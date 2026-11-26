/**
 * Projected (lightweight) run reads for the experiment page.
 *
 * The MLflow server optionally exposes `runs/search-lite` and `runs/data-keys` endpoints that
 * return only the run fields the UI actually displays (selected columns, grouping, sorting and
 * chart configuration) instead of every metric/param/tag of every run. This module owns:
 *
 * - capability detection (tri-state, based on the shared `server-info` react-query cache);
 * - building a stable, deduplicated projection from experiment page state;
 * - routing projected searches to search-lite with a single sticky legacy fallback on 404/405;
 * - exact-request coalescing and newest-response superseding for concurrent projected searches;
 * - per-experiment data-keys (column/chart selector catalog) discovery with caching, chunking,
 *   stale-generation protection and a failure fallback that never retry-storms.
 *
 * Servers that do not advertise `run_projection_enabled` keep the stock legacy behavior.
 */
import { MlflowService } from '../sdk/MlflowService';
import { ensureServerInfo, getRunProjectionEnabledSync } from '../hooks/useServerInfo';
import type { RunEntity } from '../types';

/**
 * Field selection attached to a `runs/search-lite` request. Empty key lists mean "return none"
 * (unlike legacy search, where omitted collections mean "return everything").
 */
export interface RunProjection {
  metric_keys: string[];
  param_keys: string[];
  tag_keys: string[];
  include_inputs: boolean;
  include_outputs: boolean;
}

/** A projection together with its canonical JSON serialization, used as a stable dependency hash. */
export interface RunProjectionWithHash {
  projection: RunProjection;
  projectionHash: string;
}

/** Union of discovered metric/param/tag key names, ready for selector consumption. */
export interface DiscoveredRunDataKeys {
  metric_keys: string[];
  param_keys: string[];
  tag_keys: string[];
}

/** Raw `runs/data-keys` response shape. */
interface RunDataKeysResponse {
  metric_keys?: string[];
  param_keys?: string[];
  tag_keys?: string[];
}

/**
 * Well-known operational tags consumed by the runs table, grouping, run page links and
 * model-version discovery. Always projected; arbitrary tags are never requested wholesale.
 */
export const FIXED_OPERATIONAL_TAGS = [
  'mlflow.parentRunId',
  'mlflow.log-model.history',
  'mlflow.loggedImages',
  'mlflow.source.name',
  'mlflow.source.type',
  'mlflow.source.git.commit',
  'mlflow.project.entryPoint',
  'mlflow.project.backend',
  'mlflow.user',
  'mlflow.runName',
  'mlflow.note.content',
];

/**
 * Constant projection for evaluation-runs reads: those views classify runs via
 * `outputs.modelOutputs` and the `mlflow.runType` tag only. Identity is stable so callers can
 * rely on reference equality and the object is never mutated.
 */
const EVALUATION_RUN_PROJECTION: RunProjection = {
  metric_keys: [],
  param_keys: [],
  tag_keys: ['mlflow.runType'],
  include_inputs: false,
  include_outputs: true,
};

export const getEvaluationRunProjection = (): RunProjection => EVALUATION_RUN_PROJECTION;

/**
 * Maximum run IDs per `runs/data-keys` request. Unknown IDs are chunked to this contract size.
 */
export const DATA_KEYS_MAX_RUN_IDS_PER_REQUEST = 1000;

/**
 * Tri-state capability gate:
 * - `undefined`: server-info has not loaded yet (pending) — projections are still built so the
 *   hash-keyed effects stay stable, and projected requests await the shared server-info fetch;
 * - `true`: the server advertises `run_projection_enabled`;
 * - `false`: explicit absent/false — always use legacy reads.
 */
export const getRunProjectionCapability = (): boolean | undefined => getRunProjectionEnabledSync();

// --- module state -----------------------------------------------------------------------------

interface ExperimentKeyCatalog {
  metric: Set<string>;
  param: Set<string>;
  tag: Set<string>;
}

/** Run IDs whose data keys have been fetched (and merged into the catalogs) already. */
const knownRunIds = new Set<string>();

/** Run IDs with an in-flight data-keys request, tagged with the generation that issued it. */
const inFlightRunIds = new Map<string, number>();

/** Discovered key catalogs, per experiment ID. */
const keyCatalogsByExperiment = new Map<string, ExperimentKeyCatalog>();

/** Experiment scope key -> generation at which its latest data-keys request was issued. */
const scopeGenerations = new Map<string, number>();

/** Monotonic generation, bumped on experiment-scope switch/reset. */
let currentGeneration = 0;

/** Projected mode is active after at least one successful lite search in the current scope. */
let projectionActive = false;

/** Sticky: set by a lite 404/405 or a data-keys failure; never re-tries lite afterwards. */
let projectionDisabled = false;

/** Registry of in-flight projected searches keyed by experiment IDs + page token. */
interface SupersedingEntry {
  promise: Promise<unknown> | null;
}
const supersedingEntries = new Map<string, SupersedingEntry>();
const supersedingCounts = new Map<string, number>();

/** Registry of in-flight requests with identical semantics, keyed by the full normalized payload. */
const coalescedEntries = new Map<string, { promise: Promise<unknown> }>();

const isCapabilityGateOpen = () => !projectionDisabled && getRunProjectionCapability() !== false;

/** True once a projected (lite) search has succeeded; drives e.g. the CSV export label. */
export const isRunProjectionActive = () => projectionActive === true;

/** Number of in-flight supersede keys; exposed for tests and diagnostics. */
export const getSupersededRequestCount = () => supersedingEntries.size;

// --- shared pending-capability fetch ------------------------------------------------------------

let pendingCapabilityFetch: Promise<unknown> | null = null;

/**
 * Joins the shared react-query server-info request (its query client dedupes in-flight fetches)
 * so concurrent projected callers await a single capability resolution. Coalesced here as well,
 * so even callers racing before react-query registers the query share one wave. Failures are
 * swallowed: the capability getter decides the outcome from the cache afterwards.
 */
const ensurePendingCapability = (): Promise<unknown> => {
  if (!pendingCapabilityFetch) {
    const fetchWave: Promise<unknown> = Promise.resolve(ensureServerInfo())
      .catch(() => undefined)
      .finally(() => {
        pendingCapabilityFetch = null;
      });
    pendingCapabilityFetch = fetchWave;
  }
  return pendingCapabilityFetch;
};

// --- projection construction --------------------------------------------------------------------

/** Parses a canonical column/sort key: "metrics.`name`" / "params.`name`" / "tags.`name`". */
const parseCanonicalKey = (key: string): [field: string, name: string] | null => {
  const match = /^(metrics|params|tags)\.(.+)$/.exec(key);
  if (!match) {
    return null;
  }
  let name = match[2];
  if (name.length > 1 && name.charAt(0) === '`' && name.charAt(name.length - 1) === '`') {
    name = name.slice(1, -1);
  }
  return [match[1], name];
};

const addCanonicalKey = (key: string, metrics: Set<string>, params: Set<string>, tags: Set<string>) => {
  const parsed = parseCanonicalKey(key);
  if (!parsed) {
    return;
  }
  const [field, name] = parsed;
  if (field === 'metrics') {
    metrics.add(name);
  } else if (field === 'params') {
    params.add(name);
  } else {
    tags.add(name);
  }
};

const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;

const addStrings = (target: Set<string>, values: unknown) => {
  if (!Array.isArray(values)) {
    return;
  }
  values.forEach((value) => {
    if (isNonEmptyString(value)) {
      target.add(value);
    }
  });
};

/** Loose view of a serialized chart config; only recognized fields are read. */
interface ChartConfigLike {
  metricKey?: unknown;
  dataAccessKey?: unknown;
  selectedXAxisMetricKey?: unknown;
  selectedMetricKeys?: unknown;
  selectedMetrics?: unknown;
  selectedParams?: unknown;
  yAxisExpressions?: unknown;
  xaxis?: unknown;
  yaxis?: unknown;
  zaxis?: unknown;
  nodeLevelSystemMetricConfiguration?: unknown;
}

/**
 * Generic saved-chart-config walker. Only recognized field/key structures of LINE / BAR /
 * SCATTER / CONTOUR / PARALLEL charts and metric expressions contribute keys; anything else is
 * ignored so unknown config shapes never widen the projection.
 */
const collectChartConfigKeys = (
  config: unknown,
  metrics: Set<string>,
  params: Set<string>,
  tags: Set<string>,
): void => {
  if (config === null || typeof config !== 'object') {
    return;
  }
  if (Array.isArray(config)) {
    config.forEach((item) => collectChartConfigKeys(item, metrics, params, tags));
    return;
  }
  const record = config as ChartConfigLike;
  if (isNonEmptyString(record.metricKey)) {
    metrics.add(record.metricKey);
  }
  if (isNonEmptyString(record.dataAccessKey)) {
    metrics.add(record.dataAccessKey);
  }
  if (isNonEmptyString(record.selectedXAxisMetricKey)) {
    metrics.add(record.selectedXAxisMetricKey);
  }
  addStrings(metrics, record.selectedMetricKeys);
  addStrings(metrics, record.selectedMetrics);
  addStrings(params, record.selectedParams);
  if (Array.isArray(record.yAxisExpressions)) {
    record.yAxisExpressions.forEach((expression) => {
      if (expression && Array.isArray((expression as { variables?: unknown }).variables)) {
        addStrings(metrics, (expression as { variables: unknown[] }).variables);
      }
    });
  }
  (['xaxis', 'yaxis', 'zaxis'] as const).forEach((axis) => {
    const axisConfig = record[axis] as { key?: unknown; type?: unknown } | undefined;
    if (axisConfig && isNonEmptyString(axisConfig.key)) {
      if (axisConfig.type === 'METRIC') {
        metrics.add(axisConfig.key);
      } else if (axisConfig.type === 'PARAM') {
        params.add(axisConfig.key);
      }
    }
  });
  const systemMetric = record.nodeLevelSystemMetricConfiguration as { metric?: unknown } | undefined;
  if (systemMetric && isNonEmptyString(systemMetric.metric)) {
    metrics.add(systemMetric.metric);
  }
};

/**
 * Minimal UI-state shape the projection builder reads. Compatible with ExperimentPageUIState;
 * loose so partial states and arbitrary saved chart configs work.
 */
export type RunProjectionUIState = {
  selectedColumns?: string[] | null;
  groupBy?: string | { groupByKeys?: { mode?: unknown; groupByData?: unknown }[] } | null;
  compareRunCharts?: unknown[] | null;
  globalLineChartConfig?: { selectedXAxisMetricKey?: unknown } | null;
};

/**
 * Minimal search-facets shape the projection builder reads. Compatible with
 * ExperimentPageSearchFacetsState; loose so partial and legacy states (null orderByKey) work.
 */
export type RunProjectionSearchFacets = {
  orderByKey?: string | null;
  datasetsFilter?: unknown[] | null;
};

/**
 * Builds a stable, deduplicated projection from the experiment page state: fixed operational
 * tags plus visible metric/param/tag columns, the sort key, the active group-by keys, saved
 * compare-chart configs and the global line-chart X-axis metric. Inputs are requested only for
 * the Dataset column or an active datasets filter; outputs only for the Models column (legacy
 * logged-model discovery consumes run outputs).
 *
 * Returns null when there are no facets or the server explicitly lacks the capability; the
 * pending (undefined) capability still builds so the projection hash stays stable across the
 * pending -> true transition.
 */
export const buildRunProjection = (
  searchFacets: RunProjectionSearchFacets | null | undefined,
  uiState: RunProjectionUIState | null | undefined,
): RunProjectionWithHash | null => {
  if (!searchFacets || !isCapabilityGateOpen()) {
    return null;
  }
  const metrics = new Set<string>();
  const params = new Set<string>();
  const tags = new Set<string>();
  FIXED_OPERATIONAL_TAGS.forEach((tag) => tags.add(tag));

  const selectedColumns = Array.isArray(uiState?.selectedColumns) ? uiState.selectedColumns : [];
  const hasDatasetColumn = selectedColumns.includes('attributes.`Dataset`');
  const hasModelsColumn = selectedColumns.includes('attributes.`Models`');

  selectedColumns.forEach((column) => addCanonicalKey(column, metrics, params, tags));
  if (typeof searchFacets.orderByKey === 'string') {
    addCanonicalKey(searchFacets.orderByKey, metrics, params, tags);
  }

  const groupBy = uiState?.groupBy;
  if (groupBy) {
    if (typeof groupBy === 'string') {
      // Legacy persisted group-by key: "<mode>.<aggregate>.<key>".
      const match = /^(param|tag|metric)\.[a-z]+\.(.+)$/.exec(groupBy);
      if (match) {
        if (match[1] === 'param') {
          params.add(match[2]);
        } else if (match[1] === 'tag') {
          tags.add(match[2]);
        } else {
          metrics.add(match[2]);
        }
      }
    } else if (Array.isArray(groupBy.groupByKeys)) {
      groupBy.groupByKeys.forEach((groupKey) => {
        if (!groupKey || !isNonEmptyString(groupKey.groupByData)) {
          return;
        }
        if (groupKey.mode === 'param') {
          params.add(groupKey.groupByData);
        } else if (groupKey.mode === 'tag') {
          tags.add(groupKey.groupByData);
        } else if (groupKey.mode === 'metric') {
          metrics.add(groupKey.groupByData);
        }
      });
    }
  }

  (uiState?.compareRunCharts ?? []).forEach((chart) => collectChartConfigKeys(chart, metrics, params, tags));
  const globalXAxisMetricKey = uiState?.globalLineChartConfig?.selectedXAxisMetricKey;
  if (isNonEmptyString(globalXAxisMetricKey)) {
    metrics.add(globalXAxisMetricKey);
  }

  const projection: RunProjection = {
    metric_keys: Array.from(metrics).sort(),
    param_keys: Array.from(params).sort(),
    tag_keys: Array.from(tags).sort(),
    include_inputs: hasDatasetColumn || Boolean(searchFacets.datasetsFilter && searchFacets.datasetsFilter.length),
    include_outputs: hasModelsColumn,
  };
  return { projection, projectionHash: JSON.stringify(projection) };
};

// --- endpoint routing -----------------------------------------------------------------------------

type SearchRunsExecutor<TResponse> = (payload: Record<string, any>) => Promise<TResponse>;

/** Defensive 404/405 detection across ErrorWrapper shapes (status / statusCode / response.status). */
const isEndpointUnavailableError = (error: unknown): boolean => {
  if (!error) {
    return false;
  }
  const candidate = error as {
    status?: unknown;
    statusCode?: unknown;
    response?: { status?: unknown };
  };
  const status =
    candidate.status !== undefined
      ? candidate.status
      : candidate.statusCode !== undefined
        ? candidate.statusCode
        : candidate.response?.status;
  return status === 404 || status === 405;
};

const stripProjection = <T extends { projection?: RunProjection } & Record<string, any>>(payload: T): T => ({
  ...payload,
  projection: undefined,
});

/**
 * Routes a search to `runs/search-lite` when a projection is present and supported, otherwise to
 * legacy `runs/search`. Legacy payloads ALWAYS have the projection stripped. Behavior:
 *
 * - no projection / sticky-disabled / capability explicitly false: immediate legacy, the pending
 *   server-info request is never awaited;
 * - capability pending: awaits the shared server-info fetch, then decides (lite iff true);
 * - capability true: lite directly; a lite 404/405 sticky-disables projection and reissues the
 *   legacy request exactly once; any other lite error propagates unchanged (no retry storm).
 */
export const routeProjectedSearchRuns = <TResponse>(
  payload: { projection?: RunProjection } & Record<string, any>,
  legacySearch: SearchRunsExecutor<TResponse>,
  liteSearch: SearchRunsExecutor<TResponse>,
): Promise<TResponse> => {
  const projection = payload.projection;
  if (!projection || projectionDisabled || getRunProjectionCapability() === false) {
    return legacySearch(stripProjection(payload));
  }
  const decide = (): Promise<TResponse> => {
    if (getRunProjectionCapability() === true) {
      return liteSearch({ ...payload, projection })
        .then((response) => {
          projectionActive = true;
          return response;
        })
        .catch((error) => {
          if (isEndpointUnavailableError(error)) {
            projectionDisabled = true;
            return legacySearch(stripProjection(payload));
          }
          throw error;
        });
    }
    return legacySearch(stripProjection(payload));
  };
  return getRunProjectionCapability() === undefined ? ensurePendingCapability().then(decide) : decide();
};

// --- request coalescing + superseding ---------------------------------------------------------------

const createSupersedeKey = (experimentIds: string[] | undefined, pageToken: string | null | undefined) =>
  `${(experimentIds ?? []).slice().sort().join(',')}|${pageToken === null || pageToken === undefined ? '' : String(pageToken)}`;

/**
 * Wraps the whole projected search payload (primary + pinned merge included) with:
 *
 * - newest-response superseding, keyed by sorted experiment IDs + page token: initial, manual
 *   refresh and auto-refresh of the same page share a key, while different Load-more pages stay
 *   independent. An older request settling after a newer same-key request adopts the newest
 *   promise, so Redux fulfilled actions always receive the newest response.
 * - exact-request coalescing, keyed by the full normalized request semantics: a duplicate call
 *   returns the same overall promise without re-executing the executor or any network call.
 *
 * Entries are removed on settle (rejection clears coalescing so a later refresh re-executes) and
 * on experiment reset. Calls without a projection skip registration entirely.
 */
export const coalesceAndSupersedeProjectedSearch = <TResponse>(
  projection: RunProjection | null | undefined,
  experimentIds: string[] | undefined,
  pageToken: string | null | undefined,
  requestSemantics: unknown,
  execute: () => Promise<TResponse>,
): Promise<TResponse> => {
  if (!projection) {
    return execute();
  }
  const supersedeKey = createSupersedeKey(experimentIds, pageToken);
  const entry: SupersedingEntry = { promise: null };
  supersedingEntries.set(supersedeKey, entry);
  supersedingCounts.set(supersedeKey, (supersedingCounts.get(supersedeKey) ?? 0) + 1);

  const coalesceKey =
    requestSemantics !== null && requestSemantics !== undefined ? JSON.stringify(requestSemantics) : null;

  let requestPromise: Promise<TResponse>;
  const existing = coalesceKey !== null ? coalescedEntries.get(coalesceKey) : undefined;
  if (existing) {
    requestPromise = existing.promise as Promise<TResponse>;
  } else {
    try {
      requestPromise = Promise.resolve(execute());
    } catch (error) {
      requestPromise = Promise.reject(error);
    }
    if (coalesceKey !== null) {
      const coalesced = { promise: requestPromise as Promise<unknown> };
      coalescedEntries.set(coalesceKey, coalesced);
      requestPromise = requestPromise.then(
        (response) => {
          if (coalescedEntries.get(coalesceKey) === coalesced) {
            coalescedEntries.delete(coalesceKey);
          }
          return response;
        },
        (error) => {
          if (coalescedEntries.get(coalesceKey) === coalesced) {
            coalescedEntries.delete(coalesceKey);
          }
          throw error;
        },
      );
    }
  }

  const releaseSupersedeCount = () => {
    supersedingCounts.set(supersedeKey, (supersedingCounts.get(supersedeKey) ?? 1) - 1);
  };

  entry.promise = requestPromise.then(
    (response) => {
      releaseSupersedeCount();
      const newest = supersedingEntries.get(supersedeKey);
      if (newest && newest !== entry) {
        // A newer same-key request registered while this one was in flight: adopt its outcome.
        if ((supersedingCounts.get(supersedeKey) ?? 0) === 0) {
          supersedingEntries.delete(supersedeKey);
          supersedingCounts.delete(supersedeKey);
        }
        return newest.promise as Promise<TResponse>;
      }
      if ((supersedingCounts.get(supersedeKey) ?? 0) === 0 && supersedingEntries.get(supersedeKey) === entry) {
        supersedingEntries.delete(supersedeKey);
        supersedingCounts.delete(supersedeKey);
      }
      return response;
    },
    (error) => {
      releaseSupersedeCount();
      const newest = supersedingEntries.get(supersedeKey);
      if (newest && newest !== entry) {
        if ((supersedingCounts.get(supersedeKey) ?? 0) === 0) {
          supersedingEntries.delete(supersedeKey);
          supersedingCounts.delete(supersedeKey);
        }
        return newest.promise as Promise<TResponse>;
      }
      if ((supersedingCounts.get(supersedeKey) ?? 0) === 0 && supersedingEntries.get(supersedeKey) === entry) {
        supersedingEntries.delete(supersedeKey);
        supersedingCounts.delete(supersedeKey);
      }
      throw error;
    },
  );
  return entry.promise as Promise<TResponse>;
};

// --- data-keys discovery -----------------------------------------------------------------------------

/**
 * Selector-facing catalog read: union of the per-experiment data-keys results, sorted. Combined
 * by the runs selector with the keys derivable from loaded runs so the column/chart selectors see
 * every known key without hydrating all values.
 */
export const getDiscoveredRunDataKeys = (experimentIds: string[] | undefined): DiscoveredRunDataKeys => {
  const metric = new Set<string>();
  const param = new Set<string>();
  const tag = new Set<string>();
  (experimentIds ?? []).forEach((experimentId) => {
    const catalog = keyCatalogsByExperiment.get(experimentId);
    if (catalog) {
      catalog.metric.forEach((key) => metric.add(key));
      catalog.param.forEach((key) => param.add(key));
      catalog.tag.forEach((key) => tag.add(key));
    }
  });
  return {
    metric_keys: Array.from(metric).sort(),
    param_keys: Array.from(param).sort(),
    tag_keys: Array.from(tag).sort(),
  };
};

const chunkRunIds = (runIds: string[], size: number): string[][] => {
  const chunks: string[][] = [];
  for (let index = 0; index < runIds.length; index += size) {
    chunks.push(runIds.slice(index, index + size));
  }
  return chunks;
};

const getRunIdFromEntity = (run: RunEntity | undefined): string | undefined => {
  const info = run?.info as { run_id?: string; runUuid?: string } | undefined;
  return info?.run_id || info?.runUuid;
};

/**
 * Discovers metric/param/tag key names for the given runs via `runs/data-keys` and merges them
 * into the per-experiment catalogs. Resolves `true` when new keys were applied (callers should
 * recompute selectors), `false` otherwise.
 *
 * - only unknown (not yet fetched, not in-flight) run IDs are requested, deduplicated and
 *   chunked to <= 1000 IDs per request; IDs are marked in-flight synchronously so concurrent
 *   calls never duplicate them;
 * - same-scope responses merge additively in any order; only responses issued before an
 *   experiment-scope switch/reset (generation bump) are discarded;
 * - any failure applies no partial catalog, clears the in-flight flags, leaves the failed runs
 *   unknown and sticky-disables projection, so subsequent searches safely use legacy reads
 *   instead of retry-storming data-keys on every auto-refresh.
 */
export const discoverRunDataKeys = (
  runs: (RunEntity | undefined)[] | undefined,
  experimentIds: string[] | undefined,
): Promise<boolean> => {
  if (!isCapabilityGateOpen() || !projectionActive) {
    return Promise.resolve(false);
  }
  const scopeKey = (experimentIds ?? []).slice().sort().join(',');
  if (!scopeKey) {
    return Promise.resolve(false);
  }
  const runIds = (runs ?? []).map(getRunIdFromEntity).filter((runId): runId is string => Boolean(runId));
  if (!runIds.length) {
    return Promise.resolve(false);
  }
  const unknownRunIds: string[] = [];
  const seen = new Set<string>();
  for (const runId of runIds) {
    if (!seen.has(runId) && !knownRunIds.has(runId) && !inFlightRunIds.has(runId)) {
      seen.add(runId);
      unknownRunIds.push(runId);
    }
  }
  if (!unknownRunIds.length) {
    return Promise.resolve(false);
  }
  const issuedGeneration = currentGeneration;
  unknownRunIds.forEach((runId) => inFlightRunIds.set(runId, issuedGeneration));
  scopeGenerations.set(scopeKey, currentGeneration);

  const releaseInFlight = () => {
    unknownRunIds.forEach((runId) => {
      if (inFlightRunIds.get(runId) === issuedGeneration) {
        inFlightRunIds.delete(runId);
      }
    });
  };

  return Promise.all(
    chunkRunIds(unknownRunIds, DATA_KEYS_MAX_RUN_IDS_PER_REQUEST).map(
      (runIdsChunk) => MlflowService.getRunDataKeys({ run_ids: runIdsChunk }) as Promise<RunDataKeysResponse>,
    ),
  )
    .then((responses) => {
      if (scopeGenerations.get(scopeKey) !== currentGeneration) {
        // Issued before an experiment-scope switch/reset: discard.
        releaseInFlight();
        return false;
      }
      const discovered = { metric: new Set<string>(), param: new Set<string>(), tag: new Set<string>() };
      (responses ?? []).forEach((response) => {
        (response?.metric_keys ?? []).forEach((key) => discovered.metric.add(key));
        (response?.param_keys ?? []).forEach((key) => discovered.param.add(key));
        (response?.tag_keys ?? []).forEach((key) => discovered.tag.add(key));
      });
      unknownRunIds.forEach((runId) => {
        knownRunIds.add(runId);
        if (inFlightRunIds.get(runId) === issuedGeneration) {
          inFlightRunIds.delete(runId);
        }
      });
      Array.from(new Set(experimentIds ?? [])).forEach((experimentId) => {
        let catalog = keyCatalogsByExperiment.get(experimentId);
        if (!catalog) {
          catalog = { metric: new Set(), param: new Set(), tag: new Set() };
          keyCatalogsByExperiment.set(experimentId, catalog);
        }
        discovered.metric.forEach((key) => catalog.metric.add(key));
        discovered.param.forEach((key) => catalog.param.add(key));
        discovered.tag.forEach((key) => catalog.tag.add(key));
      });
      return true;
    })
    .catch(() => {
      releaseInFlight();
      projectionDisabled = true;
      return false;
    });
};

/**
 * Experiment-scope switch: drops the catalogs of experiments no longer in scope, clears the
 * coalescing/superseding registries, bumps the global generation so in-flight pre-switch
 * discovery responses are discarded, and resets the projected-mode flag.
 */
export const resetRunProjectionScope = (experimentIds: string[] | undefined) => {
  const keep = new Set(experimentIds ?? []);
  for (const experimentId of Array.from(keyCatalogsByExperiment.keys())) {
    if (!keep.has(experimentId)) {
      keyCatalogsByExperiment.delete(experimentId);
    }
  }
  scopeGenerations.clear();
  supersedingEntries.clear();
  supersedingCounts.clear();
  coalescedEntries.clear();
  currentGeneration += 1;
  projectionActive = false;
};

/** Resets all module state; intended for tests only. */
export const resetRunProjectionStateForTests = () => {
  knownRunIds.clear();
  inFlightRunIds.clear();
  keyCatalogsByExperiment.clear();
  scopeGenerations.clear();
  supersedingEntries.clear();
  supersedingCounts.clear();
  coalescedEntries.clear();
  currentGeneration = 0;
  projectionActive = false;
  projectionDisabled = false;
  pendingCapabilityFetch = null;
};
