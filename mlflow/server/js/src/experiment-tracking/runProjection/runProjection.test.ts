import { describe, beforeEach, test, expect, jest } from '@jest/globals';
import {
  FIXED_OPERATIONAL_TAGS,
  buildRunProjection,
  coalesceAndSupersedeProjectedSearch,
  discoverRunDataKeys,
  getDiscoveredRunDataKeys,
  getEvaluationRunProjection,
  getSupersededRequestCount,
  isRunProjectionActive,
  resetRunProjectionScope,
  resetRunProjectionStateForTests,
  routeProjectedSearchRuns,
} from './runProjection';
import { MlflowService } from '../sdk/MlflowService';
import { ensureServerInfo, getRunProjectionEnabledSync } from '../hooks/useServerInfo';
import type { RunEntity } from '../types';

jest.mock('../sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
    searchRunsLite: jest.fn(),
    getRunDataKeys: jest.fn(),
  },
}));

jest.mock('../hooks/useServerInfo', () => ({
  getRunProjectionEnabledSync: jest.fn(),
  ensureServerInfo: jest.fn(),
}));

const FIXED_TAGS = [...FIXED_OPERATIONAL_TAGS].sort();

// Tri-state capability: undefined = server-info pending, false = explicit absent, true = enabled.
let capability: boolean | undefined;

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

const mkRun = (id: string): RunEntity => ({ info: { runUuid: id } }) as unknown as RunEntity;

type SearchResponse = { runs: RunEntity[]; next_page_token?: string | null };
type SearchPayload = { projection?: unknown; run_ids?: string[] } & Record<string, any>;

const makeLegacySearch = () =>
  jest.fn<(payload: SearchPayload) => Promise<SearchResponse>>().mockResolvedValue({ runs: [mkRun('legacy-r1')] });
const makeLiteSearch = () =>
  jest.fn<(payload: SearchPayload) => Promise<SearchResponse>>().mockResolvedValue({ runs: [mkRun('lite-r1')] });

/** Activates projected mode the way production does: one successful lite search. */
const activateProjectedMode = async () => {
  capability = true;
  await routeProjectedSearchRuns(
    {
      experiment_ids: ['1'],
      projection: { metric_keys: [], param_keys: [], tag_keys: [], include_inputs: false, include_outputs: false },
    },
    makeLegacySearch(),
    makeLiteSearch(),
  );
  expect(isRunProjectionActive()).toBe(true);
};

beforeEach(() => {
  resetRunProjectionStateForTests();
  capability = true;
  jest.mocked(getRunProjectionEnabledSync).mockClear();
  jest.mocked(getRunProjectionEnabledSync).mockImplementation(() => capability);
  jest.mocked(ensureServerInfo).mockClear();
  jest.mocked(ensureServerInfo).mockImplementation(() => Promise.resolve({} as any));
  jest.mocked(MlflowService.getRunDataKeys).mockReset();
  jest
    .mocked(MlflowService.getRunDataKeys)
    .mockResolvedValue({ metric_keys: ['m1'], param_keys: ['p1'], tag_keys: ['t1'] });
});

describe('buildRunProjection', () => {
  const baseFacets = { searchFilter: '', orderByKey: null, datasetsFilter: [] };

  test('builds while capability is pending and returns null only for explicit false', () => {
    capability = undefined;
    expect(buildRunProjection(baseFacets, {})).not.toBeNull();
    capability = false;
    expect(buildRunProjection(baseFacets, {})).toBeNull();
  });

  test('returns null without search facets', () => {
    expect(buildRunProjection(null, {})).toBeNull();
  });

  test('regular view gives zero metrics/params and fixed operational tags', () => {
    const built = buildRunProjection(baseFacets, {});
    expect(built?.projection.metric_keys).toEqual([]);
    expect(built?.projection.param_keys).toEqual([]);
    expect(built?.projection.tag_keys).toEqual(FIXED_TAGS);
    expect(built?.projection.include_inputs).toBe(false);
    expect(built?.projection.include_outputs).toBe(false);
  });

  test('includes selectedColumns, groupBy and orderByKey keys', () => {
    const built = buildRunProjection(
      { ...baseFacets, orderByKey: 'metrics.`loss`' },
      {
        selectedColumns: ['metrics.`acc`', 'params.`lr`', 'tags.`foo`', 'attributes.`Source`'],
        groupBy: {
          aggregateFunction: 'min',
          groupByKeys: [
            { mode: 'param', groupByData: 'batch_size' },
            { mode: 'tag', groupByData: 'env' },
            { mode: 'metric', groupByData: 'auc' },
          ],
        } as any,
        compareRunCharts: [],
      },
    );
    expect(built?.projection.metric_keys).toEqual(['acc', 'auc', 'loss']);
    expect(built?.projection.param_keys).toEqual(['batch_size', 'lr']);
    expect(built?.projection.tag_keys).toContain('env');
    expect(built?.projection.tag_keys).toContain('foo');
    expect(built?.projection.tag_keys).toHaveLength(FIXED_TAGS.length + 2);
  });

  test('parses legacy string groupBy keys', () => {
    const paramGroup = buildRunProjection(
      { ...baseFacets, orderByKey: 'params.`gamma`' },
      { groupBy: 'param.min.some.param' },
    );
    expect(paramGroup?.projection.param_keys).toEqual(['gamma', 'some.param']);
    const tagGroup = buildRunProjection(baseFacets, { groupBy: 'tag.max.some.tag' });
    expect(tagGroup?.projection.tag_keys).toContain('some.tag');
  });

  test('projection is stable, deduplicated and serialized for hashing', () => {
    const ui = {
      selectedColumns: ['metrics.`acc`'],
      compareRunCharts: [{ type: 'BAR', metricKey: 'acc' }],
    };
    const first = buildRunProjection(baseFacets, ui);
    const second = buildRunProjection(baseFacets, ui);
    expect(first?.projectionHash).toEqual(second?.projectionHash);
    expect(first?.projectionHash).toEqual(JSON.stringify(first?.projection));
    expect(first?.projection.metric_keys.filter((key) => key === 'acc')).toHaveLength(1);
    // Changed content must change the hash.
    expect(
      buildRunProjection(baseFacets, { ...ui, compareRunCharts: [{ type: 'BAR', metricKey: 'other' }] })
        ?.projectionHash,
    ).not.toEqual(first?.projectionHash);
    // Fresh-but-equivalent object identities (local state hydration) keep the same hash.
    expect(
      buildRunProjection(JSON.parse(JSON.stringify(baseFacets)), JSON.parse(JSON.stringify(ui)))?.projectionHash,
    ).toEqual(first?.projectionHash);
  });

  test('hash is identical across the pending -> true capability transition', () => {
    capability = undefined;
    const pendingHash = buildRunProjection(baseFacets, {})?.projectionHash;
    capability = true;
    expect(buildRunProjection(baseFacets, {})?.projectionHash).toEqual(pendingHash);
  });

  test('inputs/outputs flags follow Dataset/Models columns and the datasets filter only', () => {
    const both = buildRunProjection(baseFacets, {
      selectedColumns: ['attributes.`Dataset`', 'attributes.`Models`'],
    });
    expect(both?.projection.include_inputs).toBe(true);
    expect(both?.projection.include_outputs).toBe(true);

    const datasetOnly = buildRunProjection(baseFacets, { selectedColumns: ['attributes.`Dataset`'] });
    expect(datasetOnly?.projection.include_inputs).toBe(true);
    expect(datasetOnly?.projection.include_outputs).toBe(false);

    const modelsOnly = buildRunProjection(baseFacets, { selectedColumns: ['attributes.`Models`'] });
    expect(modelsOnly?.projection.include_inputs).toBe(false);
    expect(modelsOnly?.projection.include_outputs).toBe(true);

    const filterOnly = buildRunProjection({ ...baseFacets, datasetsFilter: [{ name: 'd', digest: 'x' } as any] }, {});
    expect(filterOnly?.projection.include_inputs).toBe(true);
    expect(filterOnly?.projection.include_outputs).toBe(false);

    const noIo = buildRunProjection(baseFacets, { selectedColumns: ['attributes.`Source`'] });
    expect(noIo?.projection.include_inputs).toBe(false);
    expect(noIo?.projection.include_outputs).toBe(false);
  });

  test('nine saved charts yield exactly the selected metric/param keys and no extra tags', () => {
    const nineCharts = [
      {
        type: 'LINE',
        metricKey: '',
        selectedMetricKeys: ['loss', 'acc'],
        selectedXAxisMetricKey: 'epoch',
        yAxisExpressions: [{ variables: ['loss', 'acc'], expression: 'loss/acc' }],
      },
      { type: 'BAR', metricKey: 'f1', selectedMetricKeys: ['f1', 'precision'] },
      { type: 'SCATTER', xaxis: { key: 'sx', type: 'METRIC' }, yaxis: { key: 'sy', type: 'PARAM' } },
      {
        type: 'CONTOUR',
        xaxis: { key: 'cx', type: 'METRIC' },
        yaxis: { key: 'cy', type: 'METRIC' },
        zaxis: { key: 'cz', type: 'METRIC' },
      },
      { type: 'PARALLEL', selectedParams: ['pp1', 'pp2'], selectedMetrics: ['pm1'] },
      { type: 'LINE', metricKey: '', selectedMetricKeys: null },
      { type: 'BAR', metricKey: 'single' },
      { type: 'LINE', metricKey: '', selectedMetricKeys: ['only'] },
      {
        type: 'LINE',
        metricKey: '',
        dataAccessKey: 'ds/key',
        nodeLevelSystemMetricConfiguration: { metric: 'system/gpu', type: 'gpu' },
      },
    ];
    const built = buildRunProjection(baseFacets, { compareRunCharts: nineCharts as any });
    expect(built?.projection.metric_keys).toEqual([
      'acc',
      'cx',
      'cy',
      'cz',
      'ds/key',
      'epoch',
      'f1',
      'loss',
      'only',
      'pm1',
      'precision',
      'single',
      'sx',
      'system/gpu',
    ]);
    expect(built?.projection.param_keys).toEqual(['pp1', 'pp2', 'sy']);
    expect(built?.projection.tag_keys).toEqual(FIXED_TAGS);
  });

  test('runs without saved chart config add no chart keys', () => {
    expect(buildRunProjection(baseFacets, { compareRunCharts: undefined })?.projection.metric_keys).toEqual([]);
  });

  test('globalLineChartConfig.selectedXAxisMetricKey is projected and hashed', () => {
    const globalX = buildRunProjection(baseFacets, {
      globalLineChartConfig: { selectedXAxisMetricKey: 'global_x_metric' },
    });
    expect(globalX?.projection.metric_keys).toEqual(['global_x_metric']);
    expect(globalX?.projectionHash).not.toEqual(buildRunProjection(baseFacets, {})?.projectionHash);
  });
});

describe('getEvaluationRunProjection', () => {
  test('is a stable-identity constant with runType tag and outputs only', () => {
    const projection = getEvaluationRunProjection();
    expect(projection).toEqual({
      metric_keys: [],
      param_keys: [],
      tag_keys: ['mlflow.runType'],
      include_inputs: false,
      include_outputs: true,
    });
    expect(getEvaluationRunProjection()).toBe(projection);
  });
});

describe('routeProjectedSearchRuns', () => {
  const projection = { metric_keys: [], param_keys: [], tag_keys: [], include_inputs: false, include_outputs: false };

  test('capability true uses search-lite with the projection', async () => {
    const legacy = makeLegacySearch();
    const lite = makeLiteSearch();
    const result = await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    expect((result.runs![0] as any).info.runUuid).toBe('lite-r1');
    expect(lite).toHaveBeenCalledTimes(1);
    expect(legacy).not.toHaveBeenCalled();
    expect(lite.mock.calls[0][0].projection).toEqual(projection);
    expect(isRunProjectionActive()).toBe(true);
  });

  test('capability false routes to legacy with the projection stripped from the JSON body', async () => {
    capability = false;
    const legacy = makeLegacySearch();
    const lite = makeLiteSearch();
    await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    expect(legacy).toHaveBeenCalledTimes(1);
    expect(lite).not.toHaveBeenCalled();
    expect(legacy.mock.calls[0][0].projection).toBeUndefined();
    expect('projection' in JSON.parse(JSON.stringify(legacy.mock.calls[0][0]))).toBe(false);
  });

  test('no projection goes immediate legacy without awaiting server-info', async () => {
    capability = undefined;
    const legacy = makeLegacySearch();
    await routeProjectedSearchRuns({ experiment_ids: ['1'] }, legacy, makeLiteSearch());
    expect(legacy).toHaveBeenCalledTimes(1);
    expect(ensureServerInfo).not.toHaveBeenCalled();
  });

  test('pending capability: concurrent callers share one server-info fetch, then both go lite', async () => {
    capability = undefined;
    let resolveInfo!: (value: unknown) => void;
    jest.mocked(ensureServerInfo).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveInfo = resolve;
        }) as any,
    );
    const legacy = makeLegacySearch();
    const lite = makeLiteSearch();
    const first = routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    const second = routeProjectedSearchRuns(
      { experiment_ids: ['1'], projection: getEvaluationRunProjection() },
      legacy,
      lite,
    );
    await flush();
    expect(lite).not.toHaveBeenCalled();
    expect(legacy).not.toHaveBeenCalled();
    expect(ensureServerInfo).toHaveBeenCalledTimes(1);
    capability = true;
    resolveInfo({});
    await Promise.all([first, second]);
    expect(lite).toHaveBeenCalledTimes(2);
    expect(legacy).not.toHaveBeenCalled();
  });

  test('pending then explicit false: both fall back to stripped legacy without any lite call', async () => {
    capability = undefined;
    let resolveInfo!: (value: unknown) => void;
    jest.mocked(ensureServerInfo).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveInfo = resolve;
        }) as any,
    );
    const legacy = makeLegacySearch();
    const lite = makeLiteSearch();
    const first = routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    await flush();
    capability = false;
    resolveInfo({});
    await first;
    expect(lite).not.toHaveBeenCalled();
    expect(legacy).toHaveBeenCalledTimes(1);
    expect('projection' in JSON.parse(JSON.stringify(legacy.mock.calls[0][0]))).toBe(false);
  });

  test('lite 404 disables capability and reissues legacy exactly once, then stays legacy', async () => {
    const legacy = makeLegacySearch();
    const lite = jest
      .fn<(payload: SearchPayload) => Promise<SearchResponse>>()
      .mockRejectedValue(Object.assign(new Error('not found'), { status: 404 }));
    const result = await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    expect((result.runs![0] as any).info.runUuid).toBe('legacy-r1');
    expect(lite).toHaveBeenCalledTimes(1);
    expect(legacy).toHaveBeenCalledTimes(1);
    expect('projection' in JSON.parse(JSON.stringify(legacy.mock.calls[0][0]))).toBe(false);

    await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    expect(lite).toHaveBeenCalledTimes(1);
    expect(legacy).toHaveBeenCalledTimes(2);
    // Sticky disable wins: no further server-info fetches are awaited either.
    capability = undefined;
    await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    expect(ensureServerInfo).not.toHaveBeenCalled();
    expect(legacy).toHaveBeenCalledTimes(3);
  });

  test('405 responses are detected across error shapes', async () => {
    const legacy = makeLegacySearch();
    const lite = jest
      .fn<(payload: SearchPayload) => Promise<SearchResponse>>()
      .mockRejectedValue({ response: { status: 405 } });
    const result = await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite);
    expect((result.runs![0] as any).info.runUuid).toBe('legacy-r1');
    expect(legacy).toHaveBeenCalledTimes(1);
  });

  test('unexpected lite errors surface without legacy reissue or capability loss', async () => {
    const legacy = makeLegacySearch();
    const lite = jest
      .fn<(payload: SearchPayload) => Promise<SearchResponse>>()
      .mockRejectedValue(Object.assign(new Error('boom'), { status: 500 }));
    await expect(routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, lite)).rejects.toThrow('boom');
    expect(legacy).not.toHaveBeenCalled();
    // Capability remains enabled: the next call retries lite.
    const liteOk = makeLiteSearch();
    await routeProjectedSearchRuns({ experiment_ids: ['1'], projection }, legacy, liteOk);
    expect(liteOk).toHaveBeenCalledTimes(1);
  });

  test('the constant evaluation projection routes by reference and is never mutated', async () => {
    const lite = makeLiteSearch();
    const projectionConstant = getEvaluationRunProjection();
    await routeProjectedSearchRuns({ experiment_ids: ['1'], projection: projectionConstant }, makeLegacySearch(), lite);
    expect(lite.mock.calls[0][0].projection).toBe(projectionConstant);
    capability = false;
    const legacy = makeLegacySearch();
    await routeProjectedSearchRuns({ experiment_ids: ['1'], projection: projectionConstant }, legacy, makeLiteSearch());
    expect(legacy.mock.calls[0][0].projection).toBeUndefined();
    expect(getEvaluationRunProjection()).toBe(projectionConstant);
  });
});

describe('coalesceAndSupersedeProjectedSearch', () => {
  const projection = {
    metric_keys: ['m'],
    param_keys: [],
    tag_keys: [],
    include_inputs: false,
    include_outputs: false,
  };
  const experimentIds = ['exp1'];

  type Deferred = { resolve: (value: SearchResponse) => void; reject: (error: unknown) => void };
  const makeDeferredExecute = (pending: Deferred[]) =>
    jest.fn<() => Promise<SearchResponse>>(
      () => new Promise<SearchResponse>((resolve, reject) => pending.push({ resolve, reject })),
    );

  test('calls without a projection bypass the registry entirely', async () => {
    const execute = jest.fn<() => Promise<string>>().mockResolvedValue('legacy');
    const result = await coalesceAndSupersedeProjectedSearch(undefined, experimentIds, undefined, { a: 1 }, execute);
    expect(result).toBe('legacy');
    expect(getSupersededRequestCount()).toBe(0);
  });

  test('identical concurrent requests execute once and resolve with the same response object', async () => {
    const pending: Deferred[] = [];
    const execute = makeDeferredExecute(pending);
    const semantics = { experimentIds, filter: '', pageToken: null, projection };
    const first = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    const second = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    expect(execute).toHaveBeenCalledTimes(1);
    expect(getSupersededRequestCount()).toBe(1);
    pending[0].resolve({ runs: [mkRun('shared')] });
    const [firstResult, secondResult] = await Promise.all([first, second]);
    expect(firstResult).toBe(secondResult);
    expect(getSupersededRequestCount()).toBe(0);
  });

  test('a later identical refresh executes normally again', async () => {
    const execute = jest.fn<() => Promise<SearchResponse>>().mockResolvedValue({ runs: [mkRun('refresh')] });
    const semantics = { experimentIds, filter: '' };
    await coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    await coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    expect(execute).toHaveBeenCalledTimes(2);
  });

  test('different request semantics never coalesce', async () => {
    const execute = jest.fn<() => Promise<SearchResponse>>().mockResolvedValue({ runs: [] });
    await Promise.all([
      coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { filter: 'a' }, execute),
      coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { filter: 'b' }, execute),
    ]);
    expect(execute).toHaveBeenCalledTimes(2);
  });

  test('rejection clears coalescing state so an identical retry executes', async () => {
    const execute = jest
      .fn<() => Promise<SearchResponse>>()
      .mockRejectedValueOnce(new Error('lite boom'))
      .mockResolvedValue({ runs: [mkRun('retry')] });
    const semantics = { experimentIds, filter: '' };
    const first = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    const second = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    await expect(first).rejects.toThrow('lite boom');
    await expect(second).rejects.toThrow('lite boom');
    expect(execute).toHaveBeenCalledTimes(1);
    const retry = await coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, semantics, execute);
    expect((retry.runs[0] as any).info.runUuid).toBe('retry');
    expect(execute).toHaveBeenCalledTimes(2);
  });

  test('an older same-key request adopts the newest response when the newest completes first', async () => {
    const pending: Deferred[] = [];
    const execute = makeDeferredExecute(pending);
    const older = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'a' }, execute);
    const newer = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'b' }, execute);
    expect(getSupersededRequestCount()).toBe(1);
    pending[1].resolve({ runs: [mkRun('rB')] });
    pending[0].resolve({ runs: [mkRun('rA')] });
    const [olderResult, newerResult] = await Promise.all([older, newer]);
    expect((newerResult.runs[0] as any).info.runUuid).toBe('rB');
    expect((olderResult.runs[0] as any).info.runUuid).toBe('rB');
    expect(getSupersededRequestCount()).toBe(0);
  });

  test('an older same-key request adopts the newest response even when it settles first', async () => {
    const pending: Deferred[] = [];
    const execute = makeDeferredExecute(pending);
    const older = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'a' }, execute);
    const newer = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'b' }, execute);
    pending[0].resolve({ runs: [mkRun('rA2')] });
    const olderResultPromise = older;
    pending[1].resolve({ runs: [mkRun('rB2')] });
    const [olderResult, newerResult] = await Promise.all([olderResultPromise, newer]);
    expect((newerResult.runs[0] as any).info.runUuid).toBe('rB2');
    expect((olderResult.runs[0] as any).info.runUuid).toBe('rB2');
    expect(getSupersededRequestCount()).toBe(0);
  });

  test('different page tokens are independent supersede keys', async () => {
    const pending: Deferred[] = [];
    const execute = makeDeferredExecute(pending);
    const page1 = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { t: 1 }, execute);
    const page2 = coalesceAndSupersedeProjectedSearch(projection, experimentIds, 'tok2', { t: 2 }, execute);
    expect(getSupersededRequestCount()).toBe(2);
    pending[0].resolve({ runs: [mkRun('page1')] });
    pending[1].resolve({ runs: [mkRun('page2')] });
    const [page1Result, page2Result] = await Promise.all([page1, page2]);
    expect((page1Result.runs[0] as any).info.runUuid).toBe('page1');
    expect((page2Result.runs[0] as any).info.runUuid).toBe('page2');
    expect(getSupersededRequestCount()).toBe(0);
  });

  test('the newest rejection propagates to superseded requests with cleanup', async () => {
    const pending: Deferred[] = [];
    const execute = makeDeferredExecute(pending);
    const older = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'a' }, execute);
    const newer = coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'b' }, execute);
    pending[1].reject(new Error('lite exploded'));
    await expect(newer).rejects.toThrow('lite exploded');
    pending[0].reject(new Error('old lite error'));
    await expect(older).rejects.toThrow('lite exploded');
    expect(getSupersededRequestCount()).toBe(0);
  });

  test('resetRunProjectionScope clears the registries', () => {
    const execute = jest.fn<() => Promise<SearchResponse>>(() => new Promise<SearchResponse>(() => {}));
    coalesceAndSupersedeProjectedSearch(projection, experimentIds, undefined, { p: 'a' }, execute);
    expect(getSupersededRequestCount()).toBe(1);
    resetRunProjectionScope(experimentIds);
    expect(getSupersededRequestCount()).toBe(0);
  });
});

describe('discoverRunDataKeys', () => {
  const dataKeysCalls = () => jest.mocked(MlflowService.getRunDataKeys).mock.calls.map(([payload]) => payload);

  test('does nothing until projected mode is active', async () => {
    expect(await discoverRunDataKeys([mkRun('r1')], ['exp1'])).toBe(false);
    expect(MlflowService.getRunDataKeys).not.toHaveBeenCalled();
  });

  test('fetches data-keys once per unknown run set; auto-refresh re-downloads nothing', async () => {
    await activateProjectedMode();
    const runs = [mkRun('r1'), mkRun('r2')];
    expect(await discoverRunDataKeys(runs, ['exp1'])).toBe(true);
    expect(dataKeysCalls()).toHaveLength(1);
    expect([...dataKeysCalls()[0].run_ids].sort()).toEqual(['r1', 'r2']);
    expect(await discoverRunDataKeys(runs, ['exp1'])).toBe(false);
    expect(dataKeysCalls()).toHaveLength(1);
  });

  test('Load more fetches data-keys only for newly seen run IDs', async () => {
    await activateProjectedMode();
    await discoverRunDataKeys([mkRun('r1'), mkRun('r2')], ['exp1']);
    await discoverRunDataKeys([mkRun('r1'), mkRun('r2'), mkRun('r3')], ['exp1']);
    expect(dataKeysCalls()).toHaveLength(2);
    expect(dataKeysCalls()[1].run_ids).toEqual(['r3']);
  });

  test('selector catalog unions per-experiment results only', async () => {
    await activateProjectedMode();
    await discoverRunDataKeys([mkRun('r1')], ['exp1']);
    expect(getDiscoveredRunDataKeys(['exp1'])).toEqual({
      metric_keys: ['m1'],
      param_keys: ['p1'],
      tag_keys: ['t1'],
    });
    expect(getDiscoveredRunDataKeys(['other-exp'])).toEqual({ metric_keys: [], param_keys: [], tag_keys: [] });
  });

  test('unknown run IDs chunk into <=1000-ID requests with no truncation', async () => {
    await activateProjectedMode();
    const bigRuns = Array.from({ length: 2500 }, (_, index) => mkRun(`big-${index}`));
    await discoverRunDataKeys(bigRuns, ['exp-chunk']);
    expect(dataKeysCalls().map((call) => call.run_ids.length)).toEqual([1000, 1000, 500]);
  });

  test('duplicate unknown run IDs dedupe before chunking', async () => {
    await activateProjectedMode();
    await discoverRunDataKeys([mkRun('dup1'), mkRun('dup1'), mkRun('dup1')], ['exp-dup']);
    expect(dataKeysCalls()).toHaveLength(1);
    expect(dataKeysCalls()[0].run_ids).toEqual(['dup1']);
  });

  test('concurrent calls never duplicate in-flight run IDs and merge out-of-order responses', async () => {
    await activateProjectedMode();
    const pending: ((value: unknown) => void)[] = [];
    jest
      .mocked(MlflowService.getRunDataKeys)
      .mockImplementation(() => new Promise((resolve) => pending.push(resolve)) as any);
    const first = discoverRunDataKeys([mkRun('o1'), mkRun('o2')], ['exp-overlap']);
    const second = discoverRunDataKeys([mkRun('o2'), mkRun('o3')], ['exp-overlap']);
    expect(dataKeysCalls()).toHaveLength(2);
    expect(dataKeysCalls()[0].run_ids).toEqual(['o1', 'o2']);
    expect(dataKeysCalls()[1].run_ids).toEqual(['o3']);
    // Resolve out of order; both merge additively.
    pending[1]({ metric_keys: ['overlap-m'], param_keys: [], tag_keys: [] });
    await second;
    pending[0]({ metric_keys: ['initial-m'], param_keys: [], tag_keys: [] });
    await first;
    expect(getDiscoveredRunDataKeys(['exp-overlap']).metric_keys).toEqual(['initial-m', 'overlap-m']);
    // Completed IDs are known and in-flight flags cleared.
    expect(await discoverRunDataKeys([mkRun('o1'), mkRun('o2'), mkRun('o3')], ['exp-overlap'])).toBe(false);
    expect(dataKeysCalls()).toHaveLength(2);
  });

  test('pre-switch discovery responses are discarded; post-switch discovery applies', async () => {
    await activateProjectedMode();
    const pending: ((value: unknown) => void)[] = [];
    jest
      .mocked(MlflowService.getRunDataKeys)
      .mockImplementation(() => new Promise((resolve) => pending.push(resolve)) as any);
    const preSwitch = discoverRunDataKeys([mkRun('pre1')], ['exp-iso']);
    resetRunProjectionScope(['exp-iso']);
    pending[0]({ metric_keys: ['pre-switch'], param_keys: [], tag_keys: [] });
    expect(await preSwitch).toBe(false);
    expect(getDiscoveredRunDataKeys(['exp-iso'])).toEqual({ metric_keys: [], param_keys: [], tag_keys: [] });
    expect(isRunProjectionActive()).toBe(false);

    // A post-switch lite search re-activates projected mode, then discovery applies.
    await activateProjectedMode();
    const postSwitch = discoverRunDataKeys([mkRun('post1')], ['exp-iso']);
    pending[1]({ metric_keys: ['post-switch'], param_keys: [], tag_keys: [] });
    expect(await postSwitch).toBe(true);
    expect(getDiscoveredRunDataKeys(['exp-iso']).metric_keys).toEqual(['post-switch']);
  });

  test('experiment switch resets catalogs of other experiments', async () => {
    await activateProjectedMode();
    await discoverRunDataKeys([mkRun('r1')], ['exp-iso']);
    resetRunProjectionScope(['exp2']);
    expect(getDiscoveredRunDataKeys(['exp-iso'])).toEqual({ metric_keys: [], param_keys: [], tag_keys: [] });
    expect(isRunProjectionActive()).toBe(false);
  });

  test('failure applies no partial catalog and sticky-disables projection to legacy', async () => {
    await activateProjectedMode();
    jest.mocked(MlflowService.getRunDataKeys).mockRejectedValue(new Error('data-keys unavailable'));
    expect(await discoverRunDataKeys([mkRun('fail-1')], ['exp-fail'])).toBe(false);
    expect(getDiscoveredRunDataKeys(['exp-fail'])).toEqual({ metric_keys: [], param_keys: [], tag_keys: [] });

    // Sticky disable stops further data-keys attempts...
    await discoverRunDataKeys([mkRun('fail-2')], ['exp-fail']);
    expect(MlflowService.getRunDataKeys).toHaveBeenCalledTimes(1);

    // ...and subsequent searches fall back to legacy.
    const legacy = makeLegacySearch();
    const lite = makeLiteSearch();
    await routeProjectedSearchRuns(
      {
        experiment_ids: ['1'],
        projection: { metric_keys: [], param_keys: [], tag_keys: [], include_inputs: false, include_outputs: false },
      },
      legacy,
      lite,
    );
    expect(lite).not.toHaveBeenCalled();
    expect(legacy).toHaveBeenCalledTimes(1);
  });

  test('buildRunProjection returns null after a data-keys failure (gate closed)', async () => {
    await activateProjectedMode();
    jest.mocked(MlflowService.getRunDataKeys).mockRejectedValue(new Error('data-keys unavailable'));
    await discoverRunDataKeys([mkRun('fail-1')], ['exp-fail']);
    expect(buildRunProjection({ orderByKey: null, datasetsFilter: [] }, {})).toBeNull();
  });
});
