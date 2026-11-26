import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useDispatch } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { loadMoreRunsApi, searchRunsApi } from '../../../actions';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { createSearchRunsParams, fetchModelVersionsForRuns } from '../utils/experimentPage.fetch-utils';
import type { ExperimentRunsSelectorResult } from '../utils/experimentRuns.selector';
import { experimentRunsSelector } from '../utils/experimentRuns.selector';
import type { ExperimentQueryParamsSearchFacets } from './useExperimentPageSearchFacets';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import { PredefinedError } from '@databricks/web-shared/errors';
import { mapErrorWrapperToPredefinedError } from '../../../../common/utils/ErrorUtils';
import { shouldUsePredefinedErrorsInExperimentTracking } from '../../../../common/utils/FeatureUtils';
import Utils from '../../../../common/utils/Utils';
import { useExperimentRunsAutoRefresh } from './useExperimentRunsAutoRefresh';
import type { RunEntity, SearchRunsApiResponse } from '../../../types';
import type { RunProjection } from '../../../runProjection/runProjection';
import { buildRunProjection, discoverRunDataKeys, resetRunProjectionScope } from '../../../runProjection/runProjection';

export type FetchRunsHookParams = ReturnType<typeof createSearchRunsParams> & {
  requestedFacets: ExperimentPageSearchFacetsState & {
    /** Stable projection hash stamped at request time so stale-projection responses can be discarded. */
    __projectionHash: string | null;
  };
  /** Field projection for lightweight (search-lite) reads; absent when the server does not support them. */
  projection?: RunProjection;
};

export type FetchRunsHookFunction = (
  params: FetchRunsHookParams,
  options?: {
    isAutoRefreshing?: boolean;
    discardResultsFn?: (lastRequestedParams: FetchRunsHookParams, response?: SearchRunsApiResponse) => boolean;
  },
) => Promise<{ runs: RunEntity[]; next_page_token?: string }>;

// Calculate actual params to use for fetching runs. Attaches the current field projection (when
// the server supports projected reads) and stamps requestedFacets with its stable hash so
// auto-refresh discard checks can detect stale-projection responses.
const createFetchRunsRequestParams = (
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  runsPinned: string[],
  uiState: ExperimentPageUIState,
): FetchRunsHookParams | null => {
  if (!searchFacets || !experimentIds.length) {
    return null;
  }
  const projectionWithHash = buildRunProjection(searchFacets, uiState);
  const searchParams = createSearchRunsParams(experimentIds, { ...searchFacets, runsPinned }, Date.now());
  return {
    ...(projectionWithHash ? { projection: projectionWithHash.projection } : {}),
    ...searchParams,
    requestedFacets: {
      ...searchFacets,
      __projectionHash: projectionWithHash ? projectionWithHash.projectionHash : null,
    },
  };
};

/**
 * This hook will request for new runs data based on the current experiment page search facets and UI state.
 * Replaces GetExperimentRunsContext and a substantial portion of <ExperimentRuns> component stack.
 */
export const useExperimentRuns = (
  uiState: ExperimentPageUIState,
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  disabled = false,
) => {
  const dispatch = useDispatch<ThunkDispatch>();

  const [runsData, setRunsData] = useState<ExperimentRunsSelectorResult>(() => createEmptyRunsResult());

  const enableWorkspaceModelsRegistryCall = true;

  const persistKey = useMemo(() => (experimentIds ? JSON.stringify(experimentIds.sort()) : null), [experimentIds]);
  // Canonical serialization of the search facets: genuine filter changes alter it, while local
  // state hydration replacing equivalent facet objects (fresh identities) does not.
  const serializedSearchFacets = useMemo(() => JSON.stringify(searchFacets), [searchFacets]);
  // Stable projection hash: identical across the pending -> true capability transition (the hash
  // covers projection content only) and across fresh-but-equivalent UI state identities.
  const projectionHash = useMemo(() => {
    const projectionWithHash = buildRunProjection(searchFacets, uiState);
    return projectionWithHash ? projectionWithHash.projectionHash : null;
  }, [searchFacets, uiState]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);
  const [isInitialLoadingRuns, setIsInitialLoadingRuns] = useState(true);
  const [nextPageToken, setNextPageToken] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<ErrorWrapper | Error | null>(null);
  const cachedPinnedRuns = useRef<string[]>([]);

  const lastFetchedTime = useRef<number | null>(null);
  const lastRequestedParams = useRef<FetchRunsHookParams | null>(null);
  const lastSuccessfulRequestedParams = useRef<FetchRunsHookParams | null>(null);

  // Reset initial loading state when experiment IDs change
  useEffect(() => {
    if (disabled) {
      return;
    }
    // Drop the key-discovery catalogs of experiments no longer in scope and discard any
    // in-flight pre-switch discovery responses.
    resetRunProjectionScope(experimentIds);
    setIsInitialLoadingRuns(true);
    setRunsData(createEmptyRunsResult());
    // eslint-disable-next-line react-hooks/exhaustive-deps -- persistKey is the canonical serialization of experimentIds; depending on the raw identity would reset on equivalent hydration.
  }, [persistKey, disabled]);

  const setResultRunsData = useCallback(
    (store: ReduxState, experimentIds: string[], requestedFacets: ExperimentPageSearchFacetsState) => {
      setRunsData(
        experimentRunsSelector(store, {
          datasetsFilter: requestedFacets.datasetsFilter,
          lifecycleFilter: requestedFacets.lifecycleFilter,
          modelVersionFilter: requestedFacets.modelVersionFilter,
          // In the new version of the view state, experiment IDs are used instead of full experiment entities:
          experiments: [],
          experimentIds,
        }),
      );
    },
    [],
  );

  useEffect(() => {
    cachedPinnedRuns.current = uiState.runsPinned;
  }, [uiState.runsPinned]);

  const loadModelVersions = useCallback(
    (runs: Parameters<typeof fetchModelVersionsForRuns>[0]) => {
      const handleModelVersionLoadFailure = (error: Error | ErrorWrapper) => {
        const normalizedError =
          (error instanceof ErrorWrapper ? mapErrorWrapperToPredefinedError(error) : error) ?? error;
        const message =
          normalizedError instanceof ErrorWrapper ? normalizedError.getMessageField() : normalizedError.message;
        Utils.displayGlobalErrorNotification(`Failed to load model versions for runs: ${message}`);
      };
      if (enableWorkspaceModelsRegistryCall) {
        fetchModelVersionsForRuns(runs || [], searchModelVersionsApi, dispatch).catch(handleModelVersionLoadFailure);
      }
    },
    [dispatch, enableWorkspaceModelsRegistryCall],
  );

  // Main function for fetching runs
  const fetchRuns: FetchRunsHookFunction = useCallback(
    (fetchParams, options = {}) =>
      dispatch((thunkDispatch: ThunkDispatch, getStore: () => ReduxState) => {
        // If we're auto-refreshing, we don't want to show the loading spinner and
        // we don't want to update the last requested params - they're used to determine
        // whether to discard results when the automatically fetched data changes.
        if (!options.isAutoRefreshing) {
          setIsLoadingRuns(true);
          lastRequestedParams.current = fetchParams;
        }
        return thunkDispatch((fetchParams.pageToken ? loadMoreRunsApi : searchRunsApi)(fetchParams))
          .then(async ({ value }) => {
            lastFetchedTime.current = Date.now();

            setIsLoadingRuns(false);
            setIsInitialLoadingRuns(false);
            setRequestError(null);

            if (lastRequestedParams.current && options.discardResultsFn?.(lastRequestedParams.current, value)) {
              return value;
            }

            lastSuccessfulRequestedParams.current = fetchParams;
            setNextPageToken(value.next_page_token || null);

            // We rely on redux reducer to update the state with new runs data,
            // then we pick it up from the store. This benefits other pages that use same data
            // from the same store slice (e.g. run details page). Will be changed when moving to graphQL.
            setResultRunsData(getStore(), fetchParams.experimentIds, fetchParams.requestedFacets);

            // In the end, load model versions for the fetched runs
            loadModelVersions(value.runs || []);

            // Discover metric/param/tag key names for the fetched runs (projected mode only) and
            // recompute the selectors once the catalog grew, so column/chart selectors see the
            // discovered keys without hydrating all values.
            discoverRunDataKeys(value.runs || [], fetchParams.experimentIds).then((applied) => {
              if (!applied) {
                return;
              }
              const latestParams = lastRequestedParams.current || fetchParams;
              setResultRunsData(getStore(), latestParams.experimentIds, latestParams.requestedFacets);
            });
            return value;
          })
          .catch((e: ErrorWrapper | PredefinedError) => {
            setIsLoadingRuns(false);
            setIsInitialLoadingRuns(false);
            if (shouldUsePredefinedErrorsInExperimentTracking()) {
              // If it's already a PredefinedError, we don't need to map it again
              if (e instanceof PredefinedError) {
                setRequestError(e);
                return;
              }
              const maybePredefinedError = mapErrorWrapperToPredefinedError(e);
              if (maybePredefinedError) {
                setRequestError(maybePredefinedError);
                return;
              }
            }
            setRequestError(e);
            if (!shouldUsePredefinedErrorsInExperimentTracking()) {
              Utils.logErrorAndNotifyUser(e);
            }
          });
      }),
    [dispatch, setResultRunsData, loadModelVersions],
  );

  // Fetch runs when new request params are available
  // (e.g. after search facets change). Depends on stable serialized representations instead of
  // object identities: local state hydration replaces equivalent searchFacets once during
  // initialization, which used to issue a duplicate identical request.
  useEffect(() => {
    if (disabled) {
      return;
    }
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds, cachedPinnedRuns.current, uiState);
    if (requestParams) {
      fetchRuns(requestParams);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- persistKey/serializedSearchFacets/projectionHash are the canonical serializations of experimentIds/searchFacets/projection-relevant UI state; depending on the raw identities would refetch on equivalent hydration.
  }, [fetchRuns, dispatch, disabled, persistKey, serializedSearchFacets, projectionHash]);

  const loadMoreRuns = async () => {
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds, cachedPinnedRuns.current, uiState);
    if (!nextPageToken || !requestParams) {
      return [];
    }
    return fetchRuns({ ...requestParams, pageToken: nextPageToken });
  };

  const refreshRuns = useCallback(() => {
    if (lastSuccessfulRequestedParams.current) {
      fetchRuns({ ...lastSuccessfulRequestedParams.current, pageToken: undefined });
    }
  }, [fetchRuns]);

  useExperimentRunsAutoRefresh({
    experimentIds,
    fetchRuns,
    searchFacets,
    enabled: uiState.autoRefreshEnabled,
    cachedPinnedRuns,
    runsData,
    isLoadingRuns: isLoadingRuns,
    lastFetchedTime,
    uiState,
  });

  return {
    isLoadingRuns,
    moreRunsAvailable: Boolean(nextPageToken),
    refreshRuns,
    loadMoreRuns,
    isInitialLoadingRuns,
    runsData,
    requestError,
  };
};

const createEmptyRunsResult = () => ({
  datasetsList: [],
  experimentTags: {},
  metricKeyList: [],
  tagKeyList: [],
  metricsList: [],
  modelVersionsByRunUuid: {},
  paramKeyList: [],
  paramsList: [],
  runInfos: [],
  runUuidsMatchingFilter: [],
  tagsList: [],
  inputsOutputsList: [],
});
