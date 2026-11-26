import { describe, beforeEach, test, expect, jest } from '@jest/globals';
import { searchRunsPayload } from './actions';
import { MlflowService } from './sdk/MlflowService';
import { ensureServerInfo, getRunProjectionEnabledSync } from './hooks/useServerInfo';
import { resetRunProjectionStateForTests } from './runProjection/runProjection';
import type { RunProjection } from './runProjection/runProjection';

jest.mock('./sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
    searchRunsLite: jest.fn(),
    getRun: jest.fn(),
    getRunDataKeys: jest.fn(),
  },
}));

jest.mock('./hooks/useServerInfo', () => ({
  getRunProjectionEnabledSync: jest.fn(),
  ensureServerInfo: jest.fn(),
}));

let capability: boolean | undefined;

const projection: RunProjection = {
  metric_keys: ['m'],
  param_keys: [],
  tag_keys: [],
  include_inputs: false,
  include_outputs: false,
};

const mkRun = (id: string, tags: { key: string; value: string }[] = []) =>
  ({ info: { runUuid: id, run_id: id }, data: { metrics: [], params: [], tags } }) as any;

const baseParams = () => ({
  experimentIds: ['exp1'],
  filter: undefined,
  runViewType: 'ACTIVE',
  shouldFetchParents: false,
  runsPinned: [] as string[],
});

const liteMock = () => jest.mocked(MlflowService.searchRunsLite);
const legacyMock = () => jest.mocked(MlflowService.searchRuns);

beforeEach(() => {
  resetRunProjectionStateForTests();
  capability = true;
  jest.mocked(getRunProjectionEnabledSync).mockClear();
  jest.mocked(getRunProjectionEnabledSync).mockImplementation(() => capability);
  jest.mocked(ensureServerInfo).mockClear();
  jest.mocked(ensureServerInfo).mockImplementation(() => Promise.resolve({} as any));
  legacyMock().mockReset();
  legacyMock().mockResolvedValue({ runs: [mkRun('legacy-r1')], next_page_token: null } as any);
  liteMock().mockReset();
  liteMock().mockResolvedValue({ runs: [mkRun('lite-r1')], next_page_token: null } as any);
  jest.mocked(MlflowService.getRun).mockReset();
  jest.mocked(MlflowService.getRun).mockResolvedValue({ run: mkRun('parent') } as any);
});

describe('searchRunsPayload projected routing', () => {
  test('routes the projected primary search to search-lite', async () => {
    const response = await searchRunsPayload({ ...baseParams(), projection });
    expect(liteMock()).toHaveBeenCalledTimes(1);
    expect(legacyMock()).not.toHaveBeenCalled();
    expect(liteMock().mock.calls[0][0].projection).toEqual(projection);
    expect(response.runsMatchingFilter).toEqual(response.runs);
  });

  test('pinned-run second search uses the same projection with max_results = pinned count', async () => {
    const pinnedIds = Array.from({ length: 25 }, (_, index) => `pin-${index}`);
    await searchRunsPayload({ ...baseParams(), runsPinned: pinnedIds, projection });
    expect(liteMock()).toHaveBeenCalledTimes(2);
    expect(legacyMock()).not.toHaveBeenCalled();
    expect(liteMock().mock.calls[1][0].projection).toEqual(projection);
    expect(liteMock().mock.calls[1][0].max_results).toBe(25);
    expect(liteMock().mock.calls[1][0].run_view_type).toBe('ALL');
  });

  test('calls without a projection (eval/compare) stay on legacy search', async () => {
    const response = await searchRunsPayload(baseParams());
    expect(liteMock()).not.toHaveBeenCalled();
    expect(legacyMock()).toHaveBeenCalledTimes(1);
    expect(response.runs[0].info.runUuid).toBe('legacy-r1');
  });

  test('explicit capability false falls back to legacy with the projection stripped', async () => {
    capability = false;
    await searchRunsPayload({ ...baseParams(), projection });
    expect(liteMock()).not.toHaveBeenCalled();
    expect(legacyMock()).toHaveBeenCalledTimes(1);
    expect(legacyMock().mock.calls[0][0].projection).toBeUndefined();
    expect('projection' in JSON.parse(JSON.stringify(legacyMock().mock.calls[0][0]))).toBe(false);
  });

  test('fetchMissingParents getRun path remains legacy and safe with lite results', async () => {
    const response = await searchRunsPayload({
      ...baseParams(),
      shouldFetchParents: true,
      projection,
    });
    // Lite runs carry no parent tag, so no parent fetches happen; getRun stays the legacy path.
    expect(MlflowService.getRun).not.toHaveBeenCalled();
    expect(response.runs).toHaveLength(1);
  });

  test('identical concurrent projected payloads coalesce into one network call', async () => {
    const pending: ((value: unknown) => void)[] = [];
    liteMock().mockImplementation(() => new Promise((resolve) => pending.push(resolve)) as any);
    const semantics = () => ({
      experimentIds: ['exp1'],
      filter: undefined,
      runViewType: undefined,
      maxResults: undefined,
      orderBy: undefined,
      pageToken: undefined,
      shouldFetchParents: undefined,
      runsPinned: undefined,
      projection,
    });
    const first = searchRunsPayload(semantics());
    // Normalized defaults make these two payloads semantically identical.
    const second = searchRunsPayload({
      ...semantics(),
      filter: '',
      runViewType: 'ACTIVE_ONLY',
      maxResults: 100,
      orderBy: [],
      pageToken: null,
      shouldFetchParents: false,
      runsPinned: [],
    });
    expect(liteMock()).toHaveBeenCalledTimes(1);
    pending[0]({ runs: [mkRun('shared')], next_page_token: null });
    const [firstResponse, secondResponse] = await Promise.all([first, second]);
    expect(firstResponse).toBe(secondResponse);
    expect(firstResponse.runs[0].info.runUuid).toBe('shared');
  });

  test('an older projected search resolves with the newest response', async () => {
    const pending: ((value: unknown) => void)[] = [];
    liteMock().mockImplementation(() => new Promise((resolve) => pending.push(resolve)) as any);
    const older = searchRunsPayload({ ...baseParams(), projection });
    const newer = searchRunsPayload({
      ...baseParams(),
      projection: { ...projection, metric_keys: ['other'] },
    });
    expect(liteMock()).toHaveBeenCalledTimes(2);
    pending[1]({ runs: [mkRun('rB')], next_page_token: null });
    pending[0]({ runs: [mkRun('rA')], next_page_token: null });
    const [olderResponse, newerResponse] = await Promise.all([older, newer]);
    expect(newerResponse.runs[0].info.runUuid).toBe('rB');
    expect(olderResponse.runs[0].info.runUuid).toBe('rB');
  });

  test('different page tokens stay independent and never supersede each other', async () => {
    const pending: ((value: unknown) => void)[] = [];
    liteMock().mockImplementation(() => new Promise((resolve) => pending.push(resolve)) as any);
    const page1 = searchRunsPayload({ ...baseParams(), projection });
    const page2 = searchRunsPayload({ ...baseParams(), pageToken: 'tok2', projection });
    pending[0]({ runs: [mkRun('page1')], next_page_token: null });
    pending[1]({ runs: [mkRun('page2')], next_page_token: null });
    const [page1Response, page2Response] = await Promise.all([page1, page2]);
    expect(page1Response.runs[0].info.runUuid).toBe('page1');
    expect(page2Response.runs[0].info.runUuid).toBe('page2');
  });
});
