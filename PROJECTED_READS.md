# Projected run reads

This branch adds opt-in lightweight run reads for metric-heavy experiments. It is
based on MLflow `v3.18.0` (`ba41106d110b67096384da12b6b85473457598db`).

## Motivation

Legacy `SearchRuns` and `GetRun` eagerly load and serialize every latest metric,
parameter, and tag for each returned run. A page of ten runs can therefore
contain tens of thousands of values even when the experiment UI needs no metric
values, or only the keys used by a few charts.

Projected reads preserve the legacy APIs while adding explicit alternatives:

```text
POST /api/2.0/mlflow/runs/search-lite
POST /api/2.0/mlflow/runs/get-lite
POST /api/2.0/mlflow/runs/data-keys
```

Each is also registered under `/ajax-api/2.0`. `/server-info` advertises
`run_projection_enabled` only for SQL-backed tracking stores.

## API behavior

`search-lite` accepts the usual search inputs plus:

```json
{
  "projection": {
    "metric_keys": ["loss", "accuracy"],
    "param_keys": [],
    "tag_keys": ["mlflow.parentRunId"],
    "include_inputs": false,
    "include_outputs": false
  }
}
```

Omitted or empty child-key lists return no child values. `max_results` defaults
to a bounded page of ten. Search filtering, ordering, lifecycle views, page
tokens, selected values, run-name fallback, and requested input/output metadata
match legacy behavior. Legacy `SearchRuns` and `GetRun` are unchanged.

`data-keys` accepts explicit `run_ids` and returns distinct metric, parameter,
and tag names for selector/catalog use. Projection lists and run-ID lists are
bounded and validated. No schema migration is required.

The SQL implementation reuses the legacy query builders but omits eager child
hydration, trims the N+1 pagination lookahead before child reads, and bulk-loads
only requested keys. Empty collections cause no corresponding child query.
Basic-auth, workspace scoping, and model-link redaction are preserved. Non-SQL
stores return `NOT_IMPLEMENTED` and do not advertise the capability.

## Experiment UI

The TypeScript implementation lives in
`mlflow/server/js/src/experiment-tracking/runProjection/runProjection.ts` and:

- returns no metric or parameter values for the ordinary Runs table;
- adds keys required by visible columns, grouping, sorting, and saved charts;
- requests Dataset/Model inputs and outputs only when needed;
- discovers selector key names separately, with caching, in-flight dedupe, and
  at most 1000 run IDs per request;
- uses the same projection for initial loads, pagination, manual refresh,
  auto-refresh, pinned runs, and evaluation-run searches;
- waits for the shared server-info query before choosing lite or legacy;
- coalesces identical concurrent searches and supersedes stale responses;
- falls back to legacy on unsupported endpoints or discovery failure without
  retry storms;
- labels CSV export as containing visible fields in projected mode.

Run-page GraphQL and parent-run `getRun` reads are outside this slice. A newly
introduced metric key on an already cached run is discovered after a browser
reload rather than on every auto-refresh.

## Source map

- Backend: `mlflow/server/projected_runs.py`
- Registration/capability: `mlflow/server/handlers.py`,
  `mlflow/utils/server_info.py`
- Authorization/redaction: `mlflow/server/auth/__init__.py`
- Frontend runtime: `mlflow/server/js/src/experiment-tracking/runProjection/`
- Backend tests: `tests/server/test_projected_runs.py`,
  `tests/server/auth/test_projected_runs_auth.py`
- Frontend tests: `runProjection.test.ts` and `actions.projection.test.ts`

## Validation

The implementation was validated against SQLite, disposable Postgres 15, and a
restored metric-heavy production copy. Backend suites: 55 general and 9
auth/redaction tests on each database. Frontend: 50 new Jest tests, with
affected existing suites also passing. TypeScript and lint checks are clean.

Production-copy API comparison for the same ten runs:

| Request | Legacy | Projected |
|---|---:|---:|
| Default table with required tags and IO | 1831.2 ms / 6,323,193 B | 14.1 ms / 59,519 B |
| Nine selected chart metrics | 1701.5 ms / 6,323,193 B | 25.9 ms / 20,476 B |

The nine-metric request returns exactly 90 selected latest values instead of
31,191. Two consecutive headless Chromium runs verified the actual experiment
page, key discovery, adding a selected metric through the Columns UI, zero
legacy `runs/search` calls, and zero browser runtime errors.

## Maintenance

Keep this work on a feature branch. Fetch official changes from `upstream` and
push the branch to `origin` (the fork). Rebase or merge deliberately when
moving to a new MLflow release; do not apply the deployment repository's
minified-bundle patcher to this source branch. Run the focused backend and
frontend suites, then full CI before opening an upstream pull request.
