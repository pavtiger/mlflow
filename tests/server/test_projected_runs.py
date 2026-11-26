"""Integration tests for the projected (lightweight) run read endpoints.

Covers: legacy parity, omitted/empty projection, selected-key parity,
validation, filters/ordering parity, pagination across the N+1 boundary,
lifecycle views, runName fallback, fixed query counts, data-keys, get-lite,
server-info capability, artifacts-only mode, and api/ajax path parity.

The Flask app and the tracking store are process-global, so the seeded
database is created once per module and shared by all tests in this file. Set
PROJECTION_TEST_DB_URI to run the suite against a different backend (e.g.
Postgres) instead of the default SQLite file.
"""

import json
import os

import pytest

import mlflow
from mlflow.server import handlers
from mlflow.server.constants import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR

SEARCH_LITE = "/api/2.0/mlflow/runs/search-lite"
GET_LITE = "/api/2.0/mlflow/runs/get-lite"
DATA_KEYS = "/api/2.0/mlflow/runs/data-keys"

DELETED_RUN_INDEXES = {2, 7, 13}
# Runs 25/26 are appended after seeding with the legacy pre-run-name column
# shape (runs.name empty): 25 carries an mlflow.runName tag, 26 carries none.
LEGACY_NAMED_INDEX = 25
LEGACY_NONAME_INDEX = 26

# Projects every key the seed writes, for exact parity with legacy hydration.
FULL_PROJECTION = {
    "metric_keys": ["acc", "loss", "noise_0", "noise_1", "noise_2", "noise_3", "noise_4"],
    "param_keys": ["batch_size", "lr", "unselected_param"],
    "tag_keys": ["mlflow.runName", "stage", "team", "unselected_tag"],
}


def _seed(db_uri):
    """Create an experiment with 25 runs of varying child-data cardinality."""
    from mlflow import MlflowClient

    tracking = MlflowClient(tracking_uri=db_uri)
    experiment_id = tracking.create_experiment("projection-test")

    run_ids = []
    for i in range(25):
        tags = {}
        if i % 4 != 0:
            tags["mlflow.runName"] = f"run-{i:02d}"
        else:
            tags["mlflow.runName"] = f"tagrun-{i:02d}"
        run = tracking.create_run(experiment_id, tags=tags)
        run_id = run.info.run_id
        run_ids.append(run_id)
        tracking.log_metric(run_id, "loss", round(1.0 / (i + 1), 6), step=0)
        tracking.log_metric(run_id, "acc", i / 100.0, step=0)
        # Unselected cardinality grows with i (0..5 extra metric keys per run).
        for j in range(i % 6):
            tracking.log_metric(run_id, f"noise_{j}", float(j))
        tracking.log_param(run_id, "lr", f"0.0{i % 10}")
        tracking.log_param(run_id, "batch_size", str(16 * (i % 4 + 1)))
        tracking.log_param(run_id, "unselected_param", f"u{i}")
        tracking.set_tag(run_id, "team", "a" if i % 2 else "b")
        tracking.set_tag(run_id, "stage", f"s{i % 3}")
        tracking.set_tag(run_id, "unselected_tag", f"t{i}")
        tracking.set_terminated(run_id)

    for i in sorted(DELETED_RUN_INDEXES):
        tracking.delete_run(run_ids[i])

    store = handlers._get_tracking_store()

    # Run 3 carries dataset inputs, a model input, and a model output so IO
    # projection parity is tested against real rows (legacy helpers read the
    # ``inputs`` edge table).
    from mlflow.entities import Dataset, DatasetInput, InputTag, LoggedModelOutput
    from mlflow.store.tracking.dbmodels.models import SqlInput

    io_run_id = run_ids[3]
    dataset = Dataset(name="seed-dataset", digest="d1", source_type="s3", source="s3://bucket/src")
    tracking.log_inputs(
        io_run_id, datasets=[DatasetInput(dataset=dataset, tags=[InputTag("context", "train")])]
    )
    tracking.log_outputs(io_run_id, models=[LoggedModelOutput(model_id="mo-1", step=0)])
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlInput(
                input_uuid="test-model-input-1",
                source_type="RUN_INPUT",
                source_id=io_run_id,
                destination_type="MODEL_INPUT",
                destination_id="mi-1",
            )
        )

    # Simulate legacy (pre-run-name) rows: the server always generates a name
    # today, so blank the name column by hand. One run keeps its mlflow.runName
    # tag (fallback source), the other loses it (no fallback available).
    from mlflow.store.tracking.dbmodels.models import SqlRun

    legacy_named = tracking.create_run(
        experiment_id, tags={"mlflow.runName": "legacy-named"}
    ).info.run_id
    tracking.log_metric(legacy_named, "loss", 0.25)
    tracking.log_metric(legacy_named, "acc", 0.75)
    legacy_noname = tracking.create_run(
        experiment_id, tags={"mlflow.runName": "to-drop"}
    ).info.run_id
    tracking.log_metric(legacy_noname, "loss", 0.5)
    tracking.log_metric(legacy_noname, "acc", 0.5)
    tracking.set_terminated(legacy_named)
    tracking.set_terminated(legacy_noname)
    with store.ManagedSessionMaker(read_only=False) as session:
        named_row = session.query(SqlRun).filter(SqlRun.run_uuid == legacy_named).one()
        named_row.name = None
        noname_row = session.query(SqlRun).filter(SqlRun.run_uuid == legacy_noname).one()
        noname_row.name = None
        for tag in noname_row.tags:
            if tag.key == "mlflow.runName":
                session.delete(tag)
    run_ids.extend([legacy_named, legacy_noname])

    return experiment_id, run_ids


@pytest.fixture(scope="module")
def db_uri(tmp_path_factory):
    """Point the global tracking store at a throwaway database for this module."""
    tmpdir = tmp_path_factory.mktemp("projected-runs")
    db_uri = os.environ.get("PROJECTION_TEST_DB_URI") or f"sqlite:///{tmpdir}/mlflow.db"
    artifact_root = tmpdir / "artifacts"
    artifact_root.mkdir()

    overrides = {
        BACKEND_STORE_URI_ENV_VAR: db_uri,
        ARTIFACT_ROOT_ENV_VAR: str(artifact_root),
        "MLFLOW_ARTIFACT_ROOT": str(artifact_root),
    }
    saved_env = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    saved_tracking_uri = mlflow.get_tracking_uri()
    saved_store = handlers._tracking_store
    handlers._tracking_store = None
    try:
        yield db_uri
    finally:
        handlers._tracking_store = saved_store
        mlflow.set_tracking_uri(saved_tracking_uri)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="module")
def seeded(db_uri):
    experiment_id, run_ids = _seed(db_uri)
    return experiment_id, run_ids


@pytest.fixture(scope="module")
def http(seeded):
    from mlflow.server import app as flask_app

    class Client:
        def __init__(self, inner):
            self.inner = inner

        def post(self, path, body):
            resp = self.inner.post(path, data=json.dumps(body), content_type="application/json")
            try:
                return resp.status_code, json.loads(resp.get_data(as_text=True))
            except json.JSONDecodeError:
                return resp.status_code, resp.get_data(as_text=True)

        def get(self, path, params=None):
            resp = self.inner.get(path, query_string=params or {})
            try:
                return resp.status_code, json.loads(resp.get_data(as_text=True))
            except json.JSONDecodeError:
                return resp.status_code, resp.get_data(as_text=True)

    return Client(flask_app.test_client())


def active_run_ids(all_run_ids):
    return [rid for i, rid in enumerate(all_run_ids) if i not in DELETED_RUN_INDEXES]


def comparable(run, collection):
    return sorted(
        (item["key"], json.dumps(item.get("value", item.get("value"))))
        for item in run["data"].get(collection, [])
    )


# ---------------------------------------------------------------------------
# Legacy behavior
# ---------------------------------------------------------------------------


def test_legacy_search_unchanged(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": [experiment_id],
            "max_results": 100,
            "order_by": ["attributes.start_time ASC"],
        },
    )
    assert sc == 200
    by_name = {r["info"].get("run_name", ""): r for r in body["runs"]}
    run = by_name["run-05"]
    # Legacy still hydrates everything, including inputs/outputs and unselected keys.
    assert "inputs" in run
    assert "outputs" in run
    metric_keys = {m["key"] for m in run["data"]["metrics"]}
    assert "noise_0" in metric_keys
    assert {p["key"] for p in run["data"]["params"]} >= {"lr", "batch_size", "unselected_param"}


def test_legacy_get_run_unchanged(http, seeded):
    _, run_ids = seeded
    sc, body = http.get("/api/2.0/mlflow/runs/get", {"run_id": run_ids[5]})
    assert sc == 200
    assert "inputs" in body["run"]
    assert "outputs" in body["run"]
    assert {m["key"] for m in body["run"]["data"]["metrics"]} >= {"acc", "loss", "noise_0"}


# ---------------------------------------------------------------------------
# Projection semantics
# ---------------------------------------------------------------------------


def test_search_lite_omitted_projection_returns_no_values(http, seeded):
    experiment_id, run_ids = seeded
    sc, body = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "max_results": 100,
            "order_by": ["attributes.start_time ASC"],
        },
    )
    assert sc == 200
    assert len(body["runs"]) == len(active_run_ids(run_ids))
    assert "next_page_token" not in body
    for run in body["runs"]:
        assert run["data"] == {}
        # Core RunInfo fields are always present; end_time/run_name are omitted
        # when null/empty, exactly like legacy proto3 JSON.
        assert {
            "run_id",
            "experiment_id",
            "status",
            "start_time",
            "lifecycle_stage",
            "artifact_uri",
        } <= set(run["info"])


def test_search_lite_empty_projection_equals_omitted(http, seeded):
    experiment_id, _ = seeded
    sc_omit, body_omit = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 100}
    )
    sc_empty, body_empty = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 100, "projection": {}}
    )
    assert sc_omit == sc_empty == 200
    assert body_omit == body_empty


def test_search_lite_empty_key_lists_select_nothing(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "max_results": 100,
            "projection": {"metric_keys": [], "param_keys": [], "tag_keys": []},
        },
    )
    assert sc == 200
    assert all(run["data"] == {} for run in body["runs"])


def test_search_lite_selected_values_match_legacy(http, seeded):
    experiment_id, run_ids = seeded
    sc_legacy, legacy = http.post(
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": [experiment_id],
            "max_results": 100,
            "order_by": ["attributes.start_time ASC"],
        },
    )
    sc_lite, lite = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "max_results": 100,
            "order_by": ["attributes.start_time ASC"],
            "projection": FULL_PROJECTION,
        },
    )
    assert sc_legacy == sc_lite == 200
    assert len(legacy["runs"]) == len(lite["runs"]) == len(active_run_ids(run_ids))
    for legacy_run, lite_run in zip(legacy["runs"], lite["runs"]):
        assert legacy_run["info"] == lite_run["info"]
        for collection in ("metrics", "params", "tags"):
            assert comparable(legacy_run, collection) == comparable(lite_run, collection)


def test_search_lite_partial_projection_returns_only_selected(http, seeded):
    experiment_id, run_ids = seeded
    sc, body = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "max_results": 100,
            "projection": {"metric_keys": ["acc"]},
        },
    )
    assert sc == 200
    by_id = {run["info"]["run_id"]: run for run in body["runs"]}
    for run_id in active_run_ids(run_ids):
        run = by_id[run_id]
        assert [m["key"] for m in run["data"].get("metrics", [])] == ["acc"]
        assert run["data"].get("params", []) == []
        assert run["data"].get("tags", []) == []


def test_duplicate_projection_keys_are_deduped(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "max_results": 1,
            "projection": {
                "metric_keys": ["acc", "acc", "loss", "acc"],
                "param_keys": ["lr", "lr"],
            },
        },
    )
    assert sc == 200
    keys = [m["key"] for m in body["runs"][0]["data"]["metrics"]]
    assert keys == ["acc", "loss"]  # deduped, deterministic key order


def test_metric_objects_carry_timestamp_and_step(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(GET_LITE, {"run_id": run_ids[0], "projection": {"metric_keys": ["loss"]}})
    assert sc == 200
    (metric,) = body["run"]["data"]["metrics"]
    assert metric["key"] == "loss"
    assert metric["step"] == 0
    assert "timestamp" in metric


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_projection_key_length_limit(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "projection": {"metric_keys": ["x" * 251]}}
    )
    assert sc == 400
    assert body["error_code"] == "INVALID_PARAMETER_VALUE"


def test_projection_key_count_cap(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "projection": {"param_keys": [f"p{i}" for i in range(501)]},
        },
    )
    assert sc == 400
    assert "at most" in body["message"]


def test_unknown_fields_rejected(http, seeded):
    experiment_id, _ = seeded
    sc, _ = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "bogus": 1})
    assert sc == 400
    sc, _ = http.post(
        SEARCH_LITE,
        {"experiment_ids": [experiment_id], "projection": {"metric_keys": ["loss"], "nope": 1}},
    )
    assert sc == 400
    sc, _ = http.post(GET_LITE, {"run_id": "x", "extra": True})
    assert sc == 400
    sc, _ = http.post(DATA_KEYS, {"run_ids": ["a"], "extra": True})
    assert sc == 400


def test_bad_types_rejected(http, seeded):
    experiment_id, _ = seeded
    sc, _ = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "projection": {"metric_keys": "loss"}}
    )
    assert sc == 400
    sc, _ = http.post(
        SEARCH_LITE,
        {"experiment_ids": [experiment_id], "projection": {"include_inputs": "yes"}},
    )
    assert sc == 400
    sc, _ = http.post(SEARCH_LITE, {"experiment_ids": "3"})
    assert sc == 400
    sc, _ = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": "abc"})
    assert sc == 400
    sc, _ = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "run_view_type": "SOMETIMES"}
    )
    assert sc == 400


def test_get_lite_include_inputs_outputs_parity(http, seeded):
    _, run_ids = seeded
    sc_legacy, legacy = http.get("/api/2.0/mlflow/runs/get", {"run_id": run_ids[3]})
    sc, body = http.post(
        GET_LITE,
        {"run_id": run_ids[3], "projection": {"include_inputs": True, "include_outputs": True}},
    )
    assert sc == sc_legacy == 200
    # Exact parity with legacy GetRun: dataset inputs + model inputs, model outputs.
    assert body["run"]["inputs"] == legacy["run"]["inputs"]
    assert body["run"]["outputs"] == legacy["run"]["outputs"]
    assert body["run"]["inputs"]["dataset_inputs"][0]["dataset"]["name"] == "seed-dataset"
    assert body["run"]["inputs"]["model_inputs"] == [{"model_id": "mi-1"}]
    assert body["run"]["outputs"]["model_outputs"] == [{"model_id": "mo-1", "step": 0}]
    # Metric/param/tag behavior is unchanged by the IO flags.
    assert body["run"]["data"] == {}


def test_get_lite_inputs_outputs_omitted_unless_requested(http, seeded):
    _, run_ids = seeded
    for projection in ({}, {"include_inputs": False, "include_outputs": False}):
        sc, body = http.post(GET_LITE, {"run_id": run_ids[3], "projection": projection})
        assert sc == 200
        assert "inputs" not in body["run"]
        assert "outputs" not in body["run"]


def test_get_lite_inputs_only_and_outputs_only(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(GET_LITE, {"run_id": run_ids[3], "projection": {"include_inputs": True}})
    assert sc == 200
    assert "inputs" in body["run"]
    assert "outputs" not in body["run"]
    sc, body = http.post(GET_LITE, {"run_id": run_ids[3], "projection": {"include_outputs": True}})
    assert sc == 200
    assert "outputs" in body["run"]
    assert "inputs" not in body["run"]


def test_search_lite_include_inputs_outputs_parity(http, seeded):
    experiment_id, run_ids = seeded
    base = {
        "experiment_ids": [experiment_id],
        "max_results": 100,
        "order_by": ["attributes.start_time ASC"],
    }
    sc_legacy, legacy = http.post("/api/2.0/mlflow/runs/search", base)
    sc, body = http.post(
        SEARCH_LITE,
        {**base, "projection": {"include_inputs": True, "include_outputs": True}},
    )
    assert sc_legacy == sc == 200
    assert len(legacy["runs"]) == len(body["runs"])
    for legacy_run, lite_run in zip(legacy["runs"], body["runs"]):
        # Legacy SearchRuns always attaches IO; search-lite matches it exactly
        # when the flags are set (legacy search carries dataset_inputs only).
        assert lite_run["inputs"] == legacy_run["inputs"]
        assert lite_run["outputs"] == legacy_run["outputs"]
    by_id = {run["info"]["run_id"]: run for run in body["runs"]}
    assert by_id[run_ids[3]]["inputs"]["dataset_inputs"][0]["dataset"]["name"] == "seed-dataset"
    assert by_id[run_ids[3]]["outputs"]["model_outputs"] == [{"model_id": "mo-1", "step": 0}]


def test_search_lite_inputs_outputs_omitted_by_default(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 100, "projection": {}}
    )
    assert sc == 200
    for run in body["runs"]:
        assert "inputs" not in run
        assert "outputs" not in run


def test_search_lite_io_queries_only_when_requested(seeded):
    from sqlalchemy import event

    from mlflow.entities.view_type import ViewType
    from mlflow.server.projected_runs import RunProjection, search_runs_projected

    store = handlers._get_tracking_store()
    experiment_id, _ = seeded
    base = {
        "experiment_ids": [experiment_id],
        "filter_string": "",
        "run_view_type": ViewType.ACTIVE_ONLY,
        "max_results": 10,
        "order_by": [],
        "page_token": None,
    }

    def io_query_count(projection):
        statements = []

        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(store.engine, "before_cursor_execute", before_cursor_execute)
        try:
            search_runs_projected(store, projection=projection, **base)
        finally:
            event.remove(store.engine, "before_cursor_execute", before_cursor_execute)
        return [s for s in statements if "FROM inputs" in s]

    # Unrequested: the inputs edge table is never queried.
    assert io_query_count(RunProjection()) == []
    assert io_query_count(RunProjection(metric_keys=("acc",))) == []
    # Requested: exactly one bulk query per flag, regardless of page contents.
    both = io_query_count(RunProjection(include_inputs=True, include_outputs=True))
    assert len(both) == 2
    only_in = io_query_count(RunProjection(include_inputs=True))
    assert len(only_in) == 1
    only_out = io_query_count(RunProjection(include_outputs=True))
    assert len(only_out) == 1


def test_max_results_cap_matches_legacy(http, seeded):
    experiment_id, _ = seeded
    for path in (SEARCH_LITE, "/api/2.0/mlflow/runs/search"):
        sc, _ = http.post(path, {"experiment_ids": [experiment_id], "max_results": 50001})
        assert sc == 400


def test_missing_run_id_rejected(http):
    sc, body = http.post(GET_LITE, {"projection": {}})
    assert sc == 400


# ---------------------------------------------------------------------------
# Filters, ordering, lifecycle
# ---------------------------------------------------------------------------

FILTER_ORDER_CASES = [
    ({"filter": "metrics.acc > 0.1", "order_by": ["attributes.start_time ASC"]}),
    ({"filter": "metrics.loss < 0.2", "order_by": ["metrics.acc DESC"]}),
    ({
        "filter": "params.lr = '0.03'",
        "order_by": ["params.batch_size ASC", "attributes.start_time ASC"],
    }),
    ({"filter": "tags.team = 'a'", "order_by": ["tags.stage ASC", "attributes.start_time ASC"]}),
    ({"filter": "", "order_by": ["attributes.start_time DESC"]}),
]


@pytest.mark.parametrize("case", FILTER_ORDER_CASES)
def test_filter_and_ordering_parity(http, seeded, case):
    experiment_id, _ = seeded
    base = {"experiment_ids": [experiment_id], "max_results": 100}
    sc_legacy, legacy = http.post("/api/2.0/mlflow/runs/search", {**base, **case})
    sc_lite, lite = http.post(
        SEARCH_LITE,
        {
            **base,
            **case,
            "projection": {
                "metric_keys": ["acc", "loss"],
                "param_keys": ["lr", "batch_size"],
                "tag_keys": ["team", "stage"],
            },
        },
    )
    assert sc_legacy == sc_lite == 200
    assert [r["info"]["run_id"] for r in legacy["runs"]] == [
        r["info"]["run_id"] for r in lite["runs"]
    ]


@pytest.mark.parametrize("view_type", ["ACTIVE_ONLY", "DELETED_ONLY", "ALL"])
def test_lifecycle_view_parity(http, seeded, view_type):
    experiment_id, run_ids = seeded
    sc_legacy, legacy = http.post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": [experiment_id], "run_view_type": view_type, "max_results": 100},
    )
    sc_lite, lite = http.post(
        SEARCH_LITE,
        {"experiment_ids": [experiment_id], "run_view_type": view_type, "max_results": 100},
    )
    assert sc_legacy == sc_lite == 200
    assert [r["info"]["run_id"] for r in legacy["runs"]] == [
        r["info"]["run_id"] for r in lite["runs"]
    ]


def test_deleted_run_reported_deleted(http, seeded):
    experiment_id, run_ids = seeded
    sc, body = http.post(
        SEARCH_LITE,
        {
            "experiment_ids": [experiment_id],
            "run_view_type": "ALL",
            "max_results": 100,
            "projection": {"tag_keys": ["team"]},
        },
    )
    assert sc == 200
    deleted_id = run_ids[min(DELETED_RUN_INDEXES)]
    by_id = {run["info"]["run_id"]: run for run in body["runs"]}
    assert by_id[deleted_id]["info"]["lifecycle_stage"] == "deleted"
    assert by_id[deleted_id]["data"]["tags"][0]["key"] == "team"


# ---------------------------------------------------------------------------
# Pagination across the N+1 lookahead boundary
# ---------------------------------------------------------------------------


def test_pagination_walk_matches_legacy(http, seeded):
    experiment_id, run_ids = seeded
    expected = active_run_ids(run_ids)

    def walk(path, body):
        seen = []
        while True:
            sc, resp = http.post(path, body)
            assert sc == 200
            seen.extend(r["info"]["run_id"] for r in resp["runs"])
            token = resp.get("next_page_token")
            if not token:
                return seen
            body = {**body, "page_token": token}

    # Odd page sizes straddle the N+1 lookahead boundary in different places.
    # NB: start_time ties are broken by run_uuid (not creation order), so the
    # page-by-page order is compared against legacy, and the full walk against
    # the active set as a whole.
    for page_size in (1, 3, 10, 25, 100):
        lite_ids = walk(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": page_size})
        legacy_ids = walk(
            "/api/2.0/mlflow/runs/search",
            {"experiment_ids": [experiment_id], "max_results": page_size},
        )
        assert lite_ids == legacy_ids
        assert sorted(lite_ids) == sorted(expected)
        assert len(set(lite_ids)) == len(lite_ids)


def test_page_token_invalid(http, seeded):
    experiment_id, _ = seeded
    sc, body = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "page_token": "!!!not-a-token"}
    )
    assert sc == 400


# ---------------------------------------------------------------------------
# runName fallback
# ---------------------------------------------------------------------------


def test_run_name_tag_fallback_without_leaking_tag(http, seeded):
    _, run_ids = seeded
    # Legacy-shaped run: empty runs.name, mlflow.runName tag present.
    sc, body = http.post(GET_LITE, {"run_id": run_ids[LEGACY_NAMED_INDEX]})
    assert sc == 200
    assert body["run"]["info"]["run_name"] == "legacy-named"
    assert "tags" not in body["run"]["data"]  # fallback tag is not leaked unrequested

    # Requesting the tag explicitly returns it.
    sc, body = http.post(
        GET_LITE,
        {"run_id": run_ids[LEGACY_NAMED_INDEX], "projection": {"tag_keys": ["mlflow.runName"]}},
    )
    assert sc == 200
    assert [(t["key"], t["value"]) for t in body["run"]["data"]["tags"]] == [
        ("mlflow.runName", "legacy-named")
    ]

    # A nameless run without the fallback tag reports no run_name at all
    # (proto3 JSON omits the empty string, matching legacy).
    sc, body = http.post(GET_LITE, {"run_id": run_ids[LEGACY_NONAME_INDEX]})
    assert sc == 200
    assert body["run"]["info"].get("run_name", "") == ""


def test_search_lite_run_name_fallback(http, seeded):
    experiment_id, run_ids = seeded
    sc, body = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 100})
    assert sc == 200
    by_id = {run["info"]["run_id"]: run["info"].get("run_name", "") for run in body["runs"]}
    assert by_id[run_ids[1]] == "run-01"
    assert by_id[run_ids[LEGACY_NAMED_INDEX]] == "legacy-named"
    assert by_id[run_ids[LEGACY_NONAME_INDEX]] == ""
    # Legacy parity for the fallback rows too.
    sc, legacy = http.post(
        "/api/2.0/mlflow/runs/search", {"experiment_ids": [experiment_id], "max_results": 100}
    )
    legacy_by_id = {
        run["info"]["run_id"]: run["info"].get("run_name", "") for run in legacy["runs"]
    }
    assert by_id == legacy_by_id


# ---------------------------------------------------------------------------
# Query-count behavior (store level)
# ---------------------------------------------------------------------------


def test_query_count_fixed_as_unselected_cardinality_grows(seeded, db_uri):
    from sqlalchemy import event

    from mlflow.entities.view_type import ViewType
    from mlflow.server.projected_runs import RunProjection, search_runs_projected

    store = handlers._get_tracking_store()
    experiment_id, run_ids = seeded
    active = active_run_ids(run_ids)
    target = active[-1]
    dialect = store.engine.dialect.name

    def count_queries(fn):
        statements = []

        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(store.engine, "before_cursor_execute", before_cursor_execute)
        try:
            fn()
        finally:
            event.remove(store.engine, "before_cursor_execute", before_cursor_execute)
        # DDL/alembic noise aside, count only SELECTs against data tables.
        return len([s for s in statements if s.lstrip().upper().startswith("SELECT")])

    projection = RunProjection(metric_keys=("acc",), param_keys=(), tag_keys=())
    base = {
        "experiment_ids": [experiment_id],
        "filter_string": "",
        "run_view_type": ViewType.ACTIVE_ONLY,
        "max_results": 10,
        "order_by": [],
        "page_token": None,
    }

    n_before = count_queries(lambda: search_runs_projected(store, projection=projection, **base))

    # Grow unselected cardinality substantially.
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=db_uri)
    for j in range(200):
        client.log_metric(target, f"extra_noise_{j}", float(j))

    n_after = count_queries(lambda: search_runs_projected(store, projection=projection, **base))
    assert n_before == n_after, f"query count grew with unselected metrics: {n_before} -> {n_after}"

    # A fully-empty projection performs exactly the run query (plus, on some
    # dialects, an experiment-scope lookup), and never queries children.
    empty_projection = RunProjection()

    def count_all(fn):
        executions = []

        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            executions.append((statement, parameters))

        event.listen(store.engine, "before_cursor_execute", before_cursor_execute)
        try:
            fn()
        finally:
            event.remove(store.engine, "before_cursor_execute", before_cursor_execute)
        return executions

    executions = count_all(
        lambda: search_runs_projected(store, projection=empty_projection, **base)
    )
    # No metric/param/inputs/outputs queries at all; tags are only ever touched
    # for the single mlflow.runName fallback row on nameless runs.
    banned = ("latest_metrics", "params", "inputs", "model_outputs")
    bad = [
        s
        for s, _ in executions
        if any(t in s for t in banned) and s.lstrip().upper().startswith("SELECT")
    ]
    assert not bad, f"unexpected child collection query: {bad}"
    for s, params in executions:
        if "FROM tags" in s:
            if isinstance(params, dict):
                flat = list(params.values())
            else:
                flat = [
                    p
                    for item in params
                    for p in (item if isinstance(item, (list, tuple)) else (item,))
                ]
            assert set(flat) - set(run_ids) == {"mlflow.runName"}, (
                f"unrequested tag query: {s} {params}"
            )

    # The legacy path issues strictly more queries once unselected cardinality grows.
    def legacy():
        store.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=10,
            order_by=[],
            page_token=None,
        )

    n_legacy = count_queries(legacy)
    n_lite = count_queries(lambda: search_runs_projected(store, projection=projection, **base))
    assert n_lite < n_legacy, (
        f"lite({n_lite}) should be cheaper than legacy({n_legacy}) on {dialect}"
    )


# ---------------------------------------------------------------------------
# get-lite
# ---------------------------------------------------------------------------


def test_get_lite_parity(http, seeded):
    _, run_ids = seeded
    sc_legacy, legacy = http.get("/api/2.0/mlflow/runs/get", {"run_id": run_ids[5]})
    sc_lite, lite = http.post(GET_LITE, {"run_id": run_ids[5], "projection": FULL_PROJECTION})
    assert sc_legacy == sc_lite == 200
    assert legacy["run"]["info"] == lite["run"]["info"]
    for collection in ("metrics", "params", "tags"):
        assert comparable(legacy["run"], collection) == comparable(lite["run"], collection)


def test_get_lite_omitted_projection(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(GET_LITE, {"run_id": run_ids[5]})
    assert sc == 200
    assert body["run"]["data"] == {}


def test_get_lite_missing_run(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(
        GET_LITE, {"run_id": run_ids[5][:-1] + ("0" if run_ids[5][-1] != "0" else "1")}
    )
    assert sc == 404
    assert body["error_code"] == "RESOURCE_DOES_NOT_EXIST"


def test_get_lite_deleted_run(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(GET_LITE, {"run_id": run_ids[min(DELETED_RUN_INDEXES)]})
    assert sc == 200
    assert body["run"]["info"]["lifecycle_stage"] == "deleted"


# ---------------------------------------------------------------------------
# data-keys
# ---------------------------------------------------------------------------


def test_data_keys_union_sorted_distinct(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(DATA_KEYS, {"run_ids": run_ids[:2] + run_ids[2:4]})
    assert sc == 200
    assert body["metric_keys"] == sorted({"acc", "loss"} | {f"noise_{j}" for j in range(3)})
    assert body["param_keys"] == sorted({"lr", "batch_size", "unselected_param"})
    assert body["tag_keys"] == sorted({"mlflow.runName", "team", "stage", "unselected_tag"})
    for value in body.values():
        assert value == sorted(set(value))  # distinct + sorted


def test_data_keys_duplicate_run_ids_ok(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(DATA_KEYS, {"run_ids": [run_ids[0], run_ids[0]]})
    assert sc == 200
    assert body["metric_keys"] == sorted({"acc", "loss"})


def test_data_keys_missing_run(http, seeded):
    _, run_ids = seeded
    bogus = run_ids[0][:-1] + ("0" if run_ids[0][-1] != "0" else "1")
    sc, body = http.post(DATA_KEYS, {"run_ids": [run_ids[0], bogus]})
    assert sc == 404


def test_data_keys_deleted_run_includes_keys(http, seeded):
    _, run_ids = seeded
    sc, body = http.post(DATA_KEYS, {"run_ids": [run_ids[min(DELETED_RUN_INDEXES)]]})
    assert sc == 200
    assert "loss" in body["metric_keys"]


def test_data_keys_run_id_cap(http, seeded):
    sc, body = http.post(DATA_KEYS, {"run_ids": [f"{i:032x}" for i in range(1001)]})
    assert sc == 400
    assert "at most" in body["message"]


def test_data_keys_requires_nonempty_list(http):
    sc, _ = http.post(DATA_KEYS, {"run_ids": []})
    assert sc == 400
    sc, _ = http.post(DATA_KEYS, {})
    assert sc == 400


# ---------------------------------------------------------------------------
# server-info capability
# ---------------------------------------------------------------------------


def test_server_info_advertises_projection(http):
    for path in ("/api/3.0/mlflow/server-info", "/ajax-api/3.0/mlflow/server-info"):
        sc, body = http.get(path)
        assert sc == 200
        assert body["run_projection_enabled"] is True


# ---------------------------------------------------------------------------
# Artifacts-only mode
# ---------------------------------------------------------------------------


def test_artifacts_only_disables_projected_routes(http, seeded, monkeypatch):
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACTS_ONLY", "true")
    for path, body in (
        (SEARCH_LITE, {"experiment_ids": [seeded[0]]}),
        (GET_LITE, {"run_id": seeded[1][0]}),
        (DATA_KEYS, {"run_ids": [seeded[1][0]]}),
    ):
        sc, _ = http.post(path, body)
        assert sc == 503, f"{path} should be disabled in artifacts-only mode"


# ---------------------------------------------------------------------------
# api/ajax parity
# ---------------------------------------------------------------------------


def test_ajax_paths_match_api_paths(http, seeded):
    experiment_id, run_ids = seeded
    request = {
        "experiment_ids": [experiment_id],
        "max_results": 3,
        "projection": {"metric_keys": ["loss"]},
    }
    _, api_body = http.post(SEARCH_LITE, request)
    _, ajax_body = http.post(SEARCH_LITE.replace("/api/", "/ajax-api/"), request)
    assert api_body == ajax_body

    _, api_body = http.post(GET_LITE, {"run_id": run_ids[0]})
    _, ajax_body = http.post(GET_LITE.replace("/api/", "/ajax-api/"), {"run_id": run_ids[0]})
    assert api_body == ajax_body

    _, api_body = http.post(DATA_KEYS, {"run_ids": run_ids[:2]})
    _, ajax_body = http.post(DATA_KEYS.replace("/api/", "/ajax-api/"), {"run_ids": run_ids[:2]})
    assert api_body == ajax_body


def test_max_results_string_coerced_like_proto(http, seeded):
    experiment_id, run_ids = seeded
    sc, body = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": "5"})
    assert sc == 200
    assert len(body["runs"]) == 5


def test_search_lite_omitted_or_null_max_results_defaults_to_bounded_page(http, seeded):
    experiment_id, run_ids = seeded
    expected = active_run_ids(run_ids)

    def walk(body):
        seen = []
        while True:
            sc, resp = http.post(SEARCH_LITE, body)
            assert sc == 200
            seen.extend(r["info"]["run_id"] for r in resp["runs"])
            token = resp.get("next_page_token")
            if not token:
                return seen
            body = {**body, "page_token": token}

    # Omitted: at most 10 per page, with a correct next token.
    sc, body = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id]})
    assert sc == 200
    assert len(body["runs"]) == 10
    first_page = [r["info"]["run_id"] for r in body["runs"]]
    assert body.get("next_page_token")
    sc, page_two = http.post(
        SEARCH_LITE, {"experiment_ids": [experiment_id], "page_token": body["next_page_token"]}
    )
    assert sc == 200
    assert len(page_two["runs"]) == 10
    second_page = [r["info"]["run_id"] for r in page_two["runs"]]
    assert not set(first_page) & set(second_page)
    assert page_two.get("next_page_token")

    # Explicit null behaves identically to omitted.
    sc, null_body = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": None})
    assert sc == 200
    assert [r["info"]["run_id"] for r in null_body["runs"]] == first_page
    assert null_body.get("next_page_token")

    # The full walk covers exactly the active set, bounded pages throughout.
    for request_body in (
        {"experiment_ids": [experiment_id]},
        {"experiment_ids": [experiment_id], "max_results": None},
    ):
        seen = walk(request_body)
        assert sorted(seen) == sorted(expected)
        assert len(set(seen)) == len(seen)


def test_search_lite_explicit_max_results_semantics_unchanged(http, seeded):
    experiment_id, _ = seeded
    # Positive explicit value is honored beyond the default page size.
    sc, body = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 15})
    assert sc == 200
    assert len(body["runs"]) == 15
    # Zero is rejected by the shared store validation, as is the over-cap value.
    sc, body = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 0})
    assert sc == 400
    assert body["error_code"] == "INVALID_PARAMETER_VALUE"
    sc, _ = http.post(SEARCH_LITE, {"experiment_ids": [experiment_id], "max_results": 50001})
    assert sc == 400


def test_get_lite_run_uuid_schema_and_conflict(http, seeded):
    _, run_ids = seeded
    # run_uuid alias is schema-validated and works when run_id is absent.
    sc, body = http.post(GET_LITE, {"run_uuid": run_ids[0]})
    assert sc == 200
    assert body["run"]["info"]["run_id"] == run_ids[0]
    # Matching duplicates are harmless (same value twice).
    sc, body = http.post(GET_LITE, {"run_id": run_ids[0], "run_uuid": run_ids[0]})
    assert sc == 200
    assert body["run"]["info"]["run_id"] == run_ids[0]
    # Conflicting values are rejected.
    sc, body = http.post(GET_LITE, {"run_id": run_ids[0], "run_uuid": run_ids[1]})
    assert sc == 400
    assert "Conflicting" in body["message"]
    # Non-string run_uuid fails schema validation.
    sc, _ = http.post(GET_LITE, {"run_uuid": 123})
    assert sc == 400
    # run_id keeps its existing type validation.
    sc, _ = http.post(GET_LITE, {"run_id": 123})
    assert sc == 400


def test_server_info_capability_tracks_store_type(http, seeded, monkeypatch, tmp_path):
    from mlflow.store.tracking.file_store import FileStore

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    real_store = handlers._get_tracking_store()
    try:
        monkeypatch.setattr(handlers, "_tracking_store", FileStore(str(tmp_path / "mlruns")))
        payload = handlers.build_server_info_payload()
        assert payload["store_type"] == "FileStore"
        assert payload["run_projection_enabled"] is False

        monkeypatch.setattr(handlers, "_tracking_store", real_store)
        payload = handlers.build_server_info_payload()
        assert payload["store_type"] == "SqlStore"
        assert payload["run_projection_enabled"] is True
    finally:
        handlers._tracking_store = real_store


def test_server_info_capability_false_in_artifacts_only(http, monkeypatch):
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACTS_ONLY", "true")
    try:
        payload = handlers.build_server_info_payload()
        assert payload["store_type"] is None
        assert payload["run_projection_enabled"] is False
    finally:
        monkeypatch.undo()


def test_non_sql_store_returns_501_not_implemented(http, seeded, monkeypatch, tmp_path):
    from mlflow.server import projected_runs
    from mlflow.store.tracking.file_store import FileStore

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    fake_store = FileStore(str(tmp_path / "mlruns"))
    monkeypatch.setattr(projected_runs, "_get_tracking_store", lambda: fake_store)

    for path, body in (
        (SEARCH_LITE, {"experiment_ids": [seeded[0]]}),
        (GET_LITE, {"run_id": seeded[1][0]}),
        (DATA_KEYS, {"run_ids": [seeded[1][0]]}),
    ):
        sc, resp = http.post(path, body)
        assert sc == 501, f"{path} should map a non-SQL store to 501, got {sc}"
        assert resp["error_code"] == "NOT_IMPLEMENTED"
