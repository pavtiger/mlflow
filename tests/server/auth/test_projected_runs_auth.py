"""Basic-auth (fail-closed) tests for the projected run read endpoints.

Spawns a real auth-enabled server (``mlflow.server.auth:create_app``) against a
throwaway SQLite backend, mirroring the conventions in ``test_auth.py``. The
module-scoped fixture seeds one experiment with two runs; run A links to a
logged model so the model-link redaction path can be exercised end to end.

Test order matters: ``test_granted_reader_gets_projected_reads`` grants the
reader experiment READ, and ``test_lite_runs_withhold_denied_model_links``
adds a logged-model DENY on top of it.
"""

import os

import pytest
import requests

from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_FLASK_SERVER_SECRET_KEY
from mlflow.server.auth import auth_config
from mlflow.utils.os import is_windows

from tests.tracking.integration_test_utils import _init_server

SEARCH_LITE = "/api/2.0/mlflow/runs/search-lite"
GET_LITE = "/api/2.0/mlflow/runs/get-lite"
DATA_KEYS = "/api/2.0/mlflow/runs/data-keys"

ADMIN_USERNAME = auth_config.admin_username
ADMIN_PASSWORD = "test-admin-password"
ADMIN = (ADMIN_USERNAME, ADMIN_PASSWORD)
READER = ("reader", "reader-pass-123")


def _post(base_url, path, body, creds=None):
    resp = requests.post(base_url + path, json=body, auth=creds, timeout=30)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


def _get(base_url, path, params=None, creds=None):
    resp = requests.get(base_url + path, params=params or {}, auth=creds, timeout=30)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


@pytest.fixture(scope="module")
def auth_server(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("projected-runs-auth")
    # Set PROJECTION_TEST_DB_URI to run against a different backend (e.g.
    # Postgres) instead of the default SQLite file. The auth store keeps its
    # own SQLite database either way.
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = os.environ.get("PROJECTION_TEST_DB_URI") or (
        ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    )

    auth_config_path = tmp_path / "basic_auth.ini"
    auth_config_path.write_text(
        "[mlflow]\n"
        "default_permission = NO_PERMISSIONS\n"
        f"database_uri = sqlite:///{tmp_path / 'basic_auth.db'}\n"
        f"admin_username = {ADMIN_USERNAME}\n"
        f"admin_password = {ADMIN_PASSWORD}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
    )
    extra_env = {
        "MLFLOW_AUTH_CONFIG_PATH": str(auth_config_path),
        MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
    }

    saved_creds = (
        os.environ.get("MLFLOW_TRACKING_USERNAME"),
        os.environ.get("MLFLOW_TRACKING_PASSWORD"),
    )
    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        # Seed as the admin (superuser) so creation is permitted under
        # default_permission = NO_PERMISSIONS.
        os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"] = ADMIN
        try:
            client = MlflowClient(url)
            experiment_id = client.create_experiment("auth-projection-test")
            run_a = client.create_run(experiment_id, tags={"mlflow.runName": "auth-run-a"})
            run_b = client.create_run(experiment_id, tags={"mlflow.runName": "auth-run-b"})
            for run_id in (run_a.info.run_id, run_b.info.run_id):
                client.log_metric(run_id, "loss", 0.5)
                client.log_param(run_id, "lr", "0.01")
                client.set_tag(run_id, "team", "x")
            client.set_terminated(run_a.info.run_id)
            client.set_terminated(run_b.info.run_id)

            # Run A links to a logged model (the "Models" column data), so the
            # model-link redaction path can be exercised end to end. The row is
            # written directly against the server's tracking DB.
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session

            from mlflow.store.tracking.dbmodels.models import SqlInput

            engine = create_engine(backend_uri)
            try:
                with Session(engine) as session:
                    session.add(
                        SqlInput(
                            input_uuid="auth-model-input-1",
                            source_type="RUN_INPUT",
                            source_id=run_a.info.run_id,
                            destination_type="MODEL_INPUT",
                            destination_id="auth-model-1",
                        )
                    )
                    session.commit()
            finally:
                engine.dispose()
        finally:
            if saved_creds[0] is None:
                os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
            else:
                os.environ["MLFLOW_TRACKING_USERNAME"] = saved_creds[0]
            if saved_creds[1] is None:
                os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
            else:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = saved_creds[1]

        # Bootstrap: create the non-privileged reader (validate_can_create_user
        # is admin-only).
        sc, _ = _post(
            url,
            "/api/2.0/mlflow/users/create",
            {"username": READER[0], "password": READER[1]},
            creds=ADMIN,
        )
        assert sc == 200, "failed to create reader user"

        yield url, experiment_id, run_a.info.run_id, run_b.info.run_id


def _grant_experiment_read(base_url, experiment_id, creds=ADMIN):
    sc, _ = _post(
        base_url,
        "/api/3.0/mlflow/users/permissions/grant",
        {
            "username": READER[0],
            "resource_type": "experiment",
            "resource_id": experiment_id,
            "permission": "READ",
        },
        creds=creds,
    )
    assert sc == 200, "failed to grant READ on experiment"


def test_fail_closed_wiring_present():
    from mlflow.server import auth

    assert "/mlflow/runs/search-lite" in auth._HANDLER_INTERNAL_AUTHZ_SUFFIXES
    for path, method in (
        ("/api/2.0/mlflow/runs/get-lite", "POST"),
        ("/ajax-api/2.0/mlflow/runs/get-lite", "POST"),
        ("/api/2.0/mlflow/runs/data-keys", "POST"),
        ("/ajax-api/2.0/mlflow/runs/data-keys", "POST"),
    ):
        assert (path, method) in auth.BEFORE_REQUEST_VALIDATORS
    for path in (
        "/api/2.0/mlflow/runs/search-lite",
        "/ajax-api/2.0/mlflow/runs/search-lite",
        "/api/2.0/mlflow/runs/get-lite",
        "/ajax-api/2.0/mlflow/runs/get-lite",
    ):
        assert (path, "POST") in auth.AFTER_REQUEST_HANDLERS


def test_unauthenticated_requests_rejected(auth_server):
    base_url, experiment_id, run_id, _ = auth_server
    for path, body in (
        (SEARCH_LITE, {"experiment_ids": [experiment_id]}),
        (GET_LITE, {"run_id": run_id}),
        (DATA_KEYS, {"run_ids": [run_id]}),
        (SEARCH_LITE.replace("/api/", "/ajax-api/"), {"experiment_ids": [experiment_id]}),
    ):
        sc, _ = _post(base_url, path, body, creds=None)
        assert sc == 401, f"{path} should require authentication"


def test_search_lite_self_authorizes_for_reader_without_grants(auth_server):
    base_url, experiment_id, _, _ = auth_server
    # The reader has no grants: the handler-side filter restricts to readable
    # experiments, so the route answers 200 with zero runs (legacy semantics).
    sc, body = _post(
        base_url,
        SEARCH_LITE,
        {"experiment_ids": [experiment_id], "projection": {"metric_keys": ["loss"]}},
        creds=READER,
    )
    assert sc == 200
    assert body.get("runs", []) == []  # proto3 JSON omits the empty repeated field
    # Admin (superuser) sees everything.
    sc, body = _post(base_url, SEARCH_LITE, {"experiment_ids": [experiment_id]}, creds=ADMIN)
    assert sc == 200
    assert len(body["runs"]) == 2


def test_get_lite_denied_without_grant(auth_server):
    base_url, _, run_id, _ = auth_server
    for path in (GET_LITE, GET_LITE.replace("/api/", "/ajax-api/")):
        sc, _ = _post(base_url, path, {"run_id": run_id}, creds=READER)
        assert sc == 403, f"{path} should deny a reader without grants"


def test_data_keys_denied_without_grant(auth_server):
    base_url, _, run_id, _ = auth_server
    for path in (DATA_KEYS, DATA_KEYS.replace("/api/", "/ajax-api/")):
        sc, _ = _post(base_url, path, {"run_ids": [run_id]}, creds=READER)
        assert sc == 403, f"{path} should deny a reader without grants"


def test_data_keys_missing_run_fails_closed(auth_server):
    base_url, _, run_a, _ = auth_server
    # Admin bypasses validators; the handler itself must not emit keys for a
    # run that does not exist.
    sc, _ = _post(base_url, DATA_KEYS, {"run_ids": [run_a, "f" * 32]}, creds=ADMIN)
    assert sc == 404
    # Non-admin: the per-run validator denies unknown run IDs (403, not 404),
    # so the response is not an existence oracle.
    sc, _ = _post(base_url, DATA_KEYS, {"run_ids": [run_a, "f" * 32]}, creds=READER)
    assert sc == 403


def test_granted_reader_gets_projected_reads(auth_server):
    base_url, experiment_id, run_a, _ = auth_server
    _grant_experiment_read(base_url, experiment_id)

    sc, body = _post(base_url, GET_LITE, {"run_id": run_a}, creds=READER)
    assert sc == 200
    assert body["run"]["data"] == {}

    sc, body = _post(
        base_url,
        GET_LITE,
        {"run_id": run_a, "projection": {"metric_keys": ["loss"], "tag_keys": ["team"]}},
        creds=READER,
    )
    assert sc == 200
    assert [m["key"] for m in body["run"]["data"]["metrics"]] == ["loss"]
    assert [t["key"] for t in body["run"]["data"]["tags"]] == ["team"]
    # Model-link redaction after-request handler is wired and runs without error.
    assert "run" in body

    sc, body = _post(base_url, DATA_KEYS, {"run_ids": [run_a]}, creds=READER)
    assert sc == 200
    assert body["metric_keys"] == ["loss"]

    # IO projection is permitted for authorized readers; the model-link
    # redaction after-request handler still runs on the response.
    sc, body = _post(
        base_url,
        GET_LITE,
        {"run_id": run_a, "projection": {"include_inputs": True, "include_outputs": True}},
        creds=READER,
    )
    assert sc == 200
    assert "run" in body

    sc, body = _post(base_url, SEARCH_LITE, {"experiment_ids": [experiment_id]}, creds=READER)
    assert sc == 200
    assert len(body["runs"]) == 2
    assert all(run["data"] == {} for run in body["runs"])


def test_denied_runs_never_appear_via_search_lite(auth_server):
    # Reader has READ on the experiment (granted above); experiment-level read
    # implies run read here, so both runs remain visible. The important property
    # is that search-lite goes through the identical filter as legacy search.
    base_url, experiment_id, _, _ = auth_server
    sc_lite, lite = _post(base_url, SEARCH_LITE, {"experiment_ids": [experiment_id]}, creds=READER)
    sc_legacy, legacy = _post(
        base_url, "/api/2.0/mlflow/runs/search", {"experiment_ids": [experiment_id]}, creds=READER
    )
    assert sc_lite == sc_legacy == 200
    assert [r["info"]["run_id"] for r in lite["runs"]] == [
        r["info"]["run_id"] for r in legacy["runs"]
    ]


def test_lite_runs_withhold_denied_model_links(auth_server):
    """Negative redaction test: a run's model link is withheld from a reader who
    has a logged-model DENY -- on the new lite routes exactly as on legacy.
    """
    base_url, experiment_id, run_a, _ = auth_server
    projection = {"include_inputs": True, "include_outputs": True}

    # Authorized path (experiment READ granted earlier): the reader and admin
    # both see the model link on legacy and lite alike.
    sc, legacy = _get(base_url, "/api/2.0/mlflow/runs/get", {"run_id": run_a}, creds=READER)
    assert sc == 200
    assert legacy["run"]["inputs"]["model_inputs"] == [{"model_id": "auth-model-1"}]
    sc, lite = _post(base_url, GET_LITE, {"run_id": run_a, "projection": projection}, creds=READER)
    assert sc == 200
    assert lite["run"]["inputs"]["model_inputs"] == [{"model_id": "auth-model-1"}]
    sc, body = _post(base_url, GET_LITE, {"run_id": run_a, "projection": projection}, creds=ADMIN)
    assert sc == 200
    assert body["run"]["inputs"]["model_inputs"] == [{"model_id": "auth-model-1"}]

    # Deny the logged-model tier for the reader.
    sc, _ = _post(
        base_url,
        "/api/3.0/mlflow/users/permissions/grant",
        {
            "username": READER[0],
            "resource_type": "logged_model",
            "resource_id": "*",
            "permission": "DENY",
        },
        creds=ADMIN,
    )
    assert sc == 200

    # get-lite redaction is byte-identical to legacy GetRun: the unauthorized
    # link is stripped, the run itself survives.
    sc, legacy = _get(base_url, "/api/2.0/mlflow/runs/get", {"run_id": run_a}, creds=READER)
    sc, lite = _post(base_url, GET_LITE, {"run_id": run_a, "projection": projection}, creds=READER)
    assert sc == 200
    assert "model_inputs" not in lite["run"].get("inputs", {})
    assert "model_outputs" not in lite["run"].get("outputs", {})
    assert lite["run"]["inputs"] == legacy["run"]["inputs"]
    assert lite["run"]["outputs"] == legacy["run"]["outputs"]
    assert lite["run"]["info"]["run_id"] == run_a  # the subject itself survives

    # search-lite redaction matches legacy SearchRuns row for row.
    sc, legacy_s = _post(
        base_url, "/api/2.0/mlflow/runs/search", {"experiment_ids": [experiment_id]}, creds=READER
    )
    sc, lite_s = _post(
        base_url,
        SEARCH_LITE,
        {"experiment_ids": [experiment_id], "projection": projection},
        creds=READER,
    )
    assert sc == 200
    assert [r["info"]["run_id"] for r in lite_s["runs"]] == [
        r["info"]["run_id"] for r in legacy_s["runs"]
    ]
    lite_by_id = {r["info"]["run_id"]: r for r in lite_s["runs"]}
    for legacy_run in legacy_s["runs"]:
        lite_run = lite_by_id[legacy_run["info"]["run_id"]]
        assert lite_run["inputs"] == legacy_run["inputs"]
        assert lite_run["outputs"] == legacy_run["outputs"]
    assert "model_inputs" not in lite_by_id[run_a].get("inputs", {})

    # The admin (superuser) bypass still retains the link.
    sc, body = _post(base_url, GET_LITE, {"run_id": run_a, "projection": projection}, creds=ADMIN)
    assert sc == 200
    assert body["run"]["inputs"]["model_inputs"] == [{"model_id": "auth-model-1"}]
