"""Projected (lightweight) run reads for the MLflow tracking server.

Adds three POST endpoints, registered on both ``/api/2.0`` and ``/ajax-api/2.0``
via :func:`get_projected_run_endpoints`:

- ``/mlflow/runs/search-lite`` -- like legacy ``SearchRuns`` but child
  metric/param/tag values are returned only when explicitly projected.
- ``/mlflow/runs/get-lite`` -- like legacy ``GetRun`` with the same projection.
- ``/mlflow/runs/data-keys`` -- distinct metric/param/tag key names for an
  explicit set of run IDs.

Contract:

- An omitted or empty projection returns run info with **no** child values.
- ``include_inputs`` / ``include_outputs`` (default ``false``) project run IO
  using the exact legacy bulk helpers: search-lite mirrors legacy SearchRuns
  (``_get_run_inputs`` + ``_get_model_outputs_bulk`` after trimming to the N
  run IDs), get-lite mirrors legacy GetRun (``_get_run_inputs`` +
  ``_get_model_inputs`` + ``_get_model_outputs``). When a flag is false the
  corresponding IO queries do not run and the field is omitted.
- Responses are built from the existing MLflow entities
  (``Run``/``RunInfo``/``RunData``) and serialized through ``to_proto`` +
  ``message_to_json`` so the JSON shape stays legacy-compatible snake_case.
- Filtering, ordering, lifecycle views, and offset page tokens reuse the exact
  legacy ``SqlAlchemyStore._search_runs`` building blocks.

Query behavior: the run query executes **without** eager child loaders; the
lookahead (N+1) ORM row is trimmed before any child query runs. Child rows are
fetched with manual parameterized bulk queries restricted to the returned run
IDs and requested keys, so the query count is fixed regardless of how many
*unselected* child rows exist, and an empty selection performs zero child
queries. Run IO uses the same conditional scheme: the legacy bulk IO helpers
run only for the trimmed run IDs and only when the corresponding flag is set.
``SqlRun.to_mlflow_entity`` is never called and no relationship is
accessed lazily.

These reads are only supported on the SQLAlchemy store family; other tracking
stores receive a 501 ``NOT_IMPLEMENTED`` response, and ``/mlflow/server-info``
advertises the capability via ``run_projection_enabled``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Response

from sqlalchemy import select

from mlflow.entities import Metric, Param, Run, RunData, RunInfo, RunInputs, RunOutputs, RunTag
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.view_type import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import GetRun, SearchRuns
from mlflow.server.handlers import (
    SEARCH_MAX_RESULTS_THRESHOLD,
    _assert_array,
    _assert_dict,
    _assert_item_type_string,
    _assert_string,
    _disable_if_artifacts_only,
    _get_paths,
    _get_tracking_store,
    _get_validated_flask_request_json,
    catch_mlflow_exception,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlLatestMetric,
    SqlParam,
    SqlRun,
    SqlTag,
)
from mlflow.store.tracking.sqlalchemy_store import (
    _get_orderby_clauses,
    _get_sqlalchemy_filter_clauses,
)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.validation import MAX_ENTITY_KEY_LENGTH, _parse_experiment_ids

_logger = logging.getLogger(__name__)

SEARCH_RUNS_LITE = "/mlflow/runs/search-lite"
GET_RUN_LITE = "/mlflow/runs/get-lite"
RUN_DATA_KEYS = "/mlflow/runs/data-keys"

# Limits for the projected-read contract. Projection keys and data-keys run IDs
# are caller-supplied lists that become SQL ``IN`` clauses, so they are capped;
# individual keys must satisfy the MLflow entity key length limit (250 chars).
MAX_PROJECTION_KEYS_PER_COLLECTION = 500
MAX_DATA_KEYS_RUN_IDS = 1000

# search-lite is UI-focused: an omitted or null max_results must not turn into
# an unbounded query (legacy proto parsing yields max_results=0 and rejects
# it), so it falls back to a small bounded page. Explicit values -- including
# 0, which the store rejects -- keep legacy validation semantics.
DEFAULT_SEARCH_LITE_MAX_RESULTS = 10

_ALLOWED_SEARCH_FIELDS = {
    "experiment_ids",
    "filter",
    "run_view_type",
    "max_results",
    "order_by",
    "page_token",
    "projection",
}
_ALLOWED_PROJECTION_FIELDS = {
    "metric_keys",
    "param_keys",
    "tag_keys",
    "include_inputs",
    "include_outputs",
}
_RUN_VIEW_TYPES = {
    "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
    "DELETED_ONLY": ViewType.DELETED_ONLY,
    "ALL": ViewType.ALL,
}


@dataclass(frozen=True)
class RunProjection:
    """Normalized, validated projection for a projected run read."""

    metric_keys: tuple[str, ...] = ()
    param_keys: tuple[str, ...] = ()
    tag_keys: tuple[str, ...] = ()
    include_inputs: bool = False
    include_outputs: bool = False


def _invalid_parameter(message: str) -> MlflowException:
    return MlflowException(
        message,
        error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
    )


def _reject_unknown_fields(request_json: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(request_json) - allowed)
    if unknown:
        raise _invalid_parameter(f"Unknown field(s) for {context}: {', '.join(unknown)}.")


def _validate_key_list(raw: Any, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(k, str) for k in raw):
        raise _invalid_parameter(
            f"Field '{field_name}' must be a list of strings. "
            f"Hint: Value was of type '{type(raw).__name__}'."
        )
    keys = list(dict.fromkeys(raw))  # dedupe, preserving first-seen order
    if len(keys) > MAX_PROJECTION_KEYS_PER_COLLECTION:
        raise _invalid_parameter(
            f"Field '{field_name}' contains {len(keys)} distinct keys; "
            f"at most {MAX_PROJECTION_KEYS_PER_COLLECTION} are allowed."
        )
    for key in keys:
        if not key:
            raise _invalid_parameter(f"Field '{field_name}' contains an empty key.")
        if len(key) > MAX_ENTITY_KEY_LENGTH:
            raise _invalid_parameter(
                f"A key in '{field_name}' exceeds the MLflow key length limit of "
                f"{MAX_ENTITY_KEY_LENGTH} characters."
            )
    return tuple(keys)


def _validate_run_ids(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(r, str) for r in raw):
        raise _invalid_parameter(
            "Field 'run_ids' must be a list of strings. "
            f"Hint: Value was of type '{type(raw).__name__}'."
        )
    run_ids = list(dict.fromkeys(raw))
    if not run_ids:
        raise _invalid_parameter("Field 'run_ids' must contain at least one run ID.")
    if len(run_ids) > MAX_DATA_KEYS_RUN_IDS:
        raise _invalid_parameter(
            f"Field 'run_ids' contains {len(run_ids)} distinct run IDs; "
            f"at most {MAX_DATA_KEYS_RUN_IDS} are allowed."
        )
    for run_id in run_ids:
        if not run_id:
            raise _invalid_parameter("Field 'run_ids' contains an empty run ID.")
    return tuple(run_ids)


def _parse_projection(raw: Any) -> RunProjection:
    if raw is None:
        return RunProjection()
    if not isinstance(raw, dict):
        raise _invalid_parameter(
            f"Field 'projection' must be an object. Hint: Value was of type '{type(raw).__name__}'."
        )
    _reject_unknown_fields(raw, _ALLOWED_PROJECTION_FIELDS, "projection")
    include_inputs = raw.get("include_inputs", False)
    include_outputs = raw.get("include_outputs", False)
    for flag_name, flag_value in (
        ("include_inputs", include_inputs),
        ("include_outputs", include_outputs),
    ):
        if not isinstance(flag_value, bool):
            raise _invalid_parameter(f"Field 'projection.{flag_name}' must be a boolean.")
    return RunProjection(
        metric_keys=_validate_key_list(raw.get("metric_keys"), "projection.metric_keys"),
        param_keys=_validate_key_list(raw.get("param_keys"), "projection.param_keys"),
        tag_keys=_validate_key_list(raw.get("tag_keys"), "projection.tag_keys"),
        include_inputs=include_inputs,
        include_outputs=include_outputs,
    )


def _parse_run_view_type(raw: Any) -> ViewType:
    if raw is None:
        return ViewType.ACTIVE_ONLY
    view_type = _RUN_VIEW_TYPES.get(raw) if isinstance(raw, str) else None
    if view_type is None:
        raise _invalid_parameter(
            f"Invalid run_view_type {raw!r}; expected one of: {', '.join(_RUN_VIEW_TYPES)}."
        )
    return view_type


def _filter_authorized_experiment_ids(experiment_ids: list[str]) -> list[str]:
    # Same guarded block as ``search_runs_impl``: the handler self-authorizes by
    # restricting the query to experiments the caller can read.
    try:
        from mlflow.server import auth

        if auth.auth_config:
            return auth.filter_experiment_ids(experiment_ids)
    except ImportError:
        # Auth module not available (Flask-WTF not installed), skip filtering
        pass
    return experiment_ids


def _require_sql_store(store) -> None:
    """Defensive guard: projected reads only exist on the SQLAlchemy store family.

    ``WorkspaceAwareSqlAlchemyStore`` subclasses ``SqlAlchemyStore`` and is
    accepted. Anything else (FileStore, custom stores) maps to a clear 501
    instead of an ``AttributeError`` 500 deep in the projection code.
    """
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    if not isinstance(store, SqlAlchemyStore):
        raise MlflowException(
            "Projected run reads (search-lite / get-lite / data-keys) are only supported "
            f"on the SQLAlchemy tracking store; got {type(store).__name__}.",
            error_code=databricks_pb2.NOT_IMPLEMENTED,
        )


# ---------------------------------------------------------------------------
# Store-level projected reads. These are module functions that take a tracking
# store and use only its existing (private) building blocks so behavior stays
# identical to ``SqlAlchemyStore._search_runs`` / ``get_run``.
# ---------------------------------------------------------------------------


def _run_info_from_sql(sql_run: SqlRun, all_tags: list[RunTag]) -> RunInfo:
    """Build a :class:`RunInfo` from run columns only (never ``to_mlflow_entity``).

    ``all_tags`` is the set of tags loaded for this run (requested keys plus the
    ``mlflow.runName`` fallback row when needed); it is used solely to recover
    the legacy run-name-from-tag behavior when ``runs.name`` is empty.
    """
    run_name = sql_run.name
    if not run_name:
        for tag in all_tags:
            if tag.key == MLFLOW_RUN_NAME:
                run_name = tag.value
                break
    return RunInfo(
        run_id=sql_run.run_uuid,
        run_name=run_name,
        experiment_id=str(sql_run.experiment_id),
        user_id=sql_run.user_id,
        status=sql_run.status,
        start_time=sql_run.start_time,
        end_time=sql_run.end_time,
        lifecycle_stage=sql_run.lifecycle_stage,
        artifact_uri=sql_run.artifact_uri,
    )


def _load_projected_children(session, run_ids, projection, extra_tag_keys=()):
    """Bulk-load the projected child rows for ``run_ids``.

    Zero queries per collection with no selected keys. Returns
    ``(metrics_by_run, params_by_run, tags_by_run)`` where ``tags_by_run``
    includes ``extra_tag_keys`` rows (used for the run-name fallback); callers
    strip unrequested tags from the output.
    """
    metrics_by_run: dict[str, list[Metric]] = {}
    params_by_run: dict[str, list[Param]] = {}
    tags_by_run: dict[str, list[RunTag]] = {}
    if not run_ids:
        return metrics_by_run, params_by_run, tags_by_run

    if projection.metric_keys:
        rows = (
            session
            .query(SqlLatestMetric)
            .filter(
                SqlLatestMetric.run_uuid.in_(run_ids),
                SqlLatestMetric.key.in_(projection.metric_keys),
            )
            .order_by(SqlLatestMetric.key)
            .all()
        )
        for row in rows:
            metrics_by_run.setdefault(row.run_uuid, []).append(row.to_mlflow_entity())

    if projection.param_keys:
        rows = (
            session
            .query(SqlParam)
            .filter(
                SqlParam.run_uuid.in_(run_ids),
                SqlParam.key.in_(projection.param_keys),
            )
            .order_by(SqlParam.key)
            .all()
        )
        for row in rows:
            params_by_run.setdefault(row.run_uuid, []).append(row.to_mlflow_entity())

    tag_keys = tuple(dict.fromkeys((*projection.tag_keys, *extra_tag_keys)))
    if tag_keys:
        rows = (
            session
            .query(SqlTag)
            .filter(
                SqlTag.run_uuid.in_(run_ids),
                SqlTag.key.in_(tag_keys),
            )
            .order_by(SqlTag.key)
            .all()
        )
        for row in rows:
            tags_by_run.setdefault(row.run_uuid, []).append(row.to_mlflow_entity())

    return metrics_by_run, params_by_run, tags_by_run


def _build_projected_run(
    sql_run,
    projection,
    metrics_by_run,
    params_by_run,
    tags_by_run,
    run_inputs=None,
    run_outputs=None,
):
    """Construct a :class:`Run` entity from a trimmed ORM row + bulk-loaded children."""
    run_id = sql_run.run_uuid
    all_tags = tags_by_run.get(run_id, [])
    requested_key_set = set(projection.tag_keys)
    requested_tags = [t for t in all_tags if t.key in requested_key_set]
    return Run(
        run_info=_run_info_from_sql(sql_run, all_tags),
        run_data=RunData(
            metrics=metrics_by_run.get(run_id, []),
            params=params_by_run.get(run_id, []),
            tags=requested_tags,
        ),
        run_inputs=run_inputs,
        run_outputs=run_outputs,
    )


def search_runs_projected(
    store,
    experiment_ids,
    filter_string,
    run_view_type,
    max_results,
    order_by,
    page_token,
    projection: RunProjection,
):
    """Projected variant of ``SqlAlchemyStore._search_runs``.

    Reuses the exact legacy query building (search filter parsing, order-by
    clauses, experiment scoping, offset page tokens, ``max_results`` validation)
    but executes the run query without eager child loaders, trims the lookahead
    (N+1) row before touching children, and loads only the projected child keys
    via parameterized bulk queries. Returns ``(runs, next_page_token)``.
    """
    self = store
    _require_sql_store(self)

    def compute_next_token(current_size):
        next_token = None
        if max_results is not None and current_size == max_results + 1:
            final_offset = offset + max_results
            next_token = SearchUtils.create_page_token(final_offset)
        return next_token

    self._validate_max_results_param(max_results, allow_null=True)

    stages = set(LifecycleStage.view_type_to_stages(run_view_type))

    with self.ManagedSessionMaker() as session:
        parsed_filters = SearchUtils.parse_search_filter(filter_string)
        cases_orderby, parsed_orderby, sorting_joins = _get_orderby_clauses(order_by, session)

        stmt = select(SqlRun, *cases_orderby)
        (
            attribute_filters,
            non_attribute_filters,
            dataset_filters,
        ) = _get_sqlalchemy_filter_clauses(parsed_filters, session, self._get_dialect())
        for non_attr_filter in non_attribute_filters:
            stmt = stmt.join(non_attr_filter)
        for dataset_filter in dataset_filters:
            stmt = stmt.join(
                dataset_filter,
                SqlRun.run_uuid == dataset_filter.c.destination_id,
            )
        # Outer joins (not inner) so ordering on a tag/metric/param column does
        # not drop runs missing a value for it -- same as legacy _search_runs.
        for j in sorting_joins:
            stmt = stmt.outerjoin(j)

        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        experiment_ids = _parse_experiment_ids(experiment_ids)
        experiment_ids = self._filter_experiment_ids(session, experiment_ids)
        stmt = (
            stmt
            .distinct()
            .filter(
                SqlRun.experiment_id.in_(experiment_ids),
                SqlRun.lifecycle_stage.in_(stages),
                *attribute_filters,
            )
            .order_by(*parsed_orderby)
            .offset(offset)
        )
        if max_results is not None:
            stmt = stmt.limit(max_results + 1)
        # NB: deliberately no _get_eager_run_query_options() here.
        queried_runs = session.execute(stmt).scalars(SqlRun).all()

        # Trim the lookahead row BEFORE any child query so an unselected N+1 row
        # costs nothing, then build the compatible page token.
        next_page_token = compute_next_token(len(queried_runs))
        if next_page_token and max_results is not None:
            queried_runs = queried_runs[:max_results]

        run_ids = [run.run_uuid for run in queried_runs]
        # Legacy fallback: recover run_name from the mlflow.runName tag when the
        # run row has no name, even when that tag was not projected.
        extra_tag_keys = (
            (MLFLOW_RUN_NAME,)
            if any(not run.name for run in queried_runs)
            and MLFLOW_RUN_NAME not in projection.tag_keys
            else ()
        )
        metrics_by_run, params_by_run, tags_by_run = _load_projected_children(
            session, run_ids, projection, extra_tag_keys=extra_tag_keys
        )

        # Projected IO, mirroring legacy SearchRuns but only when requested.
        # The lookahead row was already trimmed above, so the N+1 run never
        # triggers an IO query. Skipped entirely (zero queries) when the flag
        # is false, and skipped for empty pages.
        dataset_inputs_per_run = None
        model_outputs_map = None
        if run_ids:
            if projection.include_inputs:
                dataset_inputs_per_run = self._get_run_inputs(run_uuids=run_ids, session=session)
            if projection.include_outputs:
                model_outputs_map = self._get_model_outputs_bulk(run_ids=run_ids, session=session)

        runs = []
        for index, sql_run in enumerate(queried_runs):
            run_inputs = (
                RunInputs(dataset_inputs=dataset_inputs_per_run[index])
                if projection.include_inputs
                else None
            )
            run_outputs = (
                RunOutputs(model_outputs_map[sql_run.run_uuid])
                if projection.include_outputs
                else None
            )
            runs.append(
                _build_projected_run(
                    sql_run,
                    projection,
                    metrics_by_run,
                    params_by_run,
                    tags_by_run,
                    run_inputs=run_inputs,
                    run_outputs=run_outputs,
                )
            )

    return runs, next_page_token


def get_run_projected(store, run_id: str, projection: RunProjection) -> Run:
    """Projected variant of ``SqlAlchemyStore.get_run`` (workspace-aware)."""
    _require_sql_store(store)
    with store.ManagedSessionMaker() as session:
        # Workspace-aware stores scope _get_query to the active workspace; plain
        # stores query runs directly.
        sql_run = store._get_query(session, SqlRun).filter(SqlRun.run_uuid == run_id).first()
        if sql_run is None:
            raise MlflowException(
                f"Run with id={run_id} not found",
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        store._validate_run_accessible(session, run_id)

        extra_tag_keys = (
            (MLFLOW_RUN_NAME,)
            if not sql_run.name and MLFLOW_RUN_NAME not in projection.tag_keys
            else ()
        )
        metrics_by_run, params_by_run, tags_by_run = _load_projected_children(
            session, [run_id], projection, extra_tag_keys=extra_tag_keys
        )

        # Projected IO, mirroring legacy GetRun but only when requested.
        run_inputs = None
        run_outputs = None
        if projection.include_inputs:
            dataset_inputs = store._get_run_inputs(run_uuids=[run_id], session=session)[0]
            model_inputs = store._get_model_inputs(run_id, session)
            run_inputs = RunInputs(dataset_inputs=dataset_inputs, model_inputs=model_inputs)
        if projection.include_outputs:
            run_outputs = RunOutputs(store._get_model_outputs(run_id, session))

        return _build_projected_run(
            sql_run,
            projection,
            metrics_by_run,
            params_by_run,
            tags_by_run,
            run_inputs=run_inputs,
            run_outputs=run_outputs,
        )


def get_run_data_keys(store, run_ids) -> dict[str, list[str]]:
    """Distinct metric/param/tag keys across the given runs (workspace-aware)."""
    _require_sql_store(store)
    with store.ManagedSessionMaker() as session:
        found = store._get_query(session, SqlRun).filter(SqlRun.run_uuid.in_(run_ids)).all()
        found_ids = {run.run_uuid for run in found}
        missing = [run_id for run_id in run_ids if run_id not in found_ids]
        if missing:
            raise MlflowException(
                f"Run(s) not found: {', '.join(missing)}",
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )

        metric_keys = sorted(
            row[0]
            for row in session
            .query(SqlLatestMetric.key)
            .filter(SqlLatestMetric.run_uuid.in_(run_ids))
            .distinct()
            .all()
        )
        param_keys = sorted(
            row[0]
            for row in session
            .query(SqlParam.key)
            .filter(SqlParam.run_uuid.in_(run_ids))
            .distinct()
            .all()
        )
        tag_keys = sorted(
            row[0]
            for row in session
            .query(SqlTag.key)
            .filter(SqlTag.run_uuid.in_(run_ids))
            .distinct()
            .all()
        )
    return {"metric_keys": metric_keys, "param_keys": param_keys, "tag_keys": tag_keys}


# ---------------------------------------------------------------------------
# Authorization validators (registered by mlflow.server.auth).
# ---------------------------------------------------------------------------


def validate_can_read_data_keys() -> bool:
    """Before-request validator for ``data-keys``: require read on every run.

    Uses the same per-run grant resolution as ``validate_can_read_run`` for
    each requested run ID, so keys can never leak across permissions or
    workspaces. Fail-closed: an unknown run ID denies rather than 404ing.
    """
    from mlflow.server import auth

    request_json = _get_validated_flask_request_json(
        schema={"run_ids": [_assert_array, _assert_item_type_string]}
    )
    run_ids = _validate_run_ids(request_json.get("run_ids"))
    return all(auth._authorize_run_id(run_id, "read") for run_id in run_ids)


# ---------------------------------------------------------------------------
# HTTP handlers.
# ---------------------------------------------------------------------------


def _json_response(message) -> "Response":
    from flask import Response

    response = Response(mimetype="application/json")
    response.set_data(message_to_json(message))
    return response


def _parse_search_lite_request() -> dict[str, Any]:
    request_json = _get_validated_flask_request_json(
        schema={
            "experiment_ids": [_assert_array],
            "filter": [_assert_string],
            "run_view_type": [_assert_string],
            "order_by": [_assert_array, _assert_item_type_string],
            "page_token": [_assert_string],
            "projection": [_assert_dict],
        }
    )
    _reject_unknown_fields(request_json, _ALLOWED_SEARCH_FIELDS, "search-lite")
    experiment_ids = request_json.get("experiment_ids") or []
    if not all(isinstance(e, (str, int)) for e in experiment_ids):
        raise _invalid_parameter("Field 'experiment_ids' must be a list of strings.")
    # max_results is validated manually because explicit null is meaningful
    # here (it means "use the bounded default", whereas the shared schema
    # helpers would reject null for an int-like field with a 500).
    raw_max_results = request_json.get("max_results")
    if raw_max_results is None:
        request_json = {**request_json, "max_results": DEFAULT_SEARCH_LITE_MAX_RESULTS}
    else:
        if isinstance(raw_max_results, bool) or not isinstance(raw_max_results, (str, int)):
            raise _invalid_parameter(
                "Field 'max_results' must be an integer. "
                f"Hint: Value was of type '{type(raw_max_results).__name__}'."
            )
        try:
            max_results = int(raw_max_results)
        except ValueError:
            raise _invalid_parameter(
                f"Field 'max_results' must be an integer. Got: {raw_max_results!r}."
            )
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise _invalid_parameter(
                f"Field 'max_results' must be at most {SEARCH_MAX_RESULTS_THRESHOLD}. "
                f"Got: {max_results}."
            )
        request_json = {**request_json, "max_results": max_results}
    return request_json


@catch_mlflow_exception
@_disable_if_artifacts_only
def _search_runs_lite():
    """POST handler for search-lite (both /api/2.0 and /ajax-api/2.0)."""
    request_json = _parse_search_lite_request()
    projection = _parse_projection(request_json.get("projection"))
    run_view_type = _parse_run_view_type(request_json.get("run_view_type"))

    # Self-authorizing route: same auth.filter_experiment_ids guard as legacy
    # search_runs_impl (the auth before-request hook allows the route and the
    # handler restricts the query to readable experiments).
    experiment_ids = _filter_authorized_experiment_ids([
        str(e) for e in (request_json.get("experiment_ids") or [])
    ])

    store = _get_tracking_store()
    # max_results was normalized (defaulted/coerced/capped) during parsing.
    max_results = request_json["max_results"]
    runs, next_page_token = search_runs_projected(
        store,
        experiment_ids=experiment_ids,
        filter_string=request_json.get("filter") or "",
        run_view_type=run_view_type,
        max_results=max_results,
        order_by=request_json.get("order_by") or [],
        page_token=request_json.get("page_token") or None,
        projection=projection,
    )

    response_message = SearchRuns.Response()
    response_message.runs.extend([run.to_proto() for run in runs])
    if next_page_token:
        response_message.next_page_token = next_page_token
    return _json_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_run_lite():
    """POST handler for get-lite (both /api/2.0 and /ajax-api/2.0)."""
    request_json = _get_validated_flask_request_json(
        schema={
            "run_id": [_assert_string],
            "run_uuid": [_assert_string],
            "projection": [_assert_dict],
        }
    )
    _reject_unknown_fields(request_json, {"run_id", "run_uuid", "projection"}, "get-lite")
    projection = _parse_projection(request_json.get("projection"))
    run_id = request_json.get("run_id")
    run_uuid = request_json.get("run_uuid")
    if run_id and run_uuid and run_id != run_uuid:
        raise _invalid_parameter(
            "Conflicting values for parameters 'run_id' and 'run_uuid'; provide only one."
        )
    run_id = run_id or run_uuid
    if not run_id:
        raise _invalid_parameter("Missing value for required parameter 'run_id'.")

    run = get_run_projected(_get_tracking_store(), run_id, projection)
    response_message = GetRun.Response()
    response_message.run.MergeFrom(run.to_proto())
    return _json_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_run_data_keys():
    """POST handler for data-keys (both /api/2.0 and /ajax-api/2.0)."""
    request_json = _get_validated_flask_request_json(
        schema={"run_ids": [_assert_array, _assert_item_type_string]}
    )
    _reject_unknown_fields(request_json, {"run_ids"}, "data-keys")
    run_ids = _validate_run_ids(request_json.get("run_ids"))

    from flask import jsonify

    return jsonify(get_run_data_keys(_get_tracking_store(), run_ids))


def get_projected_run_endpoints() -> list[tuple[str, Any, list[str]]]:
    """Endpoint tuples for the projected run reads, on api and ajax prefixes.

    Appended to ``handlers.get_endpoints()``; the Flask app registers each
    tuple, and the auth module wires validators/redactors to these exact paths.
    """
    return [
        (http_path, handler, ["POST"])
        for path, handler in (
            (SEARCH_RUNS_LITE, _search_runs_lite),
            (GET_RUN_LITE, _get_run_lite),
            (RUN_DATA_KEYS, _get_run_data_keys),
        )
        for http_path in _get_paths(path)
    ]
