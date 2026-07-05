"""Shared manifest lookups backed by Feedbax path expressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from feedbax.contracts.expressions import (
    Compare,
    ContextItem,
    ExpressionContext,
    ExpressionEvaluationError,
    ExpressionSelectAmbiguous,
    Select,
    ValueQuery,
    evaluate_query,
)

_STANDARD_ROW_BY_SOURCE_RUN_ID = ValueQuery(
    item="manifest",
    path="rows",
    select=Select(
        where=Compare(
            item="entry",
            path="spec.parameters.source_run_id",
            op="eq",
            value="",
        )
    ),
)

_CERTIFICATE_COMPONENT_BY_NAME = ValueQuery(
    item="row",
    path="certificate_components",
    select=Select(
        where=Compare(
            item="entry",
            path="name",
            op="eq",
            value="",
        )
    ),
)


def standard_row_by_source_run_id(
    manifest: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any] | None:
    """Return the unique standard-certificate row for ``run_id``, if present."""

    query = _STANDARD_ROW_BY_SOURCE_RUN_ID.model_copy(
        update={
            "select": Select(
                where=Compare(
                    item="entry",
                    path="spec.parameters.source_run_id",
                    op="eq",
                    value=run_id,
                )
            )
        }
    )
    rows = manifest.get("rows", ())
    matches = [
        row
        for row in _sequence(rows)
        if row.get("spec", {}).get("parameters", {}).get("source_run_id") == run_id
    ]
    return _query_exact_or_none(
        query,
        _context("manifest", manifest),
        matches=matches,
        description=f"standard row for source_run_id={run_id!r}",
    )


def certificate_component_summary(
    row: Mapping[str, Any],
    component_name: str,
) -> dict[str, Any] | None:
    """Return status, summary, and reason for one certificate component."""

    component = certificate_component(row, component_name)
    if component is None:
        return None
    return {
        "status": component.get("status"),
        "summary": component.get("summary"),
        "reason": component.get("reason"),
    }


def certificate_component_summary_value(
    row: Mapping[str, Any],
    component_name: str,
    key: str,
) -> Any:
    """Return one summary value from a named certificate component."""

    component = certificate_component(row, component_name)
    if component is None:
        return None
    try:
        return evaluate_query(
            ValueQuery(item="component", path=f"summary.{key}"),
            _context("component", component),
        )
    except ExpressionEvaluationError:
        return None


def certificate_component(
    row: Mapping[str, Any],
    component_name: str,
) -> dict[str, Any] | None:
    """Return the unique named certificate component from ``row``, if present."""

    query = _CERTIFICATE_COMPONENT_BY_NAME.model_copy(
        update={
            "select": Select(
                where=Compare(item="entry", path="name", op="eq", value=component_name)
            )
        }
    )
    components = row.get("certificate_components", ())
    matches = [
        component
        for component in _sequence(components)
        if component.get("name") == component_name
    ]
    return _query_exact_or_none(
        query,
        _context("row", row),
        matches=matches,
        description=f"certificate component {component_name!r}",
    )


def _query_exact_or_none(
    query: ValueQuery,
    ctx: ExpressionContext,
    *,
    matches: Sequence[dict[str, Any]],
    description: str,
) -> dict[str, Any] | None:
    if not matches:
        return None
    try:
        result = evaluate_query(query, ctx)
    except ExpressionSelectAmbiguous as exc:
        raise ValueError(
            f"Expected exactly one {description}; found {len(matches)}."
        ) from exc
    if not isinstance(result, dict):
        raise TypeError(f"Expected {description} to resolve to a dict, got {type(result).__name__}.")
    return result


def _context(item: str, payload: Mapping[str, Any]) -> ExpressionContext:
    return ExpressionContext(items={item: ContextItem(kind=item, payload=payload)})


def _sequence(value: Any) -> Sequence[dict[str, Any]]:
    if isinstance(value, list | tuple):
        return [entry for entry in value if isinstance(entry, dict)]
    return ()
