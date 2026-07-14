"""Registered manifest-to-history payload adapter for declarative figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.analysis import StagedExecutionContext
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    ResolvedAnalysisInput,
    register_analysis_recipe,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace

from rlrmp.mappings import as_mapping as _mapping


HISTORY_PAYLOAD_ANALYSIS_TYPE = "rlrmp.history_payload"
HISTORY_PAYLOAD_ROLE = "history_figure_payload"
HISTORY_PAYLOAD_SCHEMA_ID = "rlrmp.figure_data.history"
HISTORY_PAYLOAD_SCHEMA_VERSION = "rlrmp.figure_data.history.v1"


class HistoryPayloadAnalysis(AbstractAnalysis):
    """Publish manifest-backed line histories for native figure composition."""

    def compute(self, data: AnalysisInputData, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_id": HISTORY_PAYLOAD_SCHEMA_ID,
            "schema_version": HISTORY_PAYLOAD_SCHEMA_VERSION,
            "facets": dict(data.states["facets"]),
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload = dict(result)
        context.metadata = {
            **dict(context.metadata or {}),
            "history_payload": payload,
        }
        artifact = context.record_json_artifact(
            payload,
            role=HISTORY_PAYLOAD_ROLE,
            logical_name="history_payload/payload.json",
            metadata={"schema_version": HISTORY_PAYLOAD_SCHEMA_VERSION},
        )
        return {"payload": payload, "artifact_refs": {"payload": artifact}}


def register_history_payload_recipe(*, replace: bool = True) -> None:
    """Register the generic manifest-backed history payload analysis."""
    register_analysis_recipe(
        HISTORY_PAYLOAD_ANALYSIS_TYPE,
        history_payload_recipe,
        replace=replace,
    )


def history_payload_recipe(
    spec: Any,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
    _execution_context: StagedExecutionContext,
) -> AnalysisRecipeResult:
    """Build history facets from resolved evaluation-manifest states.

    Each input may expose a ``histories`` mapping, a caller-selected mapping
    named by ``params.history_key``, or one direct ``x``/``series`` record.
    Figure intent remains intrinsic; labels and trace values stay data-bound to
    the resolved manifests.
    """
    params = dict(spec.params)
    history_key = str(params.get("history_key", "histories"))
    facets: dict[str, dict[str, Any]] = {}
    for resolved in inputs:
        source = _mapping(resolved.states)
        raw = source.get(history_key, source.get("histories", source))
        for label, history in _history_records(raw, fallback_label=resolved.ref.id):
            if label in facets:
                raise ValueError(f"duplicate history facet {label!r}")
            facets[label] = _normalise_history(history, label=label)
    if not facets:
        raise ValueError("history payload analysis resolved no history facets")

    analysis = HistoryPayloadAnalysis(variant="history_payload", cache_result=True)
    return AnalysisRecipeResult(
        analyses={"history_payload": analysis},
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"facets": facets},
            hps={"history_payload": TreeNamespace(task=TreeNamespace(eval_n=len(facets)))},
            extras={"params": params},
        ),
    )


history_payload_recipe.EVAL_DEPENDENCIES = ("rlrmp.eval.center_out_ensemble",)


def _history_records(
    value: Any,
    *,
    fallback_label: str,
) -> list[tuple[str, Mapping[str, Any]]]:
    payload = _mapping(value)
    if "x" in payload and "series" in payload:
        return [(fallback_label, payload)]
    records = []
    for label, history in payload.items():
        if isinstance(history, Mapping):
            records.append((str(label), history))
    return records


def _normalise_history(history: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    series = _mapping(history.get("series"))
    if not series:
        raise ValueError(f"history facet {label!r} has no series")
    x = history.get("x", history.get("time", history.get("steps")))
    if x is None:
        first = next(iter(series.values()))
        x = list(range(len(first)))
    return {
        "x": _jsonable(x),
        "series": _jsonable(series),
        "color": history.get("color"),
        "label": str(history.get("label", label)),
        "annotations": _jsonable(history.get("annotations", [])),
        "summary": _jsonable(history.get("summary", {})),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child) for child in value]
    return value


__all__ = [
    "HISTORY_PAYLOAD_ANALYSIS_TYPE",
    "HISTORY_PAYLOAD_ROLE",
    "HISTORY_PAYLOAD_SCHEMA_ID",
    "HISTORY_PAYLOAD_SCHEMA_VERSION",
    "HistoryPayloadAnalysis",
    "history_payload_recipe",
    "register_history_payload_recipe",
]
