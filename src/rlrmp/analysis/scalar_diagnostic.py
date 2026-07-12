"""Registered payload analysis for scalar and dual-axis diagnostic figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    ResolvedAnalysisInput,
    register_analysis_recipe,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace


SCALAR_DIAGNOSTIC_ANALYSIS_TYPE = "rlrmp.scalar_diagnostic_payload"
SCALAR_DIAGNOSTIC_PAYLOAD_ROLE = "scalar_diagnostic_figure_payload"
SCALAR_DIAGNOSTIC_SCHEMA_ID = "rlrmp.figure_data.scalar_diagnostic"
SCALAR_DIAGNOSTIC_SCHEMA_VERSION = "rlrmp.figure_data.scalar_diagnostic.v1"


class ScalarDiagnosticAnalysis(AbstractAnalysis):
    """Publish manifest-backed diagnostic traces and headline summaries."""

    def compute(self, data: AnalysisInputData, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_id": SCALAR_DIAGNOSTIC_SCHEMA_ID,
            "schema_version": SCALAR_DIAGNOSTIC_SCHEMA_VERSION,
            "intrinsic_axes": dict(data.states["intrinsic_axes"]),
            "collections": dict(data.states["collections"]),
            "headlines": dict(data.states["headlines"]),
            "provenance": dict(data.states["provenance"]),
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
            "figure_payload": payload,
        }
        artifact = context.record_json_artifact(
            payload,
            role=SCALAR_DIAGNOSTIC_PAYLOAD_ROLE,
            logical_name="scalar_diagnostic/payload.json",
            metadata={"schema_version": SCALAR_DIAGNOSTIC_SCHEMA_VERSION},
        )
        return {"payload": payload, "artifact_refs": {"payload": artifact}}


def register_scalar_diagnostic_recipe(*, replace: bool = True) -> None:
    """Register the manifest-backed scalar-diagnostic payload analysis."""
    register_analysis_recipe(
        SCALAR_DIAGNOSTIC_ANALYSIS_TYPE,
        scalar_diagnostic_recipe,
        replace=replace,
    )


def scalar_diagnostic_recipe(
    spec: Any,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Combine diagnostic records without fixing data-bound row cardinality."""
    params = dict(spec.params)
    collections: dict[str, Any] = {}
    headlines: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for resolved in inputs:
        source = _mapping(resolved.states)
        payload = _mapping(source.get("scalar_diagnostic", source))
        for name, record in _mapping(payload.get("collections")).items():
            if name in collections:
                raise ValueError(f"duplicate scalar-diagnostic collection {name!r}")
            collections[str(name)] = _jsonable(record)
        headlines.update(_jsonable(_mapping(payload.get("headlines"))))
        provenance[resolved.ref.id] = {
            "manifest_kind": resolved.ref.kind,
            "manifest_id": resolved.ref.id,
            "role": resolved.ref.role,
        }
    if not collections:
        raise ValueError("scalar-diagnostic analysis resolved no collections")

    intrinsic_axes = _jsonable(
        params.get(
            "intrinsic_axes",
            {"metric": ["value"], "condition_class": ["observed"]},
        )
    )
    analysis = ScalarDiagnosticAnalysis(variant="scalar_diagnostic", cache_result=True)
    return AnalysisRecipeResult(
        analyses={"scalar_diagnostic": analysis},
        data=AnalysisInputData(
            models={},
            tasks={},
            states={
                "intrinsic_axes": intrinsic_axes,
                "collections": collections,
                "headlines": headlines,
                "provenance": provenance,
            },
            hps={"scalar_diagnostic": TreeNamespace(task=TreeNamespace(eval_n=len(inputs)))},
            extras={"params": params},
        ),
    )


scalar_diagnostic_recipe.EVAL_DEPENDENCIES = ("rlrmp.eval.center_out_ensemble",)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child) for child in value]
    return value


__all__ = [
    "SCALAR_DIAGNOSTIC_ANALYSIS_TYPE",
    "SCALAR_DIAGNOSTIC_PAYLOAD_ROLE",
    "SCALAR_DIAGNOSTIC_SCHEMA_ID",
    "SCALAR_DIAGNOSTIC_SCHEMA_VERSION",
    "ScalarDiagnosticAnalysis",
    "register_scalar_diagnostic_recipe",
    "scalar_diagnostic_recipe",
]
