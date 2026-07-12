"""Manifest-canonical response-norm analysis payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    ResolvedAnalysisInput,
    register_analysis_recipe,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from feedbax.contracts.manifest import AnalysisRunSpec
from pydantic import BaseModel, ConfigDict, Field


RESPONSE_NORM_ANALYSIS_TYPE = "rlrmp.response_norm_comparison"
RESPONSE_NORM_PAYLOAD_ROLE = "rlrmp-response-norm-comparison-payload"
RESPONSE_NORM_PAYLOAD_SCHEMA_ID = "rlrmp.figure_data.response_norm_comparison"
RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION = "rlrmp.figure_data.response_norm_comparison.v1"
INTRINSIC_METRICS = ("delta_position", "delta_action")
INTRINSIC_CONDITION_CLASSES = ("class_a", "class_b")


class ResponseNormAnalysisParams(BaseModel):
    """Parameters for a response-norm comparison payload."""

    model_config = ConfigDict(extra="allow")

    rows: list[dict[str, Any]] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: list(INTRINSIC_METRICS))
    condition_classes: list[str] = Field(
        default_factory=lambda: list(INTRINSIC_CONDITION_CLASSES)
    )


class ResponseNormAnalysis(AbstractAnalysis):
    """Normalize post-run response curves into a custody-owned figure payload."""

    params: dict[str, Any] = eqx.field(default_factory=dict, static=True)
    analysis_type: str = eqx.field(default=RESPONSE_NORM_ANALYSIS_TYPE, static=True)

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        params = ResponseNormAnalysisParams.model_validate(self.params)
        rows = params.rows or _rows_from_states(data.states)
        return response_norm_payload(
            rows,
            metrics=params.metrics,
            condition_classes=params.condition_classes,
        )

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        del data, kwargs
        context.record_json_artifact(
            dict(result),
            role=RESPONSE_NORM_PAYLOAD_ROLE,
            logical_name="response_norm/response_norm_comparison_payload.json",
            metadata={
                "schema_boundary": RESPONSE_NORM_PAYLOAD_SCHEMA_ID,
                "analysis_type": self.analysis_type,
            },
        )
        return result


def response_norm_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = INTRINSIC_METRICS,
    condition_classes: Sequence[str] = INTRINSIC_CONDITION_CLASSES,
) -> dict[str, Any]:
    """Build schema-bearing response-norm facets from post-run curve rows.

    Metric and condition-class axes are intrinsic to the scientific view. Model
    and comparison-row axes are data-bound and therefore derived from the payload
    rather than encoded as fixed subplot counts.
    """
    normalized = [_normalize_row(row) for row in rows]
    models = list(dict.fromkeys(str(row["model_id"]) for row in normalized))
    comparison_rows = list(dict.fromkeys(str(row["row_id"]) for row in normalized))
    facets: dict[str, Any] = {}
    for metric in metrics:
        facets[str(metric)] = {}
        for condition_class in condition_classes:
            facets[str(metric)][str(condition_class)] = {
                "metric": str(metric),
                "condition_class": str(condition_class),
                "models": models,
                "comparison_rows": comparison_rows,
                "curves": [
                    row
                    for row in normalized
                    if row["metric"] == metric and row["condition_class"] == condition_class
                ],
            }
    return {
        "schema_id": RESPONSE_NORM_PAYLOAD_SCHEMA_ID,
        "schema_version": RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION,
        "intrinsic_axes": {
            "metric": {str(value): str(value) for value in metrics},
            "condition_class": {
                str(value): str(value) for value in condition_classes
            },
        },
        "data_bound_axes": {
            "model": models,
            "comparison_row": comparison_rows,
        },
        "facets": facets,
    }


def response_norm_recipe(
    spec: AnalysisRunSpec,
    _root: Any,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build response-norm analysis from upstream manifest-owned states."""
    params = ResponseNormAnalysisParams.model_validate(spec.params)
    states = {
        "inputs": [resolved.states or {} for resolved in inputs],
    }
    analysis = ResponseNormAnalysis(
        variant="response_norm_comparison",
        params=params.model_dump(mode="json"),
        analysis_type=spec.analysis_type,
    )
    data = AnalysisInputData(
        models={},
        tasks={},
        states=states,
        hps={"response_norm": TreeNamespace(n_inputs=len(inputs))},
        extras={},
    )
    return AnalysisRecipeResult(analyses={"response_norm": analysis}, data=data)


def register_response_norm_recipes(*, replace: bool = True) -> None:
    """Register the living post-run response-norm analysis recipe."""
    register_analysis_recipe(
        RESPONSE_NORM_ANALYSIS_TYPE,
        response_norm_recipe,
        replace=replace,
    )


response_norm_recipe.EVAL_DEPENDENCIES = ("rlrmp.eval.perturbation_response_bank",)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required = ("row_id", "model_id", "metric", "condition_class", "time", "mean")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"response-norm row missing required fields: {', '.join(missing)}")
    result = {str(key): value for key, value in row.items()}
    result.setdefault("sem", None)
    result.setdefault("max", None)
    result.setdefault("label", str(result["model_id"]))
    result.setdefault("is_lqg", False)
    return result


def _rows_from_states(states: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(states, Mapping):
        candidates = states.get("inputs", states.values())
    else:
        candidates = ()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        payload = candidate.get("response_norm", candidate)
        if isinstance(payload, Mapping):
            value = payload.get("rows", ())
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                rows.extend(dict(row) for row in value if isinstance(row, Mapping))
    return rows


__all__ = [
    "INTRINSIC_CONDITION_CLASSES",
    "INTRINSIC_METRICS",
    "RESPONSE_NORM_ANALYSIS_TYPE",
    "RESPONSE_NORM_PAYLOAD_ROLE",
    "RESPONSE_NORM_PAYLOAD_SCHEMA_ID",
    "RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION",
    "ResponseNormAnalysis",
    "ResponseNormAnalysisParams",
    "register_response_norm_recipes",
    "response_norm_payload",
    "response_norm_recipe",
]
