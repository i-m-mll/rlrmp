"""Cached broad-epsilon tree-attribution summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from feedbax.analysis import StagedExecutionContext
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    ResolvedAnalysisInput,
    register_analysis_recipe,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, Field

BROAD_EPSILON_EVALUATION_TYPE = "rlrmp.eval.broad_epsilon"
BROAD_EPSILON_ATTRIBUTION_ANALYSIS_TYPE = "rlrmp.analysis.broad_epsilon_attribution"


class BroadEpsilonAttributionParams(BaseModel):
    """Governed attribution selection and gradient budget."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = "rlrmp.analysis.broad_epsilon_attribution.params.v1"
    schema_version: str = "v1"
    run_ids: tuple[str, ...] = ()
    max_gradient_replicates: int = Field(default=1, ge=0)


class BroadEpsilonAttributionAnalysis(AbstractAnalysis):
    """Aggregate cached active/zero losses and parameter-gradient trees."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        rows = list(data.states["rows"])
        deltas = [
            float(row["loss"]["delta_active_minus_zero"]["total"])
            for row in rows
            if isinstance(row, Mapping) and "loss" in row
        ]
        return {
            "schema_version": "rlrmp.analysis.broad_epsilon_attribution.v1",
            "params": dict(data.extras["params"]),
            "rows": rows,
            "summary": {
                "n_rows": len(rows),
                "n_evaluated": len(deltas),
                "mean_total_loss_delta_active_minus_zero": (
                    None if not deltas else sum(deltas) / len(deltas)
                ),
            },
        }


def register_broad_epsilon_attribution_recipe(*, replace: bool = True) -> None:
    """Register cached broad-epsilon attribution analysis."""

    register_analysis_recipe(
        BROAD_EPSILON_ATTRIBUTION_ANALYSIS_TYPE,
        broad_epsilon_attribution_recipe,
        replace=replace,
    )


def broad_epsilon_attribution_recipe(
    spec,
    _root,
    inputs: Sequence[ResolvedAnalysisInput],
    _execution_context: StagedExecutionContext,
) -> AnalysisRecipeResult:
    """Build attribution analysis from evaluation-manifest states only."""

    params = BroadEpsilonAttributionParams.model_validate(spec.params)
    rows: list[Any] = []
    for resolved in inputs:
        states = resolved.states
        if isinstance(states, Mapping):
            payload = states.get("rows", states.get("runs", ()))
            if isinstance(payload, Mapping):
                rows.extend(payload.values())
            elif isinstance(payload, Sequence):
                rows.extend(payload)
    return AnalysisRecipeResult(
        analyses={
            "attribution": BroadEpsilonAttributionAnalysis(variant="broad_epsilon_attribution")
        },
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"rows": rows},
            hps={"broad_epsilon": TreeNamespace(task=TreeNamespace(eval_n=len(rows)))},
            extras={"params": params.model_dump(mode="json")},
        ),
    )


broad_epsilon_attribution_recipe.EVAL_DEPENDENCIES = (BROAD_EPSILON_EVALUATION_TYPE,)


__all__ = [
    "BROAD_EPSILON_ATTRIBUTION_ANALYSIS_TYPE",
    "BROAD_EPSILON_EVALUATION_TYPE",
    "BroadEpsilonAttributionAnalysis",
    "BroadEpsilonAttributionParams",
    "register_broad_epsilon_attribution_recipe",
]
