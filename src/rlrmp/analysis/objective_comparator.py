"""Manifest-canonical objective-term comparison analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput, register_analysis_recipe
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.eval.objective_terms import OBJECTIVE_TERMS_EVALUATION_TYPE

OBJECTIVE_COMPARATOR_ANALYSIS_TYPE = "rlrmp.analysis.objective_comparator"
OBJECTIVE_COMPARATOR_SCHEMA = "rlrmp.objective_comparator_sidecar.v6"
STANDARD_SPLIT_BANK_LENSES = (
    "deterministic_nominal",
    "x0_position_only",
    "x0_velocity_only",
    "x0_force_filter_only",
    "x0_disturbance_integrator_only",
    "process_epsilon_position_only",
    "process_epsilon_velocity_only",
    "process_epsilon_force_filter_only",
    "process_epsilon_integrator_only",
    "x0_position_velocity",
    "x0_plus_epsilon",
)


class CachedObjectiveComparatorInput(BaseModel):
    """Evaluation-manifest state sufficient for archived v6 parity."""

    model_config = ConfigDict(extra="forbid")

    source_manifest: str | None = None
    checkpoint_policy: dict[str, Any]
    extlqg_decomposition: dict[str, Any]
    same_noise_bank_monte_carlo: dict[str, Any]
    per_term_realized_scoring: dict[str, Any]
    shared_rollout_comparator: dict[str, Any]
    standard_split_bank_comparator: dict[str, Any]
    rows: list[dict[str, Any]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        lenses = self.standard_split_bank_comparator.get("lenses", {})
        missing = [name for name in STANDARD_SPLIT_BANK_LENSES if name not in lenses]
        if missing:
            raise ValueError(f"standard split-bank comparator missing lenses: {missing}")
        if "fairness" not in self.standard_split_bank_comparator:
            raise ValueError("standard split-bank comparator requires fairness metadata")


def build_objective_comparator_sidecar_from_cached(
    cached: Mapping[str, Any],
    *,
    issue: str,
    scope: str,
    source_manifest: str,
) -> dict[str, Any]:
    """Project cached evaluation summaries into the archived v6 output shape."""

    payload = CachedObjectiveComparatorInput.model_validate(cached)
    return {
        "schema_version": OBJECTIVE_COMPARATOR_SCHEMA,
        "issue": issue,
        "scope": scope,
        "source_manifest": source_manifest,
        "generated_by": "rlrmp.analysis.objective_comparator",
        "checkpoint_policy": payload.checkpoint_policy,
        "objective_lenses": {
            "gru_validation_selected_realized_full_qrf": {
                "kind": "realized_validation_objective",
                "noise_bank": "evaluation_manifest_cached",
            },
            "extlqg_covariance_inclusive_expected_cost": {"kind": "expected_cost"},
            "same_noise_bank_monte_carlo_full_qrf": {
                "kind": "realized_same_noise_bank_monte_carlo"
            },
            "realized_full_qrf_per_term_validation": {
                "kind": "realized_validation_objective_decomposition"
            },
            "shared_rollout_full_qrf": {"kind": "realized_shared_rollout_comparison"},
            "standard_split_rollout_bank_full_qrf": {
                "kind": "realized_shared_rollout_split_bank",
                "checkpoint_selection_role": "audit_only_not_used_for_checkpoint_selection",
            },
        },
        "extlqg_decomposition": payload.extlqg_decomposition,
        "same_noise_bank_monte_carlo": payload.same_noise_bank_monte_carlo,
        "per_term_realized_scoring": payload.per_term_realized_scoring,
        "shared_rollout_comparator": payload.shared_rollout_comparator,
        "standard_split_bank_comparator": payload.standard_split_bank_comparator,
        "rows": payload.rows,
        "caveats": [
            "Comparator rows are audit-only and do not select checkpoints.",
            "Shared-noise and split-bank results are consumed from evaluation-manifest states.",
            "Split-bank hidden-state conditioning and fairness metadata remain explicit.",
        ],
    }


class ObjectiveComparatorParams(BaseModel):
    """Governed parameters for cached objective comparison."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = "rlrmp.analysis.objective_comparator.params"
    schema_version: str = "v1"
    scope: str = "validation_selected_objective_comparison"
    issue: str = "3a5be47"


class ObjectiveComparatorAnalysis(AbstractAnalysis):
    """Compare cached full-QRF term summaries without executing rollouts."""

    def compute(self, data: AnalysisInputData, **_kwargs: Any) -> dict[str, Any]:
        params = data.extras["params"]
        return build_objective_comparator_sidecar_from_cached(
            data.states["cached"],
            issue=params["issue"],
            scope=params["scope"],
            source_manifest=str(data.states["cached"].get("source_manifest", "evaluation_parent")),
        )


def _comparison_row(row: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(row.get("run_id", "unknown"))
    terms = row.get("terms")
    reference = row.get("reference_terms")
    if not isinstance(terms, Mapping) or not isinstance(reference, Mapping):
        return {
            "run_id": run_id,
            "status": "not_available",
            "reason": "cached evaluation lacks objective terms or reference terms",
        }
    term_rows = {
        name: _term_comparison(terms.get(name), reference.get(name))
        for name in sorted(set(terms) | set(reference))
    }
    total = _term_comparison(terms.get("total"), reference.get("total"))
    return {
        "run_id": run_id,
        "status": "available" if total["status"] == "available" else "not_available",
        "checkpoint_selection": row.get("checkpoint_selection"),
        "bank": row.get("bank"),
        "terms": term_rows,
        "total": total,
    }


def _term_comparison(value: Any, reference: Any) -> dict[str, Any]:
    value_mean = _scalar_mean(value)
    reference_mean = _scalar_mean(reference)
    if value_mean is None or reference_mean is None:
        return {"status": "not_available", "value": value_mean, "reference": reference_mean}
    return {
        "status": "available",
        "value": value_mean,
        "reference": reference_mean,
        "delta": value_mean - reference_mean,
        "ratio_to_reference": None if reference_mean == 0.0 else value_mean / reference_mean,
    }


def _scalar_mean(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("mean", value.get("value"))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: Sequence[float | None] | Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def objective_comparator_recipe(
    spec: Any,
    _root: Any,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build the comparator exclusively from evaluation-manifest states."""

    params = ObjectiveComparatorParams.model_validate(spec.params)
    cached: Mapping[str, Any] | None = None
    for resolved in inputs:
        states = resolved.states
        if isinstance(states, Mapping) and "standard_split_bank_comparator" in states:
            cached = states
            break
    if cached is None:
        raise ValueError("objective comparator requires cached evaluation-manifest summaries")
    return AnalysisRecipeResult(
        analyses={"objective_comparator": ObjectiveComparatorAnalysis(variant="objective_comparator")},
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"cached": cached},
            hps={"objective_comparator": TreeNamespace(task=TreeNamespace(eval_n=len(cached.get("rows", ()))) )},
            extras={"params": params.model_dump(mode="json")},
        ),
    )


objective_comparator_recipe.EVAL_DEPENDENCIES = (OBJECTIVE_TERMS_EVALUATION_TYPE,)


def register_objective_comparator_recipe(*, replace: bool = True) -> None:
    """Register the cached objective-comparison analysis recipe."""

    register_analysis_recipe(
        OBJECTIVE_COMPARATOR_ANALYSIS_TYPE,
        objective_comparator_recipe,
        replace=replace,
    )


__all__ = [
    "OBJECTIVE_COMPARATOR_ANALYSIS_TYPE",
    "OBJECTIVE_COMPARATOR_SCHEMA",
    "STANDARD_SPLIT_BANK_LENSES",
    "CachedObjectiveComparatorInput",
    "ObjectiveComparatorAnalysis",
    "ObjectiveComparatorParams",
    "register_objective_comparator_recipe",
    "build_objective_comparator_sidecar_from_cached",
]
