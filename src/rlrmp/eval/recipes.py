"""Manifest-canonical rlrmp evaluation recipes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    ParentRef,
)

from rlrmp.runtime.spec_migrations import (
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    DELAYED_REACH_BANK_EVAL_PARAMS_KIND,
    FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
    WORST_CASE_EPSILON_EVAL_PARAMS_KIND,
    accept_rlrmp_spec_payload,
)

CENTER_OUT_ENSEMBLE_EVALUATION_TYPE = "rlrmp.eval.center_out_ensemble"
PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE = "rlrmp.eval.perturbation_response_bank"
FEEDBACK_ABLATION_EVALUATION_TYPE = "rlrmp.eval.feedback_ablation"
WORST_CASE_EPSILON_EVALUATION_TYPE = "rlrmp.eval.worst_case_epsilon"
DELAYED_REACH_BANK_EVALUATION_TYPE = "rlrmp.eval.delayed_reach_bank"

_RECIPE_PARAM_KINDS = {
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE: CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE: PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
    FEEDBACK_ABLATION_EVALUATION_TYPE: FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    WORST_CASE_EPSILON_EVALUATION_TYPE: WORST_CASE_EPSILON_EVAL_PARAMS_KIND,
    DELAYED_REACH_BANK_EVALUATION_TYPE: DELAYED_REACH_BANK_EVAL_PARAMS_KIND,
}


def register_rlrmp_evaluation_recipes(*, replace: bool = True) -> None:
    """Register rlrmp's manifest-canonical evaluation recipes."""

    register_evaluation_recipe(
        CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
        center_out_ensemble_recipe,
        replace=replace,
    )
    register_evaluation_recipe(
        PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
        perturbation_response_bank_recipe,
        replace=replace,
    )
    register_evaluation_recipe(
        FEEDBACK_ABLATION_EVALUATION_TYPE,
        feedback_ablation_recipe,
        replace=replace,
    )
    register_evaluation_recipe(
        WORST_CASE_EPSILON_EVALUATION_TYPE,
        worst_case_epsilon_recipe,
        replace=replace,
    )
    register_evaluation_recipe(
        DELAYED_REACH_BANK_EVALUATION_TYPE,
        delayed_reach_bank_recipe,
        replace=replace,
    )


def center_out_ensemble_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate the shared center-out/delayed-reach ensemble recipe contract."""

    params = _validated_params(run_spec)
    _require_one_of(params, ("task",), recipe=run_spec.evaluation_type)
    return _result(
        run_spec,
        params,
        product_role="center_out_ensemble_states",
        state_payload={
            "task": params.get("task"),
            "checkpoint_selector": params.get("checkpoint_selector"),
            "replicate_selector": params.get("replicate_selector"),
            "trial_specs": _subset(
                params,
                (
                    "n_directions",
                    "n_trials_per_direction",
                    "n_trials",
                    "directions",
                    "task_conditions",
                ),
            ),
            "perturbation": params.get("perturbation", params.get("pert_axis")),
            "sisu_values": params.get("sisu_values", []),
            "seed": params.get("seed"),
            "trajectories": params.get("trajectories", []),
            "kinematics_summary": params.get("kinematics_summary", {}),
            "legacy_diagnostics_manifest": params.get("legacy_diagnostics_manifest"),
            "legacy_bulk_arrays": params.get("legacy_bulk_arrays", {}),
        },
    )


def perturbation_response_bank_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate a perturbation-response bank request."""

    params = _validated_params(run_spec)
    _require_one_of(
        params,
        ("checkpoint_bank_ref", "checkpoint_bank"),
        recipe=run_spec.evaluation_type,
    )
    return _result(
        run_spec,
        params,
        product_role="perturbation_response_bank",
        state_payload={
            "checkpoint_bank": params.get("checkpoint_bank_ref", params.get("checkpoint_bank")),
            "perturbation_battery": params.get("perturbation_battery", {}),
            "alignment_mode": params.get("alignment_mode", "reach_locked"),
            "response_tensors": params.get("response_tensors", {}),
            "bank_status": params.get("bank_status", {}),
        },
    )


def feedback_ablation_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate intact-vs-ablated feedback rollout pairs."""

    params = _validated_params(run_spec)
    _require_one_of(
        params,
        ("ablation_masks", "ablation_mask_set"),
        recipe=run_spec.evaluation_type,
    )
    return _result(
        run_spec,
        params,
        product_role="feedback_ablation_rollouts",
        state_payload={
            "ablation_masks": params.get("ablation_masks", params.get("ablation_mask_set")),
            "base_task": params.get("base_task", {}),
            "rollout_pairs": params.get("rollout_pairs", []),
        },
    )


def worst_case_epsilon_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate worst-case epsilon perturbation audits."""

    params = _validated_params(run_spec)
    _require_one_of(
        params,
        ("epsilon_budget_data_product_identity", "epsilon_budget_identity"),
        recipe=run_spec.evaluation_type,
    )
    return _result(
        run_spec,
        params,
        product_role="worst_case_epsilon_rollouts",
        state_payload={
            "epsilon_budget_identity": params.get(
                "epsilon_budget_data_product_identity",
                params.get("epsilon_budget_identity"),
            ),
            "optimizer": params.get("optimizer", {}),
            "audit_inputs": params.get("audit_inputs", {}),
            "worst_case_rollouts": params.get("worst_case_rollouts", []),
        },
    )


def delayed_reach_bank_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate delayed-reach checkpoint-selection banks."""

    params = _validated_params(run_spec)
    _require_one_of(params, ("bank_spec",), recipe=run_spec.evaluation_type)
    return _result(
        run_spec,
        params,
        product_role="delayed_reach_eval_bank",
        state_payload={
            "bank_spec": params.get("bank_spec", {}),
            "bank_tensors": params.get("bank_tensors", {}),
            "selection_inputs": params.get("selection_inputs", {}),
        },
    )


def _validated_params(run_spec: EvaluationRunSpec) -> dict[str, Any]:
    kind = _RECIPE_PARAM_KINDS[run_spec.evaluation_type]
    result = accept_rlrmp_spec_payload(kind, run_spec.params)
    return dict(result.payload)


def _result(
    run_spec: EvaluationRunSpec,
    params: Mapping[str, Any],
    *,
    product_role: str,
    state_payload: Mapping[str, Any],
) -> EvaluationRecipeResult:
    manifest_id = _evaluation_manifest_id(run_spec)
    refs = _parent_refs(run_spec)
    states = {
        "schema_id": params.get("schema_id"),
        "schema_version": params.get("schema_version"),
        "evaluation_type": run_spec.evaluation_type,
        "evaluation_manifest_id": manifest_id,
        "product_role": product_role,
        "input_training_run_ids": list(run_spec.training_run_ids),
        "input_refs": refs,
        "consumed_data_identities": _consumed_data_identities(params),
        **state_payload,
    }
    summary_metrics = {
        "input_ref_count": len(refs),
        "training_run_id_count": len(run_spec.training_run_ids),
        "consumed_data_identity_count": len(states["consumed_data_identities"]),
    }
    return EvaluationRecipeResult(
        states=states,
        summary_metrics=summary_metrics,
        metadata={
            "rlrmp_evaluation_recipe": run_spec.evaluation_type,
            "product_role": product_role,
            "params_schema_id": params.get("schema_id"),
            "params_schema_version": params.get("schema_version"),
            "params_sha256": _sha256(_canonical_json_bytes(params)),
            "caching_identity": {
                "source": "EvaluationRunSpec",
                "manifest_id": manifest_id,
                "rule": (
                    "Feedbax hashes the canonical EvaluationRunSpec; all task, "
                    "seed, selector, perturbation, and data-product identity knobs "
                    "must live in params or inputs."
                ),
            },
        },
    )


def _parent_refs(run_spec: EvaluationRunSpec) -> list[dict[str, Any]]:
    refs: list[ParentRef] = list(run_spec.inputs)
    for run_id in run_spec.training_run_ids:
        ref = ParentRef(kind="TrainingRunManifest", id=run_id, role="training_run")
        if all(existing.id != ref.id or existing.kind != ref.kind for existing in refs):
            refs.append(ref)
    return [ref.model_dump(mode="json", exclude_none=True) for ref in refs]


def _consumed_data_identities(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    identities = params.get("consumed_data_identities", [])
    if isinstance(identities, Mapping):
        return [dict(identities)]
    if isinstance(identities, Sequence) and not isinstance(identities, (str, bytes)):
        return [dict(item) for item in identities if isinstance(item, Mapping)]
    return []


def _subset(params: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: params[key] for key in keys if key in params}


def _require_one_of(params: Mapping[str, Any], keys: Sequence[str], *, recipe: str) -> None:
    if not any(key in params for key in keys):
        joined = ", ".join(keys)
        raise ValueError(f"{recipe} params must include one of: {joined}")


def _evaluation_manifest_id(run_spec: EvaluationRunSpec) -> str:
    digest = _sha256(_canonical_json_bytes(run_spec))
    return f"feedbax-evaluation-run:{digest[:32]}"


def _canonical_json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "CENTER_OUT_ENSEMBLE_EVALUATION_TYPE",
    "DELAYED_REACH_BANK_EVALUATION_TYPE",
    "FEEDBACK_ABLATION_EVALUATION_TYPE",
    "PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE",
    "WORST_CASE_EPSILON_EVALUATION_TYPE",
    "center_out_ensemble_recipe",
    "delayed_reach_bank_recipe",
    "feedback_ablation_recipe",
    "perturbation_response_bank_recipe",
    "register_rlrmp_evaluation_recipes",
    "worst_case_epsilon_recipe",
]
