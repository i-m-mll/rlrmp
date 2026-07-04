"""Manifest-canonical rlrmp evaluation recipes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    ParentRef,
)
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.runtime.params_models import params_model_for, register_params_model
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


class _StrictParamsModel(BaseModel):
    """Base class for strict current-version recipe params."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    schema_version: str | None = None
    consumed_data_identities: list[dict[str, Any]] | dict[str, Any] = Field(
        default_factory=list
    )


class CenterOutEnsembleEvalParams(_StrictParamsModel):
    """Params for the shared center-out/delayed-reach ensemble eval recipe."""

    task: Any | None = None
    checkpoint_selector: Any | None = None
    replicate_selector: Any | None = None
    n_directions: Any | None = None
    n_trials_per_direction: Any | None = None
    n_trials: Any | None = None
    directions: Any | None = None
    task_conditions: Any | None = None
    perturbation: Any | None = None
    pert_axis: Any | None = None
    sisu_values: list[Any] = Field(default_factory=list)
    seed: Any | None = None
    trajectories: list[Any] = Field(default_factory=list)
    kinematics_summary: dict[str, Any] = Field(default_factory=dict)
    legacy_diagnostics_manifest: Any | None = None
    legacy_bulk_arrays: dict[str, Any] = Field(default_factory=dict)
    gru_standard_certificate: dict[str, Any] | None = None


class PerturbationResponseBankEvalParams(_StrictParamsModel):
    """Params for perturbation-response bank evaluation."""

    checkpoint_bank_ref: Any | None = None
    checkpoint_bank: Any | None = None
    perturbation_battery: Any | None = None
    bank: Any | None = None
    alignment_mode: Literal["reach_locked"] = "reach_locked"
    response_tensors: Any | None = None
    class_index_map: Any | None = None
    bank_status: dict[str, Any] = Field(default_factory=dict)
    legacy_payload_mode: bool = False
    source_experiment: str | None = None
    experiment: str | None = None
    run_ids: list[str] | None = None
    labels: list[str] | None = None
    class_set: str | list[str] | None = None
    families: str | list[str] | None = None
    family_set: str | list[str] | None = None
    perturbation_families: str | list[str] | None = None
    perturbation_ids: str | list[str] | None = None
    consume_open_loop_calibration: bool = False
    bank_mode: Literal["raw", "calibrated"] | None = None
    mode: Literal["raw", "calibrated"] = "raw"
    calibration_level: Any | None = None
    calibration_reach: Any | None = None
    feedback_scale_manifest: Any | None = None
    feedback_scale_manifest_path: str | None = None
    repo_root: str | None = None
    bulk_dir: str | None = None
    write_bulk_arrays: bool = False
    n_rollout_trials: int = Field(8, ge=1)
    extlqg_physical_dim: Literal[6, 8] = 8
    preferred_checkpoint_manifest_path: str | None = None
    checkpoint_selection_mode: Literal["sparse_history", "fixed_bank_manifest"] = (
        "sparse_history"
    )


class FeedbackAblationEvalParams(_StrictParamsModel):
    """Params for feedback-ablation evaluation."""

    ablation_masks: Any | None = None
    ablation_mask_set: Any | None = None
    base_task: dict[str, Any] = Field(default_factory=dict)
    rollout_pairs: list[Any] = Field(default_factory=list)


class WorstCaseEpsilonEvalParams(_StrictParamsModel):
    """Params for worst-case epsilon evaluation."""

    epsilon_budget_data_product_identity: Any | None = None
    epsilon_budget_identity: Any | None = None
    optimizer: dict[str, Any] = Field(default_factory=dict)
    audit_inputs: dict[str, Any] = Field(default_factory=dict)
    worst_case_rollouts: list[Any] = Field(default_factory=list)


class DelayedReachBankEvalParams(_StrictParamsModel):
    """Params for delayed-reach bank evaluation."""

    bank_spec: dict[str, Any] = Field(default_factory=dict)
    bank_tensors: dict[str, Any] = Field(default_factory=dict)
    selection_inputs: dict[str, Any] = Field(default_factory=dict)


_PARAMS_MODEL_BY_RECIPE = {
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE: CenterOutEnsembleEvalParams,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE: PerturbationResponseBankEvalParams,
    FEEDBACK_ABLATION_EVALUATION_TYPE: FeedbackAblationEvalParams,
    WORST_CASE_EPSILON_EVALUATION_TYPE: WorstCaseEpsilonEvalParams,
    DELAYED_REACH_BANK_EVALUATION_TYPE: DelayedReachBankEvalParams,
}


def register_rlrmp_evaluation_recipes(*, replace: bool = True) -> None:
    """Register rlrmp's manifest-canonical evaluation recipes."""

    for recipe_name, model_class in _PARAMS_MODEL_BY_RECIPE.items():
        register_params_model(recipe_name, model_class, replace=replace)
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
    root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate the shared center-out/delayed-reach ensemble recipe contract."""

    p, params = _validated_params(run_spec)
    _require_one_of(params, ("task",), recipe=run_spec.evaluation_type)
    return _result(
        run_spec,
        params,
        product_role="center_out_ensemble_states",
        state_payload={
            "task": p.task,
            "checkpoint_selector": p.checkpoint_selector,
            "replicate_selector": p.replicate_selector,
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
            "perturbation": p.perturbation if p.perturbation is not None else p.pert_axis,
            "sisu_values": p.sisu_values,
            "seed": p.seed,
            "trajectories": p.trajectories,
            "kinematics_summary": p.kinematics_summary,
            "legacy_diagnostics_manifest": p.legacy_diagnostics_manifest,
            "legacy_bulk_arrays": p.legacy_bulk_arrays,
            "gru_standard_certificate": _gru_standard_certificate_payload(
                run_spec,
                params,
                root=root,
            ),
        },
    )


def perturbation_response_bank_recipe(
    run_spec: EvaluationRunSpec,
    root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate a perturbation-response bank request."""

    _p, params = _validated_params(run_spec)
    payload = _perturbation_response_bank_payload(run_spec, params, root=root)
    return _result(
        run_spec,
        payload.params,
        product_role="perturbation_response_bank",
        state_payload=payload.state_payload,
        summary_metrics=payload.summary_metrics,
        metadata=payload.metadata,
    )


def feedback_ablation_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate intact-vs-ablated feedback rollout pairs."""

    p, params = _validated_params(run_spec)
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
            "ablation_masks": p.ablation_masks
            if p.ablation_masks is not None
            else p.ablation_mask_set,
            "base_task": p.base_task,
            "rollout_pairs": p.rollout_pairs,
        },
    )


def worst_case_epsilon_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate worst-case epsilon perturbation audits."""

    p, params = _validated_params(run_spec)
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
            "epsilon_budget_identity": (
                p.epsilon_budget_data_product_identity
                if p.epsilon_budget_data_product_identity is not None
                else p.epsilon_budget_identity
            ),
            "optimizer": p.optimizer,
            "audit_inputs": p.audit_inputs,
            "worst_case_rollouts": p.worst_case_rollouts,
        },
    )


def delayed_reach_bank_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate delayed-reach checkpoint-selection banks."""

    p, params = _validated_params(run_spec)
    _require_one_of(params, ("bank_spec",), recipe=run_spec.evaluation_type)
    return _result(
        run_spec,
        params,
        product_role="delayed_reach_eval_bank",
        state_payload={
            "bank_spec": p.bank_spec,
            "bank_tensors": p.bank_tensors,
            "selection_inputs": p.selection_inputs,
        },
    )


def _validated_params(run_spec: EvaluationRunSpec) -> tuple[BaseModel, dict[str, Any]]:
    kind = _RECIPE_PARAM_KINDS[run_spec.evaluation_type]
    result = accept_rlrmp_spec_payload(kind, run_spec.params)
    params = dict(result.payload)
    model_class = params_model_for(run_spec.evaluation_type)
    return model_class.model_validate(params), params


def _result(
    run_spec: EvaluationRunSpec,
    params: Mapping[str, Any],
    *,
    product_role: str,
    state_payload: Mapping[str, Any],
    summary_metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
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
        **dict(summary_metrics or {}),
    }
    return EvaluationRecipeResult(
        states=states,
        summary_metrics=summary_metrics,
        metadata={
            "rlrmp_evaluation_recipe": run_spec.evaluation_type,
            "states_schema": f"{run_spec.evaluation_type}.states.v1",
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
            **dict(metadata or {}),
        },
    )


class _PerturbationResponseBankPayload:
    """Normalized eval payload for the perturbation-response bank recipe."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any],
        state_payload: Mapping[str, Any],
        summary_metrics: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        self.params = dict(params)
        self.state_payload = dict(state_payload)
        self.summary_metrics = dict(summary_metrics)
        self.metadata = dict(metadata)


def _perturbation_response_bank_payload(
    run_spec: EvaluationRunSpec,
    params: Mapping[str, Any],
    *,
    root: Path,
) -> _PerturbationResponseBankPayload:
    legacy_payload_mode = params.get("legacy_payload_mode") is True
    response_tensors = params.get("response_tensors")
    if response_tensors is not None and not legacy_payload_mode:
        raise ValueError(
            "perturbation-response bank response_tensors params require "
            "legacy_payload_mode=true"
        )
    if legacy_payload_mode:
        return _legacy_perturbation_response_bank_payload(params)
    if response_tensors is not None:
        raise ValueError("perturbation-response bank legacy payload mode must be explicit")
    return _model_driven_perturbation_response_bank_payload(run_spec, params, root=root)


def _legacy_perturbation_response_bank_payload(
    params: Mapping[str, Any],
) -> _PerturbationResponseBankPayload:
    _require_one_of(
        params,
        ("checkpoint_bank_ref", "checkpoint_bank"),
        recipe=PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
    )
    bank = _normalized_perturbation_bank(params.get("perturbation_battery", {}))
    class_index_map = params.get("class_index_map")
    if class_index_map is None:
        class_index_map = _perturbation_class_index_map(bank)
    response_tensors = params.get("response_tensors")
    if response_tensors is None:
        raise ValueError(
            "perturbation-response bank legacy_payload_mode=true requires response_tensors"
        )
    return _PerturbationResponseBankPayload(
        params=params,
        state_payload={
            "production_mode": "legacy_payload",
            "checkpoint_bank": params.get("checkpoint_bank_ref", params.get("checkpoint_bank")),
            "perturbation_battery": bank,
            "alignment_mode": params.get("alignment_mode", "reach_locked"),
            "response_tensors": response_tensors,
            "class_index_map": class_index_map,
            "bank_status": params.get("bank_status", {}),
        },
        summary_metrics=_perturbation_bank_summary_metrics(bank, class_index_map),
        metadata={"legacy_payload_mode": True, "production_mode": "legacy_payload"},
    )


def _model_driven_perturbation_response_bank_payload(
    run_spec: EvaluationRunSpec,
    params: Mapping[str, Any],
    *,
    root: Path,
) -> _PerturbationResponseBankPayload:
    params_with_identity = _with_eval_consumed_calibration_identity(params)
    bank = _perturbation_bank_from_params(params_with_identity)
    class_index_map = _perturbation_class_index_map(bank)
    run_summaries = _evaluate_perturbation_bank_runs(
        run_spec,
        params_with_identity,
        bank=bank,
        root=root,
    )
    response_tensors = {
        "schema_id": "rlrmp.eval.perturbation_response_bank.response_tensors",
        "schema_version": "rlrmp.eval.perturbation_response_bank.response_tensors.v1",
        "runs": run_summaries,
    }
    bank_status = {
        run_id: dict(summary.get("status_counts", {}))
        for run_id, summary in run_summaries.items()
        if isinstance(summary, Mapping)
    }
    return _PerturbationResponseBankPayload(
        params=params_with_identity,
        state_payload={
            "production_mode": "model_driven",
            "checkpoint_bank": params_with_identity.get(
                "checkpoint_bank_ref",
                params_with_identity.get("checkpoint_bank"),
            ),
            "perturbation_battery": bank,
            "alignment_mode": params_with_identity.get("alignment_mode", "reach_locked"),
            "response_tensors": response_tensors,
            "class_index_map": class_index_map,
            "bank_status": bank_status,
        },
        summary_metrics={
            **_perturbation_bank_summary_metrics(bank, class_index_map),
            "evaluated_run_count": len(run_summaries),
        },
        metadata={"legacy_payload_mode": False, "production_mode": "model_driven"},
    )


def _with_eval_consumed_calibration_identity(params: Mapping[str, Any]) -> dict[str, Any]:
    if not _uses_open_loop_calibration(params):
        return dict(params)
    from rlrmp.runtime.training_run_specs import add_consumed_data_identity

    identity = _consumed_open_loop_calibration_identity()
    return add_consumed_data_identity(dict(params), **identity)


def _uses_open_loop_calibration(params: Mapping[str, Any]) -> bool:
    if params.get("consume_open_loop_calibration") is True:
        return True
    return str(params.get("bank_mode", params.get("mode", "raw"))) == "calibrated"


def _consumed_open_loop_calibration_identity() -> dict[str, str]:
    from rlrmp.data_products.calibration import consumed_calibration_identity

    return consumed_calibration_identity()


def _perturbation_bank_from_params(params: Mapping[str, Any]) -> dict[str, Any]:
    raw_bank = params.get("perturbation_battery", params.get("bank"))
    if raw_bank is None:
        raw_bank = _default_perturbation_bank(params)
    bank = _normalized_perturbation_bank(raw_bank)
    families = _requested_perturbation_families(params)
    if families is not None:
        family_set = set(families)
        perturbations = [
            row for row in bank["perturbations"] if str(row.get("family")) in family_set
        ]
        missing = sorted(family_set.difference({str(row.get("family")) for row in perturbations}))
        if missing:
            available = sorted({str(row.get("family")) for row in bank["perturbations"]})
            raise ValueError(
                "perturbation-response bank requested unavailable families "
                f"{missing}; available families: {available}"
            )
        bank["perturbations"] = perturbations
        bank["selected_families"] = list(families)
    perturbation_ids = _requested_perturbation_ids(params)
    if perturbation_ids is not None:
        id_set = set(perturbation_ids)
        perturbations = [
            row for row in bank["perturbations"] if str(row.get("perturbation_id")) in id_set
        ]
        missing = sorted(
            id_set.difference({str(row.get("perturbation_id")) for row in perturbations})
        )
        if missing:
            available = sorted(str(row.get("perturbation_id")) for row in bank["perturbations"])
            raise ValueError(
                "perturbation-response bank requested unavailable perturbation_ids "
                f"{missing}; available perturbation_ids: {available}"
            )
        bank["perturbations"] = perturbations
        bank["selected_perturbation_ids"] = list(perturbation_ids)
    if not bank["perturbations"]:
        raise ValueError("perturbation-response bank produced no perturbation rows")
    return bank


def _default_perturbation_bank(params: Mapping[str, Any]) -> dict[str, Any]:
    from rlrmp.analysis.pipelines.gru_perturbation_bank import default_cs_perturbation_bank

    feedback_scale_manifest_path = params.get("feedback_scale_manifest_path")
    return default_cs_perturbation_bank(
        mode=str(params.get("bank_mode", params.get("mode", "raw"))),
        calibration_level=params.get("calibration_level"),
        calibration_reach=params.get("calibration_reach"),
        feedback_scale_manifest=params.get("feedback_scale_manifest"),
        feedback_scale_manifest_path=(
            None if feedback_scale_manifest_path is None else Path(str(feedback_scale_manifest_path))
        ),
    )


def _normalized_perturbation_bank(raw_bank: Any) -> dict[str, Any]:
    from rlrmp.analysis.perturbation_rows import PerturbationSpec

    if not isinstance(raw_bank, Mapping):
        raise TypeError("perturbation_battery must be a mapping")
    bank = deepcopy(dict(raw_bank))
    rows = bank.get("perturbations", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("perturbation_battery.perturbations must be a sequence")
    validated_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("perturbation rows must be mappings")
        spec = PerturbationSpec.from_mapping(row)
        spec.validate()
        validated_rows.append(spec.to_json())
    bank["perturbations"] = validated_rows
    return bank


def _requested_perturbation_families(params: Mapping[str, Any]) -> tuple[str, ...] | None:
    for key in ("class_set", "families", "family_set", "perturbation_families"):
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(str(item) for item in value)
        raise TypeError(f"{key} must be a string or sequence of strings")
    return None


def _requested_perturbation_ids(params: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = params.get("perturbation_ids")
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    raise TypeError("perturbation_ids must be a string or sequence of strings")


def _perturbation_class_index_map(bank: Mapping[str, Any]) -> dict[str, Any]:
    rows = bank.get("perturbations", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("perturbation_battery.perturbations must be a sequence")
    families: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("perturbation rows must be mappings")
        family = str(row.get("family"))
        perturbation_id = str(row.get("perturbation_id"))
        entry = families.setdefault(
            family,
            {
                "family": family,
                "perturbation_ids": [],
                "row_indices": [],
                "tensor_slices": {"axis": "perturbation", "indices": []},
                "calibration_provenance": {},
            },
        )
        entry["perturbation_ids"].append(perturbation_id)
        entry["row_indices"].append(index)
        entry["tensor_slices"]["indices"].append(index)
        provenance = _row_calibration_provenance(row)
        if provenance:
            entry["calibration_provenance"][perturbation_id] = provenance
    for entry in families.values():
        indices = list(entry["row_indices"])
        contiguous = indices == list(range(indices[0], indices[-1] + 1))
        entry["tensor_slices"]["contiguous"] = contiguous
        entry["tensor_slices"]["start"] = indices[0] if contiguous else None
        entry["tensor_slices"]["stop"] = indices[-1] + 1 if contiguous else None
    return families


def _row_calibration_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    explicit = row.get("calibration_provenance")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    known_keys = {
        "level_name",
        "level_fraction_of_reach",
        "reach_label",
        "reach_length_m",
        "open_loop_peak_dx_per_unit_m",
        "open_loop_auc_dx_per_unit_m_s",
        "target_open_loop_peak_dx_m",
        "target_open_loop_auc_dx_m_s",
        "native_unit_rule",
        "reference_position_scale_m",
        "nominal_peak_speed_m_s",
        "reference_force_filter_scale_N",
        "controller_feedback_scale",
        "feedback_quantity",
        "feedback_payload_index",
        "force_filter_feedback_only",
        "false_feedback_probe",
    }
    return {
        key: value
        for key, value in row.items()
        if key.startswith("calibration_") or key in known_keys
    }


def _perturbation_bank_summary_metrics(
    bank: Mapping[str, Any],
    class_index_map: Mapping[str, Any],
) -> dict[str, int]:
    perturbations = bank.get("perturbations", [])
    return {
        "perturbation_family_count": len(class_index_map),
        "perturbation_row_count": (
            len(perturbations)
            if isinstance(perturbations, Sequence) and not isinstance(perturbations, (str, bytes))
            else 0
        ),
    }


def _evaluate_perturbation_bank_runs(
    run_spec: EvaluationRunSpec,
    params: Mapping[str, Any],
    *,
    bank: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    source_experiment = _source_experiment(params)
    run_ids = tuple(str(run_id) for run_id in params.get("run_ids", run_spec.training_run_ids))
    labels = params.get("labels")
    if labels is not None:
        labels = tuple(str(label) for label in labels)
    runs = _resolve_perturbation_run_inputs(
        experiment=source_experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=_repo_root_for_eval(params, root=root),
    )
    bulk_dir = _perturbation_eval_bulk_dir(params, root=root)
    write_bulk_arrays = bool(params.get("write_bulk_arrays", False))
    if write_bulk_arrays:
        bulk_dir.mkdir(parents=True, exist_ok=True)
    return {
        run.run_id: _evaluate_single_perturbation_bank_run(
            run,
            source_experiment=source_experiment,
            bank=bank,
            n_rollout_trials=int(params.get("n_rollout_trials", 8)),
            write_bulk_arrays=write_bulk_arrays,
            bulk_dir=bulk_dir,
            extlqg_physical_dim=int(params.get("extlqg_physical_dim", 8)),
            preferred_checkpoint_manifest_path=_optional_path(
                params.get("preferred_checkpoint_manifest_path")
            ),
            checkpoint_selection_mode=str(params.get("checkpoint_selection_mode", "sparse_history")),
            repo_root=_repo_root_for_eval(params, root=root),
        )
        for run in runs
    }


def _source_experiment(params: Mapping[str, Any]) -> str:
    source = params.get("source_experiment", params.get("experiment"))
    if source is None:
        raise ValueError(
            "model-driven perturbation-response bank params must include source_experiment"
        )
    return str(source)


def _repo_root_for_eval(params: Mapping[str, Any], *, root: Path) -> Path:
    repo_root = params.get("repo_root")
    if repo_root is not None:
        return Path(str(repo_root))
    from rlrmp.paths import REPO_ROOT

    return REPO_ROOT if REPO_ROOT is not None else root


def _perturbation_eval_bulk_dir(params: Mapping[str, Any], *, root: Path) -> Path:
    bulk_dir = params.get("bulk_dir")
    if bulk_dir is not None:
        return Path(str(bulk_dir))
    return root / "cache" / "perturbation_response_bank"


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(str(value))


def _resolve_perturbation_run_inputs(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    repo_root: Path,
) -> Sequence[Any]:
    from rlrmp.analysis.pipelines.gru_pilot_figures import resolve_run_inputs

    return resolve_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )


def _evaluate_single_perturbation_bank_run(
    run: Any,
    *,
    source_experiment: str,
    bank: Mapping[str, Any],
    n_rollout_trials: int,
    write_bulk_arrays: bool,
    bulk_dir: Path,
    extlqg_physical_dim: int,
    preferred_checkpoint_manifest_path: Path | None,
    checkpoint_selection_mode: str,
    repo_root: Path,
) -> dict[str, Any]:
    from rlrmp.analysis.pipelines.gru_perturbation_bank import evaluate_run_perturbation_bank

    return evaluate_run_perturbation_bank(
        run,
        source_experiment=source_experiment,
        bank=bank,
        n_rollout_trials=n_rollout_trials,
        write_bulk_arrays=write_bulk_arrays,
        bulk_dir=bulk_dir,
        extlqg_physical_dim=extlqg_physical_dim,  # type: ignore[arg-type]
        preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=checkpoint_selection_mode,  # type: ignore[arg-type]
        repo_root=repo_root,
    )


def _parent_refs(run_spec: EvaluationRunSpec) -> list[dict[str, Any]]:
    refs: list[ParentRef] = list(run_spec.inputs)
    for run_id in run_spec.training_run_ids:
        ref = ParentRef(kind="TrainingRunManifest", id=run_id, role="training_run")
        if all(existing.id != ref.id or existing.kind != ref.kind for existing in refs):
            refs.append(ref)
    return [ref.model_dump(mode="json", exclude_none=True) for ref in refs]


def _gru_standard_certificate_payload(
    run_spec: EvaluationRunSpec,
    params: Mapping[str, Any],
    *,
    root: Path,
) -> Mapping[str, Any]:
    request = params.get("gru_standard_certificate")
    if not isinstance(request, Mapping):
        return {}
    mode = request.get("mode", "precomputed")
    if mode == "precomputed":
        return dict(request)
    if mode != "evaluate_clean_actions":
        raise ValueError(
            "gru_standard_certificate.mode must be 'precomputed' or "
            f"'evaluate_clean_actions', got {mode!r}"
        )

    from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
        evaluate_gru_clean_actions,
    )
    from rlrmp.paths import REPO_ROOT
    from rlrmp.runtime.run_specs import resolve_run_record

    repo_root = Path(request.get("repo_root", root if root is not None else REPO_ROOT))
    experiment = str(request["experiment"])
    run_ids = tuple(str(run_id) for run_id in request.get("run_ids", run_spec.training_run_ids))
    runs = {}
    for run_id in run_ids:
        run_record = resolve_run_record(experiment, run_id, repo_root=repo_root)
        actions, response_maps, metadata = evaluate_gru_clean_actions(
            run_id,
            run_spec=run_record,
            experiment=experiment,
            use_validation_selected_checkpoints=bool(
                request.get("use_validation_selected_checkpoints", False)
            ),
            preferred_checkpoint_manifest_path=(
                None
                if request.get("preferred_checkpoint_manifest_path") is None
                else Path(str(request["preferred_checkpoint_manifest_path"]))
            ),
            repo_root=repo_root,
        )
        covariance = metadata.pop("_observation_history_covariance_array", None)
        runs[run_id] = {
            "run_spec": run_record,
            "candidate_actions": actions,
            "candidate_observation_to_action_map": response_maps,
            "observation_history_covariance": covariance,
            "evaluation_metadata": metadata,
        }
    return {
        "mode": mode,
        "experiment": experiment,
        "run_ids": list(run_ids),
        "runs": runs,
    }


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
    "CenterOutEnsembleEvalParams",
    "DELAYED_REACH_BANK_EVALUATION_TYPE",
    "DelayedReachBankEvalParams",
    "FEEDBACK_ABLATION_EVALUATION_TYPE",
    "FeedbackAblationEvalParams",
    "PerturbationResponseBankEvalParams",
    "PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE",
    "WorstCaseEpsilonEvalParams",
    "WORST_CASE_EPSILON_EVALUATION_TYPE",
    "center_out_ensemble_recipe",
    "delayed_reach_bank_recipe",
    "feedback_ablation_recipe",
    "perturbation_response_bank_recipe",
    "register_rlrmp_evaluation_recipes",
    "worst_case_epsilon_recipe",
]
