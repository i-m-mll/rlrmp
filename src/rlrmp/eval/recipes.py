"""Manifest-canonical rlrmp evaluation recipes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    ParentRef,
)
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.runtime.params_models import params_model_for, register_params_model
from rlrmp.eval.feedback_ablation import (
    evaluate_feedback_ablation_runs,
    evaluate_projected_feedback_ablation_run,
)
from rlrmp.eval.model_slots import ModelSlotProjection
from rlrmp.eval.broad_epsilon import evaluate_broad_epsilon_runs
from rlrmp.eval.evaluation_diagnostics import (
    DEFAULT_N_ROLLOUT_TRIALS,
    evaluate_gru_diagnostics_runs,
)
from rlrmp.runtime.spec_migrations import (
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    DELAYED_REACH_BANK_EVAL_PARAMS_KIND,
    FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    GRU_DIAGNOSTICS_EVAL_PARAMS_KIND,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
    WORST_CASE_EPSILON_EVAL_PARAMS_KIND,
    accept_rlrmp_spec_payload,
)

CENTER_OUT_ENSEMBLE_EVALUATION_TYPE = "rlrmp.eval.center_out_ensemble"
GRU_DIAGNOSTICS_EVALUATION_TYPE = "rlrmp.eval.gru_diagnostics"
PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE = "rlrmp.eval.perturbation_response_bank"
FEEDBACK_ABLATION_EVALUATION_TYPE = "rlrmp.eval.feedback_ablation"
WORST_CASE_EPSILON_EVALUATION_TYPE = "rlrmp.eval.worst_case_epsilon"
BROAD_EPSILON_EVALUATION_TYPE = "rlrmp.eval.broad_epsilon"
DELAYED_REACH_BANK_EVALUATION_TYPE = "rlrmp.eval.delayed_reach_bank"
DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_ID = "rlrmp.figure_data.delayed_velocity_profiles"
DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_VERSION = "rlrmp.figure_data.delayed_velocity_profiles.v1"

_RECIPE_PARAM_KINDS = {
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE: CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    GRU_DIAGNOSTICS_EVALUATION_TYPE: GRU_DIAGNOSTICS_EVAL_PARAMS_KIND,
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
    consumed_data_identities: list[dict[str, Any]] | dict[str, Any] = Field(default_factory=list)


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
    policy_diagnostics: dict[str, Any] | None = None
    recurrent_jacobians: dict[str, Any] | None = None


class GRUDiagnosticsEvalParams(_StrictParamsModel):
    """Params for selected-checkpoint GRU diagnostics from cached states."""

    source_experiment: str = ""
    run_ids: list[str] = Field(default_factory=list)
    labels: list[str] | None = None
    n_rollout_trials: int = Field(DEFAULT_N_ROLLOUT_TRIALS, ge=1)
    preferred_checkpoint_manifest_path: str | None = None
    jacobian_timepoints: list[Literal["first", "peak_forward_velocity", "terminal"]] = Field(
        default_factory=lambda: ["first", "peak_forward_velocity", "terminal"]
    )
    repo_root: str | None = None


class PerturbationResponseBankEvalParams(_StrictParamsModel):
    """Params for perturbation-response bank evaluation."""

    checkpoint_bank_ref: Any | None = None
    checkpoint_bank: Any | None = None
    bank_params: dict[str, Any] | None = None
    perturbation_battery: Any | None = None
    bank: Any | None = None
    alignment_mode: Literal["reach_locked"] = "reach_locked"
    response_tensors: Any | None = None
    class_index_map: Any | None = None
    bank_status: dict[str, Any] = Field(default_factory=dict)
    bundle_contract: dict[str, Any] = Field(default_factory=dict)
    states_custody: Literal["cache", "durable"] | None = None
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
    n_rollout_trials: int = Field(8, ge=1)
    extlqg_physical_dim: Literal[6, 8] = 8
    preferred_checkpoint_manifest_path: str | None = None
    checkpoint_selection_mode: Literal["sparse_history", "fixed_bank_manifest"] = "sparse_history"
    checkpoint_custody_root: str | None = None


class FeedbackAblationEvalParams(_StrictParamsModel):
    """Params for feedback-ablation evaluation."""

    source_experiment: str = ""
    run_ids: list[str] = Field(default_factory=list)
    labels: list[str] | None = None
    scope: str = "feedback_ablation"
    n_rollout_trials: int = Field(4, ge=1)
    include_checkpoint_rescore: bool = True
    bank_mode: Literal["raw", "calibrated"] = "raw"
    calibration_level: str | list[str] | None = None
    calibration_reach: str | float | None = None
    feedback_selection_level: str = "small"
    feedback_scale_manifest_path: str | None = None
    preferred_checkpoint_manifest_path: str | None = None
    repo_root: str | None = None
    bank: dict[str, Any] | None = None
    evaluation_bins: dict[str, str | None] | None = None
    checkpoint_custody_root: str | None = None


class WorstCaseEpsilonEvalParams(_StrictParamsModel):
    """Params for worst-case epsilon evaluation."""

    run_ids: list[str] = Field(default_factory=list)
    budget_level: str | None = None
    budget_scale: float | None = Field(default=None, gt=0)
    n_steps: int = Field(default=12, ge=0)
    n_restarts: int = Field(default=3, ge=0)
    step_size: float | None = Field(default=None, gt=0)
    backend: Literal["serial", "staged"] = "serial"
    epsilon_budget_data_product_identity: Any | None = None
    epsilon_budget_identity: Any | None = None
    optimizer: dict[str, Any] = Field(default_factory=dict)
    audit_inputs: dict[str, Any] = Field(default_factory=dict)
    worst_case_rollouts: list[Any] = Field(default_factory=list)


class BroadEpsilonEvalParams(_StrictParamsModel):
    """Params for paired active/zero broad-epsilon evaluation."""

    source_experiment: str = ""
    run_ids: list[str] = Field(default_factory=list)
    labels: list[str] | None = None
    n_rollout_trials: int = Field(default=8, ge=1)
    max_gradient_replicates: int = Field(default=1, ge=0)
    checkpoint_policy: Literal["validation_selected_per_replicate"] = (
        "validation_selected_per_replicate"
    )
    paired_rollouts: list[Any] = Field(default_factory=list)


class DelayedReachBankEvalParams(_StrictParamsModel):
    """Params for delayed-reach bank evaluation."""

    bank_spec: dict[str, Any] = Field(default_factory=dict)
    bank_tensors: dict[str, Any] = Field(default_factory=dict)
    selection_inputs: dict[str, Any] = Field(default_factory=dict)
    profile_payloads: dict[str, Any] = Field(default_factory=dict)


_PARAMS_MODEL_BY_RECIPE = {
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE: CenterOutEnsembleEvalParams,
    GRU_DIAGNOSTICS_EVALUATION_TYPE: GRUDiagnosticsEvalParams,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE: PerturbationResponseBankEvalParams,
    FEEDBACK_ABLATION_EVALUATION_TYPE: FeedbackAblationEvalParams,
    WORST_CASE_EPSILON_EVALUATION_TYPE: WorstCaseEpsilonEvalParams,
    BROAD_EPSILON_EVALUATION_TYPE: BroadEpsilonEvalParams,
    DELAYED_REACH_BANK_EVALUATION_TYPE: DelayedReachBankEvalParams,
}


def register_rlrmp_evaluation_recipes(*, replace: bool = True) -> None:
    """Register rlrmp's manifest-canonical evaluation recipes."""

    from rlrmp.analysis.standard_certificate import (
        register_standard_certificate_component_provider,
    )
    from rlrmp.eval.linear_recurrent_certificate import (
        LINEAR_RECURRENT_AUGMENTED_PROVIDER,
        linear_recurrent_augmented_component_kwargs,
    )

    register_standard_certificate_component_provider(
        LINEAR_RECURRENT_AUGMENTED_PROVIDER,
        linear_recurrent_augmented_component_kwargs,
        replace=replace,
    )

    for recipe_name, model_class in _PARAMS_MODEL_BY_RECIPE.items():
        register_params_model(recipe_name, model_class, replace=replace)
    from rlrmp.eval.perturbation_bank import (
        PERTURBATION_BANK_PARAMS_TYPE,
        PerturbationBankParams,
    )

    register_params_model(PERTURBATION_BANK_PARAMS_TYPE, PerturbationBankParams, replace=replace)
    register_evaluation_recipe(
        CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
        center_out_ensemble_recipe,
        replace=replace,
    )
    register_evaluation_recipe(
        GRU_DIAGNOSTICS_EVALUATION_TYPE,
        gru_diagnostics_recipe,
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
        BROAD_EPSILON_EVALUATION_TYPE,
        broad_epsilon_recipe,
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
            "policy_diagnostics": p.policy_diagnostics,
            "recurrent_jacobians": p.recurrent_jacobians,
        },
    )


def gru_diagnostics_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate and cache behavior, feedback, gate, and Jacobian diagnostics."""

    p, params = _validated_params(run_spec)
    if not p.source_experiment or not p.run_ids:
        raise ValueError("GRU diagnostics require source_experiment and non-empty run_ids")
    repo_root = Path(p.repo_root).expanduser() if p.repo_root else Path.cwd()
    payload = evaluate_gru_diagnostics_runs(params, repo_root=repo_root)
    return _result(
        run_spec,
        params,
        product_role="gru_diagnostic_states",
        state_payload=payload,
        summary_metrics={
            "gru_diagnostic_run_count": len(payload["runs"]),
            "gru_diagnostic_rollout_count": sum(
                int(run["n_replicates"]) * int(run["n_rollout_trials_per_replicate"])
                for run in payload["runs"].values()
            ),
        },
        metadata={"execution_owner": "registered_evaluation_recipe"},
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
    root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate intact-vs-ablated feedback rollout pairs."""

    p, params = _validated_params(run_spec)
    has_legacy_selector = bool(params.get("source_experiment")) and bool(params.get("run_ids"))
    if run_spec.inputs and has_legacy_selector:
        raise ValueError(
            "feedback-ablation evaluation cannot mix exact native parents with legacy "
            "source_experiment/run_ids selectors"
        )
    if run_spec.inputs:
        projection = _native_model_projection(
            run_spec,
            manifest_root=root,
            checkpoint_custody_root=params.get("checkpoint_custody_root"),
        )
        payload = evaluate_projected_feedback_ablation_run(
            projection,
            p.model_dump(mode="python"),
        )
    elif not p.source_experiment or not p.run_ids:
        raise ValueError(
            "feedback-ablation evaluation requires source_experiment and non-empty run_ids"
        )
    else:
        execution_params = p.model_dump(mode="python")
        repo_root_value = execution_params.get("repo_root")
        repo_root = Path(str(repo_root_value)).expanduser() if repo_root_value else Path.cwd()
        payload = evaluate_feedback_ablation_runs(
            execution_params,
            repo_root=repo_root,
        )
    return _result(
        run_spec,
        params,
        product_role="feedback_ablation_rollouts",
        state_payload=payload,
        summary_metrics={
            "feedback_ablation_run_count": len(payload["runs"]),
            "feedback_ablation_row_count": sum(
                len(run.get("ablations", ()))
                for run in payload["runs"].values()
                if isinstance(run, Mapping)
            ),
        },
        metadata={"execution_owner": "registered_evaluation_recipe"},
    )


def worst_case_epsilon_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate worst-case epsilon perturbation audits."""

    p, params = _validated_params(run_spec)
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
            "run_ids": p.run_ids,
            "budget_level": p.budget_level,
            "budget_scale": p.budget_scale,
            "n_steps": p.n_steps,
            "n_restarts": p.n_restarts,
            "step_size": p.step_size,
            "backend": p.backend,
            "audit_inputs": p.audit_inputs,
            "worst_case_rollouts": p.worst_case_rollouts,
        },
    )


def broad_epsilon_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Cache paired active/zero rollout and differentiable-gradient records."""

    p, params = _validated_params(run_spec)
    payload = (
        {
            "source_experiment": p.source_experiment,
            "checkpoint_policy": p.checkpoint_policy,
            "rows": p.paired_rollouts,
        }
        if p.paired_rollouts
        else evaluate_broad_epsilon_runs(params)
    )
    return _result(
        run_spec,
        params,
        product_role="broad_epsilon_paired_rollouts",
        state_payload={
            **payload,
            "run_ids": p.run_ids,
            "n_rollout_trials": p.n_rollout_trials,
            "max_gradient_replicates": p.max_gradient_replicates,
        },
        metadata={"execution_owner": "registered_evaluation_recipe"},
    )


def delayed_reach_bank_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate delayed-reach checkpoint-selection banks."""

    p, params = _validated_params(run_spec)
    _require_one_of(params, ("bank_spec",), recipe=run_spec.evaluation_type)
    figure_payload = delayed_velocity_profile_payload(p.profile_payloads)
    return _result(
        run_spec,
        params,
        product_role="delayed_reach_eval_bank",
        state_payload={
            "bank_spec": p.bank_spec,
            "bank_tensors": p.bank_tensors,
            "selection_inputs": p.selection_inputs,
            "profile_payloads": p.profile_payloads,
        },
        metadata={"figure_payload": figure_payload} if figure_payload is not None else None,
    )


def delayed_velocity_profile_payload(
    profile_payloads: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Normalize delayed-bank profiles for the native profile template.

    The registered evaluation recipe owns the numerical profile payload. Figure
    intent owns only intrinsic panel/axis semantics and binds its facets to this
    manifest metadata. Each run is one facet; no-catch and catch banks remain
    separate labeled series within that facet.
    """

    if not profile_payloads:
        return None
    banks = profile_payloads.get("banks", profile_payloads)
    if not isinstance(banks, Mapping):
        raise ValueError("delayed profile payloads must map bank kind to profiles")

    facets: dict[str, dict[str, Any]] = {}
    for bank_kind in ("no_catch", "catch"):
        profiles = banks.get(bank_kind, ())
        if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
            raise ValueError(f"delayed {bank_kind} profiles must be a sequence")
        for raw_profile in profiles:
            if not isinstance(raw_profile, Mapping):
                raise ValueError(f"delayed {bank_kind} profile entries must be mappings")
            experiment = str(raw_profile.get("experiment", ""))
            run_id = str(raw_profile.get("run_id", ""))
            label = str(raw_profile.get("label", raw_profile.get("run_label", run_id)))
            if not run_id or not label:
                raise ValueError(f"delayed {bank_kind} profile requires run_id and label")
            facet_id = f"{experiment}/{run_id}" if experiment else run_id
            facet = facets.setdefault(
                facet_id,
                {
                    "run_id": run_id,
                    "experiment": experiment or None,
                    "display_name": label,
                    "forward_velocity": {"series": []},
                    "forward_velocity_by_replicate": {"series": []},
                },
            )
            profile = _delayed_profile_band(raw_profile)
            facet["forward_velocity"]["series"].append(
                {
                    "label": f"{label} ({bank_kind.replace('_', ' ')})",
                    "color": raw_profile.get("color"),
                    "profile": profile,
                    "bank_kind": bank_kind,
                    "alignment": raw_profile.get("alignment", {}),
                    "evaluation_bank": raw_profile.get("evaluation_bank", {}),
                }
            )
            for replicate, replicate_profile in enumerate(_delayed_replicate_bands(raw_profile)):
                facet["forward_velocity_by_replicate"]["series"].append(
                    {
                        "label": (
                            f"{label} ({bank_kind.replace('_', ' ')}, replicate {replicate + 1})"
                        ),
                        "color": raw_profile.get("color"),
                        "profile": replicate_profile,
                        "bank_kind": bank_kind,
                        "replicate": replicate,
                    }
                )

    if not facets:
        raise ValueError("delayed profile payloads contain no profiles")
    return {
        "schema_id": DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_ID,
        "schema_version": DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_VERSION,
        "facets": {"condition": facets},
    }


def _delayed_profile_band(profile: Mapping[str, Any]) -> dict[str, Any]:
    time = list(profile.get("time", profile.get("time_s", ())))
    mean = list(profile.get("mean", ()))
    if not time or len(time) != len(mean):
        raise ValueError("delayed profile time and mean must be non-empty and aligned")
    upper = profile.get("upper")
    lower = profile.get("lower")
    if upper is None or lower is None:
        std = list(profile.get("std", ()))
        if len(std) != len(mean):
            raise ValueError("delayed profile requires aligned upper/lower or std")
        upper = [float(value) + float(delta) for value, delta in zip(mean, std, strict=True)]
        lower = [float(value) - float(delta) for value, delta in zip(mean, std, strict=True)]
    return {
        "time": time,
        "mean": mean,
        "upper": list(upper),
        "lower": list(lower),
    }


def _delayed_replicate_bands(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    means = profile.get("replicate_mean", ())
    stds = profile.get("replicate_std", ())
    if not means and not stds:
        return []
    if not isinstance(means, Sequence) or not isinstance(stds, Sequence):
        raise ValueError("delayed replicate profile means/stds must be sequences")
    if len(means) != len(stds):
        raise ValueError("delayed replicate profile means/stds must have equal length")
    time = list(profile.get("time", profile.get("time_s", ())))
    bands = []
    for mean, std in zip(means, stds, strict=True):
        bands.append(_delayed_profile_band({"time": time, "mean": mean, "std": std}))
    return bands


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
            "perturbation-response bank response_tensors params require legacy_payload_mode=true"
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
    raw_bank = params.get("perturbation_battery")
    if raw_bank is None:
        raw_bank = _default_perturbation_bank(params)
    bank = _normalized_perturbation_bank(raw_bank)
    bank_params = _canonical_perturbation_bank_params(params)
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
            "bank_params": bank_params,
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
    bank_params = _canonical_perturbation_bank_params(params_with_identity)
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
            "bank_params": bank_params,
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
    from rlrmp.data_products.calibration import (
        CALIBRATION_PRODUCT_ROLE,
        CALIBRATION_PRODUCT_SCHEMA_VERSION,
        load_open_loop_calibration,
    )
    from rlrmp.data_products.envelope import consumed_identity_from_loader
    from rlrmp.runtime.training_run_specs import add_consumed_data_identity

    identity = consumed_identity_from_loader(
        load_product=load_open_loop_calibration,
        role=CALIBRATION_PRODUCT_ROLE,
        schema=CALIBRATION_PRODUCT_SCHEMA_VERSION,
    )
    return add_consumed_data_identity(dict(params), **identity)


def _uses_open_loop_calibration(params: Mapping[str, Any]) -> bool:
    if params.get("consume_open_loop_calibration") is True:
        return True
    return str(params.get("bank_mode", params.get("mode", "raw"))) == "calibrated"


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
    from rlrmp.eval.perturbation_bank import (
        PerturbationBankParams,
        expand_perturbation_bank,
    )

    return expand_perturbation_bank(
        PerturbationBankParams.model_validate(_canonical_perturbation_bank_params(params)),
    )


def _canonical_perturbation_bank_params(params: Mapping[str, Any]) -> dict[str, Any]:
    from rlrmp.eval.perturbation_bank import PerturbationBankParams

    bank_params = dict(params.get("bank_params") or {})
    bank_params.setdefault("mode", params.get("bank_mode", params.get("mode", "raw")))
    for key in (
        "calibration_level",
        "calibration_reach",
        "feedback_scale_manifest",
        "feedback_scale_manifest_path",
    ):
        if params.get(key) is not None:
            bank_params.setdefault(key, params[key])
    return PerturbationBankParams.model_validate(bank_params).model_dump(
        mode="json",
        exclude_none=True,
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
    legacy_run_ids = params.get("run_ids")
    has_legacy_selector = bool(params.get("source_experiment")) and bool(legacy_run_ids)
    if run_spec.inputs and has_legacy_selector:
        raise ValueError(
            "perturbation-response evaluation cannot mix exact native parents with legacy "
            "source_experiment/run_ids selectors"
        )
    if run_spec.inputs:
        projection = _native_model_projection(
            run_spec,
            manifest_root=root,
            checkpoint_custody_root=params.get("checkpoint_custody_root"),
        )
        labels = params.get("labels")
        label = (
            str(labels[0])
            if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)) and labels
            else projection.provenance.run_id
        )
        run = SimpleNamespace(label=label)
        return {
            projection.provenance.training_manifest_id: _evaluate_single_perturbation_bank_run(
                run,
                source_experiment=None,
                bank=bank,
                n_rollout_trials=int(params.get("n_rollout_trials", 8)),
                extlqg_physical_dim=int(params.get("extlqg_physical_dim", 8)),
                preferred_checkpoint_manifest_path=None,
                checkpoint_selection_mode="sparse_history",
                repo_root=root,
                model_projection=projection,
            )
        }
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
    return {
        run.run_id: _evaluate_single_perturbation_bank_run(
            run,
            source_experiment=source_experiment,
            bank=bank,
            n_rollout_trials=int(params.get("n_rollout_trials", 8)),
            extlqg_physical_dim=int(params.get("extlqg_physical_dim", 8)),
            preferred_checkpoint_manifest_path=_optional_path(
                params.get("preferred_checkpoint_manifest_path")
            ),
            checkpoint_selection_mode=str(
                params.get("checkpoint_selection_mode", "sparse_history")
            ),
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


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(str(value))


def _resolve_perturbation_run_inputs(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    repo_root: Path,
) -> Sequence[Any]:
    from rlrmp.eval.trial_inputs import resolve_evaluation_run_inputs

    return resolve_evaluation_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )


def _evaluate_single_perturbation_bank_run(
    run: Any,
    *,
    source_experiment: str | None,
    bank: Mapping[str, Any],
    n_rollout_trials: int,
    extlqg_physical_dim: int,
    preferred_checkpoint_manifest_path: Path | None,
    checkpoint_selection_mode: str,
    repo_root: Path,
    model_projection: ModelSlotProjection | None = None,
) -> dict[str, Any]:
    from rlrmp.eval.perturbation_bank import evaluate_run_perturbation_bank

    return evaluate_run_perturbation_bank(
        run,
        source_experiment=source_experiment,
        bank=bank,
        n_rollout_trials=n_rollout_trials,
        extlqg_physical_dim=extlqg_physical_dim,  # type: ignore[arg-type]
        preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=checkpoint_selection_mode,  # type: ignore[arg-type]
        repo_root=repo_root,
        model_projection=model_projection,
    )


def _native_model_projection(
    run_spec: EvaluationRunSpec,
    *,
    manifest_root: Path,
    checkpoint_custody_root: str | Path | None,
) -> ModelSlotProjection:
    """Resolve and project the one exact native training parent for a recipe."""

    try:
        from feedbax.analysis import resolve_evaluation_inputs
    except ImportError as exc:
        raise RuntimeError(
            "native model-driven evaluation requires Feedbax resolved evaluation inputs"
        ) from exc
    from rlrmp.eval.model_slots import _project_training_model_slot_from_custody_root

    if (
        not isinstance(checkpoint_custody_root, (str, Path))
        or not str(checkpoint_custody_root).strip()
    ):
        raise ValueError("native model-driven evaluation requires explicit checkpoint_custody_root")
    checkpoint_root = Path(checkpoint_custody_root).expanduser()
    if not checkpoint_root.is_absolute():
        raise ValueError("native model-driven evaluation requires absolute checkpoint_custody_root")
    resolved = resolve_evaluation_inputs(run_spec, manifest_root=manifest_root)
    if len(resolved) != 1:
        raise ValueError("native model-driven evaluation requires exactly one training parent")
    return _project_training_model_slot_from_custody_root(
        resolved[0],
        checkpoint_root=checkpoint_root,
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

    from rlrmp.analysis.gru_standard_certificate import (
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
    "BROAD_EPSILON_EVALUATION_TYPE",
    "BroadEpsilonEvalParams",
    "CENTER_OUT_ENSEMBLE_EVALUATION_TYPE",
    "CenterOutEnsembleEvalParams",
    "DELAYED_REACH_BANK_EVALUATION_TYPE",
    "DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_ID",
    "DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_VERSION",
    "DelayedReachBankEvalParams",
    "FEEDBACK_ABLATION_EVALUATION_TYPE",
    "FeedbackAblationEvalParams",
    "GRU_DIAGNOSTICS_EVALUATION_TYPE",
    "GRUDiagnosticsEvalParams",
    "PerturbationResponseBankEvalParams",
    "PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE",
    "WorstCaseEpsilonEvalParams",
    "WORST_CASE_EPSILON_EVALUATION_TYPE",
    "center_out_ensemble_recipe",
    "broad_epsilon_recipe",
    "delayed_reach_bank_recipe",
    "delayed_velocity_profile_payload",
    "feedback_ablation_recipe",
    "gru_diagnostics_recipe",
    "perturbation_response_bank_recipe",
    "register_rlrmp_evaluation_recipes",
    "worst_case_epsilon_recipe",
]
