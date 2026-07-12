"""Native C&S supervised training orchestration and chunk execution."""
# ruff: noqa: F401

from __future__ import annotations

from rlrmp.train.run_spec_authoring import (
    _config_default,
    _policy_adversary_training_enabled,
    _training_diagnostics_metadata,
    write_run_spec,
)
from rlrmp.train.config_materialization import (
    DEFAULT_STOCHASTIC_PRESET,
    _apply_smoke_overrides,
    _training_diagnostics_enabled,
)
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    TrainingMethodRegistry,
    TrainingRunSpec,
)
from feedbax.objectives.loss import AbstractLoss
from feedbax.objectives.service import LossService, LoweredObjective
from feedbax.objectives.spec import ObjectiveExecutionRequirements
from feedbax.orchestration.events import RunEventEmitter
from feedbax.training.executor import execute_training_run_spec
from feedbax.training.checkpoint_custody import (
    load_latest_checkpoint as load_feedbax_checkpoint,
)
from jax_cookbook.tree import filter_spec_leaves
from rlrmp.model.feedbax_graph import (
    build_runtime_rlrmp_feedbax_graph_bundle,
)
from rlrmp.loss import (
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
)
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.run_specs import run_spec_sidecar_dir, validate_nominal_gru_run_spec
from rlrmp.runtime.checkpoint_custody import cs_custody_training_spec
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    RLRMP_RUN_SPEC_PAYLOAD_KEY,
    assert_runtime_graph_matches_training_spec,
    feedbax_training_run_spec_from_payload,
    hydrate_compact_run_spec_envelope,
)
from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
)
from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.initial_slots import RlrmpRuntime, split_initial_keys
from rlrmp.train.executor.slots import (
    COMPLETED_BATCHES,
    HISTORY_CHUNK_BYTES,
    MODEL,
    OPTIMIZER,
    PRNG,
    TRAIN_LOSS,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    DEFAULT_TARGET_SUPPORT_PROFILE,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_PLAIN_MODE,
    make_broad_epsilon_pgd_pre_step,
)
from rlrmp.train.progress import format_batch_line, make_executor_batch_log_callback
from rlrmp.train.resume_control import (
    LaunchContinuation,
    attach_cs_supervised_checkpoint_continuation,
    emit_launch_continuation,
    resolve_launch_continuation,
)
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
    setup_task_model_pair,
)
from rlrmp.model.trainable import staged_network_trainable_parts
from rlrmp.train.training_configs import (
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    CS_CONTROL_SCALE,
    CS_POSITION_SCALE,
    CS_VELOCITY_SCALE,
    CsNominalGruConfig,
    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
    ISSUE_ID,
)
from rlrmp.train.executor.checkpoints import (
    ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON,
    SCHEMA_VERSION,
    TrainingState,
    _atomic_write_json,
    _load_latest_checkpoint_materialization,
    _save_pytree,
    latest_checkpoint_path,
    load_latest_checkpoint as load_latest_checkpoint,
    save_training_checkpoint,
)

logger = logging.getLogger(__name__)

VolumeCommit = Callable[[], None]

DEFAULT_OUTPUT_DIR = str(CsNominalGruConfig.model_fields["output_dir"].default)

DEFAULT_CHECKPOINT_INTERVAL_BATCHES = int(
    CsNominalGruConfig.model_fields["checkpoint_interval_batches"].default
)

DEFAULT_DELAYED_GO_CUE_MIN_STEP = int(
    CsNominalGruConfig.model_fields["delayed_reach_go_cue_min_step"].default
)

DEFAULT_DELAYED_GO_CUE_MAX_STEP = int(
    CsNominalGruConfig.model_fields["delayed_reach_go_cue_max_step"].default
)

DEFAULT_DELAYED_P_CATCH_TRIAL = float(
    CsNominalGruConfig.model_fields["delayed_reach_p_catch_trial"].default
)


@dataclass(frozen=True)
class RunSpecExecutionContext:
    """Validated C&S GRU training contract used by the execution path."""

    run_spec_path: Path
    run_spec: dict[str, Any]
    args: argparse.Namespace
    hps: TreeNamespace


@dataclass(frozen=True)
class CsSupervisedNativeChunkRecord:
    """Host-side record for one native cs-supervised executor chunk."""

    state: TrainingState
    history_chunk: Any
    pgd_diagnostics: dict[str, np.ndarray]
    chunk_batches: int
    duration_seconds: float


@dataclass
class CsSupervisedNativeRuntime:
    """Runtime-only objects used by the native cs-supervised chunk kernel."""

    args: argparse.Namespace
    hps: TreeNamespace
    pair: Any
    optimizer: optax.GradientTransformation
    where_train: Callable[[Any], Any]
    model_array_template: Any
    optimizer_template: Any
    pre_step_fn: Callable[..., Any] | None
    current_model: Any | None = None
    current_optimizer_state: Any | None = None
    current_completed_batches: int = 0
    history: Any | None = None
    records: list[CsSupervisedNativeChunkRecord] = field(default_factory=list)


class GradientDiagnosticsState(NamedTuple):
    """Optimizer-side scalar diagnostics captured before global-norm clipping."""

    count: Any
    gradient_norm_pre_clip: Any
    gradient_clipped: Any
    learning_rate: Any


class UpdateDiagnosticsState(NamedTuple):
    """Optimizer-side scalar diagnostics captured after the optimizer update."""

    count: Any
    update_norm: Any
    parameter_norm: Any
    update_parameter_norm_ratio: Any


def load_validated_run_spec(
    run_spec_path: Path | str,
    *,
    require_graph_sidecars: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Load and validate a composed C&S GRU ``TrainingRunSpec`` recipe."""

    payload_path = Path(run_spec_path)
    raw_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, Mapping):
        raise ValueError("C&S GRU run spec must be a JSON object")
    payload = hydrate_compact_run_spec_envelope(raw_payload)
    validate_nominal_gru_run_spec(
        payload,
        spec_dir=run_spec_sidecar_dir(payload_path),
        require_graph_sidecars=require_graph_sidecars,
    )
    _validate_composed_training_spec_payload(payload)
    _validate_adaptive_epsilon_cross_mirrors(payload)
    return payload_path, payload


def build_execution_context_from_spec(
    run_spec_path: Path | str,
    *,
    dry_run: bool = False,
    resume: bool = False,
    allow_fresh_start: bool = False,
    stop_after_batches: int | None = None,
    disable_progress: bool = False,
    quiet_progress: bool = False,
    log_step: int | None = None,
) -> RunSpecExecutionContext:
    """Build an execution context directly from a resolved run-spec recipe.

    This library entry point keeps internal tooling independent of the retired
    flag-per-training-field authoring parser. Only operational lifecycle and
    presentation controls may be supplied alongside the spec-owned values.
    """

    payload_path, run_spec = load_validated_run_spec(run_spec_path)
    values = _args_values_from_run_spec(run_spec)
    values.update(
        {
            "run_spec": str(payload_path),
            "dry_run": dry_run,
            "resume": resume,
            "allow_fresh_start": allow_fresh_start,
            "stop_after_batches": stop_after_batches,
            "disable_progress": disable_progress,
            "quiet_progress": quiet_progress,
            "log_step": values["log_step"] if log_step is None else log_step,
            "verify_resume_only": False,
        }
    )
    return RunSpecExecutionContext(
        run_spec_path=payload_path,
        run_spec=run_spec,
        args=argparse.Namespace(**values),
        hps=_hps_from_run_spec(run_spec),
    )


def _validate_composed_training_spec_payload(run_spec: dict[str, Any]) -> None:
    missing = [
        key
        for key in (FEEDBAX_TRAINING_RUN_SPEC_KEY, RLRMP_RUN_SPEC_PAYLOAD_KEY)
        if key not in run_spec
    ]
    if missing:
        raise ValueError(
            "C&S GRU run spec must embed composed TrainingRunSpec payloads: " + ", ".join(missing)
        )
    feedbax_training_run_spec_from_payload(run_spec)
    extension = run_spec[RLRMP_RUN_SPEC_PAYLOAD_KEY]
    if not isinstance(extension, dict):
        raise ValueError("C&S GRU run spec rlrmp_run_spec payload must be an object")
    accept_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        extension,
        source_version=extension.get("schema_version"),
        path=RLRMP_RUN_SPEC_PAYLOAD_KEY,
    )


def _validate_adaptive_epsilon_cross_mirrors(run_spec: dict[str, Any]) -> None:
    """Reject drift between runtime HPS and the governed adaptive method payload."""

    hps = _dict_value(run_spec, "hps")
    hps_adaptive = _dict_value(hps, "adaptive_epsilon_curriculum")
    if hps_adaptive.get("enabled") is not True:
        return
    training_spec = feedbax_training_run_spec_from_payload(run_spec)
    method_ref = training_spec.method_ref
    if f"{method_ref.package}/{method_ref.name}/{method_ref.version}" != (
        "rlrmp/adaptive_epsilon_curriculum/v1"
    ):
        raise ValueError(
            "Adaptive-epsilon runtime HPS requires the governed adaptive-epsilon method payload"
        )
    payload = training_spec.method_payload.payload
    if not isinstance(payload, Mapping):
        raise ValueError("Adaptive-epsilon method payload must be an object")
    payload_config = _dict_value(payload, "config")
    payload_damage = _dict_value(payload, "damage_schedule")
    payload_lambda = _dict_value(payload, "lambda_update")
    hps_damage = _dict_value(hps_adaptive, "damage_schedule")
    hps_lambda = _dict_value(hps_adaptive, "lambda_update")

    mirrors = {
        "damage_schedule.start": (
            hps_damage.get("start"),
            payload_damage.get("start"),
            payload_config.get("adaptive_epsilon_damage_start"),
        ),
        "damage_schedule.peak": (
            hps_damage.get("peak"),
            payload_damage.get("peak"),
            payload_config.get("adaptive_epsilon_damage_peak"),
        ),
        "damage_schedule.final": (
            hps_damage.get("final"),
            payload_damage.get("final"),
            payload_config.get("adaptive_epsilon_damage_final"),
        ),
        "lambda_update.interval_batches": (
            hps_lambda.get("interval_batches"),
            payload_lambda.get("interval_batches"),
            payload_config.get("adaptive_epsilon_update_interval_batches"),
        ),
        "lambda_update.ema_alpha": (
            hps_lambda.get("ema_alpha"),
            payload_lambda.get("ema_alpha"),
            payload_config.get("adaptive_epsilon_ema_alpha"),
        ),
        "lambda_update.eta": (
            hps_lambda.get("eta"),
            payload_lambda.get("eta"),
            payload_config.get("adaptive_epsilon_eta"),
        ),
        "lambda_update.deadband_frac": (
            hps_lambda.get("deadband_frac"),
            payload_lambda.get("deadband_frac"),
            payload_config.get("adaptive_epsilon_deadband_frac"),
        ),
        "lambda_update.max_log_step": (
            hps_lambda.get("max_log_step"),
            payload_lambda.get("max_log_step"),
            payload_config.get("adaptive_epsilon_max_log_step"),
        ),
        "lambda_update.lambda_min": (
            hps_lambda.get("lambda_min"),
            payload_lambda.get("lambda_min"),
            payload_config.get("adaptive_epsilon_lambda_min"),
        ),
        "lambda_update.freeze_during_application_ramp": (
            hps_lambda.get("freeze_during_application_ramp"),
            payload_lambda.get("freeze_during_application_ramp"),
            payload_config.get("adaptive_epsilon_freeze_during_application_ramp"),
        ),
    }
    for field_name, values in mirrors.items():
        if values[0] != values[1] or values[1] != values[2]:
            raise ValueError(
                "Adaptive-epsilon cross-mirror mismatch "
                f"field={field_name}: hps={values[0]!r} payload={values[1]!r} "
                f"config={values[2]!r}"
            )


def _run_spec_payload_schema_version(run_spec: dict[str, Any]) -> str:
    """Return the inline payload version that manifest preflight must bind."""

    return str(run_spec[RLRMP_RUN_SPEC_PAYLOAD_KEY]["schema_version"])


def _hps_from_run_spec(run_spec: dict[str, Any]) -> TreeNamespace:
    hps = dict(run_spec["hps"])
    if hps.get("hidden_type") == "equinox.nn._rnn.GRUCell":
        hps["hidden_type"] = eqx.nn.GRUCell
    return dict_to_namespace(hps, to_type=TreeNamespace)


def _args_values_from_run_spec(run_spec: dict[str, Any]) -> dict[str, Any]:
    hps = run_spec.get("hps") or {}
    model = _dict_value(hps, "model")
    loss = _dict_value(hps, "loss")
    loss_weights = _dict_value(loss, "weights")
    perturbation = _dict_value(hps, "perturbation_training")
    broad = _dict_value(hps, "broad_epsilon_training")
    broad_pgd = _dict_value(hps, "broad_epsilon_pgd_training")
    policy_adversary = _dict_value(hps, "policy_adversary_training")
    policy_payload = _dict_value(policy_adversary, "policy")
    policy_optimizer = _dict_value(policy_adversary, "inner_optimizer")
    policy_objective = _dict_value(policy_adversary, "objective")
    policy_budget = _dict_value(policy_adversary, "budget_contract")
    policy_budget_source = _dict_value(policy_budget, "budget_source")
    broad_pgd_schedule = _dict_value(broad_pgd, "budget_schedule")
    broad_pgd_conditioning = _dict_value(broad_pgd_schedule, "conditioning_scalar")
    broad_pgd_max_radius_source = _dict_value(broad_pgd_schedule, "max_radius_source")
    broad_pgd_budget = _dict_value(broad_pgd, "budget_contract")
    broad_pgd_budget_source = _dict_value(broad_pgd_budget, "budget_source")
    broad_pgd_objective = _dict_value(broad_pgd, "objective")
    broad_pgd_mechanism = _dict_value(broad_pgd, "mechanism")
    broad_pgd_safety_cap = _dict_value(broad_pgd, "safety_cap")
    broad_pgd_safety_cap_source = _dict_value(broad_pgd_safety_cap, "source")
    adaptive_epsilon = _dict_value(hps, "adaptive_epsilon_curriculum")
    adaptive_damage = _dict_value(adaptive_epsilon, "damage_schedule")
    adaptive_lambda = _dict_value(adaptive_epsilon, "lambda_update")
    adaptive_outer_weight = _dict_value(adaptive_epsilon, "outer_adversarial_weight")
    target_relative = _dict_value(hps, "target_relative_multitarget")
    target_distribution = _dict_value(target_relative, "target_distribution")
    delayed = _dict_value(hps, "delayed_reach")
    delayed_go = _dict_value(delayed, "go_cue_sampling")
    delayed_catch = _dict_value(delayed, "catch_trials")
    delayed_norm = _dict_value(loss, "delayed_trial_type_normalization")
    population = _dict_value(model, "population_structure")
    pgd_inner = _dict_value(broad_pgd, "inner_maximizer")

    values = {
        "output_dir": str(run_spec.get("artifact_output_dir", DEFAULT_OUTPUT_DIR)),
        "spec_dir": str(run_spec.get("spec_dir")) if run_spec.get("spec_dir") else None,
        "issue": str(run_spec.get("issue", ISSUE_ID)),
        "seed": int(run_spec.get("seed", 42)),
        "n_train_batches": int(
            run_spec.get("n_train_batches", hps.get("n_batches_condition", 12000))
        ),
        "batch_size": int(run_spec.get("batch_size", hps.get("batch_size", 250))),
        "controller_lr": float(run_spec.get("controller_lr", hps.get("learning_rate_0", 1e-2))),
        "lr_warmup_batches": int(hps.get("constant_lr_iterations", 0)),
        "lr_warmup_init_fraction": float(hps.get("warmup_init_fraction", 0.1)),
        "lr_cosine_alpha": float(hps.get("cosine_annealing_alpha", 1.0)),
        "gradient_clip_norm": hps.get("gradient_clip_norm"),
        "n_replicates": int(model.get("n_replicates", 5)),
        "hidden_size": int(model.get("hidden_size", 180)),
        "plant_backend": str(model.get("plant_backend", CS_LSS_PLANT_BACKEND)),
        "no_integrator_state": bool(model.get("no_integrator_state", False)),
        "stochastic_preset": str(model.get("stochastic_preset", DEFAULT_STOCHASTIC_PRESET)),
        "n_input_only": int(population.get("n_input_only", 0)),
        "n_readout_only": int(population.get("n_readout_only", 0)),
        "n_recurrent_only": int(population.get("n_recurrent_only", 0)),
        "effector_pos_running": float(loss_weights.get("effector_pos_running", CS_POSITION_SCALE)),
        "effector_vel_running": float(loss_weights.get("effector_vel_running", CS_VELOCITY_SCALE)),
        "effector_terminal_pos": float(
            loss_weights.get("effector_terminal_pos", CS_POSITION_SCALE)
        ),
        "effector_terminal_vel": float(
            loss_weights.get("effector_terminal_vel", CS_VELOCITY_SCALE)
        ),
        "effector_final_vel": float(loss_weights.get("effector_final_vel", 0.0)),
        "nn_output": float(loss_weights.get("nn_output", CS_CONTROL_SCALE)),
        "nn_output_jerk": float(loss_weights.get("nn_output_jerk", 0.0)),
        "nn_output_pre_go": float(loss_weights.get("nn_output_pre_go", 0.0)),
        "delayed_pre_go_force_filter_hold": float(
            loss_weights.get("delayed_pre_go_force_filter_hold", 0.0)
        ),
        "delayed_pre_go_start_pos_hold": float(
            loss_weights.get("delayed_pre_go_start_pos_hold", 0.0)
        ),
        "delayed_pre_go_start_pos_hold_norm": str(
            loss.get("delayed_pre_go_start_pos_hold_norm", "l2")
        ),
        "delayed_pre_go_zero_vel_hold": float(
            loss_weights.get("delayed_pre_go_zero_vel_hold", 0.0)
        ),
        "loss_objective": str(
            run_spec.get("loss_objective", loss.get("objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
        ),
        "regularized_fidelity": float(loss_weights.get("nn_hidden", 0.0)) > 0.0,
        "perturbation_training": bool(perturbation.get("enabled", False)),
        "perturbation_nominal_fraction": float(perturbation.get("nominal_fraction", 0.45)),
        "perturbation_single_fraction": float(perturbation.get("single_fraction", 0.45)),
        "perturbation_combined_fraction": float(perturbation.get("combined_fraction", 0.10)),
        "perturbation_combined_amplitude_scale": float(
            perturbation.get("combined_amplitude_scale", 0.5)
        ),
        "perturbation_initial_position_offset_m": float(
            _family_amplitude(perturbation, "initial_position", 0.01)
        ),
        "perturbation_initial_velocity_offset_m_s": float(
            _family_amplitude(perturbation, "initial_velocity", 0.05)
        ),
        "perturbation_process_epsilon_scale": float(
            _family_amplitude(perturbation, "process_epsilon", 0.01)
        ),
        "perturbation_command_input_pulse_n": float(
            _family_amplitude(perturbation, "command_input", 1.0)
        ),
        "perturbation_sensory_feedback_offset_m": float(
            _family_amplitude(perturbation, "sensory_feedback", 0.01)
        ),
        "perturbation_delayed_observation_offset_m": float(
            _family_amplitude(perturbation, "delayed_observation", 0.01)
        ),
        "perturbation_pulse_start_step": int(_pulse_value(perturbation, "start_step", 20)),
        "perturbation_pulse_duration_steps": int(_pulse_value(perturbation, "duration_steps", 5)),
        "perturbation_calibrated_timing": bool(perturbation.get("calibrated_timing", False)),
        "perturbation_movement_age_timing": bool(perturbation.get("movement_age_timing", False)),
        "perturbation_physical_level": str(perturbation.get("physical_level", "moderate")),
        "perturbation_calibration_regime": str(
            perturbation.get("calibration_regime", "open_loop_all")
        ),
        "perturbation_closed_loop_calibration_table": perturbation.get(
            "closed_loop_calibration_table_path",
            None,
        ),
        "target_relative_multitarget": bool(target_relative.get("enabled", False)),
        "target_support_profile": str(
            target_distribution.get("target_support_profile", DEFAULT_TARGET_SUPPORT_PROFILE)
        ),
        "delayed_reach": bool(delayed.get("enabled", False)),
        "delayed_reach_go_cue_min_step": int(
            delayed_go.get("min_step_inclusive", DEFAULT_DELAYED_GO_CUE_MIN_STEP)
        ),
        "delayed_reach_go_cue_max_step": int(
            delayed_go.get("max_step_inclusive", DEFAULT_DELAYED_GO_CUE_MAX_STEP)
        ),
        "delayed_reach_p_catch_trial": float(
            delayed_catch.get("p_catch_trial", DEFAULT_DELAYED_P_CATCH_TRIAL)
        ),
        "delayed_reach_trial_type_normalized_loss": bool(delayed_norm.get("enabled", False)),
        "delayed_reach_no_catch_qrf_weight": float(delayed_norm.get("no_catch_weight", 1.0)),
        "delayed_reach_catch_qrf_weight": float(delayed_norm.get("catch_weight", 1.0)),
        "delayed_movement_cost_tail_mode": str(
            loss.get(
                "delayed_movement_cost_tail_mode",
                _dict_value(delayed, "movement_epoch").get(
                    "cost_tail_mode",
                    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
                ),
            )
        ),
        "force_filter_feedback": bool(model.get("force_filter_feedback", False)),
        "broad_epsilon_training": bool(broad.get("enabled", False)),
        "broad_epsilon_pgd_training": bool(broad_pgd.get("enabled", False)),
        "broad_epsilon_pgd_mechanism": str(
            broad_pgd.get(
                "adversary_mechanism",
                broad_pgd_mechanism.get("name", BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM),
            )
        ),
        "broad_epsilon_level": str(broad_pgd.get("level", broad.get("level", "moderate"))),
        "broad_epsilon_budget_scale": float(
            broad_pgd.get("budget_scale", broad.get("budget_scale", 1.0))
        ),
        "broad_epsilon_reach_scaling": bool(
            broad_pgd.get("reach_length_scaling", broad.get("reach_length_scaling", True))
        ),
        "broad_epsilon_pgd_steps": int(pgd_inner.get("n_steps", broad_pgd.get("n_steps", 3))),
        "broad_epsilon_pgd_step_size_fraction": float(
            pgd_inner.get(
                "step_size_fraction_of_l2_radius",
                broad_pgd.get("step_size_fraction", 0.25),
            )
        ),
        "broad_epsilon_pgd_inner_optimizer_method": str(
            pgd_inner.get(
                "method",
                broad_pgd.get(
                    "inner_optimizer_method",
                    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
                ),
            )
        ),
        "broad_epsilon_pgd_adam_lr": float(
            _dict_value(pgd_inner, "adam").get(
                "learning_rate",
                pgd_inner.get("learning_rate", broad_pgd.get("adam_learning_rate", 3e-4)),
            )
        ),
        "broad_epsilon_pgd_adam_b1": float(
            _dict_value(pgd_inner, "adam").get("b1", broad_pgd.get("adam_b1", 0.9))
        ),
        "broad_epsilon_pgd_adam_b2": float(
            _dict_value(pgd_inner, "adam").get("b2", broad_pgd.get("adam_b2", 0.999))
        ),
        "broad_epsilon_pgd_adam_eps": float(
            _dict_value(pgd_inner, "adam").get("eps", broad_pgd.get("adam_eps", 1e-8))
        ),
        "broad_epsilon_pgd_budget_schedule": str(
            broad_pgd_schedule.get("mode", broad_pgd.get("budget_schedule_mode", "fixed"))
        ),
        "broad_epsilon_pgd_fixed_radius_15cm": broad_pgd_budget.get(
            "effective_l2_radius_15cm",
            broad_pgd.get("fixed_l2_radius_15cm"),
        ),
        "broad_epsilon_pgd_fixed_radius_source": broad_pgd_budget_source.get(
            "key",
            broad_pgd.get("fixed_radius_source"),
        ),
        "broad_epsilon_pgd_objective": str(
            broad_pgd_objective.get(
                "kind",
                broad_pgd.get("objective_kind", BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE),
            )
        ),
        "broad_epsilon_pgd_energy_gamma_star": broad_pgd_objective.get(
            "gamma_star",
            broad_pgd.get("energy_gamma_star"),
        ),
        "broad_epsilon_pgd_energy_gamma_factor": broad_pgd_objective.get(
            "gamma_factor",
            broad_pgd.get("energy_gamma_factor"),
        ),
        "broad_epsilon_pgd_energy_gamma": broad_pgd_objective.get(
            "gamma",
            broad_pgd.get("energy_gamma"),
        ),
        "broad_epsilon_pgd_energy_penalty_scale": float(
            broad_pgd_objective.get(
                "penalty_scale_c",
                broad_pgd.get("energy_penalty_scale", 1.0),
            )
        ),
        "broad_epsilon_pgd_energy_lambda": broad_pgd_objective.get(
            "lambda",
            broad_pgd.get("energy_lambda"),
        ),
        "broad_epsilon_pgd_safety_cap_15cm": broad_pgd_safety_cap.get(
            "l2_radius_15cm",
            broad_pgd.get("safety_cap_l2_radius_15cm"),
        ),
        "broad_epsilon_pgd_safety_cap_source": broad_pgd_safety_cap_source.get(
            "key",
            broad_pgd.get("safety_cap_source"),
        ),
        "broad_epsilon_pgd_sisu_condition_input": str(
            broad_pgd_conditioning.get(
                "input_key",
                broad_pgd.get("sisu_condition_input", "auto"),
            )
        ),
        "broad_epsilon_pgd_sisu_max_radius": broad_pgd_schedule.get(
            "max_l2_radius_15cm",
            broad_pgd.get("sisu_max_l2_radius_15cm"),
        ),
        "broad_epsilon_pgd_sisu_max_radius_source": broad_pgd_max_radius_source.get(
            "key",
            broad_pgd.get("sisu_max_radius_source"),
        ),
        "adaptive_epsilon_curriculum": bool(
            adaptive_epsilon.get("enabled", _config_default("adaptive_epsilon_curriculum"))
        ),
        "adaptive_epsilon_controller_training_mode": str(
            adaptive_epsilon.get(
                "controller_training_mode",
                ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
            )
        ),
        "adaptive_epsilon_damage_start": float(
            adaptive_damage.get("start", _config_default("adaptive_epsilon_damage_start"))
        ),
        "adaptive_epsilon_damage_peak": float(
            adaptive_damage.get("peak", _config_default("adaptive_epsilon_damage_peak"))
        ),
        "adaptive_epsilon_damage_final": float(
            adaptive_damage.get("final", _config_default("adaptive_epsilon_damage_final"))
        ),
        "adaptive_epsilon_damage_ramp_batches": int(
            adaptive_damage.get(
                "ramp_batches",
                _config_default("adaptive_epsilon_damage_ramp_batches"),
            )
        ),
        "adaptive_epsilon_damage_anneal_batches": int(
            adaptive_damage.get(
                "anneal_batches",
                _config_default("adaptive_epsilon_damage_anneal_batches"),
            )
        ),
        "adaptive_epsilon_update_interval_batches": int(
            adaptive_lambda.get(
                "interval_batches",
                _config_default("adaptive_epsilon_update_interval_batches"),
            )
        ),
        "adaptive_epsilon_ema_alpha": float(
            adaptive_lambda.get("ema_alpha", _config_default("adaptive_epsilon_ema_alpha"))
        ),
        "adaptive_epsilon_eta": float(
            adaptive_lambda.get("eta", _config_default("adaptive_epsilon_eta"))
        ),
        "adaptive_epsilon_deadband_frac": float(
            adaptive_lambda.get(
                "deadband_frac",
                _config_default("adaptive_epsilon_deadband_frac"),
            )
        ),
        "adaptive_epsilon_hysteresis_frac": adaptive_lambda.get(
            "hysteresis_frac",
            _config_default("adaptive_epsilon_hysteresis_frac"),
        ),
        "adaptive_epsilon_freeze_during_application_ramp": bool(
            adaptive_lambda.get(
                "freeze_during_application_ramp",
                _config_default("adaptive_epsilon_freeze_during_application_ramp"),
            )
        ),
        "adaptive_epsilon_gain_normalization": bool(
            adaptive_lambda.get(
                "gain_normalization",
                _config_default("adaptive_epsilon_gain_normalization"),
            )
        ),
        "adaptive_epsilon_gain_ema_alpha": float(
            adaptive_lambda.get(
                "gain_ema_alpha",
                _config_default("adaptive_epsilon_gain_ema_alpha"),
            )
        ),
        "adaptive_epsilon_gain_min": float(
            adaptive_lambda.get(
                "gain_min",
                _config_default("adaptive_epsilon_gain_min"),
            )
        ),
        "adaptive_epsilon_gain_max": float(
            adaptive_lambda.get(
                "gain_max",
                _config_default("adaptive_epsilon_gain_max"),
            )
        ),
        "adaptive_epsilon_lambda_min": (
            None
            if adaptive_lambda.get("lambda_min", None) is None
            else float(adaptive_lambda["lambda_min"])
        ),
        "adaptive_epsilon_lambda_max": adaptive_lambda.get("lambda_max"),
        "adaptive_epsilon_max_log_step": float(
            adaptive_lambda.get(
                "max_log_step",
                _config_default("adaptive_epsilon_max_log_step"),
            )
        ),
        "adaptive_epsilon_outer_weight_start": float(
            adaptive_outer_weight.get(
                "start",
                _config_default("adaptive_epsilon_outer_weight_start"),
            )
        ),
        "adaptive_epsilon_outer_weight_final": float(
            adaptive_outer_weight.get(
                "final",
                _config_default("adaptive_epsilon_outer_weight_final"),
            )
        ),
        "adaptive_epsilon_outer_weight_ramp_batches": int(
            adaptive_outer_weight.get(
                "ramp_batches",
                _config_default("adaptive_epsilon_outer_weight_ramp_batches"),
            )
        ),
        "policy_adversary_training": bool(policy_adversary.get("enabled", False)),
        "policy_adversary_policy_class": str(
            policy_adversary.get(
                "policy_class",
                policy_payload.get("kind", POLICY_ADVERSARY_MEMORYLESS_MLP),
            )
        ),
        "policy_adversary_mode": str(
            policy_adversary.get(
                "row_mode",
                policy_objective.get("active", POLICY_ADVERSARY_PLAIN_MODE),
            )
        ),
        "policy_adversary_width": int(policy_payload.get("width", 64)),
        "policy_adversary_depth": int(policy_payload.get("depth", 2)),
        "policy_adversary_steps": int(
            policy_optimizer.get("n_ascent_steps_per_controller_step", 5)
        ),
        "policy_adversary_lr": float(policy_optimizer.get("learning_rate", 3e-4)),
        "policy_adversary_energy_gamma": float(policy_objective.get("energy_penalty_gamma", 1.0)),
        "policy_adversary_radius_15cm": policy_budget.get(
            "effective_l2_radius_15cm",
            policy_budget.get("active_max_l2_radius_15cm"),
        ),
        "policy_adversary_radius_source": policy_budget_source.get("key"),
        "initial_hidden_encoder": bool(model.get("initial_hidden_encoder", False)),
        "full_train": (
            run_spec.get("mode") == "full_train"
            or run_spec.get("full_training_launch") == "requested"
        ),
        "checkpoint_interval_batches": int(
            _dict_value(run_spec, "checkpointing").get(
                "interval_batches", DEFAULT_CHECKPOINT_INTERVAL_BATCHES
            )
        ),
        "training_diagnostics": bool(
            _dict_value(run_spec, "training_diagnostics").get(
                "enabled", hps.get("training_diagnostics", True)
            )
        ),
        "dry_run": False,
        "resume": False,
        "allow_fresh_start": False,
        "stop_after_batches": None,
        "smoke": False,
    }
    return CsNominalGruConfig.model_validate(values).model_dump(mode="python")


def _dict_value(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def _family_amplitude(
    perturbation: dict[str, Any],
    family: str,
    default: float,
) -> float:
    families = _dict_value(perturbation, "families")
    payload = _dict_value(families, family)
    return float(payload.get("amplitude", default))


def _pulse_value(perturbation: dict[str, Any], key: str, default: int) -> int:
    pulse = _dict_value(perturbation, "pulse")
    return int(pulse.get(key, perturbation.get(f"pulse_{key}", default)))


def run_full_training(
    args: argparse.Namespace,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    """Compatibility adapter that enters full training through a validated spec."""

    if getattr(args, "run_spec", None) is None:
        args = _apply_smoke_overrides(args)
        spec_result = write_run_spec(args)
        run_spec_path = spec_result["run_spec_path"]
    else:
        run_spec_path = args.run_spec
    context = build_execution_context_from_spec(
        run_spec_path,
        dry_run=bool(getattr(args, "dry_run", False)),
        resume=bool(getattr(args, "resume", False)),
        allow_fresh_start=bool(getattr(args, "allow_fresh_start", False)),
        stop_after_batches=getattr(args, "stop_after_batches", None),
        disable_progress=bool(getattr(args, "disable_progress", False)),
        quiet_progress=bool(getattr(args, "quiet_progress", False)),
        log_step=getattr(args, "log_step", None),
    )
    return _run_full_training_from_context(context, volume_commit=volume_commit)


class CsSupervisedExternalObjectiveLoss(AbstractLoss):
    """Placeholder lowered loss for runtime-owned C&S supervised objectives."""

    label: str = "rlrmp_cs_supervised_external_objective"

    def term(self, states: Any, trial_specs: Any, model: Any) -> Any:
        del states, trial_specs, model
        return jnp.asarray(0.0)


class CsSupervisedExternalObjectiveLossService(LossService):
    """Lower the governed C&S external objective for native execution."""

    def lower_objective_slot(
        self,
        slot: Any,
        *,
        graph: Any = None,
        trial_axis: str = "batch",
        path: str = "/objective",
    ) -> LoweredObjective:
        if slot.kind == "external" and slot.schema_id == "rlrmp.cs_gru_objective":
            del graph, trial_axis, path
            return LoweredObjective(
                loss=CsSupervisedExternalObjectiveLoss(),
                requirements=ObjectiveExecutionRequirements(),
                source_kind="objective_spec",
            )
        return super().lower_objective_slot(
            slot,
            graph=graph,
            trial_axis=trial_axis,
            path=path,
        )


def _cs_supervised_native_supported(hps: TreeNamespace) -> bool:
    """Return whether this run belongs to the R2 native executor lane."""

    return not (
        _adaptive_epsilon_curriculum_enabled(hps) or _policy_adversary_training_enabled(hps)
    )


def build_cs_supervised_native_initial_slots(
    *,
    run_spec: Mapping[str, Any],
    hps: TreeNamespace,
    args: argparse.Namespace,
    key: Any,
) -> tuple[dict[str, Any], RlrmpRuntime]:
    """Build cs-supervised native-executor initial slots and runtime context."""

    del run_spec
    key_init, key_train, _key_adversary = split_initial_keys(key)
    pair = setup_task_model_pair(hps, key=key_init)
    optimizer = _build_optimizer(hps)
    where_train = _where_train()[0]
    template_state = _initial_training_state(
        model=pair.model,
        trainer=optimizer,
        where_train=where_train,
        key=key_train,
    )
    history_base = _native_resume_history_base(
        output_dir=Path(args.output_dir),
        args=args,
        model_template=pair.model,
        optimizer_state_template=template_state.optimizer_state,
    )
    model_array_template = eqx.filter(pair.model, eqx.is_array)
    runtime = CsSupervisedNativeRuntime(
        args=args,
        hps=hps,
        pair=pair,
        optimizer=optimizer,
        where_train=where_train,
        model_array_template=model_array_template,
        optimizer_template=template_state.optimizer_state,
        pre_step_fn=make_broad_epsilon_pgd_pre_step(hps.broad_epsilon_pgd_training),
        current_model=pair.model,
        current_optimizer_state=template_state.optimizer_state,
        history=history_base,
    )
    return (
        {
            MODEL: _cs_model_slot(pair.model, model_array_template),
            OPTIMIZER: _cs_optimizer_slot(template_state.optimizer_state),
            PRNG: key_train,
            COMPLETED_BATCHES: jnp.asarray(0, dtype=jnp.int32),
            TRAIN_LOSS: 0.0,
            HISTORY_CHUNK_BYTES: b"",
        },
        RlrmpRuntime(
            components={"cs_supervised": runtime},
            stop_after_batches=args.stop_after_batches,
        ),
    )


def _execute_native_training_run_spec(
    training_spec: Any,
    *,
    progress_phase: str,
    total_batches: int,
    **kwargs: Any,
) -> Any:
    """Execute one native method with the shared readiness/progress observers."""
    emitter = RunEventEmitter.from_env()
    try:
        return execute_training_run_spec(
            training_spec,
            progress_callback=make_executor_batch_log_callback(
                {progress_phase: total_batches},
                logger=_native_progress_logger(),
            ),
            run_event_emitter=emitter,
            **kwargs,
        )
    finally:
        if emitter is not None:
            emitter.close()


def _native_progress_logger() -> logging.Logger:
    """Return a per-run INFO logger whose stdout handler flushes each BATCH line."""
    progress_logger = logging.Logger(f"{__name__}.native_progress", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    progress_logger.addHandler(handler)
    progress_logger.propagate = False
    return progress_logger


def _run_cs_supervised_native_from_context(
    context: RunSpecExecutionContext,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    args = context.args
    run_spec = context.run_spec
    run_spec_path = context.run_spec_path
    hps = context.hps
    if int(args.n_train_batches) < 1:
        raise ValueError("--n-train-batches must be positive for --full-train")
    if int(args.checkpoint_interval_batches) < 1:
        raise ValueError("--checkpoint-interval-batches must be positive")
    stop_after_batches = None if args.stop_after_batches is None else int(args.stop_after_batches)
    if stop_after_batches is not None:
        if stop_after_batches < 1:
            raise ValueError("--stop-after-batches must be positive when provided")
        if stop_after_batches > int(args.n_train_batches):
            raise ValueError("--stop-after-batches cannot exceed --n-train-batches")

    spec_result = _spec_result_from_execution_context(context)
    output_dir = mkdir_p(Path(args.output_dir))
    key_init, _key_train, _key_adversary = split_initial_keys(jr.PRNGKey(int(args.seed)))
    pair_for_graph = setup_task_model_pair(hps, key=key_init)
    runtime_graph_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair_for_graph.model)
    assert_runtime_graph_matches_training_spec(
        run_spec,
        graph_spec=runtime_graph_bundle.graph_spec,
    )
    checkpoint_root = output_dir / "checkpoints"
    context, resume_native, continuation = _resolve_full_train_launch_context(
        context,
        checkpoint_root=checkpoint_root,
        stop_after_batches=stop_after_batches,
    )
    training_spec = cs_custody_training_spec(run_spec)
    training_spec = attach_cs_supervised_checkpoint_continuation(training_spec, continuation)
    execution_registry = _cs_supervised_execution_registry(training_spec)
    args = context.args
    initial_slots, runtime = build_cs_supervised_native_initial_slots(
        run_spec=training_spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(int(args.seed)),
    )
    started = time.perf_counter()
    execution = _execute_native_training_run_spec(
        training_spec,
        progress_phase="train_chunk",
        total_batches=int(args.n_train_batches),
        run_id=_cs_supervised_native_run_id(args, run_spec_path),
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        checkpoint_root=checkpoint_root,
        loss_service=CsSupervisedExternalObjectiveLossService(),
        training_spec_payload=run_spec.get(RLRMP_RUN_SPEC_PAYLOAD_KEY),
        training_spec_payload_kind=RUN_SPEC_KIND,
        training_spec_payload_schema_id=RUN_SPEC_SCHEMA_ID,
        training_spec_payload_schema_version=_run_spec_payload_schema_version(run_spec),
        training_spec_payload_ref=str(run_spec_path),
        resume=resume_native,
        resume_slot_transform=_cs_supervised_resume_slot_transform(),
        registry=execution_registry,
        issues=[str(args.issue)],
    )
    training_duration_seconds = time.perf_counter() - started
    native_runtime = runtime.component("cs_supervised")
    if not isinstance(native_runtime, CsSupervisedNativeRuntime):
        raise TypeError("cs_supervised runtime context was not installed")
    return _materialize_cs_supervised_native_result(
        context=context,
        spec_result={
            **spec_result,
            "training_manifest_path": str(execution.manifest_path),
        },
        runtime=native_runtime,
        training_duration_seconds=training_duration_seconds,
        stop_after_batches=stop_after_batches,
        checkpoint_writes=execution.checkpoint_writes,
        volume_commit=volume_commit,
    )


def _cs_supervised_execution_registry(training_spec: TrainingRunSpec) -> TrainingMethodRegistry:
    """Bind native execution to the exact governed worker contract being resumed."""

    registration = DEFAULT_TRAINING_METHOD_REGISTRY.resolve(
        training_spec.method_ref,
        path="/method_ref",
    )
    registry = TrainingMethodRegistry()
    registry.register(
        replace(
            registration,
            contract_factory=lambda: training_spec.worker_execution.method_contract,
        )
    )
    return registry


def _cs_supervised_native_run_id(args: argparse.Namespace, run_spec_path: Path) -> str:
    base = Path(run_spec_path).stem
    output_hash = hashlib.sha256(str(Path(args.output_dir).resolve()).encode()).hexdigest()[:8]
    if args.stop_after_batches is None:
        return f"{base}-{output_hash}"
    return f"{base}-{output_hash}-stop-after-{int(args.stop_after_batches)}"


def _resolve_full_train_launch_context(
    context: RunSpecExecutionContext,
    *,
    checkpoint_root: Path,
    stop_after_batches: int | None,
) -> tuple[RunSpecExecutionContext, bool, LaunchContinuation]:
    """Emit launch continuation summary and return executor resume semantics."""

    args = context.args
    stop_target = int(
        stop_after_batches if stop_after_batches is not None else args.n_train_batches
    )
    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=bool(args.resume),
        allow_fresh_start=bool(args.allow_fresh_start),
        stop_target_batches=stop_target,
    )
    emit_launch_continuation(continuation, logger=logger)
    if continuation.resume == bool(args.resume):
        return context, continuation.resume, continuation
    resolved_args = argparse.Namespace(**{**vars(args), "resume": continuation.resume})
    return replace(context, args=resolved_args), continuation.resume, continuation


def _cs_model_slot(model: Any, array_template: Any) -> tuple[Any, ...]:
    del array_template
    return tuple(jt.leaves(eqx.filter(model, eqx.is_array)))


def _cs_optimizer_slot(optimizer_state: Any) -> tuple[Any, ...]:
    return tuple(jt.leaves(optimizer_state))


def _native_resume_history_base(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    model_template: Any,
    optimizer_state_template: Any,
) -> Any | None:
    if not bool(args.resume):
        return None
    checkpoint_path = latest_checkpoint_path(output_dir / "checkpoints")
    if not checkpoint_path.exists():
        return None
    try:
        state = _load_latest_checkpoint_materialization(
            output_dir / "checkpoints",
            model_template=model_template,
            optimizer_state_template=optimizer_state_template,
            history_template=None,
            adversary_policy_template=None,
            adversary_optimizer_state_template=None,
        )
    except Exception:
        return None
    return state.history


def _materialize_cs_supervised_native_result(
    *,
    context: RunSpecExecutionContext,
    spec_result: dict[str, Any],
    runtime: CsSupervisedNativeRuntime,
    training_duration_seconds: float,
    stop_after_batches: int | None,
    checkpoint_writes: Sequence[Any],
    volume_commit: VolumeCommit | None,
) -> dict[str, Any]:
    args = context.args
    run_spec = context.run_spec
    run_spec_path = context.run_spec_path
    output_dir = mkdir_p(Path(args.output_dir))
    checkpoint_root = output_dir / "checkpoints"
    history_chunks_dir = output_dir / "history_chunks"
    chunks: list[dict[str, float | int | str]] = []
    pgd_diagnostic_chunks: list[dict[str, np.ndarray]] = []
    training_started = time.perf_counter() - training_duration_seconds
    final_state: TrainingState | None = None
    custody_by_completed = _checkpoint_writes_by_completed_batch(checkpoint_writes)
    for record in runtime.records:
        completed = int(record.state.completed_batches)
        history_chunk_path = history_chunks_dir / f"history_{completed:07d}.eqx"
        history_chunk_path.parent.mkdir(parents=True, exist_ok=True)
        _save_pytree(history_chunk_path, record.history_chunk)
        checkpoint_path = save_training_checkpoint(
            checkpoint_root,
            record.state,
            args=args,
            run_spec=run_spec,
            write_custody=False,
            custody_result=custody_by_completed.get(int(record.state.completed_batches)),
        )
        if record.pgd_diagnostics:
            pgd_diagnostic_chunks.append(record.pgd_diagnostics)
        if _training_diagnostics_enabled(args):
            write_training_diagnostics_sidecar(
                output_dir,
                args=args,
                run_spec=run_spec,
                state=record.state,
                training_history_path=output_dir / "training_history.eqx",
                pgd_diagnostic_chunks=pgd_diagnostic_chunks,
            )
        if not bool(args.disable_progress):
            _emit_checkpoint_progress(
                record.history_chunk,
                record.pgd_diagnostics,
                chunk_batches=record.chunk_batches,
                completed_batches=completed,
                total_batches=int(args.n_train_batches),
                elapsed_seconds=time.perf_counter() - training_started,
            )
        _commit_volume(volume_commit)
        chunks.append(
            {
                "completed_batches": completed,
                "checkpoint": str(checkpoint_path),
                "history_chunk": str(history_chunk_path),
                "chunk_batches": record.chunk_batches,
                "duration_seconds": record.duration_seconds,
                "batches_per_second": record.chunk_batches / record.duration_seconds,
            }
        )
        final_state = record.state
    if final_state is None:
        raise RuntimeError("native cs-supervised executor produced no training chunks")

    final_model_path = output_dir / "trained_model.eqx"
    final_history_path = output_dir / "training_history.eqx"
    final_summary_path = output_dir / "training_summary.json"
    _save_pytree(final_model_path, final_state.model, hyperparameters=run_spec)
    if final_state.history is not None:
        _save_pytree(final_history_path, final_state.history)
    diagnostics_metadata = write_training_diagnostics_sidecar(
        output_dir,
        args=args,
        run_spec=run_spec,
        state=final_state,
        training_history_path=final_history_path,
        pgd_diagnostic_chunks=pgd_diagnostic_chunks,
    )
    stopped_for_gate = stop_after_batches is not None and final_state.completed_batches < int(
        args.n_train_batches
    )
    final_summary = {
        "schema_version": f"{SCHEMA_VERSION}.training.v1",
        "issue": str(args.issue),
        "completed_batches": final_state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "stopped_early_for_checkpoint_gate": stopped_for_gate,
        "stopped_early_for_adaptive_epsilon_zero_adversary": False,
        "stop_reason": "checkpoint_gate_stop_after_batches" if stopped_for_gate else None,
        "stop_after_batches": stop_after_batches,
        "adaptive_epsilon_zero_adversary_guard": None,
        "training_duration_seconds": training_duration_seconds,
        "training_batches_per_second": (
            final_state.completed_batches / training_duration_seconds
            if training_duration_seconds > 0
            else None
        ),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": None,
        "training_history_path": str(final_history_path),
        "run_spec_path": str(run_spec_path),
        "graph_spec_path": spec_result["graph_spec_path"],
        "training_diagnostics": diagnostics_metadata,
        "chunks": chunks,
    }
    _atomic_write_json(final_summary_path, final_summary)
    _commit_volume(volume_commit)
    return {
        **spec_result,
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": None,
        "training_history_path": str(final_history_path),
        "training_summary_path": str(final_summary_path),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "completed_batches": final_state.completed_batches,
    }


def _run_full_training_from_context(
    context: RunSpecExecutionContext,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    """Run a validated C&S training spec through the owned execution path."""

    if _adaptive_epsilon_curriculum_enabled(context.hps):
        return _run_adaptive_epsilon_native_from_context(
            context,
            volume_commit=volume_commit,
        )
    if _policy_adversary_training_enabled(context.hps):
        return _run_policy_adversary_native_from_context(
            context,
            volume_commit=volume_commit,
        )
    if _cs_supervised_native_supported(context.hps):
        return _run_cs_supervised_native_from_context(
            context,
            volume_commit=volume_commit,
        )
    raise ValueError(
        "This C&S run is not covered by a registered native executor method. "
        "Legacy full-training loop fallback has been deleted."
    )


def verify_resume_from_context(context: RunSpecExecutionContext) -> dict[str, Any]:
    """Load and strictly validate the configured checkpoint without training."""

    args = argparse.Namespace(
        **{
            **vars(context.args),
            "resume": True,
            "allow_fresh_start": False,
        }
    )
    context = replace(context, args=args)
    checkpoint_root = Path(args.output_dir) / "checkpoints"
    context, resume_native, continuation = _resolve_full_train_launch_context(
        context,
        checkpoint_root=checkpoint_root,
        stop_after_batches=None,
    )
    if not resume_native:
        raise RuntimeError("--verify-resume-only requires an existing checkpoint")

    training_spec = feedbax_training_run_spec_from_payload(context.run_spec)
    resume_slot_transform: Any
    if _adaptive_epsilon_curriculum_enabled(context.hps):
        from rlrmp.train.adaptive_epsilon_native import (
            _resume_slot_transform,
            attach_adaptive_epsilon_checkpoint_continuation,
            build_adaptive_epsilon_native_initial_slots,
        )

        training_spec = attach_adaptive_epsilon_checkpoint_continuation(
            training_spec,
            source_completed_batches=continuation.completed_batches,
            target_total_batches=continuation.stop_target_batches,
        )
        initial_slots, _runtime = build_adaptive_epsilon_native_initial_slots(
            run_spec=context.run_spec,
            hps=context.hps,
            args=context.args,
            key=jr.PRNGKey(int(context.args.seed)),
        )
        resume_slot_transform = _resume_slot_transform(None)
    elif _policy_adversary_training_enabled(context.hps):
        from rlrmp.train.policy_adversary_native import (
            _resume_slot_transform,
            build_policy_adversary_native_initial_slots,
        )

        initial_slots, _runtime = build_policy_adversary_native_initial_slots(
            run_spec=context.run_spec,
            hps=context.hps,
            args=context.args,
            key=jr.PRNGKey(int(context.args.seed)),
        )
        resume_slot_transform = _resume_slot_transform(None)
    elif _cs_supervised_native_supported(context.hps):
        training_spec = attach_cs_supervised_checkpoint_continuation(
            training_spec,
            continuation,
        )
        initial_slots, _runtime = build_cs_supervised_native_initial_slots(
            run_spec=context.run_spec,
            hps=context.hps,
            args=context.args,
            key=jr.PRNGKey(int(context.args.seed)),
        )
        resume_slot_transform = _cs_supervised_resume_slot_transform()
    else:
        raise ValueError("run spec is not covered by a registered native executor method")

    loaded = load_feedbax_checkpoint(
        checkpoint_root,
        expected_run_spec=training_spec,
        expected_phase_program=training_spec.worker_execution.method_contract.phase_program,
        expected_slots=initial_slots,
        resume_slot_transform=resume_slot_transform,
        continuation_request=training_spec.checkpoint_progress.continuation,
        allow_new_lineage_override=(
            training_spec.checkpoint_progress.continuation is not None
        ),
    )
    return {
        "verified_resume": True,
        "checkpoint_root": str(checkpoint_root),
        "transaction_id": loaded.manifest.transaction_id,
        "completed_batches": continuation.completed_batches,
        "continuation_batches": continuation.continuation_batches,
    }


def _adaptive_runtime_template_inputs(
    args: argparse.Namespace,
    hps: TreeNamespace,
    continuation: LaunchContinuation,
) -> tuple[argparse.Namespace, TreeNamespace]:
    """Copy governed arguments and size runtime histories to the resumed segment."""

    runtime_template_args = copy.copy(args)
    runtime_template_hps = copy.copy(hps)
    if continuation.resume:
        runtime_template_args.n_train_batches = continuation.continuation_batches
        runtime_template_hps.n_batches_condition = continuation.continuation_batches
    return runtime_template_args, runtime_template_hps


def _run_adaptive_epsilon_native_from_context(
    context: RunSpecExecutionContext,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    from rlrmp.train.adaptive_epsilon_native import (
        AdaptiveEpsilonNativeRuntime,
        AdaptiveEpsilonExternalObjectiveLossService,
        _resume_slot_transform,
        attach_adaptive_epsilon_checkpoint_continuation,
        build_adaptive_epsilon_native_initial_slots,
    )

    args = context.args
    run_spec = context.run_spec
    run_spec_path = context.run_spec_path
    hps = context.hps
    if int(args.n_train_batches) < 1:
        raise ValueError("--n-train-batches must be positive for --full-train")
    if int(args.checkpoint_interval_batches) < 1:
        raise ValueError("--checkpoint-interval-batches must be positive")
    stop_after_batches = None if args.stop_after_batches is None else int(args.stop_after_batches)
    if stop_after_batches is not None:
        if stop_after_batches < 1:
            raise ValueError("--stop-after-batches must be positive when provided")
        if stop_after_batches > int(args.n_train_batches):
            raise ValueError("--stop-after-batches cannot exceed --n-train-batches")

    spec_result = _spec_result_from_execution_context(context)
    output_dir = mkdir_p(Path(args.output_dir))
    key_init, _key_train, _key_adversary = split_initial_keys(jr.PRNGKey(int(args.seed)))
    pair_for_graph = setup_task_model_pair(hps, key=key_init)
    runtime_graph_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair_for_graph.model)
    assert_runtime_graph_matches_training_spec(
        run_spec,
        graph_spec=runtime_graph_bundle.graph_spec,
    )
    training_spec = feedbax_training_run_spec_from_payload(run_spec)
    checkpoint_root = output_dir / "checkpoints"
    context, resume_native, continuation = _resolve_full_train_launch_context(
        context,
        checkpoint_root=checkpoint_root,
        stop_after_batches=stop_after_batches,
    )
    if continuation.resume:
        training_spec = attach_adaptive_epsilon_checkpoint_continuation(
            training_spec,
            source_completed_batches=continuation.completed_batches,
            target_total_batches=continuation.stop_target_batches,
        )
    args = context.args
    runtime_template_args, runtime_template_hps = _adaptive_runtime_template_inputs(
        args,
        hps,
        continuation,
    )
    initial_slots, runtime = build_adaptive_epsilon_native_initial_slots(
        # The optimizer builder consumes the typed method payload. Passing the
        # outer tracked recipe here hides lr_continuation_mode under
        # feedbax_training_run_spec and silently falls back to "continue".
        run_spec=training_spec,
        hps=runtime_template_hps,
        args=runtime_template_args,
        key=jr.PRNGKey(int(args.seed)),
    )
    started = time.perf_counter()
    execution = _execute_native_training_run_spec(
        training_spec,
        progress_phase="adaptive_epsilon_train_chunk",
        total_batches=int(args.n_train_batches),
        run_id=_cs_supervised_native_run_id(args, run_spec_path),
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        checkpoint_root=checkpoint_root,
        loss_service=AdaptiveEpsilonExternalObjectiveLossService(),
        training_spec_payload=run_spec.get(RLRMP_RUN_SPEC_PAYLOAD_KEY),
        training_spec_payload_kind=RUN_SPEC_KIND,
        training_spec_payload_schema_id=RUN_SPEC_SCHEMA_ID,
        training_spec_payload_schema_version=_run_spec_payload_schema_version(run_spec),
        training_spec_payload_ref=str(run_spec_path),
        resume=resume_native,
        resume_slot_transform=_resume_slot_transform(None),
        issues=[str(args.issue)],
    )
    training_duration_seconds = time.perf_counter() - started
    native_runtime = runtime.component("adaptive_epsilon")
    if not isinstance(native_runtime, AdaptiveEpsilonNativeRuntime):
        raise TypeError("adaptive_epsilon runtime context was not installed")
    return _materialize_adaptive_epsilon_native_result(
        context=context,
        spec_result={
            **spec_result,
            "training_manifest_path": str(execution.manifest_path),
        },
        runtime=native_runtime,
        final_slots=execution.final_slots,
        training_duration_seconds=training_duration_seconds,
        stop_after_batches=stop_after_batches,
        checkpoint_writes=execution.checkpoint_writes,
        volume_commit=volume_commit,
    )


def _run_policy_adversary_native_from_context(
    context: RunSpecExecutionContext,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    from rlrmp.train.policy_adversary_native import (
        PolicyAdversaryExternalObjectiveLossService,
        PolicyAdversaryNativeRuntime,
        _resume_slot_transform,
        build_policy_adversary_native_initial_slots,
    )

    args = context.args
    run_spec = context.run_spec
    run_spec_path = context.run_spec_path
    hps = context.hps
    if int(args.n_train_batches) < 1:
        raise ValueError("--n-train-batches must be positive for --full-train")
    if int(args.checkpoint_interval_batches) < 1:
        raise ValueError("--checkpoint-interval-batches must be positive")
    stop_after_batches = None if args.stop_after_batches is None else int(args.stop_after_batches)
    if stop_after_batches is not None:
        if stop_after_batches < 1:
            raise ValueError("--stop-after-batches must be positive when provided")
        if stop_after_batches > int(args.n_train_batches):
            raise ValueError("--stop-after-batches cannot exceed --n-train-batches")

    spec_result = _spec_result_from_execution_context(context)
    output_dir = mkdir_p(Path(args.output_dir))
    key_init, _key_train, _key_adversary = split_initial_keys(jr.PRNGKey(int(args.seed)))
    pair_for_graph = setup_task_model_pair(hps, key=key_init)
    runtime_graph_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair_for_graph.model)
    assert_runtime_graph_matches_training_spec(
        run_spec,
        graph_spec=runtime_graph_bundle.graph_spec,
    )
    training_spec = feedbax_training_run_spec_from_payload(run_spec)
    checkpoint_root = output_dir / "checkpoints"
    context, resume_native, _continuation = _resolve_full_train_launch_context(
        context,
        checkpoint_root=checkpoint_root,
        stop_after_batches=stop_after_batches,
    )
    args = context.args
    initial_slots, runtime = build_policy_adversary_native_initial_slots(
        run_spec=run_spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(int(args.seed)),
    )
    started = time.perf_counter()
    execution = _execute_native_training_run_spec(
        training_spec,
        progress_phase="policy_adversary_train_chunk",
        total_batches=int(args.n_train_batches),
        run_id=_cs_supervised_native_run_id(args, run_spec_path),
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        checkpoint_root=checkpoint_root,
        loss_service=PolicyAdversaryExternalObjectiveLossService(),
        training_spec_payload=run_spec.get(RLRMP_RUN_SPEC_PAYLOAD_KEY),
        training_spec_payload_kind=RUN_SPEC_KIND,
        training_spec_payload_schema_id=RUN_SPEC_SCHEMA_ID,
        training_spec_payload_schema_version=_run_spec_payload_schema_version(run_spec),
        training_spec_payload_ref=str(run_spec_path),
        resume=resume_native,
        resume_slot_transform=_resume_slot_transform(None),
        issues=[str(args.issue)],
    )
    training_duration_seconds = time.perf_counter() - started
    native_runtime = runtime.component("policy_adversary")
    if not isinstance(native_runtime, PolicyAdversaryNativeRuntime):
        raise TypeError("policy_adversary runtime context was not installed")
    return _materialize_policy_adversary_native_result(
        context=context,
        spec_result={
            **spec_result,
            "training_manifest_path": str(execution.manifest_path),
        },
        runtime=native_runtime,
        final_slots=execution.final_slots,
        training_duration_seconds=training_duration_seconds,
        stop_after_batches=stop_after_batches,
        checkpoint_writes=execution.checkpoint_writes,
        volume_commit=volume_commit,
    )


def _materialize_policy_adversary_native_result(
    *,
    context: RunSpecExecutionContext,
    spec_result: dict[str, Any],
    runtime: Any,
    final_slots: Mapping[str, Any],
    training_duration_seconds: float,
    stop_after_batches: int | None,
    checkpoint_writes: Sequence[Any],
    volume_commit: VolumeCommit | None,
) -> dict[str, Any]:
    from rlrmp.train.policy_adversary_native import (
        ADVERSARY_OPTIMIZER,
        ADVERSARY_POLICY,
        MODEL,
        OPTIMIZER,
        PRNG,
        _deserialize_pytree_slot_value,
    )

    args = context.args
    run_spec = context.run_spec
    run_spec_path = context.run_spec_path
    output_dir = mkdir_p(Path(args.output_dir))
    checkpoint_root = output_dir / "checkpoints"
    state = TrainingState(
        model=_deserialize_pytree_slot_value(
            final_slots[MODEL], runtime.model_template, slot=MODEL
        ),
        optimizer_state=_deserialize_pytree_slot_value(
            final_slots[OPTIMIZER],
            runtime.optimizer_template,
            slot=OPTIMIZER,
        ),
        completed_batches=int(final_slots[COMPLETED_BATCHES]),
        key=jnp.asarray(final_slots[PRNG], dtype=jnp.uint32),
        history=getattr(runtime, "history", None),
        adversary_policy=_deserialize_pytree_slot_value(
            final_slots[ADVERSARY_POLICY],
            runtime.adversary_policy_template,
            slot=ADVERSARY_POLICY,
        ),
        adversary_optimizer_state=_deserialize_pytree_slot_value(
            final_slots[ADVERSARY_OPTIMIZER],
            runtime.adversary_optimizer_template,
            slot=ADVERSARY_OPTIMIZER,
        ),
    )
    checkpoint_path = save_training_checkpoint(
        checkpoint_root,
        state,
        args=args,
        run_spec=run_spec,
        write_custody=False,
        custody_result=_latest_checkpoint_write(checkpoint_writes),
    )
    records = list(getattr(runtime, "records", []))
    history_chunk_dir = mkdir_p(output_dir / "history_chunks")
    chunks: list[dict[str, float | int | str | None]] = []
    per_chunk_duration = (
        training_duration_seconds / len(records)
        if records and training_duration_seconds > 0
        else None
    )
    for record in records:
        completed = int(record["completed_batches"])
        chunk_batches = int(record["chunk_batches"])
        history_chunk_path = history_chunk_dir / f"history_{completed:07d}.eqx"
        _save_pytree(history_chunk_path, record["history_chunk"])
        chunks.append(
            {
                "completed_batches": completed,
                "checkpoint": str(checkpoint_path)
                if completed == state.completed_batches
                else None,
                "history_chunk": str(history_chunk_path),
                "chunk_batches": chunk_batches,
                "duration_seconds": per_chunk_duration,
                "batches_per_second": (
                    chunk_batches / per_chunk_duration if per_chunk_duration else None
                ),
            }
        )
    final_model_path = output_dir / "trained_model.eqx"
    final_adversary_policy_path = output_dir / "trained_policy_adversary.eqx"
    final_history_path = output_dir / "training_history.eqx"
    final_summary_path = output_dir / "training_summary.json"
    _save_pytree(final_model_path, state.model, hyperparameters=run_spec)
    _save_pytree(final_adversary_policy_path, state.adversary_policy, hyperparameters=run_spec)
    if state.history is not None:
        _save_pytree(final_history_path, state.history)
    policy_diagnostic_chunks = [record["diagnostics"] for record in records]
    diagnostics_metadata = write_training_diagnostics_sidecar(
        output_dir,
        args=args,
        run_spec=run_spec,
        state=state,
        training_history_path=final_history_path,
        policy_adversary_diagnostic_chunks=policy_diagnostic_chunks,
    )
    stopped_for_gate = stop_after_batches is not None and state.completed_batches < int(
        args.n_train_batches
    )
    final_summary = {
        "schema_version": f"{SCHEMA_VERSION}.training.v1",
        "issue": str(args.issue),
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "stopped_early_for_checkpoint_gate": stopped_for_gate,
        "stopped_early_for_adaptive_epsilon_zero_adversary": False,
        "stop_reason": "checkpoint_gate_stop_after_batches" if stopped_for_gate else None,
        "stop_after_batches": stop_after_batches,
        "adaptive_epsilon_zero_adversary_guard": None,
        "training_duration_seconds": training_duration_seconds,
        "training_batches_per_second": (
            state.completed_batches / training_duration_seconds
            if training_duration_seconds > 0
            else None
        ),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": str(final_adversary_policy_path),
        "training_history_path": str(final_history_path) if state.history is not None else None,
        "run_spec_path": str(run_spec_path),
        "graph_spec_path": spec_result["graph_spec_path"],
        "training_diagnostics": diagnostics_metadata,
        "chunks": chunks,
    }
    _atomic_write_json(final_summary_path, final_summary)
    _commit_volume(volume_commit)
    return {
        **spec_result,
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": str(final_adversary_policy_path),
        "training_history_path": str(final_history_path) if state.history is not None else None,
        "training_summary_path": str(final_summary_path),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "completed_batches": state.completed_batches,
    }


def _materialize_adaptive_epsilon_native_result(
    *,
    context: RunSpecExecutionContext,
    spec_result: dict[str, Any],
    runtime: Any,
    final_slots: Mapping[str, Any],
    training_duration_seconds: float,
    stop_after_batches: int | None,
    checkpoint_writes: Sequence[Any],
    volume_commit: VolumeCommit | None,
) -> dict[str, Any]:
    from rlrmp.train.adaptive_epsilon_native import (
        ADAPTIVE_EPSILON_STATE,
        MODEL,
        OPTIMIZER,
        PRNG,
        ZERO_ADVERSARY_GUARD,
        _adaptive_state_from_slot,
        _deserialize_pytree_slot_value,
        _guard_from_slot,
    )

    args = context.args
    run_spec = context.run_spec
    run_spec_path = context.run_spec_path
    output_dir = mkdir_p(Path(args.output_dir))
    checkpoint_root = output_dir / "checkpoints"
    adaptive_state = _adaptive_state_from_slot(final_slots[ADAPTIVE_EPSILON_STATE])
    zero_guard = _guard_from_slot(final_slots[ZERO_ADVERSARY_GUARD])
    if adaptive_state is not None:
        adaptive_state = replace(adaptive_state, zero_adversary_guard=zero_guard)
    state = TrainingState(
        model=_deserialize_pytree_slot_value(
            final_slots[MODEL],
            runtime.model_template,
            slot=MODEL,
        ),
        optimizer_state=_deserialize_pytree_slot_value(
            final_slots[OPTIMIZER],
            runtime.optimizer_template,
            slot=OPTIMIZER,
        ),
        completed_batches=int(final_slots[COMPLETED_BATCHES]),
        key=jnp.asarray(final_slots[PRNG], dtype=jnp.uint32),
        history=getattr(runtime, "history", None),
        adaptive_epsilon_state=adaptive_state,
    )
    checkpoint_path = save_training_checkpoint(
        checkpoint_root,
        state,
        args=args,
        run_spec=run_spec,
        write_custody=False,
        custody_result=_latest_checkpoint_write(checkpoint_writes),
    )
    diagnostics_metadata = write_training_diagnostics_sidecar(
        output_dir,
        args=args,
        run_spec=run_spec,
        state=state,
        training_history_path=output_dir / "training_history.eqx",
        adaptive_epsilon_diagnostic_chunks=[
            record["diagnostics"] for record in getattr(runtime, "records", [])
        ],
    )
    final_model_path = output_dir / "trained_model.eqx"
    final_history_path = output_dir / "training_history.eqx"
    final_summary_path = output_dir / "training_summary.json"
    _save_pytree(final_model_path, state.model, hyperparameters=run_spec)
    if state.history is not None:
        _save_pytree(final_history_path, state.history)
    stopped_for_gate = stop_after_batches is not None and state.completed_batches < int(
        args.n_train_batches
    )
    stopped_for_zero = bool(zero_guard.get("should_stop")) and state.completed_batches < int(
        args.n_train_batches
    )
    stop_reason = None
    if stopped_for_zero:
        stop_reason = ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON
    elif stopped_for_gate:
        stop_reason = "checkpoint_gate_stop_after_batches"
    final_summary = {
        "schema_version": f"{SCHEMA_VERSION}.training.v1",
        "issue": str(args.issue),
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "stopped_early_for_checkpoint_gate": stopped_for_gate,
        "stopped_early_for_adaptive_epsilon_zero_adversary": stopped_for_zero,
        "stop_reason": stop_reason,
        "stop_after_batches": stop_after_batches,
        "adaptive_epsilon_zero_adversary_guard": zero_guard,
        "training_duration_seconds": training_duration_seconds,
        "training_batches_per_second": (
            state.completed_batches / training_duration_seconds
            if training_duration_seconds > 0
            else None
        ),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": None,
        "training_history_path": str(final_history_path) if state.history is not None else None,
        "run_spec_path": str(run_spec_path),
        "graph_spec_path": spec_result["graph_spec_path"],
        "training_diagnostics": diagnostics_metadata,
        "chunks": [
            {
                "completed_batches": state.completed_batches,
                "checkpoint": str(checkpoint_path),
                "history_chunk": None,
                "chunk_batches": state.completed_batches,
                "duration_seconds": training_duration_seconds,
                "batches_per_second": (
                    state.completed_batches / training_duration_seconds
                    if training_duration_seconds > 0
                    else None
                ),
            }
        ],
    }
    _atomic_write_json(final_summary_path, final_summary)
    _commit_volume(volume_commit)
    return {
        **spec_result,
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": None,
        "training_history_path": str(final_history_path) if state.history is not None else None,
        "training_summary_path": str(final_summary_path),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "completed_batches": state.completed_batches,
    }


def _spec_result_from_execution_context(context: RunSpecExecutionContext) -> dict[str, Any]:
    spec_dir = Path(context.args.spec_dir)
    graph_metadata = context.run_spec.get("feedbax_graph", {})
    graph_spec_path = graph_metadata.get("graph_spec_path")
    graph_manifest_path = graph_metadata.get("manifest_path")
    return {
        "run_spec_path": str(context.run_spec_path),
        "graph_spec_path": None
        if graph_spec_path is None
        else str(spec_dir / str(graph_spec_path)),
        "graph_manifest_path": None
        if graph_manifest_path is None
        else str(spec_dir / str(graph_manifest_path)),
        "training_manifest_path": None,
    }


def _checkpoint_writes_by_completed_batch(
    checkpoint_writes: Sequence[Any],
) -> dict[int, Any]:
    """Index Feedbax executor checkpoint writes by completed program step."""

    indexed: dict[int, Any] = {}
    for write in checkpoint_writes:
        coordinate = getattr(getattr(write, "manifest", None), "coordinate", None)
        completed = getattr(coordinate, "program_step", None)
        if completed is None:
            continue
        indexed[int(completed)] = write
    return indexed


def _latest_checkpoint_write(checkpoint_writes: Sequence[Any]) -> Any | None:
    return checkpoint_writes[-1] if checkpoint_writes else None


def _tree_global_norm(tree: Any) -> Any:
    leaves = [leaf for leaf in jt.leaves(tree) if eqx.is_array(leaf)]
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _trainable_parameter_tree(updates: Any, params: Any | None) -> Any:
    if params is None:
        return None
    return jt.map(
        lambda update, param: param if eqx.is_array(update) else None,
        updates,
        params,
        is_leaf=lambda x: x is None,
    )


def _empty_diagnostic_series(n_batches: int, *, dtype: Any = jnp.float32) -> Any:
    return jnp.full((n_batches,), jnp.nan, dtype=dtype)


def _gradient_diagnostics_transform(
    *,
    schedule: Callable[[Any], Any],
    n_batches: int,
    gradient_clip_norm: float | None,
) -> optax.GradientTransformation:
    """Return an Optax transform that records pre-clip gradient scalar diagnostics."""

    clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)

    def init_fn(_params: Any) -> GradientDiagnosticsState:
        return GradientDiagnosticsState(
            count=jnp.asarray(0, dtype=jnp.int32),
            gradient_norm_pre_clip=_empty_diagnostic_series(n_batches),
            gradient_clipped=jnp.zeros((n_batches,), dtype=bool),
            learning_rate=_empty_diagnostic_series(n_batches),
        )

    def update_fn(
        updates: Any,
        state: GradientDiagnosticsState,
        params: Any | None = None,
    ) -> tuple[Any, GradientDiagnosticsState]:
        del params
        index = jnp.minimum(state.count, max(n_batches - 1, 0))
        grad_norm = _tree_global_norm(updates)
        clipped = jnp.asarray(False) if clip_norm is None else grad_norm > clip_norm
        grad_norm = jnp.asarray(grad_norm, dtype=state.gradient_norm_pre_clip.dtype)
        learning_rate = jnp.asarray(schedule(state.count), dtype=state.learning_rate.dtype)
        new_state = GradientDiagnosticsState(
            count=state.count + jnp.asarray(1, dtype=jnp.int32),
            gradient_norm_pre_clip=state.gradient_norm_pre_clip.at[index].set(grad_norm),
            gradient_clipped=state.gradient_clipped.at[index].set(clipped),
            learning_rate=state.learning_rate.at[index].set(learning_rate),
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def _update_diagnostics_transform(*, n_batches: int) -> optax.GradientTransformation:
    """Return an Optax transform that records post-update scalar diagnostics."""

    def init_fn(_params: Any) -> UpdateDiagnosticsState:
        return UpdateDiagnosticsState(
            count=jnp.asarray(0, dtype=jnp.int32),
            update_norm=_empty_diagnostic_series(n_batches),
            parameter_norm=_empty_diagnostic_series(n_batches),
            update_parameter_norm_ratio=_empty_diagnostic_series(n_batches),
        )

    def update_fn(
        updates: Any,
        state: UpdateDiagnosticsState,
        params: Any | None = None,
    ) -> tuple[Any, UpdateDiagnosticsState]:
        index = jnp.minimum(state.count, max(n_batches - 1, 0))
        update_norm = _tree_global_norm(updates)
        parameter_tree = _trainable_parameter_tree(updates, params)
        parameter_norm = _tree_global_norm(parameter_tree)
        ratio = update_norm / jnp.maximum(parameter_norm, jnp.asarray(1e-12))
        update_norm = jnp.asarray(update_norm, dtype=state.update_norm.dtype)
        parameter_norm = jnp.asarray(parameter_norm, dtype=state.parameter_norm.dtype)
        ratio = jnp.asarray(ratio, dtype=state.update_parameter_norm_ratio.dtype)
        new_state = UpdateDiagnosticsState(
            count=state.count + jnp.asarray(1, dtype=jnp.int32),
            update_norm=state.update_norm.at[index].set(update_norm),
            parameter_norm=state.parameter_norm.at[index].set(parameter_norm),
            update_parameter_norm_ratio=state.update_parameter_norm_ratio.at[index].set(ratio),
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def _resize_optimizer_diagnostics_for_batches(optimizer_state: Any, n_batches: int) -> Any:
    """Resize host-side optimizer diagnostic buffers for cross-length checkpoint resume."""

    target_batches = max(0, int(n_batches))

    def resize_leaf(leaf: Any) -> Any:
        if isinstance(leaf, GradientDiagnosticsState):
            return GradientDiagnosticsState(
                count=leaf.count,
                gradient_norm_pre_clip=_resize_diagnostic_series(
                    leaf.gradient_norm_pre_clip,
                    target_batches,
                ),
                gradient_clipped=_resize_diagnostic_series(
                    leaf.gradient_clipped,
                    target_batches,
                ),
                learning_rate=_resize_diagnostic_series(leaf.learning_rate, target_batches),
            )
        if isinstance(leaf, UpdateDiagnosticsState):
            return UpdateDiagnosticsState(
                count=leaf.count,
                update_norm=_resize_diagnostic_series(leaf.update_norm, target_batches),
                parameter_norm=_resize_diagnostic_series(leaf.parameter_norm, target_batches),
                update_parameter_norm_ratio=_resize_diagnostic_series(
                    leaf.update_parameter_norm_ratio,
                    target_batches,
                ),
            )
        return leaf

    return jt.map(
        resize_leaf,
        optimizer_state,
        is_leaf=lambda leaf: isinstance(leaf, (GradientDiagnosticsState, UpdateDiagnosticsState)),
    )


def _cs_supervised_resume_slot_transform(
    *,
    transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Return the C&S checkpoint normalizer without changing horizon leaves.

    Feedbax owns declared batch-axis extension.  Resizing here would hide the
    source shape before Feedbax can verify the continuation declaration.
    """

    def normalize(slots: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(transform(slots) if transform is not None else slots)
        payload[TRAIN_LOSS] = 0.0
        return payload

    return normalize


def _resize_diagnostic_series(series: Any, n_batches: int) -> Any:
    target_batches = max(0, int(n_batches))
    values = jnp.asarray(series)
    if values.ndim < 1:
        return values
    current_batches = int(values.shape[-1])
    if current_batches == target_batches:
        return values
    if current_batches > target_batches:
        return values[..., :target_batches]
    pad_shape = (*values.shape[:-1], target_batches - current_batches)
    if jnp.issubdtype(values.dtype, jnp.floating):
        pad = jnp.full(pad_shape, jnp.nan, dtype=values.dtype)
    else:
        pad = jnp.zeros(pad_shape, dtype=values.dtype)
    return jnp.concatenate([values, pad], axis=-1)


def _build_optimizer(hps: TreeNamespace) -> optax.GradientTransformation:
    schedule = _learning_rate_schedule(hps)
    transforms = []
    if bool(getattr(hps, "training_diagnostics", True)):
        transforms.append(
            _gradient_diagnostics_transform(
                schedule=schedule,
                n_batches=int(hps.n_batches_condition),
                gradient_clip_norm=hps.gradient_clip_norm,
            )
        )
    if hps.gradient_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(hps.gradient_clip_norm)))
    transforms.append(
        optax.inject_hyperparams(partial(optax.adamw, weight_decay=float(hps.weight_decay)))(
            learning_rate=schedule
        )
    )
    if bool(getattr(hps, "training_diagnostics", True)):
        transforms.append(_update_diagnostics_transform(n_batches=int(hps.n_batches_condition)))
    return optax.chain(*transforms)


def _learning_rate_schedule(hps: TreeNamespace) -> Callable[[Any], Any]:
    """Return the controller learning-rate schedule declared by ``hps``."""

    schedule_name = str(getattr(hps, "lr_schedule", "delayed_cosine"))
    warmup_batches = int(getattr(hps, "constant_lr_iterations", 0))
    total_batches = int(hps.n_batches_condition)
    learning_rate = float(hps.learning_rate_0)
    warmup_init_fraction = float(getattr(hps, "warmup_init_fraction", 0.0))
    alpha = float(hps.cosine_annealing_alpha)
    if schedule_name == "warmup_cosine":
        if warmup_batches < 1:
            raise ValueError("warmup_cosine requires constant_lr_iterations >= 1")
        if warmup_batches >= total_batches:
            raise ValueError("warmup_cosine requires warmup batches < total batches")
        return optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * warmup_init_fraction,
            peak_value=learning_rate,
            warmup_steps=warmup_batches,
            decay_steps=total_batches,
            end_value=learning_rate * alpha,
        )
    if schedule_name == "delayed_cosine":
        return make_delayed_cosine_schedule(
            learning_rate,
            constant_steps=warmup_batches,
            total_steps=total_batches,
            alpha=alpha,
        )
    raise ValueError(f"Unsupported learning-rate schedule {schedule_name!r}")


def make_delayed_cosine_schedule(
    init_lr: float,
    constant_steps: int,
    total_steps: int,
    alpha: float = 0.001,
) -> optax.Schedule:
    """Return a constant-then-cosine learning-rate schedule."""

    return optax.join_schedules(
        schedules=[
            optax.constant_schedule(init_lr),
            optax.cosine_decay_schedule(
                init_value=init_lr,
                decay_steps=max(0, int(total_steps) - int(constant_steps)),
                alpha=alpha,
            ),
        ],
        boundaries=[int(constant_steps)],
    )


def _where_train() -> dict[int, Callable[[Any], tuple[Any, ...]]]:
    def where_train_fn(model):
        net = model.nodes["net"]
        return staged_network_trainable_parts(net)

    return {0: where_train_fn}


def _initial_training_state(
    *,
    model: Any,
    trainer: optax.GradientTransformation,
    where_train: Callable[[Any], Any],
    key: Any,
) -> TrainingState:
    where_train_spec = filter_spec_leaves(model, where_train)
    model_parameters = get_model_parameters(model, where_train_spec)
    optimizer_state = eqx.filter_vmap(trainer.init)(model_parameters)
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=0,
        key=key,
        history=None,
    )


def get_model_parameters(model: Any, where_train_spec: Any) -> Any:
    """Return trainable array parameters selected by a where-train spec."""

    return eqx.filter(eqx.filter(model, where_train_spec), eqx.is_array)


def _adaptive_epsilon_curriculum_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "adaptive_epsilon_curriculum", None), "enabled", False))


def _commit_volume(volume_commit: VolumeCommit | None) -> None:
    if volume_commit is not None:
        volume_commit()


def write_training_diagnostics_sidecar(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    run_spec: dict[str, Any],
    state: TrainingState,
    training_history_path: Path,
    optimizer_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
    history_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
    pgd_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
    policy_adversary_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
    adaptive_epsilon_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Write compact training-process scalar sidecars for future optimizer audits."""

    metadata = _training_diagnostics_metadata(args, output_dir)
    if not metadata["enabled"]:
        return metadata

    gradient_state = _find_diagnostics_state(
        state.optimizer_state,
        GradientDiagnosticsState,
    )
    update_state = _find_diagnostics_state(
        state.optimizer_state,
        UpdateDiagnosticsState,
    )
    arrays: dict[str, np.ndarray] = {
        "batch_index": np.arange(state.completed_batches, dtype=np.int64),
    }
    if optimizer_diagnostic_chunks:
        arrays.update(_combine_history_diagnostic_chunks(optimizer_diagnostic_chunks))
    elif gradient_state is not None:
        arrays.update(_gradient_diagnostics_arrays(gradient_state, state.completed_batches))
    if not optimizer_diagnostic_chunks and update_state is not None:
        arrays.update(_update_diagnostics_arrays(update_state, state.completed_batches))
    if history_diagnostic_chunks:
        arrays.update(_combine_history_diagnostic_chunks(history_diagnostic_chunks))
    elif state.history is not None:
        arrays.update(_history_diagnostics_arrays(state.history, state.completed_batches))
    if pgd_diagnostic_chunks:
        arrays.update(_combine_history_diagnostic_chunks(pgd_diagnostic_chunks))
    if policy_adversary_diagnostic_chunks:
        arrays.update(_combine_history_diagnostic_chunks(policy_adversary_diagnostic_chunks))
    if adaptive_epsilon_diagnostic_chunks:
        arrays.update(_combine_history_diagnostic_chunks(adaptive_epsilon_diagnostic_chunks))

    npz_path = Path(metadata["sidecar_path"])
    manifest_path = Path(metadata["manifest_path"])
    if bool(args.resume):
        arrays = _prepend_existing_training_diagnostics(
            npz_path,
            arrays,
            completed_batches=state.completed_batches,
        )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_name(f".{npz_path.name}.tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)

    manifest = {
        **metadata,
        "issue": str(args.issue),
        "run_spec_issue": run_spec.get("issue"),
        "completed_batches": int(state.completed_batches),
        "n_train_batches": int(args.n_train_batches),
        "gradient_clip_active": args.gradient_clip_norm is not None,
        "gradient_clip_norm": (
            None if args.gradient_clip_norm is None else float(args.gradient_clip_norm)
        ),
        "training_history_path": str(training_history_path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in sorted(arrays.items())
        },
    }
    _atomic_write_json(manifest_path, manifest)
    return {
        **metadata,
        "written": True,
        "array_count": len(arrays),
        "completed_batches": int(state.completed_batches),
    }


def _find_diagnostics_state(tree: Any, state_type: type) -> Any | None:
    if isinstance(tree, state_type):
        return tree
    if isinstance(tree, dict):
        for value in tree.values():
            found = _find_diagnostics_state(value, state_type)
            if found is not None:
                return found
    if isinstance(tree, tuple | list):
        for value in tree:
            found = _find_diagnostics_state(value, state_type)
            if found is not None:
                return found
    return None


def _diagnostic_series(array: Any, completed_batches: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 0:
        return values.reshape((1,))
    if values.ndim == 1:
        return values[:completed_batches]
    if values.shape[0] == completed_batches:
        return values[:completed_batches]
    return np.moveaxis(values[..., :completed_batches], -1, 0)


def _optimizer_diagnostic_series(array: Any, completed_batches: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 0:
        return values.reshape((1,))
    if values.ndim == 1:
        return values[:completed_batches]
    return np.moveaxis(values[..., :completed_batches], -1, 0)


def _optimizer_diagnostic_series_range(
    array: Any,
    start_batches: int,
    completed_batches: int,
) -> np.ndarray:
    return _optimizer_diagnostic_series(array, completed_batches)[start_batches:completed_batches]


def _gradient_diagnostics_arrays(
    state: GradientDiagnosticsState,
    completed_batches: int,
    *,
    start_batches: int = 0,
) -> dict[str, np.ndarray]:
    clipped = _optimizer_diagnostic_series_range(
        state.gradient_clipped,
        start_batches,
        completed_batches,
    ).astype(bool)
    arrays = {
        "optimizer_gradient_norm_pre_clip": _optimizer_diagnostic_series_range(
            state.gradient_norm_pre_clip,
            start_batches,
            completed_batches,
        ),
        "optimizer_gradient_clipped": clipped,
        "optimizer_learning_rate": _optimizer_diagnostic_series_range(
            state.learning_rate,
            start_batches,
            completed_batches,
        ),
    }
    clipped_float = clipped.astype(np.float32)
    if clipped_float.ndim == 1:
        arrays["optimizer_clipping_fraction"] = clipped_float
    else:
        arrays["optimizer_clipping_fraction"] = clipped_float.mean(
            axis=tuple(range(1, clipped_float.ndim))
        )
    return arrays


def _update_diagnostics_arrays(
    state: UpdateDiagnosticsState,
    completed_batches: int,
    *,
    start_batches: int = 0,
) -> dict[str, np.ndarray]:
    return {
        "optimizer_update_norm": _optimizer_diagnostic_series_range(
            state.update_norm,
            start_batches,
            completed_batches,
        ),
        "optimizer_parameter_norm": _optimizer_diagnostic_series_range(
            state.parameter_norm,
            start_batches,
            completed_batches,
        ),
        "optimizer_update_parameter_norm_ratio": _optimizer_diagnostic_series_range(
            state.update_parameter_norm_ratio,
            start_batches,
            completed_batches,
        ),
    }


def _emit_checkpoint_progress(
    history_chunk: Any,
    pgd_diagnostics: dict[str, np.ndarray],
    *,
    chunk_batches: int,
    completed_batches: int,
    total_batches: int,
    elapsed_seconds: float,
) -> None:
    """Emit one compact host-side progress line at a checkpoint boundary."""

    extras: dict[str, float | int] = {"completed": int(completed_batches)}
    loss_scalars = _latest_loss_scalars(history_chunk, chunk_batches=chunk_batches)
    loss = loss_scalars.pop("total", None)
    extras.update(loss_scalars)
    extras.update(_latest_pgd_progress_scalars(pgd_diagnostics))
    print(
        format_batch_line(
            "checkpoint",
            max(0, int(completed_batches) - 1),
            int(total_batches),
            loss=loss,
            elapsed=elapsed_seconds,
            **extras,
        ),
        flush=True,
    )


def _latest_loss_scalars(
    history_chunk: Any,
    *,
    chunk_batches: int,
    max_terms: int = 8,
) -> dict[str, float]:
    loss_tree = getattr(history_chunk, "loss", None)
    arrays = _loss_tree_arrays(
        loss_tree,
        prefix="train_loss",
        completed_batches=int(chunk_batches),
    )
    scalars: dict[str, float] = {}
    total = _latest_scalar(arrays.get("train_loss__total"))
    if total is not None:
        scalars["total"] = total
    terms: list[tuple[str, float]] = []
    for name, values in arrays.items():
        if name == "train_loss__total":
            continue
        value = _latest_scalar(values)
        if value is None:
            continue
        terms.append((f"loss_{name.removeprefix('train_loss__')}", value))
    for name, value in sorted(terms)[:max_terms]:
        scalars[name] = value
    return scalars


def _latest_pgd_progress_scalars(
    pgd_diagnostics: dict[str, np.ndarray],
) -> dict[str, float]:
    key_map = {
        "pgd_broad_epsilon_raw_task_loss_selected": "adv_task_loss",
        "pgd_broad_epsilon_energy_penalty_term_selected": "adv_penalty",
        "pgd_broad_epsilon_penalized_objective_selected": "adv_objective",
        "pgd_broad_epsilon_epsilon_energy_mean": "adv_energy",
        "pgd_broad_epsilon_selected_objective_gain_over_zero": "adv_gain",
        "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean": "adv_radius_ratio",
        "pgd_broad_epsilon_inner_objective_nonfinite_seen": "adv_nonfinite",
    }
    scalars: dict[str, float] = {}
    for source, target in key_map.items():
        value = _latest_scalar(pgd_diagnostics.get(source))
        if value is not None:
            scalars[target] = value
    return scalars


def _latest_scalar(values: Any) -> float | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.size == 0:
        return None
    latest = array.reshape((1,)) if array.ndim == 0 else array[-1]
    latest = np.asarray(latest, dtype=np.float64)
    finite = latest[np.isfinite(latest)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _history_diagnostics_arrays(history: Any, completed_batches: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    learning_rate = getattr(history, "learning_rate", None)
    if learning_rate is not None:
        arrays["history_learning_rate"] = _diagnostic_series(learning_rate, completed_batches)
    arrays.update(
        _loss_tree_arrays(
            getattr(history, "loss", None),
            prefix="train_loss",
            completed_batches=completed_batches,
        )
    )
    arrays.update(
        _loss_tree_arrays(
            getattr(history, "loss_validation", None),
            prefix="validation_loss",
            completed_batches=completed_batches,
        )
    )
    return arrays


def _combine_history_diagnostic_chunks(
    chunks: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    keys = sorted({key for chunk in chunks for key in chunk})
    combined: dict[str, np.ndarray] = {}
    for key in keys:
        pieces = [chunk[key] for chunk in chunks if key in chunk]
        if not pieces:
            continue
        combined[key] = np.concatenate(pieces, axis=0)
    return combined


def _axis_removed_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return (*shape[:axis], *shape[axis + 1 :])


def _slice_axis(array: np.ndarray, axis: int, start: int) -> np.ndarray:
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(start, None)
    return array[tuple(slices)]


def _has_time_axis(array: np.ndarray, completed_batches: int) -> bool:
    return any(dim == int(completed_batches) for dim in array.shape)


def _stitch_training_diagnostic_array(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    completed_batches: int,
) -> np.ndarray | None:
    """Stitch one diagnostic array while preserving its non-time axes.

    Resumed diagnostics can be time-major, e.g. ``(batch, replicate)``, or
    replicate-major, e.g. ``(replicate, batch)``.  Repeated checkpoint-cadence
    sidecar writes also mean ``previous`` may already contain a prefix of the
    in-memory continuation.  Infer the append axis from the only dimension pair
    that can produce ``completed_batches`` after removing any overlap.
    """

    completed = int(completed_batches)
    candidates: list[tuple[int, int, int, int]] = []
    for previous_axis in range(previous.ndim):
        previous_len = int(previous.shape[previous_axis])
        for current_axis in range(current.ndim):
            if _axis_removed_shape(previous.shape, previous_axis) != _axis_removed_shape(
                current.shape,
                current_axis,
            ):
                continue
            current_len = int(current.shape[current_axis])
            continuation_start = completed - current_len
            if continuation_start < 0:
                continue
            overlap = previous_len - continuation_start
            if 0 <= overlap <= current_len and previous_len + current_len - overlap == completed:
                same_axis = int(previous_axis != current_axis)
                nonzero_overlap = int(overlap > 0)
                candidates.append((same_axis, nonzero_overlap, previous_axis, current_axis))

    if not candidates:
        return None

    _, _, previous_axis, current_axis = sorted(candidates)[0]
    aligned_current = (
        current
        if current_axis == previous_axis
        else np.moveaxis(current, current_axis, previous_axis)
    )
    overlap = int(previous.shape[previous_axis]) - (
        int(completed_batches) - int(aligned_current.shape[previous_axis])
    )
    current_suffix = _slice_axis(aligned_current, previous_axis, max(0, overlap))
    if current_suffix.shape[previous_axis] == 0:
        return previous
    return np.concatenate([previous, current_suffix], axis=previous_axis)


def _prepend_existing_training_diagnostics(
    npz_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    completed_batches: int,
) -> dict[str, np.ndarray]:
    """Prepend prior sidecar arrays when a resumed run writes only new chunks."""

    if not npz_path.exists():
        return arrays
    with np.load(npz_path) as prior_npz:
        prior = {name: prior_npz[name] for name in prior_npz.files}
    stitched = dict(arrays)
    for name, current in arrays.items():
        if name == "batch_index" or current.ndim == 0:
            continue
        if _has_time_axis(current, completed_batches):
            continue
        previous = prior.get(name)
        if previous is None:
            logger.warning(
                "Cannot stitch resumed training diagnostics for %r: no prior array is "
                "available; writing continuation-only array with shape %s.",
                name,
                current.shape,
            )
            continue
        stitched_array = _stitch_training_diagnostic_array(
            previous,
            current,
            completed_batches=completed_batches,
        )
        if stitched_array is None:
            if _has_time_axis(previous, completed_batches):
                stitched[name] = previous
                continue
            logger.warning(
                "Cannot stitch resumed training diagnostics for %r: prior shape %s "
                "and current shape %s are not axis-compatible for %d completed batches; "
                "writing continuation-only array.",
                name,
                previous.shape,
                current.shape,
                int(completed_batches),
            )
            continue
        stitched[name] = stitched_array
    for name, previous in prior.items():
        if name in stitched or name == "batch_index" or previous.ndim == 0:
            continue
        if _has_time_axis(previous, completed_batches):
            stitched[name] = previous
    return stitched


def _loss_tree_arrays(
    loss_tree: Any,
    *,
    prefix: str,
    completed_batches: int,
) -> dict[str, np.ndarray]:
    if loss_tree is None:
        return {}
    arrays: dict[str, np.ndarray] = {}
    total = _loss_tree_total_array(loss_tree)
    if total is not None:
        arrays[f"{prefix}__total"] = _diagnostic_series(total, completed_batches)
    if not hasattr(loss_tree, "flatten"):
        return arrays
    for name, value in loss_tree.flatten().items():
        if not eqx.is_array(value):
            continue
        arrays[f"{prefix}__{_sanitize_array_name(str(name))}"] = _diagnostic_series(
            value,
            completed_batches,
        )
    return arrays


def _loss_tree_total_array(loss_tree: Any) -> Any | None:
    value = getattr(loss_tree, "value", None)
    weight = getattr(loss_tree, "weight", 1.0)
    if eqx.is_array(value):
        return value * weight
    children = getattr(loss_tree, "children", None)
    if not children:
        return None
    child_values = [
        child_total
        for child in children
        if (child_total := _loss_tree_total_array(child)) is not None
    ]
    if not child_values:
        return None
    total = child_values[0]
    for child_value in child_values[1:]:
        total = total + child_value
    return total * weight


def _sanitize_array_name(name: str) -> str:
    sanitized = [char if char.isalnum() else "_" for char in name]
    return "_".join("".join(sanitized).split("_"))


__all__ = [
    "CsSupervisedExternalObjectiveLoss",
    "CsSupervisedExternalObjectiveLossService",
    "CsSupervisedNativeChunkRecord",
    "CsSupervisedNativeRuntime",
    "DEFAULT_CHECKPOINT_INTERVAL_BATCHES",
    "DEFAULT_DELAYED_GO_CUE_MAX_STEP",
    "DEFAULT_DELAYED_GO_CUE_MIN_STEP",
    "DEFAULT_DELAYED_P_CATCH_TRIAL",
    "DEFAULT_OUTPUT_DIR",
    "GradientDiagnosticsState",
    "RunSpecExecutionContext",
    "UpdateDiagnosticsState",
    "VolumeCommit",
    "_adaptive_epsilon_curriculum_enabled",
    "_args_values_from_run_spec",
    "_axis_removed_shape",
    "_build_optimizer",
    "_checkpoint_writes_by_completed_batch",
    "_combine_history_diagnostic_chunks",
    "_commit_volume",
    "_config_default",
    "_cs_model_slot",
    "_cs_optimizer_slot",
    "_cs_supervised_native_run_id",
    "_cs_supervised_native_supported",
    "_cs_supervised_resume_slot_transform",
    "_diagnostic_series",
    "_dict_value",
    "_emit_checkpoint_progress",
    "_empty_diagnostic_series",
    "_family_amplitude",
    "_find_diagnostics_state",
    "_gradient_diagnostics_arrays",
    "_gradient_diagnostics_transform",
    "_has_time_axis",
    "_history_diagnostics_arrays",
    "_hps_from_run_spec",
    "_initial_training_state",
    "_latest_checkpoint_write",
    "_latest_loss_scalars",
    "_latest_pgd_progress_scalars",
    "_latest_scalar",
    "_learning_rate_schedule",
    "_loss_tree_arrays",
    "_loss_tree_total_array",
    "_materialize_adaptive_epsilon_native_result",
    "_materialize_cs_supervised_native_result",
    "_materialize_policy_adversary_native_result",
    "_native_resume_history_base",
    "_optimizer_diagnostic_series",
    "_optimizer_diagnostic_series_range",
    "_prepend_existing_training_diagnostics",
    "_pulse_value",
    "_resize_diagnostic_series",
    "_resize_optimizer_diagnostics_for_batches",
    "_resolve_full_train_launch_context",
    "_run_adaptive_epsilon_native_from_context",
    "_run_cs_supervised_native_from_context",
    "_run_full_training_from_context",
    "_run_policy_adversary_native_from_context",
    "_sanitize_array_name",
    "_slice_axis",
    "_spec_result_from_execution_context",
    "_stitch_training_diagnostic_array",
    "_trainable_parameter_tree",
    "_tree_global_norm",
    "_update_diagnostics_arrays",
    "_update_diagnostics_transform",
    "_validate_composed_training_spec_payload",
    "_where_train",
    "build_cs_supervised_native_initial_slots",
    "build_execution_context_from_spec",
    "get_model_parameters",
    "load_validated_run_spec",
    "logger",
    "make_delayed_cosine_schedule",
    "run_full_training",
    "verify_resume_from_context",
    "write_training_diagnostics_sidecar",
]
