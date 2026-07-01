"""Stochastic C&S-fidelity GRU run-spec construction and training.

This module prepares nominal, hold-free C&S-aligned GRU runs for issue
``30f2313``. The default CLI mode writes only the lightweight run spec and
GraphSpec; ``--full-train`` performs the explicitly launched training path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
import optax
from jax_cookbook import save as fbx_save
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.intervene.schedule import TimeSeriesParam
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.runtime.parameter_constraints import project_component_parameters
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule
from feedbax.training.trainer import get_model_parameters, init_task_trainer_history
from feedbax.tasks import (
    extract_timeseries_params,
    infer_n_steps,
    prepare_inputs,
    set_state_by_path,
    where_key_to_path,
)
from jax_cookbook.tree import array_set as tree_set
from jax_cookbook.tree import filter_spec_leaves

from rlrmp.analysis.math.cs_game_card import (
    INIT_POS,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
    TARGET_POS,
    build_canonical_game,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    default_cs_noise_covariances,
)
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.model.cs_lss_gru import CS_H0_CONTEXT_DIM, CS_H0_ENCODER_INIT
from rlrmp.model.feedbax_graph import (
    EXECUTION_BACKEND,
    GRAPH_PLANT_INTERVENOR_NODE,
    RLRMPFeedbaxGraphBundle,
    build_point_mass_sensorimotor_graph_spec,
    write_graph_spec_bundle,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_LOSS_OBJECTIVES,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.io import compact_json_dumps, write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path
from rlrmp.runtime.run_specs import validate_nominal_gru_run_spec
from rlrmp.model.stochastic_runtime import (
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_ADAM,
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
    BROAD_EPSILON_PGD_INNER_OPTIMIZER_METHODS,
    BROAD_EPSILON_PGD_MECHANISMS,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    BROAD_EPSILON_PGD_TRAINING_MODE,
    BROAD_EPSILON_TRAINING_MODE,
    HISTORICAL_020A65B_PGD_RADIUS_15CM,
    DEFAULT_TARGET_SUPPORT_PROFILE,
    LEGACY_PERTURBATION_TRAINING_MODE,
    PERTURBATION_TRAINING_MODE,
    POLICY_ADVERSARY_ENERGY_MODE,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_PLAIN_MODE,
    POLICY_ADVERSARY_POLICY_CLASSES,
    POLICY_ADVERSARY_TRAINING_MODE,
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
    TARGET_SUPPORT_PROFILES,
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
    BroadFullStateEpsilonTrainingConfig,
    FixedTargetPerturbationTrainingConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    config_from_broad_epsilon_pgd_hps,
    config_from_policy_adversary_hps,
    make_broad_epsilon_pgd_pre_step,
    make_policy_adversary,
    make_policy_adversary_pre_step,
    planned_33b0dcb_target_support_rows,
    planned_020a65b_h0_pgd_rows,
    planned_7c1f7ed_delayed_sisu_spectrum_rows,
    planned_e4800d6_sisu_spectrum_rows,
    planned_fixed_target_perturbation_rows,
    planned_target_relative_multitarget_h0_rows,
    planned_target_relative_multitarget_rows,
    policy_adversary_objective,
    run_broad_epsilon_pgd_inner_maximizer,
    target_relative_target_support_config,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.progress import batch_log_every, format_batch_line, should_log_batch
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
    LEGACY_CAUSAL_PLANT_BACKEND,
    setup_task_model_pair,
)
from rlrmp.model.trainable import staged_network_trainable_parts, staged_network_trainable_paths

ISSUE_ID = "30f2313"
SCHEMA_VERSION = "rlrmp.cs_stochastic_gru.v1"
DEFAULT_EXPERIMENT = ISSUE_ID
DEFAULT_RUN = "cs_stochastic_gru__no_hidden_penalty"
DEFAULT_OUTPUT_DIR = f"_artifacts/{DEFAULT_EXPERIMENT}/runs/{DEFAULT_RUN}"
DEFAULT_STOCHASTIC_PRESET = "cs2019-rollout"
DEFAULT_CHECKPOINT_INTERVAL_BATCHES = 500
CS_STAGE_COUNT = 60
CS_FEEDBAX_N_STEPS = CS_STAGE_COUNT + 1
CS_POSITION_SCALE = 1e6
CS_VELOCITY_SCALE = 1e5
CS_CONTROL_SCALE = 1.0
CS_REGULARIZED_NN_HIDDEN = 1e-5
CS_DELAYED_REACH_TASK_TYPE = "delayed_reach"
CS_DELAYED_REACH_TASK_PRESET = "delayed_center_out"
LEGACY_CS_DELAYED_REACH_TASK_TYPE = "cs_delayed_center_out_reach"
DELAYED_REACH_TRAINING_MODE = "delayed_reach_target_visible_go_cue"
DEFAULT_DELAYED_GO_CUE_MIN_STEP = 10
DEFAULT_DELAYED_GO_CUE_MAX_STEP = 30
DEFAULT_DELAYED_P_CATCH_TRIAL = 0.5
DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW = "canonical_window"
DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON = "flat_after_canonical_horizon"
DELAYED_MOVEMENT_COST_TAIL_MODES = (
    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
)
TRAINING_DIAGNOSTICS_NPZ = "training_diagnostics.npz"
TRAINING_DIAGNOSTICS_MANIFEST = "training_diagnostics.json"
VolumeCommit = Callable[[], None]


@dataclass(frozen=True)
class AdaptiveEpsilonState:
    """Host-side adaptive-lambda state for soft-energy direct-epsilon training."""

    lambda_value: float
    damage_ema: float | None = None
    last_update_batch: int | None = None
    update_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "lambda_value": float(self.lambda_value),
            "damage_ema": None if self.damage_ema is None else float(self.damage_ema),
            "last_update_batch": self.last_update_batch,
            "update_count": int(self.update_count),
        }


@dataclass(frozen=True)
class TrainingState:
    """Serializable state needed to resume the chunked C&S GRU training loop."""

    model: Any
    optimizer_state: Any
    completed_batches: int
    key: Any
    history: Any | None
    adversary_policy: Any | None = None
    adversary_optimizer_state: Any | None = None
    adaptive_epsilon_state: AdaptiveEpsilonState | None = None


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


@dataclass(frozen=True)
class StochasticPreset:
    """Named stochastic rollout preset for Feedbax-backed GRU runs."""

    name: str
    sensory_noise_std: float
    additive_motor_noise_std: float
    signal_dependent_motor_noise_std: float
    plant_process_force_noise_std: float
    source_contract: dict[str, Any]
    projection_notes: dict[str, str]

    def hps_fields(self) -> dict[str, float]:
        """Return model-hyperparameter fields controlled by the preset."""

        return {
            "sensory_noise_std": self.sensory_noise_std,
            "additive_motor_noise_std": self.additive_motor_noise_std,
            "signal_dependent_motor_noise_std": self.signal_dependent_motor_noise_std,
            "plant_process_force_noise_std": self.plant_process_force_noise_std,
        }

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable preset metadata."""

        return {
            "name": self.name,
            **self.hps_fields(),
            "source_contract": self.source_contract,
            "projection_notes": self.projection_notes,
        }


def stochastic_preset(name: str) -> StochasticPreset:
    """Return a named stochastic preset for the dedicated C&S GRU runner."""

    if name != DEFAULT_STOCHASTIC_PRESET:
        raise ValueError(
            f"Unknown stochastic preset {name!r}; expected {DEFAULT_STOCHASTIC_PRESET!r}"
        )
    plant, _schedule = build_canonical_game()
    output_config = OutputFeedbackConfig()
    noise_config = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG
    covariances = default_cs_noise_covariances(
        plant,
        output_config,
        motor_covariance_scale=noise_config.motor_covariance_scale,
        process_covariance_scale=noise_config.process_covariance_scale,
        signal_dependent_scale=noise_config.signal_dependent_scale,
    )
    sensory_diag = jnp.diag(covariances.sensory)
    if not bool(jnp.allclose(sensory_diag, sensory_diag[0])):
        raise ValueError("C&S sensory covariance projection expects isotropic diagonal covariance")
    return StochasticPreset(
        name=name,
        sensory_noise_std=float(jnp.sqrt(sensory_diag[0])),
        additive_motor_noise_std=math.sqrt(noise_config.motor_covariance_scale),
        signal_dependent_motor_noise_std=noise_config.signal_dependent_scale,
        plant_process_force_noise_std=math.sqrt(
            output_config.process_covariance_scale * noise_config.process_covariance_scale
        ),
        source_contract={
            **noise_config.summary(),
            "output_feedback_process_covariance_scale": output_config.process_covariance_scale,
            "sensory_noise_scale": output_config.sensory_noise_scale,
            "sensory_covariance_diag": [float(x) for x in sensory_diag.tolist()],
            "motor_covariance_shape": list(covariances.motor.shape),
            "process_covariance_shape": list(covariances.process.shape),
            "signal_dependent_state_shape": list(covariances.signal_dependent_state.shape),
        },
        projection_notes={
            "sensory": (
                "Use the C&S sensory covariance diagonal standard deviation on the "
                "Feedbax delayed pos/vel feedback channel."
            ),
            "additive_motor": (
                "Project C&S input-image motor covariance to command-channel "
                "additive noise with std sqrt(motor_covariance_scale)."
            ),
            "signal_dependent_motor": (
                "Use the C&S Csdn scale as Feedbax pre-force-filter multiplicative command noise."
            ),
            "plant_process": (
                "Project C&S process/load covariance to independent force noise "
                "immediately upstream of mechanics, after the force filter."
            ),
            "state_diffusion": "No arbitrary full-state diffusion is used in Feedbax GRU rollout.",
        },
    )


def derive_spec_dir(output_dir: Path) -> Path:
    """Return the tracked spec directory corresponding to an artifact directory."""

    out = Path(output_dir)
    artifact_root = REPO_ROOT / "_artifacts"
    spec_root = REPO_ROOT / "results"
    logical_out = out if out.is_absolute() else REPO_ROOT / out
    try:
        rel = logical_out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        pass
    resolved_out = out.resolve()
    try:
        rel = resolved_out.relative_to(artifact_root.resolve())
        return spec_root / rel
    except ValueError:
        return resolved_out.parent / f"{resolved_out.name}_spec"


def derive_spec_path(output_dir: Path) -> Path:
    """Return the canonical flat run-recipe file for an artifact directory.

    The recipe is written to ``results/<exp>/runs/<run>.json``. The sibling
    ``results/<exp>/runs/<run>/`` directory remains available for lightweight
    sidecars such as GraphSpec manifests.
    """

    sidecar_dir = derive_spec_dir(output_dir)
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = sidecar_dir.resolve().relative_to(spec_root)
    except ValueError:
        return sidecar_dir.parent / f"{sidecar_dir.name}.json"
    parts = rel.parts
    if len(parts) == 3 and parts[1] == "runs":
        return run_spec_path(parts[0], parts[2], for_write=True)
    return sidecar_dir.parent / f"{sidecar_dir.name}.json"


def _run_spec_path_for_write(*, output_dir: Path, spec_dir: Path, explicit_spec_dir: bool) -> Path:
    """Return the flat recipe path paired with ``spec_dir`` sidecars."""

    if explicit_spec_dir:
        return spec_dir.parent / f"{spec_dir.name}.json"
    return derive_spec_path(output_dir)


def _dump_json_metadata_bytes(file: Any, hyperparameters: dict[str, Any] | None) -> None:
    file.write(json.dumps(hyperparameters, sort_keys=True).encode("utf-8") + b"\n")


def _save_pytree(path: Path, tree: Any, *, hyperparameters: dict[str, Any] | None = None) -> None:
    fbx_save(
        path,
        tree,
        hyperparameters=hyperparameters,
        dump_fn=_dump_json_metadata_bytes,
    )


def resolve_run_spec_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Return executable CLI arguments replayed from a modern nominal-GRU run spec.

    Explicit CLI values override the checked-in run spec. This keeps replay useful
    for smoke gates that should write to a new output directory or stop after an
    intermediate checkpoint without modifying the durable historical run recipe.
    """

    run_spec_path = getattr(args, "run_spec", None)
    if run_spec_path is None:
        return args
    parser = parser or build_parser()
    defaults = parser.parse_args([])
    payload_path = Path(run_spec_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    validate_nominal_gru_run_spec(
        payload,
        spec_dir=payload_path.parent,
        require_graph_sidecars=False,
    )

    values = vars(defaults).copy()
    values.update(_args_values_from_run_spec(payload))
    for key, value in vars(args).items():
        if key == "run_spec":
            values[key] = value
            continue
        if value != getattr(defaults, key):
            values[key] = value
    return argparse.Namespace(**values)


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
    target_relative = _dict_value(hps, "target_relative_multitarget")
    target_distribution = _dict_value(target_relative, "target_distribution")
    delayed = _dict_value(hps, "delayed_reach")
    delayed_go = _dict_value(delayed, "go_cue_sampling")
    delayed_catch = _dict_value(delayed, "catch_trials")
    delayed_norm = _dict_value(loss, "delayed_trial_type_normalization")
    population = _dict_value(model, "population_structure")
    pgd_inner = _dict_value(broad_pgd, "inner_maximizer")

    return {
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
        "stop_after_batches": None,
        "smoke": False,
    }


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


def _delayed_reach_contract_from_args(
    *,
    enabled: bool,
    go_cue_min_step: int,
    go_cue_max_step: int,
    p_catch_trial: float,
    movement_cost_tail_mode: str = DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
) -> dict[str, Any]:
    """Return the delayed-reach task contract embedded in hps/run specs."""

    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "mode": DELAYED_REACH_TRAINING_MODE,
        "task_type": CS_DELAYED_REACH_TASK_TYPE,
        "task_preset": CS_DELAYED_REACH_TASK_PRESET,
        "legacy_task_type": LEGACY_CS_DELAYED_REACH_TASK_TYPE,
        "target_visibility": "visible_from_trial_start",
        "target_on_input": "not_used_target_always_visible",
        "go_cue_input": {
            "input_port": "input",
            "shape": [1],
            "sign": "0_during_prep_1_during_movement",
            "source": "1 - DelayedReachTaskInputs.hold",
        },
        "go_cue_sampling": {
            "min_step_inclusive": int(go_cue_min_step),
            "max_step_inclusive": int(go_cue_max_step),
            "distribution": "uniform_integer",
        },
        "catch_trials": {
            "p_catch_trial": float(p_catch_trial),
            "semantics": (
                "target remains visible, movement target is replaced by the initial "
                "position, and DelayedReachTaskInputs.hold stays 1 for the full trial"
            ),
            "go_cue_value": 0.0,
        },
        "movement_epoch": {
            "epoch_name": "movement",
            "epoch_index": 1,
            "source": "trial_specs.timeline.epoch_bounds[-2:]",
            "cs_schedule_horizon_steps": CS_STAGE_COUNT,
            "cost_indexing": "movement_age_not_trial_age",
            "cost_tail_mode": movement_cost_tail_mode,
            "cost_tail_semantics": (
                "score exactly the canonical 60 movement-age stages, then stop"
                if movement_cost_tail_mode == DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
                else (
                    "score canonical movement-age stages 0..59, then reuse stage 59 "
                    "Q/R weights through the remaining trial tail"
                )
            ),
        },
        "prep_epoch": {
            "epoch_name": "prep",
            "target_directed_movement_loss": "zero",
            "anti_anticipation": "nn_output_pre_go",
        },
        "pgd_mask": {
            "mode": "movement_epoch_only",
            "prep_support": "zero",
        },
        "multi_target_contract": "same structured target-relative target bank as non-delayed rows",
    }


def _delayed_reach_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "delayed_reach", None), "enabled", False))


def _resolve_auto_bool(value: bool | None, *, default: bool) -> bool:
    """Resolve tri-state CLI booleans that have context-dependent defaults."""

    return bool(default) if value is None else bool(value)


def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Build nominal C&S-aligned GRU hyperparameters from CLI arguments."""

    args = _apply_smoke_overrides(args)
    if (
        str(args.loss_objective)
        in {
            CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
        }
        and str(args.plant_backend) != CS_LSS_PLANT_BACKEND
    ):
        raise ValueError(
            f"--loss-objective {args.loss_objective} requires --plant-backend cs_lss "
            "because the full 48D C&S state is unavailable on the legacy backend."
        )
    if str(args.loss_objective) == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE and bool(
        args.regularized_fidelity
    ):
        raise ValueError(
            "--regularized-fidelity cannot be combined with "
            "--loss-objective full_analytical_qrf because nn_hidden is not an analytical "
            "Q/R/Q_f objective term."
        )
    no_integrator_state = bool(getattr(args, "no_integrator_state", False))
    if no_integrator_state and str(args.plant_backend) != CS_LSS_PLANT_BACKEND:
        raise ValueError("--no-integrator-state requires --plant-backend cs_lss.")
    plant, schedule = build_no_integrator_game() if no_integrator_state else build_canonical_game()
    preset = stochastic_preset(args.stochastic_preset)
    delayed_reach = bool(getattr(args, "delayed_reach", False))
    delayed_go_min = int(getattr(args, "delayed_reach_go_cue_min_step", 10))
    delayed_go_max = int(getattr(args, "delayed_reach_go_cue_max_step", 30))
    delayed_p_catch_trial = float(
        getattr(args, "delayed_reach_p_catch_trial", DEFAULT_DELAYED_P_CATCH_TRIAL)
    )
    delayed_movement_cost_tail_mode = str(
        getattr(
            args,
            "delayed_movement_cost_tail_mode",
            DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
        )
        or DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
    )
    if int(schedule.T) != CS_STAGE_COUNT:
        raise ValueError(f"Expected C&S stage count {CS_STAGE_COUNT}, got {schedule.T}")
    if delayed_go_min < 0 or delayed_go_max < delayed_go_min:
        raise ValueError(
            "--delayed-reach-go-cue-max-step must be >= --delayed-reach-go-cue-min-step >= 0"
        )
    if delayed_p_catch_trial < 0.0 or delayed_p_catch_trial > 1.0:
        raise ValueError("--delayed-reach-p-catch-trial must be between 0 and 1")
    if delayed_movement_cost_tail_mode not in DELAYED_MOVEMENT_COST_TAIL_MODES:
        raise ValueError(
            "--delayed-movement-cost-tail-mode must be one of: "
            + ", ".join(DELAYED_MOVEMENT_COST_TAIL_MODES)
        )
    if (
        delayed_movement_cost_tail_mode != DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
        and not delayed_reach
    ):
        raise ValueError("--delayed-movement-cost-tail-mode requires --delayed-reach.")
    delayed_trial_type_normalized_loss = bool(
        getattr(args, "delayed_reach_trial_type_normalized_loss", False)
    )
    if delayed_trial_type_normalized_loss and not delayed_reach:
        raise ValueError("--delayed-reach-trial-type-normalized-loss requires --delayed-reach.")
    if (
        delayed_trial_type_normalized_loss
        and str(args.loss_objective) != CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
    ):
        raise ValueError(
            "--delayed-reach-trial-type-normalized-loss requires "
            "--loss-objective full_analytical_qrf."
        )
    nn_hidden = CS_REGULARIZED_NN_HIDDEN if args.regularized_fidelity else 0.0
    nn_output_pre_go = (
        1.0
        if delayed_reach and getattr(args, "nn_output_pre_go", None) is None
        else float(getattr(args, "nn_output_pre_go", 0.0) or 0.0)
    )
    delayed_pre_go_force_filter_hold = float(
        getattr(args, "delayed_pre_go_force_filter_hold", 0.0) or 0.0
    )
    delayed_pre_go_start_pos_hold = float(
        getattr(args, "delayed_pre_go_start_pos_hold", 0.0) or 0.0
    )
    delayed_pre_go_start_pos_hold_norm = str(
        getattr(args, "delayed_pre_go_start_pos_hold_norm", "l2") or "l2"
    )
    if delayed_pre_go_start_pos_hold_norm not in {"l2", "l1"}:
        raise ValueError("--delayed-pre-go-start-pos-hold-norm must be one of: l2, l1")
    delayed_pre_go_zero_vel_hold = float(getattr(args, "delayed_pre_go_zero_vel_hold", 0.0) or 0.0)
    delayed_pre_go_aux_weights = {
        "delayed_pre_go_force_filter_hold": delayed_pre_go_force_filter_hold,
        "delayed_pre_go_start_pos_hold": delayed_pre_go_start_pos_hold,
        "delayed_pre_go_zero_vel_hold": delayed_pre_go_zero_vel_hold,
    }
    if any(weight != 0.0 for weight in delayed_pre_go_aux_weights.values()) and not delayed_reach:
        raise ValueError("Delayed pre-go hold penalties require --delayed-reach.")
    n_input_readout = int(args.hidden_size) - (
        int(args.n_input_only) + int(args.n_readout_only) + int(args.n_recurrent_only)
    )
    if n_input_readout < 0:
        raise ValueError(
            "Population subgroups exceed hidden_size: "
            f"hidden_size={args.hidden_size}, "
            f"n_input_only={args.n_input_only}, "
            f"n_readout_only={args.n_readout_only}, "
            f"n_recurrent_only={args.n_recurrent_only}"
        )
    force_filter_feedback = _resolve_auto_bool(
        getattr(args, "force_filter_feedback", None),
        default=delayed_reach,
    )
    perturbation_training_enabled = _resolve_auto_bool(
        getattr(args, "perturbation_training", None),
        default=delayed_reach,
    )
    perturbation_calibrated_timing = _resolve_auto_bool(
        getattr(args, "perturbation_calibrated_timing", None),
        default=delayed_reach and perturbation_training_enabled,
    )
    perturbation_movement_age_timing = _resolve_auto_bool(
        getattr(args, "perturbation_movement_age_timing", None),
        default=delayed_reach and perturbation_training_enabled and perturbation_calibrated_timing,
    )
    perturbation_physical_level = str(
        getattr(args, "perturbation_physical_level", None)
        or ("small" if delayed_reach else "moderate")
    )
    perturbation_training = FixedTargetPerturbationTrainingConfig(
        enabled=perturbation_training_enabled,
        nominal_fraction=float(args.perturbation_nominal_fraction),
        single_fraction=float(args.perturbation_single_fraction),
        combined_fraction=float(args.perturbation_combined_fraction),
        combined_amplitude_scale=float(args.perturbation_combined_amplitude_scale),
        initial_position_offset_m=float(args.perturbation_initial_position_offset_m),
        initial_velocity_offset_m_s=float(args.perturbation_initial_velocity_offset_m_s),
        process_epsilon_scale=float(args.perturbation_process_epsilon_scale),
        command_input_pulse_n=float(args.perturbation_command_input_pulse_n),
        sensory_feedback_offset_m=float(args.perturbation_sensory_feedback_offset_m),
        delayed_observation_offset_m=float(args.perturbation_delayed_observation_offset_m),
        pulse_start_step=int(args.perturbation_pulse_start_step),
        pulse_duration_steps=int(args.perturbation_pulse_duration_steps),
        calibrated_timing=perturbation_calibrated_timing,
        movement_age_timing=perturbation_movement_age_timing,
        physical_level=perturbation_physical_level,
        force_filter_feedback=force_filter_feedback,
        calibration_regime=str(args.perturbation_calibration_regime),
        closed_loop_calibration_table_path=args.perturbation_closed_loop_calibration_table,
    )
    if perturbation_movement_age_timing and not perturbation_calibrated_timing:
        raise ValueError(
            "--perturbation-movement-age-timing requires --perturbation-calibrated-timing."
        )
    broad_epsilon_training = BroadFullStateEpsilonTrainingConfig(
        enabled=bool(args.broad_epsilon_training),
        level=str(args.broad_epsilon_level),
        budget_scale=float(args.broad_epsilon_budget_scale),
        reach_length_scaling=bool(args.broad_epsilon_reach_scaling),
        movement_epoch_only=delayed_reach,
        epsilon_dim=int(plant.m_w),
    )
    broad_epsilon_pgd_training = PgdFullStateEpsilonTrainingConfig(
        enabled=bool(args.broad_epsilon_pgd_training),
        adversary_mechanism=str(args.broad_epsilon_pgd_mechanism),
        level=str(args.broad_epsilon_level),
        budget_scale=float(args.broad_epsilon_budget_scale),
        reach_length_scaling=bool(args.broad_epsilon_reach_scaling),
        n_steps=int(args.broad_epsilon_pgd_steps),
        step_size_fraction=float(args.broad_epsilon_pgd_step_size_fraction),
        inner_optimizer_method=str(args.broad_epsilon_pgd_inner_optimizer_method),
        adam_learning_rate=float(args.broad_epsilon_pgd_adam_lr),
        adam_b1=float(args.broad_epsilon_pgd_adam_b1),
        adam_b2=float(args.broad_epsilon_pgd_adam_b2),
        adam_eps=float(args.broad_epsilon_pgd_adam_eps),
        movement_epoch_only=delayed_reach,
        epsilon_dim=int(plant.m_w),
        budget_schedule=str(args.broad_epsilon_pgd_budget_schedule),
        sisu_condition_input=str(args.broad_epsilon_pgd_sisu_condition_input),
        sisu_max_l2_radius_15cm=args.broad_epsilon_pgd_sisu_max_radius,
        sisu_max_radius_source=args.broad_epsilon_pgd_sisu_max_radius_source,
        fixed_l2_radius_15cm=args.broad_epsilon_pgd_fixed_radius_15cm,
        fixed_radius_source=args.broad_epsilon_pgd_fixed_radius_source,
        objective_kind=str(args.broad_epsilon_pgd_objective),
        energy_gamma_star=args.broad_epsilon_pgd_energy_gamma_star,
        energy_gamma_factor=args.broad_epsilon_pgd_energy_gamma_factor,
        energy_gamma=args.broad_epsilon_pgd_energy_gamma,
        energy_penalty_scale=float(args.broad_epsilon_pgd_energy_penalty_scale),
        energy_lambda=args.broad_epsilon_pgd_energy_lambda,
        safety_cap_l2_radius_15cm=args.broad_epsilon_pgd_safety_cap_15cm,
        safety_cap_source=args.broad_epsilon_pgd_safety_cap_source,
    )
    adaptive_epsilon_curriculum = _adaptive_epsilon_curriculum_config_from_args(args)
    if adaptive_epsilon_curriculum["enabled"]:
        if not broad_epsilon_pgd_training.enabled:
            raise ValueError(
                "--adaptive-epsilon-curriculum requires --broad-epsilon-pgd-training."
            )
        if (
            broad_epsilon_pgd_training.adversary_mechanism
            != BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
        ):
            raise ValueError(
                "--adaptive-epsilon-curriculum currently applies only to direct_epsilon."
            )
        if broad_epsilon_pgd_training.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE:
            raise ValueError(
                "--adaptive-epsilon-curriculum requires --broad-epsilon-pgd-objective soft_energy."
            )
    policy_adversary_training = PolicyFullStateEpsilonTrainingConfig(
        enabled=bool(args.policy_adversary_training),
        policy_class=str(args.policy_adversary_policy_class),
        mode=str(args.policy_adversary_mode),
        width=int(args.policy_adversary_width),
        depth=int(args.policy_adversary_depth),
        n_steps=int(args.policy_adversary_steps),
        learning_rate=float(args.policy_adversary_lr),
        energy_penalty_gamma=float(args.policy_adversary_energy_gamma),
        reference_l2_radius_15cm=args.policy_adversary_radius_15cm,
        reach_length_scaling=bool(args.broad_epsilon_reach_scaling),
        movement_epoch_only=delayed_reach,
        epsilon_dim=int(plant.m_w),
        state_feature_dim=int(plant.n),
        budget_source=args.policy_adversary_radius_source,
    )
    if (
        delayed_reach
        and broad_epsilon_pgd_training.enabled
        and broad_epsilon_pgd_training.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
        and broad_epsilon_pgd_training.sisu_condition_input == "input"
    ):
        raise ValueError(
            "Delayed SISU-conditioned PGD must use --broad-epsilon-pgd-sisu-condition-input "
            "sisu (or auto) so the delayed go cue and SISU budget key are distinct."
        )
    target_relative_multitarget = target_relative_target_support_config(
        enabled=bool(args.target_relative_multitarget),
        force_filter_feedback=force_filter_feedback,
        profile=str(args.target_support_profile),
    )
    enabled_broad_lanes = [
        broad_epsilon_training.enabled,
        broad_epsilon_pgd_training.enabled,
        policy_adversary_training.enabled,
    ]
    if sum(bool(enabled) for enabled in enabled_broad_lanes) > 1:
        raise ValueError(
            "--broad-epsilon-training, --broad-epsilon-pgd-training, and "
            "--policy-adversary-training are separate broad-epsilon lanes and cannot "
            "be combined in the same row."
        )
    broad_epsilon_needs_target_relative = (
        (broad_epsilon_training.enabled and broad_epsilon_training.reach_length_scaling)
        or (broad_epsilon_pgd_training.enabled and broad_epsilon_pgd_training.reach_length_scaling)
        or (policy_adversary_training.enabled and policy_adversary_training.reach_length_scaling)
    )
    if broad_epsilon_needs_target_relative and not target_relative_multitarget.enabled:
        raise ValueError(
            "Reach-scaled broad-epsilon training requires --target-relative-multitarget "
            "so budgets are computed after explicit target sampling. For fixed-target "
            "scalar/SISU rows, use --no-broad-epsilon-reach-scaling."
        )
    if force_filter_feedback and not target_relative_multitarget.enabled:
        raise ValueError(
            "--force-filter-feedback requires --target-relative-multitarget because it "
            "extends the target-relative delayed feedback vector."
        )
    if delayed_reach and not target_relative_multitarget.enabled:
        raise ValueError(
            "--delayed-reach requires --target-relative-multitarget so the target remains "
            "visible from trial start through the documented controller input surface."
        )
    if delayed_reach and str(args.plant_backend) != CS_LSS_PLANT_BACKEND:
        raise ValueError("--delayed-reach currently requires --plant-backend cs_lss.")
    initial_hidden_encoder = bool(args.initial_hidden_encoder)
    if initial_hidden_encoder and not target_relative_multitarget.enabled:
        raise ValueError(
            "--initial-hidden-encoder currently requires --target-relative-multitarget so "
            "H0 is conditioned only on controller-visible target-relative feedback."
        )
    if delayed_reach and initial_hidden_encoder:
        raise ValueError(
            "--delayed-reach and --initial-hidden-encoder are separate task-contract lanes."
        )
    task_n_steps = CS_STAGE_COUNT + delayed_go_max if delayed_reach else CS_FEEDBAX_N_STEPS
    task_type = CS_DELAYED_REACH_TASK_TYPE if delayed_reach else "fixed_simple_reach"
    task_workspace = (
        [[-0.20, -0.20], [0.20, 0.20]]
        if delayed_reach
        else [[-0.02, -0.02], [float(TARGET_POS[0]) + 0.02, 0.02]]
    )
    hps_dict = {
        "method": "nominal-cs-gru",
        "dt": float(plant.dt),
        "n_batches_condition": int(args.n_train_batches),
        "n_batches_baseline": 0,
        "batch_size": int(args.batch_size),
        "learning_rate_0": float(args.controller_lr),
        "gradient_clip_norm": (
            None if args.gradient_clip_norm is None else float(args.gradient_clip_norm)
        ),
        "n_scaleup_batches": 0,
        "constant_lr_iterations": int(args.lr_warmup_batches),
        "warmup_init_fraction": float(args.lr_warmup_init_fraction),
        "cosine_annealing_alpha": float(args.lr_cosine_alpha),
        "lr_schedule": "warmup_cosine" if int(args.lr_warmup_batches) > 0 else "delayed_cosine",
        "weight_decay": 0.0,
        "training_diagnostics": _training_diagnostics_enabled(args),
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": int(args.n_replicates),
            "effector_mass": 1.0,
            "hidden_size": int(args.hidden_size),
            "feedback_delay_steps": 5,
            "feedback_noise_std": 0.0,
            "motor_noise_std": 0.0,
            **preset.hps_fields(),
            "stochastic_preset": preset.name,
            "plant_backend": str(args.plant_backend),
            "no_integrator_state": no_integrator_state,
            "state_dim": int(plant.n),
            "physical_state_dim": int(plant.m_w),
            "delay_blocks": int(plant.n // plant.m_w),
            "force_filter_feedback": force_filter_feedback,
            "initial_hidden_encoder": initial_hidden_encoder,
            "initial_hidden_encoder_config": _initial_hidden_encoder_config(
                enabled=initial_hidden_encoder,
                hidden_size=int(args.hidden_size),
                context_dim=6 if force_filter_feedback else CS_H0_CONTEXT_DIM,
                context_basis=(
                    "target_relative_delayed_feedback_plus_force_filter"
                    if force_filter_feedback
                    else "target_relative_delayed_feedback"
                ),
            ),
            "damping": 0.1,
            "tau_rise": 0.066,
            "population_structure": {
                "n_input_only": int(args.n_input_only),
                "n_readout_only": int(args.n_readout_only),
                "n_recurrent_only": int(args.n_recurrent_only),
                "n_input_readout": n_input_readout,
            },
        },
        "task": {
            "type": task_type,
            "preset": CS_DELAYED_REACH_TASK_PRESET if delayed_reach else None,
            "n_steps": task_n_steps,
            "n_control_stages": task_n_steps - 1 if delayed_reach else None,
            "workspace": task_workspace,
            "fixed_init_pos": (None if delayed_reach else [float(x) for x in INIT_POS.tolist()]),
            "fixed_target_pos": (
                None if delayed_reach else [float(x) for x in TARGET_POS.tolist()]
            ),
            "eval_grid_n": 1,
            "eval_n_directions": 1,
            "eval_reach_length": float(TARGET_POS[0]),
            "epoch_len_ranges": (
                [[delayed_go_min, delayed_go_max + 1]]
                if delayed_reach
                else [[0, 1], [CS_STAGE_COUNT, CS_STAGE_COUNT + 1]]
            ),
            "target_on_epochs": [0, 1] if delayed_reach else [0],
            "hold_epochs": [0] if delayed_reach else [],
            "move_epochs": [1] if delayed_reach else [0],
            "p_catch_trial": delayed_p_catch_trial if delayed_reach else 0.0,
            "target_visible_from_start": True if delayed_reach else None,
            "go_cue_event_name": "go_cue" if delayed_reach else None,
            "catch_metadata_policy": "flag" if delayed_reach else None,
        },
        "delayed_reach": _delayed_reach_contract_from_args(
            enabled=delayed_reach,
            go_cue_min_step=delayed_go_min,
            go_cue_max_step=delayed_go_max,
            p_catch_trial=delayed_p_catch_trial if delayed_reach else 0.0,
            movement_cost_tail_mode=delayed_movement_cost_tail_mode,
        ),
        "pert": {
            "type": "gusts",
            "std": 0.0,
            "duration_mean": 0,
            "n_expected": 0,
        },
        "perturbation_training": perturbation_training.to_hps_dict(),
        "broad_epsilon_training": broad_epsilon_training.to_hps_dict(),
        "broad_epsilon_pgd_training": broad_epsilon_pgd_training.to_hps_dict(),
        "adaptive_epsilon_curriculum": adaptive_epsilon_curriculum,
        "policy_adversary_training": policy_adversary_training.to_hps_dict(),
        "target_relative_multitarget": target_relative_multitarget.to_hps_dict(),
        "loss": {
            "objective": str(args.loss_objective),
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": float(args.effector_pos_running),
                "effector_vel_running": float(args.effector_vel_running),
                "effector_terminal_pos": float(args.effector_terminal_pos),
                "effector_terminal_vel": float(args.effector_terminal_vel),
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.0,
                "effector_vel_late": 0.0,
                "effector_hold_pos": 0.0,
                "effector_hold_vel": 0.0,
                "effector_final_vel": float(args.effector_final_vel),
                "nn_output": float(args.nn_output),
                "nn_hidden": nn_hidden,
                "nn_hidden_derivative": 0.0,
                "nn_output_jerk": float(args.nn_output_jerk),
                "nn_output_pre_go": nn_output_pre_go,
                "delayed_pre_go_force_filter_hold": delayed_pre_go_force_filter_hold,
                "delayed_pre_go_start_pos_hold": delayed_pre_go_start_pos_hold,
                "delayed_pre_go_zero_vel_hold": delayed_pre_go_zero_vel_hold,
                "nn_hidden_derivative_pre_go": 0.0,
                "mechanics_force_filter": (
                    1.0 / float(schedule.Q.shape[-1] // 8)
                    if str(args.loss_objective) == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE
                    else 0.0
                ),
            },
            "delayed_pre_go_start_pos_hold_norm": delayed_pre_go_start_pos_hold_norm,
            "effector_pos_late": {
                "start_step_after_go": int(schedule.T),
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": int(schedule.T),
                "final_scale_factor": 1.0,
            },
            "effector_pos_running_schedule": "cs_eq15_power6",
            "delayed_movement_cost_tail_mode": delayed_movement_cost_tail_mode,
            "effector_hold_pos_schedule": "disabled",
            "position_powerlaw_power": 6.0,
            "movement_ramp_shape": "none",
            "movement_ramp_duration_steps": 0,
            "movement_ramp_power": 1.0,
            "delayed_trial_type_normalization": {
                "enabled": delayed_trial_type_normalized_loss,
                "no_catch_weight": float(args.delayed_reach_no_catch_qrf_weight),
                "catch_weight": float(args.delayed_reach_catch_qrf_weight),
                "semantics": (
                    "When enabled, split full_analytical_qrf into no-catch and catch "
                    "terms, normalize each over its selected trial type, then combine "
                    "with explicit weights so p_catch controls sampling rather than "
                    "implicit objective dilution. This is an RLRMP bridge pending "
                    "Feedbax grouped reductions from Mandible issue 69d8d76."
                ),
            },
        },
        "loss_update": {
            "enabled": False,
            "target_ratio": 0.0,
            "alpha": 0.0,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_vel_running"],
            "start_iteration": 0,
        },
        "where": {
            0: (
                ["nodes.net.hidden", "nodes.net.readout", "nodes.net.h0_encoder"]
                if initial_hidden_encoder
                else ["nodes.net.hidden", "nodes.net.readout"]
            ),
        },
        "hidden_type": eqx.nn.GRUCell,
        "sisu_gating": "additive",
    }
    return dict_to_namespace(hps_dict, to_type=TreeNamespace)


def build_game_card_provenance() -> dict[str, Any]:
    """Return lightweight C&S game-card provenance without solving Riccati systems."""

    plant, schedule = build_canonical_game()
    target = [float(x) for x in TARGET_POS.tolist()]
    init = [float(x) for x in INIT_POS.tolist()]
    return {
        "source_module": "rlrmp.analysis.math.cs_game_card",
        "canonical_builder": "build_canonical_game",
        "discretization": plant.discretization,
        "dt": float(plant.dt),
        "horizon_steps": int(schedule.T),
        "feedbax_task_n_steps": CS_FEEDBAX_N_STEPS,
        "feedbax_control_cost_stages": CS_STAGE_COUNT,
        "init_pos_m": init,
        "target_pos_m": target,
        "target_distance_m": float(TARGET_POS[0]),
        "hold_free": True,
        "single_reach": True,
        "plant": {
            "state_dim": int(plant.n),
            "control_dim": int(plant.m_u),
            "disturbance_dim": int(plant.m_w),
            "delay_steps": 5,
            "physical_state_dim": 8,
            "bw_shape": list(plant.Bw.shape),
            "bw_contract": "top physical 8x8 block is identity; lag rows are zero",
            "mass": 1.0,
            "damping": 0.1,
            "tau": 0.066,
        },
        "cost": {
            "schedule": "C&S Eq. 15 physical 8-state schedule with 5-step delay distribution",
            "position_weight": "fact_t * 1e6",
            "velocity_weight": "fact_t * 1e5",
            "force_and_integrator_weight": "1.0",
            "fact_t": "((t + 1) / T)^6, capped at 1",
            "R": "I_2",
            "terminal_Q_f": "diag([1e6, 1e6, 1e5, 1e5, 1, 1, 1, 1]) on physical state",
            "feedbax_force_filter_state_cost": "not_available",
            "feedbax_force_filter_state_cost_note": (
                "The GRU task metadata records the analytical force/integrator "
                "cost, but the current Feedbax loss only exposes clean effector "
                "position, velocity, and efferent-output terms here; no synthetic "
                "force/filter-state cost is added."
            ),
        },
        "output_feedback_certificate_gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "output_feedback_gamma_selection_issue": OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
        "gamma_usage": (
            "Recorded for C&S provenance only. This nominal run has no robust/minimax "
            "adversarial phase and does not claim a certificate pass."
        ),
    }


def build_loss_game_card_provenance(hps: TreeNamespace) -> dict[str, Any]:
    """Return game-card provenance with objective-specific loss notes."""

    card = build_game_card_provenance()
    no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
    if no_integrator_state:
        card["canonical_builder"] = "build_no_integrator_game"
        card["comparator_variant"] = {
            "enabled": True,
            "name": "no_integrator_state",
            "canonical_cs2019_fidelity": False,
            "omitted_coordinates": ["eps_x_int", "eps_y_int"],
        }
        card["plant"] = {
            **card["plant"],
            "state_dim": int(getattr(hps.model, "state_dim", 36)),
            "disturbance_dim": int(getattr(hps.model, "physical_state_dim", 6)),
            "physical_state_dim": int(getattr(hps.model, "physical_state_dim", 6)),
            "bw_shape": [
                int(getattr(hps.model, "state_dim", 36)),
                int(getattr(hps.model, "physical_state_dim", 6)),
            ],
            "bw_contract": "top physical 6x6 block is identity; lag rows are zero",
        }
        card["cost"] = {
            **card["cost"],
            "schedule": "C&S Eq. 15 physical 6-state schedule with 5-step delay distribution",
            "force_and_integrator_weight": "force/filter entries only; integrator entries omitted",
            "terminal_Q_f": "diag([1e6, 1e6, 1e5, 1e5, 1, 1]) on physical state",
        }
    if _delayed_reach_enabled(hps):
        card["delayed_reach_projection"] = {
            "enabled": True,
            "rollout_control_stages": int(hps.task.n_steps),
            "canonical_cs_movement_horizon_steps": CS_STAGE_COUNT,
            "cost_indexing": "canonical Q/R/Q_f schedule starts at sampled go cue",
            "cost_tail_mode": str(hps.loss.delayed_movement_cost_tail_mode),
            "prep_epoch": "not part of canonical movement-stage C&S cost",
        }
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    if objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE:
        card["cost"] = {
            **card["cost"],
            "feedbax_force_filter_state_cost": "included_as_partial_ablation_running_term",
            "feedbax_force_filter_state_cost_note": (
                "This ablation preserves the historical partial Feedbax position/velocity "
                "terms, moves command cost to intended net.output, and adds a running "
                "force/filter state penalty over mechanics.vector force coordinates. It "
                "still omits disturbance-integrator state and terminal full-state Q_f costs."
            ),
        }
    elif objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        card["cost"] = {
            **card["cost"],
            "feedbax_force_filter_state_cost": "included_via_full_qrf",
            "feedbax_force_filter_state_cost_note": (
                "Full analytical Q/R/Q_f loss scores force/filter state through the "
                "delay-augmented Q_t and Q_f matrices."
                if no_integrator_state
                else (
                    "Full analytical Q/R/Q_f loss scores force/filter and "
                    "disturbance-integrator state through the canonical delay-augmented "
                    "C&S Q_t and Q_f matrices."
                )
            ),
        }
    return card


def build_model_structure_summary(hps: TreeNamespace) -> dict[str, Any]:
    """Return the model/training summary embedded in ``run.json``."""

    pop = hps.model.population_structure
    stochastic_runtime = _stochastic_runtime_contract(hps)
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    exact_lss = plant_backend == CS_LSS_PLANT_BACKEND
    h0 = _initial_hidden_encoder_metadata(hps)
    delayed_reach = _delayed_reach_enabled(hps)
    physical_state_dim = int(getattr(hps.model, "physical_state_dim", 8))
    state_dim = int(getattr(hps.model, "state_dim", 48))
    no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
    go_cue_dim = 1 if delayed_reach else 0
    sisu_condition_input = _sisu_conditioned_pgd_input_key(hps)
    sisu_condition_dim = 1 if sisu_condition_input is not None else 0
    return {
        "controller_kind": "gru",
        "plant_backend": plant_backend,
        "plant_backend_warning": (
            "legacy causal SimpleFeedback has a same-step force-filter-to-mechanics timing problem"
            if plant_backend == LEGACY_CAUSAL_PLANT_BACKEND
            else None
        ),
        "exact_cs_linear_state_space": exact_lss,
        "no_integrator_state": no_integrator_state,
        "state_dim": state_dim,
        "physical_state_dim": physical_state_dim,
        "fixed_plant_parameters": (
            ["nodes.mechanics.A", "nodes.mechanics.B", "nodes.mechanics.B_w"] if exact_lss else []
        ),
        "hidden_size": int(hps.model.hidden_size),
        "n_replicates": int(hps.model.n_replicates),
        "trainable": staged_network_trainable_paths(
            sisu_gating=str(getattr(hps, "sisu_gating", "additive")),
            initial_hidden_encoder=bool(h0["enabled"]),
        ),
        "initial_hidden_encoder": h0,
        "population_structure": {
            "n_input_only": int(pop.n_input_only),
            "n_readout_only": int(pop.n_readout_only),
            "n_recurrent_only": int(pop.n_recurrent_only),
            "n_input_readout": int(pop.n_input_readout),
        },
        "feedback": {
            "delay_steps": int(hps.model.feedback_delay_steps),
            "basis": _controller_feedback_basis(hps),
            "dimension": _controller_feedback_dim(hps),
            "noise_std": stochastic_runtime["sensory_noise_std"],
            "noise_role": "sensory_feedback",
            "noise_timing": (
                "Feedbax sensory Channel after delayed LSS feedback selector"
                if exact_lss
                else "Feedbax feedback Channel before controller"
            ),
            "delay_source": (
                f"C&S {state_dim}D LinearStateSpace delay-augmented state"
                if exact_lss
                else "Feedbax feedback Channel queue"
            ),
        },
        "go_cue": {
            "enabled": delayed_reach,
            "input_port": "input" if delayed_reach else None,
            "dimension": go_cue_dim,
            "sign": "0_during_prep_1_during_movement" if delayed_reach else None,
            "controller_input_index": 0 if delayed_reach else None,
        },
        "sisu_conditioning": {
            "enabled": sisu_condition_input is not None,
            "input_key": sisu_condition_input,
            "controller_input_port": "input" if sisu_condition_input is not None else None,
            "controller_input_index": 1 if delayed_reach and sisu_condition_input else 0,
            "budget_role": "pgd_energy_fraction" if sisu_condition_input is not None else None,
        },
        "controller_input_dimension": (
            _controller_feedback_dim(hps) + go_cue_dim + sisu_condition_dim
        ),
        "efferent": {
            "additive_motor_noise_std": stochastic_runtime["additive_motor_noise_std"],
            "signal_dependent_motor_noise_std": (
                stochastic_runtime["signal_dependent_motor_noise_std"]
            ),
            "noise_timing": (
                "Feedbax efferent Channel immediately before LinearStateSpace.force"
                if exact_lss
                else "pre_force_filter"
            ),
            "force_filter_tau_s": float(hps.model.tau_rise),
            "force_filter_role": (
                "coupled inside C&S LinearStateSpace dynamics"
                if exact_lss
                else "separate Feedbax FirstOrderFilter node"
            ),
        },
        "plant_process": {
            "force_noise_std": stochastic_runtime["plant_process_force_noise_std"],
            "noise_timing": (
                "mechanics.epsilon_sampled_task_input"
                if exact_lss
                else "post_force_filter_pre_mechanics"
            ),
            "state_diffusion": "mechanics.epsilon" if exact_lss else "not_used",
            "epsilon_bridge": (
                "sampled physical-process/load epsilon Task input bound to C&S "
                "LinearStateSpace mechanics.epsilon using the physical block of "
                "the released C&S process covariance"
                if exact_lss
                else "not_used"
            ),
        },
        "stochastic_runtime": stochastic_runtime,
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "nominal_only": _nominal_only(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "delayed_reach": _plain(hps.delayed_reach),
        "adversarial_phase": _adversarial_phase(hps),
        "certificate_lens": "input_output_map_certificate",
        "certificate_coordinate_claim": "not_same_coordinate_gain",
        "analytical_delay_augmented_state_input": False,
        "certificate_claim": (
            "I/O map certificate framing only; the output-feedback GRU is not fed "
            "the 48D delay-augmented analytical state and is not claimed to share "
            "same-coordinate gains with the analytical controller."
        ),
    }


def build_graph_bundle(hps: TreeNamespace) -> RLRMPFeedbaxGraphBundle:
    """Build the GraphSpec bundle for the nominal GRU run."""

    graph_spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        controller_kind="gru",
        intervention_type="FixedField",
    )
    task_spec = _task_spec(hps)
    loss_spec = _loss_spec(hps)
    training_spec = {
        "dt": float(hps.dt),
        "batch_size": int(hps.batch_size),
        "n_replicates": int(hps.model.n_replicates),
        "controller_kind": "gru",
        "plant_backend": str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)),
        "trainable": staged_network_trainable_paths(
            sisu_gating=str(getattr(hps, "sisu_gating", "additive")),
            initial_hidden_encoder=_initial_hidden_encoder_enabled(hps),
        ),
        "method": str(hps.method),
        "nominal_only": _nominal_only(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "adversarial_phase": _adversarial_phase(hps),
        "certificate_lens": "input_output_map_certificate",
        "analytical_delay_augmented_state_input": False,
        "stochastic_runtime": _stochastic_runtime_contract(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "loss_objective": str(hps.loss.objective),
        "initial_hidden_encoder": _initial_hidden_encoder_metadata(hps),
    }
    model_structure = build_model_structure_summary(hps)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "provenance_refs": {
            "delayed_reach": "$.delayed_reach",
            "loss_objective": "$.loss_objective",
            "model_structure.delayed_reach": "$.delayed_reach",
            "model_structure.stochastic_preset": "$.stochastic_preset",
            "model_structure.stochastic_runtime": "$.stochastic_runtime",
            "model_structure.training_distribution": "$.training_spec.training_distribution",
            "stochastic_preset": "$.stochastic_preset",
            "stochastic_runtime": "$.stochastic_runtime",
            "training_distribution": "$.training_spec.training_distribution",
        },
        "component_policy": {
            "rlrmp_component_types": [
                "RLRMPSimpleStagedNetwork",
                "FixedField",
            ],
            "feedbax_native_component_types": [
                "FeedbackChannels",
                "PointMass",
                "Channel",
            ],
            "nominal_intervention_policy": (
                f"{GRAPH_PLANT_INTERVENOR_NODE} is present only as an inactive legacy "
                "GraphSpec compatibility component; no robust/minimax adversary is scheduled."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.train.task_model.setup_task_model_pair",
            "checkpoint_format": "jax_cookbook.save/load_with_hyperparameters",
        },
        "task_spec": task_spec,
        "loss_spec": loss_spec,
        "training_spec": training_spec,
        "game_card_provenance": build_loss_game_card_provenance(hps),
        "model_structure": model_structure,
        "delayed_reach": _plain(hps.delayed_reach),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "stochastic_runtime": _stochastic_runtime_contract(hps),
        "loss_objective": str(hps.loss.objective),
    }
    return RLRMPFeedbaxGraphBundle(
        graph_spec=graph_spec,
        task_spec=task_spec,
        loss_spec=loss_spec,
        training_spec=training_spec,
        manifest=manifest,
    )


def _should_write_graph_spec(hps: TreeNamespace) -> bool:
    return str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)) != CS_LSS_PLANT_BACKEND


def _write_graph_bundle_for_backend(
    hps: TreeNamespace,
    graph_bundle: RLRMPFeedbaxGraphBundle,
    spec_dir: Path,
) -> Path | None:
    manifest_path = spec_dir / "model.graph.manifest.json"
    if _should_write_graph_spec(hps):
        return write_graph_spec_bundle(graph_bundle, spec_dir)
    manifest = {
        **graph_bundle.manifest,
        "graph_export": {
            "status": "unavailable",
            "reason": (
                "C&S cs_lss runs use LinearStateSpace mechanics and delayed "
                "position/velocity feedback; the current compatibility GraphSpec "
                "builder serializes the legacy FirstOrderFilter -> PointMass path."
            ),
            "authoritative_sources": [
                "run.json.model_summary",
                "run.json.hps.model.plant_backend",
                "trained_model.eqx",
            ],
        },
    }
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")
    return None


def _stochastic_runtime_contract(hps: TreeNamespace) -> dict[str, Any]:
    contract = graphspec_noise_contract(stochastic_runtime_config_from_model(hps.model))
    if str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)) != CS_LSS_PLANT_BACKEND:
        return contract
    return {
        **contract,
        "sensory_runtime": (
            "Feedbax sensory Channel after the 4D delayed LSS feedback selector; "
            "the delay itself is represented by the C&S 48D LSS state"
        ),
        "command_runtime": (
            "Feedbax efferent Channel immediately before LinearStateSpace.force; "
            "additive and signal-dependent motor noise are both command-channel noise"
        ),
        "plant_process_runtime": (
            "Task-sampled physical-process/load epsilon bound to LinearStateSpace.epsilon"
        ),
        "state_diffusion": "mechanics.epsilon",
    }


def build_run_spec(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    spec_dir: Path,
    graph_bundle: RLRMPFeedbaxGraphBundle,
) -> dict[str, Any]:
    """Build the JSON payload for ``run.json``."""

    hps = build_hps(args)
    training_distribution = _training_distribution_metadata(hps)
    validation_bins = _validation_bins_metadata(hps)
    delayed_reach = _plain(hps.delayed_reach)
    model_summary = build_model_structure_summary(hps)
    training_summary = {
        **graph_bundle.training_spec,
        "training_mode": _training_mode(hps),
        "n_train_batches": int(args.n_train_batches),
        "n_adversary_batches": 0,
        "n_policy_adversary_ascent_steps_per_controller_step": (
            int(config_from_policy_adversary_hps(hps.policy_adversary_training).n_steps)
            if _policy_adversary_training_enabled(hps)
            else 0
        ),
        "training_diagnostics": _training_diagnostics_metadata(args, output_dir),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": str(args.issue),
        "training_script": "scripts/train_cs_nominal_gru.py",
        "mode": _run_mode(args),
        "artifact_output_dir": str(output_dir),
        "spec_dir": str(spec_dir),
        "nominal_only": _nominal_only(hps),
        "training_distribution": training_distribution,
        "delayed_reach": delayed_reach,
        "validation_bins": validation_bins,
        "provenance_refs": {
            "delayed_reach": "$.delayed_reach",
            "loss_objective": "$.loss_objective",
            "model_summary.training_distribution": "$.training_distribution",
            "stochastic_preset": "$.stochastic_preset",
            "training_summary.training_distribution": "$.training_distribution",
            "training_summary.validation_bins": "$.validation_bins",
            "validation_bins": "$.validation_bins",
        },
        "adversarial_phase": _adversarial_phase(hps),
        "modal_launch": "not_requested",
        "full_training_launch": "requested" if args.full_train else "not_requested",
        "seed": int(args.seed),
        "n_train_batches": int(args.n_train_batches),
        "batch_size": int(args.batch_size),
        "controller_lr": float(args.controller_lr),
        "optimizer": _optimizer_metadata(args),
        "checkpointing": _checkpoint_metadata(args, output_dir),
        "training_diagnostics": _training_diagnostics_metadata(args, output_dir),
        "loss_objective": str(hps.loss.objective),
        "fidelity_status": _fidelity_status(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "game_card": build_loss_game_card_provenance(hps),
        "model_summary": model_summary,
        "task_timing": graph_bundle.task_spec,
        "loss_summary": graph_bundle.loss_spec,
        "training_summary": training_summary,
        "feedbax_graph": graph_bundle.to_run_metadata(),
        "hps": _plain(hps),
        "provenance": {
            "git": _get_git_metadata(),
            "dependencies": _get_dependency_metadata(),
            "modal": {
                "launch": "not_requested",
                "app_name": "rlrmp-cs-stochastic-gru",
                "mode": "not_requested",
            },
            "gpu": _get_gpu_metadata(),
            "runtime": _get_runtime_metadata(),
        },
    }


def write_run_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Write, or dry-run, the stochastic C&S GRU spec artifacts."""

    args = _apply_smoke_overrides(args)
    output_dir = Path(args.output_dir)
    explicit_spec_dir = args.spec_dir is not None
    spec_dir = Path(args.spec_dir) if explicit_spec_dir else derive_spec_dir(output_dir)
    run_path = _run_spec_path_for_write(
        output_dir=output_dir,
        spec_dir=spec_dir,
        explicit_spec_dir=explicit_spec_dir,
    )
    hps = build_hps(args)
    graph_bundle = build_graph_bundle(hps)
    payload = build_run_spec(
        args,
        output_dir=output_dir,
        spec_dir=spec_dir,
        graph_bundle=graph_bundle,
    )

    if args.dry_run:
        would_write = [str(run_path), str(spec_dir / "model.graph.manifest.json")]
        if _should_write_graph_spec(hps):
            would_write.append(str(spec_dir / "model.graph.json"))
        return {
            "run_spec": payload,
            "would_write": would_write,
        }

    mkdir_p(spec_dir)
    mkdir_p(run_path.parent)
    graph_path = _write_graph_bundle_for_backend(hps, graph_bundle, spec_dir)
    payload["feedbax_graph"] = graph_bundle.to_run_metadata(
        graph_spec_path=None if graph_path is None else graph_path.name,
    )
    validate_nominal_gru_run_spec(payload, spec_dir=spec_dir)
    run_path.write_text(_json_dumps(payload), encoding="utf-8")
    return {
        "run_spec_path": str(run_path),
        "graph_spec_path": None if graph_path is None else str(graph_path),
        "graph_manifest_path": str(spec_dir / "model.graph.manifest.json"),
    }


def planned_ef9c882_start_pos_hold_rows(
    *,
    experiment: str = "ef9c882",
) -> list[dict[str, Any]]:
    """Return the locked delayed pre-go start-position hold rows for ef9c882."""

    common_command = [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_cs_nominal_gru.py",
        "--issue",
        experiment,
        "--n-train-batches",
        "12000",
        "--batch-size",
        "64",
        "--gradient-clip-norm",
        "5",
        "--lr-warmup-batches",
        "500",
        "--lr-warmup-init-fraction",
        "0.1",
        "--lr-cosine-alpha",
        "0.1",
        "--n-replicates",
        "5",
        "--hidden-size",
        "180",
        "--loss-objective",
        CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "--delayed-reach",
        "--delayed-reach-go-cue-min-step",
        "10",
        "--delayed-reach-go-cue-max-step",
        "30",
        "--delayed-reach-p-catch-trial",
        "0.5",
        "--target-relative-multitarget",
        "--force-filter-feedback",
        "--perturbation-training",
        "--perturbation-calibrated-timing",
        "--perturbation-movement-age-timing",
        "--perturbation-physical-level",
        "small",
        "--nn-output-pre-go",
        "0",
        "--delayed-pre-go-force-filter-hold",
        "0",
    ]
    row_specs = [
        ("hold_start_pos_l2_ffpert__w1e6_lr3e-3", "l2", 1e6, 3e-3, 0.0),
        ("hold_start_pos_l2_ffpert__w1e8_lr3e-3", "l2", 1e8, 3e-3, 0.0),
        ("hold_start_pos_l1_ffpert__w1e6_lr3e-3", "l1", 1e6, 3e-3, 0.0),
        ("hold_start_pos_l1_ffpert__w1e5_lr3e-3", "l1", 1e5, 3e-3, 0.0),
        ("hold_start_pos_l2_ffpert__w1e8_lr1e-2", "l2", 1e8, 1e-2, 0.0),
        ("hold_start_pos_l1_ffpert__w1e5_lr1e-2", "l1", 1e5, 1e-2, 0.0),
        ("hold__start_pos_zero_vel_lr1e-2", "l2", 1e6, 1e-2, 1e5),
        ("hold__start_pos_zero_vel_lr3e-2", "l2", 1e6, 3e-2, 1e5),
    ]
    rows = []
    for run, norm, weight, controller_lr, zero_vel_hold in row_specs:
        command = [
            *common_command,
            "--controller-lr",
            f"{controller_lr:g}",
            "--delayed-pre-go-zero-vel-hold",
            f"{zero_vel_hold:g}",
            "--delayed-pre-go-start-pos-hold",
            f"{weight:g}",
            "--delayed-pre-go-start-pos-hold-norm",
            norm,
            "--output-dir",
            f"_artifacts/{experiment}/runs/{run}",
        ]
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "row_kind": "full_training_contract",
                "adversarial_phase": "none",
                "broad_epsilon_pgd_training": False,
                "delayed_reach": True,
                "target_visible_from_start": True,
                "go_cue_min_step": 10,
                "go_cue_max_step": 30,
                "p_catch_trial": 0.5,
                "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
                "movement_period_position_error_norm": "l2",
                "movement_window_qrf_comparator": "full_analytical_qrf_movement_age",
                "nn_output_pre_go": 0.0,
                "delayed_pre_go_force_filter_hold": 0.0,
                "delayed_pre_go_zero_vel_hold": float(zero_vel_hold),
                "delayed_pre_go_start_pos_hold": float(weight),
                "delayed_pre_go_start_pos_hold_norm": norm,
                "force_filter_feedback": True,
                "perturbation_training": True,
                "perturbation_calibrated_timing": True,
                "perturbation_movement_age_timing": True,
                "perturbation_physical_level": "small",
                "hidden_size": 180,
                "batch_size": 64,
                "controller_lr": float(controller_lr),
                "lr_schedule": "warmup_cosine",
                "lr_warmup_batches": 500,
                "lr_warmup_init_fraction": 0.1,
                "lr_cosine_alpha": 0.1,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "n_train_batches": 12000,
                "n_adversary_batches": 0,
                "eval_bank": "current_corrected_fixed_delayed_movement_bank",
                "spec_command": [*command, "--dry-run"],
                "command": [*command, "--full-train", "--resume"],
            }
        )
    return rows


def planned_246182c_post_movement_cost_tail_rows(
    *,
    experiment: str = "246182c",
) -> list[dict[str, Any]]:
    """Return the locked delayed movement cost-tail diagnostic row for 246182c."""

    run = "hold__start_pos_zero_vel_lr1e-2_flat_tail"
    command = [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_cs_nominal_gru.py",
        "--issue",
        experiment,
        "--n-train-batches",
        "12000",
        "--batch-size",
        "64",
        "--gradient-clip-norm",
        "5",
        "--lr-warmup-batches",
        "500",
        "--lr-warmup-init-fraction",
        "0.1",
        "--lr-cosine-alpha",
        "0.1",
        "--n-replicates",
        "5",
        "--hidden-size",
        "180",
        "--loss-objective",
        CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "--delayed-reach",
        "--delayed-reach-go-cue-min-step",
        "10",
        "--delayed-reach-go-cue-max-step",
        "30",
        "--delayed-reach-p-catch-trial",
        "0.5",
        "--delayed-movement-cost-tail-mode",
        DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
        "--target-relative-multitarget",
        "--force-filter-feedback",
        "--perturbation-training",
        "--perturbation-calibrated-timing",
        "--perturbation-movement-age-timing",
        "--perturbation-physical-level",
        "small",
        "--nn-output-pre-go",
        "0",
        "--delayed-pre-go-force-filter-hold",
        "0",
        "--controller-lr",
        "0.01",
        "--delayed-pre-go-zero-vel-hold",
        "100000",
        "--delayed-pre-go-start-pos-hold",
        "1000000",
        "--delayed-pre-go-start-pos-hold-norm",
        "l2",
        "--output-dir",
        f"_artifacts/{experiment}/runs/{run}",
    ]
    return [
        {
            "experiment": experiment,
            "run": run,
            "row_kind": "full_training_contract",
            "comparator": "ef9c882/hold__start_pos_zero_vel_lr1e-2",
            "adversarial_phase": "none",
            "broad_epsilon_pgd_training": False,
            "delayed_reach": True,
            "target_visible_from_start": True,
            "go_cue_min_step": 10,
            "go_cue_max_step": 30,
            "p_catch_trial": 0.5,
            "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            "movement_window_qrf_comparator": "full_analytical_qrf_movement_age",
            "delayed_movement_cost_tail_mode": DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
            "cost_tail_semantics": (
                "canonical movement-age Q/R stages 0..59, then stage 59 Q/R held "
                "flat through the remaining sampled trial tail; Q_f scores the final "
                "rollout state"
            ),
            "nn_output_pre_go": 0.0,
            "delayed_pre_go_force_filter_hold": 0.0,
            "delayed_pre_go_zero_vel_hold": 1e5,
            "delayed_pre_go_start_pos_hold": 1e6,
            "delayed_pre_go_start_pos_hold_norm": "l2",
            "force_filter_feedback": True,
            "perturbation_training": True,
            "perturbation_calibrated_timing": True,
            "perturbation_movement_age_timing": True,
            "perturbation_physical_level": "small",
            "hidden_size": 180,
            "batch_size": 64,
            "controller_lr": 1e-2,
            "lr_schedule": "warmup_cosine",
            "lr_warmup_batches": 500,
            "lr_warmup_init_fraction": 0.1,
            "lr_cosine_alpha": 0.1,
            "gradient_clip_norm": 5.0,
            "n_replicates": 5,
            "n_train_batches": 12000,
            "n_adversary_batches": 0,
            "planned_run_spec_path": f"results/{experiment}/runs/{run}.json",
            "spec_command": [*command, "--dry-run"],
            "command": [*command, "--full-train", "--resume"],
        }
    ]


def planned_e901a20_policy_adversary_rows(
    *,
    experiment: str = "e901a20",
) -> list[dict[str, Any]]:
    """Return the two H0 policy-adversary gate rows for e901a20."""

    common_command = [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_cs_nominal_gru.py",
        "--issue",
        experiment,
        "--n-train-batches",
        "12000",
        "--stop-after-batches",
        "1000",
        "--batch-size",
        "64",
        "--controller-lr",
        "0.003",
        "--gradient-clip-norm",
        "5",
        "--lr-warmup-batches",
        "500",
        "--lr-warmup-init-fraction",
        "0.1",
        "--lr-cosine-alpha",
        "0.01",
        "--n-replicates",
        "5",
        "--hidden-size",
        "180",
        "--loss-objective",
        CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "--target-relative-multitarget",
        "--force-filter-feedback",
        "--initial-hidden-encoder",
        "--perturbation-training",
        "--perturbation-calibrated-timing",
        "--perturbation-physical-level",
        "small",
        "--policy-adversary-training",
        "--policy-adversary-width",
        "64",
        "--policy-adversary-depth",
        "2",
        "--policy-adversary-steps",
        "5",
        "--policy-adversary-lr",
        "0.0003",
        "--policy-adversary-radius-15cm",
        f"{HISTORICAL_020A65B_PGD_RADIUS_15CM:.18g}",
        "--policy-adversary-radius-source",
        "effective_020a65b_pgd_training_radius",
    ]
    rows = []
    for label in (POLICY_ADVERSARY_PLAIN_MODE, POLICY_ADVERSARY_ENERGY_MODE):
        run = f"h0_policy_adversary__{label}"
        command = [
            *common_command,
            "--policy-adversary-mode",
            label,
            "--output-dir",
            f"_artifacts/{experiment}/runs/{run}",
        ]
        if label == POLICY_ADVERSARY_ENERGY_MODE:
            command.extend(["--policy-adversary-energy-gamma", "1"])
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "row": label,
                "row_kind": "checkpoint_gate_contract",
                "base_contract": "latest 020a65b H0 PGD row excluding PGD",
                "adversarial_phase": "learned_memoryless_policy_adversary",
                "policy_adversary_mode": label,
                "policy_width": 64,
                "policy_depth": 2,
                "policy_output_dim": 8,
                "policy_ascent_steps_per_controller_step": 5,
                "policy_weights_persist_across_batches": True,
                "h_infinity_style_energy_stabilizer": label == POLICY_ADVERSARY_ENERGY_MODE,
                "formal_certificate": False,
                "broad_epsilon_pgd_training": False,
                "broad_epsilon_reach_scaling": True,
                "effective_l2_radius_15cm": HISTORICAL_020A65B_PGD_RADIUS_15CM,
                "radius_source": "effective_020a65b_pgd_training_radius",
                "single_fixed_budget": True,
                "sisu_modulation": False,
                "force_filter_feedback": True,
                "initial_hidden_encoder": True,
                "perturbation_training": True,
                "perturbation_calibrated_timing": True,
                "perturbation_physical_level": "small",
                "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
                "batch_size": 64,
                "controller_lr": 3e-3,
                "gradient_clip_norm": 5.0,
                "lr_schedule": "warmup_cosine",
                "lr_warmup_batches": 500,
                "lr_warmup_init_fraction": 0.1,
                "lr_cosine_alpha": 0.01,
                "n_replicates": 5,
                "n_train_batches": 12000,
                "stop_after_batches": 1000,
                "gate_batches": "500-1000; planner command stops at 1000",
                "diagnostics": [
                    "policy_adversary_epsilon_norm_radius_ratio",
                    "policy_adversary_epsilon_energy",
                    "policy_adversary_boundary_fraction",
                    "policy_adversary_adversary_objective",
                    "policy_adversary_controller_loss",
                    "policy_adversary_stabilizer_term",
                ],
                "spec_command": [*command, "--dry-run"],
                "command": [*command, "--full-train", "--resume"],
            }
        )
    return rows


def run_full_training(
    args: argparse.Namespace,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    """Run chunked stochastic C&S GRU training with durable checkpoints.

    Feedbax's trainer can accept an optimizer state, but its checkpoint restore
    path is model-centric. This wrapper owns the resume contract explicitly:
    checkpoints store model leaves, optimizer state, completed batch count,
    PRNG state, run/config metadata, and training history snapshots.
    """

    args = _apply_smoke_overrides(args)
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

    spec_result = write_run_spec(args)
    output_dir = mkdir_p(Path(args.output_dir))
    run_spec_path = Path(spec_result["run_spec_path"])
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))

    hps = build_hps(args)
    key_init, key_train, key_adversary = jr.split(jr.PRNGKey(int(args.seed)), 3)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = _build_trainer(hps)
    adaptive_epsilon_enabled = _adaptive_epsilon_curriculum_enabled(hps)
    pre_step_fn = (
        None
        if adaptive_epsilon_enabled
        else make_broad_epsilon_pgd_pre_step(hps.broad_epsilon_pgd_training)
    )
    policy_adversary_enabled = _policy_adversary_training_enabled(hps)
    policy_adversary_optimizer = None
    adversary_policy_template = None
    adversary_optimizer_state_template = None
    if policy_adversary_enabled:
        adversary_cfg = config_from_policy_adversary_hps(hps.policy_adversary_training)
        adversary_policy_template = make_policy_adversary(
            adversary_cfg,
            key=key_adversary,
            horizon=max(1, int(hps.task.n_steps) - 1),
        )
        policy_adversary_optimizer = optax.adam(float(adversary_cfg.learning_rate))
        adversary_optimizer_state_template = policy_adversary_optimizer.init(
            eqx.filter(adversary_policy_template, eqx.is_array)
        )
    where_train = _where_train()
    template_state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=where_train[0],
        key=key_train,
    )
    if adaptive_epsilon_enabled:
        template_state = replace(
            template_state,
            adaptive_epsilon_state=_initial_adaptive_epsilon_state(hps),
        )
    checkpoint_root = output_dir / "checkpoints"
    state = (
        load_latest_checkpoint(
            checkpoint_root,
            model_template=pair.model,
            optimizer_state_template=template_state.optimizer_state,
            history_template=None,
            adversary_policy_template=adversary_policy_template,
            adversary_optimizer_state_template=adversary_optimizer_state_template,
        )
        if args.resume and latest_checkpoint_path(checkpoint_root).exists()
        else template_state
    )
    if policy_adversary_enabled:
        if state.adversary_policy is None:
            state = replace(
                state,
                adversary_policy=adversary_policy_template,
            )
        if state.adversary_optimizer_state is None:
            state = replace(
                state,
                adversary_optimizer_state=adversary_optimizer_state_template,
            )
    if adaptive_epsilon_enabled and state.adaptive_epsilon_state is None:
        state = replace(state, adaptive_epsilon_state=_initial_adaptive_epsilon_state(hps))

    chunks: list[dict[str, float | int | str]] = []
    pgd_diagnostic_chunks: list[dict[str, np.ndarray]] = []
    adaptive_epsilon_diagnostic_chunks: list[dict[str, np.ndarray]] = []
    policy_adversary_diagnostic_chunks: list[dict[str, np.ndarray]] = []
    training_started = time.perf_counter()
    while state.completed_batches < int(args.n_train_batches):
        if stop_after_batches is not None and state.completed_batches >= stop_after_batches:
            break
        remaining = int(args.n_train_batches) - state.completed_batches
        chunk_batches = min(
            int(args.checkpoint_interval_batches),
            remaining,
        )
        if stop_after_batches is not None:
            chunk_batches = min(chunk_batches, stop_after_batches - state.completed_batches)
        key_chunk, key_next = jr.split(state.key, 2)
        chunk_started = time.perf_counter()
        pgd_diagnostics: dict[str, np.ndarray] = {}
        policy_adversary_diagnostics = None
        adaptive_epsilon_diagnostics = None
        if policy_adversary_enabled:
            if policy_adversary_optimizer is None:
                raise ValueError("Policy adversary optimizer was not initialized.")
            (
                model,
                history_chunk,
                optimizer_state,
                adversary_policy,
                adversary_optimizer_state,
                policy_adversary_diagnostics,
            ) = _run_policy_adversary_training_chunk(
                trainer=trainer,
                task=pair.task,
                model=state.model,
                optimizer_state=state.optimizer_state,
                adversary_policy=state.adversary_policy,
                adversary_optimizer_state=state.adversary_optimizer_state,
                adversary_optimizer=policy_adversary_optimizer,
                hps=hps,
                where_train=where_train[0],
                key=key_chunk,
                start_batch=state.completed_batches,
                chunk_batches=chunk_batches,
                log_progress=not bool(args.disable_progress),
            )
        elif adaptive_epsilon_enabled:
            (
                model,
                history_chunk,
                optimizer_state,
                adaptive_epsilon_state,
                adaptive_epsilon_diagnostics,
            ) = _run_adaptive_epsilon_training_chunk(
                trainer=trainer,
                task=pair.task,
                model=state.model,
                optimizer_state=state.optimizer_state,
                adaptive_state=state.adaptive_epsilon_state,
                hps=hps,
                where_train=where_train[0],
                key=key_chunk,
                start_batch=state.completed_batches,
                chunk_batches=chunk_batches,
                log_progress=not bool(args.disable_progress),
            )
            adversary_policy = state.adversary_policy
            adversary_optimizer_state = state.adversary_optimizer_state
            state = replace(state, adaptive_epsilon_state=adaptive_epsilon_state)
        else:
            adversary_policy = state.adversary_policy
            adversary_optimizer_state = state.adversary_optimizer_state
            model, history_chunk, optimizer_state = trainer(
                pair.task,
                state.model,
                n_batches=chunk_batches,
                # Keep Feedbax's batch index local to the chunk: its PRNG key array
                # is chunk-local. Passing the stable selector function avoids the
                # dict-at-local-batch-0 path that would reinitialise optimizer state.
                idx_start=0,
                opt_state=state.optimizer_state,
                key=key_chunk,
                ensembled=True,
                loss_func=pair.task.loss_func,
                where_train=where_train[0],
                batch_size=int(hps.batch_size),
                log_step=max(1, int(args.log_step)),
                disable_progress=bool(args.disable_progress),
                verbose_progress=not bool(args.quiet_progress),
                pre_step_fn=pre_step_fn,
            )
        chunk_duration_seconds = time.perf_counter() - chunk_started
        completed = state.completed_batches + chunk_batches
        history = _append_history(state.history, history_chunk)
        if _training_diagnostics_enabled(args):
            pgd_diagnostics = _broad_epsilon_pgd_diagnostics_arrays(
                pair.task,
                model,
                hps,
                key=key_chunk,
                batch_index=completed - 1,
                chunk_batches=chunk_batches,
            )
            if pgd_diagnostics:
                pgd_diagnostic_chunks.append(pgd_diagnostics)
            if adaptive_epsilon_diagnostics:
                adaptive_epsilon_diagnostic_chunks.append(adaptive_epsilon_diagnostics)
            if policy_adversary_diagnostics:
                policy_adversary_diagnostic_chunks.append(
                    _policy_adversary_diagnostics_arrays(
                        policy_adversary_diagnostics,
                        batch_index=completed - 1,
                        chunk_batches=chunk_batches,
                    )
                )
        if not bool(args.disable_progress):
            _emit_checkpoint_progress(
                history_chunk,
                pgd_diagnostics,
                chunk_batches=chunk_batches,
                completed_batches=completed,
                total_batches=int(args.n_train_batches),
                elapsed_seconds=time.perf_counter() - training_started,
            )
        history_chunk_path = output_dir / "history_chunks" / f"history_{completed:07d}.eqx"
        history_chunk_path.parent.mkdir(parents=True, exist_ok=True)
        _save_pytree(history_chunk_path, history_chunk)
        state = replace(
            state,
            model=model,
            optimizer_state=optimizer_state,
            completed_batches=completed,
            key=key_next,
            history=history,
            adversary_policy=adversary_policy,
            adversary_optimizer_state=adversary_optimizer_state,
        )
        checkpoint_path = save_training_checkpoint(
            checkpoint_root,
            state,
            args=args,
            run_spec=run_spec,
        )
        _commit_volume(volume_commit)
        chunks.append(
            {
                "completed_batches": completed,
                "checkpoint": str(checkpoint_path),
                "history_chunk": str(history_chunk_path),
                "chunk_batches": chunk_batches,
                "duration_seconds": chunk_duration_seconds,
                "batches_per_second": chunk_batches / chunk_duration_seconds,
            }
        )
        if stop_after_batches is not None and state.completed_batches >= stop_after_batches:
            break
    training_duration_seconds = time.perf_counter() - training_started

    final_model_path = output_dir / "trained_model.eqx"
    final_adversary_policy_path = output_dir / "trained_policy_adversary.eqx"
    final_history_path = output_dir / "training_history.eqx"
    final_summary_path = output_dir / "training_summary.json"
    _save_pytree(final_model_path, state.model, hyperparameters=run_spec)
    if state.adversary_policy is not None:
        _save_pytree(final_adversary_policy_path, state.adversary_policy, hyperparameters=run_spec)
    if state.history is not None:
        _save_pytree(final_history_path, state.history)
    diagnostics_metadata = write_training_diagnostics_sidecar(
        output_dir,
        args=args,
        run_spec=run_spec,
        state=state,
        training_history_path=final_history_path,
        pgd_diagnostic_chunks=pgd_diagnostic_chunks,
        policy_adversary_diagnostic_chunks=policy_adversary_diagnostic_chunks,
        adaptive_epsilon_diagnostic_chunks=adaptive_epsilon_diagnostic_chunks,
    )
    final_summary = {
        "schema_version": f"{SCHEMA_VERSION}.training.v1",
        "issue": str(args.issue),
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "stopped_early_for_checkpoint_gate": (
            stop_after_batches is not None and state.completed_batches < int(args.n_train_batches)
        ),
        "stop_after_batches": stop_after_batches,
        "training_duration_seconds": training_duration_seconds,
        "training_batches_per_second": (
            state.completed_batches / training_duration_seconds
            if training_duration_seconds > 0
            else None
        ),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "final_model_path": str(final_model_path),
        "final_adversary_policy_path": (
            str(final_adversary_policy_path) if state.adversary_policy is not None else None
        ),
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
        "final_adversary_policy_path": (
            str(final_adversary_policy_path) if state.adversary_policy is not None else None
        ),
        "training_history_path": str(final_history_path),
        "training_summary_path": str(final_summary_path),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "completed_batches": state.completed_batches,
    }


def save_training_checkpoint(
    checkpoint_root: Path,
    state: TrainingState,
    *,
    args: argparse.Namespace,
    run_spec: dict[str, Any],
) -> Path:
    """Write a numbered checkpoint and atomically repoint ``checkpoint_latest``."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"checkpoint_{state.completed_batches:07d}"
    target = checkpoint_root / checkpoint_name
    tmp = checkpoint_root / f".{checkpoint_name}.tmp"
    if tmp.exists():
        _remove_tree(tmp)
    tmp.mkdir(parents=True)

    eqx.tree_serialise_leaves(tmp / "model.eqx", state.model)
    eqx.tree_serialise_leaves(tmp / "optimizer_state.eqx", state.optimizer_state)
    if state.adversary_policy is not None:
        eqx.tree_serialise_leaves(tmp / "adversary_policy.eqx", state.adversary_policy)
    if state.adversary_optimizer_state is not None:
        eqx.tree_serialise_leaves(
            tmp / "adversary_optimizer_state.eqx",
            state.adversary_optimizer_state,
        )
    if state.history is not None:
        _save_pytree(tmp / "history.eqx", state.history)
    metadata = {
        "schema_version": f"{SCHEMA_VERSION}.checkpoint.v1",
        "issue": str(args.issue),
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "seed": int(args.seed),
        "next_prng_key": _plain(state.key),
        "stochastic_preset": str(args.stochastic_preset),
        "run_spec": run_spec,
    }
    if state.adaptive_epsilon_state is not None:
        metadata["adaptive_epsilon_state"] = state.adaptive_epsilon_state.to_json()
    _atomic_write_json(tmp / "metadata.json", metadata)
    if target.exists():
        _remove_tree(target)
    os.replace(tmp, target)
    _atomic_latest_link(checkpoint_root, checkpoint_name)
    _atomic_write_json(
        checkpoint_root / "checkpoint_index.json",
        {
            "latest": checkpoint_name,
            "latest_path": str(latest_checkpoint_path(checkpoint_root)),
            "completed_batches": state.completed_batches,
        },
    )
    return target


def load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    model_template: Any,
    optimizer_state_template: Any,
    history_template: Any | None = None,
    adversary_policy_template: Any | None = None,
    adversary_optimizer_state_template: Any | None = None,
) -> TrainingState:
    """Load ``checkpoint_latest`` using explicit model and optimizer templates."""

    checkpoint_path = latest_checkpoint_path(checkpoint_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_latest found under {checkpoint_root}")
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", model_template)
    optimizer_state = eqx.tree_deserialise_leaves(
        checkpoint_path / "optimizer_state.eqx",
        optimizer_state_template,
    )
    history_path = checkpoint_path / "history.eqx"
    history = (
        eqx.tree_deserialise_leaves(history_path, history_template)
        if history_template is not None and history_path.exists()
        else None
    )
    adversary_policy_path = checkpoint_path / "adversary_policy.eqx"
    adversary_policy = (
        eqx.tree_deserialise_leaves(adversary_policy_path, adversary_policy_template)
        if adversary_policy_template is not None and adversary_policy_path.exists()
        else None
    )
    adversary_optimizer_state_path = checkpoint_path / "adversary_optimizer_state.eqx"
    adversary_optimizer_state = (
        eqx.tree_deserialise_leaves(
            adversary_optimizer_state_path,
            adversary_optimizer_state_template,
        )
        if (
            adversary_optimizer_state_template is not None
            and adversary_optimizer_state_path.exists()
        )
        else None
    )
    adaptive_payload = metadata.get("adaptive_epsilon_state")
    adaptive_epsilon_state = (
        AdaptiveEpsilonState(
            lambda_value=float(adaptive_payload["lambda_value"]),
            damage_ema=(
                None
                if adaptive_payload.get("damage_ema") is None
                else float(adaptive_payload["damage_ema"])
            ),
            last_update_batch=adaptive_payload.get("last_update_batch"),
            update_count=int(adaptive_payload.get("update_count", 0)),
        )
        if isinstance(adaptive_payload, dict)
        else None
    )
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=int(metadata["completed_batches"]),
        key=jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32),
        history=history,
        adversary_policy=adversary_policy,
        adversary_optimizer_state=adversary_optimizer_state,
        adaptive_epsilon_state=adaptive_epsilon_state,
    )


def latest_checkpoint_path(checkpoint_root: Path) -> Path:
    """Return the path used by the durable latest-checkpoint contract."""

    return checkpoint_root / "checkpoint_latest"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Prepare a stochastic C&S-fidelity GRU run spec.",
    )
    parser.add_argument(
        "--run-spec",
        default=None,
        help=(
            "Replay a modern tracked nominal-GRU run.json through the current "
            "training/spec writer. Explicit CLI flags override the run spec."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spec-dir", default=None)
    parser.add_argument("--issue", default=ISSUE_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-batches", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--controller-lr", type=float, default=1e-2)
    parser.add_argument(
        "--lr-warmup-batches",
        type=int,
        default=0,
        help=(
            "If positive, linearly warm the controller LR from "
            "--lr-warmup-init-fraction * --controller-lr to --controller-lr over this "
            "many batches, then cosine-decay."
        ),
    )
    parser.add_argument(
        "--lr-warmup-init-fraction",
        type=float,
        default=0.1,
        help="Initial LR fraction for warmup-cosine schedules.",
    )
    parser.add_argument(
        "--lr-cosine-alpha",
        type=float,
        default=1.0,
        help="Final LR fraction for cosine schedules.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=180)
    parser.add_argument(
        "--plant-backend",
        choices=[CS_LSS_PLANT_BACKEND, LEGACY_CAUSAL_PLANT_BACKEND],
        default=CS_LSS_PLANT_BACKEND,
        help=(
            "Plant backend for nominal GRU training. The default uses exact C&S "
            "LinearStateSpace mechanics; legacy_causal_simplefeedback preserves the "
            "old point-mass/force-filter path and emits a timing warning."
        ),
    )
    parser.add_argument(
        "--no-integrator-state",
        action="store_true",
        help=(
            "Use the reduced C&S LSS comparator with 6D physical state "
            "[pos, vel, force/filter] and no disturbance-integrator coordinates."
        ),
    )
    parser.add_argument(
        "--stochastic-preset",
        choices=[DEFAULT_STOCHASTIC_PRESET],
        default=DEFAULT_STOCHASTIC_PRESET,
        help=(
            "Named stochastic rollout contract. The preset fixes sensory, command, "
            "signal-dependent, and plant/load force noise and records concrete "
            "values in run.json."
        ),
    )
    parser.add_argument(
        "--target-m",
        type=float,
        default=float(TARGET_POS[0]),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--n-input-only", type=int, default=0)
    parser.add_argument("--n-readout-only", type=int, default=0)
    parser.add_argument("--n-recurrent-only", type=int, default=0)
    parser.add_argument("--effector-pos-running", type=float, default=CS_POSITION_SCALE)
    parser.add_argument("--effector-vel-running", type=float, default=CS_VELOCITY_SCALE)
    parser.add_argument("--effector-terminal-pos", type=float, default=CS_POSITION_SCALE)
    parser.add_argument("--effector-terminal-vel", type=float, default=CS_VELOCITY_SCALE)
    parser.add_argument("--effector-final-vel", type=float, default=0.0)
    parser.add_argument("--nn-output", type=float, default=CS_CONTROL_SCALE)
    parser.add_argument("--nn-output-jerk", type=float, default=0.0)
    parser.add_argument(
        "--nn-output-pre-go",
        type=float,
        default=None,
        help=(
            "Anti-anticipation controller-output penalty during delayed-reach prep. "
            "Defaults to 1.0 only when --delayed-reach is active; otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--delayed-pre-go-force-filter-hold",
        type=float,
        default=0.0,
        help=(
            "Prep-only delayed-reach auxiliary penalty on the C&S force/filter state. "
            "Default 0.0 preserves the movement-window Q/R/Q_f comparator."
        ),
    )
    parser.add_argument(
        "--delayed-pre-go-start-pos-hold",
        type=float,
        default=0.0,
        help=(
            "Prep-only delayed-reach auxiliary penalty on effector position away from "
            "the sampled start position. Default 0.0."
        ),
    )
    parser.add_argument(
        "--delayed-pre-go-start-pos-hold-norm",
        choices=["l2", "l1"],
        default="l2",
        help=(
            "Norm for --delayed-pre-go-start-pos-hold. l2 preserves the existing "
            "squared-distance penalty; l1 scores absolute coordinate displacement."
        ),
    )
    parser.add_argument(
        "--delayed-pre-go-zero-vel-hold",
        type=float,
        default=0.0,
        help=(
            "Prep-only delayed-reach auxiliary penalty on nonzero effector velocity "
            "before the go cue. Default 0.0."
        ),
    )
    parser.add_argument(
        "--loss-objective",
        choices=CS_LOSS_OBJECTIVES,
        default=CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        help=(
            "Training objective. Default partial_feedbax_terms preserves the historical "
            "Feedbax pos/vel/control term subset. full_analytical_qrf uses the canonical "
            "C&S Q/R/Q_f matrices on the cs_lss 48D state plus command history."
        ),
    )
    parser.add_argument(
        "--regularized-fidelity",
        action="store_true",
        help="Mark the paired non-exact run and use nn_hidden=1e-5.",
    )
    parser.add_argument(
        "--perturbation-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable fixed-target perturbation-generalized training using external "
            "task/plant/channel adapters. Defaults to on for --delayed-reach and off "
            "otherwise. Target-position streams are not added."
        ),
    )
    parser.add_argument("--perturbation-nominal-fraction", type=float, default=0.45)
    parser.add_argument("--perturbation-single-fraction", type=float, default=0.45)
    parser.add_argument("--perturbation-combined-fraction", type=float, default=0.10)
    parser.add_argument("--perturbation-combined-amplitude-scale", type=float, default=0.5)
    parser.add_argument("--perturbation-initial-position-offset-m", type=float, default=0.01)
    parser.add_argument("--perturbation-initial-velocity-offset-m-s", type=float, default=0.05)
    parser.add_argument("--perturbation-process-epsilon-scale", type=float, default=0.01)
    parser.add_argument("--perturbation-command-input-pulse-n", type=float, default=1.0)
    parser.add_argument("--perturbation-sensory-feedback-offset-m", type=float, default=0.01)
    parser.add_argument(
        "--perturbation-delayed-observation-offset-m",
        type=float,
        default=0.01,
        help=(
            "Legacy run-spec compatibility only; delayed_observation is no longer "
            "sampled or validated in the active final perturbation bank."
        ),
    )
    parser.add_argument("--perturbation-pulse-start-step", type=int, default=20)
    parser.add_argument("--perturbation-pulse-duration-steps", type=int, default=5)
    parser.add_argument(
        "--perturbation-calibrated-timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use timing-bin calibrated perturbation training: plant process/command "
            "pulses sample starts 5/15/35 uniformly and controller-visible sensory "
            "feedback offsets sample starts 10/20/40 uniformly, all with 5-step "
            "pulses. Defaults to on for --delayed-reach perturbation training."
        ),
    )
    parser.add_argument(
        "--perturbation-movement-age-timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Index calibrated perturbation timing bins by movement age: plant "
            "process/command pulses use movement_start + 5/15/35, controller-visible "
            "sensory-feedback pulses use movement_start + 10/20/40, and "
            "initial position/velocity diagnostics use movement-onset process-epsilon "
            "impulses. Requires --perturbation-calibrated-timing and defaults to on "
            "for --delayed-reach perturbation training."
        ),
    )
    parser.add_argument(
        "--perturbation-physical-level",
        choices=("small", "moderate", "stress"),
        default=None,
        help=(
            "Declared reach-relative perturbation level for calibrated screens. "
            "Small/moderate are training rows; stress is reserved for evaluation. "
            "Defaults to small for --delayed-reach and moderate otherwise."
        ),
    )
    parser.add_argument(
        "--perturbation-calibration-regime",
        choices=(
            "open_loop_all",
            "closed_loop_sensory",
            "closed_loop_sensory_command_lateral",
        ),
        default="open_loop_all",
        help=(
            "Select how calibrated perturbation families resolve amplitudes: all "
            "families from open-loop calibration, sensory feedback from the "
            "closed-loop table, or sensory plus command/random force plus "
            "target-aligned lateral loads from the closed-loop table."
        ),
    )
    parser.add_argument(
        "--perturbation-closed-loop-calibration-table",
        default=None,
        help=(
            "Path to a closed-loop calibration table used by mixed "
            "perturbation-calibration regimes."
        ),
    )
    parser.add_argument(
        "--target-relative-multitarget",
        action="store_true",
        help=(
            "Enable static target-relative multi-target training. The C&S LSS GRU "
            "receives [target_x - delayed_x, target_y - delayed_y, -delayed_vx, "
            "-delayed_vy] feedback and uses structured seen/held-out target bins."
        ),
    )
    parser.add_argument(
        "--target-support-profile",
        choices=TARGET_SUPPORT_PROFILES,
        default=DEFAULT_TARGET_SUPPORT_PROFILE,
        help=(
            "Named finite target-support profile for --target-relative-multitarget. "
            "Defaults to const_band16: fixed 0.15 m reaches on the dense validation "
            "grid with 16 held-out angular-band directions. Pass old_020a65b to "
            "replay the old seen/held-out target bank."
        ),
    )
    parser.add_argument(
        "--delayed-reach",
        action="store_true",
        help=(
            "Use the explicit delayed-reach C&S task contract: target-relative "
            "multi-target feedback plus one scalar go-cue input, target visible "
            "from trial start, and movement-epoch scoring."
        ),
    )
    parser.add_argument(
        "--delayed-reach-go-cue-min-step",
        type=int,
        default=DEFAULT_DELAYED_GO_CUE_MIN_STEP,
        help="Inclusive minimum sampled go-cue/prep length for --delayed-reach.",
    )
    parser.add_argument(
        "--delayed-reach-go-cue-max-step",
        type=int,
        default=DEFAULT_DELAYED_GO_CUE_MAX_STEP,
        help="Inclusive maximum sampled go-cue/prep length for --delayed-reach.",
    )
    parser.add_argument(
        "--delayed-reach-p-catch-trial",
        type=float,
        default=DEFAULT_DELAYED_P_CATCH_TRIAL,
        help=(
            "Probability of delayed-reach no-go catch trials. Catch trials keep the "
            "target visible but keep the go cue at 0 and score holding the initial "
            "position. Ignored unless --delayed-reach is active."
        ),
    )
    parser.add_argument(
        "--delayed-movement-cost-tail-mode",
        choices=DELAYED_MOVEMENT_COST_TAIL_MODES,
        default=DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
        help=(
            "Delayed-reach full-Q/R/Qf tail support. canonical_window preserves the "
            "60 movement-age stage objective; flat_after_canonical_horizon reuses the "
            "terminal running Q/R stage after movement age 59 through the trial tail."
        ),
    )
    parser.add_argument(
        "--delayed-reach-trial-type-normalized-loss",
        "--delayed-reach-trial-type-normalization",
        action="store_true",
        dest="delayed_reach_trial_type_normalized_loss",
        help=(
            "For delayed full_analytical_qrf rows, split Q/R/Q_f into no-catch and "
            "catch trial terms and normalize each over its own selected trials before "
            "applying explicit weights. This separates p_catch sampling from objective "
            "weighting."
        ),
    )
    parser.add_argument(
        "--delayed-reach-no-catch-qrf-weight",
        type=float,
        default=1.0,
        help=(
            "Explicit weight for the no-catch movement Q/R/Q_f mean when "
            "--delayed-reach-trial-type-normalized-loss is active."
        ),
    )
    parser.add_argument(
        "--delayed-reach-catch-qrf-weight",
        type=float,
        default=1.0,
        help=(
            "Explicit weight for the catch/no-go Q/R/Q_f mean when "
            "--delayed-reach-trial-type-normalized-loss is active."
        ),
    )
    parser.add_argument(
        "--force-filter-feedback",
        "--proprioceptive-feedback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Extend target-relative delayed feedback with delayed force/filter x/y "
            "coordinates. Defaults to on for --delayed-reach and off otherwise. "
            "Requires --target-relative-multitarget."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-training",
        action="store_true",
        help=(
            "Enable randomized full-state C&S epsilon training: iid 8D epsilon over "
            "time/components, projected per trial to the declared L2 budget."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-training",
        action="store_true",
        help=(
            "Enable training-time projected-gradient ascent over the full T x 8 "
            "C&S epsilon channel before each controller update."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-level",
        choices=("moderate", "strong"),
        default="moderate",
        help=(
            "Analytical broad-epsilon budget anchor. moderate uses gamma factor 1.4; "
            "strong uses gamma factor 1.05."
        ),
    )
    parser.add_argument("--broad-epsilon-budget-scale", type=float, default=1.0)
    parser.add_argument(
        "--broad-epsilon-pgd-fixed-radius-15cm",
        type=float,
        default=None,
        help=(
            "Override the fixed broad-epsilon PGD L2 radius at the 15 cm reference reach. "
            "Use with --broad-epsilon-pgd-fixed-radius-source for provenance."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-fixed-radius-source",
        default=None,
        help="Provenance key for --broad-epsilon-pgd-fixed-radius-15cm.",
    )
    parser.add_argument(
        "--broad-epsilon-reach-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Scale the 15 cm analytical epsilon L2 radius by sampled reach length. "
            "This is an explicit multi-target normalization choice."
        ),
    )
    parser.add_argument("--broad-epsilon-pgd-steps", type=int, default=3)
    parser.add_argument(
        "--broad-epsilon-pgd-step-size-fraction",
        type=float,
        default=0.25,
        help="PGD ascent step size as a fraction of each trial's L2 radius.",
    )
    parser.add_argument(
        "--broad-epsilon-pgd-inner-optimizer-method",
        choices=BROAD_EPSILON_PGD_INNER_OPTIMIZER_METHODS,
        default=BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
        help=(
            "Inner optimizer for broad-epsilon adversary selection. "
            f"{BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT} preserves the historical "
            "normalized projected-gradient ascent behavior; "
            f"{BROAD_EPSILON_PGD_ADAM} runs Adam ascent over live finite-policy "
            "graph-component parameters."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-adam-lr",
        type=float,
        default=3e-4,
        help=(
            "Adam ascent learning rate for broad-epsilon direct-epsilon sequences "
            "or live finite-policy mechanisms."
        ),
    )
    parser.add_argument("--broad-epsilon-pgd-adam-b1", type=float, default=0.9)
    parser.add_argument("--broad-epsilon-pgd-adam-b2", type=float, default=0.999)
    parser.add_argument("--broad-epsilon-pgd-adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--broad-epsilon-pgd-mechanism",
        choices=BROAD_EPSILON_PGD_MECHANISMS,
        default=BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
        help=(
            "Adversary mechanism for broad-epsilon PGD. direct_epsilon preserves the "
            "existing exogenous epsilon-sequence path; finite mechanisms install live "
            "graph-component policy inputs for closed-loop rollout evaluation."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-objective",
        choices=(BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE, BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE),
        default=BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
        help=(
            "Inner PGD objective. hard_l2 preserves the existing projected objective; "
            "soft_energy maximizes task_loss - lambda * epsilon_energy with an explicit "
            "stabilization cap."
        ),
    )
    parser.add_argument("--broad-epsilon-pgd-energy-gamma-star", type=float, default=None)
    parser.add_argument("--broad-epsilon-pgd-energy-gamma-factor", type=float, default=None)
    parser.add_argument("--broad-epsilon-pgd-energy-gamma", type=float, default=None)
    parser.add_argument(
        "--broad-epsilon-pgd-energy-penalty-scale",
        type=float,
        default=1.0,
        help="Soft-energy c multiplier in lambda = c * gamma^2 unless lambda is explicit.",
    )
    parser.add_argument(
        "--broad-epsilon-pgd-energy-lambda",
        type=float,
        default=None,
        help="Explicit soft-energy lambda; otherwise derived as c * gamma^2.",
    )
    parser.add_argument(
        "--broad-epsilon-pgd-safety-cap-15cm",
        type=float,
        default=None,
        help=(
            "Optional 15 cm L2 trust-region cap for soft-energy PGD stabilization. "
            "This cap is metadata-marked as not the scientific hard-budget constraint."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-safety-cap-source",
        type=str,
        default=None,
        help="Provenance key/source for --broad-epsilon-pgd-safety-cap-15cm.",
    )
    parser.add_argument(
        "--broad-epsilon-pgd-budget-schedule",
        choices=("fixed", "sisu_energy_fraction"),
        default="fixed",
        help=(
            "Select the PGD L2 budget schedule. fixed preserves the existing single "
            "radius; sisu_energy_fraction maps SISU to radius via sqrt(SISU)."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-sisu-condition-input",
        choices=("auto", "input", "sisu"),
        default="auto",
        help=(
            "Trial input that carries the scalar SISU value for sisu_energy_fraction PGD budgets."
        ),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-sisu-max-radius",
        type=float,
        default=None,
        help=("Maximum 15 cm PGD L2 radius at SISU=1 for sisu_energy_fraction budgets."),
    )
    parser.add_argument(
        "--broad-epsilon-pgd-sisu-max-radius-source",
        type=str,
        default=None,
        help=(
            "Metadata key/source for the SISU PGD max radius, e.g. "
            "raw_strong_gamma_1p05_radius or effective_020a65b_pgd_training_radius."
        ),
    )
    parser.add_argument(
        "--adaptive-epsilon-curriculum",
        action="store_true",
        help=(
            "Enable adaptive-lambda soft-energy direct-epsilon training with a paired "
            "clean/adversarial controller loss. Requires --broad-epsilon-pgd-training "
            "and --broad-epsilon-pgd-objective soft_energy."
        ),
    )
    parser.add_argument("--adaptive-epsilon-damage-start", type=float, default=0.0)
    parser.add_argument("--adaptive-epsilon-damage-peak", type=float, default=3500.0)
    parser.add_argument("--adaptive-epsilon-damage-final", type=float, default=1000.0)
    parser.add_argument("--adaptive-epsilon-damage-ramp-batches", type=int, default=2500)
    parser.add_argument("--adaptive-epsilon-damage-anneal-batches", type=int, default=5000)
    parser.add_argument("--adaptive-epsilon-update-interval-batches", type=int, default=50)
    parser.add_argument("--adaptive-epsilon-ema-alpha", type=float, default=0.1)
    parser.add_argument("--adaptive-epsilon-eta", type=float, default=0.1)
    parser.add_argument("--adaptive-epsilon-deadband-frac", type=float, default=0.10)
    parser.add_argument("--adaptive-epsilon-lambda-min", type=float, default=1e-12)
    parser.add_argument("--adaptive-epsilon-lambda-max", type=float, default=None)
    parser.add_argument("--adaptive-epsilon-max-log-step", type=float, default=0.25)
    parser.add_argument("--adaptive-epsilon-outer-weight-start", type=float, default=0.0)
    parser.add_argument("--adaptive-epsilon-outer-weight-final", type=float, default=1.0)
    parser.add_argument("--adaptive-epsilon-outer-weight-ramp-batches", type=int, default=2500)
    parser.add_argument(
        "--policy-adversary-training",
        action="store_true",
        help=(
            "Enable learned full-state epsilon policy-adversary training. "
            "This replaces PGD for policy-adversary rows and keeps adversary weights "
            "persistent across controller batches."
        ),
    )
    parser.add_argument(
        "--policy-adversary-policy-class",
        choices=POLICY_ADVERSARY_POLICY_CLASSES,
        default=POLICY_ADVERSARY_MEMORYLESS_MLP,
        help=(
            "Adversary policy parameterization: memoryless_mlp for the existing MLP lane, "
            "or linear_no_bias/affine for finite time-varying policies optimized by Adam."
        ),
    )
    parser.add_argument(
        "--policy-adversary-mode",
        choices=(POLICY_ADVERSARY_PLAIN_MODE, POLICY_ADVERSARY_ENERGY_MODE),
        default=POLICY_ADVERSARY_PLAIN_MODE,
        help=(
            "plain uses hard projection only; energy adds an H-infinity-style "
            "energy stabilizer to the adversary objective."
        ),
    )
    parser.add_argument("--policy-adversary-width", type=int, default=64)
    parser.add_argument("--policy-adversary-depth", type=int, default=2)
    parser.add_argument("--policy-adversary-steps", type=int, default=5)
    parser.add_argument("--policy-adversary-lr", type=float, default=3e-4)
    parser.add_argument(
        "--policy-adversary-energy-gamma",
        type=float,
        default=1.0,
        help=(
            "Multiplier on epsilon energy in energy mode. This is a stabilizer "
            "term, not a formal H-infinity certificate parameter."
        ),
    )
    parser.add_argument(
        "--policy-adversary-radius-15cm",
        type=float,
        default=None,
        help=(
            "Explicit 15 cm L2 epsilon radius for policy-adversary training. "
            "Required when --policy-adversary-training is enabled."
        ),
    )
    parser.add_argument(
        "--policy-adversary-radius-source",
        type=str,
        default=None,
        help=(
            "Metadata key/source for --policy-adversary-radius-15cm. Required when "
            "--policy-adversary-training is enabled."
        ),
    )
    parser.add_argument(
        "--initial-hidden-encoder",
        "--h0-encoder",
        action="store_true",
        help=(
            "Enable the conservative C&S GRU H0 path: a zero-initialized affine map "
            "from the first controller-visible target-relative delayed feedback vector "
            "to the GRU hidden state. Requires --target-relative-multitarget."
        ),
    )
    parser.add_argument(
        "--planned-perturbation-rows",
        action="store_true",
        help="Print the two planned issue aacb9ed local training row commands and exit.",
    )
    parser.add_argument(
        "--planned-target-relative-rows",
        action="store_true",
        help="Print the planned issue ba82f3d smoke/main target-relative rows and exit.",
    )
    parser.add_argument(
        "--planned-target-relative-h0-rows",
        action="store_true",
        help="Print the planned issue 643f101 smoke/main target-relative H0 rows and exit.",
    )
    parser.add_argument(
        "--planned-020a65b-h0-pgd-rows",
        action="store_true",
        help="Print the two planned local issue 020a65b H0 no-PGD/PGD gate rows and exit.",
    )
    parser.add_argument(
        "--planned-33b0dcb-target-support-rows",
        action="store_true",
        help="Print the planned issue 33b0dcb no-PGD H0 target-support rows and exit.",
    )
    parser.add_argument(
        "--planned-e901a20-policy-adversary-rows",
        action="store_true",
        help="Print the two planned issue e901a20 H0 policy-adversary gate rows and exit.",
    )
    parser.add_argument(
        "--planned-e4800d6-sisu-spectrum-rows",
        action="store_true",
        help="Print the two planned local issue e4800d6 SISU-conditioned PGD rows and exit.",
    )
    parser.add_argument(
        "--planned-ef9c882-start-pos-hold-rows",
        action="store_true",
        help="Print the five planned issue ef9c882 delayed pre-go start-position hold rows.",
    )
    parser.add_argument(
        "--planned-246182c-post-movement-cost-tail-rows",
        action="store_true",
        help="Print the planned issue 246182c delayed post-movement cost-tail row.",
    )
    parser.add_argument(
        "--planned-7c1f7ed-delayed-sisu-spectrum-rows",
        action="store_true",
        help="Print the two planned issue 7c1f7ed delayed SISU-conditioned PGD rows and exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny local values; with --full-train this runs a one-batch smoke.",
    )
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after-batches",
        type=int,
        default=None,
        help=(
            "For full-train checkpoint-gate smoke runs, stop cleanly after the "
            "first checkpoint at or beyond this completed-batch count while "
            "preserving the original --n-train-batches run contract for resume."
        ),
    )
    parser.add_argument(
        "--training-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write compact optimizer/loss scalar sidecars for full training runs. "
            "Use --no-training-diagnostics to opt out."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_BATCHES,
    )
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--disable-progress", action="store_true", default=False)
    parser.add_argument("--quiet-progress", action="store_true", default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the would-write payload without creating files.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    volume_commit: VolumeCommit | None = None,
) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = resolve_run_spec_args(parser.parse_args(argv), parser=parser)
    if args.planned_perturbation_rows:
        print(_json_dumps({"planned_rows": planned_fixed_target_perturbation_rows()}), end="")
        return 0
    if args.planned_target_relative_rows:
        print(_json_dumps({"planned_rows": planned_target_relative_multitarget_rows()}), end="")
        return 0
    if args.planned_target_relative_h0_rows:
        print(_json_dumps({"planned_rows": planned_target_relative_multitarget_h0_rows()}), end="")
        return 0
    if args.planned_020a65b_h0_pgd_rows:
        print(_json_dumps({"planned_rows": planned_020a65b_h0_pgd_rows()}), end="")
        return 0
    if args.planned_33b0dcb_target_support_rows:
        print(_json_dumps({"planned_rows": planned_33b0dcb_target_support_rows()}), end="")
        return 0
    if args.planned_e901a20_policy_adversary_rows:
        print(_json_dumps({"planned_rows": planned_e901a20_policy_adversary_rows()}), end="")
        return 0
    if args.planned_e4800d6_sisu_spectrum_rows:
        print(_json_dumps({"planned_rows": planned_e4800d6_sisu_spectrum_rows()}), end="")
        return 0
    if args.planned_ef9c882_start_pos_hold_rows:
        print(_json_dumps({"planned_rows": planned_ef9c882_start_pos_hold_rows()}), end="")
        return 0
    if args.planned_246182c_post_movement_cost_tail_rows:
        print(
            _json_dumps({"planned_rows": planned_246182c_post_movement_cost_tail_rows()}),
            end="",
        )
        return 0
    if args.planned_7c1f7ed_delayed_sisu_spectrum_rows:
        print(
            _json_dumps({"planned_rows": planned_7c1f7ed_delayed_sisu_spectrum_rows()}),
            end="",
        )
        return 0
    result = (
        run_full_training(args, volume_commit=volume_commit)
        if args.full_train and not args.dry_run
        else write_run_spec(args)
    )
    print(_json_dumps(result), end="")
    return 0


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    values = vars(args).copy()
    values.update(
        {
            "n_train_batches": 1,
            "batch_size": 2,
            "n_replicates": 1,
            "hidden_size": 4,
            "n_input_only": 0,
            "n_readout_only": 0,
            "n_recurrent_only": 0,
            "checkpoint_interval_batches": 1,
            "log_step": 1,
        }
    )
    return argparse.Namespace(**values)


def _training_diagnostics_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "training_diagnostics", True))


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


def _build_trainer(hps: TreeNamespace) -> TaskTrainer:
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
    optimizer = optax.chain(*transforms)
    return TaskTrainer(optimizer=optimizer, checkpointing=False)


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


def _where_train() -> dict[int, Callable[[Any], tuple[Any, ...]]]:
    def where_train_fn(model):
        net = model.nodes["net"]
        return staged_network_trainable_parts(net)

    return {0: where_train_fn}


def _initial_training_state(
    *,
    model: Any,
    trainer: TaskTrainer,
    where_train: Callable[[Any], Any],
    key: Any,
) -> TrainingState:
    where_train_spec = filter_spec_leaves(model, where_train)
    model_parameters = get_model_parameters(model, where_train_spec)
    optimizer_state = eqx.filter_vmap(trainer.optimizer.init)(model_parameters)
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=0,
        key=key,
        history=None,
    )


def _append_history(history: Any | None, chunk: Any) -> Any:
    if history is None:
        return chunk
    return jt.map(_append_history_leaf, history, chunk, is_leaf=lambda x: x is None)


def _append_history_leaf(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if eqx.is_array(left) and eqx.is_array(right):
        if left.ndim == 0 or right.ndim == 0:
            return right
        return jnp.concatenate([left, right], axis=0)
    return right


def _checkpoint_metadata(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "enabled": bool(args.full_train),
        "resume": bool(args.resume),
        "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
        "latest_checkpoint": str(Path(output_dir) / "checkpoints" / "checkpoint_latest"),
        "numbered_pattern": "checkpoint_{completed_batches:07d}",
        "interval_batches": int(args.checkpoint_interval_batches),
        "contents": [
            "model.eqx",
            "optimizer_state.eqx",
            "history.eqx",
            "adversary_policy.eqx when --policy-adversary-training is active",
            "adversary_optimizer_state.eqx when --policy-adversary-training is active",
            "adaptive_epsilon_state in metadata.json when adaptive epsilon curriculum is active",
            "metadata.json",
        ],
    }


def _optimizer_metadata(args: argparse.Namespace) -> dict[str, Any]:
    schedule_name = "warmup_cosine" if int(args.lr_warmup_batches) > 0 else "delayed_cosine"
    return {
        "name": "adamw",
        "learning_rate_0": float(args.controller_lr),
        "schedule": schedule_name,
        "warmup_batches": int(args.lr_warmup_batches),
        "warmup_init_fraction": float(args.lr_warmup_init_fraction),
        "warmup_initial_learning_rate": float(args.controller_lr)
        * float(args.lr_warmup_init_fraction),
        "constant_lr_iterations": int(args.lr_warmup_batches),
        "cosine_annealing_alpha": float(args.lr_cosine_alpha),
        "final_learning_rate": float(args.controller_lr) * float(args.lr_cosine_alpha),
        "weight_decay": 0.0,
        "gradient_clip_norm": (
            None if args.gradient_clip_norm is None else float(args.gradient_clip_norm)
        ),
        "gradient_clip_kind": (None if args.gradient_clip_norm is None else "global_norm"),
        "training_diagnostics": _training_diagnostics_metadata(
            args,
            Path(args.output_dir),
        ),
    }


def _perturbation_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.perturbation_training, "enabled", False))


def _target_relative_multitarget_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.target_relative_multitarget, "enabled", False))


def _initial_hidden_encoder_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.model, "initial_hidden_encoder", False))


def _initial_hidden_encoder_config(
    *,
    enabled: bool,
    hidden_size: int,
    context_dim: int = CS_H0_CONTEXT_DIM,
    context_basis: str = "target_relative_delayed_feedback",
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "architecture": "affine",
        "context_source": "first_controller_visible_target_relative_delayed_feedback",
        "context_basis": context_basis,
        "context_shape": [int(context_dim)],
        "output_shape": [int(hidden_size)],
        "initialization": CS_H0_ENCODER_INIT,
        "initialization_note": (
            "Exact zero affine weights and bias preserve the zero-H0 baseline at "
            "initialization while remaining trainable through ordinary rollout loss."
        ),
        "separate_hidden_width": None,
        "teacher_or_jacobian_supervision": False,
        "plant_live_preview": False,
        "delayed_reach": False,
    }


def _initial_hidden_encoder_metadata(hps: TreeNamespace) -> dict[str, Any]:
    config = getattr(hps.model, "initial_hidden_encoder_config", None)
    if config is None:
        return _initial_hidden_encoder_config(
            enabled=_initial_hidden_encoder_enabled(hps),
            hidden_size=int(hps.model.hidden_size),
            context_dim=_controller_feedback_dim(hps),
            context_basis=_controller_feedback_basis(hps),
        )
    return _plain(config)


def _broad_epsilon_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "broad_epsilon_training", None), "enabled", False))


def _broad_epsilon_pgd_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "broad_epsilon_pgd_training", None), "enabled", False))


def _adaptive_epsilon_curriculum_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(args, "adaptive_epsilon_curriculum", False))
    cfg = {
        "enabled": enabled,
        "damage_schedule": {
            "kind": "linear_ramp_then_cosine_anneal",
            "start": float(args.adaptive_epsilon_damage_start),
            "peak": float(args.adaptive_epsilon_damage_peak),
            "final": float(args.adaptive_epsilon_damage_final),
            "ramp_batches": int(args.adaptive_epsilon_damage_ramp_batches),
            "anneal_batches": int(args.adaptive_epsilon_damage_anneal_batches),
        },
        "lambda_update": {
            "interval_batches": int(args.adaptive_epsilon_update_interval_batches),
            "ema_alpha": float(args.adaptive_epsilon_ema_alpha),
            "eta": float(args.adaptive_epsilon_eta),
            "deadband_frac": float(args.adaptive_epsilon_deadband_frac),
            "lambda_min": float(args.adaptive_epsilon_lambda_min),
            "lambda_max": (
                None
                if args.adaptive_epsilon_lambda_max is None
                else float(args.adaptive_epsilon_lambda_max)
            ),
            "max_log_step": float(args.adaptive_epsilon_max_log_step),
        },
        "outer_adversarial_weight": {
            "kind": "linear_ramp_then_hold",
            "start": float(args.adaptive_epsilon_outer_weight_start),
            "final": float(args.adaptive_epsilon_outer_weight_final),
            "ramp_batches": int(args.adaptive_epsilon_outer_weight_ramp_batches),
            "applies_to": "optimized_direct_epsilon_loss_only",
            "perturbation_bank_policy": "orthogonal_unweighted_by_outer_adversarial_weight",
        },
    }
    if not enabled:
        return cfg
    damage = cfg["damage_schedule"]
    if damage["ramp_batches"] < 0 or damage["anneal_batches"] < 0:
        raise ValueError("Adaptive epsilon damage schedule batch counts must be nonnegative.")
    if damage["start"] < 0.0 or damage["peak"] < 0.0 or damage["final"] < 0.0:
        raise ValueError("Adaptive epsilon damage targets must be nonnegative.")
    if cfg["lambda_update"]["interval_batches"] < 1:
        raise ValueError("Adaptive epsilon update interval must be positive.")
    if not 0.0 < cfg["lambda_update"]["ema_alpha"] <= 1.0:
        raise ValueError("Adaptive epsilon EMA alpha must be in (0, 1].")
    if cfg["lambda_update"]["eta"] <= 0.0:
        raise ValueError("Adaptive epsilon eta must be positive.")
    if cfg["lambda_update"]["deadband_frac"] < 0.0:
        raise ValueError("Adaptive epsilon deadband fraction must be nonnegative.")
    if cfg["lambda_update"]["lambda_min"] <= 0.0:
        raise ValueError("Adaptive epsilon lambda_min must be positive.")
    if (
        cfg["lambda_update"]["lambda_max"] is not None
        and cfg["lambda_update"]["lambda_max"] <= cfg["lambda_update"]["lambda_min"]
    ):
        raise ValueError("Adaptive epsilon lambda_max must be greater than lambda_min.")
    if cfg["lambda_update"]["max_log_step"] <= 0.0:
        raise ValueError("Adaptive epsilon max_log_step must be positive.")
    outer = cfg["outer_adversarial_weight"]
    if outer["ramp_batches"] < 0:
        raise ValueError("Adaptive epsilon outer-weight ramp batches must be nonnegative.")
    if not 0.0 <= outer["start"] <= 1.0 or not 0.0 <= outer["final"] <= 1.0:
        raise ValueError("Adaptive epsilon outer adversarial weights must lie in [0, 1].")
    return cfg


def _adaptive_epsilon_curriculum_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "adaptive_epsilon_curriculum", None), "enabled", False))


def _adaptive_epsilon_damage_target(config: Any, batch_index: int) -> float:
    schedule = getattr(config, "damage_schedule")
    start = float(schedule.start)
    peak = float(schedule.peak)
    final = float(schedule.final)
    ramp_batches = int(schedule.ramp_batches)
    anneal_batches = int(schedule.anneal_batches)
    batch = max(0, int(batch_index))
    if ramp_batches > 0 and batch < ramp_batches:
        frac = batch / float(ramp_batches)
        return start + frac * (peak - start)
    if anneal_batches > 0 and batch < ramp_batches + anneal_batches:
        frac = (batch - ramp_batches) / float(anneal_batches)
        cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
        return final + cosine * (peak - final)
    return final


def _adaptive_epsilon_outer_weight(config: Any, batch_index: int) -> float:
    schedule = getattr(config, "outer_adversarial_weight")
    start = float(schedule.start)
    final = float(schedule.final)
    ramp_batches = int(schedule.ramp_batches)
    batch = max(0, int(batch_index))
    if ramp_batches < 1:
        return final
    frac = min(1.0, batch / float(ramp_batches))
    return start + frac * (final - start)


def _initial_adaptive_epsilon_state(hps: TreeNamespace) -> AdaptiveEpsilonState | None:
    if not _adaptive_epsilon_curriculum_enabled(hps):
        return None
    cfg = config_from_broad_epsilon_pgd_hps(hps.broad_epsilon_pgd_training)
    if cfg.soft_energy_lambda is None:
        raise ValueError("Adaptive epsilon curriculum requires a resolved positive energy lambda.")
    return AdaptiveEpsilonState(lambda_value=float(cfg.soft_energy_lambda))


def _policy_adversary_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "policy_adversary_training", None), "enabled", False))


def _policy_adversary_policy_class(hps: TreeNamespace) -> str:
    if not _policy_adversary_training_enabled(hps):
        return "disabled"
    return config_from_policy_adversary_hps(hps.policy_adversary_training).policy_class


def _broad_epsilon_pgd_mechanism(hps: TreeNamespace) -> str:
    pgd = getattr(hps, "broad_epsilon_pgd_training", None)
    return str(getattr(pgd, "adversary_mechanism", BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM))


def _broad_epsilon_pgd_finite_policy_inputs(hps: TreeNamespace) -> list[str]:
    if not _broad_epsilon_pgd_training_enabled(hps):
        return []
    mechanism = _broad_epsilon_pgd_mechanism(hps)
    if mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM:
        return []
    keys = [FINITE_POLICY_GAINS_INPUT]
    mechanism_payload = getattr(getattr(hps, "broad_epsilon_pgd_training", None), "mechanism", None)
    has_bias = bool(
        getattr(getattr(mechanism_payload, "required_policy_contract", None), "has_bias", False)
    )
    if has_bias:
        keys.append(FINITE_POLICY_BIAS_INPUT)
    return keys


def _adversarial_phase(hps: TreeNamespace) -> str:
    if _policy_adversary_training_enabled(hps):
        policy_class = _policy_adversary_policy_class(hps)
        if policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP:
            return "learned_memoryless_policy_adversary"
        return f"learned_finite_{policy_class}_policy_adversary"
    if _broad_epsilon_pgd_training_enabled(hps):
        mechanism = _broad_epsilon_pgd_mechanism(hps)
        if mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM:
            return "broad_epsilon_pgd_direct_epsilon"
        return f"broad_epsilon_pgd_live_finite_policy_{mechanism}"
    return "none"


def _nominal_only(hps: TreeNamespace) -> bool:
    return (
        not _perturbation_training_enabled(hps)
        and not _broad_epsilon_training_enabled(hps)
        and not _broad_epsilon_pgd_training_enabled(hps)
        and not _policy_adversary_training_enabled(hps)
        and not _target_relative_multitarget_enabled(hps)
        and not _initial_hidden_encoder_enabled(hps)
        and not _delayed_reach_enabled(hps)
    )


def _training_mode(hps: TreeNamespace) -> str:
    if _target_relative_multitarget_enabled(hps):
        parts = [
            (
                TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE
                if _initial_hidden_encoder_enabled(hps)
                else TARGET_RELATIVE_MULTITARGET_TRAINING_MODE
            )
        ]
        if _broad_epsilon_training_enabled(hps):
            parts.append(BROAD_EPSILON_TRAINING_MODE)
        if _broad_epsilon_pgd_training_enabled(hps):
            parts.append(BROAD_EPSILON_PGD_TRAINING_MODE)
        if _policy_adversary_training_enabled(hps):
            parts.append(POLICY_ADVERSARY_TRAINING_MODE)
        if _perturbation_training_enabled(hps):
            parts.append(PERTURBATION_TRAINING_MODE)
        if _delayed_reach_enabled(hps):
            parts.insert(0, DELAYED_REACH_TRAINING_MODE)
        return "+".join(parts)
    if _perturbation_training_enabled(hps):
        parts = [PERTURBATION_TRAINING_MODE]
    else:
        parts = []
    if _broad_epsilon_training_enabled(hps):
        parts.append(BROAD_EPSILON_TRAINING_MODE)
    if _broad_epsilon_pgd_training_enabled(hps):
        parts.append(BROAD_EPSILON_PGD_TRAINING_MODE)
    if _policy_adversary_training_enabled(hps):
        parts.append(POLICY_ADVERSARY_TRAINING_MODE)
    return "+".join(parts) if parts else "nominal"


def _controller_feedback_basis(hps: TreeNamespace) -> str:
    if _target_relative_multitarget_enabled(hps):
        if bool(getattr(hps.target_relative_multitarget, "force_filter_feedback", False)):
            return "target_relative_delayed_feedback_plus_force_filter"
        return "target_relative_delayed_feedback"
    return "raw_delayed_position_velocity"


def _controller_feedback_dim(hps: TreeNamespace) -> int:
    if _target_relative_multitarget_enabled(hps):
        return (
            6
            if bool(getattr(hps.target_relative_multitarget, "force_filter_feedback", False))
            else 4
        )
    return 4


def _validation_bins_metadata(hps: TreeNamespace) -> dict[str, Any]:
    if _target_relative_multitarget_enabled(hps):
        return target_relative_validation_manifest(hps.target_relative_multitarget)
    return validation_bin_manifest(hps.perturbation_training)


def _training_distribution_metadata(hps: TreeNamespace) -> dict[str, Any]:
    config = hps.perturbation_training
    target_config = hps.target_relative_multitarget
    if _target_relative_multitarget_enabled(hps):
        target_payload = target_config.target_distribution
        h0 = _initial_hidden_encoder_metadata(hps)
        return {
            "mode": _training_mode(hps),
            "training_axes": {
                "target_relative_multitarget": True,
                "delayed_reach": _delayed_reach_enabled(hps),
                "initial_hidden_encoder": bool(h0["enabled"]),
                "calibrated_perturbation_training": _perturbation_training_enabled(hps),
                "broad_full_state_epsilon_training": _broad_epsilon_training_enabled(hps),
                "broad_full_state_epsilon_pgd_training": (_broad_epsilon_pgd_training_enabled(hps)),
                "policy_adversary_training": _policy_adversary_training_enabled(hps),
                "force_filter_feedback": bool(
                    getattr(target_config, "force_filter_feedback", False)
                ),
            },
            "fixed_target_only": False,
            "target_stream": {
                "status": "consumed_as_static_target_relative_feedback",
                "input_port": "target",
                "contract": _plain(target_config.input_contract),
            },
            "go_cue_stream": (
                _plain(hps.delayed_reach.go_cue_input)
                if _delayed_reach_enabled(hps)
                else {"enabled": False}
            ),
            "delayed_reach": _plain(hps.delayed_reach),
            "initial_hidden_encoder": h0,
            "force_filter_feedback": _plain(target_config.force_filter_feedback),
            "broad_epsilon_training": (
                _plain(hps.broad_epsilon_training)
                if _broad_epsilon_training_enabled(hps)
                else {"enabled": False}
            ),
            "broad_epsilon_pgd_training": (
                _plain(hps.broad_epsilon_pgd_training)
                if _broad_epsilon_pgd_training_enabled(hps)
                else {"enabled": False}
            ),
            "policy_adversary_training": (
                _plain(hps.policy_adversary_training)
                if _policy_adversary_training_enabled(hps)
                else {"enabled": False}
            ),
            "perturbation_training": (
                _plain(hps.perturbation_training)
                if _perturbation_training_enabled(hps)
                else {"enabled": False}
            ),
            "target_distribution": _plain(target_payload),
            "original_target_anchor_m": _plain(target_payload.original_target_anchor_m),
            "seen_targets_m": _plain(target_payload.seen_targets_m),
            "held_out_targets_m": _plain(target_payload.held_out_targets_m),
            "validation_bins": _plain(target_config.validation_bins),
            "perturbation_mixture_emphasis": _plain(target_config.perturbation_mixture_emphasis),
            "checkpoint_selection_role": ("target_relative_multitarget_rollout_validation"),
            "nominal_quality_role": "original_anchor_and_seen_held_out_targets_reported",
            "controller_internal_mutation": False,
            "adversarial_phase": _adversarial_phase(hps),
        }
    if (
        not bool(getattr(config, "enabled", False))
        and not _broad_epsilon_training_enabled(hps)
        and not _broad_epsilon_pgd_training_enabled(hps)
        and not _policy_adversary_training_enabled(hps)
    ):
        return {
            "mode": "nominal",
            "fixed_target_only": True,
            "target_stream": "not_consumed",
        }
    return {
        "mode": str(getattr(config, "mode", PERTURBATION_TRAINING_MODE)),
        "legacy_mode": LEGACY_PERTURBATION_TRAINING_MODE,
        "fixed_target_only": True,
        "target_stream": {
            "status": "not_consumed",
            "reason": (
                "Current C&S GRU input is scalar external input plus delayed "
                "feedback; no target-position stream is supplied to the controller."
            ),
        },
        "mixture": {
            "nominal_fraction": float(config.nominal_fraction),
            "single_family_fraction": float(config.single_fraction),
            "mild_combined_fraction": float(config.combined_fraction),
            "combined_amplitude_scale": float(config.combined_amplitude_scale),
            "sampling": (
                "prng_driven_signed_random_axes_components_calibrated_timing_levels"
                if bool(getattr(config, "calibrated_timing", False))
                else "prng_driven_signed_random_axes_components_timings_levels"
            ),
            "calibrated_timing": bool(getattr(config, "calibrated_timing", False)),
            "movement_age_timing": bool(getattr(config, "movement_age_timing", False)),
            "physical_level": str(getattr(config, "physical_level", "moderate")),
            "physical_level_fraction_of_reach": float(
                getattr(config, "physical_level_fraction_of_reach", 0.10)
            ),
        },
        "mild_combined_families": ["initial_position", "command_input"],
        "single_family_bins": list(config.single_family_bins),
        "validation_bins": list(config.validation_bins),
        "timing_basis": _plain(config.timing_basis),
        "timing_bins": _plain(config.timing_bins),
        "calibrated_levels": _plain(config.mixture_semantics.calibrated_levels),
        "broad_epsilon_training": (
            _plain(hps.broad_epsilon_training)
            if _broad_epsilon_training_enabled(hps)
            else {"enabled": False}
        ),
        "broad_epsilon_pgd_training": (
            _plain(hps.broad_epsilon_pgd_training)
            if _broad_epsilon_pgd_training_enabled(hps)
            else {"enabled": False}
        ),
        "policy_adversary_training": (
            _plain(hps.policy_adversary_training)
            if _policy_adversary_training_enabled(hps)
            else {"enabled": False}
        ),
        "perturbation_training": (
            _plain(hps.perturbation_training)
            if _perturbation_training_enabled(hps)
            else {"enabled": False}
        ),
        "checkpoint_selection_role": "generalized_held_out_perturbation_validation",
        "nominal_quality_role": "reported_quality_sidecar_gate",
        "controller_internal_mutation": False,
        "adversarial_phase": _adversarial_phase(hps),
    }


def _training_diagnostics_metadata(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    enabled = _training_diagnostics_enabled(args)
    return {
        "enabled": enabled,
        "default_enabled": True,
        "opt_out_flag": "--no-training-diagnostics",
        "schema_version": f"{SCHEMA_VERSION}.training_diagnostics.v1",
        "format": "npz+json_manifest",
        "sidecar_path": str(Path(output_dir) / TRAINING_DIAGNOSTICS_NPZ) if enabled else None,
        "manifest_path": (
            str(Path(output_dir) / TRAINING_DIAGNOSTICS_MANIFEST) if enabled else None
        ),
        "source": (
            "optimizer_state plus Feedbax TaskTrainer history; no raw gradients, "
            "batches, or activations are persisted"
        ),
        "scalar_groups": [
            "optimizer_gradient_norm_pre_clip",
            "optimizer_gradient_clipped",
            "optimizer_clipping_fraction",
            "optimizer_update_parameter_norm_ratio",
            "optimizer_learning_rate",
            "train_loss_terms",
            "validation_loss_terms",
            "pgd_broad_epsilon_inner_maximizer",
            "policy_adversary_inner_optimizer",
            "adaptive_epsilon_curriculum",
        ],
    }


def _run_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    if args.full_train:
        return "full_train"
    return "spec_write"


def _commit_volume(volume_commit: VolumeCommit | None) -> None:
    if volume_commit is not None:
        volume_commit()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    write_compact_json(path, payload, atomic=True)


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


def _optimizer_diagnostics_arrays(
    optimizer_state: Any,
    *,
    start_batches: int = 0,
    completed_batches: int,
) -> dict[str, np.ndarray]:
    """Return scalar optimizer diagnostics for one completed batch range."""

    arrays: dict[str, np.ndarray] = {}
    gradient_state = _find_diagnostics_state(
        optimizer_state,
        GradientDiagnosticsState,
    )
    update_state = _find_diagnostics_state(
        optimizer_state,
        UpdateDiagnosticsState,
    )
    if gradient_state is not None:
        arrays.update(
            _gradient_diagnostics_arrays(
                gradient_state,
                completed_batches,
                start_batches=start_batches,
            )
        )
    if update_state is not None:
        arrays.update(
            _update_diagnostics_arrays(
                update_state,
                completed_batches,
                start_batches=start_batches,
            )
        )
    return arrays


def _diagnostic_series(array: Any, completed_batches: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 0:
        return values.reshape((1,))
    if values.ndim == 1:
        return values[:completed_batches]
    if values.shape[0] == completed_batches:
        return values[:completed_batches]
    return np.moveaxis(values[..., :completed_batches], -1, 0)


def _diagnostic_series_range(array: Any, start_batches: int, completed_batches: int) -> np.ndarray:
    return _diagnostic_series(array, completed_batches)[start_batches:completed_batches]


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


def _make_policy_adversary_pre_step(policy: Any, config: Any) -> Callable:
    """Return a stable PyTree hook applying the current learned policy adversary."""

    return make_policy_adversary_pre_step(policy, config)


def _run_adaptive_epsilon_training_chunk(
    *,
    trainer: TaskTrainer,
    task: Any,
    model: Any,
    optimizer_state: Any,
    adaptive_state: AdaptiveEpsilonState | None,
    hps: TreeNamespace,
    where_train: Callable[[Any], Any],
    key: Any,
    start_batch: int,
    chunk_batches: int,
    log_progress: bool,
) -> tuple[Any, Any, Any, AdaptiveEpsilonState, dict[str, np.ndarray]]:
    """Run one paired clean/adversarial adaptive direct-epsilon training chunk."""

    if chunk_batches < 1:
        raise ValueError("chunk_batches must be positive")
    if adaptive_state is None:
        adaptive_state = _initial_adaptive_epsilon_state(hps)
    if adaptive_state is None:
        raise ValueError("Adaptive epsilon state is required for adaptive training.")

    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    where_train_spec = filter_spec_leaves(model, where_train)
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(optimizer_state)

    def _ensemble_in_axis(leaf):
        if _is_replicate_axis_array(leaf, n_replicates):
            return 0
        return None

    flat_model_arr_spec = jt.map(_ensemble_in_axis, flat_model)
    train_step = eqx.filter_vmap(
        _adaptive_epsilon_train_step,
        in_axes=(
            None,
            None,
            None,
            flat_model_arr_spec,
            None,
            0,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
        ),
        out_axes=(
            eqx.if_array(0),
            0,
            flat_model_arr_spec,
            eqx.if_array(0),
            eqx.if_array(0),
            eqx.if_array(0),
        ),
    )
    history = init_task_trainer_history(
        task.loss_func,
        chunk_batches,
        n_replicates,
        ensembled=True,
        ensemble_random_trials=True,
        start_batch=0,
        task=task,
        batch_size=batch_size,
        model=model,
        where_train=where_train,
    )
    keys = jr.split(key, chunk_batches)
    progress_every = batch_log_every(int(hps.n_batches_condition))
    chunk_started = time.perf_counter()
    diagnostic_series: dict[str, list[np.ndarray]] = {}

    for local_batch in range(chunk_batches):
        global_batch = start_batch + local_batch
        key_train, key_eval = jr.split(keys[local_batch], 2)
        target_damage = _adaptive_epsilon_damage_target(
            hps.adaptive_epsilon_curriculum,
            global_batch,
        )
        outer_weight = _adaptive_epsilon_outer_weight(
            hps.adaptive_epsilon_curriculum,
            global_batch,
        )
        batch_info = BatchInfo(
            size=batch_size,
            start=jnp.asarray(0),
            current=jnp.asarray(global_batch),
            total=jnp.asarray(hps.n_batches_condition),
        )
        key_train = jr.split(key_train, n_replicates)
        losses, _trial_specs, flat_model, flat_opt_state, _grads, diagnostics = train_step(
            task,
            task.loss_func,
            batch_info,
            flat_model,
            treedef_model,
            flat_opt_state,
            treedef_opt_state,
            where_train_spec,
            trainer.optimizer,
            hps.broad_epsilon_pgd_training,
            key_train,
            jnp.asarray(adaptive_state.lambda_value, dtype=jnp.float32),
            jnp.asarray(outer_weight, dtype=jnp.float32),
        )
        damage_raw = float(np.asarray(jax.device_get(jnp.mean(diagnostics["damage_raw"]))))
        adaptive_state, update_diagnostics = _update_adaptive_epsilon_state(
            adaptive_state,
            hps.adaptive_epsilon_curriculum,
            batch_index=global_batch,
            target_damage=target_damage,
            measured_damage=damage_raw,
        )
        host_diagnostics = {
            name: np.asarray(jax.device_get(value))
            for name, value in diagnostics.items()
            if eqx.is_array(value) or np.isscalar(value)
        }
        host_diagnostics.update(update_diagnostics)
        host_diagnostics["target_damage"] = np.asarray(target_damage, dtype=np.float32)
        host_diagnostics["outer_weight"] = np.asarray(outer_weight, dtype=np.float32)
        host_diagnostics["lambda_value"] = np.asarray(
            adaptive_state.lambda_value,
            dtype=np.float32,
        )
        host_diagnostics["global_batch"] = np.asarray(global_batch, dtype=np.float32)
        _append_adaptive_epsilon_diagnostics(diagnostic_series, host_diagnostics)

        history = eqx.tree_at(
            lambda history: history.loss,
            history,
            tree_set(
                history.loss,
                losses.map(lambda arr: jnp.mean(arr, axis=-1)),
                local_batch,
            ),
        )
        opt_state_for_history = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)
        if (hyperparams := getattr(opt_state_for_history, "hyperparams", None)) is not None:
            history = eqx.tree_at(
                lambda history: history.learning_rate,
                history,
                history.learning_rate.at[local_batch].set(hyperparams["learning_rate"]),
            )
        if log_progress and should_log_batch(
            global_batch,
            int(hps.n_batches_condition),
            every=progress_every,
        ):
            loss_mean = losses.map(jnp.mean)
            print(
                format_batch_line(
                    "adaptive_epsilon",
                    global_batch,
                    int(hps.n_batches_condition),
                    loss=float(jax.device_get(loss_mean.total)),
                    damage=damage_raw,
                    target=target_damage,
                    lambda_value=float(adaptive_state.lambda_value),
                    outer=outer_weight,
                    elapsed=time.perf_counter() - chunk_started,
                ),
                flush=True,
            )

    model = jtu.tree_unflatten(treedef_model, flat_model)
    optimizer_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)
    states_validation, losses_validation = task.eval_ensemble_with_loss(
        model,
        n_replicates,
        key_eval,
        ensemble_random_trials=True,
    )
    del states_validation
    history = eqx.tree_at(
        lambda history: history.loss_validation,
        history,
        tree_set(
            history.loss_validation,
            losses_validation.map(lambda arr: jnp.mean(arr, axis=-1)),
            chunk_batches - 1,
        ),
    )
    return (
        model,
        history,
        optimizer_state,
        adaptive_state,
        _adaptive_epsilon_diagnostics_arrays(diagnostic_series),
    )


@eqx.filter_jit
def _adaptive_epsilon_train_step(
    task: Any,
    loss_func: Any,
    batch_info: BatchInfo,
    flat_model: Any,
    treedef_model: Any,
    flat_opt_state: Any,
    treedef_opt_state: Any,
    where_train_spec: Any,
    optimizer: optax.GradientTransformation,
    pgd_config: Any,
    key: Any,
    energy_lambda: Any,
    outer_weight: Any,
) -> tuple[Any, Any, Any, Any, Any, dict[str, jnp.ndarray]]:
    key_trials, key_init, key_model = jr.split(key, 3)
    keys_trials = jr.split(key_trials, batch_info.size)
    keys_init = jr.split(key_init, batch_info.size)
    keys_model = jr.split(key_model, batch_info.size)
    trial_specs = eqx.filter_vmap(
        partial(
            task.get_train_trial_with_intervenor_params,
            batch_info=batch_info,
        )
    )(keys_trials)
    model = jtu.tree_unflatten(treedef_model, flat_model)
    adv_specs, inner_diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        task,
        model,
        trial_specs,
        loss_func,
        keys_model,
        config=pgd_config,
        soft_energy_lambda_override=energy_lambda,
        return_diagnostics=True,
    )
    adv_specs = jt.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        adv_specs,
    )
    init_states = eqx.filter_vmap(lambda _: init_state_from_component(model))(keys_init)
    init_states = eqx.filter_vmap(
        lambda state, trial_spec: _apply_trial_spec_initial_state(model, state, trial_spec)
    )(init_states, trial_specs)
    diff_model, static_model = eqx.partition(model, where_train_spec)
    opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

    def paired_loss(current_diff_model):
        current_model = eqx.combine(current_diff_model, static_model)
        clean_states = _eval_trial_specs_for_training(
            current_model,
            trial_specs,
            init_states,
            keys_model,
        )
        adv_states = _eval_trial_specs_for_training(
            current_model,
            adv_specs,
            init_states,
            keys_model,
        )
        clean_losses = loss_func(clean_states, trial_specs, current_model)
        adv_losses = loss_func(adv_states, adv_specs, current_model)
        weighted_losses = _weighted_loss_tree(clean_losses, adv_losses, outer_weight)
        diagnostics = {
            "damage_raw": jnp.asarray(adv_losses.total - clean_losses.total),
            "clean_loss_total": jnp.asarray(clean_losses.total),
            "adversarial_loss_total": jnp.asarray(adv_losses.total),
            "weighted_loss_total": jnp.asarray(weighted_losses.total),
            "energy_lambda_used": jnp.asarray(energy_lambda, dtype=jnp.float32),
            "outer_weight_used": jnp.asarray(outer_weight, dtype=jnp.float32),
        }
        diagnostics.update({f"inner_{name}": value for name, value in inner_diagnostics.items()})
        return weighted_losses.total, (weighted_losses, diagnostics)

    (_, (losses, diagnostics)), grads = eqx.filter_value_and_grad(
        paired_loss,
        has_aux=True,
    )(diff_model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = project_component_parameters(model)
    flat_model = jtu.tree_leaves(model)
    flat_opt_state = jtu.tree_leaves(opt_state)
    return losses, adv_specs, flat_model, flat_opt_state, grads, diagnostics


def _weighted_loss_tree(clean_losses: Any, adv_losses: Any, outer_weight: Any) -> Any:
    def combine(clean_value: Any, adv_value: Any) -> Any:
        if eqx.is_array(clean_value) and eqx.is_array(adv_value):
            return (1.0 - outer_weight) * clean_value + outer_weight * adv_value
        return clean_value

    return jt.map(combine, clean_losses, adv_losses)


def _apply_trial_spec_initial_state(model: Any, state: Any, trial_spec: Any) -> Any:
    for where_substate, init_substate in trial_spec.inits.items():
        path = where_key_to_path(where_substate)
        state = set_state_by_path(model, state, path, init_substate)

    if trial_spec.intervene:
        intervention_indices = model.intervention_state_indices()
        for label, params in trial_spec.intervene.items():
            if label not in intervention_indices:
                raise ValueError(f"Unknown intervention label '{label}'")
            idx = intervention_indices[label]
            current = state.get(idx)

            def _merge_leaf(p, c):
                if isinstance(p, TimeSeriesParam):
                    return c
                if p is None:
                    return c
                return p

            merged = jt.map(
                _merge_leaf,
                params,
                current,
                is_leaf=lambda x: x is None or isinstance(x, TimeSeriesParam),
            )
            state = _state_set_matching_dtypes(state, idx, merged)

    return model.state_consistency_update(state)


def _eval_trial_specs_for_training(model: Any, trial_specs: Any, init_states: Any, keys: Any) -> Any:
    def _run_trial(trial_spec, init_state, key):
        inputs = prepare_inputs(model, trial_spec.inputs)
        n_steps = infer_n_steps(inputs, getattr(trial_spec, "timeline", None))
        if trial_spec.intervene:
            intervene_inputs = _extract_intervene_inputs(trial_spec.intervene, model)
            if intervene_inputs:
                inputs = {**inputs, **intervene_inputs}
        _, _, state_history = run_component(
            model,
            inputs,
            init_state,
            key=key,
            n_steps=n_steps,
        )
        return jt.map(lambda x: x[1:] if x is not None else x, state_history)

    return eqx.filter_vmap(_run_trial)(trial_specs, init_states, keys)


def _extract_intervene_inputs(intervene: Any, model: Any) -> dict[str, Any]:
    indices = model.intervention_state_indices()
    result = {}
    for label, params in intervene.items():
        if label not in indices:
            continue
        idx = indices[label]
        tv_params = extract_timeseries_params(params, idx.init)
        if tv_params is not None:
            result[f"intervene:{label}"] = tv_params
    return result


def _cast_to_state_dtypes(new_value: Any, current_value: Any) -> Any:
    def _cast_leaf(new_leaf, current_leaf):
        if eqx.is_array(new_leaf) and eqx.is_array(current_leaf):
            if getattr(new_leaf, "dtype", None) != getattr(current_leaf, "dtype", None):
                return jnp.asarray(new_leaf, dtype=current_leaf.dtype)
        return new_leaf

    return jt.map(_cast_leaf, new_value, current_value)


def _state_set_matching_dtypes(state: Any, idx: Any, new_value: Any) -> Any:
    current_value = state.get(idx)
    return state.set(idx, _cast_to_state_dtypes(new_value, current_value))


def _update_adaptive_epsilon_state(
    state: AdaptiveEpsilonState,
    config: Any,
    *,
    batch_index: int,
    target_damage: float,
    measured_damage: float,
) -> tuple[AdaptiveEpsilonState, dict[str, np.ndarray]]:
    update_cfg = getattr(config, "lambda_update")
    alpha = float(update_cfg.ema_alpha)
    damage_ema = (
        float(measured_damage)
        if state.damage_ema is None
        else (1.0 - alpha) * float(state.damage_ema) + alpha * float(measured_damage)
    )
    completed_batches = int(batch_index) + 1
    interval = int(update_cfg.interval_batches)
    update_due = completed_batches % interval == 0
    target = float(target_damage)
    relative_error = (
        (damage_ema - target) / max(target, 1e-12)
        if target > 0.0
        else 0.0
    )
    deadband = float(update_cfg.deadband_frac)
    lambda_value = float(state.lambda_value)
    updated = False
    log_step = 0.0
    if update_due and target > 0.0 and abs(relative_error) > deadband:
        eta = float(update_cfg.eta)
        max_log_step = float(update_cfg.max_log_step)
        log_step = max(-max_log_step, min(max_log_step, eta * relative_error))
        lambda_value *= math.exp(log_step)
        lambda_value = max(lambda_value, float(update_cfg.lambda_min))
        lambda_max = getattr(update_cfg, "lambda_max", None)
        if lambda_max is not None:
            lambda_value = min(lambda_value, float(lambda_max))
        updated = True
    next_state = AdaptiveEpsilonState(
        lambda_value=lambda_value,
        damage_ema=damage_ema,
        last_update_batch=int(batch_index) if updated else state.last_update_batch,
        update_count=state.update_count + (1 if updated else 0),
    )
    return next_state, {
        "damage_ema": np.asarray(damage_ema, dtype=np.float32),
        "relative_error": np.asarray(relative_error, dtype=np.float32),
        "lambda_updated": np.asarray(updated, dtype=bool),
        "lambda_log_step": np.asarray(log_step, dtype=np.float32),
        "update_due": np.asarray(update_due, dtype=bool),
        "update_count": np.asarray(next_state.update_count, dtype=np.float32),
    }


def _append_adaptive_epsilon_diagnostics(
    series: dict[str, list[np.ndarray]],
    diagnostics: dict[str, np.ndarray],
) -> None:
    for name, value in diagnostics.items():
        series.setdefault(name, []).append(np.asarray(value))


def _adaptive_epsilon_diagnostics_arrays(
    series: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        f"adaptive_epsilon_{name}": np.stack(values, axis=0)
        for name, values in sorted(series.items())
    }


def _run_policy_adversary_training_chunk(
    *,
    trainer: TaskTrainer,
    task: Any,
    model: Any,
    optimizer_state: Any,
    adversary_policy: Any,
    adversary_optimizer_state: Any,
    adversary_optimizer: optax.GradientTransformation,
    hps: TreeNamespace,
    where_train: Callable[[Any], Any],
    key: Any,
    start_batch: int,
    chunk_batches: int,
    log_progress: bool,
) -> tuple[Any, Any, Any, Any, Any, dict[str, np.ndarray]]:
    """Run a policy-adversary chunk without per-batch trainer re-entry."""

    if chunk_batches < 1:
        raise ValueError("chunk_batches must be positive")
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    where_train_spec = filter_spec_leaves(model, where_train)
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(optimizer_state)

    def _ensemble_in_axis(leaf):
        if eqx.is_array(leaf) and leaf.ndim > 0 and leaf.shape[0] == n_replicates:
            return 0
        return None

    flat_model_arr_spec = jt.map(_ensemble_in_axis, flat_model)
    train_step = eqx.filter_vmap(
        trainer._train_step,
        in_axes=(
            None,
            None,
            None,
            flat_model_arr_spec,
            None,
            0,
            None,
            None,
            None,
            0,
            None,
            None,
        ),
        out_axes=(
            eqx.if_array(0),
            0,
            flat_model_arr_spec,
            eqx.if_array(0),
            eqx.if_array(0),
        ),
    )
    history = init_task_trainer_history(
        task.loss_func,
        chunk_batches,
        n_replicates,
        ensembled=True,
        ensemble_random_trials=True,
        start_batch=0,
        task=task,
        batch_size=batch_size,
        model=model,
        where_train=where_train,
    )
    keys = jr.split(key, chunk_batches)
    progress_every = batch_log_every(int(hps.n_batches_condition))
    chunk_started = time.perf_counter()
    diagnostics: dict[str, np.ndarray] = {}

    for local_batch in range(chunk_batches):
        global_batch = start_batch + local_batch
        key_adversary, key_controller, key_eval = jr.split(keys[local_batch], 3)
        model_for_adversary = jtu.tree_unflatten(treedef_model, flat_model)
        (
            adversary_policy,
            adversary_optimizer_state,
            diagnostics,
        ) = _advance_policy_adversary(
            adversary_policy,
            adversary_optimizer_state,
            adversary_optimizer,
            task,
            model_for_adversary,
            hps,
            key=key_adversary,
            batch_index=global_batch,
        )
        pre_step_fn = _make_policy_adversary_pre_step(
            adversary_policy,
            hps.policy_adversary_training,
        )
        batch_info = BatchInfo(
            size=batch_size,
            start=jnp.asarray(0),
            current=jnp.asarray(local_batch),
            total=jnp.asarray(chunk_batches),
        )
        key_train = jr.split(key_controller, n_replicates)
        losses, _trial_specs, flat_model, flat_opt_state, _grads = train_step(
            task,
            task.loss_func,
            batch_info,
            flat_model,
            treedef_model,
            flat_opt_state,
            treedef_opt_state,
            where_train_spec,
            [],
            key_train,
            None,
            pre_step_fn,
        )
        history = eqx.tree_at(
            lambda history: history.loss,
            history,
            tree_set(
                history.loss,
                losses.map(lambda arr: jnp.mean(arr, axis=-1)),
                local_batch,
            ),
        )
        opt_state_for_history = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)
        if (hyperparams := getattr(opt_state_for_history, "hyperparams", None)) is not None:
            history = eqx.tree_at(
                lambda history: history.learning_rate,
                history,
                history.learning_rate.at[local_batch].set(hyperparams["learning_rate"]),
            )
        if log_progress and should_log_batch(
            global_batch,
            int(hps.n_batches_condition),
            every=progress_every,
        ):
            loss_mean = losses.map(jnp.mean)
            print(
                format_batch_line(
                    "policy_adversary",
                    global_batch,
                    int(hps.n_batches_condition),
                    loss=float(jax.device_get(loss_mean.total)),
                    adv=float(np.asarray(diagnostics.get("adversary_objective", np.nan))),
                    elapsed=time.perf_counter() - chunk_started,
                ),
                flush=True,
            )

    model = jtu.tree_unflatten(treedef_model, flat_model)
    optimizer_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)
    states_validation, losses_validation = task.eval_ensemble_with_loss(
        model,
        n_replicates,
        key_eval,
        ensemble_random_trials=True,
    )
    del states_validation
    history = eqx.tree_at(
        lambda history: history.loss_validation,
        history,
        tree_set(
            history.loss_validation,
            losses_validation.map(lambda arr: jnp.mean(arr, axis=-1)),
            chunk_batches - 1,
        ),
    )
    return (
        model,
        history,
        optimizer_state,
        adversary_policy,
        adversary_optimizer_state,
        diagnostics,
    )


def _advance_policy_adversary(
    policy: Any,
    optimizer_state: Any,
    optimizer: optax.GradientTransformation,
    task: Any,
    model: Any,
    hps: TreeNamespace,
    *,
    key: Any,
    batch_index: int,
) -> tuple[Any, Any, dict[str, np.ndarray]]:
    """Run the persistent policy-adversary ascent steps for one controller batch."""

    policy, optimizer_state, diagnostics = _advance_policy_adversary_compiled(
        policy,
        optimizer_state,
        optimizer,
        task,
        model,
        hps,
        key,
        jnp.asarray(batch_index),
    )
    arrays = {
        name: np.asarray(jax.device_get(value))
        for name, value in diagnostics.items()
        if eqx.is_array(value) or np.isscalar(value)
    }
    return policy, optimizer_state, arrays


@eqx.filter_jit
def _advance_policy_adversary_compiled(
    policy: Any,
    optimizer_state: Any,
    optimizer: optax.GradientTransformation,
    task: Any,
    model: Any,
    hps: TreeNamespace,
    key: Any,
    batch_index: Any,
) -> tuple[Any, Any, dict[str, jnp.ndarray]]:
    """Run the persistent policy-adversary ascent steps in one compiled update."""

    cfg = config_from_policy_adversary_hps(hps.policy_adversary_training)

    def loss_for_policy(candidate_policy):
        objective, diagnostics = _policy_adversary_batch_objective(
            candidate_policy,
            task,
            model,
            hps,
            key=key,
            batch_index=batch_index,
        )
        return -objective, diagnostics

    diagnostics = {}
    for _ in range(int(cfg.n_steps)):
        (_loss, diagnostics), grads = eqx.filter_value_and_grad(
            loss_for_policy,
            has_aux=True,
        )(policy)
        updates, optimizer_state = optimizer.update(
            grads,
            optimizer_state,
            eqx.filter(policy, eqx.is_array),
        )
        policy = eqx.apply_updates(policy, updates)
    diagnostics = {
        **diagnostics,
        "n_ascent_steps": jnp.asarray(cfg.n_steps, dtype=jnp.float32),
        "learning_rate": jnp.asarray(cfg.learning_rate, dtype=jnp.float32),
    }
    return policy, optimizer_state, diagnostics


def _policy_adversary_batch_objective(
    policy: Any,
    task: Any,
    model: Any,
    hps: TreeNamespace,
    *,
    key: Any,
    batch_index: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    batch_info = BatchInfo(
        size=batch_size,
        start=jnp.asarray(0),
        current=jnp.asarray(batch_index),
        total=jnp.asarray(hps.n_batches_condition),
    )
    keys = jr.split(key, n_replicates)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_axis_array(leaf, n_replicates),
    )
    objectives = []
    diagnostics_by_replicate = []
    for replicate_index, key_replicate in enumerate(keys):
        key_trials, _, key_model = jr.split(key_replicate, 3)
        keys_trials = jr.split(key_trials, batch_size)
        keys_model = jr.split(key_model, batch_size)
        trial_specs = eqx.filter_vmap(
            partial(
                task.get_train_trial_with_intervenor_params,
                batch_info=batch_info,
            )
        )(keys_trials)
        replicate_arrays = jt.map(
            lambda leaf: None if leaf is None else leaf[replicate_index],
            model_arrays,
            is_leaf=lambda leaf: leaf is None,
        )
        model_replicate = eqx.combine(replicate_arrays, model_other)
        model_replicate = _with_single_replicate_state_initializers(
            model_replicate,
            n_replicates=n_replicates,
            replicate_index=replicate_index,
        )
        objective, diagnostics = policy_adversary_objective(
            policy,
            task,
            model_replicate,
            trial_specs,
            task.loss_func,
            keys_model,
            hps.policy_adversary_training,
        )
        objectives.append(objective)
        diagnostics_by_replicate.append(diagnostics)
    objective = jnp.mean(jnp.stack(objectives))
    diagnostics = jt.map(lambda *values: jnp.mean(jnp.stack(values)), *diagnostics_by_replicate)
    return objective, diagnostics


def _policy_adversary_diagnostics_arrays(
    diagnostics: dict[str, np.ndarray],
    *,
    batch_index: int,
    chunk_batches: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "policy_adversary_diagnostic_sampled": np.zeros(chunk_batches, dtype=bool),
        "policy_adversary_diagnostic_global_batch": np.full(
            chunk_batches,
            np.nan,
            dtype=np.float32,
        ),
    }
    arrays["policy_adversary_diagnostic_sampled"][-1] = True
    arrays["policy_adversary_diagnostic_global_batch"][-1] = float(batch_index)
    for name, value in diagnostics.items():
        sampled = np.asarray(value)
        if sampled.ndim == 0:
            sampled = sampled.reshape((1,))
        chunk = np.full((chunk_batches, *sampled.shape), np.nan, dtype=sampled.dtype)
        chunk[-1] = sampled
        arrays[f"policy_adversary_{name}"] = chunk
    return arrays


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


def _broad_epsilon_pgd_diagnostics_arrays(
    task: Any,
    model: Any,
    hps: TreeNamespace,
    *,
    key: Any,
    batch_index: int,
    chunk_batches: int,
) -> dict[str, np.ndarray]:
    if not _broad_epsilon_pgd_training_enabled(hps):
        return {}

    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    batch_info = BatchInfo(
        size=batch_size,
        start=0,
        current=int(batch_index),
        total=int(hps.n_batches_condition),
    )

    def diagnostic_for_replicate(model_replicate: Any, key_replicate: Any):
        key_trials, _, key_model = jr.split(key_replicate, 3)
        keys_trials = jr.split(key_trials, batch_size)
        keys_model = jr.split(key_model, batch_size)
        trial_specs = eqx.filter_vmap(
            partial(
                task.get_train_trial_with_intervenor_params,
                batch_info=batch_info,
            )
        )(keys_trials)
        _, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
            task,
            model_replicate,
            trial_specs,
            task.loss_func,
            keys_model,
            hps.broad_epsilon_pgd_training,
            return_diagnostics=True,
        )
        return diagnostics

    keys = jr.split(key, n_replicates)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_axis_array(leaf, n_replicates),
    )
    per_replicate_diagnostics = []
    for replicate_index, key_replicate in enumerate(keys):
        replicate_arrays = jt.map(
            lambda leaf: None if leaf is None else leaf[replicate_index],
            model_arrays,
            is_leaf=lambda leaf: leaf is None,
        )
        model_replicate = eqx.combine(replicate_arrays, model_other)
        model_replicate = _with_single_replicate_state_initializers(
            model_replicate,
            n_replicates=n_replicates,
            replicate_index=replicate_index,
        )
        per_replicate_diagnostics.append(diagnostic_for_replicate(model_replicate, key_replicate))
    diagnostics = jt.map(lambda *values: jnp.stack(values), *per_replicate_diagnostics)
    arrays: dict[str, np.ndarray] = {
        "pgd_broad_epsilon_diagnostic_sampled": np.zeros(chunk_batches, dtype=bool),
        "pgd_broad_epsilon_diagnostic_global_batch": np.full(
            chunk_batches,
            np.nan,
            dtype=np.float32,
        ),
    }
    arrays["pgd_broad_epsilon_diagnostic_sampled"][-1] = True
    arrays["pgd_broad_epsilon_diagnostic_global_batch"][-1] = float(batch_index)
    for name, value in diagnostics.items():
        sampled = np.asarray(jax.device_get(value))
        if sampled.ndim == 0:
            sampled = sampled.reshape((1,))
        chunk = np.full(
            (chunk_batches, *sampled.shape),
            np.nan,
            dtype=sampled.dtype,
        )
        chunk[-1] = sampled
        arrays[f"pgd_broad_epsilon_{name}"] = chunk
    return arrays


def _is_replicate_axis_array(leaf: Any, n_replicates: int) -> bool:
    return (
        eqx.is_array(leaf)
        and leaf.ndim > 0
        and int(getattr(leaf, "shape", (0,))[0]) == int(n_replicates)
    )


def _with_single_replicate_state_initializers(
    model: Any,
    *,
    n_replicates: int,
    replicate_index: int,
) -> Any:
    nodes = getattr(model, "nodes", {})
    for node_name, node in nodes.items():
        state_index = getattr(node, "state_index", None)
        if not isinstance(state_index, eqx.nn.StateIndex):
            continue
        changed = False

        def unbatch_init_leaf(leaf: Any) -> Any:
            nonlocal changed
            if _is_replicate_axis_array(leaf, n_replicates):
                changed = True
                return leaf[replicate_index]
            return leaf

        init = jt.map(unbatch_init_leaf, state_index.init)
        if changed:
            model = eqx.tree_at(
                lambda graph, name=node_name: graph.nodes[name].state_index.init,
                model,
                init,
            )
    return model


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
        if current.shape[0] == int(completed_batches):
            continue
        previous = prior.get(name)
        if previous is None:
            raise ValueError(
                f"Cannot stitch resumed training diagnostics for {name!r}: "
                "no prior array is available."
            )
        if (
            previous.ndim == current.ndim
            and previous.shape[1:] == current.shape[:-1]
            and previous.shape[0] + current.shape[-1] == int(completed_batches)
        ):
            current = np.moveaxis(current, -1, 0)
        if previous.ndim != current.ndim or previous.shape[1:] != current.shape[1:]:
            raise ValueError(
                f"Cannot stitch resumed training diagnostics for {name!r}: "
                f"prior shape {previous.shape} and current shape {current.shape} differ."
            )
        if previous.shape[0] + current.shape[0] != int(completed_batches):
            raise ValueError(
                f"Cannot stitch resumed training diagnostics for {name!r}: "
                f"{previous.shape[0]} + {current.shape[0]} != {completed_batches}."
            )
        stitched[name] = np.concatenate([previous, current], axis=0)
    for name, previous in prior.items():
        if name in stitched or name == "batch_index" or previous.ndim == 0:
            continue
        if previous.shape[0] == int(completed_batches):
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


def _atomic_latest_link(checkpoint_root: Path, checkpoint_name: str) -> None:
    latest = checkpoint_root / "checkpoint_latest"
    tmp_link = checkpoint_root / ".checkpoint_latest.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(checkpoint_name, tmp_link)
    os.replace(tmp_link, latest)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _task_spec(hps: TreeNamespace) -> dict[str, Any]:
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    target_relative = _target_relative_multitarget_enabled(hps)
    delayed_reach = _delayed_reach_enabled(hps)
    sisu_condition_input = _sisu_conditioned_pgd_input_key(hps)
    sisu_conditioned_pgd_budget = sisu_condition_input is not None
    rollout_steps = int(hps.task.n_steps) if delayed_reach else int(hps.task.n_steps) - 1
    if delayed_reach:
        movement_window = {
            "kind": "delayed_reach_movement_epoch",
            "start_transition": "sampled_go_cue_step",
            "go_cue_min_step": int(hps.delayed_reach.go_cue_sampling.min_step_inclusive),
            "go_cue_max_step": int(hps.delayed_reach.go_cue_sampling.max_step_inclusive),
            "cs_horizon_steps": CS_STAGE_COUNT,
            "cost_indexing": "movement_age_not_trial_age",
            "cost_tail_mode": str(hps.loss.delayed_movement_cost_tail_mode),
        }
        time_axis_contract = (
            "Delayed C&S task: target is visible from trial start, prep has no "
            "target-directed movement loss, and C&S stage costs are indexed by "
            "movement age from the sampled go cue."
        )
    else:
        movement_window = {
            "kind": "full_simple_reach_trial",
            "start_transition": 0,
            "end_transition": int(hps.task.n_steps) - 2,
        }
        time_axis_contract = (
            "Hold-free fixed nominal task: Feedbax n_steps=61 yields exactly 60 "
            "transition/control-cost stages and one position target per transition; "
            "delayed-reach epoch masks are not used."
        )
    if (
        plant_backend == CS_LSS_PLANT_BACKEND
        and target_relative
        and delayed_reach
        and sisu_conditioned_pgd_budget
    ):
        extra_inputs = ["input", "sisu", "target", "epsilon"]
    elif (
        plant_backend == CS_LSS_PLANT_BACKEND
        and target_relative
        and (delayed_reach or sisu_conditioned_pgd_budget)
    ):
        extra_inputs = ["input", "target", "epsilon"]
    elif plant_backend == CS_LSS_PLANT_BACKEND and target_relative:
        extra_inputs = ["target", "epsilon"]
    elif plant_backend == CS_LSS_PLANT_BACKEND:
        extra_inputs = ["input", "epsilon"]
    else:
        extra_inputs = ["sisu", f"intervene:{GRAPH_PLANT_INTERVENOR_NODE}"]
    extra_inputs = [*extra_inputs, *_broad_epsilon_pgd_finite_policy_inputs(hps)]
    return {
        "type": str(hps.task.type),
        "preset": _plain(getattr(hps.task, "preset", None)),
        "n_steps": int(hps.task.n_steps),
        "n_control_stages": _plain(getattr(hps.task, "n_control_stages", None)),
        "control_cost_stages": rollout_steps,
        "workspace": _plain(hps.task.workspace),
        "fixed_init_pos": _plain(hps.task.fixed_init_pos),
        "fixed_target_pos": _plain(hps.task.fixed_target_pos),
        "eval_grid_n": int(hps.task.eval_grid_n),
        "eval_n_directions": int(hps.task.eval_n_directions),
        "eval_reach_length": float(hps.task.eval_reach_length),
        "epoch_len_ranges": _plain(hps.task.epoch_len_ranges),
        "target_on_epochs": _plain(hps.task.target_on_epochs),
        "hold_epochs": _plain(hps.task.hold_epochs),
        "move_epochs": _plain(hps.task.move_epochs),
        "p_catch_trial": float(hps.task.p_catch_trial),
        "target_visible_from_start": _plain(getattr(hps.task, "target_visible_from_start", None)),
        "go_cue_event_name": _plain(getattr(hps.task, "go_cue_event_name", None)),
        "catch_metadata_policy": _plain(getattr(hps.task, "catch_metadata_policy", None)),
        "coordinate_contract": (
            "Feedbax SimpleReaches supplies mechanics.effector.pos targets in the same "
            "Cartesian metre coordinates as the point-mass effector state."
        ),
        "time_axis_contract": time_axis_contract,
        "movement_window": movement_window,
        "extra_inputs": extra_inputs,
        "delayed_reach": _plain(hps.delayed_reach),
        "target_relative_multitarget": (
            _plain(hps.target_relative_multitarget) if target_relative else {"enabled": False}
        ),
        "broad_epsilon_training": (
            _plain(hps.broad_epsilon_training)
            if _broad_epsilon_training_enabled(hps)
            else {"enabled": False}
        ),
        "broad_epsilon_pgd_training": (
            _plain(hps.broad_epsilon_pgd_training)
            if _broad_epsilon_pgd_training_enabled(hps)
            else {"enabled": False}
        ),
        "initial_hidden_encoder": _initial_hidden_encoder_metadata(hps),
    }


def _sisu_conditioned_pgd_input_key(hps: TreeNamespace) -> str | None:
    pgd = getattr(hps, "broad_epsilon_pgd_training", None)
    if not bool(getattr(pgd, "enabled", False)):
        return None
    pgd_schedule = getattr(pgd, "budget_schedule", None)
    pgd_schedule_mode = (
        getattr(pgd_schedule, "mode", None)
        if pgd_schedule is not None
        else getattr(pgd, "budget_schedule", "")
    )
    if str(pgd_schedule_mode) != BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE:
        return None
    pgd_conditioning = getattr(pgd_schedule, "conditioning_scalar", None)
    pgd_condition_input = (
        getattr(pgd_conditioning, "input_key", None)
        if pgd_conditioning is not None
        else getattr(pgd, "sisu_condition_input", "auto")
    )
    if str(pgd_condition_input) == "auto":
        return "sisu" if _delayed_reach_enabled(hps) else "input"
    return str(pgd_condition_input)


def _delayed_pre_go_auxiliary_terms_metadata(hps: TreeNamespace) -> dict[str, Any]:
    weights = getattr(hps.loss, "weights", TreeNamespace())
    start_pos_norm = str(getattr(hps.loss, "delayed_pre_go_start_pos_hold_norm", "l2"))
    terms = {
        "delayed_pre_go_force_filter_hold": {
            "scale": float(getattr(weights, "delayed_pre_go_force_filter_hold", 0.0)),
            "state_key": "states.mechanics.vector delay blocks[..., 4:6]",
            "target": "zero_force_filter_state",
        },
        "delayed_pre_go_start_pos_hold": {
            "scale": float(getattr(weights, "delayed_pre_go_start_pos_hold", 0.0)),
            "state_key": "states.mechanics.effector.pos",
            "target": "trial_specs.inits['mechanics.vector'][..., :2]",
            "norm": start_pos_norm,
        },
        "delayed_pre_go_zero_vel_hold": {
            "scale": float(getattr(weights, "delayed_pre_go_zero_vel_hold", 0.0)),
            "state_key": "states.mechanics.effector.vel",
            "target": "zero_velocity",
        },
    }
    active = {name: meta for name, meta in terms.items() if meta["scale"] != 0.0}
    return {
        "scope": "prep_epoch_only" if _delayed_reach_enabled(hps) else "inactive",
        "epoch_indices": [0] if _delayed_reach_enabled(hps) else [],
        "movement_window_qrf_comparator": "unchanged",
        "terms": terms,
        "active_terms": active,
    }


def _loss_spec(hps: TreeNamespace) -> dict[str, Any]:
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    delayed_reach = _delayed_reach_enabled(hps)
    cs_time_indexing = (
        {
            "stage_schedule": "movement_age_from_go_cue",
            "movement_epoch_source": "trial_specs.timeline.epoch_bounds[-2:]",
            "prep_target_directed_movement_loss": "zero",
            "canonical_movement_horizon_steps": CS_STAGE_COUNT,
            "cost_tail_mode": str(hps.loss.delayed_movement_cost_tail_mode),
            "post_horizon_tail": (
                "zero_weight_after_canonical_horizon"
                if str(hps.loss.delayed_movement_cost_tail_mode)
                == DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
                else "hold_terminal_running_qr_weights_flat_to_trial_end"
            ),
        }
        if delayed_reach
        else {
            "stage_schedule": "trial_age_full_simple_reach",
            "canonical_movement_horizon_steps": CS_STAGE_COUNT,
        }
    )
    cs_fact_t = "((movement_age + 1) / 60)^6, capped at 1" if delayed_reach else "((t + 1) / T)^6"
    if objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
        _plant, schedule = (
            build_no_integrator_game() if no_integrator_state else build_canonical_game()
        )
        physical_state_dim = 6 if no_integrator_state else 8
        q_diag = jnp.diag(schedule.Q[0])
        qf_diag = jnp.diag(schedule.Q_f)
        trial_type_normalization = _plain(
            getattr(hps.loss, "delayed_trial_type_normalization", {"enabled": False})
        )
        return {
            "weights": _plain(hps.loss.weights),
            "delayed_pre_go_auxiliary_terms": _delayed_pre_go_auxiliary_terms_metadata(hps),
            "delayed_trial_type_normalization": trial_type_normalization,
            "delayed_reach": _plain(hps.delayed_reach),
            "objective_profile": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            "objective_kind": "finite_horizon_quadratic",
            "grouped_reduction_implementation": (
                "rlrmp_bridge_pending_feedbax_69d8d76"
                if bool(trial_type_normalization.get("enabled", False))
                else "not_enabled"
            ),
            "source_module": (
                "rlrmp.analysis.math.cs_game_card.build_no_integrator_game"
                if no_integrator_state
                else "rlrmp.analysis.math.cs_game_card.build_canonical_game"
            ),
            "comparator_variant": "no_integrator_state" if no_integrator_state else None,
            "state_basis": {
                "state_key": "states.mechanics.vector",
                "dimension": int(schedule.Q.shape[-1]),
                "physical_block_size": physical_state_dim,
                "delay_blocks": int(schedule.Q.shape[-1] // physical_state_dim),
                "coordinate_transform": (
                    "absolute Feedbax position entries are converted to target-centred "
                    "analytical coordinates before applying Q_t and Q_f"
                ),
            },
            "time_indexing": {
                "running_state": (
                    "state before each movement command from sampled go cue"
                    if delayed_reach
                    else "trial init plus rollout states[:-1], paired with commands"
                ),
                "terminal_state": (
                    (
                        "final rollout state after the variable post-horizon tail"
                        if str(hps.loss.delayed_movement_cost_tail_mode)
                        == DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON
                        else "state after 60 movement commands from sampled go cue"
                    )
                    if delayed_reach
                    else "rollout states[-1]"
                ),
                "horizon_steps": int(schedule.T),
                **cs_time_indexing,
            },
            "matrix_shapes": {
                "Q": list(schedule.Q.shape),
                "R": list(schedule.R.shape),
                "Q_f": list(schedule.Q_f.shape),
            },
            "active_cs_terms": {
                "state_running_q": {
                    "term": "mechanics.vector^T Q_t mechanics.vector",
                    "source": "canonical delay-augmented C&S schedule.Q",
                    "initial_diag_first_block": [float(x) for x in q_diag[:8].tolist()],
                },
                "control_r": {
                    "term": "net.output^T R_t net.output",
                    "source": (
                        "canonical C&S schedule.R on intended controller command "
                        "before efferent/motor-channel noise"
                    ),
                    "diag": [float(x) for x in jnp.diag(schedule.R[0]).tolist()],
                },
                "terminal_q_f": {
                    "term": "mechanics.vector_T^T Q_f mechanics.vector_T",
                    "source": "canonical delay-augmented C&S schedule.Q_f",
                    "diag_first_block": [float(x) for x in qf_diag[:8].tolist()],
                },
            },
            "force_filter_state_cost": "included_via_Q_entries_4_5_each_delay_block",
            "disturbance_integrator_state_cost": "included_via_Q_entries_6_7_each_delay_block",
            "hidden_regularizer": {
                "term": "not_in_full_analytical_qrf_loss",
                "configured_weight": float(hps.loss.weights.nn_hidden),
            },
        }
    if objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE:
        return {
            "weights": _plain(hps.loss.weights),
            "delayed_pre_go_auxiliary_terms": _delayed_pre_go_auxiliary_terms_metadata(hps),
            "delayed_reach": _plain(hps.delayed_reach),
            "effector_pos_late": _plain(hps.loss.effector_pos_late),
            "effector_vel_late": _plain(hps.loss.effector_vel_late),
            "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
            "objective_profile": CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
            "objective_kind": "partial_feedbax_ablation",
            "hypothesis": (
                "historical partial position/velocity terms plus intended-command "
                "control cost and LSS force/filter state cost"
            ),
            "active_cs_terms": {
                "stage_position": {
                    "term": "effector_pos_running",
                    "scale": float(hps.loss.weights.effector_pos_running),
                    "fact_t": cs_fact_t,
                },
                "stage_velocity": {
                    "term": "effector_vel_running",
                    "scale": float(hps.loss.weights.effector_vel_running),
                    "fact_t": cs_fact_t,
                },
                "control": {
                    "term": "nn_output",
                    "state_key": "states.net.output",
                    "scale": float(hps.loss.weights.nn_output),
                    "equivalent_R": "I_2 on intended controller command before noise",
                },
                "force_filter": {
                    "term": "mechanics_force_filter",
                    "state_key": "states.mechanics.vector delay blocks[..., 4:6]",
                    "scale": float(hps.loss.weights.mechanics_force_filter),
                    "basis": "force/filter coordinates from every 8D physical delay block",
                },
                "terminal_position": {
                    "term": "effector_terminal_pos",
                    "scale": float(hps.loss.weights.effector_terminal_pos),
                },
                "terminal_velocity": {
                    "term": "effector_terminal_vel",
                    "scale": float(hps.loss.weights.effector_terminal_vel),
                },
            },
            "force_filter_state_cost": "included_as_partial_ablation_running_term",
            "disturbance_integrator_state_cost": "omitted_in_this_ablation",
            "hidden_regularizer": {
                "term": "nn_hidden",
                "scale": float(hps.loss.weights.nn_hidden),
                "exact_fidelity_default": 0.0,
                "regularized_pair_scale": CS_REGULARIZED_NN_HIDDEN,
            },
            "simple_reach_position_loss_contract": (
                "effector_pos_running compares mechanics.effector.pos to the SimpleReaches "
                "same-coordinate target sequence over every transition, using the configured "
                "C&S Eq. 15 power-law discount when requested."
            ),
            "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
            "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
            "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
            "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
            "movement_ramp_power": float(hps.loss.movement_ramp_power),
            "time_indexing": cs_time_indexing,
        }

    return {
        "weights": _plain(hps.loss.weights),
        "delayed_pre_go_auxiliary_terms": _delayed_pre_go_auxiliary_terms_metadata(hps),
        "delayed_reach": _plain(hps.delayed_reach),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "objective_profile": CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        "active_cs_terms": {
            "stage_position": {
                "term": "effector_pos_running",
                "scale": float(hps.loss.weights.effector_pos_running),
                "fact_t": cs_fact_t,
            },
            "stage_velocity": {
                "term": "effector_vel_running",
                "scale": float(hps.loss.weights.effector_vel_running),
                "fact_t": cs_fact_t,
            },
            "control": {
                "term": "nn_output",
                "scale": float(hps.loss.weights.nn_output),
                "equivalent_R": "I_2 on efferent output",
            },
            "terminal_position": {
                "term": "effector_terminal_pos",
                "scale": float(hps.loss.weights.effector_terminal_pos),
            },
            "terminal_velocity": {
                "term": "effector_terminal_vel",
                "scale": float(hps.loss.weights.effector_terminal_vel),
            },
        },
        "force_filter_state_cost": "not_available",
        "force_filter_state_cost_note": (
            "No force/filter-state quadratic term is synthesized because this "
            "nominal Feedbax loss path has no clean C&S physical force/integrator "
            "state target exposed through the task state contract."
        ),
        "hidden_regularizer": {
            "term": "nn_hidden",
            "scale": float(hps.loss.weights.nn_hidden),
            "exact_fidelity_default": 0.0,
            "regularized_pair_scale": CS_REGULARIZED_NN_HIDDEN,
        },
        "simple_reach_position_loss_contract": (
            "effector_pos_running compares mechanics.effector.pos to the SimpleReaches "
            "same-coordinate target sequence over every transition, using the configured "
            "C&S Eq. 15 power-law discount when requested."
        ),
        "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
        "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
        "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
        "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
        "movement_ramp_power": float(hps.loss.movement_ramp_power),
        "time_indexing": cs_time_indexing,
    }


def _fidelity_status(hps: TreeNamespace) -> dict[str, Any]:
    nn_hidden = float(hps.loss.weights.nn_hidden)
    no_extra_regularizer = nn_hidden == 0.0
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    exact_lss = plant_backend == CS_LSS_PLANT_BACKEND
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    if objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        return {
            "objective": "cs_fidelity_stochastic_rollout",
            "loss_objective": objective,
            "exact_fidelity": False,
            "exact_objective_terms": exact_lss,
            "exact_objective_terms_scope": (
                "true for the implemented training scalar when plant_backend='cs_lss': "
                "the loss evaluates canonical C&S delay-augmented Q_t, R_t, and Q_f "
                "on the exposed LinearStateSpace state and command history"
            ),
            "objective_fidelity": {
                "implemented_terms": [
                    "delay_augmented_state_running_Q_t",
                    "command_running_R_t",
                    "delay_augmented_terminal_Q_f",
                ],
                "omitted_terms": [] if exact_lss else ["cs_lss_state_unavailable"],
                "extra_terms": [],
                "selection_policy": (
                    "rollout validation loss uses the same full analytical Q/R/Q_f "
                    "training scalar; analytical action and I/O metrics remain audit-only"
                ),
            },
            "exact_stochastic_rollout": False,
            "exact_stochastic_noise_sources": exact_lss,
            "exact_plant_matrices": exact_lss,
            "plant_backend": plant_backend,
            "temporary_stochastic_bridge": (
                "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
                "signal-dependent motor Channel, and sampled physical-process mechanics.epsilon; "
                "future Feedbax acausal/ODE plant support should subsume this wrapper"
                if exact_lss
                else None
            ),
            "stochastic_preset": str(hps.model.stochastic_preset),
            "stochastic_projection": (
                "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
                "and plant/load force noise channels without feeding the 48D "
                "delay-augmented analytical state to the GRU."
            ),
            "regularized_pair": False,
            "regularizer": "none",
            "nn_hidden": nn_hidden,
            "certificate_lens": "input_output_map_certificate",
            "same_coordinate_gain_certificate": False,
        }
    if objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE:
        extra_terms = (
            []
            if no_extra_regularizer
            else [
                {
                    "term": "nn_hidden",
                    "scale": nn_hidden,
                    "status": "auxiliary_regularizer_not_in_analytical_objective",
                }
            ]
        )
        return {
            "objective": "cs_fidelity_stochastic_rollout",
            "loss_objective": objective,
            "exact_fidelity": False,
            "exact_objective_terms": False,
            "exact_objective_terms_scope": (
                "ablation only: old partial position/velocity terms are kept, "
                "control is moved to intended net.output, and running force/filter "
                "state cost is added; this is not the full Q/R/Q_f objective"
            ),
            "objective_fidelity": {
                "implemented_terms": [
                    "running_position_cs_eq15_power6",
                    "terminal_position",
                    "running_velocity_cs_eq15_power6",
                    "terminal_velocity",
                    "intended_command_quadratic_net_output",
                    "running_force_filter_state_cost",
                ],
                "omitted_terms": [
                    {
                        "term": "disturbance_integrator_state_cost",
                        "analytical_role": (
                            "unit-weight disturbance-integrator state cost in the C&S 8D schedule"
                        ),
                        "status": "intentionally_omitted_for_force_filter_ablation",
                    },
                    {
                        "term": "terminal_force_filter_and_integrator_Q_f",
                        "analytical_role": "terminal full-state Q_f costs",
                        "status": "not_synthesized_in_partial_ablation",
                    },
                ],
                "extra_terms": extra_terms,
                "selection_policy": (
                    "rollout validation loss only; analytical action and I/O metrics are audit-only"
                ),
            },
            "exact_stochastic_rollout": False,
            "exact_stochastic_noise_sources": exact_lss,
            "exact_plant_matrices": exact_lss,
            "plant_backend": plant_backend,
            "temporary_stochastic_bridge": (
                "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
                "signal-dependent motor Channel, and sampled physical-process mechanics.epsilon; "
                "future Feedbax acausal/ODE plant support should subsume this wrapper"
                if exact_lss
                else None
            ),
            "stochastic_preset": str(hps.model.stochastic_preset),
            "stochastic_projection": (
                "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
                "and plant/load force noise channels without feeding the 48D "
                "delay-augmented analytical state to the GRU."
            ),
            "regularized_pair": not no_extra_regularizer,
            "regularizer": "none" if no_extra_regularizer else "nn_hidden",
            "nn_hidden": nn_hidden,
            "certificate_lens": "input_output_map_certificate",
            "same_coordinate_gain_certificate": False,
            "analytical_delay_augmented_state_input": False,
        }
    omitted_terms = [
        {
            "term": "force_filter_state_cost",
            "analytical_role": "unit-weight force/filter state cost in the C&S 8D schedule",
            "status": "not_synthesized_in_feedbax_gru_loss",
        },
        {
            "term": "disturbance_integrator_state_cost",
            "analytical_role": (
                "unit-weight disturbance-integrator state cost in the C&S 8D schedule"
            ),
            "status": "not_synthesized_in_feedbax_gru_loss",
        },
    ]
    extra_terms = (
        []
        if no_extra_regularizer
        else [
            {
                "term": "nn_hidden",
                "scale": nn_hidden,
                "status": "auxiliary_regularizer_not_in_analytical_objective",
            }
        ]
    )
    return {
        "objective": "cs_fidelity_stochastic_rollout",
        "loss_objective": objective,
        "exact_fidelity": False,
        "exact_objective_terms": False,
        "exact_objective_terms_scope": (
            "false because force/filter-state and disturbance-integrator state costs from "
            "the analytical C&S schedule are omitted from the current Feedbax GRU loss contract"
        ),
        "objective_fidelity": {
            "implemented_terms": [
                "running_position_cs_eq15_power6",
                "terminal_position",
                "running_velocity_cs_eq15_power6",
                "terminal_velocity",
                "command_quadratic_nn_output",
            ],
            "omitted_terms": omitted_terms,
            "extra_terms": extra_terms,
            "selection_policy": (
                "rollout validation loss only; analytical action and I/O metrics are audit-only"
            ),
        },
        "exact_stochastic_rollout": False,
        "exact_stochastic_noise_sources": exact_lss,
        "exact_plant_matrices": exact_lss,
        "plant_backend": plant_backend,
        "temporary_stochastic_bridge": (
            "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
            "signal-dependent motor Channel, and sampled physical-process mechanics.epsilon; "
            "future Feedbax acausal/ODE plant support should subsume this wrapper"
            if exact_lss
            else None
        ),
        "stochastic_preset": str(hps.model.stochastic_preset),
        "stochastic_projection": (
            "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
            "and plant/load force noise channels without feeding the 48D "
            "delay-augmented analytical state to the GRU."
        ),
        "regularized_pair": not no_extra_regularizer,
        "regularizer": "none" if no_extra_regularizer else "nn_hidden",
        "nn_hidden": nn_hidden,
        "certificate_lens": "input_output_map_certificate",
        "same_coordinate_gain_certificate": False,
        "analytical_delay_augmented_state_input": False,
    }


def _get_git_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for cmd, key in [
        (["git", "rev-parse", "HEAD"], "rlrmp_commit"),
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "rlrmp_branch"),
    ]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=REPO_ROOT,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            metadata[key] = result.stdout.strip()
    return metadata


def _get_runtime_metadata() -> dict[str, Any]:
    metadata = {"jax_version": jax.__version__}
    try:
        import feedbax

        metadata["feedbax_version"] = getattr(feedbax, "__version__", "unknown")
    except ImportError:
        metadata["feedbax_version"] = "unavailable"
    return metadata


def _get_dependency_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "rlrmp": "local",
        "jax": jax.__version__,
    }
    for package in ("feedbax", "jax_cookbook", "modal"):
        try:
            module = __import__(package)
            metadata[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            metadata[package] = "unavailable"
    return metadata


def _get_gpu_metadata() -> dict[str, Any]:
    try:
        devices = jax.devices()
        return {
            "device_kinds": [device.device_kind for device in devices],
            "device_count": len(devices),
        }
    except Exception as exc:
        return {
            "device_kinds": None,
            "device_count": 0,
            "error": str(exc),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _json_dumps(payload: dict[str, Any]) -> str:
    return compact_json_dumps(payload)


if __name__ == "__main__":
    raise SystemExit(main())
