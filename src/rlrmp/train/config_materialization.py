"""Validated C&S nominal-GRU hyperparameter materialization."""
# ruff: noqa: F401

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from rlrmp.analysis.math.cs_game_card import (
    INIT_POS,
    TARGET_POS,
    build_canonical_game,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    default_cs_noise_covariances,
)
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.model.cs_lss_gru import (
    CS_H0_CONTEXT_DIM,
    CS_H0_ENCODER_INIT,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.train.broad_epsilon_training import (
    _batch_shape,
    run_broad_epsilon_pgd_inner_maximizer,
)
from rlrmp.train.closed_loop_finite_adversary import (
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
)
from rlrmp.train.fixed_target_perturbation_training import add_zero_graph_channel_inputs
from rlrmp.train.cs_perturbation_training import (
    consumed_calibration_budget_identities,
    make_broad_epsilon_pgd_pre_step,
    make_policy_adversary_pre_step,
    policy_adversary_objective,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.training_configs import (
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    BROAD_EPSILON_PGD_TRAINING_MODE,
    BROAD_EPSILON_TRAINING_MODE,
    DEFAULT_TARGET_SUPPORT_PROFILE,
    LEGACY_PERTURBATION_TRAINING_MODE,
    PERTURBATION_TRAINING_MODE,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_PLAIN_MODE,
    POLICY_ADVERSARY_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
    BroadFullStateEpsilonTrainingConfig,
    FixedTargetPerturbationTrainingConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    target_relative_target_support_config,
)
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
)
from rlrmp.train.training_configs import (
    CS_CONTROL_SCALE,
    CS_POSITION_SCALE,
    CS_VELOCITY_SCALE,
    CsNominalGruConfig,
    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
)

CS_STAGE_COUNT = 60

CS_FEEDBAX_N_STEPS = CS_STAGE_COUNT + 1

CS_REGULARIZED_NN_HIDDEN = 1e-5

CS_DELAYED_REACH_TASK_TYPE = "delayed_reach"

CS_DELAYED_REACH_TASK_PRESET = "delayed_center_out"

LEGACY_CS_DELAYED_REACH_TASK_TYPE = "cs_delayed_center_out_reach"

DELAYED_REACH_TRAINING_MODE = "delayed_reach_target_visible_go_cue"

DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON = "flat_after_canonical_horizon"

DELAYED_MOVEMENT_COST_TAIL_MODES = (
    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
)

ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER = "epsilon_scaled_outer_training"

ADAPTIVE_EPSILON_TRAINING_MODES = (
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
)

DEFAULT_STOCHASTIC_PRESET = str(CsNominalGruConfig.model_fields["stochastic_preset"].default)


def _config_payload_from_args(args: argparse.Namespace | Mapping[str, Any]) -> dict[str, Any]:
    """Return a config payload from a namespace or mapping without unknown attrs."""

    raw = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    return {key: raw[key] for key in CsNominalGruConfig.model_fields if key in raw}


def cs_nominal_gru_config_from_args(
    args: argparse.Namespace | Mapping[str, Any] | CsNominalGruConfig,
) -> CsNominalGruConfig:
    """Validate a nominal-GRU config from CLI-compatible args."""

    if isinstance(args, CsNominalGruConfig):
        return args
    return CsNominalGruConfig.model_validate(_config_payload_from_args(args))


def _config_namespace(
    args: argparse.Namespace | Mapping[str, Any] | CsNominalGruConfig,
) -> argparse.Namespace:
    config = cs_nominal_gru_config_from_args(args)
    return argparse.Namespace(**config.model_dump(mode="python"))


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
    output_config = OutputFeedbackConfig()
    noise_config = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG
    with jax.enable_x64(False):
        plant, _schedule = build_canonical_game()
        covariances = default_cs_noise_covariances(
            plant,
            output_config,
            motor_covariance_scale=noise_config.motor_covariance_scale,
            process_covariance_scale=noise_config.process_covariance_scale,
            signal_dependent_scale=noise_config.signal_dependent_scale,
        )
    sensory_diag = np.asarray(jax.device_get(jnp.diag(covariances.sensory)), dtype=np.float32)
    if not bool(np.allclose(sensory_diag, sensory_diag[0])):
        raise ValueError("C&S sensory covariance projection expects isotropic diagonal covariance")
    return StochasticPreset(
        name=name,
        sensory_noise_std=float(np.sqrt(sensory_diag[0])),
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


def _resolve_auto_bool(value: bool | None, *, default: bool) -> bool:
    """Resolve tri-state CLI booleans that have context-dependent defaults."""

    return bool(default) if value is None else bool(value)


def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Build nominal C&S-aligned GRU hyperparameters from CLI arguments."""

    args = _config_namespace(args)
    args = _apply_smoke_overrides(args)
    args = _config_namespace(args)
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
    no_integrator_state = bool(args.no_integrator_state)
    if no_integrator_state and str(args.plant_backend) != CS_LSS_PLANT_BACKEND:
        raise ValueError("--no-integrator-state requires --plant-backend cs_lss.")
    plant, schedule = build_no_integrator_game() if no_integrator_state else build_canonical_game()
    preset = stochastic_preset(args.stochastic_preset)
    delayed_reach = bool(args.delayed_reach)
    delayed_go_min = int(args.delayed_reach_go_cue_min_step)
    delayed_go_max = int(args.delayed_reach_go_cue_max_step)
    delayed_p_catch_trial = float(args.delayed_reach_p_catch_trial)
    delayed_movement_cost_tail_mode = str(
        args.delayed_movement_cost_tail_mode or DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
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
    delayed_trial_type_normalized_loss = bool(args.delayed_reach_trial_type_normalized_loss)
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
        if delayed_reach and args.nn_output_pre_go is None
        else float(args.nn_output_pre_go or 0.0)
    )
    delayed_pre_go_force_filter_hold = float(args.delayed_pre_go_force_filter_hold or 0.0)
    delayed_pre_go_start_pos_hold = float(args.delayed_pre_go_start_pos_hold or 0.0)
    delayed_pre_go_start_pos_hold_norm = str(args.delayed_pre_go_start_pos_hold_norm or "l2")
    if delayed_pre_go_start_pos_hold_norm not in {"l2", "l1"}:
        raise ValueError("--delayed-pre-go-start-pos-hold-norm must be one of: l2, l1")
    delayed_pre_go_zero_vel_hold = float(args.delayed_pre_go_zero_vel_hold or 0.0)
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
        args.force_filter_feedback,
        default=delayed_reach,
    )
    perturbation_training_enabled = _resolve_auto_bool(
        args.perturbation_training,
        default=delayed_reach,
    )
    perturbation_calibrated_timing = _resolve_auto_bool(
        args.perturbation_calibrated_timing,
        default=delayed_reach and perturbation_training_enabled,
    )
    perturbation_movement_age_timing = _resolve_auto_bool(
        args.perturbation_movement_age_timing,
        default=delayed_reach and perturbation_training_enabled and perturbation_calibrated_timing,
    )
    perturbation_physical_level = str(
        args.perturbation_physical_level or ("small" if delayed_reach else "moderate")
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
            raise ValueError("--adaptive-epsilon-curriculum requires --broad-epsilon-pgd-training.")
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
        adaptive_lambda = adaptive_epsilon_curriculum["lambda_update"]
        if adaptive_lambda["lambda_min"] is None:
            seed_lambda = getattr(broad_epsilon_pgd_training, "energy_lambda", None)
            adaptive_lambda["lambda_min"] = (
                1.0e-3 * float(seed_lambda)
                if seed_lambda is not None and float(seed_lambda) > 0.0
                else 1.0e-12
            )
        lambda_max = adaptive_lambda.get("lambda_max")
        if lambda_max is not None and float(lambda_max) <= float(adaptive_lambda["lambda_min"]):
            raise ValueError("Adaptive epsilon lambda_max must be greater than lambda_min.")
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


def _adaptive_epsilon_curriculum_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(args, "adaptive_epsilon_curriculum", False))
    controller_training_mode = str(
        getattr(
            args,
            "adaptive_epsilon_controller_training_mode",
            ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
        )
    )
    cfg = {
        "enabled": enabled,
        "controller_training_mode": controller_training_mode,
        "damage_schedule": {
            "kind": "linear_ramp_then_cosine_anneal",
            "setpoint_basis": "damage_to_clean_loss_ratio",
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
            "hysteresis_frac": (
                None
                if args.adaptive_epsilon_hysteresis_frac is None
                else float(args.adaptive_epsilon_hysteresis_frac)
            ),
            "freeze_until_burn_in": bool(args.adaptive_epsilon_freeze_until_burn_in),
            "gain_normalization": bool(args.adaptive_epsilon_gain_normalization),
            "gain_ema_alpha": float(args.adaptive_epsilon_gain_ema_alpha),
            "gain_min": float(args.adaptive_epsilon_gain_min),
            "gain_max": float(args.adaptive_epsilon_gain_max),
            "lambda_min": (
                None
                if args.adaptive_epsilon_lambda_min is None
                else float(args.adaptive_epsilon_lambda_min)
            ),
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
            "applies_to": (
                "optimized_direct_epsilon_loss_only"
                if controller_training_mode == ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND
                else "optimized_direct_epsilon_channel_scale_for_controller_rollout"
            ),
            "perturbation_bank_policy": "orthogonal_unweighted_by_outer_adversarial_weight",
        },
    }
    if not enabled:
        return cfg
    if controller_training_mode not in ADAPTIVE_EPSILON_TRAINING_MODES:
        raise ValueError(
            "Adaptive epsilon controller training mode must be one of "
            f"{', '.join(ADAPTIVE_EPSILON_TRAINING_MODES)}."
        )
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
    if (
        cfg["lambda_update"]["hysteresis_frac"] is not None
        and cfg["lambda_update"]["hysteresis_frac"] < 0.0
    ):
        raise ValueError("Adaptive epsilon hysteresis fraction must be nonnegative.")
    if not 0.0 < cfg["lambda_update"]["gain_ema_alpha"] <= 1.0:
        raise ValueError("Adaptive epsilon gain EMA alpha must be in (0, 1].")
    if cfg["lambda_update"]["gain_min"] <= 0.0:
        raise ValueError("Adaptive epsilon gain_min must be positive.")
    if cfg["lambda_update"]["gain_max"] < cfg["lambda_update"]["gain_min"]:
        raise ValueError("Adaptive epsilon gain_max must be greater than or equal to gain_min.")
    if cfg["lambda_update"]["lambda_min"] is not None and cfg["lambda_update"]["lambda_min"] <= 0.0:
        raise ValueError("Adaptive epsilon lambda_min must be positive.")
    if (
        cfg["lambda_update"]["lambda_max"] is not None
        and cfg["lambda_update"]["lambda_min"] is not None
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


__all__ = [
    "ADAPTIVE_EPSILON_TRAINING_MODES",
    "ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER",
    "CS_DELAYED_REACH_TASK_PRESET",
    "CS_DELAYED_REACH_TASK_TYPE",
    "CS_FEEDBAX_N_STEPS",
    "CS_REGULARIZED_NN_HIDDEN",
    "CS_STAGE_COUNT",
    "DEFAULT_STOCHASTIC_PRESET",
    "DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON",
    "DELAYED_MOVEMENT_COST_TAIL_MODES",
    "DELAYED_REACH_TRAINING_MODE",
    "LEGACY_CS_DELAYED_REACH_TASK_TYPE",
    "StochasticPreset",
    "_adaptive_epsilon_curriculum_config_from_args",
    "_apply_smoke_overrides",
    "_config_namespace",
    "_config_payload_from_args",
    "_delayed_reach_contract_from_args",
    "_initial_hidden_encoder_config",
    "_resolve_auto_bool",
    "_training_diagnostics_enabled",
    "build_hps",
    "cs_nominal_gru_config_from_args",
    "stochastic_preset",
]
