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
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
from feedbax._io import save as fbx_save
from feedbax.train import filter_spec_leaves, get_model_parameters
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.cs_game_card import (
    INIT_POS,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
    TARGET_POS,
    build_canonical_game,
)
from rlrmp.analysis.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    default_cs_noise_covariances,
)
from rlrmp.analysis.output_feedback import OutputFeedbackConfig
from rlrmp.feedbax_graph import (
    EXECUTION_BACKEND,
    PLANT_INTERVENOR_LABEL,
    RLRMPFeedbaxGraphBundle,
    build_point_mass_sensorimotor_graph_spec,
    write_graph_spec_bundle,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_LOSS_OBJECTIVES,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
)
from rlrmp.modules.training.part2 import (
    CS_LSS_PLANT_BACKEND,
    LEGACY_CAUSAL_PLANT_BACKEND,
    setup_task_model_pair,
)
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.run_specs import validate_nominal_gru_run_spec
from rlrmp.train.cs_perturbation_training import (
    FixedTargetPerturbationTrainingConfig,
    LEGACY_PERTURBATION_TRAINING_MODE,
    PERTURBATION_TRAINING_MODE,
    planned_fixed_target_perturbation_rows,
    validation_bin_manifest,
)
from rlrmp.stochastic_runtime import (
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)

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
TRAINING_DIAGNOSTICS_NPZ = "training_diagnostics.npz"
TRAINING_DIAGNOSTICS_MANIFEST = "training_diagnostics.json"
VolumeCommit = Callable[[], None]


@dataclass(frozen=True)
class TrainingState:
    """Serializable state needed to resume the chunked C&S GRU training loop."""

    model: Any
    optimizer_state: Any
    completed_batches: int
    key: Any
    history: Any | None


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
                "Use the C&S Csdn scale as Feedbax pre-force-filter "
                "multiplicative command noise."
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

    out = Path(output_dir).resolve()
    artifact_root = (REPO_ROOT / "_artifacts").resolve()
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        return out.parent / f"{out.name}_spec"


def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Build nominal C&S-aligned GRU hyperparameters from CLI arguments."""

    args = _apply_smoke_overrides(args)
    if str(args.loss_objective) in {
        CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    } and str(args.plant_backend) != CS_LSS_PLANT_BACKEND:
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
    plant, schedule = build_canonical_game()
    preset = stochastic_preset(args.stochastic_preset)
    if int(schedule.T) != CS_STAGE_COUNT:
        raise ValueError(f"Expected C&S stage count {CS_STAGE_COUNT}, got {schedule.T}")
    nn_hidden = CS_REGULARIZED_NN_HIDDEN if args.regularized_fidelity else 0.0
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
    perturbation_training = FixedTargetPerturbationTrainingConfig(
        enabled=bool(args.perturbation_training),
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
            "type": "fixed_simple_reach",
            "n_steps": CS_FEEDBAX_N_STEPS,
            "workspace": [[-0.02, -0.02], [float(TARGET_POS[0]) + 0.02, 0.02]],
            "fixed_init_pos": [float(x) for x in INIT_POS.tolist()],
            "fixed_target_pos": [float(x) for x in TARGET_POS.tolist()],
            "eval_grid_n": 1,
            "eval_n_directions": 1,
            "eval_reach_length": float(TARGET_POS[0]),
            "epoch_len_ranges": [[0, 1], [CS_STAGE_COUNT, CS_STAGE_COUNT + 1]],
            "target_on_epochs": [0],
            "hold_epochs": [],
            "move_epochs": [0],
            "p_catch_trial": 0.0,
        },
        "pert": {
            "type": "gusts",
            "std": 0.0,
            "duration_mean": 0,
            "n_expected": 0,
        },
        "perturbation_training": perturbation_training.to_hps_dict(),
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
                "nn_output_pre_go": 0.0,
                "nn_hidden_derivative_pre_go": 0.0,
                "mechanics_force_filter": (
                    1.0 / float(schedule.Q.shape[-1] // 8)
                    if str(args.loss_objective) == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE
                    else 0.0
                ),
            },
            "effector_pos_late": {
                "start_step_after_go": int(schedule.T),
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": int(schedule.T),
                "final_scale_factor": 1.0,
            },
            "effector_pos_running_schedule": "cs_eq15_power6",
            "effector_hold_pos_schedule": "disabled",
            "position_powerlaw_power": 6.0,
            "movement_ramp_shape": "none",
            "movement_ramp_duration_steps": 0,
            "movement_ramp_power": 1.0,
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
            0: ["nodes.net.hidden", "nodes.net.readout"],
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
        "source_module": "rlrmp.analysis.cs_game_card",
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
                "Full analytical Q/R/Q_f loss scores force/filter and disturbance-integrator "
                "state through the canonical delay-augmented C&S Q_t and Q_f matrices."
            ),
        }
    return card


def build_model_structure_summary(hps: TreeNamespace) -> dict[str, Any]:
    """Return the model/training summary embedded in ``run.json``."""

    pop = hps.model.population_structure
    stochastic_runtime = _stochastic_runtime_contract(hps)
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    exact_lss = plant_backend == CS_LSS_PLANT_BACKEND
    return {
        "controller_kind": "gru",
        "plant_backend": plant_backend,
        "plant_backend_warning": (
            "legacy causal SimpleFeedback has a same-step force-filter-to-mechanics "
            "timing problem"
            if plant_backend == LEGACY_CAUSAL_PLANT_BACKEND
            else None
        ),
        "exact_cs_linear_state_space": exact_lss,
        "fixed_plant_parameters": (
            ["nodes.mechanics.A", "nodes.mechanics.B", "nodes.mechanics.B_w"]
            if exact_lss
            else []
        ),
        "hidden_size": int(hps.model.hidden_size),
        "n_replicates": int(hps.model.n_replicates),
        "trainable": ["nodes.net.hidden", "nodes.net.readout"],
        "population_structure": {
            "n_input_only": int(pop.n_input_only),
            "n_readout_only": int(pop.n_readout_only),
            "n_recurrent_only": int(pop.n_recurrent_only),
            "n_input_readout": int(pop.n_input_readout),
        },
        "feedback": {
            "delay_steps": int(hps.model.feedback_delay_steps),
            "noise_std": stochastic_runtime["sensory_noise_std"],
            "noise_role": "sensory_feedback",
            "noise_timing": (
                "Feedbax sensory Channel after 4D delayed LSS feedback selector"
                if exact_lss
                else "Feedbax feedback Channel before controller"
            ),
            "delay_source": (
                "C&S 48D LinearStateSpace delay-augmented state"
                if exact_lss
                else "Feedbax feedback Channel queue"
            ),
        },
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
        "nominal_only": not _perturbation_training_enabled(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "adversarial_phase": "none",
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
        "trainable": ["nodes.net.hidden", "nodes.net.readout"],
        "method": str(hps.method),
        "nominal_only": not _perturbation_training_enabled(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "adversarial_phase": "none",
        "certificate_lens": "input_output_map_certificate",
        "analytical_delay_augmented_state_input": False,
        "stochastic_runtime": _stochastic_runtime_contract(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "loss_objective": str(hps.loss.objective),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "component_policy": {
            "rlrmp_component_types": [
                "RLRMPFeedbackChannels",
                "RLRMPSimpleStagedNetwork",
                "FixedField",
            ],
            "nominal_intervention_policy": (
                f"{PLANT_INTERVENOR_LABEL} is present only as an inactive legacy "
                "GraphSpec compatibility component; no robust/minimax adversary is scheduled."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.modules.training.part2.setup_task_model_pair",
            "checkpoint_format": "feedbax._io.save/load_with_hyperparameters",
        },
        "task_spec": task_spec,
        "loss_spec": loss_spec,
        "training_spec": training_spec,
        "game_card_provenance": build_loss_game_card_provenance(hps),
        "model_structure": build_model_structure_summary(hps),
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
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": str(args.issue),
        "training_script": "scripts/train_cs_nominal_gru.py",
        "mode": _run_mode(args),
        "artifact_output_dir": str(output_dir),
        "spec_dir": str(spec_dir),
        "nominal_only": not _perturbation_training_enabled(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "validation_bins": validation_bin_manifest(hps.perturbation_training),
        "adversarial_phase": "none",
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
        "model_summary": build_model_structure_summary(hps),
        "task_timing": graph_bundle.task_spec,
        "loss_summary": graph_bundle.loss_spec,
        "training_summary": {
            **graph_bundle.training_spec,
            "training_mode": _training_mode(hps),
            "n_train_batches": int(args.n_train_batches),
            "n_adversary_batches": 0,
            "validation_bins": validation_bin_manifest(hps.perturbation_training),
            "training_diagnostics": _training_diagnostics_metadata(args, output_dir),
        },
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
    spec_dir = Path(args.spec_dir) if args.spec_dir is not None else derive_spec_dir(output_dir)
    hps = build_hps(args)
    graph_bundle = build_graph_bundle(hps)
    payload = build_run_spec(
        args,
        output_dir=output_dir,
        spec_dir=spec_dir,
        graph_bundle=graph_bundle,
    )

    if args.dry_run:
        would_write = [str(spec_dir / "run.json"), str(spec_dir / "model.graph.manifest.json")]
        if _should_write_graph_spec(hps):
            would_write.append(str(spec_dir / "model.graph.json"))
        return {
            "run_spec": payload,
            "would_write": would_write,
        }

    mkdir_p(spec_dir)
    graph_path = _write_graph_bundle_for_backend(hps, graph_bundle, spec_dir)
    payload["feedbax_graph"] = graph_bundle.to_run_metadata(
        graph_spec_path=None if graph_path is None else graph_path.name,
    )
    validate_nominal_gru_run_spec(payload, spec_dir=spec_dir)
    run_path = spec_dir / "run.json"
    run_path.write_text(_json_dumps(payload), encoding="utf-8")
    return {
        "run_spec_path": str(run_path),
        "graph_spec_path": None if graph_path is None else str(graph_path),
        "graph_manifest_path": str(spec_dir / "model.graph.manifest.json"),
    }


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

    spec_result = write_run_spec(args)
    output_dir = mkdir_p(Path(args.output_dir))
    run_spec_path = Path(spec_result["run_spec_path"])
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))

    hps = build_hps(args)
    key_init, key_train = jr.split(jr.PRNGKey(int(args.seed)), 2)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = _build_trainer(hps)
    where_train = _where_train()
    template_state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=where_train[0],
        key=key_train,
    )
    checkpoint_root = output_dir / "checkpoints"
    state = (
        load_latest_checkpoint(
            checkpoint_root,
            model_template=pair.model,
            optimizer_state_template=template_state.optimizer_state,
            history_template=None,
        )
        if args.resume and latest_checkpoint_path(checkpoint_root).exists()
        else template_state
    )

    chunks: list[dict[str, float | int | str]] = []
    history_diagnostic_chunks: list[dict[str, np.ndarray]] = []
    optimizer_diagnostic_chunks: list[dict[str, np.ndarray]] = []
    training_started = time.perf_counter()
    while state.completed_batches < int(args.n_train_batches):
        remaining = int(args.n_train_batches) - state.completed_batches
        chunk_batches = min(int(args.checkpoint_interval_batches), remaining)
        key_chunk, key_next = jr.split(state.key, 2)
        chunk_started = time.perf_counter()
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
        )
        chunk_duration_seconds = time.perf_counter() - chunk_started
        completed = state.completed_batches + chunk_batches
        history = _append_history(state.history, history_chunk)
        if _training_diagnostics_enabled(args):
            optimizer_diagnostic_chunks.append(
                _optimizer_diagnostics_arrays(
                    optimizer_state,
                    start_batches=state.completed_batches,
                    completed_batches=completed,
                )
            )
            history_diagnostic_chunks.append(
                _history_diagnostics_arrays(history_chunk, chunk_batches)
            )
        history_chunk_path = output_dir / "history_chunks" / f"history_{completed:07d}.eqx"
        history_chunk_path.parent.mkdir(parents=True, exist_ok=True)
        fbx_save(history_chunk_path, history_chunk)
        state = TrainingState(
            model=model,
            optimizer_state=optimizer_state,
            completed_batches=completed,
            key=key_next,
            history=history,
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
    training_duration_seconds = time.perf_counter() - training_started

    final_model_path = output_dir / "trained_model.eqx"
    final_history_path = output_dir / "training_history.eqx"
    final_summary_path = output_dir / "training_summary.json"
    fbx_save(final_model_path, state.model, hyperparameters=run_spec)
    if state.history is not None:
        fbx_save(final_history_path, state.history)
    diagnostics_metadata = write_training_diagnostics_sidecar(
        output_dir,
        args=args,
        run_spec=run_spec,
        state=state,
        training_history_path=final_history_path,
        optimizer_diagnostic_chunks=optimizer_diagnostic_chunks,
        history_diagnostic_chunks=history_diagnostic_chunks,
    )
    final_summary = {
        "schema_version": f"{SCHEMA_VERSION}.training.v1",
        "issue": str(args.issue),
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "training_duration_seconds": training_duration_seconds,
        "training_batches_per_second": (
            state.completed_batches / training_duration_seconds
            if training_duration_seconds > 0
            else None
        ),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "final_model_path": str(final_model_path),
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
    if state.history is not None:
        fbx_save(tmp / "history.eqx", state.history)
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
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=int(metadata["completed_batches"]),
        key=jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32),
        history=history,
    )


def latest_checkpoint_path(checkpoint_root: Path) -> Path:
    """Return the path used by the durable latest-checkpoint contract."""

    return checkpoint_root / "checkpoint_latest"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Prepare a stochastic C&S-fidelity GRU run spec.",
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
        action="store_true",
        help=(
            "Enable fixed-target perturbation-generalized training using external "
            "task/plant/channel adapters. Target-position streams are not added."
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
    parser.add_argument("--perturbation-delayed-observation-offset-m", type=float, default=0.01)
    parser.add_argument("--perturbation-pulse-start-step", type=int, default=20)
    parser.add_argument("--perturbation-pulse-duration-steps", type=int, default=5)
    parser.add_argument(
        "--planned-perturbation-rows",
        action="store_true",
        help="Print the two planned issue aacb9ed local training row commands and exit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny local values; with --full-train this runs a one-batch smoke.",
    )
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    parser.add_argument("--disable-progress", action="store_true", default=True)
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

    args = build_parser().parse_args(argv)
    if args.planned_perturbation_rows:
        print(_json_dumps({"planned_rows": planned_fixed_target_perturbation_rows()}), end="")
        return 0
    result = (
        run_full_training(args, volume_commit=volume_commit)
        if args.full_train
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
        optax.inject_hyperparams(
            partial(optax.adamw, weight_decay=float(hps.weight_decay))
        )(learning_rate=schedule)
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


def _where_train() -> dict[int, Callable[[Any], tuple[Any, Any]]]:
    def where_train_fn(model):
        net = model.nodes["net"]
        return (net.hidden, net.readout)

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
        "gradient_clip_kind": (
            None if args.gradient_clip_norm is None else "global_norm"
        ),
        "training_diagnostics": _training_diagnostics_metadata(
            args,
            Path(args.output_dir),
        ),
    }


def _perturbation_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.perturbation_training, "enabled", False))


def _training_mode(hps: TreeNamespace) -> str:
    if _perturbation_training_enabled(hps):
        return PERTURBATION_TRAINING_MODE
    return "nominal"


def _training_distribution_metadata(hps: TreeNamespace) -> dict[str, Any]:
    config = hps.perturbation_training
    if not bool(getattr(config, "enabled", False)):
        return {
            "mode": "nominal",
            "fixed_target_only": True,
            "target_stream": "not_consumed",
        }
    return {
        "mode": PERTURBATION_TRAINING_MODE,
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
            "sampling": "prng_driven_signed_random_axes_components_timings_levels",
        },
        "mild_combined_families": ["initial_position", "command_input"],
        "single_family_bins": list(config.single_family_bins),
        "validation_bins": list(config.validation_bins),
        "checkpoint_selection_role": "generalized_held_out_perturbation_validation",
        "nominal_quality_role": "reported_quality_sidecar_gate",
        "controller_internal_mutation": False,
        "adversarial_phase": "none",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(_json_dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def write_training_diagnostics_sidecar(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    run_spec: dict[str, Any],
    state: TrainingState,
    training_history_path: Path,
    optimizer_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
    history_diagnostic_chunks: list[dict[str, np.ndarray]] | None = None,
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

    npz_path = Path(metadata["sidecar_path"])
    manifest_path = Path(metadata["manifest_path"])
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
    return {
        "type": str(hps.task.type),
        "n_steps": int(hps.task.n_steps),
        "control_cost_stages": int(hps.task.n_steps) - 1,
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
        "coordinate_contract": (
            "Feedbax SimpleReaches supplies mechanics.effector.pos targets in the same "
            "Cartesian metre coordinates as the point-mass effector state."
        ),
        "time_axis_contract": (
            "Hold-free fixed nominal task: Feedbax n_steps=61 yields exactly 60 "
            "transition/control-cost stages and one position target per transition; "
            "delayed-reach epoch masks are not used."
        ),
        "movement_window": {
            "kind": "full_simple_reach_trial",
            "start_transition": 0,
            "end_transition": int(hps.task.n_steps) - 2,
        },
        "extra_inputs": (
            ["input", "epsilon"]
            if plant_backend == CS_LSS_PLANT_BACKEND
            else ["sisu", f"intervene:{PLANT_INTERVENOR_LABEL}"]
        ),
    }


def _loss_spec(hps: TreeNamespace) -> dict[str, Any]:
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    if objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        _plant, schedule = build_canonical_game()
        q_diag = jnp.diag(schedule.Q[0])
        qf_diag = jnp.diag(schedule.Q_f)
        return {
            "weights": _plain(hps.loss.weights),
            "objective_profile": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            "objective_kind": "finite_horizon_quadratic",
            "source_module": "rlrmp.analysis.cs_game_card.build_canonical_game",
            "state_basis": {
                "state_key": "states.mechanics.vector",
                "dimension": int(schedule.Q.shape[-1]),
                "physical_block_size": 8,
                "delay_blocks": int(schedule.Q.shape[-1] // 8),
                "coordinate_transform": (
                    "absolute Feedbax position entries are converted to target-centred "
                    "analytical coordinates before applying Q_t and Q_f"
                ),
            },
            "time_indexing": {
                "running_state": "trial init plus rollout states[:-1], paired with commands",
                "terminal_state": "rollout states[-1]",
                "horizon_steps": int(schedule.T),
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
                    "fact_t": "((t + 1) / T)^6",
                },
                "stage_velocity": {
                    "term": "effector_vel_running",
                    "scale": float(hps.loss.weights.effector_vel_running),
                    "fact_t": "((t + 1) / T)^6",
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
        }

    return {
        "weights": _plain(hps.loss.weights),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "objective_profile": CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        "active_cs_terms": {
            "stage_position": {
                "term": "effector_pos_running",
                "scale": float(hps.loss.weights.effector_pos_running),
                "fact_t": "((t + 1) / T)^6",
            },
            "stage_velocity": {
                "term": "effector_vel_running",
                "scale": float(hps.loss.weights.effector_vel_running),
                "fact_t": "((t + 1) / T)^6",
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
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
