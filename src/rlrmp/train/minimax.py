"""Hyperparameter construction for the minimax adversarial trainer.

Bug: 8404108 — extracted from ``scripts/train_minimax.py`` so analysis /
eval / diagnostic scripts can reconstruct the same hyperparameter tree
from a saved ``config.json`` without ``sys.path``-injecting the training
script.

The corresponding training loop lives in ``scripts/train_minimax.py``; only
the hyperparameter construction is library-grade.
"""

from __future__ import annotations

import argparse
from functools import partial

import equinox as eqx
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.model.trainable import staged_network_trainable_paths

__all__ = ["build_hps"]


def _trainable_paths_for_hidden_type(hidden_type: str, sisu_gating: str) -> list[str]:
    if hidden_type == "linear":
        return ["nodes.net.gain"]
    if hidden_type == "linear_tracker":
        return ["nodes.net.gain", "nodes.net.feedforward"]
    return staged_network_trainable_paths(sisu_gating=sisu_gating)


def _resolve_hidden_type(hidden_type_str: str, dt: float):
    """Map a CLI hidden-type string to the corresponding recurrent cell class/partial.

    Args:
        hidden_type_str: One of ``"gru"``, ``"vanilla_rnn"``, ``"linear"``,
            ``"linear_tracker"``.
        dt: Simulation timestep (used to set ``alpha = dt / tau`` for
            ``VanillaRNNCell``).

    Returns:
        A class or partial-applied constructor compatible with
        ``point_mass_nn``'s ``hidden_type`` parameter (i.e. callable as
        ``hidden_type(input_size, hidden_size, use_bias=..., key=...)``), or the
        sentinel string for linear controllers.
    """
    if hidden_type_str == "gru":
        return eqx.nn.GRUCell
    elif hidden_type_str == "vanilla_rnn":
        from rlrmp.model import VanillaRNNCell
        # tau=0.1 s (100 ms) => alpha=dt/tau=0.1 at dt=0.01 — matches cortical-neuron
        # time constant in motor-control RNN literature (Yang 2019, Sussillo 2015).
        return partial(VanillaRNNCell, dt=dt, tau=0.1)
    elif hidden_type_str in ("linear", "linear_tracker"):
        # Sentinel string forwarded to setup_task_model_pair, which dispatches to
        # create_point_mass_linear_ensemble. Linear controllers have no recurrent
        # cell — they replace SimpleStagedNetwork entirely. Bug: 410d7ac.
        return hidden_type_str
    else:
        raise ValueError(f"Unknown hidden_type: {hidden_type_str!r}")


def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Construct minimax-trainer hyperparameters from CLI args.

    Uses the same task config as :func:`rlrmp.train.standard.build_hps`
    (running_cost loss mode), so the two trainers produce comparable models.
    """
    dt = 0.01
    hps_dict = {
        "method": "pai-asf",
        "dt": dt,
        # n_batches_condition drives setup_task_model_pair's loss schedule;
        # set to total training length so late-ramp terms are calibrated correctly.
        "n_batches_condition": args.n_warmup_batches + args.n_adversary_batches,
        "n_batches_baseline": 0,
        "batch_size": getattr(args, "batch_size", 250),
        "learning_rate_0": args.controller_lr,
        "n_scaleup_batches": 0,
        "constant_lr_iterations": 0,
        "cosine_annealing_alpha": 1.0,
        "weight_decay": 0.0,
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": getattr(args, "n_replicates", 5),
            "effector_mass": 1.0,
            "hidden_size": 180,
            "feedback_delay_steps": 5,
            "feedback_noise_std": 0.01,
            "motor_noise_std": 0.01,
            "sensory_noise_std": getattr(args, "sensory_noise_std", None),
            "additive_motor_noise_std": getattr(args, "additive_motor_noise_std", None),
            "signal_dependent_motor_noise_std": getattr(
                args,
                "signal_dependent_motor_noise_std",
                None,
            ),
            "plant_process_force_noise_std": getattr(
                args,
                "plant_process_force_noise_std",
                0.0,
            ),
            "damping": 10.0,
            "tau_rise": 0.05,
            "population_structure": {
                "n_input_only": 60,
                "n_readout_only": 60,
                "n_recurrent_only": 60,
                "n_input_readout": 0,
            },
        },
        "task": {
            "type": "delayed_reach",
            "n_steps": 140,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "eval_grid_n": 1,
            "eval_n_directions": 8,
            "eval_reach_length": 0.5,
            "train_endpoint_mode": "center_out",
            # Drop pure-hold to 0 steps; target-on now 100-300 ms (10-30 steps
            # at dt=0.01 s), matching Shahbazi 2025 §4.2. Bug: 2bc95fd
            "epoch_len_ranges": [[0, 1], [10, 30]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
            "p_catch_trial": getattr(args, "p_catch_trial", 0.5),
        },
        "pert": {
            "type": "gusts",
            # Warm-start uses pert_std=1.0 (normal gusts).
            "std": 1.0,
            "duration_mean": 8,
            "n_expected": 3,
        },
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": getattr(args, "effector_pos_running", 1.0),
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": getattr(args, "effector_pos_late_weight", 0.5),
                "effector_vel_late": getattr(args, "effector_vel_late", 0.1),
                "effector_hold_pos": getattr(args, "effector_hold_pos", 10.0),
                "effector_hold_vel": getattr(args, "effector_hold_vel", 10.0),
                # Terminal-step velocity penalty (historical simple_reach_loss
                # shape). Fires only at t=T; strong "come-to-rest" signal.
                # Default 0.0 = disabled (preserves baseline behaviour).
                # Activate via --effector-final-vel 1.0. Bug: 2bc95fd
                "effector_final_vel": getattr(args, "effector_final_vel", 0.0),
                "nn_output": getattr(args, "nn_output", 1e-5),
                "nn_hidden": getattr(args, "nn_hidden", 1e-5),
                # Compositional ||h_t - h_{t-1}||² hidden-state smoothness
                # term, off-by-default. Enable via --nn-hidden-derivative
                # (e.g. 1e-3 per Shahbazi et al. 2025 Eq. 1). Bug: efc4d68
                "nn_hidden_derivative": getattr(args, "nn_hidden_derivative", 0.0),
                # Compositional ||v_{t+1} - 2 v_t + v_{t-1}||² output-jerk
                # term, off-by-default. Enable via --nn-output-jerk
                # (e.g. 1e5 per Shahbazi et al. 2025 Eq. 1). Bug: efc4d68
                # (feedbax 7e1d257)
                "nn_output_jerk": getattr(args, "nn_output_jerk", 0.0),
                # Pre-go controller-output penalty (epochs 0+1, before the go
                # cue). Wraps the standard nn_output squared-L2 term in
                # EpochMaskedLoss; off-by-default. Enable via
                # --nn-output-pre-go (suggested 1e-2 ≈ 1000x the post-aggregated
                # nn_output weight). Bug: efc4d68 (feedbax 50507a9)
                "nn_output_pre_go": getattr(args, "nn_output_pre_go", 0.0),
                # Pre-go hidden-state-derivative penalty (epochs 0+1).
                # Companion to the motor-pre-go term — included so the
                # "suppress preparation too" comparator is one flag away.
                # Off-by-default. Bug: efc4d68 (feedbax 50507a9)
                "nn_hidden_derivative_pre_go": getattr(
                    args, "nn_hidden_derivative_pre_go", 0.0
                ),
            },
            "effector_pos_late": {
                "start_step_after_go": getattr(
                    args, "effector_pos_late_start_step", 80
                ),
                "final_scale_factor": getattr(
                    args, "effector_pos_late_final_scale", 2.0
                ),
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            # Power-law schedule: "flat" (default) or "powerlaw" ((t/T-1)^power).
            # Bug: 2e1a6ad
            "effector_pos_running_schedule": getattr(
                args, "effector_pos_running_schedule", "flat"
            ),
            "effector_hold_pos_schedule": getattr(
                args, "effector_hold_pos_schedule", "flat"
            ),
            "position_powerlaw_power": getattr(args, "position_powerlaw_power", 6.0),
            "movement_ramp_shape": getattr(args, "movement_ramp_shape", "linear"),
            "movement_ramp_duration_steps": getattr(
                args, "movement_ramp_duration_steps", 60
            ),
            "movement_ramp_power": getattr(args, "movement_ramp_power", 2.0),
        },
        "loss_update": {
            "enabled": args.loss_update_enabled,
            "target_ratio": args.loss_update_ratio,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
        "where": {
            0: _trainable_paths_for_hidden_type(args.hidden_type, args.sisu_gating),
        },
        # hidden_type is a callable (class or partial), not serialisable to JSON.
        # It is resolved here from the CLI string and stored directly in the namespace.
        "hidden_type": _resolve_hidden_type(args.hidden_type, dt),
        "sisu_gating": args.sisu_gating,
    }
    return dict_to_namespace(hps_dict, to_type=TreeNamespace)
