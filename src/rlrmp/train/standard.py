"""Hyperparameter construction for the non-adversarial (standard) trainer.

Bug: 8404108 — extracted from ``scripts/train_part2_5.py`` so analysis /
eval / diagnostic scripts can reconstruct the same hyperparameter tree
from a saved ``config.json`` without ``sys.path``-injecting the training
script.

The training driver itself remains in ``scripts/train_part2_5.py`` along
with its CVaR / APT wrappers; only the hyperparameter constructors and the
loss-mode menu are library-grade.

The four loss modes are:

- ``running_cost`` — uniform position penalty during movement.
- ``softmin`` — goal-hit-in-window objective with soft minimum.
- ``combined`` — weak running cost + strong goal-hit window.
- ``default`` — structured mid/late terms (standard Part 2 config).
"""

from __future__ import annotations

import argparse

from feedbax.types import TreeNamespace, dict_to_namespace

__all__ = [
    "LOSS_MODE_CONFIGS",
    "build_hps",
]


# ---------------------------------------------------------------------------
# Base hyperparameter override
# ---------------------------------------------------------------------------

def _base_hps(args: argparse.Namespace) -> dict:
    """Return the base hyperparameter overrides shared across all loss modes."""
    return {
        "method": "pai-asf",
        "dt": 0.01,
        "n_batches_condition": args.n_batches,
        "n_batches_baseline": 0,
        "batch_size": 250,
        "learning_rate_0": 0.01,
        "n_scaleup_batches": 0,
        "constant_lr_iterations": 0,
        "cosine_annealing_alpha": 1.0,
        "weight_decay": 0.0,
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": 5,
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
            "type": "center_out_delayed_reach",
            "n_steps": 130,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "eval_grid_n": 1,
            "eval_n_directions": 8,
            "eval_reach_length": 0.5,
            "epoch_len_ranges": [[10, 11], [5, 20]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
            "p_catch_trial": 0.5,
        },
        "pert": {
            "type": "gusts",
            "std": args.pert_std,
            "duration_mean": 8,
            "n_expected": 3,
        },
        "where": {
            0: ["nodes.net.hidden", "nodes.net.readout"],
        },
    }


# ---------------------------------------------------------------------------
# Loss-mode configurations
# ---------------------------------------------------------------------------

def _loss_cfg_running_cost() -> dict:
    """Loss config for running-cost mode: uniform position penalty during movement."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": 1.0,
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.5,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-5,
                "nn_hidden": 1e-5,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 2.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
        },
        "loss_update": {
            "enabled": False,  # Disabled by default; enable with --target-ratio to use adaptive control penalty
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


def _loss_cfg_softmin() -> dict:
    """Loss config for softmin mode: goal-hit-in-window objective."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 1.0,
                "effector_pos": 0.0,
                "effector_pos_running": 0.0,
                "effector_pos_mid": 1.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.5,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-5,
                "nn_hidden": 1e-5,
            },
            "effector_pos_mid": {
                "start_step_after_go": 0,
                "end_step_after_go": 80,
                "ramp_init_weight": 0.0,
                "ramp_final_weight": 0.1,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "goal_hit_in_window": {
                "start_step_after_go": 60,
                "end_step_after_go": 80,
                "softmin_tau": 0.2,
                "post_pos_sigma_t": 5.0,
                "weights": {
                    "pos": 1.0,
                    "vel": 0.1,
                    "post_pos": 1.0,
                },
            },
        },
        "loss_update": {
            "enabled": False,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_mid", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


def _loss_cfg_combined() -> dict:
    """Loss config for combined mode: weak running cost + strong goal-hit window."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 1.0,
                "effector_pos": 0.0,
                "effector_pos_running": 0.3,
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.3,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-5,
                "nn_hidden": 1e-5,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "goal_hit_in_window": {
                "start_step_after_go": 60,
                "end_step_after_go": 80,
                "softmin_tau": 0.2,
                "post_pos_sigma_t": 5.0,
                "weights": {
                    "pos": 1.0,
                    "vel": 0.1,
                    "post_pos": 1.0,
                },
            },
        },
        "loss_update": {
            "enabled": False,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


def _loss_cfg_default() -> dict:
    """Loss config for default mode: structured mid/late terms (standard Part 2 config)."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": 0.0,
                "effector_pos_mid": 1.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 1.0,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-5,
                "nn_hidden": 1e-5,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 3.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 3.0,
            },
            "effector_pos_mid": {
                "start_step_after_go": 0,
                "end_step_after_go": 80,
                "ramp_init_weight": 0.0,
                "ramp_final_weight": 0.1,
            },
        },
        "loss_update": {
            "enabled": False,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_mid", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


LOSS_MODE_CONFIGS = {
    "running_cost": _loss_cfg_running_cost,
    "softmin": _loss_cfg_softmin,
    "combined": _loss_cfg_combined,
    "default": _loss_cfg_default,
}


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Construct the full hyperparameter namespace from CLI args."""
    base = _base_hps(args)
    loss_cfg = LOSS_MODE_CONFIGS[args.loss_mode]()

    # Deep merge loss config into base
    merged = {**base, **loss_cfg}

    # Always set target_ratio from args; loss_cfg_* functions all hardcode 0.5
    # so without this the user's --target-ratio default of 0.3 would never apply.
    merged["loss_update"]["target_ratio"] = args.target_ratio

    # Enable loss update if --enable-loss-update was passed.
    # Bug: previously this flag didn't exist, so loss_update was always disabled regardless
    # of --target-ratio. The enabled flag must be set explicitly.
    if getattr(args, "enable_loss_update", False):
        merged["loss_update"]["enabled"] = True

    # Override nn_output weight if specified
    if hasattr(args, "nn_output"):
        merged["loss"]["weights"]["nn_output"] = args.nn_output
        merged["loss"]["weights"]["nn_hidden"] = args.nn_output

    # Recursively convert nested dicts to TreeNamespaces so that dot-access works
    # throughout (e.g. hps.pert.type, hps.loss_update.target_ratio).
    # TreeNamespace(**merged) is only a shallow conversion; dict_to_namespace
    # recurses into nested dicts.
    return dict_to_namespace(merged, to_type=TreeNamespace)
