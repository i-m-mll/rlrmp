"""Shared orchestration primitives for multi-cell reach analyses."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


_TRAINING_ARG_PROFILES: dict[str, dict[str, Any]] = {
    "lit_replication": {
        "n_adversary_batches": 0,
        "batch_size": 250,
        "nn_output_jerk": 0.0,
        "seed": 42,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "loss_update_enabled": False,
        "loss_update_ratio": 0.5,
        "effector_pos_running": 1.0,
        "effector_hold_pos": 1.0,
        "effector_hold_vel": 0.0,
        "effector_pos_late_weight": 0.0,
        "effector_vel_late": 0.0,
        "effector_final_vel": 0.0,
        "effector_pos_late_final_scale": 2.0,
        "effector_pos_late_start_step": 80,
        "p_catch_trial": 0.5,
        "nn_output": 1e-5,
        "nn_hidden": 1e-5,
        "nn_hidden_derivative": 0.001,
        "nn_output_pre_go": 0.0,
        "nn_hidden_derivative_pre_go": 0.0,
        "effector_pos_running_schedule": "flat",
        "effector_hold_pos_schedule": "flat",
        "position_powerlaw_power": 6.0,
        "controller_lr": 1e-4,
    },
    "movement_ramp": {
        "n_adversary_batches": 0,
        "batch_size": 250,
        "seed": 42,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "loss_update_enabled": False,
        "loss_update_ratio": 0.5,
        "effector_pos_running": 1.0,
        "effector_hold_pos": 0.0,
        "effector_hold_vel": 0.0,
        "effector_pos_late_weight": 0.0,
        "effector_vel_late": 0.0,
        "effector_final_vel": 0.0,
        "effector_pos_late_final_scale": 2.0,
        "effector_pos_late_start_step": 80,
        "p_catch_trial": 0.5,
        "nn_output": 1e-5,
        "nn_hidden": 1e-5,
        "nn_output_jerk": 0.0,
        "nn_hidden_derivative": 0.001,
        "nn_hidden_derivative_pre_go": 0.0,
        "effector_pos_running_schedule": "movement_ramp",
        "effector_hold_pos_schedule": "flat",
        "position_powerlaw_power": 6.0,
        "controller_lr": 1e-4,
    },
}


def args_namespace(
    defaults: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    profile: str | None = None,
    n_warmup_batches: int | None = None,
    n_replicates: int | None = None,
) -> argparse.Namespace:
    """Build a CLI-compatible namespace from shared defaults and row overrides."""

    if profile is not None and defaults is not None:
        raise ValueError("pass defaults or profile, not both")
    values = dict(_TRAINING_ARG_PROFILES[profile]) if profile is not None else dict(defaults or {})
    if n_warmup_batches is not None:
        values["n_warmup_batches"] = n_warmup_batches
    if n_replicates is not None:
        values["n_replicates"] = n_replicates
    values.update(overrides or {})
    return argparse.Namespace(**values)


def legacy_task_trainer_history_skeleton(*args: Any, **kwargs: Any) -> Any:
    """Build a producing-era TaskTrainer history skeleton or fail clearly.

    These result-only readers deserialize the removed fixed-array
    ``TaskTrainerHistory`` format. Current Feedbax exposes event history instead and has no
    public equivalent for the old Equinox tree shape, so executing these historical readers
    requires their producing Feedbax checkout.
    """

    try:
        legacy_train = importlib.import_module("feedbax.train")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "historical TaskTrainerHistory reader requires the producing Feedbax checkout; "
            "current Feedbax has no compatible fixed-array history API"
        ) from error
    return legacy_train.init_task_trainer_history(*args, **kwargs)


def compute_kinematics_per_replicate(
    states: Any,
    trial_specs: Any,
    *,
    pre_go_window_steps: int = 20,
) -> dict[str, np.ndarray]:
    """Compute per-replicate forward-kinematic and pre-go drift metrics."""

    pos = states.mechanics.effector.pos
    vel = states.mechanics.effector.vel
    target_key = list(trial_specs.targets.keys())[0]
    goal = trial_specs.targets[target_key].value[:, -1, :]
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]
    _n_rep, _n_trials, n_steps, _dims = pos.shape
    time = jnp.arange(n_steps)
    after_go = time[None, None, :] >= go_idx[None, :, None]
    before_go = time[None, None, :] < go_idx[None, :, None]
    pre_go_window = before_go & (
        time[None, None, :] >= go_idx[None, :, None] - pre_go_window_steps
    )

    def positions_at_go(pos_rep: Any, indices: Any) -> Any:
        return jax.vmap(lambda trial, index: trial[index])(pos_rep, indices)

    pos_at_go = jax.vmap(positions_at_go, in_axes=(0, None))(pos, go_idx)
    direction = goal[None, :, :] - pos_at_go
    direction /= jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    forward_velocity = jnp.sum(vel * direction[:, :, None, :], axis=-1)
    post_go_velocity = jnp.where(after_go, forward_velocity, 0.0)
    peak_forward = jnp.max(post_go_velocity, axis=-1)
    time_to_peak = jnp.maximum(jnp.argmax(post_go_velocity, axis=-1) - go_idx[None, :], 0)
    relative_position = pos - pos[:, :, :1, :]
    forward_position = jnp.sum(relative_position * direction[:, :, None, :], axis=-1)
    hold_drift = jnp.where(before_go, forward_position, -jnp.inf).max(axis=-1)
    hold_drift = jnp.where(jnp.isinf(hold_drift), 0.0, hold_drift) * 1000.0
    masked = jnp.where(pre_go_window, forward_position, 0.0)
    counts = jnp.maximum(jnp.sum(pre_go_window, axis=-1), 1)
    return {
        "peak_forward_velocity": np.asarray(peak_forward),
        "time_to_peak_after_go": np.asarray(time_to_peak),
        "forward_vel_profile": np.asarray(forward_velocity),
        "pos_forward_profile": np.asarray(forward_position),
        "hold_drift_mm": np.asarray(hold_drift),
        "pre_go_rms_mm": np.asarray(jnp.sqrt(jnp.sum(masked**2, axis=-1) / counts) * 1000.0),
        "pre_go_mean_mm": np.asarray(jnp.sum(masked, axis=-1) / counts * 1000.0),
        "go_idx": np.asarray(go_idx),
    }
