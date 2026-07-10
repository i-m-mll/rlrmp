"""Shared orchestration primitives for multi-cell reach analyses."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def args_namespace(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Build a CLI-compatible namespace from shared defaults and row overrides."""

    values = dict(defaults)
    values.update(overrides or {})
    return argparse.Namespace(**values)


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
