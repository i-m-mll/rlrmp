"""Kinematic metric extraction from evaluation states.

Bug: 8404108 — extracted from ``scripts/eval_part2_5_figures.py`` and made
training-method-agnostic. The implementation handles both single-model and
ensembled-model state PyTrees (i.e. with or without a leading replicate
dimension).

Returned metrics: ``peak_velocity``, ``endpoint_error``, ``max_lateral_deviation``.
All metrics are computed only for timesteps at or after the per-trial go cue
(``trial_specs.timeline.epoch_bounds[:, 2]``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["compute_kinematics", "initial_effector_position", "initial_effector_velocity"]


def initial_effector_position(trial_specs) -> np.ndarray:
    """Return initial effector positions with shape ``(n_trials, 2)``."""

    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return np.asarray(position, dtype=np.float64)
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return np.asarray(init_state, dtype=np.float64)[..., :2]
    raise ValueError("trial spec does not include effector position initial state")


def initial_effector_velocity(trial_specs) -> np.ndarray:
    """Return initial effector velocities with shape ``(n_trials, 2)``."""

    for init_state in trial_specs.inits.values():
        velocity = getattr(init_state, "vel", None)
        if velocity is not None:
            return np.asarray(velocity, dtype=np.float64)
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 4:
            return np.asarray(init_state, dtype=np.float64)[..., 2:4]
    raise ValueError("trial spec does not include effector velocity initial state")


def compute_kinematics(states, trial_specs) -> dict[str, np.ndarray]:
    """Compute kinematic metrics from trial-evaluation states.

    Args:
        states: A states PyTree of shape ``(n_replicates, n_trials, n_steps, ...)``
            or ``(n_trials, n_steps, ...)``. Must have a ``.mechanics.effector``
            substructure with ``.pos`` and ``.vel`` attributes shaped accordingly.
        trial_specs: ``TaskTrialSpec`` for the trials (same length as the
            trial axis of ``states``). Reads ``trial_specs.targets`` (final
            timestep treated as the endpoint goal) and
            ``trial_specs.timeline.epoch_bounds`` (column 2 → go-cue step).

    Returns:
        A dict with three NumPy arrays of shape ``(n_replicates, n_trials)`` or
        ``(n_trials,)`` matching the input shape:

        - ``peak_velocity``: max speed across post-go-cue timesteps.
        - ``endpoint_error``: distance from final-step position to goal.
        - ``max_lateral_deviation``: max signed-distance-perpendicular-to-the
          line-from-init-position-to-goal across post-go-cue timesteps.
    """
    pos = states.mechanics.effector.pos  # (..., n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel  # (..., n_trials, n_steps, 2)

    # Goal position: targets has shape (n_trials, n_steps, 2); take final step
    # as the endpoint target. The target key is a lambda function (WhereDict).
    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value  # (n_trials, n_steps, 2)
    goal = goal_seq[:, -1, :]  # (n_trials, 2) — final timestep target

    # Go cue step: epoch_bounds[:, 2] = start of movement epoch
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)

    # Handle optional leading replicate dim
    has_rep_dim = pos.ndim == 4  # (n_rep, n_trials, n_steps, 2)

    if has_rep_dim:
        n_rep, n_trials, n_steps, _ = pos.shape
        t = jnp.arange(n_steps)
        # after_go: (n_trials, n_steps) broadcast to (n_rep, n_trials, n_steps)
        after_go = t[None, None, :] >= go_idx[None, :, None]

        speed = jnp.linalg.norm(vel, axis=-1)  # (n_rep, n_trials, n_steps)
        masked_speed = jnp.where(after_go, speed, 0.0)
        peak_velocity = jnp.max(masked_speed, axis=-1)  # (n_rep, n_trials)

        final_pos = pos[:, :, -1, :]  # (n_rep, n_trials, 2)
        endpoint_error = jnp.linalg.norm(
            final_pos - goal[None, :, :], axis=-1
        )  # (n_rep, n_trials)

        # Lateral deviation: per trial, get initial pos at go cue
        def get_init_pos_rep(pos_rep, go_idx_arr):
            # pos_rep: (n_trials, n_steps, 2), go_idx_arr: (n_trials,)
            return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_idx_arr)

        init_pos = jax.vmap(get_init_pos_rep, in_axes=(0, None))(
            pos, go_idx
        )  # (n_rep, n_trials, 2)

        direction = goal[None, :, :] - init_pos  # (n_rep, n_trials, 2)
        direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        direction_unit = direction / jnp.maximum(direction_norm, 1e-12)

        displacement = pos - init_pos[:, :, None, :]  # (n_rep, n_trials, n_steps, 2)
        along = jnp.sum(
            displacement * direction_unit[:, :, None, :], axis=-1, keepdims=True
        )
        lateral = displacement - along * direction_unit[:, :, None, :]
        lateral_dist = jnp.linalg.norm(lateral, axis=-1)  # (n_rep, n_trials, n_steps)
        masked_lateral = jnp.where(after_go, lateral_dist, 0.0)
        max_lateral_deviation = jnp.max(masked_lateral, axis=-1)  # (n_rep, n_trials)

    else:
        n_trials, n_steps, _ = pos.shape
        t = jnp.arange(n_steps)
        after_go = t[None, :] >= go_idx[:, None]

        speed = jnp.linalg.norm(vel, axis=-1)
        masked_speed = jnp.where(after_go, speed, 0.0)
        peak_velocity = jnp.max(masked_speed, axis=-1)

        final_pos = pos[:, -1, :]
        endpoint_error = jnp.linalg.norm(final_pos - goal, axis=-1)

        init_pos = jax.vmap(lambda p, idx: p[idx])(pos, go_idx)
        direction = goal - init_pos
        direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        direction_unit = direction / jnp.maximum(direction_norm, 1e-12)
        displacement = pos - init_pos[:, None, :]
        along = jnp.sum(displacement * direction_unit[:, None, :], axis=-1, keepdims=True)
        lateral = displacement - along * direction_unit[:, None, :]
        lateral_dist = jnp.linalg.norm(lateral, axis=-1)
        masked_lateral = jnp.where(after_go, lateral_dist, 0.0)
        max_lateral_deviation = jnp.max(masked_lateral, axis=-1)

    return {
        "peak_velocity": np.array(peak_velocity),
        "endpoint_error": np.array(endpoint_error),
        "max_lateral_deviation": np.array(max_lateral_deviation),
    }
