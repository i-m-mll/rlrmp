"""Shared numerical and orchestration helpers for the 08483d5 analyses."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.math.cs_game_card import TARGET_POS
from rlrmp.analysis.math.cs_released_simulation import CSForwardNoiseDraws, CSStochasticRollout
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    robust_estimator_covariances,
)


def stochastic_policy_rollout(
    plant: Any,
    schedule: Any,
    solution: Any,
    x0: Any,
    draws: CSForwardNoiseDraws,
    covariances: Any,
    gains: Any,
    policy: Any,
    *,
    adversarial: bool,
    config: OutputFeedbackConfig,
) -> CSStochasticRollout:
    """Roll out one robust-estimator policy under fixed stochastic draws."""

    horizon = int(gains.shape[0])
    observation = delayed_observation_matrix(plant, config)
    estimator_covariances = robust_estimator_covariances(
        plant, schedule, solution.gamma, config
    )
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)

    x_seq = [x0.astype(jnp.float64)]
    xhat_seq = [x0.astype(jnp.float64)]
    y_clean_seq = []
    y_seq = []
    u_command_seq = []
    motor_seq = []
    sdn_seq = []
    process_seq = []
    sensory_seq = []
    eps_seq = []
    zero_eps = jnp.zeros((plant.m_w,), dtype=jnp.float64)

    for step in range(horizon):
        x_t = x_seq[-1]
        xhat_t = xhat_seq[-1]
        sigma = estimator_covariances[step]
        precision = (
            jnp.linalg.inv(sigma)
            + observation.T @ observation
            - inv_gamma2 * schedule.Q[step]
        )
        middle = jnp.linalg.inv(precision)
        y_clean = observation @ x_t
        sensory = draws.sensory[step]
        y_t = y_clean + sensory
        u_command = -gains[step] @ xhat_t
        eps_t = (
            policy[step] @ jnp.concatenate([x_t, xhat_t], axis=0)
            if adversarial
            else zero_eps
        )
        motor = draws.motor[step]
        signal_dependent = jnp.einsum(
            "j,nmj,m->n",
            draws.signal_dependent_standard[step],
            covariances.signal_dependent_state,
            u_command,
        )
        process = draws.process[step]
        innovation = y_t - observation @ xhat_t
        correction = inv_gamma2 * schedule.Q[step] @ xhat_t + observation.T @ innovation
        xhat_next = (
            plant.A @ xhat_t
            + plant.B @ u_command
            + plant.A @ middle @ correction
        )
        x_next = (
            plant.A @ x_t
            + plant.B @ u_command
            + plant.Bw @ eps_t
            + motor
            + signal_dependent
            + process
        )
        y_clean_seq.append(y_clean)
        y_seq.append(y_t)
        u_command_seq.append(u_command)
        motor_seq.append(motor)
        sdn_seq.append(signal_dependent)
        process_seq.append(process)
        sensory_seq.append(sensory)
        eps_seq.append(eps_t)
        x_seq.append(x_next)
        xhat_seq.append(xhat_next)

    x = jnp.stack(x_seq, axis=0)
    u_applied = jnp.stack(u_command_seq, axis=0)
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    pos_abs = pos + TARGET_POS[None, :]
    return CSStochasticRollout(
        x=x,
        x_hat=jnp.stack(xhat_seq, axis=0),
        y_clean=jnp.stack(y_clean_seq, axis=0),
        y=jnp.stack(y_seq, axis=0),
        u_command=u_applied,
        u_applied=u_applied,
        motor_noise=jnp.stack(motor_seq, axis=0),
        signal_dependent_standard=draws.signal_dependent_standard,
        signal_dependent_noise=jnp.stack(sdn_seq, axis=0),
        process_noise=jnp.stack(process_seq, axis=0),
        sensory_noise=jnp.stack(sensory_seq, axis=0),
        adversary_epsilon=jnp.stack(eps_seq, axis=0),
        perturbations=jnp.zeros((horizon, plant.n), dtype=jnp.float64),
        peak_forward_velocity=_float(jnp.max(forward)),
        peak_forward_velocity_idx=int(jnp.argmax(forward)),
        terminal_position_error=_float(jnp.linalg.norm(pos_abs[-1] - TARGET_POS)),
        control_effort=_float(jnp.sum(u_applied**2)),
    )


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def projected_gradient_ascent(
    objective: Any,
    initial_delta: Any,
    *,
    radius: float,
    step_radius: float,
    n_steps: int,
    normalize: Any,
    project: Any,
    flattened_norm: Any,
) -> tuple[Any, Any, list[dict[str, float | int]]]:
    """Maximize a frozen-batch objective with per-trial projected gradients."""

    value_and_grad = jax.value_and_grad(objective)
    delta = initial_delta
    best_delta = delta
    best_objective = objective(delta)
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "objective": float(best_objective),
            "best_objective": float(best_objective),
            "epsilon_l2_mean": 0.0,
            "epsilon_l2_max": 0.0,
        }
    ]
    for step in range(1, n_steps + 1):
        value, grad = value_and_grad(delta)
        proposal = project(delta + normalize(grad) * step_radius, radius)
        proposal_objective = objective(proposal)
        improved = proposal_objective > best_objective
        best_delta = jnp.where(improved, proposal, best_delta)
        best_objective = jnp.where(improved, proposal_objective, best_objective)
        delta = proposal
        norms = flattened_norm(proposal)
        history.append(
            {
                "step": step,
                "objective": float(proposal_objective),
                "best_objective": float(best_objective),
                "pre_step_objective": float(value),
                "epsilon_l2_mean": float(jnp.mean(norms)),
                "epsilon_l2_max": float(jnp.max(norms)),
                "gradient_l2_mean": float(jnp.mean(flattened_norm(grad))),
            }
        )
    return best_delta, best_objective, history
