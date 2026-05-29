"""Robust Bellman diagnostics for the linear same-game gate."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as scipy_opt
from jaxtyping import Array, Float

from rlrmp.analysis.cs_game_card import (
    TARGET_POS,
    GameCardReference,
    materialize_reference,
)
from rlrmp.analysis.hinf_riccati import CostSchedule, PlantLinearization, simulate_closed_loop
from rlrmp.analysis.linear_round_trip import LinearTrainingConfig, ensemble_initial_states
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    exact_output_feedback_adversary_audit,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    robust_estimator_covariances,
    robust_estimator_joint_matrices,
    robust_output_feedback_gains,
    rollout_with_robust_estimator,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

ISSUE_ID = "583d764"
UMBRELLA_ID = "43e8728"
GAMMA_SWEEP_ISSUE_ID = "97604a8"
GAMMA_FACTORS = (1.35, 1.4, 1.5)
NUMERICAL_MINMAX_TIME_INDICES = (0, 1, 10, 30, -1)


@dataclass(frozen=True)
class RobustBellmanFit:
    """One robust Bellman fitting result."""

    label: str
    gamma_factor: float
    gamma: float
    K: Float[Array, "T m_u n"]
    best_objective: float
    final_objective: float
    reference_objective: float
    zero_objective: float
    objective_ratio_to_reference: float
    gain_relative_error: float
    optimizer_status: str
    n_iterations: int
    n_function_evaluations: int
    clean_cost: float
    clean_peak_forward_velocity: float


@dataclass(frozen=True)
class NumericalMinmaxFit:
    """One per-time numerical min-max Bellman fitting result."""

    label: str
    gamma_factor: float
    gamma: float
    time_index: int
    final_objective: float
    reference_objective: float
    zero_objective: float
    objective_ratio_to_reference: float
    gain_relative_error: float
    adversary_relative_error: float | None
    feasibility_margin: float
    optimizer_status: str
    inner_optimizer_status: str
    n_outer_iterations: int
    n_outer_function_evaluations: int
    n_inner_function_evaluations: int


def deterministic_robust_bellman_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    P_next: Float[Array, "T n n"],
    K: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    gamma: float,
) -> Float[Array, ""]:
    """Return one-step deterministic full-state H-infinity Bellman objective."""

    states = states.astype(jnp.float64)
    weights = weights.astype(jnp.float64)
    gamma2 = jnp.asarray(gamma * gamma, dtype=jnp.float64)
    eye_w = jnp.eye(plant.m_w, dtype=jnp.float64)

    def one_time_loss(inputs):
        K_t, Q_t, R_t, P_t_next = inputs
        closed_loop = plant.A - plant.B @ K_t
        next_nominal = states @ closed_loop.T
        disturbance_lhs = gamma2 * eye_w - plant.Bw.T @ P_t_next @ plant.Bw
        rhs = states @ closed_loop.T @ P_t_next @ plant.Bw
        w_star = jnp.linalg.solve(disturbance_lhs, rhs.T).T
        next_state = next_nominal + w_star @ plant.Bw.T
        u = -states @ K_t.T
        state_terms = jnp.einsum("bi,ij,bj->b", states, Q_t, states)
        control_terms = jnp.einsum("bi,ij,bj->b", u, R_t, u)
        next_terms = jnp.einsum("bi,ij,bj->b", next_state, P_t_next, next_state)
        penalty_terms = gamma2 * jnp.einsum("bi,bi->b", w_star, w_star)
        return jnp.mean(weights * (state_terms + control_terms + next_terms - penalty_terms))

    losses = jax.vmap(one_time_loss)((K, schedule.Q, schedule.R, P_next))
    return jnp.mean(losses)


def deterministic_inner_max_margin(
    plant: PlantLinearization,
    P_next: Float[Array, "T n n"],
    gamma: float,
) -> float:
    """Return the minimum eigenvalue of ``gamma^2 I - Bw.T P[t+1] Bw``."""

    gamma2 = jnp.asarray(gamma * gamma, dtype=jnp.float64)
    eye_w = jnp.eye(plant.m_w, dtype=jnp.float64)
    lhs = gamma2 * eye_w[None] - jnp.einsum("ni,tno,om->tim", plant.Bw, P_next, plant.Bw)
    eigs = jax.vmap(lambda matrix: jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T)))(lhs)
    return float(jnp.min(eigs))


def deterministic_numerical_minmax_bellman_objective(
    plant: PlantLinearization,
    Q_t: Float[Array, "n n"],
    R_t: Float[Array, "m_u m_u"],
    P_next: Float[Array, "n n"],
    K_t: Float[Array, "m_u n"],
    W_t: Float[Array, "m_w n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    gamma: float,
) -> Float[Array, ""]:
    """Return one-step full-state Bellman loss for numerical min-max maps."""

    states = states.astype(jnp.float64)
    weights = weights.astype(jnp.float64)
    u = -states @ K_t.T
    w = states @ W_t.T
    next_state = states @ plant.A.T + u @ plant.B.T + w @ plant.Bw.T
    gamma2 = jnp.asarray(gamma * gamma, dtype=jnp.float64)
    state_terms = jnp.einsum("bi,ij,bj->b", states, Q_t, states)
    control_terms = jnp.einsum("bi,ij,bj->b", u, R_t, u)
    next_terms = jnp.einsum("bi,ij,bj->b", next_state, P_next, next_state)
    penalty_terms = gamma2 * jnp.einsum("bi,bi->b", w, w)
    return jnp.mean(weights * (state_terms + control_terms + next_terms - penalty_terms))


def information_state_bellman_matrices(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution,
) -> tuple[
    Float[Array, "T n n"],
    Float[Array, "T n m_u"],
    Float[Array, "T m_u m_u"],
]:
    """Return formal time-indexed information-state Bellman quadratic blocks."""

    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)
    eye_n = jnp.eye(plant.n, dtype=jnp.float64)
    blocks_l = []
    blocks_n = []
    blocks_m = []
    for t in range(schedule.T):
        P_next = solution.P[t + 1].astype(jnp.float64)
        lambda_t = jnp.linalg.solve(
            eye_n - inv_gamma2 * P_next @ plant.Bw @ plant.Bw.T,
            P_next,
        )
        blocks_l.append(schedule.Q[t].astype(jnp.float64) + plant.A.T @ lambda_t @ plant.A)
        blocks_n.append(plant.A.T @ lambda_t @ plant.B)
        blocks_m.append(schedule.R[t].astype(jnp.float64) + plant.B.T @ lambda_t @ plant.B)
    return jnp.stack(blocks_l), jnp.stack(blocks_n), jnp.stack(blocks_m)


def information_state_numerical_minmax_bellman_objective(
    L_t: Float[Array, "n n"],
    N_t: Float[Array, "n m_u"],
    M_u_t: Float[Array, "m_u m_u"],
    sigma_t: Float[Array, "n n"],
    K_t: Float[Array, "m_u n"],
    M_t: Float[Array, "n n"],
    estimates: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    gamma: float,
) -> Float[Array, ""]:
    """Return the formal information-state one-step min-max Bellman loss."""

    estimates = estimates.astype(jnp.float64)
    weights = weights.astype(jnp.float64)
    u = -estimates @ K_t.T
    x_adv = estimates @ M_t.T
    deviation = x_adv - estimates
    sigma_inv = jnp.linalg.inv(sigma_t.astype(jnp.float64))
    gamma2 = jnp.asarray(gamma * gamma, dtype=jnp.float64)
    cost_terms = (
        jnp.einsum("bi,ij,bj->b", x_adv, L_t, x_adv)
        + 2.0 * jnp.einsum("bi,ij,bj->b", x_adv, N_t, u)
        + jnp.einsum("bi,ij,bj->b", u, M_u_t, u)
    )
    penalty_terms = gamma2 * jnp.einsum("bi,ij,bj->b", deviation, sigma_inv, deviation)
    return jnp.mean(weights * (cost_terms - penalty_terms))


def information_state_feasibility_margin(
    L_t: Float[Array, "n n"],
    sigma_t: Float[Array, "n n"],
    gamma: float,
) -> float:
    """Return the minimum eigenvalue of ``gamma^2 Sigma^-1 - L_t``."""

    sigma_inv = jnp.linalg.inv(sigma_t.astype(jnp.float64))
    matrix = gamma * gamma * sigma_inv - L_t.astype(jnp.float64)
    matrix = 0.5 * (matrix + matrix.T)
    return float(jnp.min(jnp.linalg.eigvalsh(matrix)))


def _selected_time_indices(T: int, time_indices: tuple[int, ...]) -> tuple[int, ...]:
    """Return unique valid diagnostic time indices."""

    indices = []
    for idx in time_indices:
        normalized = idx if idx >= 0 else T + idx
        if normalized < 0 or normalized >= T:
            raise ValueError(f"time index {idx} is outside [0, {T}).")
        if normalized not in indices:
            indices.append(normalized)
    return tuple(indices)


def _fit_one_step_numerical_minmax(
    *,
    label: str,
    gamma_factor: float,
    gamma: float,
    time_index: int,
    objective,
    K_ref: Float[Array, "m_u n"],
    adversary_shape: tuple[int, int],
    feasibility_margin: float,
    K_scale: float = 500.0,
    adversary_scale: float = 1.0,
    adversary_initial: Array | None = None,
    max_outer_iterations: int = 80,
    max_inner_iterations: int = 60,
) -> NumericalMinmaxFit:
    """Fit one convex-concave quadratic saddle with nested numerical optimizers."""

    k_shape = tuple(K_ref.shape)
    k_size = int(np.prod(k_shape))
    adversary_size = int(np.prod(adversary_shape))

    @jax.jit
    def scaled_objective(
        theta_k: Float[Array, " k_flat"],
        theta_adv: Float[Array, " adv_flat"],
    ) -> Float[Array, ""]:
        K_t = K_scale * theta_k.reshape(k_shape)
        adversary_t = adversary_scale * theta_adv.reshape(adversary_shape)
        return objective(K_t, adversary_t)

    inner_value_and_grad = jax.jit(
        jax.value_and_grad(lambda theta_adv, theta_k: -scaled_objective(theta_k, theta_adv))
    )
    outer_value_and_grad = jax.jit(
        jax.value_and_grad(lambda theta_k, theta_adv: scaled_objective(theta_k, theta_adv))
    )
    if adversary_initial is None:
        adversary_theta0 = np.zeros(adversary_size, dtype=np.float64)
    else:
        adversary_theta0 = (
            np.asarray(adversary_initial, dtype=np.float64).reshape(-1) / adversary_scale
        )
    inner_cache = {"theta": adversary_theta0}
    inner_stats = {"nfev": 0, "status": ""}

    def maximize_for_controller(theta_k: np.ndarray, warm_start: np.ndarray | None = None):
        theta0 = inner_cache["theta"] if warm_start is None else warm_start

        def scipy_inner(theta_adv: np.ndarray) -> tuple[float, np.ndarray]:
            value, grads = inner_value_and_grad(
                jnp.asarray(theta_adv, dtype=jnp.float64),
                jnp.asarray(theta_k, dtype=jnp.float64),
            )
            return float(value), np.asarray(grads, dtype=np.float64)

        result = scipy_opt.minimize(
            scipy_inner,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": max_inner_iterations,
                "ftol": 1e-11,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )
        inner_cache["theta"] = result.x
        inner_stats["nfev"] += int(result.nfev)
        inner_stats["status"] = str(result.message)
        return result

    def scipy_outer(theta_k: np.ndarray) -> tuple[float, np.ndarray]:
        inner = maximize_for_controller(theta_k)
        value, grads = outer_value_and_grad(
            jnp.asarray(theta_k, dtype=jnp.float64),
            jnp.asarray(inner.x, dtype=jnp.float64),
        )
        return float(value), np.asarray(grads, dtype=np.float64)

    outer = scipy_opt.minimize(
        scipy_outer,
        np.zeros(k_size, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_outer_iterations,
            "ftol": 1e-11,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    final_inner = maximize_for_controller(outer.x)
    K = K_scale * jnp.asarray(outer.x, dtype=jnp.float64).reshape(k_shape)
    adversary = adversary_scale * jnp.asarray(final_inner.x, dtype=jnp.float64).reshape(
        adversary_shape
    )
    final_objective = float(objective(K, adversary))

    ref_inner = maximize_for_controller(
        np.asarray(K_ref, dtype=np.float64).reshape(-1) / K_scale,
        warm_start=final_inner.x,
    )
    reference_adversary = adversary_scale * jnp.asarray(ref_inner.x, dtype=jnp.float64).reshape(
        adversary_shape
    )
    reference_objective = float(objective(K_ref, reference_adversary))

    zero_inner = maximize_for_controller(np.zeros(k_size, dtype=np.float64))
    zero_adversary = adversary_scale * jnp.asarray(zero_inner.x, dtype=jnp.float64).reshape(
        adversary_shape
    )
    zero_objective = float(objective(jnp.zeros_like(K_ref), zero_adversary))
    reference_adv_norm = float(jnp.linalg.norm(reference_adversary))
    adversary_relative_error = (
        None
        if reference_adv_norm == 0.0
        else float(jnp.linalg.norm(adversary - reference_adversary) / reference_adv_norm)
    )
    return NumericalMinmaxFit(
        label=label,
        gamma_factor=gamma_factor,
        gamma=gamma,
        time_index=time_index,
        final_objective=final_objective,
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        objective_ratio_to_reference=final_objective / reference_objective,
        gain_relative_error=float(jnp.linalg.norm(K - K_ref) / jnp.linalg.norm(K_ref)),
        adversary_relative_error=adversary_relative_error,
        feasibility_margin=feasibility_margin,
        optimizer_status=str(outer.message),
        inner_optimizer_status=inner_stats["status"],
        n_outer_iterations=int(outer.nit),
        n_outer_function_evaluations=int(outer.nfev),
        n_inner_function_evaluations=inner_stats["nfev"],
    )


def train_deterministic_robust_bellman(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    config: LinearTrainingConfig = LinearTrainingConfig(n_steps=250),
) -> RobustBellmanFit:
    """Fit full-state robust gains against the deterministic H-infinity Bellman objective."""

    gamma_ref = next(ref for ref in reference.gamma_references if ref.factor == gamma_factor)
    plant = reference.plant
    schedule = reference.schedule
    K_ref = gamma_ref.solution.K
    shape = K_ref.shape
    states, weights = ensemble_initial_states(plant, config)
    P_next = gamma_ref.solution.P[1:].astype(jnp.float64)
    gain_scale = jnp.asarray(500.0, dtype=jnp.float64)

    def objective(z: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        gains = gain_scale * z
        return deterministic_robust_bellman_objective(
            plant,
            schedule,
            P_next,
            gains,
            states,
            weights,
            gamma_ref.gamma,
        )

    @jax.jit
    def value_and_grad_flat(theta: Float[Array, " flat"]) -> tuple[Float[Array, ""], Array]:
        z = theta.reshape(shape)
        value, grads = jax.value_and_grad(objective)(z)
        return value, grads.reshape(-1)

    def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grads = value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
        return float(value), np.asarray(grads, dtype=np.float64)

    scipy_result = scipy_opt.minimize(
        scipy_value_and_grad,
        np.zeros(int(np.prod(shape)), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": config.n_steps,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "maxls": 50,
        },
    )
    K = gain_scale * jnp.asarray(scipy_result.x, dtype=jnp.float64).reshape(shape)
    final_objective = float(
        deterministic_robust_bellman_objective(
            plant, schedule, P_next, K, states, weights, gamma_ref.gamma
        )
    )
    reference_objective = float(
        deterministic_robust_bellman_objective(
            plant,
            schedule,
            P_next,
            K_ref,
            states,
            weights,
            gamma_ref.gamma,
        )
    )
    zero_objective = float(
        deterministic_robust_bellman_objective(
            plant,
            schedule,
            P_next,
            jnp.zeros_like(K_ref),
            states,
            weights,
            gamma_ref.gamma,
        )
    )
    x0 = reference.lqr_rollout.x[0]
    rollout = simulate_closed_loop(plant, K, x0, target_pos=TARGET_POS)
    clean_cost = float(
        jnp.sum(jnp.einsum("ti,tij,tj->t", rollout.x[:-1], schedule.Q, rollout.x[:-1]))
        + jnp.sum(jnp.einsum("ti,tij,tj->t", rollout.u, schedule.R, rollout.u))
        + rollout.x[-1] @ schedule.Q_f @ rollout.x[-1]
    )
    return RobustBellmanFit(
        label=f"deterministic_full_state_gamma_{gamma_factor:g}",
        gamma_factor=gamma_factor,
        gamma=gamma_ref.gamma,
        K=K,
        best_objective=float(min(scipy_result.fun, final_objective)),
        final_objective=final_objective,
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        objective_ratio_to_reference=final_objective / reference_objective,
        gain_relative_error=float(jnp.linalg.norm(K - K_ref) / jnp.linalg.norm(K_ref)),
        optimizer_status=str(scipy_result.message),
        n_iterations=int(scipy_result.nit),
        n_function_evaluations=int(scipy_result.nfev),
        clean_cost=clean_cost,
        clean_peak_forward_velocity=rollout.peak_forward_velocity,
    )


def train_deterministic_numerical_minmax_bellman(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    time_indices: tuple[int, ...] = NUMERICAL_MINMAX_TIME_INDICES,
    config: LinearTrainingConfig = LinearTrainingConfig(n_steps=80, n_random_states=8),
) -> tuple[NumericalMinmaxFit, ...]:
    """Fit full-state robust gains with a numerical inner-outer saddle optimizer."""

    gamma_ref = next(ref for ref in reference.gamma_references if ref.factor == gamma_factor)
    plant = reference.plant
    schedule = reference.schedule
    states, weights = ensemble_initial_states(plant, config)
    fits = []
    for t in _selected_time_indices(schedule.T, time_indices):
        P_next = gamma_ref.solution.P[t + 1].astype(jnp.float64)

        def objective(K_t, W_t, *, t=t, P_next=P_next):
            return deterministic_numerical_minmax_bellman_objective(
                plant,
                schedule.Q[t],
                schedule.R[t],
                P_next,
                K_t,
                W_t,
                states,
                weights,
                gamma_ref.gamma,
            )

        margin = deterministic_inner_max_margin(
            plant,
            P_next[None],
            gamma_ref.gamma,
        )
        fits.append(
            _fit_one_step_numerical_minmax(
                label=f"deterministic_numerical_minmax_gamma_{gamma_factor:g}_t{t}",
                gamma_factor=gamma_factor,
                gamma=gamma_ref.gamma,
                time_index=t,
                objective=objective,
                K_ref=gamma_ref.solution.K[t].astype(jnp.float64),
                adversary_shape=(plant.m_w, plant.n),
                feasibility_margin=margin,
                max_outer_iterations=config.n_steps,
            )
        )
    return tuple(fits)


def joint_state_ensemble(
    plant: PlantLinearization,
    config: LinearTrainingConfig,
) -> tuple[Float[Array, "batch two_n"], Float[Array, " batch"]]:
    """Build full-rank diagnostic states for ``z=[x,xhat]``."""

    states, weights = ensemble_initial_states(plant, config)
    zeros = jnp.zeros_like(states)
    z_matched = jnp.concatenate([states, states], axis=1)
    z_true_only = jnp.concatenate([states, zeros], axis=1)
    z_est_only = jnp.concatenate([zeros, states], axis=1)
    z_states = jnp.concatenate([z_matched, z_true_only, z_est_only], axis=0)
    z_weights = jnp.concatenate([weights, weights, weights], axis=0) / 3.0
    return z_states, z_weights


def output_feedback_joint_value_sequence(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution,
    gains: Float[Array, "T m_u n"],
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "T_plus_1 two_n two_n"]:
    """Evaluate the robust value sequence for fixed output-feedback gains."""

    A_joint, G_joint = robust_estimator_joint_matrices(
        plant,
        schedule,
        solution,
        gains,
        estimator_covariances,
        config,
    )
    n = plant.n
    zeros = jnp.zeros((n, n), dtype=jnp.float64)
    terminal = jnp.block([[schedule.Q_f.astype(jnp.float64), zeros], [zeros, zeros]])
    gamma2 = solution.gamma * solution.gamma
    eye_w = jnp.eye(plant.m_w, dtype=jnp.float64)
    values = [terminal]
    S_next = terminal
    for t in range(schedule.T - 1, -1, -1):
        K_t = gains[t]
        stage = jnp.block(
            [
                [schedule.Q[t].astype(jnp.float64), zeros],
                [zeros, K_t.T @ schedule.R[t].astype(jnp.float64) @ K_t],
            ]
        )
        lhs = gamma2 * eye_w - G_joint.T @ S_next @ G_joint
        rhs = G_joint.T @ S_next @ A_joint[t]
        S_t = stage + A_joint[t].T @ S_next @ A_joint[t] + rhs.T @ jnp.linalg.solve(lhs, rhs)
        S_t = 0.5 * (S_t + S_t.T)
        values.append(S_t)
        S_next = S_t
    return jnp.stack(list(reversed(values)), axis=0)


def output_feedback_joint_robust_bellman_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution,
    value_next: Float[Array, "T two_n two_n"],
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    K: Float[Array, "T m_u n"],
    z_states: Float[Array, "batch two_n"],
    weights: Float[Array, " batch"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, ""]:
    """Return one-step joint-state output-feedback robust Bellman objective."""

    A_joint, G_joint = robust_estimator_joint_matrices(
        plant,
        schedule,
        solution,
        K,
        estimator_covariances,
        config,
    )
    n = plant.n
    x = z_states[:, :n].astype(jnp.float64)
    x_hat = z_states[:, n:].astype(jnp.float64)
    gamma2 = jnp.asarray(solution.gamma * solution.gamma, dtype=jnp.float64)
    eye_w = jnp.eye(plant.m_w, dtype=jnp.float64)

    def one_time_loss(inputs):
        K_t, Q_t, R_t, S_next, A_t = inputs
        u = -x_hat @ K_t.T
        next_nominal = z_states @ A_t.T
        disturbance_lhs = gamma2 * eye_w - G_joint.T @ S_next @ G_joint
        rhs = next_nominal @ S_next @ G_joint
        w_star = jnp.linalg.solve(disturbance_lhs, rhs.T).T
        next_state = next_nominal + w_star @ G_joint.T
        state_terms = jnp.einsum("bi,ij,bj->b", x, Q_t, x)
        control_terms = jnp.einsum("bi,ij,bj->b", u, R_t, u)
        next_terms = jnp.einsum("bi,ij,bj->b", next_state, S_next, next_state)
        penalty_terms = gamma2 * jnp.einsum("bi,bi->b", w_star, w_star)
        return jnp.mean(weights * (state_terms + control_terms + next_terms - penalty_terms))

    losses = jax.vmap(one_time_loss)((K, schedule.Q, schedule.R, value_next, A_joint))
    return jnp.mean(losses)


def train_output_feedback_joint_robust_bellman(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    config: LinearTrainingConfig = LinearTrainingConfig(n_steps=250),
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Fit C&S output-feedback gains against a joint-state robust Bellman diagnostic."""

    gamma_ref = next(ref for ref in reference.gamma_references if ref.factor == gamma_factor)
    plant = reference.plant
    schedule = reference.schedule
    covs = robust_estimator_covariances(
        plant,
        schedule,
        gamma_ref.gamma,
        output_feedback_config,
    )
    K_ref = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        output_feedback_config,
    )
    value_next = output_feedback_joint_value_sequence(
        plant,
        schedule,
        gamma_ref.solution,
        K_ref,
        covs,
        output_feedback_config,
    )[1:]
    z_states, weights = joint_state_ensemble(plant, config)
    shape = K_ref.shape
    gain_scale = jnp.asarray(500.0, dtype=jnp.float64)

    def objective(z: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        gains = gain_scale * z
        return output_feedback_joint_robust_bellman_objective(
            plant,
            schedule,
            gamma_ref.solution,
            value_next,
            covs,
            gains,
            z_states,
            weights,
            output_feedback_config,
        )

    @jax.jit
    def value_and_grad_flat(theta: Float[Array, " flat"]) -> tuple[Float[Array, ""], Array]:
        z = theta.reshape(shape)
        value, grads = jax.value_and_grad(objective)(z)
        return value, grads.reshape(-1)

    def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grads = value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
        return float(value), np.asarray(grads, dtype=np.float64)

    scipy_result = scipy_opt.minimize(
        scipy_value_and_grad,
        np.zeros(int(np.prod(shape)), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": config.n_steps,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "maxls": 50,
        },
    )
    K = gain_scale * jnp.asarray(scipy_result.x, dtype=jnp.float64).reshape(shape)
    final_objective = float(
        output_feedback_joint_robust_bellman_objective(
            plant,
            schedule,
            gamma_ref.solution,
            value_next,
            covs,
            K,
            z_states,
            weights,
            output_feedback_config,
        )
    )
    reference_objective = float(
        output_feedback_joint_robust_bellman_objective(
            plant,
            schedule,
            gamma_ref.solution,
            value_next,
            covs,
            K_ref,
            z_states,
            weights,
            output_feedback_config,
        )
    )
    zero_objective = float(
        output_feedback_joint_robust_bellman_objective(
            plant,
            schedule,
            gamma_ref.solution,
            value_next,
            covs,
            jnp.zeros_like(K_ref),
            z_states,
            weights,
            output_feedback_config,
        )
    )
    x0 = make_cs_output_feedback_initial_state(plant, output_feedback_config)
    clean = rollout_with_robust_estimator(
        plant,
        schedule,
        gamma_ref.solution,
        x0,
        gains=K,
        config=output_feedback_config,
    )
    return {
        "label": f"output_feedback_joint_gamma_{gamma_factor:g}",
        "gamma_factor": gamma_factor,
        "gamma": gamma_ref.gamma,
        "best_objective": float(min(scipy_result.fun, final_objective)),
        "final_objective": final_objective,
        "reference_objective": reference_objective,
        "zero_objective": zero_objective,
        "objective_ratio_to_reference": final_objective / reference_objective,
        "gain_relative_error": float(jnp.linalg.norm(K - K_ref) / jnp.linalg.norm(K_ref)),
        "optimizer_status": str(scipy_result.message),
        "n_iterations": int(scipy_result.nit),
        "n_function_evaluations": int(scipy_result.nfev),
        "clean_cost": output_feedback_cost(
            schedule,
            clean,
            gamma=gamma_ref.gamma,
        ).total_without_disturbance_penalty,
        "clean_peak_forward_velocity": clean.peak_forward_velocity,
    }


def train_output_feedback_information_state_numerical_minmax_bellman(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    time_indices: tuple[int, ...] = NUMERICAL_MINMAX_TIME_INDICES,
    config: LinearTrainingConfig = LinearTrainingConfig(n_steps=80, n_random_states=8),
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Fit the formal time-indexed information-state robust Bellman saddle."""

    gamma_ref = next(ref for ref in reference.gamma_references if ref.factor == gamma_factor)
    plant = reference.plant
    schedule = reference.schedule
    formal_config = replace(output_feedback_config, use_matlab_persistent_m_index=False)
    covs = robust_estimator_covariances(
        plant,
        schedule,
        gamma_ref.gamma,
        formal_config,
    )
    formal_K_ref = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        formal_config,
    )
    cs_K_ref = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        output_feedback_config,
    )
    L, N, M_u = information_state_bellman_matrices(plant, schedule, gamma_ref.solution)
    estimates, weights = ensemble_initial_states(plant, config)
    fits = []
    for t in _selected_time_indices(schedule.T, time_indices):

        def objective(K_t, M_t, *, t=t):
            return information_state_numerical_minmax_bellman_objective(
                L[t],
                N[t],
                M_u[t],
                covs[t],
                K_t,
                M_t,
                estimates,
                weights,
                gamma_ref.gamma,
            )

        fits.append(
            _fit_one_step_numerical_minmax(
                label=f"output_feedback_information_state_minmax_gamma_{gamma_factor:g}_t{t}",
                gamma_factor=gamma_factor,
                gamma=gamma_ref.gamma,
                time_index=t,
                objective=objective,
                K_ref=formal_K_ref[t].astype(jnp.float64),
                adversary_shape=(plant.n, plant.n),
                adversary_initial=jnp.eye(plant.n, dtype=jnp.float64),
                feasibility_margin=information_state_feasibility_margin(
                    L[t],
                    covs[t],
                    gamma_ref.gamma,
                ),
                max_outer_iterations=config.n_steps,
            )
        )
    gain_errors = jnp.asarray([fit.gain_relative_error for fit in fits], dtype=jnp.float64)
    margins = jnp.asarray([fit.feasibility_margin for fit in fits], dtype=jnp.float64)
    cs_error = float(jnp.linalg.norm(formal_K_ref - cs_K_ref) / jnp.linalg.norm(formal_K_ref))
    return {
        "label": f"output_feedback_information_state_minmax_gamma_{gamma_factor:g}",
        "gamma_factor": gamma_factor,
        "gamma": gamma_ref.gamma,
        "target": "formal_time_indexed_information_state",
        "cs_persistent_index_gain_relative_error": cs_error,
        "recovers_formal_target": bool(jnp.max(gain_errors) < 2e-2),
        "max_gain_relative_error": float(jnp.max(gain_errors)),
        "mean_gain_relative_error": float(jnp.mean(gain_errors)),
        "min_feasibility_margin": float(jnp.min(margins)),
        "fits": [_numerical_minmax_summary(fit) for fit in fits],
    }


def _fit_summary(fit: RobustBellmanFit) -> dict[str, Any]:
    return {
        "label": fit.label,
        "gamma_factor": fit.gamma_factor,
        "gamma": fit.gamma,
        "best_objective": fit.best_objective,
        "final_objective": fit.final_objective,
        "reference_objective": fit.reference_objective,
        "zero_objective": fit.zero_objective,
        "objective_ratio_to_reference": fit.objective_ratio_to_reference,
        "gain_relative_error": fit.gain_relative_error,
        "optimizer_status": fit.optimizer_status,
        "n_iterations": fit.n_iterations,
        "n_function_evaluations": fit.n_function_evaluations,
        "clean_cost": fit.clean_cost,
        "clean_peak_forward_velocity": fit.clean_peak_forward_velocity,
    }


def _numerical_minmax_summary(fit: NumericalMinmaxFit) -> dict[str, Any]:
    return {
        "label": fit.label,
        "gamma_factor": fit.gamma_factor,
        "gamma": fit.gamma,
        "time_index": fit.time_index,
        "final_objective": fit.final_objective,
        "reference_objective": fit.reference_objective,
        "zero_objective": fit.zero_objective,
        "objective_ratio_to_reference": fit.objective_ratio_to_reference,
        "gain_relative_error": fit.gain_relative_error,
        "adversary_relative_error": fit.adversary_relative_error,
        "feasibility_margin": fit.feasibility_margin,
        "optimizer_status": fit.optimizer_status,
        "inner_optimizer_status": fit.inner_optimizer_status,
        "n_outer_iterations": fit.n_outer_iterations,
        "n_outer_function_evaluations": fit.n_outer_function_evaluations,
        "n_inner_function_evaluations": fit.n_inner_function_evaluations,
    }


def analyze_robust_bellman(
    gamma_factors: tuple[float, ...] = GAMMA_FACTORS,
    config: LinearTrainingConfig = LinearTrainingConfig(n_steps=250),
    numerical_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=80, n_random_states=8),
    numerical_time_indices: tuple[int, ...] = NUMERICAL_MINMAX_TIME_INDICES,
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Run deterministic and output-feedback robust Bellman diagnostics."""

    reference = materialize_reference(gamma_factors=gamma_factors)
    deterministic_fits = tuple(
        train_deterministic_robust_bellman(
            reference,
            gamma_factor=factor,
            config=config,
        )
        for factor in gamma_factors
    )
    deterministic_numerical_minmax = tuple(
        fit
        for factor in gamma_factors
        for fit in train_deterministic_numerical_minmax_bellman(
            reference,
            gamma_factor=factor,
            time_indices=numerical_time_indices,
            config=numerical_config,
        )
    )
    output_feedback_rows = []
    x0 = make_cs_output_feedback_initial_state(reference.plant, output_feedback_config)
    for gamma_ref in reference.gamma_references:
        covs = robust_estimator_covariances(
            reference.plant,
            reference.schedule,
            gamma_ref.gamma,
            output_feedback_config,
        )
        gains = robust_output_feedback_gains(
            reference.plant,
            reference.schedule,
            gamma_ref.solution,
            covs,
            output_feedback_config,
        )
        clean = rollout_with_robust_estimator(
            reference.plant,
            reference.schedule,
            gamma_ref.solution,
            x0,
            gains=gains,
            config=output_feedback_config,
        )
        audit = exact_output_feedback_adversary_audit(
            label=f"output_feedback_robust_gamma_{gamma_ref.factor:g}",
            plant=reference.plant,
            schedule=reference.schedule,
            controller_gains=gains,
            x0=x0,
            budget=1e-8,
            estimator_kind="robust",
            solution=gamma_ref.solution,
            penalty_gamma=gamma_ref.gamma,
            config=output_feedback_config,
        )
        output_feedback_rows.append(
            {
                "gamma_factor": gamma_ref.factor,
                "gamma": gamma_ref.gamma,
                "clean_cost": output_feedback_cost(
                    reference.schedule,
                    clean,
                    gamma=gamma_ref.gamma,
                ).total_without_disturbance_penalty,
                "clean_peak_forward_velocity": clean.peak_forward_velocity,
                "gamma_penalized_feasible": audit["gamma_penalized"]["feasible"],
                "lambda_over_gamma_squared": audit["gamma_penalized"][
                    "max_eigenvalue_over_gamma_squared"
                ],
                "note": (
                    "Output-feedback row is a feasibility diagnostic for the C&S "
                    "released-code command law, not an independent robust Bellman fit."
                ),
            }
        )
    output_feedback_fits = tuple(
        train_output_feedback_joint_robust_bellman(
            reference,
            gamma_factor=factor,
            config=config,
            output_feedback_config=output_feedback_config,
        )
        for factor in gamma_factors
    )
    output_feedback_information_minmax = tuple(
        train_output_feedback_information_state_numerical_minmax_bellman(
            reference,
            gamma_factor=factor,
            time_indices=numerical_time_indices,
            config=numerical_config,
            output_feedback_config=output_feedback_config,
        )
        for factor in gamma_factors
    )
    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "gamma_sweep_issue": GAMMA_SWEEP_ISSUE_ID,
        "gamma_factors": list(gamma_factors),
        "gamma_star": reference.gamma_star,
        "training_config": config.__dict__,
        "numerical_minmax_config": numerical_config.__dict__,
        "numerical_minmax_time_indices": list(numerical_time_indices),
        "deterministic_full_state": {
            "fits": [_fit_summary(fit) for fit in deterministic_fits],
            "numerical_minmax_fits": [
                _numerical_minmax_summary(fit) for fit in deterministic_numerical_minmax
            ],
            "numerical_minmax_max_gain_relative_error": float(
                max(fit.gain_relative_error for fit in deterministic_numerical_minmax)
            ),
            "inner_max_min_margins": {
                str(factor): deterministic_inner_max_margin(
                    reference.plant,
                    next(ref for ref in reference.gamma_references if ref.factor == factor)
                    .solution.P[1:]
                    .astype(jnp.float64),
                    next(ref for ref in reference.gamma_references if ref.factor == factor).gamma,
                )
                for factor in gamma_factors
            },
        },
        "output_feedback_joint_diagnostic": {
            "rows": output_feedback_rows,
            "fits": list(output_feedback_fits),
            "status": (
                "joint_policy_improvement_diagnostic; value sequence is policy-evaluated "
                "from the C&S released-code output-feedback gains"
            ),
        },
        "output_feedback_information_state_numerical_minmax": {
            "fits": list(output_feedback_information_minmax),
            "status": (
                "formal_time_indexed_target; controller u=-K xhat and adversarial "
                "hidden-state selector x_adv=M xhat optimized by nested inner-outer L-BFGS-B"
            ),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render robust Bellman diagnostics."""

    det_rows = [
        "| gamma factor | objective ratio | gain rel err | clean peak vel | status |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in summary["deterministic_full_state"]["fits"]:
        det_rows.append(
            "| "
            f"{row['gamma_factor']:.6g} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['gain_relative_error']:.8g} | "
            f"{row['clean_peak_forward_velocity']:.8g} | "
            f"{row['optimizer_status']} |"
        )
    det_minmax_rows = [
        "| gamma factor | t | objective ratio | gain rel err | margin | outer nfev | inner nfev | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["deterministic_full_state"]["numerical_minmax_fits"]:
        det_minmax_rows.append(
            "| "
            f"{row['gamma_factor']:.6g} | "
            f"{row['time_index']} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['gain_relative_error']:.8g} | "
            f"{row['feasibility_margin']:.8g} | "
            f"{row['n_outer_function_evaluations']} | "
            f"{row['n_inner_function_evaluations']} | "
            f"{row['optimizer_status']} |"
        )
    of_rows = [
        "| gamma factor | lambda/gamma^2 | penalized feasible | clean peak vel |",
        "|---:|---:|---|---:|",
    ]
    for row in summary["output_feedback_joint_diagnostic"]["rows"]:
        of_rows.append(
            "| "
            f"{row['gamma_factor']:.6g} | "
            f"{row['lambda_over_gamma_squared']:.8g} | "
            f"{str(row['gamma_penalized_feasible']).lower()} | "
            f"{row['clean_peak_forward_velocity']:.8g} |"
        )
    of_fit_rows = [
        "| gamma factor | objective ratio | gain rel err | clean peak vel | status |",
        "|---:|---:|---:|---:|---|",
    ]
    min_of_gain_error = min(
        row["gain_relative_error"] for row in summary["output_feedback_joint_diagnostic"]["fits"]
    )
    for row in summary["output_feedback_joint_diagnostic"]["fits"]:
        of_fit_rows.append(
            "| "
            f"{row['gamma_factor']:.6g} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['gain_relative_error']:.8g} | "
            f"{row['clean_peak_forward_velocity']:.8g} | "
            f"{row['optimizer_status']} |"
        )
    of_info_rows = [
        "| gamma factor | target | recovers formal | max gain err | mean gain err | min margin | C&S persistent err |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["output_feedback_information_state_numerical_minmax"]["fits"]:
        of_info_rows.append(
            "| "
            f"{row['gamma_factor']:.6g} | "
            f"{row['target']} | "
            f"{str(row['recovers_formal_target']).lower()} | "
            f"{row['max_gain_relative_error']:.8g} | "
            f"{row['mean_gain_relative_error']:.8g} | "
            f"{row['min_feasibility_margin']:.8g} | "
            f"{row['cs_persistent_index_gain_relative_error']:.8g} |"
        )
    of_info_detail_rows = [
        "| gamma factor | t | objective ratio | gain rel err | margin | outer nfev | inner nfev | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in summary["output_feedback_information_state_numerical_minmax"]["fits"]:
        for row in group["fits"]:
            of_info_detail_rows.append(
                "| "
                f"{row['gamma_factor']:.6g} | "
                f"{row['time_index']} | "
                f"{row['objective_ratio_to_reference']:.8g} | "
                f"{row['gain_relative_error']:.8g} | "
                f"{row['feasibility_margin']:.8g} | "
                f"{row['n_outer_function_evaluations']} | "
                f"{row['n_inner_function_evaluations']} | "
                f"{row['optimizer_status']} |"
            )
    return f"""# Robust Bellman Diagnostics

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.
Gamma-sweep precursor: `{summary["gamma_sweep_issue"]}`.

This diagnostic tests the robust Bellman/oracle lane before any rollout
retraining. The deterministic full-state rows fit time-varying gains against
the one-step finite-horizon H-infinity Bellman objective with the inner
disturbance maximized in closed form.

Gamma star: `{summary["gamma_star"]:.8g}`.

## Deterministic Full-State Robust Bellman

{"\n".join(det_rows)}

## Deterministic Numerical Min-Max Smoke

{"\n".join(det_minmax_rows)}

## Output-Feedback Joint Diagnostic

{"\n".join(of_rows)}

## Output-Feedback Joint Policy-Improvement Fit

{"\n".join(of_fit_rows)}

Status: {summary["output_feedback_joint_diagnostic"]["status"]}.

## Output-Feedback Information-State Numerical Min-Max

{"\n".join(of_info_rows)}

Status: {summary["output_feedback_information_state_numerical_minmax"]["status"]}.

Per-time fits:

{"\n".join(of_info_detail_rows)}

The output-feedback fit is a diagnostic, not a proof of the C&S robust
separation theorem. It policy-evaluates the released-code-compatible
output-feedback gains into a joint value sequence over `z=[x,xhat]`, then asks
whether one-step robust Bellman fitting recovers those gains when control is
restricted to `u=-K xhat`. In this run it does not recover those gains: the
smallest output-feedback gain relative error is `{min_of_gain_error:.8g}`, and
the fitted objectives are lower than the released-code-compatible reference
objectives. This suggests the C&S output-feedback command law is not a fixed
point of this simple joint policy-improvement objective, or that the objective
is still underconstrained relative to the released-code robust estimator law.

The information-state numerical min-max section is the formal time-indexed
target, not the C&S persistent-index target. It optimizes `u=-K xhat` and
`x_adv=M xhat` with nested inner-outer L-BFGS-B and reports the persistent-index
gain mismatch separately. On the default time grid it does not recover the full
formal target: early steps recover tightly, while later steps find nearly
reference-valued objectives with large gain mismatch, so this remains a
diagnostic rather than a success claim.
"""


def write_outputs(issue_id: str = ISSUE_ID) -> dict[str, Any]:
    """Write robust Bellman diagnostics."""

    summary = analyze_robust_bellman()
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Robust Bellman diagnostics for the linear same-game gate. "
            "See `notes/robust_bellman.md`.\n",
            encoding="utf-8",
        )
    summary["tracked_note"] = f"results/{issue_id}/notes/robust_bellman.md"
    summary["tracked_manifest"] = f"results/{issue_id}/notes/robust_bellman_manifest.json"
    note_path = notes_dir / "robust_bellman.md"
    manifest_path = notes_dir / "robust_bellman_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "GAMMA_FACTORS",
    "analyze_robust_bellman",
    "deterministic_inner_max_margin",
    "deterministic_numerical_minmax_bellman_objective",
    "deterministic_robust_bellman_objective",
    "information_state_bellman_matrices",
    "information_state_feasibility_margin",
    "information_state_numerical_minmax_bellman_objective",
    "joint_state_ensemble",
    "output_feedback_joint_robust_bellman_objective",
    "output_feedback_joint_value_sequence",
    "render_markdown",
    "train_deterministic_numerical_minmax_bellman",
    "train_deterministic_robust_bellman",
    "train_output_feedback_information_state_numerical_minmax_bellman",
    "train_output_feedback_joint_robust_bellman",
    "write_outputs",
]
