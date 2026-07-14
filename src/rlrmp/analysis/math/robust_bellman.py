"""Robust Bellman diagnostics for the linear same-game gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as scipy_opt
from jaxtyping import Array, Float

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    TARGET_POS,
    GameCardReference,
    materialize_reference,
)
from rlrmp.analysis.math.hinf_riccati import CostSchedule, PlantLinearization, simulate_closed_loop
from rlrmp.analysis.math.linear_round_trip import LinearOptimizationConfig, ensemble_initial_states
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    exact_output_feedback_adversary_audit,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    robust_estimator_covariances,
    robust_estimator_joint_matrices,
    robust_output_feedback_gains,
    rollout_with_robust_estimator,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.analysis.math import require_jax_x64

ISSUE_ID = "583d764"
UMBRELLA_ID = "43e8728"
GAMMA_SWEEP_ISSUE_ID = "97604a8"
GAMMA_FACTORS = (1.35, OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR, 1.5)
_ANALYSIS_PRESET = load_analysis_parameter_preset("robust_bellman").parameters
NUMERICAL_MINMAX_TIME_INDICES = tuple(_ANALYSIS_PRESET["numerical_minmax_time_indices"])
EXACT_INNER_TIME_INDICES = tuple(_ANALYSIS_PRESET["exact_inner_time_indices"])
PRIMARY_EXACT_INNER_GAMMA_FACTORS = (OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,)


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


@dataclass(frozen=True)
class ExactInnerBellmanFit:
    """One per-time information-state Bellman fit with the inner max eliminated."""

    label: str
    gamma_factor: float
    gamma: float
    time_index: int
    K: Float[Array, "m_u n"] | None
    final_objective: float | None
    reference_objective: float | None
    zero_objective: float | None
    objective_ratio_to_reference: float | None
    gain_relative_error: float | None
    feasibility_margin: float
    feasible: bool
    optimizer_status: str
    n_iterations: int
    n_function_evaluations: int


@dataclass(frozen=True)
class FlattenedEpsilonFit:
    """Full-horizon exact epsilon fit with the disturbance trajectory eliminated."""

    label: str
    gamma_factor: float
    gamma: float
    target: str
    final_objective: float | None
    reference_objective: float | None
    cs_persistent_reference_objective: float | None
    zero_objective: float | None
    objective_ratio_to_reference: float | None
    gain_relative_error: float | None
    cs_persistent_index_gain_relative_error: float
    feasibility_margin: float
    lambda_over_gamma_squared: float
    reference_feasibility_margin: float
    reference_lambda_over_gamma_squared: float
    feasible: bool
    optimizer_status: str
    n_iterations: int
    n_function_evaluations: int


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


def information_state_persistent_index_bellman_matrices(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution,
) -> tuple[
    Float[Array, "T n n"],
    Float[Array, "T n m_u"],
    Float[Array, "T m_u m_u"],
]:
    """Return a C&S-code-fidelity persistent-index Bellman block variant.

    This keeps the formal one-step control Hessian/cross terms implied by
    ``P[t + 1]`` but replaces the hidden-state Schur complement with the
    released C&S persistent Riccati slice ``P[0]``. The resulting objective is a
    code-fidelity bridge for the released persistent-index controller, not a
    formal finite-horizon theorem.
    """

    formal_l, blocks_n, blocks_m = information_state_bellman_matrices(
        plant,
        schedule,
        solution,
    )
    del formal_l
    persistent_p = solution.P[0].astype(jnp.float64)
    blocks_l = []
    for N_t, M_t in zip(blocks_n, blocks_m, strict=True):
        schur_lift = N_t @ jnp.linalg.solve(M_t, N_t.T)
        blocks_l.append(persistent_p + schur_lift)
    return jnp.stack(blocks_l), blocks_n, blocks_m


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


def information_state_exact_inner_bellman_objective(
    L_t: Float[Array, "n n"],
    N_t: Float[Array, "n m_u"],
    M_u_t: Float[Array, "m_u m_u"],
    sigma_t: Float[Array, "n n"],
    K_t: Float[Array, "m_u n"],
    estimates: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    gamma: float,
) -> Float[Array, ""]:
    """Return information-state Bellman loss after analytic max over hidden state.

    For each estimate ``h`` and action ``u=-K h`` this maximizes
    ``x.T L x + 2 x.T N u + u.T M u - gamma^2 (x-h).T Sigma^-1 (x-h)``.
    The expression is finite only when ``gamma^2 Sigma^-1 - L`` is positive
    definite; callers should check :func:`information_state_feasibility_margin`
    before using this objective.
    """

    estimates = estimates.astype(jnp.float64)
    weights = weights.astype(jnp.float64)
    sigma_inv = jnp.linalg.inv(sigma_t.astype(jnp.float64))
    gamma2 = jnp.asarray(gamma * gamma, dtype=jnp.float64)
    D_t = gamma2 * sigma_inv - L_t.astype(jnp.float64)
    u = -estimates @ K_t.T
    b = u @ N_t.T + gamma2 * estimates @ sigma_inv.T
    maximized_terms = jnp.einsum("bi,ij,bj->b", b, jnp.linalg.inv(D_t), b)
    control_terms = jnp.einsum("bi,ij,bj->b", u, M_u_t, u)
    estimate_penalty = gamma2 * jnp.einsum("bi,ij,bj->b", estimates, sigma_inv, estimates)
    return jnp.mean(weights * (maximized_terms + control_terms - estimate_penalty))


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


def _fit_one_step_exact_inner_bellman(
    *,
    label: str,
    gamma_factor: float,
    gamma: float,
    time_index: int,
    objective,
    K_ref: Float[Array, "m_u n"],
    feasibility_margin: float,
    K_scale: float = 500.0,
    max_iterations: int = 80,
) -> ExactInnerBellmanFit:
    """Fit one information-state Bellman objective with an analytic inner max."""

    if feasibility_margin <= 0.0:
        return ExactInnerBellmanFit(
            label=label,
            gamma_factor=gamma_factor,
            gamma=gamma,
            time_index=time_index,
            K=None,
            final_objective=None,
            reference_objective=None,
            zero_objective=None,
            objective_ratio_to_reference=None,
            gain_relative_error=None,
            feasibility_margin=feasibility_margin,
            feasible=False,
            optimizer_status="unbounded: gamma^2 Sigma^-1 - L is not positive definite",
            n_iterations=0,
            n_function_evaluations=0,
        )

    k_shape = tuple(K_ref.shape)
    k_size = int(np.prod(k_shape))

    @jax.jit
    def scaled_objective(theta_k: Float[Array, " k_flat"]) -> Float[Array, ""]:
        K_t = K_scale * theta_k.reshape(k_shape)
        return objective(K_t)

    value_and_grad = jax.jit(jax.value_and_grad(scaled_objective))

    def scipy_value_and_grad(theta_k: np.ndarray) -> tuple[float, np.ndarray]:
        value, grads = value_and_grad(jnp.asarray(theta_k, dtype=jnp.float64))
        return float(value), np.asarray(grads, dtype=np.float64)

    result = scipy_opt.minimize(
        scipy_value_and_grad,
        np.zeros(k_size, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "ftol": 1e-11,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    K = K_scale * jnp.asarray(result.x, dtype=jnp.float64).reshape(k_shape)
    final_objective = float(objective(K))
    reference_objective = float(objective(K_ref))
    zero_objective = float(objective(jnp.zeros_like(K_ref)))
    return ExactInnerBellmanFit(
        label=label,
        gamma_factor=gamma_factor,
        gamma=gamma,
        time_index=time_index,
        K=K,
        final_objective=final_objective,
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        objective_ratio_to_reference=final_objective / reference_objective,
        gain_relative_error=float(jnp.linalg.norm(K - K_ref) / jnp.linalg.norm(K_ref)),
        feasibility_margin=feasibility_margin,
        feasible=True,
        optimizer_status=str(result.message),
        n_iterations=int(result.nit),
        n_function_evaluations=int(result.nfev),
    )


def train_deterministic_robust_bellman(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=250),
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
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=80, n_random_states=8),
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
    config: LinearOptimizationConfig,
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
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=250),
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
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=80, n_random_states=8),
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


def train_output_feedback_information_state_exact_inner_bellman(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    time_indices: tuple[int, ...] = EXACT_INNER_TIME_INDICES,
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=80, n_random_states=8),
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Fit the information-state Bellman objective with exact hidden-state max."""

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

        def objective(K_t, *, t=t):
            return information_state_exact_inner_bellman_objective(
                L[t],
                N[t],
                M_u[t],
                covs[t],
                K_t,
                estimates,
                weights,
                gamma_ref.gamma,
            )

        margin = information_state_feasibility_margin(L[t], covs[t], gamma_ref.gamma)
        fits.append(
            _fit_one_step_exact_inner_bellman(
                label=f"output_feedback_information_state_exact_inner_gamma_{gamma_factor:g}_t{t}",
                gamma_factor=gamma_factor,
                gamma=gamma_ref.gamma,
                time_index=t,
                objective=objective,
                K_ref=formal_K_ref[t].astype(jnp.float64),
                feasibility_margin=margin,
                max_iterations=config.n_steps,
            )
        )
    gain_errors = [fit.gain_relative_error for fit in fits if fit.gain_relative_error is not None]
    margins = jnp.asarray([fit.feasibility_margin for fit in fits], dtype=jnp.float64)
    cs_error = float(jnp.linalg.norm(formal_K_ref - cs_K_ref) / jnp.linalg.norm(formal_K_ref))
    max_gain_error = None if not gain_errors else float(max(gain_errors))
    mean_gain_error = None if not gain_errors else float(np.mean(gain_errors))
    return {
        "label": f"output_feedback_information_state_exact_inner_gamma_{gamma_factor:g}",
        "gamma_factor": gamma_factor,
        "gamma": gamma_ref.gamma,
        "target": "formal_time_indexed_information_state_exact_hidden_state_inner",
        "cs_persistent_index_gain_relative_error": cs_error,
        "recovers_formal_target": bool(max_gain_error is not None and max_gain_error < 2e-2),
        "max_gain_relative_error": max_gain_error,
        "mean_gain_relative_error": mean_gain_error,
        "min_feasibility_margin": float(jnp.min(margins)),
        "all_feasible": all(fit.feasible for fit in fits),
        "fits": [_exact_inner_summary(fit) for fit in fits],
    }


def train_output_feedback_information_state_exact_inner_persistent_index(
    reference: GameCardReference,
    *,
    gamma_factor: float,
    time_indices: tuple[int, ...] = EXACT_INNER_TIME_INDICES,
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=80, n_random_states=8),
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Fit the C&S-code-fidelity persistent-index exact-inner objective."""

    gamma_ref = next(ref for ref in reference.gamma_references if ref.factor == gamma_factor)
    plant = reference.plant
    schedule = reference.schedule
    formal_config = replace(output_feedback_config, use_matlab_persistent_m_index=False)
    persistent_config = replace(output_feedback_config, use_matlab_persistent_m_index=True)
    covs = robust_estimator_covariances(
        plant,
        schedule,
        gamma_ref.gamma,
        persistent_config,
    )
    formal_K_ref = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        formal_config,
    )
    persistent_K_ref = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        persistent_config,
    )
    L, N, M_u = information_state_persistent_index_bellman_matrices(
        plant,
        schedule,
        gamma_ref.solution,
    )
    estimates, weights = ensemble_initial_states(plant, config)
    fits = []
    fitted_gains = []
    for t in _selected_time_indices(schedule.T, time_indices):

        def objective(K_t, *, t=t):
            return information_state_exact_inner_bellman_objective(
                L[t],
                N[t],
                M_u[t],
                covs[t],
                K_t,
                estimates,
                weights,
                gamma_ref.gamma,
            )

        margin = information_state_feasibility_margin(L[t], covs[t], gamma_ref.gamma)
        fit = _fit_one_step_exact_inner_bellman(
            label=(
                "output_feedback_information_state_exact_inner_persistent_index_"
                f"gamma_{gamma_factor:g}_t{t}"
            ),
            gamma_factor=gamma_factor,
            gamma=gamma_ref.gamma,
            time_index=t,
            objective=objective,
            K_ref=persistent_K_ref[t].astype(jnp.float64),
            feasibility_margin=margin,
            max_iterations=config.n_steps,
        )
        fits.append(fit)
        if fit.K is not None:
            fitted_gains.append(
                {
                    "time_index": t,
                    "gain_error_to_formal_target": float(
                        jnp.linalg.norm(fit.K - formal_K_ref[t]) / jnp.linalg.norm(formal_K_ref[t])
                    ),
                }
            )
        else:
            fitted_gains.append({"time_index": t, "gain_error_to_formal_target": None})
    persistent_errors = [
        fit.gain_relative_error for fit in fits if fit.gain_relative_error is not None
    ]
    margins = jnp.asarray([fit.feasibility_margin for fit in fits], dtype=jnp.float64)
    formal_persistent_error = float(
        jnp.linalg.norm(formal_K_ref - persistent_K_ref) / jnp.linalg.norm(formal_K_ref)
    )
    formal_errors = [
        row["gain_error_to_formal_target"]
        for row in fitted_gains
        if row["gain_error_to_formal_target"] is not None
    ]
    max_persistent_error = None if not persistent_errors else float(max(persistent_errors))
    mean_persistent_error = None if not persistent_errors else float(np.mean(persistent_errors))
    max_formal_error = None if not formal_errors else float(max(formal_errors))
    mean_formal_error = None if not formal_errors else float(np.mean(formal_errors))
    fit_rows = []
    formal_error_by_t = {
        row["time_index"]: row["gain_error_to_formal_target"] for row in fitted_gains
    }
    for fit in fits:
        row = _exact_inner_summary(fit)
        row["gain_error_to_persistent_target"] = row.pop("gain_relative_error")
        row["gain_error_to_formal_target"] = formal_error_by_t[fit.time_index]
        fit_rows.append(row)
    return {
        "label": (
            f"output_feedback_information_state_exact_inner_persistent_index_gamma_{gamma_factor:g}"
        ),
        "gamma_factor": gamma_factor,
        "gamma": gamma_ref.gamma,
        "target": "cs_code_fidelity_persistent_index_exact_hidden_state_inner",
        "formal_target": "formal_time_indexed_information_state_exact_hidden_state_inner",
        "formal_vs_persistent_reference_gain_relative_error": formal_persistent_error,
        "recovers_persistent_target": bool(
            max_persistent_error is not None and max_persistent_error < 2e-2
        ),
        "max_gain_error_to_persistent_target": max_persistent_error,
        "mean_gain_error_to_persistent_target": mean_persistent_error,
        "max_gain_error_to_formal_target": max_formal_error,
        "mean_gain_error_to_formal_target": mean_formal_error,
        "min_feasibility_margin": float(jnp.min(margins)),
        "all_feasible": all(fit.feasible for fit in fits),
        "fits": fit_rows,
    }


def flattened_epsilon_penalized_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution,
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Return exact closed-loop ``max_epsilon cost - gamma^2 ||epsilon||^2``.

    The returned tuple is ``(objective, margin, lambda_over_gamma_squared)``.
    ``margin`` is the minimum eigenvalue of ``gamma^2 I - H_epsilon``. The
    maximization is finite only when the margin is positive.
    """

    A_joint, G_joint = robust_estimator_joint_matrices(
        plant,
        schedule,
        solution,
        K,
        estimator_covariances,
        config,
    )
    n = plant.n
    zeros = jnp.zeros((n, n), dtype=jnp.float64)
    z_dim = 2 * n
    flat_dim = schedule.T * plant.m_w
    z0 = jnp.concatenate([x0.astype(jnp.float64), x0.astype(jnp.float64)], axis=0)
    S = jnp.zeros((z_dim, flat_dim), dtype=jnp.float64)
    c = z0
    H_acc = jnp.zeros((flat_dim, flat_dim), dtype=jnp.float64)
    g_acc = jnp.zeros((flat_dim,), dtype=jnp.float64)
    const = jnp.asarray(0.0, dtype=jnp.float64)
    for t in range(schedule.T):
        K_t = K[t].astype(jnp.float64)
        M_t = jnp.block(
            [
                [schedule.Q[t].astype(jnp.float64), zeros],
                [zeros, K_t.T @ schedule.R[t].astype(jnp.float64) @ K_t],
            ]
        )
        H_acc = H_acc + S.T @ M_t @ S
        g_acc = g_acc + S.T @ M_t @ c
        const = const + c @ M_t @ c
        S_next = A_joint[t] @ S
        S_next = S_next.at[:, t * plant.m_w : (t + 1) * plant.m_w].add(G_joint)
        c = A_joint[t] @ c
        S = S_next
    terminal = jnp.block([[schedule.Q_f.astype(jnp.float64), zeros], [zeros, zeros]])
    H_acc = H_acc + S.T @ terminal @ S
    g_acc = g_acc + S.T @ terminal @ c
    const = const + c @ terminal @ c
    H_sym = 0.5 * (H_acc + H_acc.T)
    eigvals = jnp.linalg.eigvalsh(H_sym)
    max_eig = eigvals[-1]
    gamma2 = jnp.asarray(solution.gamma * solution.gamma, dtype=jnp.float64)
    margin = gamma2 - max_eig
    lhs = gamma2 * jnp.eye(flat_dim, dtype=jnp.float64) - H_sym
    epsilon_star = jnp.linalg.solve(lhs, g_acc)
    objective = const + g_acc @ epsilon_star
    return objective, margin, max_eig / gamma2


def _flattened_objective_with_barrier(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution,
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    """Return flattened objective plus a large penalty when the inner max fails."""

    objective, margin, _ratio = flattened_epsilon_penalized_objective(
        plant,
        schedule,
        solution,
        estimator_covariances,
        K,
        x0,
        config,
    )
    gamma2 = jnp.asarray(solution.gamma * solution.gamma, dtype=jnp.float64)
    tolerance = jnp.maximum(1e-9, 1e-10 * gamma2)
    violation = jnp.maximum(tolerance - margin, 0.0)
    return objective + 1e10 * (violation / tolerance) ** 2


def train_output_feedback_flattened_epsilon_exact_inner(
    reference: GameCardReference,
    *,
    gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=25),
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> FlattenedEpsilonFit:
    """Minimize the full-horizon exact flattened epsilon objective over gains."""

    gamma_ref = next(ref for ref in reference.gamma_references if ref.factor == gamma_factor)
    plant = reference.plant
    schedule = reference.schedule
    formal_config = replace(output_feedback_config, use_matlab_persistent_m_index=False)
    covs = robust_estimator_covariances(plant, schedule, gamma_ref.gamma, formal_config)
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
    x0 = make_cs_output_feedback_initial_state(plant, output_feedback_config)
    gain_scale = jnp.asarray(500.0, dtype=jnp.float64)
    shape = tuple(formal_K_ref.shape)

    @jax.jit
    def scaled_objective(theta: Float[Array, " flat"]) -> Float[Array, ""]:
        K = gain_scale * theta.reshape(shape)
        return _flattened_objective_with_barrier(
            plant,
            schedule,
            gamma_ref.solution,
            covs,
            K,
            x0,
            formal_config,
        )

    value_and_grad = jax.jit(jax.value_and_grad(scaled_objective))

    def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grads = value_and_grad(jnp.asarray(theta, dtype=jnp.float64))
        return float(value), np.asarray(grads, dtype=np.float64)

    starts = (
        np.asarray(formal_K_ref, dtype=np.float64).reshape(-1) / float(gain_scale),
        np.asarray(cs_K_ref, dtype=np.float64).reshape(-1) / float(gain_scale),
    )
    results = []
    for theta0 in starts:
        results.append(
            scipy_opt.minimize(
                scipy_value_and_grad,
                theta0,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": config.n_steps,
                    "ftol": 1e-10,
                    "gtol": 1e-7,
                    "maxls": 30,
                },
            )
        )
    result = min(results, key=lambda res: float(res.fun))
    K = gain_scale * jnp.asarray(result.x, dtype=jnp.float64).reshape(shape)

    def evaluate(K_eval) -> tuple[float, float, float]:
        objective, margin, ratio = flattened_epsilon_penalized_objective(
            plant,
            schedule,
            gamma_ref.solution,
            covs,
            K_eval,
            x0,
            formal_config,
        )
        return float(objective), float(margin), float(ratio)

    final_objective, final_margin, final_ratio = evaluate(K)
    reference_objective, reference_margin, reference_ratio = evaluate(formal_K_ref)
    cs_objective, _cs_margin, _cs_ratio = evaluate(cs_K_ref)
    zero_objective, _zero_margin, _zero_ratio = evaluate(jnp.zeros_like(formal_K_ref))
    tolerance = max(1e-9, 1e-10 * gamma_ref.gamma * gamma_ref.gamma)
    feasible = final_margin > tolerance
    return FlattenedEpsilonFit(
        label=f"output_feedback_flattened_epsilon_exact_inner_gamma_{gamma_factor:g}",
        gamma_factor=gamma_factor,
        gamma=gamma_ref.gamma,
        target="formal_time_indexed_closed_loop_flattened_epsilon",
        final_objective=final_objective if feasible else None,
        reference_objective=reference_objective,
        cs_persistent_reference_objective=cs_objective,
        zero_objective=zero_objective,
        objective_ratio_to_reference=(final_objective / reference_objective if feasible else None),
        gain_relative_error=(
            float(jnp.linalg.norm(K - formal_K_ref) / jnp.linalg.norm(formal_K_ref))
            if feasible
            else None
        ),
        cs_persistent_index_gain_relative_error=float(
            jnp.linalg.norm(formal_K_ref - cs_K_ref) / jnp.linalg.norm(formal_K_ref)
        ),
        feasibility_margin=final_margin,
        lambda_over_gamma_squared=final_ratio,
        reference_feasibility_margin=reference_margin,
        reference_lambda_over_gamma_squared=reference_ratio,
        feasible=feasible,
        optimizer_status=str(result.message) if feasible else "unbounded/infeasible margin",
        n_iterations=int(result.nit),
        n_function_evaluations=int(result.nfev),
    )


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


def _exact_inner_summary(fit: ExactInnerBellmanFit) -> dict[str, Any]:
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
        "feasibility_margin": fit.feasibility_margin,
        "feasible": fit.feasible,
        "optimizer_status": fit.optimizer_status,
        "n_iterations": fit.n_iterations,
        "n_function_evaluations": fit.n_function_evaluations,
    }


def _flattened_epsilon_summary(fit: FlattenedEpsilonFit) -> dict[str, Any]:
    return {
        "label": fit.label,
        "gamma_factor": fit.gamma_factor,
        "gamma": fit.gamma,
        "target": fit.target,
        "final_objective": fit.final_objective,
        "reference_objective": fit.reference_objective,
        "cs_persistent_reference_objective": fit.cs_persistent_reference_objective,
        "zero_objective": fit.zero_objective,
        "objective_ratio_to_reference": fit.objective_ratio_to_reference,
        "gain_relative_error": fit.gain_relative_error,
        "cs_persistent_index_gain_relative_error": fit.cs_persistent_index_gain_relative_error,
        "feasibility_margin": fit.feasibility_margin,
        "lambda_over_gamma_squared": fit.lambda_over_gamma_squared,
        "reference_feasibility_margin": fit.reference_feasibility_margin,
        "reference_lambda_over_gamma_squared": fit.reference_lambda_over_gamma_squared,
        "feasible": fit.feasible,
        "optimizer_status": fit.optimizer_status,
        "n_iterations": fit.n_iterations,
        "n_function_evaluations": fit.n_function_evaluations,
    }


def analyze_robust_bellman(
    gamma_factors: tuple[float, ...] = GAMMA_FACTORS,
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=250),
    numerical_config: LinearOptimizationConfig = LinearOptimizationConfig(
        n_steps=80, n_random_states=8
    ),
    numerical_time_indices: tuple[int, ...] = NUMERICAL_MINMAX_TIME_INDICES,
    exact_inner_gamma_factors: tuple[float, ...] = PRIMARY_EXACT_INNER_GAMMA_FACTORS,
    exact_inner_time_indices: tuple[int, ...] = EXACT_INNER_TIME_INDICES,
    exact_inner_config: LinearOptimizationConfig = LinearOptimizationConfig(
        n_steps=80, n_random_states=8
    ),
    flattened_config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=25),
    output_feedback_config: OutputFeedbackConfig = OutputFeedbackConfig(),
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Run deterministic and output-feedback robust Bellman diagnostics."""

    require_jax_x64("robust Bellman analysis")
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
    output_feedback_information_exact_inner = tuple(
        train_output_feedback_information_state_exact_inner_bellman(
            reference,
            gamma_factor=factor,
            time_indices=exact_inner_time_indices,
            config=exact_inner_config,
            output_feedback_config=output_feedback_config,
        )
        for factor in exact_inner_gamma_factors
    )
    output_feedback_information_persistent_index = tuple(
        train_output_feedback_information_state_exact_inner_persistent_index(
            reference,
            gamma_factor=factor,
            time_indices=exact_inner_time_indices,
            config=exact_inner_config,
            output_feedback_config=output_feedback_config,
        )
        for factor in exact_inner_gamma_factors
    )
    output_feedback_flattened_epsilon = tuple(
        train_output_feedback_flattened_epsilon_exact_inner(
            reference,
            gamma_factor=factor,
            config=flattened_config,
            output_feedback_config=output_feedback_config,
        )
        for factor in exact_inner_gamma_factors
    )
    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "gamma_sweep_issue": GAMMA_SWEEP_ISSUE_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="robust_bellman",
        ),
        "gamma_factors": list(gamma_factors),
        "gamma_star": reference.gamma_star,
        "training_config": config.__dict__,
        "numerical_minmax_config": numerical_config.__dict__,
        "numerical_minmax_time_indices": list(numerical_time_indices),
        "exact_inner_gamma_factors": list(exact_inner_gamma_factors),
        "exact_inner_time_indices": list(exact_inner_time_indices),
        "exact_inner_config": exact_inner_config.__dict__,
        "flattened_config": flattened_config.__dict__,
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
        "output_feedback_information_state_exact_inner": {
            "fits": list(output_feedback_information_exact_inner),
            "status": (
                "formal_time_indexed_target; hidden true state is maximized analytically "
                "when gamma^2 Sigma^-1 - L is positive definite"
            ),
        },
        "output_feedback_information_state_exact_inner_persistent_index": {
            "fits": list(output_feedback_information_persistent_index),
            "status": (
                "cs_code_fidelity_target; control Hessian/cross terms remain tied to P[t+1] "
                "but the hidden-state Schur complement uses the released persistent P[0] slice"
            ),
        },
        "output_feedback_flattened_epsilon_exact_inner": {
            "fits": [_flattened_epsilon_summary(fit) for fit in output_feedback_flattened_epsilon],
            "status": (
                "full_horizon_closed_loop_target; flattened epsilon trajectory is maximized "
                "analytically when gamma^2 I - H_epsilon is positive definite"
            ),
        },
    }


__all__ = [
    "GAMMA_FACTORS",
    "analyze_robust_bellman",
    "deterministic_inner_max_margin",
    "deterministic_numerical_minmax_bellman_objective",
    "deterministic_robust_bellman_objective",
    "flattened_epsilon_penalized_objective",
    "information_state_bellman_matrices",
    "information_state_exact_inner_bellman_objective",
    "information_state_feasibility_margin",
    "information_state_numerical_minmax_bellman_objective",
    "information_state_persistent_index_bellman_matrices",
    "joint_state_ensemble",
    "output_feedback_joint_robust_bellman_objective",
    "output_feedback_joint_value_sequence",
    "train_deterministic_numerical_minmax_bellman",
    "train_deterministic_robust_bellman",
    "train_output_feedback_flattened_epsilon_exact_inner",
    "train_output_feedback_information_state_exact_inner_bellman",
    "train_output_feedback_information_state_exact_inner_persistent_index",
    "train_output_feedback_information_state_numerical_minmax_bellman",
    "train_output_feedback_joint_robust_bellman",
]
