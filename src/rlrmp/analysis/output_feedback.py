"""C&S output-feedback / estimator-in-loop analytical lane.

The existing game-card lane is deterministic full augmented-state replay:
``u_t = -K_t x_t``.  C&S's released simulation code instead observes only the
delayed physical block, maintains a full augmented-state estimate, and applies
full augmented-state gains to that estimate.  This module keeps that lane local
and analytical so Phase 0/1/3 can compare both information structures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float

from rlrmp.analysis.cs_game_card import (
    INIT_POS,
    PRIMARY_GAMMA_FACTOR,
    TARGET_POS,
    CostBreakdown,
    GameCardReference,
    GammaReference,
    materialize_reference,
    reference_summary,
)
from rlrmp.analysis.hinf_riccati import (
    CostSchedule,
    PlantLinearization,
    RiccatiSolution,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

ISSUE_ID = "83fc5b5"
UMBRELLA_ID = "43e8728"
DETERMINISTIC_PHASE0_ISSUE_ID = "cb98e58"
DETERMINISTIC_PHASE1_ISSUE_ID = "a7dad8a"
DETERMINISTIC_PHASE3_ISSUE_ID = "6f5c79e"
DETERMINISTIC_CERTIFICATE_ISSUE_ID = "d01c35a"


@dataclass(frozen=True)
class OutputFeedbackConfig:
    """Numerical contract for the C&S estimator-in-loop lane."""

    n_phys: int = 8
    delay_steps: int = 5
    estimator_initial_covariance: float = 1e-2
    process_covariance_scale: float = 1e-3
    sensory_noise_scale: float = 1.0
    use_matlab_persistent_m_index: bool = True
    denominator_floor: float = 1e-12


@dataclass(frozen=True)
class OutputFeedbackRollout:
    """Plant and estimator trajectory under delayed output feedback."""

    x: Float[Array, "T_plus_1 n"]
    x_hat: Float[Array, "T_plus_1 n"]
    y: Float[Array, "T obs"]
    u: Float[Array, "T m_u"]
    epsilon: Float[Array, "T m_w"]
    estimator_covariances: Float[Array, "T_plus_1 n n"]
    peak_forward_velocity: float
    peak_forward_velocity_idx: int
    terminal_position_error: float
    control_effort: float


@dataclass(frozen=True)
class OutputFeedbackPhase1Result:
    """Phase 1 adversary comparison under estimator-in-loop dynamics."""

    reference: GameCardReference
    gamma_ref: GammaReference
    config: OutputFeedbackConfig
    fixed_policy: Float[Array, "T m_w two_n"]
    riccati_rollout: OutputFeedbackRollout
    riccati_cost: CostBreakdown
    open_loop_results: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OutputFeedbackPhase3Result:
    """Linear same-game summary under estimator-in-loop dynamics."""

    reference: GameCardReference
    gamma_ref: GammaReference
    config: OutputFeedbackConfig
    lqr_reference_rollout: OutputFeedbackRollout
    lqr_reference_cost: CostBreakdown
    lqr_reference_under_riccati_epsilon: OutputFeedbackRollout
    lqr_reference_under_riccati_cost: CostBreakdown
    hinf_reference_rollout: OutputFeedbackRollout
    hinf_reference_cost: CostBreakdown
    fitted_controller_evaluations: tuple[dict[str, Any], ...]


def delayed_observation_matrix(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "n_phys n"]:
    """Return C&S ``H`` selecting the oldest delayed physical block."""

    h = config.delay_steps
    n_phys = config.n_phys
    expected_n = (h + 1) * n_phys
    if plant.n != expected_n:
        raise ValueError(f"Expected plant.n={expected_n}; got {plant.n}.")
    H = jnp.zeros((n_phys, plant.n), dtype=jnp.float64)
    start = h * n_phys
    return H.at[:, start : start + n_phys].set(jnp.eye(n_phys, dtype=jnp.float64))


def make_cs_output_feedback_initial_state(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, " n"]:
    """Return C&S's delay-augmented initial condition.

    C&S initialize the augmented delay chain with ``kron(ones(h, 1), xinit)``:
    every physical-history block starts at the same goal-centered physical
    state. This differs from the deterministic full-state helper, which leaves
    lag blocks at zero because full-state feedback does not consume them as
    observations at ``t=0``.
    """

    x_phys = jnp.zeros((config.n_phys,), dtype=jnp.float64)
    pos_lo, pos_hi = plant.pos_slice
    x_phys = x_phys.at[pos_lo:pos_hi].set((INIT_POS - TARGET_POS).astype(jnp.float64))
    return jnp.tile(x_phys, config.delay_steps + 1)


def process_covariance(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "n n"]:
    """Return the C&S process covariance proxy for estimator recursions."""

    Q_proc = config.process_covariance_scale * (plant.B @ plant.B.T)
    top_force = Q_proc[4:6, 4:6]
    Q_proc = Q_proc.at[6:8, 6:8].set(top_force)
    return 0.5 * (Q_proc + Q_proc.T)


def measurement_covariance(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "n_phys n_phys"]:
    """Return the C&S sensory covariance proxy."""

    Q_proc = process_covariance(plant, config)
    return (
        jnp.eye(config.n_phys, dtype=jnp.float64)
        * Q_proc[4, 4]
        * config.sensory_noise_scale
    )


def robust_estimator_covariances(
    plant: PlantLinearization,
    schedule: CostSchedule,
    gamma: float,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "T_plus_1 n n"]:
    """Return C&S robust estimator covariance sequence ``Sigma``.

    This follows ``minmaxfc_pointMass.m`` lines 253-255 with ``D D^T`` replaced
    by the equivalent ``B_w B_w^T`` selector for the current physical block.
    """

    A = plant.A.astype(jnp.float64)
    H = delayed_observation_matrix(plant, config)
    Q_proc = plant.Bw @ plant.Bw.T
    gamma_arr = jnp.asarray(gamma, dtype=jnp.float64)
    inv_gamma2 = jnp.where(jnp.isfinite(gamma_arr), 1.0 / (gamma_arr * gamma_arr), 0.0)
    Sigma = (
        jnp.eye(plant.n, dtype=jnp.float64)
        * jnp.asarray(config.estimator_initial_covariance, dtype=jnp.float64)
    )
    covariances = [Sigma]
    for t in range(schedule.T):
        precision = (
            jnp.linalg.inv(Sigma)
            + H.T @ H
            - inv_gamma2 * schedule.Q[t].astype(jnp.float64)
        )
        middle = jnp.linalg.inv(precision)
        Sigma = A @ middle @ A.T + Q_proc
        Sigma = 0.5 * (Sigma + Sigma.T)
        covariances.append(Sigma)
    return jnp.stack(covariances, axis=0)


def kalman_estimator_gains(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "T n obs"]:
    """Return Kalman-style gains for the delayed observation.

    This is the local analogue of the C&S ``computeExtKalman`` path for the
    non-robust LQG arm. It keeps the state estimator explicit while avoiding
    the broader extended-LQG iteration machinery in the first fidelity lane.
    """

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    H = delayed_observation_matrix(plant, config)
    Q_proc = process_covariance(plant, config)
    R_obs = measurement_covariance(plant, config)
    Sigma_e = (
        jnp.eye(plant.n, dtype=jnp.float64)
        * jnp.asarray(config.estimator_initial_covariance, dtype=jnp.float64)
    )
    Sigma_x = Sigma_e
    Sigma_ex = jnp.zeros((plant.n, plant.n), dtype=jnp.float64)
    gains = []
    for t in range(controller_gains.shape[0]):
        L = controller_gains[t].astype(jnp.float64)
        s_temp = Sigma_e + Sigma_x + Sigma_ex + Sigma_ex.T
        S = H @ Sigma_e @ H.T + R_obs
        K_est = A @ Sigma_e @ H.T @ jnp.linalg.inv(S)
        gains.append(K_est)
        Sigma_e_prev = Sigma_e
        Sigma_e = Q_proc + (A - K_est @ H) @ Sigma_e @ A.T
        term = (A - B @ L) @ Sigma_ex @ H.T @ K_est.T
        Sigma_x = (
            K_est @ H @ Sigma_e_prev @ A.T
            + (A - B @ L) @ Sigma_x @ (A - B @ L).T
            + term
            + term.T
        )
        Sigma_ex = (A - B @ L) @ Sigma_ex @ (A - K_est @ H).T
        Sigma_e = 0.5 * (Sigma_e + Sigma_e.T)
        Sigma_x = 0.5 * (Sigma_x + Sigma_x.T)
        _ = s_temp
    return jnp.stack(gains, axis=0)


def robust_output_feedback_gains(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "T m_u n"]:
    """Return C&S-style robust gains applied to the estimated augmented state."""

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    BwBwT = plant.Bw @ plant.Bw.T
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)
    eye_n = jnp.eye(plant.n, dtype=jnp.float64)
    gains = []
    for t in range(schedule.T):
        P_next = solution.P[t + 1]
        base = B.T @ jnp.linalg.inv(jnp.linalg.inv(P_next) + B @ B.T - inv_gamma2 * BwBwT) @ A
        p_idx = 0 if config.use_matlab_persistent_m_index else t
        correction = jnp.linalg.inv(eye_n - inv_gamma2 * estimator_covariances[t] @ solution.P[p_idx])
        gains.append(base @ correction)
    return jnp.stack(gains, axis=0)


def robust_estimator_joint_matrices(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    gains: Float[Array, "T m_u n"],
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[Float[Array, "T two_n two_n"], Float[Array, "two_n m_w"]]:
    """Return joint true/estimated-state dynamics for fixed robust feedback.

    The joint state is ``z_t = [x_t, xhat_t]``. This is the correct closed-loop
    state for an output-feedback adversary because ``u_t`` depends on
    ``xhat_t`` while the disturbance enters the true plant.
    """

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)
    H = delayed_observation_matrix(plant, config)
    HH = H.T @ H
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)
    matrices = []
    for t in range(schedule.T):
        Sigma = estimator_covariances[t]
        middle = jnp.linalg.inv(jnp.linalg.inv(Sigma) + HH - inv_gamma2 * schedule.Q[t])
        estimator_from_x = A @ middle @ HH
        estimator_from_xhat = (
            A - B @ gains[t] + A @ middle @ (inv_gamma2 * schedule.Q[t] - HH)
        )
        top = jnp.concatenate([A, -B @ gains[t]], axis=1)
        bottom = jnp.concatenate([estimator_from_x, estimator_from_xhat], axis=1)
        matrices.append(jnp.concatenate([top, bottom], axis=0))
    G = jnp.concatenate([Bw, jnp.zeros_like(Bw)], axis=0)
    return jnp.stack(matrices, axis=0), G


def robust_estimator_fixed_adversary_policy(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    gains: Float[Array, "T m_u n"],
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "T m_w two_n"]:
    """Return the fixed-policy worst-case disturbance over ``[x, xhat]``."""

    A_joint, G_joint = robust_estimator_joint_matrices(
        plant,
        schedule,
        solution,
        gains,
        estimator_covariances,
        config,
    )
    n = plant.n
    m_w = plant.m_w
    gamma2 = solution.gamma * solution.gamma
    eye_w = jnp.eye(m_w, dtype=jnp.float64)
    P_next = jnp.block(
        [
            [schedule.Q_f.astype(jnp.float64), jnp.zeros((n, n), dtype=jnp.float64)],
            [jnp.zeros((n, n), dtype=jnp.float64), jnp.zeros((n, n), dtype=jnp.float64)],
        ]
    )
    policies = []
    for t in range(schedule.T - 1, -1, -1):
        K_t = gains[t]
        Q_joint = jnp.block(
            [
                [schedule.Q[t].astype(jnp.float64), jnp.zeros((n, n), dtype=jnp.float64)],
                [
                    jnp.zeros((n, n), dtype=jnp.float64),
                    K_t.T @ schedule.R[t].astype(jnp.float64) @ K_t,
                ],
            ]
        )
        lhs = gamma2 * eye_w - G_joint.T @ P_next @ G_joint
        rhs = G_joint.T @ P_next @ A_joint[t]
        F_t = jnp.linalg.solve(lhs, rhs)
        P_t = Q_joint + A_joint[t].T @ P_next @ A_joint[t] + rhs.T @ F_t
        P_t = 0.5 * (P_t + P_t.T)
        policies.append(F_t)
        P_next = P_t
    return jnp.stack(list(reversed(policies)), axis=0)


def _rollout_summary_fields(
    plant: PlantLinearization,
    x: Float[Array, "T_plus_1 n"],
    u: Float[Array, "T m_u"],
    target_pos: Float[Array, " 2"] = TARGET_POS,
) -> tuple[float, int, float, float]:
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos_abs = pos + target_pos[None, :]
    terminal = jnp.linalg.norm(pos_abs[-1] - target_pos)
    return (
        float(jnp.max(forward)),
        int(jnp.argmax(forward)),
        float(terminal),
        float(jnp.sum(u**2)),
    )


def _robust_estimator_rollout_arrays(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"],
    *,
    gains: Float[Array, "T m_u n"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[Float[Array, "T_plus_1 n"], Float[Array, "T_plus_1 n"], Float[Array, "T n_phys"], Float[Array, "T m_u"], Float[Array, "T_plus_1 n n"]]:
    """Return robust estimator-loop arrays without Python scalar conversion."""

    H = delayed_observation_matrix(plant, config)
    covs = robust_estimator_covariances(plant, schedule, solution.gamma, config)
    K_eff = (
        robust_output_feedback_gains(plant, schedule, solution, covs, config)
        if gains is None
        else gains
    )
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)

    def step(carry, inputs):
        x_t, xhat_t = carry
        eps_t, K_t, Sigma_t, Q_t = inputs
        precision = jnp.linalg.inv(Sigma_t) + H.T @ H - inv_gamma2 * Q_t
        middle = jnp.linalg.inv(precision)
        y_t = H @ x_t
        u_t = -K_t @ xhat_t
        innovation = y_t - H @ xhat_t
        robust_correction = inv_gamma2 * Q_t @ xhat_t + H.T @ innovation
        xhat_next = plant.A @ xhat_t + plant.B @ u_t + plant.A @ middle @ robust_correction
        x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t
        return (x_next, xhat_next), (x_next, xhat_next, y_t, u_t)

    (_, _), (x_tail, xhat_tail, y, u) = jax.lax.scan(
        step,
        (x0.astype(jnp.float64), x0.astype(jnp.float64)),
        (epsilon.astype(jnp.float64), K_eff, covs[:-1], schedule.Q.astype(jnp.float64)),
    )
    x = jnp.concatenate([x0[None].astype(jnp.float64), x_tail], axis=0)
    x_hat = jnp.concatenate([x0[None].astype(jnp.float64), xhat_tail], axis=0)
    return x, x_hat, y, u, covs


def rollout_with_kalman_estimator(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackRollout:
    """Roll ``u_t = -K_t xhat_t`` with a Kalman-style delayed observer."""

    T = K.shape[0]
    eps = jnp.zeros((T, plant.m_w), dtype=jnp.float64) if epsilon is None else epsilon
    H = delayed_observation_matrix(plant, config)
    gains = kalman_estimator_gains(plant, K, config)
    Sigma = (
        jnp.eye(plant.n, dtype=jnp.float64)
        * jnp.asarray(config.estimator_initial_covariance, dtype=jnp.float64)
    )
    covariances = [Sigma]
    x_seq = [x0.astype(jnp.float64)]
    xhat_seq = [x0.astype(jnp.float64)]
    y_seq = []
    u_seq = []
    for t in range(T):
        x_t = x_seq[-1]
        xhat_t = xhat_seq[-1]
        y_t = H @ x_t
        u_t = -K[t] @ xhat_t
        xhat_next = plant.A @ xhat_t + plant.B @ u_t + gains[t] @ (y_t - H @ xhat_t)
        x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps[t]
        Sigma = (plant.A - gains[t] @ H) @ Sigma @ plant.A.T + process_covariance(plant, config)
        Sigma = 0.5 * (Sigma + Sigma.T)
        y_seq.append(y_t)
        u_seq.append(u_t)
        x_seq.append(x_next)
        xhat_seq.append(xhat_next)
        covariances.append(Sigma)
    x = jnp.stack(x_seq, axis=0)
    u = jnp.stack(u_seq, axis=0)
    peak, peak_idx, terminal, effort = _rollout_summary_fields(plant, x, u)
    return OutputFeedbackRollout(
        x=x,
        x_hat=jnp.stack(xhat_seq, axis=0),
        y=jnp.stack(y_seq, axis=0),
        u=u,
        epsilon=eps,
        estimator_covariances=jnp.stack(covariances, axis=0),
        peak_forward_velocity=peak,
        peak_forward_velocity_idx=peak_idx,
        terminal_position_error=terminal,
        control_effort=effort,
    )


def rollout_with_robust_estimator(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"] | None = None,
    *,
    gains: Float[Array, "T m_u n"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackRollout:
    """Roll the C&S robust estimator-in-loop command law."""

    T = solution.K.shape[0]
    eps = jnp.zeros((T, plant.m_w), dtype=jnp.float64) if epsilon is None else epsilon
    x, x_hat, y, u, covs = _robust_estimator_rollout_arrays(
        plant,
        schedule,
        solution,
        x0,
        eps,
        gains=gains,
        config=config,
    )
    peak, peak_idx, terminal, effort = _rollout_summary_fields(plant, x, u)
    return OutputFeedbackRollout(
        x=x,
        x_hat=x_hat,
        y=y,
        u=u,
        epsilon=eps,
        estimator_covariances=covs,
        peak_forward_velocity=peak,
        peak_forward_velocity_idx=peak_idx,
        terminal_position_error=terminal,
        control_effort=effort,
    )


def rollout_with_robust_estimator_policy(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    x0: Float[Array, " n"],
    policy: Float[Array, "T m_w two_n"],
    *,
    gains: Float[Array, "T m_u n"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackRollout:
    """Roll robust output feedback with ``epsilon_t = F_t [x_t, xhat_t]``."""

    H = delayed_observation_matrix(plant, config)
    covs = robust_estimator_covariances(plant, schedule, solution.gamma, config)
    K_eff = (
        robust_output_feedback_gains(plant, schedule, solution, covs, config)
        if gains is None
        else gains
    )
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)
    x_seq = [x0.astype(jnp.float64)]
    xhat_seq = [x0.astype(jnp.float64)]
    y_seq = []
    u_seq = []
    eps_seq = []
    for t in range(schedule.T):
        x_t = x_seq[-1]
        xhat_t = xhat_seq[-1]
        Sigma = covs[t]
        middle = jnp.linalg.inv(jnp.linalg.inv(Sigma) + H.T @ H - inv_gamma2 * schedule.Q[t])
        y_t = H @ x_t
        u_t = -K_eff[t] @ xhat_t
        eps_t = policy[t] @ jnp.concatenate([x_t, xhat_t], axis=0)
        innovation = y_t - H @ xhat_t
        correction = inv_gamma2 * schedule.Q[t] @ xhat_t + H.T @ innovation
        xhat_next = plant.A @ xhat_t + plant.B @ u_t + plant.A @ middle @ correction
        x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t
        y_seq.append(y_t)
        u_seq.append(u_t)
        eps_seq.append(eps_t)
        x_seq.append(x_next)
        xhat_seq.append(xhat_next)
    x = jnp.stack(x_seq, axis=0)
    u = jnp.stack(u_seq, axis=0)
    peak, peak_idx, terminal, effort = _rollout_summary_fields(plant, x, u)
    return OutputFeedbackRollout(
        x=x,
        x_hat=jnp.stack(xhat_seq, axis=0),
        y=jnp.stack(y_seq, axis=0),
        u=u,
        epsilon=jnp.stack(eps_seq, axis=0),
        estimator_covariances=covs,
        peak_forward_velocity=peak,
        peak_forward_velocity_idx=peak_idx,
        terminal_position_error=terminal,
        control_effort=effort,
    )


def output_feedback_cost(
    schedule: CostSchedule,
    rollout: OutputFeedbackRollout,
    *,
    gamma: float | None = None,
) -> CostBreakdown:
    """Return the same task-cost decomposition for an output-feedback rollout."""

    x = rollout.x.astype(jnp.float64)
    u = rollout.u.astype(jnp.float64)
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    state_stage = float(jnp.sum(state_terms))
    control_stage = float(jnp.sum(control_terms))
    terminal_state = float(terminal)
    total = state_stage + control_stage + terminal_state
    disturbance_energy = float(jnp.sum(rollout.epsilon**2))
    h_inf_objective = None
    if gamma is not None:
        h_inf_objective = total - float(gamma * gamma) * disturbance_energy
    return CostBreakdown(
        state_stage=state_stage,
        control_stage=control_stage,
        terminal_state=terminal_state,
        total_without_disturbance_penalty=total,
        disturbance_energy=disturbance_energy,
        h_infinity_objective=h_inf_objective,
    )


def analyze_phase0b_output_feedback(
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Materialize Phase 0B output-feedback reference metrics."""

    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    lqr_rollout = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        config=config,
    )
    hinf_rollout = rollout_with_robust_estimator(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        config=config,
    )
    return {
        "reference": reference,
        "gamma_ref": gamma_ref,
        "config": config,
        "lqr_rollout": lqr_rollout,
        "lqr_cost": output_feedback_cost(reference.schedule, lqr_rollout),
        "hinf_rollout": hinf_rollout,
        "hinf_cost": output_feedback_cost(reference.schedule, hinf_rollout),
    }


def _objective_value_output_feedback(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    x, _x_hat, _y, u, _covs = _robust_estimator_rollout_arrays(
        plant,
        schedule,
        solution,
        x0,
        epsilon,
        config=config,
    )
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    return jnp.sum(state_terms) + jnp.sum(control_terms) + terminal


def _project_l2_ball(epsilon: Float[Array, "T m_w"], radius: float) -> Float[Array, "T m_w"]:
    norm = jnp.linalg.norm(epsilon)
    return epsilon * jnp.minimum(1.0, radius / (norm + 1e-30))


def optimize_open_loop_output_feedback(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    x0: Float[Array, " n"],
    *,
    budget: float,
    n_steps: int,
    n_restarts: int,
    learning_rate: float,
    seed: int,
    initial_candidates: tuple[Float[Array, "T m_w"], ...],
    config: OutputFeedbackConfig,
) -> dict[str, Any]:
    """Projected-ascent open-loop epsilon search through the estimator loop."""

    radius = float(jnp.sqrt(jnp.asarray(budget, dtype=jnp.float64)))
    shape = (solution.K.shape[0], plant.m_w)
    n_random = max(0, n_restarts - len(initial_candidates))
    keys = jr.split(jr.PRNGKey(seed), n_random)
    starts = [_project_l2_ball(candidate.astype(jnp.float64), radius) for candidate in initial_candidates]
    starts.extend(_project_l2_ball(jr.normal(key, shape, dtype=jnp.float64), radius) for key in keys)
    if not starts:
        starts = [jnp.zeros(shape, dtype=jnp.float64)]
    optimizer = optax.adam(learning_rate)
    value_and_grad = jax.value_and_grad(
        lambda eps: _objective_value_output_feedback(plant, schedule, solution, x0, eps, config)
    )

    @jax.jit
    def run(start: Float[Array, "T m_w"]):
        eps = start
        opt_state = optimizer.init(eps)
        best_eps = eps
        best_value = _objective_value_output_feedback(plant, schedule, solution, x0, eps, config)

        def step(carry, _):
            eps, opt_state, best_eps, best_value = carry
            _value, grads = value_and_grad(eps)
            updates, opt_state = optimizer.update(-grads, opt_state, eps)
            eps = _project_l2_ball(optax.apply_updates(eps, updates), radius)
            value = _objective_value_output_feedback(plant, schedule, solution, x0, eps, config)
            improved = value > best_value
            best_eps = jnp.where(improved, eps, best_eps)
            best_value = jnp.maximum(value, best_value)
            return (eps, opt_state, best_eps, best_value), None

        (eps, _opt_state, best_eps, best_value), _ = jax.lax.scan(
            step,
            (eps, opt_state, best_eps, best_value),
            None,
            length=n_steps,
        )
        return best_eps, eps, best_value

    initial_objectives = []
    final_objectives = []
    best_objectives = []
    final_energies = []
    best_energies = []
    best_epsilons = []
    for start in starts:
        initial_objectives.append(float(_objective_value_output_feedback(plant, schedule, solution, x0, start, config)))
        best_eps, final_eps, best_value = run(start)
        best_epsilons.append(best_eps)
        final_objectives.append(float(_objective_value_output_feedback(plant, schedule, solution, x0, final_eps, config)))
        best_objectives.append(float(best_value))
        final_energies.append(float(jnp.sum(final_eps**2)))
        best_energies.append(float(jnp.sum(best_eps**2)))
    best_idx = int(jnp.argmax(jnp.asarray(best_objectives)))
    best_epsilon = best_epsilons[best_idx]
    best_rollout = rollout_with_robust_estimator(plant, schedule, solution, x0, best_epsilon, config=config)
    best_cost = output_feedback_cost(schedule, best_rollout, gamma=solution.gamma)
    return {
        "n_steps": n_steps,
        "n_restarts": n_restarts,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_restart_idx": best_idx,
        "epsilon": best_epsilon,
        "rollout": best_rollout,
        "cost": best_cost,
        "initial_objectives": tuple(initial_objectives),
        "final_objectives": tuple(final_objectives),
        "best_objectives": tuple(best_objectives),
        "final_energies": tuple(final_energies),
        "best_energies": tuple(best_energies),
    }


def analyze_phase1_output_feedback(
    *,
    step_sweep: tuple[int, ...] = (50, 200),
    n_restarts: int = 4,
    learning_rate: float = 3e-2,
    seed: int = 0,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackPhase1Result:
    """Rerun Phase 1 under C&S estimator-in-loop dynamics."""

    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    covs = robust_estimator_covariances(
        reference.plant, reference.schedule, gamma_ref.gamma, config
    )
    gains = robust_output_feedback_gains(
        reference.plant, reference.schedule, gamma_ref.solution, covs, config
    )
    F = robust_estimator_fixed_adversary_policy(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        gains,
        covs,
        config,
    )
    riccati_rollout = rollout_with_robust_estimator_policy(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        F,
        gains=gains,
        config=config,
    )
    riccati_cost = output_feedback_cost(reference.schedule, riccati_rollout, gamma=gamma_ref.gamma)
    budget = riccati_cost.disturbance_energy
    results = []
    for idx, n_steps in enumerate(step_sweep):
        results.append(
            optimize_open_loop_output_feedback(
                reference.plant,
                reference.schedule,
                gamma_ref.solution,
                x0,
                budget=budget,
                n_steps=int(n_steps),
                n_restarts=n_restarts,
                learning_rate=learning_rate,
                seed=seed + 1009 * idx,
                initial_candidates=(riccati_rollout.epsilon,),
                config=config,
            )
        )
    return OutputFeedbackPhase1Result(
        reference=reference,
        gamma_ref=gamma_ref,
        config=config,
        fixed_policy=F,
        riccati_rollout=riccati_rollout,
        riccati_cost=riccati_cost,
        open_loop_results=tuple(results),
    )


def analyze_phase3_output_feedback(
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackPhase3Result:
    """Compute Phase 3 estimator-in-loop reference audits."""

    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    covs = robust_estimator_covariances(
        reference.plant, reference.schedule, gamma_ref.gamma, config
    )
    gains = robust_output_feedback_gains(
        reference.plant, reference.schedule, gamma_ref.solution, covs, config
    )
    F = robust_estimator_fixed_adversary_policy(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        gains,
        covs,
        config,
    )
    nominal_hinf = rollout_with_robust_estimator_policy(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        F,
        gains=gains,
        config=config,
    )
    riccati_epsilon = nominal_hinf.epsilon
    lqr_rollout = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        config=config,
    )
    lqr_under_eps = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        riccati_epsilon,
        config=config,
    )
    hinf_rollout = rollout_with_robust_estimator(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        riccati_epsilon,
        config=config,
    )
    fitted_evaluations = []
    round_trip_artifact = (
        REPO_ROOT / "_artifacts" / DETERMINISTIC_PHASE3_ISSUE_ID
        / "linear_round_trip" / "linear_round_trip.npz"
    )
    if round_trip_artifact.exists():
        with np.load(round_trip_artifact) as arrays:
            for label in ("adam_lqr_fit", "lbfgsb_after_adam_lqr_fit"):
                K = jnp.asarray(arrays[f"{label.replace('_fit', '')}_K"], dtype=jnp.float64)
                clean = rollout_with_kalman_estimator(reference.plant, K, x0, config=config)
                under_eps = rollout_with_kalman_estimator(
                    reference.plant,
                    K,
                    x0,
                    riccati_epsilon,
                    config=config,
                )
                action_delta = clean.u - lqr_rollout.u
                fitted_evaluations.append(
                    {
                        "label": label,
                        "source_artifact": (
                            f"_artifacts/{DETERMINISTIC_PHASE3_ISSUE_ID}/"
                            "linear_round_trip/linear_round_trip.npz"
                        ),
                        "clean_rollout": clean,
                        "clean_cost": output_feedback_cost(reference.schedule, clean),
                        "under_riccati_epsilon_rollout": under_eps,
                        "under_riccati_epsilon_cost": output_feedback_cost(
                            reference.schedule,
                            under_eps,
                            gamma=gamma_ref.gamma,
                        ),
                        "clean_action_mismatch_rms_vs_lqr_reference": float(
                            jnp.sqrt(jnp.mean(action_delta**2))
                        ),
                        "clean_action_reference_rms": float(jnp.sqrt(jnp.mean(lqr_rollout.u**2))),
                    }
                )
    return OutputFeedbackPhase3Result(
        reference=reference,
        gamma_ref=gamma_ref,
        config=config,
        lqr_reference_rollout=lqr_rollout,
        lqr_reference_cost=output_feedback_cost(reference.schedule, lqr_rollout),
        lqr_reference_under_riccati_epsilon=lqr_under_eps,
        lqr_reference_under_riccati_cost=output_feedback_cost(
            reference.schedule,
            lqr_under_eps,
            gamma=gamma_ref.gamma,
        ),
        hinf_reference_rollout=hinf_rollout,
        hinf_reference_cost=output_feedback_cost(reference.schedule, hinf_rollout, gamma=gamma_ref.gamma),
        fitted_controller_evaluations=tuple(fitted_evaluations),
    )


def _cost_summary(cost: CostBreakdown) -> dict[str, float | None]:
    return {
        "state_stage": cost.state_stage,
        "control_stage": cost.control_stage,
        "terminal_state": cost.terminal_state,
        "total_without_disturbance_penalty": cost.total_without_disturbance_penalty,
        "disturbance_energy": cost.disturbance_energy,
        "h_infinity_objective": cost.h_infinity_objective,
    }


def _rollout_summary(rollout: OutputFeedbackRollout) -> dict[str, float | int]:
    return {
        "peak_forward_velocity": rollout.peak_forward_velocity,
        "time_to_peak_step": rollout.peak_forward_velocity_idx,
        "terminal_position_error_m": rollout.terminal_position_error,
        "control_effort": rollout.control_effort,
        "estimation_error_final_l2": float(jnp.linalg.norm(rollout.x[-1] - rollout.x_hat[-1])),
        "estimation_error_rms": float(jnp.sqrt(jnp.mean((rollout.x - rollout.x_hat) ** 2))),
    }


def result_summary(
    phase0b: dict[str, Any],
    phase1: OutputFeedbackPhase1Result,
    phase3: OutputFeedbackPhase3Result,
) -> dict[str, Any]:
    """Return JSON-serializable combined output-feedback lane summary."""

    deterministic = reference_summary(phase0b["reference"])
    lqr_cost = phase3.lqr_reference_cost.total_without_disturbance_penalty
    hinf_cost = phase3.hinf_reference_cost.total_without_disturbance_penalty
    phase1_rows = []
    ric_total = phase1.riccati_cost.total_without_disturbance_penalty
    for result in phase1.open_loop_results:
        total = result["cost"].total_without_disturbance_penalty
        phase1_rows.append(
            {
                "n_steps": result["n_steps"],
                "n_restarts": result["n_restarts"],
                "best_restart_idx": result["best_restart_idx"],
                "best_cost": _cost_summary(result["cost"]),
                "total_cost_ratio_to_riccati": total / ric_total,
                "epsilon_l2_distance_to_riccati": float(
                    jnp.linalg.norm(result["epsilon"] - phase1.riccati_rollout.epsilon)
                ),
                "initial_objectives": list(result["initial_objectives"]),
                "final_objectives": list(result["final_objectives"]),
                "best_objectives": list(result["best_objectives"]),
                "final_energies": list(result["final_energies"]),
                "best_energies": list(result["best_energies"]),
                "rollout": _rollout_summary(result["rollout"]),
            }
        )
    fitted_rows = []
    for evaluation in phase3.fitted_controller_evaluations:
        action_ref = evaluation["clean_action_reference_rms"]
        fitted_rows.append(
            {
                "label": evaluation["label"],
                "source_artifact": evaluation["source_artifact"],
                "clean_cost": _cost_summary(evaluation["clean_cost"]),
                "clean_rollout": _rollout_summary(evaluation["clean_rollout"]),
                "under_riccati_epsilon_cost": _cost_summary(
                    evaluation["under_riccati_epsilon_cost"]
                ),
                "under_riccati_epsilon_rollout": _rollout_summary(
                    evaluation["under_riccati_epsilon_rollout"]
                ),
                "clean_cost_ratio_to_lqr_reference": (
                    evaluation["clean_cost"].total_without_disturbance_penalty / lqr_cost
                ),
                "under_epsilon_cost_ratio_to_lqr_reference": (
                    evaluation["under_riccati_epsilon_cost"].total_without_disturbance_penalty
                    / phase3.lqr_reference_under_riccati_cost.total_without_disturbance_penalty
                ),
                "clean_action_mismatch_rms_vs_lqr_reference": evaluation[
                    "clean_action_mismatch_rms_vs_lqr_reference"
                ],
                "clean_action_reference_rms": action_ref,
                "clean_action_mismatch_ratio": (
                    evaluation["clean_action_mismatch_rms_vs_lqr_reference"]
                    / max(action_ref, phase3.config.denominator_floor)
                ),
            }
        )
    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "supersedes_prior_disposition": (
            "83fc5b5 was previously demoted for reproducing Delta-v, but is now "
            "canonical for C&S information-structure fidelity."
        ),
        "phase0a_deterministic_reference": {
            "issue": DETERMINISTIC_PHASE0_ISSUE_ID,
            "status": "preserved_as_full_augmented_true_state_replay",
            "summary": deterministic,
        },
        "phase0b_output_feedback_reference": {
            "status": "canonical_for_cs2019_information_structure",
            "config": phase0b["config"].__dict__,
            "initial_condition": (
                "C&S-compatible repeated physical initial state in every "
                "delay-history block."
            ),
            "robust_controller_indexing": (
                "MATLAB-compatible: released C&S code applies M(:,:,k) after "
                "the backward loop, so k is the first Riccati slice."
            ),
            "observation": "H selects delayed x_(t-5) physical block; gain acts on x_hat_aug.",
            "lqr_rollout": _rollout_summary(phase0b["lqr_rollout"]),
            "lqr_cost": _cost_summary(phase0b["lqr_cost"]),
            "hinf_rollout": _rollout_summary(phase0b["hinf_rollout"]),
            "hinf_cost": _cost_summary(phase0b["hinf_cost"]),
        },
        "phase1_output_feedback_adversary_equivalence": {
            "deterministic_phase1_issue": DETERMINISTIC_PHASE1_ISSUE_ID,
            "budget": phase1.riccati_cost.disturbance_energy,
            "budget_l2": float(jnp.sqrt(jnp.asarray(phase1.riccati_cost.disturbance_energy))),
            "riccati_feedback": {
                "cost": _cost_summary(phase1.riccati_cost),
                "rollout": _rollout_summary(phase1.riccati_rollout),
            },
            "open_loop": phase1_rows,
        },
        "phase3_output_feedback_linear_gate": {
            "deterministic_phase3_issue": DETERMINISTIC_PHASE3_ISSUE_ID,
            "deterministic_certificate_issue": DETERMINISTIC_CERTIFICATE_ISSUE_ID,
            "status": "output_feedback_reference_gate_materialized",
            "interpretation": (
                "The linear gate now evaluates controllers through x_hat_aug. "
                "The full fitted-controller certificate from d01c35a remains the "
                "deterministic Phase 0A check; this lane additionally replays "
                "the fitted LQR controllers through the C&S output-feedback "
                "estimator dynamics when the Phase 3 artifact is available."
            ),
            "lqr_comparator_scope": (
                "simplified delayed Kalman baseline, not a full extLQG parity "
                "implementation with signal-dependent estimator noise terms"
            ),
            "lqr_reference": {
                "cost": _cost_summary(phase3.lqr_reference_cost),
                "rollout": _rollout_summary(phase3.lqr_reference_rollout),
            },
            "lqr_under_riccati_epsilon": {
                "cost": _cost_summary(phase3.lqr_reference_under_riccati_cost),
                "rollout": _rollout_summary(phase3.lqr_reference_under_riccati_epsilon),
            },
            "hinf_under_riccati_epsilon": {
                "cost": _cost_summary(phase3.hinf_reference_cost),
                "rollout": _rollout_summary(phase3.hinf_reference_rollout),
            },
            "hinf_vs_lqr_cost_ratio_under_riccati_epsilon": hinf_cost
            / phase3.lqr_reference_under_riccati_cost.total_without_disturbance_penalty,
            "hinf_vs_lqr_peak_velocity_delta_percent": 100.0
            * (
                phase3.hinf_reference_rollout.peak_forward_velocity
                - phase3.lqr_reference_under_riccati_epsilon.peak_forward_velocity
            )
            / phase3.lqr_reference_under_riccati_epsilon.peak_forward_velocity,
            "clean_hinf_vs_lqr_cost_ratio": hinf_cost / lqr_cost,
            "fitted_controller_replays": fitted_rows,
        },
        "phase4_implication": (
            "Phase 4 should preserve the output-feedback information structure: "
            "Feedbax/GraphSpec should feed delayed observations and let hidden "
            "state serve as an implicit estimator, rather than exposing true "
            "x_aug. The deterministic and output-feedback linear certificates "
            "should be treated as separate gates."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a tracked combined output-feedback lane note."""

    p0 = summary["phase0b_output_feedback_reference"]
    p1 = summary["phase1_output_feedback_adversary_equivalence"]
    p3 = summary["phase3_output_feedback_linear_gate"]
    rows = [
        "| PGD steps | best cost | ratio to Riccati | epsilon L2 distance |",
        "|---:|---:|---:|---:|",
    ]
    for row in p1["open_loop"]:
        rows.append(
            "| "
            f"{row['n_steps']} | "
            f"{row['best_cost']['total_without_disturbance_penalty']:.8g} | "
            f"{row['total_cost_ratio_to_riccati']:.8g} | "
            f"{row['epsilon_l2_distance_to_riccati']:.8g} |"
        )
    fitted_rows = [
        "| controller | clean cost ratio | under-epsilon cost ratio | action mismatch ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in p3["fitted_controller_replays"]:
        fitted_rows.append(
            "| "
            f"{row['label']} | "
            f"{row['clean_cost_ratio_to_lqr_reference']:.8g} | "
            f"{row['under_epsilon_cost_ratio_to_lqr_reference']:.8g} | "
            f"{row['clean_action_mismatch_ratio']:.8g} |"
        )
    return f"""# C&S Output-Feedback / Estimator-In-Loop Lane

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

This note adds the C&S information-structure lane while preserving the older
deterministic full augmented-state replay as Phase 0A. The canonical C&S
fidelity card is now Phase 0B:

```text
y_t = H x_aug,t
x_hat_aug,t+1 = estimator_update(x_hat_aug,t, u_t, y_t)
u_t = gain_t x_hat_aug,t
```

The gain still has full augmented-state support, but it acts on an estimated
augmented state, not directly on the true augmented state.

## Phase 0B Reference

- Observation: {p0["observation"]}
- Initial condition: {p0["initial_condition"]}
- Robust command indexing: {p0["robust_controller_indexing"]}
- LQR output-feedback cost: `{p0["lqr_cost"]["total_without_disturbance_penalty"]:.8g}`.
- LQR peak forward velocity: `{p0["lqr_rollout"]["peak_forward_velocity"]:.8g}`.
- H-infinity output-feedback cost: `{p0["hinf_cost"]["total_without_disturbance_penalty"]:.8g}`.
- H-infinity peak forward velocity: `{p0["hinf_rollout"]["peak_forward_velocity"]:.8g}`.
- H-infinity estimator RMS error: `{p0["hinf_rollout"]["estimation_error_rms"]:.8g}`.

## Phase 1 Output-Feedback Adversary Equivalence

Riccati realized disturbance budget:
`{p1["budget"]:.8g}` (`L2={p1["budget_l2"]:.8g}`).

Riccati feedback cost:
`{p1["riccati_feedback"]["cost"]["total_without_disturbance_penalty"]:.8g}`.

{"\n".join(rows)}

## Phase 3 Output-Feedback Linear Gate

The output-feedback reference gate evaluates action and cost through
`x_hat_aug`. The existing fitted-controller certificate remains the Phase 0A
deterministic gate; this lane defines the C&S estimator-in-loop target and
disturbance card that later Feedbax/GraphSpec work should preserve.

LQR comparator scope: {p3["lqr_comparator_scope"]}.

- LQR clean output-feedback cost:
  `{p3["lqr_reference"]["cost"]["total_without_disturbance_penalty"]:.8g}`.
- LQR under Riccati epsilon cost:
  `{p3["lqr_under_riccati_epsilon"]["cost"]["total_without_disturbance_penalty"]:.8g}`.
- H-infinity under Riccati epsilon cost:
  `{p3["hinf_under_riccati_epsilon"]["cost"]["total_without_disturbance_penalty"]:.8g}`.
- H-infinity / LQR cost ratio under Riccati epsilon:
  `{p3["hinf_vs_lqr_cost_ratio_under_riccati_epsilon"]:.8g}`.
- H-infinity vs LQR peak-velocity delta under Riccati epsilon:
  `{p3["hinf_vs_lqr_peak_velocity_delta_percent"]:.8g}%`.

Fitted deterministic Phase 3 controllers replayed through the output-feedback
estimator loop:

{"\n".join(fitted_rows)}

## Phase 4 Implication

{summary["phase4_implication"]}
"""


def _npz_arrays(
    phase0b: dict[str, Any],
    phase1: OutputFeedbackPhase1Result,
    phase3: OutputFeedbackPhase3Result,
) -> dict[str, np.ndarray]:
    arrays = {
        "phase0b_lqr_x": np.asarray(phase0b["lqr_rollout"].x),
        "phase0b_lqr_x_hat": np.asarray(phase0b["lqr_rollout"].x_hat),
        "phase0b_lqr_u": np.asarray(phase0b["lqr_rollout"].u),
        "phase0b_hinf_x": np.asarray(phase0b["hinf_rollout"].x),
        "phase0b_hinf_x_hat": np.asarray(phase0b["hinf_rollout"].x_hat),
        "phase0b_hinf_u": np.asarray(phase0b["hinf_rollout"].u),
        "phase1_riccati_x": np.asarray(phase1.riccati_rollout.x),
        "phase1_riccati_x_hat": np.asarray(phase1.riccati_rollout.x_hat),
        "phase1_riccati_u": np.asarray(phase1.riccati_rollout.u),
        "phase1_riccati_epsilon": np.asarray(phase1.riccati_rollout.epsilon),
        "phase3_lqr_x": np.asarray(phase3.lqr_reference_rollout.x),
        "phase3_lqr_x_hat": np.asarray(phase3.lqr_reference_rollout.x_hat),
        "phase3_lqr_under_eps_x": np.asarray(phase3.lqr_reference_under_riccati_epsilon.x),
        "phase3_hinf_x": np.asarray(phase3.hinf_reference_rollout.x),
    }
    for evaluation in phase3.fitted_controller_evaluations:
        key = f"phase3_{evaluation['label']}"
        arrays[f"{key}_clean_x"] = np.asarray(evaluation["clean_rollout"].x)
        arrays[f"{key}_clean_x_hat"] = np.asarray(evaluation["clean_rollout"].x_hat)
        arrays[f"{key}_clean_u"] = np.asarray(evaluation["clean_rollout"].u)
        arrays[f"{key}_under_eps_x"] = np.asarray(
            evaluation["under_riccati_epsilon_rollout"].x
        )
        arrays[f"{key}_under_eps_x_hat"] = np.asarray(
            evaluation["under_riccati_epsilon_rollout"].x_hat
        )
        arrays[f"{key}_under_eps_u"] = np.asarray(
            evaluation["under_riccati_epsilon_rollout"].u
        )
    for result in phase1.open_loop_results:
        key = f"phase1_open_loop_{result['n_steps']}"
        arrays[f"{key}_x"] = np.asarray(result["rollout"].x)
        arrays[f"{key}_x_hat"] = np.asarray(result["rollout"].x_hat)
        arrays[f"{key}_u"] = np.asarray(result["rollout"].u)
        arrays[f"{key}_epsilon"] = np.asarray(result["epsilon"])
    return arrays


def write_outputs(issue_id: str = ISSUE_ID) -> dict[str, Any]:
    """Write combined Phase 0B/1/3 output-feedback lane artifacts."""

    phase0b = analyze_phase0b_output_feedback()
    phase1 = analyze_phase1_output_feedback()
    phase3 = analyze_phase3_output_feedback()
    summary = result_summary(phase0b, phase1, phase3)
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "output_feedback_lane")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "C&S output-feedback / estimator-in-loop fidelity lane for phases 0B, 1, and 3. "
            "See `notes/output_feedback_lane.md`.\n",
            encoding="utf-8",
        )
    arrays = _npz_arrays(phase0b, phase1, phase3)
    npz_path = artifact_dir / "output_feedback_lane.npz"
    np.savez_compressed(npz_path, **arrays)
    summary["tracked_note"] = f"results/{issue_id}/notes/output_feedback_lane.md"
    summary["tracked_manifest"] = f"results/{issue_id}/notes/output_feedback_lane_manifest.json"
    summary["artifact_npz"] = f"_artifacts/{issue_id}/output_feedback_lane/{npz_path.name}"
    summary["artifact_npz_keys"] = sorted(arrays.keys())
    note_path = notes_dir / "output_feedback_lane.md"
    manifest_path = notes_dir / "output_feedback_lane_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "OutputFeedbackConfig",
    "OutputFeedbackPhase1Result",
    "OutputFeedbackPhase3Result",
    "OutputFeedbackRollout",
    "analyze_phase0b_output_feedback",
    "analyze_phase1_output_feedback",
    "analyze_phase3_output_feedback",
    "delayed_observation_matrix",
    "kalman_estimator_gains",
    "make_cs_output_feedback_initial_state",
    "measurement_covariance",
    "optimize_open_loop_output_feedback",
    "process_covariance",
    "render_markdown",
    "result_summary",
    "robust_estimator_covariances",
    "robust_estimator_fixed_adversary_policy",
    "robust_estimator_joint_matrices",
    "robust_output_feedback_gains",
    "rollout_with_kalman_estimator",
    "rollout_with_robust_estimator",
    "rollout_with_robust_estimator_policy",
    "write_outputs",
]
