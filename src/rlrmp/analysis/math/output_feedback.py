"""Legacy scope note: `analyze_phase0b_output_feedback`,
`analyze_phase1_output_feedback`, `analyze_phase3_output_feedback`,
`write_gamma_sweep_outputs`, and `write_outputs` are frozen writer/driver
surfaces. The output-feedback math core in this module remains LIVE library
code consumed by registered recipes.

C&S output-feedback / estimator-in-loop analytical lane.

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
import scipy.optimize as scipy_opt
from jaxtyping import Array, Float

from rlrmp.analysis.math.cs_game_card import (
    INIT_POS,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    TARGET_POS,
    CostBreakdown,
    GameCardReference,
    GammaReference,
    materialize_reference,
    reference_summary,
)
from rlrmp.analysis.math.hinf_riccati import (
    CostSchedule,
    PlantLinearization,
    RiccatiSolution,
    simulate_closed_loop,
)
from rlrmp.analysis.math.linear_round_trip import (
    LinearTrainingConfig,
    LinearTrainingResult,
    ensemble_initial_states,
    rollout_task_cost,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.analysis.math import require_jax_x64
from rlrmp.paths import REPO_ROOT, mkdir_p

ISSUE_ID = "83fc5b5"
UMBRELLA_ID = "43e8728"
DETERMINISTIC_PHASE0_ISSUE_ID = "cb98e58"
DETERMINISTIC_PHASE1_ISSUE_ID = "a7dad8a"
DETERMINISTIC_PHASE3_ISSUE_ID = "6f5c79e"
DETERMINISTIC_CERTIFICATE_ISSUE_ID = "d01c35a"
EXACT_OUTPUT_FEEDBACK_PHASE1_ISSUE_ID = "60d105d"
OUTPUT_FEEDBACK_PHASE3_TRAINING_ISSUE_ID = "4008843"
GAMMA_FEASIBILITY_SWEEP_ISSUE_ID = "97604a8"


@dataclass(frozen=True)
class OutputFeedbackConfig:
    """Numerical contract for the C&S estimator-in-loop lane."""

    n_phys: int = 8
    delay_steps: int = 5
    observed_physical_indices: tuple[int, ...] | None = None
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
    exact_audits: tuple[dict[str, Any], ...]
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
    output_feedback_trainings: tuple[LinearTrainingResult, ...]
    output_feedback_training_audits: tuple[dict[str, Any], ...]
    fitted_controller_evaluations: tuple[dict[str, Any], ...]


def delayed_observation_matrix(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "obs n"]:
    """Return C&S ``H`` selecting channels from the oldest delayed physical block."""

    h = config.delay_steps
    n_phys = config.n_phys
    expected_n = (h + 1) * n_phys
    if plant.n != expected_n:
        raise ValueError(f"Expected plant.n={expected_n}; got {plant.n}.")
    observed = config.observed_physical_indices
    physical_indices = tuple(range(n_phys)) if observed is None else tuple(observed)
    if not physical_indices:
        raise ValueError("At least one observed physical channel is required.")
    if min(physical_indices) < 0 or max(physical_indices) >= n_phys:
        raise ValueError(
            "observed_physical_indices must select channels inside one physical block; "
            f"got {physical_indices} for n_phys={n_phys}."
        )
    H = jnp.zeros((len(physical_indices), plant.n), dtype=jnp.float64)
    start = h * n_phys
    cols = jnp.asarray([start + idx for idx in physical_indices], dtype=jnp.int32)
    return H.at[jnp.arange(len(physical_indices)), cols].set(1.0)


def position_velocity_observation_config(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackConfig:
    """Return a config observing delayed position and velocity only.

    The resulting channel is the 4D Feedbax GRU pilot feedback domain:
    position and velocity from the oldest delayed physical block, excluding the
    remaining C&S force/filter physical states.
    """

    pos_lo, pos_hi = plant.pos_slice
    vel_lo, vel_hi = plant.vel_slice
    physical_indices = tuple(range(pos_lo, pos_hi)) + tuple(range(vel_lo, vel_hi))
    return OutputFeedbackConfig(
        n_phys=config.n_phys,
        delay_steps=config.delay_steps,
        observed_physical_indices=physical_indices,
        estimator_initial_covariance=config.estimator_initial_covariance,
        process_covariance_scale=config.process_covariance_scale,
        sensory_noise_scale=config.sensory_noise_scale,
        use_matlab_persistent_m_index=config.use_matlab_persistent_m_index,
        denominator_floor=config.denominator_floor,
    )


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
    if config.n_phys >= 8:
        top_force = Q_proc[4:6, 4:6]
        Q_proc = Q_proc.at[6:8, 6:8].set(top_force)
    return 0.5 * (Q_proc + Q_proc.T)


def measurement_covariance(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, "obs obs"]:
    """Return the C&S sensory covariance proxy."""

    Q_proc = process_covariance(plant, config)
    obs_dim = delayed_observation_matrix(plant, config).shape[0]
    return jnp.eye(obs_dim, dtype=jnp.float64) * Q_proc[4, 4] * config.sensory_noise_scale


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
    Sigma = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance, dtype=jnp.float64
    )
    covariances = [Sigma]
    for t in range(schedule.T):
        precision = jnp.linalg.inv(Sigma) + H.T @ H - inv_gamma2 * schedule.Q[t].astype(jnp.float64)
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
    Sigma_e = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance, dtype=jnp.float64
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
            K_est @ H @ Sigma_e_prev @ A.T + (A - B @ L) @ Sigma_x @ (A - B @ L).T + term + term.T
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
        correction = jnp.linalg.inv(
            eye_n - inv_gamma2 * estimator_covariances[t] @ solution.P[p_idx]
        )
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
        estimator_from_xhat = A - B @ gains[t] + A @ middle @ (inv_gamma2 * schedule.Q[t] - HH)
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


def kalman_estimator_joint_matrices(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[Float[Array, "T two_n two_n"], Float[Array, "two_n m_w"]]:
    """Return joint true/estimated-state dynamics for fixed Kalman feedback."""

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)
    H = delayed_observation_matrix(plant, config)
    estimator_gains = kalman_estimator_gains(plant, controller_gains, config)
    matrices = []
    for t in range(controller_gains.shape[0]):
        K_t = controller_gains[t].astype(jnp.float64)
        G_t = estimator_gains[t].astype(jnp.float64)
        top = jnp.concatenate([A, -B @ K_t], axis=1)
        bottom = jnp.concatenate([G_t @ H, A - B @ K_t - G_t @ H], axis=1)
        matrices.append(jnp.concatenate([top, bottom], axis=0))
    G = jnp.concatenate([Bw, jnp.zeros_like(Bw)], axis=0)
    return jnp.stack(matrices, axis=0), G


def _joint_cost_matrices(
    schedule: CostSchedule,
    controller_gains: Float[Array, "T m_u n"],
) -> tuple[Float[Array, "T two_n two_n"], Float[Array, "two_n two_n"]]:
    """Return quadratic cost matrices over ``z_t = [x_t, xhat_t]``."""

    n = schedule.Q.shape[-1]
    zeros = jnp.zeros((n, n), dtype=jnp.float64)
    stage = []
    for t in range(schedule.T):
        K_t = controller_gains[t].astype(jnp.float64)
        state_block = schedule.Q[t].astype(jnp.float64)
        control_block = K_t.T @ schedule.R[t].astype(jnp.float64) @ K_t
        stage.append(jnp.block([[state_block, zeros], [zeros, control_block]]))
    terminal = jnp.block([[schedule.Q_f.astype(jnp.float64), zeros], [zeros, zeros]])
    return jnp.stack(stage, axis=0), terminal


def _flattened_epsilon_quadratic(
    A_joint: Float[Array, "T z z"],
    G_joint: Float[Array, "z m_w"],
    stage_costs: Float[Array, "T z z"],
    terminal_cost: Float[Array, "z z"],
    z0: Float[Array, " z"],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``H, g, c`` for ``cost(eps) = eps^T H eps + 2 g^T eps + c``."""

    T = A_joint.shape[0]
    z_dim = A_joint.shape[1]
    m_w = G_joint.shape[1]
    flat_dim = T * m_w
    S = jnp.zeros((z_dim, flat_dim), dtype=jnp.float64)
    c = z0.astype(jnp.float64)
    H_acc = jnp.zeros((flat_dim, flat_dim), dtype=jnp.float64)
    g_acc = jnp.zeros((flat_dim,), dtype=jnp.float64)
    const = jnp.asarray(0.0, dtype=jnp.float64)
    for t in range(T):
        M_t = stage_costs[t].astype(jnp.float64)
        H_acc = H_acc + S.T @ M_t @ S
        g_acc = g_acc + S.T @ M_t @ c
        const = const + c @ M_t @ c
        S_next = A_joint[t] @ S
        S_next = S_next.at[:, t * m_w : (t + 1) * m_w].add(G_joint)
        c = A_joint[t] @ c
        S = S_next
    H_acc = H_acc + S.T @ terminal_cost @ S
    g_acc = g_acc + S.T @ terminal_cost @ c
    const = const + c @ terminal_cost @ c
    H_np = np.asarray(0.5 * (H_acc + H_acc.T), dtype=np.float64)
    return H_np, np.asarray(g_acc, dtype=np.float64), float(const)


def _gamma_penalized_quadratic_diagnostic(
    H: np.ndarray,
    g: np.ndarray,
    constant: float,
    *,
    gamma: float,
) -> dict[str, Any]:
    """Return exact diagnostics for ``max cost(eps) - gamma^2 ||eps||^2``.

    The flattened rollout cost is ``eps.T @ H @ eps + 2 g.T @ eps + c``.
    The gamma-penalized H-infinity maximization is finite only when
    ``gamma^2 I - H`` is positive definite.
    """

    H_sym = 0.5 * (H + H.T)
    eigvals = np.linalg.eigvalsh(H_sym)
    max_eig = float(eigvals[-1])
    gamma2 = float(gamma * gamma)
    margin = gamma2 - max_eig
    ratio = max_eig / gamma2 if gamma2 > 0.0 else float("inf")
    tol = max(1e-9, 1e-10 * max(1.0, abs(gamma2), abs(max_eig)))
    feasible = bool(margin > tol)
    result: dict[str, Any] = {
        "gamma": float(gamma),
        "gamma_squared": gamma2,
        "max_eigenvalue": max_eig,
        "max_eigenvalue_over_gamma_squared": ratio,
        "feasibility_margin": margin,
        "feasible": feasible,
        "unbounded": not feasible,
        "tolerance": tol,
    }
    if not feasible:
        return result

    lhs = gamma2 * np.eye(H_sym.shape[0], dtype=np.float64) - H_sym
    epsilon = np.linalg.solve(lhs, g)
    total_without_penalty = float(epsilon @ H_sym @ epsilon + 2.0 * g @ epsilon + constant)
    energy = float(epsilon @ epsilon)
    result.update(
        {
            "epsilon": jnp.asarray(epsilon.reshape((-1,)), dtype=jnp.float64),
            "epsilon_energy": energy,
            "epsilon_l2": float(np.sqrt(max(energy, 0.0))),
            "total_without_disturbance_penalty": total_without_penalty,
            "h_infinity_objective": total_without_penalty - gamma2 * energy,
        }
    )
    return result


def _maximize_quadratic_on_l2_ball(
    H: np.ndarray,
    g: np.ndarray,
    *,
    radius: float,
) -> tuple[np.ndarray, float, float, bool]:
    """Solve ``max eps^T H eps + 2 g^T eps`` subject to ``||eps|| <= radius``."""

    if radius <= 0.0:
        eps = np.zeros_like(g)
        return eps, 0.0, float(np.max(np.linalg.eigvalsh(H))), False
    eigvals, eigvecs = np.linalg.eigh(0.5 * (H + H.T))
    h = eigvecs.T @ g
    d_max = float(eigvals[-1])
    tol = 1e-12
    if np.linalg.norm(g) <= tol:
        eps = radius * eigvecs[:, -1]
        value = float(eps @ H @ eps + 2.0 * g @ eps)
        return eps, value, d_max, True

    def norm_at(lam: float) -> float:
        return float(np.linalg.norm(h / (lam - eigvals)))

    lo = d_max + max(1e-12, abs(d_max) * 1e-12)
    if norm_at(lo) < radius:
        # Hard case: the linear term has no component in the top eigenspace.
        non_top = np.abs(eigvals - d_max) > 1e-10
        y = np.zeros_like(h)
        y[non_top] = h[non_top] / (d_max - eigvals[non_top])
        remaining = max(0.0, radius * radius - float(y @ y))
        top = np.flatnonzero(~non_top)
        y[top[0]] = np.sqrt(remaining)
        eps = eigvecs @ y
        value = float(eps @ H @ eps + 2.0 * g @ eps)
        return eps, value, d_max, True
    hi = max(lo + 1.0, 2.0 * abs(d_max) + 1.0)
    while norm_at(hi) > radius:
        hi = 2.0 * hi + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) > radius:
            lo = mid
        else:
            hi = mid
    lam = hi
    eps = eigvecs @ (h / (lam - eigvals))
    value = float(eps @ H @ eps + 2.0 * g @ eps)
    return eps, value, lam, True


def exact_output_feedback_adversary_audit(
    *,
    label: str,
    plant: PlantLinearization,
    schedule: CostSchedule,
    controller_gains: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    budget: float,
    estimator_kind: str,
    solution: RiccatiSolution | None = None,
    penalty_gamma: float | None = None,
    eigenspectrum_modes: int = 0,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Maximize fixed-controller output-feedback task cost under an L2 epsilon budget."""

    if estimator_kind == "kalman":
        A_joint, G_joint = kalman_estimator_joint_matrices(plant, controller_gains, config)
    elif estimator_kind == "robust":
        if solution is None:
            raise ValueError("solution is required for robust estimator audit.")
        covs = robust_estimator_covariances(plant, schedule, solution.gamma, config)
        A_joint, G_joint = robust_estimator_joint_matrices(
            plant,
            schedule,
            solution,
            controller_gains,
            covs,
            config,
        )
    else:
        raise ValueError(f"Unknown estimator_kind={estimator_kind!r}.")
    stage_costs, terminal_cost = _joint_cost_matrices(schedule, controller_gains)
    z0 = jnp.concatenate([x0.astype(jnp.float64), x0.astype(jnp.float64)], axis=0)
    H_quad, g_quad, constant = _flattened_epsilon_quadratic(
        A_joint, G_joint, stage_costs, terminal_cost, z0
    )
    radius = float(np.sqrt(max(budget, 0.0)))
    eps_flat, variable_value, kkt_lambda, boundary_active = _maximize_quadratic_on_l2_ball(
        H_quad,
        g_quad,
        radius=radius,
    )
    epsilon = jnp.asarray(eps_flat.reshape((schedule.T, plant.m_w)), dtype=jnp.float64)
    if estimator_kind == "kalman":
        rollout = rollout_with_kalman_estimator(plant, controller_gains, x0, epsilon, config)
        gamma = None
    else:
        assert solution is not None
        rollout = rollout_with_robust_estimator(
            plant,
            schedule,
            solution,
            x0,
            epsilon,
            gains=controller_gains,
            config=config,
        )
        gamma = solution.gamma
    cost = output_feedback_cost(schedule, rollout, gamma=gamma)
    quadratic_total = constant + variable_value
    result = {
        "label": label,
        "estimator_kind": estimator_kind,
        "epsilon": epsilon,
        "rollout": rollout,
        "cost": cost,
        "budget": budget,
        "budget_l2": radius,
        "epsilon_energy": float(jnp.sum(epsilon**2)),
        "quadratic_total": quadratic_total,
        "rollout_total": cost.total_without_disturbance_penalty,
        "quadratic_rollout_abs_error": abs(
            quadratic_total - cost.total_without_disturbance_penalty
        ),
        "kkt_lambda": kkt_lambda,
        "boundary_active": boundary_active,
        "max_eigenvalue": float(np.max(np.linalg.eigvalsh(H_quad))),
    }
    if eigenspectrum_modes > 0:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (H_quad + H_quad.T))
        n_modes = min(int(eigenspectrum_modes), eigvecs.shape[1])
        order = np.arange(eigvecs.shape[1] - n_modes, eigvecs.shape[1])[::-1]
        result["eigenspectrum"] = {
            "eigenvalues": jnp.asarray(eigvals[order], dtype=jnp.float64),
            "epsilon_modes": jnp.asarray(
                eigvecs[:, order].T.reshape((n_modes, schedule.T, plant.m_w)),
                dtype=jnp.float64,
            ),
        }
    if penalty_gamma is not None:
        penalized = _gamma_penalized_quadratic_diagnostic(
            H_quad,
            g_quad,
            constant,
            gamma=float(penalty_gamma),
        )
        if penalized.get("feasible"):
            eps_star = jnp.asarray(
                np.asarray(penalized["epsilon"]).reshape((schedule.T, plant.m_w)),
                dtype=jnp.float64,
            )
            if estimator_kind == "kalman":
                penalized_rollout = rollout_with_kalman_estimator(
                    plant, controller_gains, x0, eps_star, config
                )
                rollout_gamma = None
            else:
                assert solution is not None
                penalized_rollout = rollout_with_robust_estimator(
                    plant,
                    schedule,
                    solution,
                    x0,
                    eps_star,
                    gains=controller_gains,
                    config=config,
                )
                rollout_gamma = solution.gamma
            penalized_cost = output_feedback_cost(
                schedule,
                penalized_rollout,
                gamma=rollout_gamma,
            )
            penalized["rollout"] = penalized_rollout
            penalized["rollout_cost"] = penalized_cost
            penalized["quadratic_rollout_abs_error"] = abs(
                penalized["total_without_disturbance_penalty"]
                - penalized_cost.total_without_disturbance_penalty
            )
            penalized["epsilon"] = eps_star
        result["gamma_penalized"] = penalized
    return result


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
) -> tuple[
    Float[Array, "T_plus_1 n"],
    Float[Array, "T_plus_1 n"],
    Float[Array, "T n_phys"],
    Float[Array, "T m_u"],
    Float[Array, "T_plus_1 n n"],
]:
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


def _kalman_estimator_rollout_arrays(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[
    Float[Array, "T_plus_1 n"],
    Float[Array, "T_plus_1 n"],
    Float[Array, "T n_phys"],
    Float[Array, "T m_u"],
    Float[Array, "T_plus_1 n n"],
]:
    """Return Kalman estimator-loop arrays without Python scalar conversion."""

    H = delayed_observation_matrix(plant, config)
    gains = kalman_estimator_gains(plant, K, config)
    Q_proc = process_covariance(plant, config)
    Sigma0 = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance, dtype=jnp.float64
    )

    def step(carry, inputs):
        x_t, xhat_t, Sigma_t = carry
        eps_t, K_t, G_t = inputs
        y_t = H @ x_t
        u_t = -K_t @ xhat_t
        xhat_next = plant.A @ xhat_t + plant.B @ u_t + G_t @ (y_t - H @ xhat_t)
        x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t
        Sigma_next = (plant.A - G_t @ H) @ Sigma_t @ plant.A.T + Q_proc
        Sigma_next = 0.5 * (Sigma_next + Sigma_next.T)
        return (x_next, xhat_next, Sigma_next), (x_next, xhat_next, y_t, u_t, Sigma_next)

    (_, _, _), (x_tail, xhat_tail, y, u, cov_tail) = jax.lax.scan(
        step,
        (x0.astype(jnp.float64), x0.astype(jnp.float64), Sigma0),
        (epsilon.astype(jnp.float64), K.astype(jnp.float64), gains),
    )
    x = jnp.concatenate([x0[None].astype(jnp.float64), x_tail], axis=0)
    x_hat = jnp.concatenate([x0[None].astype(jnp.float64), xhat_tail], axis=0)
    covs = jnp.concatenate([Sigma0[None], cov_tail], axis=0)
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
    Sigma = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance, dtype=jnp.float64
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


def output_feedback_clean_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Float[Array, ""]:
    """Mean weighted clean cost through the output-feedback estimator loop."""

    epsilon = jnp.zeros((schedule.T, plant.m_w), dtype=jnp.float64)

    def one_cost(x0: Float[Array, " n"]) -> Float[Array, ""]:
        x, _x_hat, _y, u, _covs = _kalman_estimator_rollout_arrays(
            plant,
            K,
            x0,
            epsilon,
            config,
        )
        return rollout_task_cost(schedule, x, u)

    costs = jax.vmap(one_cost)(states)
    return jnp.mean(costs * weights)


def train_output_feedback_lqr_controller(
    reference: GameCardReference,
    training_config: LinearTrainingConfig = LinearTrainingConfig(),
    *,
    quasi_newton_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=500),
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[LinearTrainingResult, LinearTrainingResult, LinearTrainingResult]:
    """Train clean LQR gains through the estimator-in-loop rollout from zero.

    With clean dynamics and ``xhat_0 = x_0``, the estimator innovation remains
    zero, so this objective is algebraically equivalent to the deterministic
    full-state clean objective. Keeping it explicit prevents later code from
    accidentally replaying old deterministic gains as the canonical
    output-feedback Phase 3 run.
    """

    plant = reference.plant
    schedule = reference.schedule
    states, weights = ensemble_initial_states(plant, training_config)
    K_ref = reference.lqr_solution.K

    def objective(gains: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        return output_feedback_clean_objective(plant, schedule, gains, states, weights, config)

    K = jnp.zeros_like(K_ref)
    optimizer = optax.adam(training_config.learning_rate)
    opt_state = optimizer.init(K)
    value_and_grad = jax.value_and_grad(objective)

    @jax.jit
    def run_adam(
        K: Float[Array, "T m_u n"],
        opt_state: optax.OptState,
    ) -> tuple[Float[Array, "T m_u n"], Float[Array, ""], Float[Array, ""]]:
        best_K = K
        best_value = objective(K)

        def step(carry, _):
            K, opt_state, best_K, best_value = carry
            _value, grads = value_and_grad(K)
            updates, opt_state = optimizer.update(grads, opt_state, K)
            K = optax.apply_updates(K, updates)
            value = objective(K)
            improved = value < best_value
            best_K = jnp.where(improved, K, best_K)
            best_value = jnp.minimum(best_value, value)
            return (K, opt_state, best_K, best_value), value

        (_K, _opt_state, best_K, best_value), values = jax.lax.scan(
            step,
            (K, opt_state, best_K, best_value),
            None,
            length=training_config.n_steps,
        )
        return best_K, best_value, values[-1]

    best_K, best_value, final_value = run_adam(K, opt_state)
    x0 = make_cs_output_feedback_initial_state(plant, config)
    reference_objective = float(objective(K_ref))
    zero_objective = float(objective(jnp.zeros_like(K_ref)))
    adam_rollout = simulate_closed_loop(plant, best_K, x0, target_pos=TARGET_POS)
    adam_result = LinearTrainingResult(
        label="of_adam_lqr_fit",
        config=training_config,
        K=best_K,
        best_objective=float(best_value),
        final_objective=float(final_value),
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        gain_relative_error=float(jnp.linalg.norm(best_K - K_ref) / jnp.linalg.norm(K_ref)),
        canonical_rollout=adam_rollout,
        optimizer_status="completed",
        n_iterations=training_config.n_steps,
        n_function_evaluations=training_config.n_steps,
    )

    states_qn, weights_qn = ensemble_initial_states(plant, quasi_newton_config)

    def qn_objective(gains: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        return output_feedback_clean_objective(
            plant,
            schedule,
            gains,
            states_qn,
            weights_qn,
            config,
        )

    shape = K_ref.shape

    @jax.jit
    def value_and_grad_flat(theta: Float[Array, " flat"]) -> tuple[Float[Array, ""], Array]:
        gains = theta.reshape(shape)
        value, grads = jax.value_and_grad(qn_objective)(gains)
        return value, grads.reshape(-1)

    def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grads = value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
        return float(value), np.asarray(grads, dtype=np.float64)

    def run_lbfgsb(theta0: np.ndarray, label: str) -> LinearTrainingResult:
        scipy_result = scipy_opt.minimize(
            scipy_value_and_grad,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": quasi_newton_config.n_steps,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )
        K_qn = jnp.asarray(scipy_result.x, dtype=jnp.float64).reshape(shape)
        qn_reference_objective = float(qn_objective(K_ref))
        qn_zero_objective = float(qn_objective(jnp.zeros_like(K_ref)))
        qn_final_objective = float(qn_objective(K_qn))
        return LinearTrainingResult(
            label=label,
            config=quasi_newton_config,
            K=K_qn,
            best_objective=float(min(scipy_result.fun, qn_final_objective)),
            final_objective=qn_final_objective,
            reference_objective=qn_reference_objective,
            zero_objective=qn_zero_objective,
            gain_relative_error=float(jnp.linalg.norm(K_qn - K_ref) / jnp.linalg.norm(K_ref)),
            canonical_rollout=simulate_closed_loop(plant, K_qn, x0, target_pos=TARGET_POS),
            optimizer_status=str(scipy_result.message),
            n_iterations=int(scipy_result.nit),
            n_function_evaluations=int(scipy_result.nfev),
        )

    qn_zero_result = run_lbfgsb(
        np.zeros(int(np.prod(shape)), dtype=np.float64),
        "of_lbfgsb_zero_lqr_fit",
    )
    qn_after_adam_result = run_lbfgsb(
        np.asarray(adam_result.K, dtype=np.float64).reshape(-1),
        "of_lbfgsb_after_of_adam_lqr_fit",
    )
    return adam_result, qn_zero_result, qn_after_adam_result


def output_feedback_lqr_bellman_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    P_next: Float[Array, "T n n"],
    K: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
) -> Float[Array, ""]:
    """Return the one-step finite-horizon LQR Bellman objective for gains.

    This diagnostic objective asks whether optimization can recover the
    analytical Riccati feedback when trained against the dynamic-programming
    local objective, rather than against a finite rollout ensemble. It uses the
    clean output-feedback condition ``xhat=x`` and paired full-rank state
    samples, not an independent joint covariance over ``[x, xhat]``.
    """

    states = states.astype(jnp.float64)
    weights = weights.astype(jnp.float64)

    def one_time_loss(inputs):
        K_t, Q_t, R_t, P_t_next = inputs
        u = -states @ K_t.T
        x_next = states @ (plant.A - plant.B @ K_t).T
        state_terms = jnp.einsum("bi,ij,bj->b", states, Q_t, states)
        control_terms = jnp.einsum("bi,ij,bj->b", u, R_t, u)
        next_terms = jnp.einsum("bi,ij,bj->b", x_next, P_t_next, x_next)
        return jnp.mean(weights * (state_terms + control_terms + next_terms))

    losses = jax.vmap(one_time_loss)((K, schedule.Q, schedule.R, P_next))
    return jnp.mean(losses)


def train_output_feedback_lqr_bellman_controller(
    reference: GameCardReference,
    config: LinearTrainingConfig = LinearTrainingConfig(n_steps=200),
) -> LinearTrainingResult:
    """Train an output-feedback LQR controller with the one-step Bellman objective."""

    plant = reference.plant
    schedule = reference.schedule
    K_ref = reference.lqr_solution.K
    shape = K_ref.shape
    P_next = reference.lqr_solution.P[1:].astype(jnp.float64)
    states, weights = ensemble_initial_states(plant, config)
    gain_scale = jnp.asarray(500.0, dtype=jnp.float64)

    def objective(z: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        gains = gain_scale * z
        return output_feedback_lqr_bellman_objective(
            plant, schedule, P_next, gains, states, weights
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
    x0 = make_cs_output_feedback_initial_state(plant)
    final_objective = float(
        output_feedback_lqr_bellman_objective(plant, schedule, P_next, K, states, weights)
    )
    reference_objective = float(
        output_feedback_lqr_bellman_objective(plant, schedule, P_next, K_ref, states, weights)
    )
    zero_objective = float(
        output_feedback_lqr_bellman_objective(
            plant, schedule, P_next, jnp.zeros_like(K_ref), states, weights
        )
    )
    return LinearTrainingResult(
        label="of_bellman_lbfgsb_lqr_fit",
        config=config,
        K=K,
        best_objective=float(min(scipy_result.fun, final_objective)),
        final_objective=final_objective,
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        gain_relative_error=float(jnp.linalg.norm(K - K_ref) / jnp.linalg.norm(K_ref)),
        canonical_rollout=simulate_closed_loop(plant, K, x0, target_pos=TARGET_POS),
        optimizer_status=str(scipy_result.message),
        n_iterations=int(scipy_result.nit),
        n_function_evaluations=int(scipy_result.nfev),
    )


def analyze_phase0b_output_feedback(
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """LEGACY (frozen 2026-07-03, issue 64d5f13).

    This writer/driver is not contract-native: it predates the feedbax recipe,
    bundle, and manifest contracts. It may not run without deliberate
    realignment. Do not copy it as a pattern for new analyses. The
    port-or-delete decision is deferred to the report-stage era (feedbax
    132f98c) / publication.

    Scoped legacy surface: `analyze_phase0b_output_feedback`. The math core in
    this module is LIVE library code consumed by registered recipes; this
    banner does not apply to the math core.
    """

    require_jax_x64("output-feedback phase0b analysis")
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
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
    starts = [
        _project_l2_ball(candidate.astype(jnp.float64), radius) for candidate in initial_candidates
    ]
    starts.extend(
        _project_l2_ball(jr.normal(key, shape, dtype=jnp.float64), radius) for key in keys
    )
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
        initial_objectives.append(
            float(_objective_value_output_feedback(plant, schedule, solution, x0, start, config))
        )
        best_eps, final_eps, best_value = run(start)
        best_epsilons.append(best_eps)
        final_objectives.append(
            float(
                _objective_value_output_feedback(plant, schedule, solution, x0, final_eps, config)
            )
        )
        best_objectives.append(float(best_value))
        final_energies.append(float(jnp.sum(final_eps**2)))
        best_energies.append(float(jnp.sum(best_eps**2)))
    best_idx = int(jnp.argmax(jnp.asarray(best_objectives)))
    best_epsilon = best_epsilons[best_idx]
    best_rollout = rollout_with_robust_estimator(
        plant, schedule, solution, x0, best_epsilon, config=config
    )
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
    """LEGACY (frozen 2026-07-03, issue 64d5f13).

    This writer/driver is not contract-native: it predates the feedbax recipe,
    bundle, and manifest contracts. It may not run without deliberate
    realignment. Do not copy it as a pattern for new analyses. The
    port-or-delete decision is deferred to the report-stage era (feedbax
    132f98c) / publication.

    Scoped legacy surface: `analyze_phase1_output_feedback`. The math core in
    this module is LIVE library code consumed by registered recipes; this
    banner does not apply to the math core.
    """

    require_jax_x64("output-feedback phase1 analysis")
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
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
    exact_audits = [
        exact_output_feedback_adversary_audit(
            label="analytical_lqr_kalman",
            plant=reference.plant,
            schedule=reference.schedule,
            controller_gains=reference.lqr_solution.K,
            x0=x0,
            budget=budget,
            estimator_kind="kalman",
            config=config,
        ),
        exact_output_feedback_adversary_audit(
            label="analytical_hinf_robust",
            plant=reference.plant,
            schedule=reference.schedule,
            controller_gains=gains,
            x0=x0,
            budget=budget,
            estimator_kind="robust",
            solution=gamma_ref.solution,
            config=config,
        ),
    ]
    round_trip_artifact = (
        REPO_ROOT
        / "_artifacts"
        / DETERMINISTIC_PHASE3_ISSUE_ID
        / "linear_round_trip"
        / "linear_round_trip.npz"
    )
    if round_trip_artifact.exists():
        with np.load(round_trip_artifact) as arrays:
            for label in ("adam_lqr_fit", "lbfgsb_after_adam_lqr_fit"):
                K = jnp.asarray(arrays[f"{label.replace('_fit', '')}_K"], dtype=jnp.float64)
                exact_audits.append(
                    exact_output_feedback_adversary_audit(
                        label=f"{label}_kalman",
                        plant=reference.plant,
                        schedule=reference.schedule,
                        controller_gains=K,
                        x0=x0,
                        budget=budget,
                        estimator_kind="kalman",
                        config=config,
                    )
                )
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
        exact_audits=tuple(exact_audits),
        open_loop_results=tuple(results),
    )


GAMMA_SWEEP_FACTORS = (
    1.001,
    1.005,
    1.01,
    1.02,
    1.05,
    1.1,
    1.2,
    1.25,
    1.3,
    1.32,
    1.33,
    1.34,
    1.345,
    1.35,
    1.4,
    1.45,
    1.5,
    2.0,
    3.0,
)


def _min_symmetric_eig(matrix: Float[Array, "n n"]) -> float:
    return float(jnp.min(jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T))))


def robust_output_feedback_feasibility_diagnostics(
    plant: PlantLinearization,
    schedule: CostSchedule,
    solution: RiccatiSolution,
    gains: Float[Array, "T m_u n"],
    estimator_covariances: Float[Array, "T_plus_1 n n"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Return matrix-condition diagnostics for the robust output-feedback lane."""

    H_obs = delayed_observation_matrix(plant, config)
    HH = H_obs.T @ H_obs
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)
    eye_n = jnp.eye(plant.n, dtype=jnp.float64)
    estimator_precision_min_eigs = []
    gain_correction_min_eigs = []
    for t in range(schedule.T):
        estimator_precision = (
            jnp.linalg.inv(estimator_covariances[t])
            + HH
            - inv_gamma2 * schedule.Q[t].astype(jnp.float64)
        )
        estimator_precision_min_eigs.append(_min_symmetric_eig(estimator_precision))
        p_idx = 0 if config.use_matlab_persistent_m_index else t
        correction_lhs = eye_n - inv_gamma2 * estimator_covariances[t] @ solution.P[p_idx]
        gain_correction_min_eigs.append(_min_symmetric_eig(correction_lhs))

    A_joint, G_joint = robust_estimator_joint_matrices(
        plant,
        schedule,
        solution,
        gains,
        estimator_covariances,
        config,
    )
    stage_costs, terminal_cost = _joint_cost_matrices(schedule, gains)
    P_next = terminal_cost
    gamma2 = solution.gamma * solution.gamma
    eye_w = jnp.eye(plant.m_w, dtype=jnp.float64)
    fixed_policy_lhs_min_eigs = []
    for t in range(schedule.T - 1, -1, -1):
        lhs = gamma2 * eye_w - G_joint.T @ P_next @ G_joint
        fixed_policy_lhs_min_eigs.append(_min_symmetric_eig(lhs))
        rhs = G_joint.T @ P_next @ A_joint[t]
        F_t = jnp.linalg.solve(lhs, rhs)
        P_t = stage_costs[t] + A_joint[t].T @ P_next @ A_joint[t] + rhs.T @ F_t
        P_next = 0.5 * (P_t + P_t.T)
    fixed_policy_lhs_min_eigs = list(reversed(fixed_policy_lhs_min_eigs))
    return {
        "estimator_precision_min_eigs": estimator_precision_min_eigs,
        "estimator_precision_min_eig": float(min(estimator_precision_min_eigs)),
        "gain_correction_min_eigs": gain_correction_min_eigs,
        "gain_correction_min_eig": float(min(gain_correction_min_eigs)),
        "fixed_policy_lhs_min_eigs": fixed_policy_lhs_min_eigs,
        "fixed_policy_lhs_min_eig": float(min(fixed_policy_lhs_min_eigs)),
    }


def _factor_key(factor: float) -> str:
    return str(factor).replace(".", "p").replace("-", "m")


def _penalized_summary(penalized: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in penalized.items()
        if key not in {"epsilon", "rollout", "rollout_cost"}
    }
    if "rollout" in penalized:
        summary["rollout"] = _rollout_summary(penalized["rollout"])
    if "rollout_cost" in penalized:
        summary["rollout_cost"] = _cost_summary(penalized["rollout_cost"])
    return summary


def _gamma_sweep_audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": audit["label"],
        "estimator_kind": audit["estimator_kind"],
        "cost": _cost_summary(audit["cost"]),
        "epsilon_energy": audit["epsilon_energy"],
        "budget": audit["budget"],
        "budget_l2": audit["budget_l2"],
        "quadratic_total": audit["quadratic_total"],
        "rollout_total": audit["rollout_total"],
        "quadratic_rollout_abs_error": audit["quadratic_rollout_abs_error"],
        "kkt_lambda": audit["kkt_lambda"],
        "boundary_active": audit["boundary_active"],
        "max_eigenvalue": audit["max_eigenvalue"],
        "gamma_penalized": _penalized_summary(audit["gamma_penalized"]),
        "rollout": _rollout_summary(audit["rollout"]),
    }


def analyze_output_feedback_gamma_sweep(
    gamma_factors: tuple[float, ...] = GAMMA_SWEEP_FACTORS,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Sweep gamma factors for robust output-feedback feasibility and audit metrics."""

    require_jax_x64("output-feedback gamma sweep")
    reference = materialize_reference(gamma_factors=tuple(gamma_factors))
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for gamma_ref in reference.gamma_references:
        factor = gamma_ref.factor
        key = f"gamma_{_factor_key(factor)}"
        row: dict[str, Any] = {
            "gamma_factor": factor,
            "gamma": gamma_ref.gamma,
            "gamma_star": reference.gamma_star,
            "riccati_solution_admissible": bool(gamma_ref.solution.admissible),
        }
        try:
            covs = robust_estimator_covariances(
                reference.plant,
                reference.schedule,
                gamma_ref.gamma,
                config,
            )
            gains = robust_output_feedback_gains(
                reference.plant,
                reference.schedule,
                gamma_ref.solution,
                covs,
                config,
            )
            feasibility = robust_output_feedback_feasibility_diagnostics(
                reference.plant,
                reference.schedule,
                gamma_ref.solution,
                gains,
                covs,
                config,
            )
            clean_rollout = rollout_with_robust_estimator(
                reference.plant,
                reference.schedule,
                gamma_ref.solution,
                x0,
                gains=gains,
                config=config,
            )
            clean_cost = output_feedback_cost(
                reference.schedule,
                clean_rollout,
                gamma=gamma_ref.gamma,
            )
            fixed_policy = robust_estimator_fixed_adversary_policy(
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
                fixed_policy,
                gains=gains,
                config=config,
            )
            riccati_cost = output_feedback_cost(
                reference.schedule,
                riccati_rollout,
                gamma=gamma_ref.gamma,
            )
            budget = riccati_cost.disturbance_energy
            robust_audit = exact_output_feedback_adversary_audit(
                label="analytical_hinf_robust",
                plant=reference.plant,
                schedule=reference.schedule,
                controller_gains=gains,
                x0=x0,
                budget=budget,
                estimator_kind="robust",
                solution=gamma_ref.solution,
                penalty_gamma=gamma_ref.gamma,
                config=config,
            )
            lqr_audit = exact_output_feedback_adversary_audit(
                label="analytical_lqr_kalman",
                plant=reference.plant,
                schedule=reference.schedule,
                controller_gains=reference.lqr_solution.K,
                x0=x0,
                budget=budget,
                estimator_kind="kalman",
                penalty_gamma=gamma_ref.gamma,
                config=config,
            )
            lqr_total = lqr_audit["cost"].total_without_disturbance_penalty
            robust_total = robust_audit["cost"].total_without_disturbance_penalty
            row.update(
                {
                    "status": "ok",
                    "feasibility": feasibility,
                    "clean_rollout": _rollout_summary(clean_rollout),
                    "clean_cost": _cost_summary(clean_cost),
                    "riccati_feedback": {
                        "cost": _cost_summary(riccati_cost),
                        "rollout": _rollout_summary(riccati_rollout),
                    },
                    "budget": budget,
                    "budget_l2": float(jnp.sqrt(jnp.asarray(budget))),
                    "exact_fixed_controller_audits": [
                        _gamma_sweep_audit_summary(lqr_audit),
                        _gamma_sweep_audit_summary(robust_audit),
                    ],
                    "robust_exact_cost_ratio_to_lqr": robust_total / lqr_total,
                    "robust_exact_cost_ratio_to_riccati_feedback": (
                        robust_total / riccati_cost.total_without_disturbance_penalty
                    ),
                    "robust_gamma_penalized_feasible": robust_audit["gamma_penalized"]["feasible"],
                    "robust_lambda_over_gamma_squared": robust_audit["gamma_penalized"][
                        "max_eigenvalue_over_gamma_squared"
                    ],
                    "lqr_lambda_over_gamma_squared": lqr_audit["gamma_penalized"][
                        "max_eigenvalue_over_gamma_squared"
                    ],
                }
            )
            arrays[f"{key}_clean_x"] = np.asarray(clean_rollout.x)
            arrays[f"{key}_clean_x_hat"] = np.asarray(clean_rollout.x_hat)
            arrays[f"{key}_clean_u"] = np.asarray(clean_rollout.u)
            arrays[f"{key}_riccati_x"] = np.asarray(riccati_rollout.x)
            arrays[f"{key}_riccati_x_hat"] = np.asarray(riccati_rollout.x_hat)
            arrays[f"{key}_riccati_u"] = np.asarray(riccati_rollout.u)
            arrays[f"{key}_riccati_epsilon"] = np.asarray(riccati_rollout.epsilon)
            for audit in (lqr_audit, robust_audit):
                audit_key = f"{key}_{audit['label']}"
                arrays[f"{audit_key}_l2_epsilon"] = np.asarray(audit["epsilon"])
                if audit["gamma_penalized"].get("feasible"):
                    arrays[f"{audit_key}_penalized_epsilon"] = np.asarray(
                        audit["gamma_penalized"]["epsilon"]
                    )
        except Exception as exc:  # noqa: BLE001 - sweep rows should record failures.
            row.update({"status": "failed", "error": repr(exc)})
        rows.append(row)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    eligible = [
        row
        for row in ok_rows
        if row["robust_gamma_penalized_feasible"]
        and row["feasibility"]["estimator_precision_min_eig"] > 0.0
        and row["feasibility"]["gain_correction_min_eig"] > 0.0
        and row["feasibility"]["fixed_policy_lhs_min_eig"] > 0.0
    ]
    recommendation = None
    if eligible:
        chosen = min(eligible, key=lambda row: row["gamma_factor"])
        recommendation = {
            "chosen_gamma_factor": chosen["gamma_factor"],
            "chosen_gamma": chosen["gamma"],
            "reason": (
                "Smallest swept gamma ratio with positive estimator, gain-correction, "
                "fixed-policy, and gamma-penalized exact-audit margins."
            ),
        }
    return {
        "issue": GAMMA_FEASIBILITY_SWEEP_ISSUE_ID,
        "output_feedback_issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "gamma_factors": list(gamma_factors),
        "gamma_star": reference.gamma_star,
        "selected_output_feedback_gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "config": config.__dict__,
        "rows": rows,
        "recommendation": recommendation,
        "arrays": arrays,
    }


def gamma_sweep_summary(sweep: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable gamma-sweep summary without bulk arrays."""

    return {key: value for key, value in sweep.items() if key != "arrays"}


def analyze_phase3_output_feedback(
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
    training_config: LinearTrainingConfig = LinearTrainingConfig(),
    quasi_newton_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=500),
) -> OutputFeedbackPhase3Result:
    """LEGACY (frozen 2026-07-03, issue 64d5f13).

    This writer/driver is not contract-native: it predates the feedbax recipe,
    bundle, and manifest contracts. It may not run without deliberate
    realignment. Do not copy it as a pattern for new analyses. The
    port-or-delete decision is deferred to the report-stage era (feedbax
    132f98c) / publication.

    Scoped legacy surface: `analyze_phase3_output_feedback`. The math core in
    this module is LIVE library code consumed by registered recipes; this
    banner does not apply to the math core.
    """

    require_jax_x64("output-feedback phase3 analysis")
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
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
    output_feedback_trainings = train_output_feedback_lqr_controller(
        reference,
        training_config,
        quasi_newton_config=quasi_newton_config,
        config=config,
    )
    bellman_training = train_output_feedback_lqr_bellman_controller(
        reference,
        LinearTrainingConfig(n_steps=200),
    )
    output_feedback_trainings = (*output_feedback_trainings, bellman_training)
    output_feedback_training_audits = tuple(
        exact_output_feedback_adversary_audit(
            label=f"{training.label}_exact_kalman",
            plant=reference.plant,
            schedule=reference.schedule,
            controller_gains=training.K,
            x0=x0,
            budget=float(jnp.sum(riccati_epsilon**2)),
            estimator_kind="kalman",
            config=config,
        )
        for training in output_feedback_trainings
    )
    fitted_evaluations = []
    round_trip_artifact = (
        REPO_ROOT
        / "_artifacts"
        / DETERMINISTIC_PHASE3_ISSUE_ID
        / "linear_round_trip"
        / "linear_round_trip.npz"
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
        hinf_reference_cost=output_feedback_cost(
            reference.schedule, hinf_rollout, gamma=gamma_ref.gamma
        ),
        output_feedback_trainings=output_feedback_trainings,
        output_feedback_training_audits=output_feedback_training_audits,
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


def _training_summary(training: LinearTrainingResult, schedule: CostSchedule) -> dict[str, Any]:
    canonical_clean_cost = float(
        rollout_task_cost(schedule, training.canonical_rollout.x, training.canonical_rollout.u)
    )
    return {
        "label": training.label,
        "optimizer_status": training.optimizer_status,
        "n_iterations": training.n_iterations,
        "n_function_evaluations": training.n_function_evaluations,
        "best_objective": training.best_objective,
        "final_objective": training.final_objective,
        "reference_objective": training.reference_objective,
        "zero_objective": training.zero_objective,
        "objective_ratio_to_reference": training.best_objective / training.reference_objective,
        "gain_relative_error": training.gain_relative_error,
        "canonical_clean_cost": canonical_clean_cost,
        "canonical_peak_forward_velocity": training.canonical_rollout.peak_forward_velocity,
        "canonical_terminal_position_error_m": training.canonical_rollout.terminal_position_error,
    }


def result_summary(
    phase0b: dict[str, Any],
    phase1: OutputFeedbackPhase1Result,
    phase3: OutputFeedbackPhase3Result,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return JSON-serializable combined output-feedback lane summary."""

    deterministic = reference_summary(phase0b["reference"])
    lqr_cost = phase3.lqr_reference_cost.total_without_disturbance_penalty
    hinf_cost = phase3.hinf_reference_cost.total_without_disturbance_penalty
    phase1_rows = []
    ric_total = phase1.riccati_cost.total_without_disturbance_penalty
    exact_rows = []
    for audit in phase1.exact_audits:
        total = audit["cost"].total_without_disturbance_penalty
        exact_rows.append(
            {
                "label": audit["label"],
                "estimator_kind": audit["estimator_kind"],
                "cost": _cost_summary(audit["cost"]),
                "total_cost_ratio_to_riccati": total / ric_total,
                "total_cost_ratio_to_lqr_exact": None,
                "epsilon_energy": audit["epsilon_energy"],
                "budget": audit["budget"],
                "budget_l2": audit["budget_l2"],
                "quadratic_total": audit["quadratic_total"],
                "rollout_total": audit["rollout_total"],
                "quadratic_rollout_abs_error": audit["quadratic_rollout_abs_error"],
                "kkt_lambda": audit["kkt_lambda"],
                "boundary_active": audit["boundary_active"],
                "max_eigenvalue": audit["max_eigenvalue"],
                "rollout": _rollout_summary(audit["rollout"]),
            }
        )
    lqr_exact_total = next(
        (
            row["cost"]["total_without_disturbance_penalty"]
            for row in exact_rows
            if row["label"] == "analytical_lqr_kalman"
        ),
        None,
    )
    if lqr_exact_total is not None:
        for row in exact_rows:
            row["total_cost_ratio_to_lqr_exact"] = (
                row["cost"]["total_without_disturbance_penalty"] / lqr_exact_total
            )
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
    training_rows = []
    for training in phase3.output_feedback_trainings:
        training_rows.append(_training_summary(training, phase3.reference.schedule))
    training_audit_rows = []
    for audit in phase3.output_feedback_training_audits:
        total = audit["cost"].total_without_disturbance_penalty
        training_audit_rows.append(
            {
                "label": audit["label"],
                "estimator_kind": audit["estimator_kind"],
                "cost": _cost_summary(audit["cost"]),
                "total_cost_ratio_to_lqr_exact": (
                    total
                    / next(
                        row["cost"]["total_without_disturbance_penalty"]
                        for row in exact_rows
                        if row["label"] == "analytical_lqr_kalman"
                    )
                ),
                "total_cost_ratio_to_hinf_exact": (
                    total
                    / next(
                        row["cost"]["total_without_disturbance_penalty"]
                        for row in exact_rows
                        if row["label"] == "analytical_hinf_robust"
                    )
                ),
                "epsilon_energy": audit["epsilon_energy"],
                "quadratic_rollout_abs_error": audit["quadratic_rollout_abs_error"],
                "rollout": _rollout_summary(audit["rollout"]),
            }
        )
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
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="output_feedback_lane",
        ),
        "follow_up_issues": {
            "exact_output_feedback_phase1": EXACT_OUTPUT_FEEDBACK_PHASE1_ISSUE_ID,
            "output_feedback_phase3_training": OUTPUT_FEEDBACK_PHASE3_TRAINING_ISSUE_ID,
        },
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
            "gamma_factor": phase0b["gamma_ref"].factor,
            "config": phase0b["config"].__dict__,
            "initial_condition": (
                "C&S-compatible repeated physical initial state in every delay-history block."
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
            "gamma_factor": phase1.gamma_ref.factor,
            "budget": phase1.riccati_cost.disturbance_energy,
            "budget_l2": float(jnp.sqrt(jnp.asarray(phase1.riccati_cost.disturbance_energy))),
            "riccati_feedback": {
                "cost": _cost_summary(phase1.riccati_cost),
                "rollout": _rollout_summary(phase1.riccati_rollout),
            },
            "exact_fixed_controller_audits": exact_rows,
            "open_loop": phase1_rows,
            "open_loop_interpretation": (
                "The PGD rows include the Riccati epsilon as an initial candidate. "
                "They are retained as diagnostics only; exact_fixed_controller_audits "
                "is the strengthened evidence-bearing Phase 1 audit."
            ),
        },
        "phase3_output_feedback_linear_gate": {
            "deterministic_phase3_issue": DETERMINISTIC_PHASE3_ISSUE_ID,
            "deterministic_certificate_issue": DETERMINISTIC_CERTIFICATE_ISSUE_ID,
            "gamma_factor": phase3.gamma_ref.factor,
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
            "hinf_under_riccati_epsilon_vs_clean_lqr_cost_ratio": hinf_cost / lqr_cost,
            "output_feedback_training_note": (
                "Clean estimator-in-loop training starts from zero. Because xhat_0=x_0 "
                "and clean innovations remain zero, the clean objective is algebraically "
                "equivalent to full-state clean training; the estimator-loop distinction "
                "is tested by the exact disturbance audits. The Bellman row is a "
                "diagnostic one-step LQR dynamic-programming objective using the "
                "analytical P[t+1] value matrices; it tests recoverability when the "
                "objective identifies the Riccati law, not robust/H-infinity training."
            ),
            "output_feedback_trainings": training_rows,
            "output_feedback_training_exact_audits": training_audit_rows,
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
    exact_rows = [
        "| controller | estimator | exact cost | ratio to LQR exact | ratio to Riccati feedback | quadratic error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in p1["exact_fixed_controller_audits"]:
        exact_rows.append(
            "| "
            f"{row['label']} | "
            f"{row['estimator_kind']} | "
            f"{row['cost']['total_without_disturbance_penalty']:.8g} | "
            f"{row['total_cost_ratio_to_lqr_exact']:.8g} | "
            f"{row['total_cost_ratio_to_riccati']:.8g} | "
            f"{row['quadratic_rollout_abs_error']:.3g} |"
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
    training_rows = [
        "| optimizer | objective ratio | gain rel err | clean cost | exact cost ratio to LQR | exact cost ratio to H-inf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    audits_by_prefix = {
        row["label"].replace("_exact_kalman", ""): row
        for row in p3["output_feedback_training_exact_audits"]
    }
    for row in p3["output_feedback_trainings"]:
        audit = audits_by_prefix[row["label"]]
        training_rows.append(
            "| "
            f"{row['label']} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['gain_relative_error']:.8g} | "
            f"{row['canonical_clean_cost']:.8g} | "
            f"{audit['total_cost_ratio_to_lqr_exact']:.8g} | "
            f"{audit['total_cost_ratio_to_hinf_exact']:.8g} |"
        )
    return f"""# C&S Output-Feedback / Estimator-In-Loop Lane

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

This note adds the C&S information-structure lane while preserving the older
deterministic full augmented-state replay as Phase 0A. The canonical C&S
deterministic estimator-in-loop card is now Phase 0B:

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

```text
y_t = H x_aug,t
x_hat_aug,t+1 = estimator_update(x_hat_aug,t, u_t, y_t)
u_t = gain_t x_hat_aug,t
```

The gain still has full augmented-state support, but it acts on an estimated
augmented state, not directly on the true augmented state.

## Phase 0B Reference

- Gamma factor: `{p0["gamma_factor"]:.6g}`.
- Observation: {p0["observation"]}
- Initial condition: {p0["initial_condition"]}
- Robust command indexing: {p0["robust_controller_indexing"]}
- LQR output-feedback cost: `{p0["lqr_cost"]["total_without_disturbance_penalty"]:.8g}`.
- LQR peak forward velocity: `{p0["lqr_rollout"]["peak_forward_velocity"]:.8g}`.
- H-infinity output-feedback cost: `{p0["hinf_cost"]["total_without_disturbance_penalty"]:.8g}`.
- H-infinity peak forward velocity: `{p0["hinf_rollout"]["peak_forward_velocity"]:.8g}`.
- H-infinity estimator RMS error: `{p0["hinf_rollout"]["estimation_error_rms"]:.8g}`.

## Phase 1 Output-Feedback Adversary Equivalence

Gamma factor: `{p1["gamma_factor"]:.6g}`.

Riccati realized disturbance budget:
`{p1["budget"]:.8g}` (`L2={p1["budget_l2"]:.8g}`).

Riccati feedback cost:
`{p1["riccati_feedback"]["cost"]["total_without_disturbance_penalty"]:.8g}`.

Exact fixed-controller L2-budget audits:

{"\n".join(exact_rows)}

The projected-gradient rows below are retained as diagnostics. They include the
Riccati epsilon as an initial candidate, so they should not be read as an
independent proof that unseeded open-loop ascent recovered the same sequence.

{"\n".join(rows)}

## Phase 3 Output-Feedback Linear Gate

Gamma factor for disturbance/audit comparisons: `{p3["gamma_factor"]:.6g}`.

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

Canonical output-feedback retraining starts from zero, not from the old
deterministic fit:

{p3["output_feedback_training_note"]}

{"\n".join(training_rows)}

Fitted deterministic Phase 3 controllers replayed through the output-feedback
estimator loop:

{"\n".join(fitted_rows)}

## Phase 4 Implication

{summary["phase4_implication"]}
"""


def render_gamma_sweep_markdown(summary: dict[str, Any]) -> str:
    """Render the gamma-penalized output-feedback robust sweep note."""

    rows = [
        "| gamma factor | status | robust lambda/gamma^2 | robust penalized feasible | "
        "min estimator eig | min gain-correction eig | min fixed-policy eig | "
        "robust exact/LQR exact | H-inf peak velocity |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        if row["status"] != "ok":
            rows.append(
                "| "
                f"{row['gamma_factor']:.6g} | failed | nan | false | nan | nan | nan | nan | nan |"
            )
            continue
        feasibility = row["feasibility"]
        rows.append(
            "| "
            f"{row['gamma_factor']:.6g} | ok | "
            f"{row['robust_lambda_over_gamma_squared']:.8g} | "
            f"{str(row['robust_gamma_penalized_feasible']).lower()} | "
            f"{feasibility['estimator_precision_min_eig']:.8g} | "
            f"{feasibility['gain_correction_min_eig']:.8g} | "
            f"{feasibility['fixed_policy_lhs_min_eig']:.8g} | "
            f"{row['robust_exact_cost_ratio_to_lqr']:.8g} | "
            f"{row['clean_rollout']['peak_forward_velocity']:.8g} |"
        )

    recommendation = summary["recommendation"]
    selected_factor = summary["selected_output_feedback_gamma_factor"]
    selected_row = next(
        (
            row
            for row in summary["rows"]
            if row["status"] == "ok" and row["gamma_factor"] == selected_factor
        ),
        None,
    )
    if recommendation is None:
        recommendation_text = (
            "No swept gamma ratio passed all positive-margin feasibility checks. "
            "Do not promote this robust output-feedback lane as a formal H-infinity "
            "same-game target without either sweeping larger gamma ratios or changing "
            "the formal target."
        )
    else:
        chosen_row = next(
            row
            for row in summary["rows"]
            if row["status"] == "ok"
            and row["gamma_factor"] == recommendation["chosen_gamma_factor"]
        )
        chosen_ratio = chosen_row["robust_lambda_over_gamma_squared"]
        next_safer = next(
            (
                row
                for row in summary["rows"]
                if row["status"] == "ok"
                and row["gamma_factor"] > recommendation["chosen_gamma_factor"]
                and row["robust_gamma_penalized_feasible"]
            ),
            None,
        )
        safety_sentence = ""
        if next_safer is not None and chosen_ratio > 0.99:
            safety_sentence = (
                f" The margin at `{recommendation['chosen_gamma_factor']:.6g}` is thin "
                f"(`lambda/gamma^2={chosen_ratio:.8g}`), so `{next_safer['gamma_factor']:.6g}` "
                "is the nearest more conservative swept fallback if we want numerical slack."
            )
        selected_sentence = ""
        if selected_row is not None:
            selected_sentence = (
                f" The current working output-feedback gamma factor is "
                f"`{selected_factor:.6g}` because it keeps additional slack "
                f"(`lambda/gamma^2="
                f"{selected_row['robust_lambda_over_gamma_squared']:.8g}`)."
            )
        recommendation_text = (
            f"The smallest swept passing gamma factor is "
            f"`{recommendation['chosen_gamma_factor']:.6g}` "
            f"(`gamma={recommendation['chosen_gamma']:.8g}`). This identifies the "
            "boundary of feasibility for this sweep, not the mandatory working "
            "default. The working default for later output-feedback Phase 1/3 "
            f"diagnostics is selected separately as `{selected_factor:.6g}` from "
            f"this sweep.{selected_sentence}{safety_sentence}"
        )

    return f"""# Gamma-Penalized Output-Feedback Robust Feasibility Sweep

Issue: `{summary["issue"]}`. Output-feedback lane: `{summary["output_feedback_issue"]}`.
Umbrella: `{summary["umbrella"]}`.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

This note extends the exact output-feedback Phase 1 audit from an L2-budget
trust-region check to a gamma-penalized H-infinity feasibility check. For each
gamma factor, the robust output-feedback controller is built in the C&S
estimator-in-loop lane, the coupled `[x, xhat]` closed-loop quadratic is
flattened over the whole epsilon trajectory, and the condition
`gamma^2 I - Q_epsilon` is checked.

`robust lambda/gamma^2` below is `lambda_max(Q_epsilon) / gamma^2` for the
analytical H-infinity robust controller. Values below 1 are finite for the
penalized maximization; values at or above 1 indicate an unbounded penalized
open-loop epsilon objective for that frozen controller.

Gamma star: `{summary["gamma_star"]:.8g}`.

Selected working output-feedback gamma factor:
`{summary["selected_output_feedback_gamma_factor"]:.6g}`.

{"\n".join(rows)}

## Recommendation

{recommendation_text}

## Interpretation

The sweep is a Phase 1/0B certificate step. It does not train Phase 3
controllers and does not implement robust Bellman. Its purpose is to identify
which robust analytical output-feedback target is coherent enough for later
Phase 3 training and certification.
"""


def write_gamma_sweep_outputs(
    issue_id: str = GAMMA_FEASIBILITY_SWEEP_ISSUE_ID,
    gamma_factors: tuple[float, ...] = GAMMA_SWEEP_FACTORS,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """LEGACY (frozen 2026-07-03, issue 64d5f13).

    This writer/driver is not contract-native: it predates the feedbax recipe,
    bundle, and manifest contracts. It may not run without deliberate
    realignment. Do not copy it as a pattern for new analyses. The
    port-or-delete decision is deferred to the report-stage era (feedbax
    132f98c) / publication.

    Scoped legacy surface: `write_gamma_sweep_outputs`. The math core in this
    module is LIVE library code consumed by registered recipes; this banner
    does not apply to the math core.
    """

    sweep = analyze_output_feedback_gamma_sweep(gamma_factors=gamma_factors)
    summary = gamma_sweep_summary(sweep)
    summary["rerun_metadata"] = build_rerun_metadata(
        discretization=discretization,
        lane=lane,
        materializer="output_feedback_gamma_sweep",
    )
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "output_feedback_gamma_sweep")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Gamma-penalized feasibility sweep for the C&S output-feedback robust lane. "
            "See `notes/output_feedback_gamma_sweep.md`.\n",
            encoding="utf-8",
        )
    npz_path = artifact_dir / "output_feedback_gamma_sweep.npz"
    np.savez_compressed(npz_path, **sweep["arrays"])
    summary["tracked_note"] = f"results/{issue_id}/notes/output_feedback_gamma_sweep.md"
    summary["tracked_manifest"] = (
        f"results/{issue_id}/notes/output_feedback_gamma_sweep_manifest.json"
    )
    summary["artifact_npz"] = f"_artifacts/{issue_id}/output_feedback_gamma_sweep/{npz_path.name}"
    summary["artifact_npz_keys"] = sorted(sweep["arrays"].keys())
    note_path = notes_dir / "output_feedback_gamma_sweep.md"
    manifest_path = notes_dir / "output_feedback_gamma_sweep_manifest.json"
    note_path.write_text(render_gamma_sweep_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


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
        arrays[f"{key}_under_eps_x"] = np.asarray(evaluation["under_riccati_epsilon_rollout"].x)
        arrays[f"{key}_under_eps_x_hat"] = np.asarray(
            evaluation["under_riccati_epsilon_rollout"].x_hat
        )
        arrays[f"{key}_under_eps_u"] = np.asarray(evaluation["under_riccati_epsilon_rollout"].u)
    for training in phase3.output_feedback_trainings:
        key = f"phase3_{training.label}"
        arrays[f"{key}_K"] = np.asarray(training.K)
        arrays[f"{key}_clean_x"] = np.asarray(training.canonical_rollout.x)
        arrays[f"{key}_clean_u"] = np.asarray(training.canonical_rollout.u)
    for audit in phase3.output_feedback_training_audits:
        key = f"phase3_{audit['label']}"
        arrays[f"{key}_exact_x"] = np.asarray(audit["rollout"].x)
        arrays[f"{key}_exact_x_hat"] = np.asarray(audit["rollout"].x_hat)
        arrays[f"{key}_exact_u"] = np.asarray(audit["rollout"].u)
        arrays[f"{key}_exact_epsilon"] = np.asarray(audit["epsilon"])
    for result in phase1.open_loop_results:
        key = f"phase1_open_loop_{result['n_steps']}"
        arrays[f"{key}_x"] = np.asarray(result["rollout"].x)
        arrays[f"{key}_x_hat"] = np.asarray(result["rollout"].x_hat)
        arrays[f"{key}_u"] = np.asarray(result["rollout"].u)
        arrays[f"{key}_epsilon"] = np.asarray(result["epsilon"])
    for audit in phase1.exact_audits:
        key = f"phase1_exact_{audit['label']}"
        arrays[f"{key}_x"] = np.asarray(audit["rollout"].x)
        arrays[f"{key}_x_hat"] = np.asarray(audit["rollout"].x_hat)
        arrays[f"{key}_u"] = np.asarray(audit["rollout"].u)
        arrays[f"{key}_epsilon"] = np.asarray(audit["epsilon"])
    return arrays


def write_outputs(
    issue_id: str = ISSUE_ID,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """LEGACY (frozen 2026-07-03, issue 64d5f13).

    This writer/driver is not contract-native: it predates the feedbax recipe,
    bundle, and manifest contracts. It may not run without deliberate
    realignment. Do not copy it as a pattern for new analyses. The
    port-or-delete decision is deferred to the report-stage era (feedbax
    132f98c) / publication.

    Scoped legacy surface: `write_outputs`. The math core in this module is
    LIVE library code consumed by registered recipes; this banner does not
    apply to the math core.
    """

    require_jax_x64("output-feedback materialization")
    phase0b = analyze_phase0b_output_feedback()
    phase1 = analyze_phase1_output_feedback()
    phase3 = analyze_phase3_output_feedback()
    summary = result_summary(
        phase0b,
        phase1,
        phase3,
        discretization=discretization,
        lane=lane,
    )
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "output_feedback_lane")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "C&S output-feedback / estimator-in-loop deterministic lane for phases 0B, 1, and 3. "
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
    "analyze_output_feedback_gamma_sweep",
    "delayed_observation_matrix",
    "gamma_sweep_summary",
    "kalman_estimator_gains",
    "make_cs_output_feedback_initial_state",
    "measurement_covariance",
    "optimize_open_loop_output_feedback",
    "process_covariance",
    "render_markdown",
    "result_summary",
    "render_gamma_sweep_markdown",
    "robust_estimator_covariances",
    "robust_estimator_fixed_adversary_policy",
    "robust_estimator_joint_matrices",
    "robust_output_feedback_feasibility_diagnostics",
    "robust_output_feedback_gains",
    "rollout_with_kalman_estimator",
    "rollout_with_robust_estimator",
    "rollout_with_robust_estimator_policy",
    "write_gamma_sweep_outputs",
    "write_outputs",
]
