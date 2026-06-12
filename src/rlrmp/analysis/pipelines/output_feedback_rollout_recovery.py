"""Output-feedback rollout recovery diagnostics for the Phase 3 linear bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize as scipy_opt
from jaxtyping import Array, Float

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.hinf_riccati import CostSchedule, PlantLinearization
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig, rollout_task_cost
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    OutputFeedbackRollout,
    _kalman_estimator_rollout_arrays,
    _rollout_summary,
    delayed_observation_matrix,
    exact_output_feedback_adversary_audit,
    kalman_estimator_gains,
    make_cs_output_feedback_initial_state,
    output_feedback_clean_objective,
    output_feedback_cost,
    output_feedback_lqr_bellman_objective,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator_policy,
    train_output_feedback_lqr_bellman_controller,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

ISSUE_ID = "7a459bb"
UMBRELLA_ID = "43e8728"
OUTPUT_FEEDBACK_LANE_ISSUE_ID = "83fc5b5"
BELLMAN_DIAGNOSTIC_ISSUE_ID = "583d764"
GAMMA_SWEEP_ISSUE_ID = "97604a8"


@dataclass(frozen=True)
class EigenspectrumCoverageConfig:
    """Coverage samples generated from leading exact-audit epsilon eigenmodes."""

    n_modes: int = 0
    scale: float = 1.0
    weight: float = 0.1
    objective: str = "trajectory"
    reference: str = "lqr_exact_budget_l2"


@dataclass(frozen=True)
class ObserverErrorCoverageConfig:
    """Coverage samples generated from leading observer-error disturbance modes."""

    n_modes: int = 0
    scale: float = 1.0
    weight: float = 0.1
    objective: str = "trajectory"
    reference: str = "kalman_observer_error_svd"


@dataclass(frozen=True)
class RolloutRecoveryCondition:
    """One objective-preserving rollout optimizer condition."""

    label: str
    optimizer: str = "lbfgsb"
    use_whitening: bool = False
    use_time_block_preconditioning: bool = False
    maxiter: int = 500
    ftol: float = 1e-12
    gtol: float = 1e-8
    maxls: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    adam_schedule: str = "fixed"
    adam_warmup_fraction: float = 0.05
    adam_end_lr_fraction: float = 0.01
    adam_clip_norm: float | None = None
    polish_maxiter: int | None = None
    initializations: tuple[str, ...] = ("scratch", "bellman_init")
    auxiliary_bellman_weights: tuple[float, ...] = ()
    eigenspectrum_coverage: EigenspectrumCoverageConfig | None = None
    observer_error_coverage: ObserverErrorCoverageConfig | None = None


@dataclass(frozen=True)
class RolloutRecoveryFit:
    """One output-feedback rollout fit."""

    label: str
    condition: RolloutRecoveryCondition
    initialization: str
    K: Float[Array, "T m_u n"]
    objective_initial: float
    objective_final: float
    objective_reference: float
    objective_zero: float
    objective_ratio_to_reference: float
    gain_relative_error: float
    gradient_norm_initial: float
    gradient_norm_final: float
    projected_gradient_norm_final: float | None
    best_objective: float
    best_checkpoint_iteration: int | None
    optimizer_status: str
    optimizer_success: bool
    n_iterations: int
    n_function_evaluations: int
    clean_rollout: OutputFeedbackRollout
    clean_cost: float
    clean_action_mismatch_ratio: float
    under_epsilon_rollout: OutputFeedbackRollout
    under_epsilon_cost: float
    under_epsilon_cost_ratio_to_lqr: float
    under_epsilon_action_mismatch_ratio: float
    exact_l2_cost: float
    exact_l2_cost_ratio_to_lqr: float
    exact_l2_cost_ratio_to_hinf: float
    gamma_penalized_feasible: bool
    gamma_penalized_lambda_over_gamma_squared: float


@dataclass(frozen=True)
class RolloutRecoveryResult:
    """Complete 7a459bb rollout-recovery bundle."""

    issue_id: str
    conditions: tuple[RolloutRecoveryCondition, ...]
    fits: tuple[RolloutRecoveryFit, ...]
    bellman_initialization_gain_relative_error: float
    diagnostics: dict[str, Any]
    arrays: dict[str, np.ndarray]


DEFAULT_CONDITIONS: tuple[RolloutRecoveryCondition, ...] = (
    RolloutRecoveryCondition(label="clean"),
    RolloutRecoveryCondition(
        label="strong_optimizer", maxiter=2000, ftol=1e-14, gtol=1e-10, maxls=100
    ),
    RolloutRecoveryCondition(label="whitened", use_whitening=True),
    RolloutRecoveryCondition(
        label="strong_optimizer_whitened",
        use_whitening=True,
        maxiter=2000,
        ftol=1e-14,
        gtol=1e-10,
        maxls=100,
    ),
    RolloutRecoveryCondition(
        label="strong_optimizer_whitened_block_time",
        use_whitening=True,
        use_time_block_preconditioning=True,
        maxiter=2000,
        ftol=1e-14,
        gtol=1e-10,
        maxls=100,
    ),
    RolloutRecoveryCondition(
        label="strong_optimizer_whitened_bellman_aux",
        use_whitening=True,
        maxiter=500,
        ftol=1e-14,
        gtol=1e-10,
        maxls=100,
        initializations=("scratch",),
        auxiliary_bellman_weights=(0.1, 0.03, 0.01, 0.0),
    ),
)

STRONG_OPTIMIZER_WHITENED = RolloutRecoveryCondition(
    label="strong_optimizer_whitened",
    use_whitening=True,
    maxiter=2000,
    ftol=1e-14,
    gtol=1e-10,
    maxls=100,
    initializations=("scratch",),
)


def adamw_optimizer_whitened(
    label: str,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    maxiter: int = 5000,
    optimizer: str = "adamw",
    adam_schedule: str = "fixed",
    adam_clip_norm: float | None = None,
    polish_maxiter: int | None = None,
    initializations: tuple[str, ...] = ("scratch", "bellman_init"),
) -> RolloutRecoveryCondition:
    """Return a full-batch AdamW condition using the standard whitening map."""

    return RolloutRecoveryCondition(
        label=label,
        optimizer=optimizer,
        use_whitening=True,
        maxiter=maxiter,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_schedule=adam_schedule,
        adam_clip_norm=adam_clip_norm,
        polish_maxiter=polish_maxiter,
        initializations=initializations,
    )


def strong_optimizer_whitened_with_coverage(
    label: str,
    coverage: EigenspectrumCoverageConfig,
) -> RolloutRecoveryCondition:
    """Return the standard sweep optimizer with one named coverage condition."""

    return replace(
        STRONG_OPTIMIZER_WHITENED,
        label=label,
        eigenspectrum_coverage=coverage,
    )


def strong_optimizer_whitened_with_observer_error_coverage(
    label: str,
    coverage: ObserverErrorCoverageConfig,
) -> RolloutRecoveryCondition:
    """Return the standard sweep optimizer with observer-error coverage."""

    return replace(
        STRONG_OPTIMIZER_WHITENED,
        label=label,
        observer_error_coverage=coverage,
    )


def _weighted_covariance(
    samples: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
) -> Float[Array, "n n"]:
    weights = weights.astype(jnp.float64)
    weights = weights / jnp.sum(weights)
    centered = samples - jnp.sum(samples * weights[:, None], axis=0)
    return (centered * weights[:, None]).T @ centered


def _effective_rank_from_covariance(covariance: Float[Array, "n n"]) -> dict[str, float]:
    eigvals = jnp.clip(jnp.linalg.eigvalsh(0.5 * (covariance + covariance.T)), min=0.0)
    total = jnp.sum(eigvals)
    probs = jnp.where(total > 0.0, eigvals / total, jnp.zeros_like(eigvals))
    entropy = -jnp.sum(jnp.where(probs > 0.0, probs * jnp.log(probs), 0.0))
    participation = jnp.where(total > 0.0, total * total / jnp.sum(eigvals * eigvals), 0.0)
    tol = jnp.maximum(1e-14, eigvals[-1] * 1e-10)
    return {
        "trace": float(total),
        "effective_rank_entropy": float(jnp.exp(entropy)),
        "effective_rank_participation": float(participation),
        "numerical_rank": int(jnp.sum(eigvals > tol)),
        "min_eigenvalue": float(eigvals[0]),
        "max_eigenvalue": float(eigvals[-1]),
        "condition_number": float(eigvals[-1] / jnp.maximum(eigvals[0], 1e-30)),
    }


def _state_scales(
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    *,
    floor: float = 1e-4,
) -> Float[Array, " n"]:
    weights = weights.astype(jnp.float64)
    weights = weights / jnp.sum(weights)
    mean = jnp.sum(states * weights[:, None], axis=0)
    var = jnp.sum(weights[:, None] * (states - mean) ** 2, axis=0)
    return jnp.maximum(jnp.sqrt(var), floor)


def _rollout_xhat_coverage(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    config: OutputFeedbackConfig,
) -> dict[str, float]:
    epsilon = jnp.zeros((K.shape[0], plant.m_w), dtype=jnp.float64)

    def one_rollout(x0):
        _x, x_hat, _y, _u, _covs = _kalman_estimator_rollout_arrays(
            plant,
            K,
            x0,
            epsilon,
            config,
        )
        return x_hat[:-1]

    xhats = jax.vmap(one_rollout)(states).reshape((-1, plant.n))
    repeated_weights = jnp.repeat(weights / jnp.sum(weights), K.shape[0])
    cov = _weighted_covariance(xhats, repeated_weights)
    return _effective_rank_from_covariance(cov)


def _time_block_scales(
    plant: PlantLinearization,
    horizon: int,
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    config: OutputFeedbackConfig,
    *,
    floor: float = 1e-4,
) -> Float[Array, "T n"]:
    """Return per-time, per-physical-block xhat scales under zero control."""

    weights = weights.astype(jnp.float64)
    weights = weights / jnp.sum(weights)
    zero_K = jnp.zeros((horizon, plant.m_u, plant.n), dtype=jnp.float64)
    epsilon = jnp.zeros((zero_K.shape[0], plant.m_w), dtype=jnp.float64)

    def one_rollout(x0):
        _x, x_hat, _y, _u, _covs = _kalman_estimator_rollout_arrays(
            plant,
            zero_K,
            x0,
            epsilon,
            config,
        )
        return x_hat[:-1]

    xhats = jax.vmap(one_rollout)(states)
    mean = jnp.sum(xhats * weights[:, None, None], axis=0)
    var = jnp.sum(weights[:, None, None] * (xhats - mean[None]) ** 2, axis=0)
    state_scale = jnp.maximum(jnp.sqrt(var), floor)
    blocks = state_scale.reshape((state_scale.shape[0], config.delay_steps + 1, config.n_phys))
    block_scale = jnp.mean(blocks, axis=-1, keepdims=True)
    expanded = jnp.broadcast_to(block_scale, blocks.shape).reshape(state_scale.shape)
    median = jnp.maximum(jnp.median(expanded), floor)
    return jnp.maximum(expanded / median, floor)


def _training_ensemble(
    plant: PlantLinearization,
    training_config: LinearTrainingConfig,
    output_config: OutputFeedbackConfig,
) -> tuple[Float[Array, "batch n"], Float[Array, " batch"]]:
    x0 = make_cs_output_feedback_initial_state(plant, output_config)
    basis = jnp.eye(plant.n, dtype=jnp.float64)
    key = jax.random.PRNGKey(training_config.seed)
    random_states = jax.random.normal(
        key,
        (training_config.n_random_states, plant.n),
        dtype=jnp.float64,
    )
    states = jnp.concatenate(
        [
            x0[None],
            training_config.basis_scale * basis,
            -training_config.basis_scale * basis,
            training_config.random_state_scale * random_states,
        ],
        axis=0,
    )
    weights = jnp.ones((states.shape[0],), dtype=jnp.float64)
    weights = weights.at[0].set(training_config.reach_weight)
    return states, weights


def _coverage_trajectory_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    coverage_epsilons: Float[Array, "samples T m_w"],
    coverage_weights: Float[Array, " samples"],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    """Mean full-trial task cost under eigenspectrum perturbation trajectories."""

    if coverage_epsilons.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)

    def one_cost(epsilon):
        x, _xhat, _y, u, _covs = _kalman_estimator_rollout_arrays(
            plant,
            K,
            x0,
            epsilon,
            config,
        )
        return rollout_task_cost(schedule, x, u)

    costs = jax.vmap(one_cost)(coverage_epsilons)
    return jnp.sum(coverage_weights * costs) / jnp.maximum(jnp.sum(coverage_weights), 1e-30)


def _kalman_masked_suffix_task_cost(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    gains: Float[Array, "T n obs"],
    x0: Float[Array, " n"],
    xhat0: Float[Array, " n"],
    start_time: Float[Array, ""],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    """Return remaining clean task cost with JAX-scalar start time."""

    H = delayed_observation_matrix(plant, config)
    x_t = x0.astype(jnp.float64)
    xhat_t = xhat0.astype(jnp.float64)
    cost = jnp.asarray(0.0, dtype=jnp.float64)
    for t in range(schedule.T):
        active = jnp.asarray(t, dtype=jnp.int32) >= start_time
        u_t = -K[t] @ xhat_t
        y_t = H @ x_t
        stage = x_t @ schedule.Q[t] @ x_t + u_t @ schedule.R[t] @ u_t
        xhat_next = plant.A @ xhat_t + plant.B @ u_t + gains[t] @ (y_t - H @ xhat_t)
        x_next = plant.A @ x_t + plant.B @ u_t
        cost = cost + jnp.where(active, stage, 0.0)
        x_t = jnp.where(active, x_next, x_t)
        xhat_t = jnp.where(active, xhat_next, xhat_t)
    return cost + x_t @ schedule.Q_f @ x_t


def _coverage_state_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    coverage_x: Float[Array, "samples n"],
    coverage_xhat: Float[Array, "samples n"],
    coverage_times: Float[Array, " samples"],
    coverage_state_weights: Float[Array, " samples"],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    """Mean remaining-horizon task cost on time-indexed coverage states."""

    if coverage_x.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    gains = kalman_estimator_gains(plant, K, config)

    def one_cost(x, xhat, start_time):
        return _kalman_masked_suffix_task_cost(
            plant,
            schedule,
            K,
            gains,
            x,
            xhat,
            start_time,
            config,
        )

    costs = jax.vmap(one_cost)(coverage_x, coverage_xhat, coverage_times)
    return jnp.sum(coverage_state_weights * costs) / jnp.maximum(
        jnp.sum(coverage_state_weights), 1e-30
    )


def _empty_coverage_samples(
    plant: PlantLinearization,
) -> tuple[
    Float[Array, "samples T m_w"],
    Float[Array, " samples"],
    Float[Array, "state_samples n"],
    Float[Array, "state_samples n"],
    Float[Array, " state_samples"],
    Float[Array, " state_samples"],
]:
    _ = plant
    return (
        jnp.zeros((0, 0, 0), dtype=jnp.float64),
        jnp.zeros((0,), dtype=jnp.float64),
        jnp.zeros((0, plant.n), dtype=jnp.float64),
        jnp.zeros((0, plant.n), dtype=jnp.float64),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.float64),
    )


def _eigenspectrum_coverage_samples(
    *,
    plant: PlantLinearization,
    schedule: CostSchedule,
    K_ref: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    budget_l2: float,
    gamma: float,
    output_config: OutputFeedbackConfig,
    coverage_config: EigenspectrumCoverageConfig,
) -> tuple[
    Float[Array, "samples T m_w"],
    Float[Array, " samples"],
    Float[Array, "state_samples n"],
    Float[Array, "state_samples n"],
    Float[Array, " state_samples"],
    Float[Array, " state_samples"],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Create signed full-trial perturbation trajectories from leading epsilon modes."""

    if coverage_config.n_modes <= 0 or coverage_config.weight <= 0.0:
        empty = _empty_coverage_samples(plant)
        return (*empty, {"enabled": False}, {})

    audit = exact_output_feedback_adversary_audit(
        label="lqr_eigenspectrum_coverage_source",
        plant=plant,
        schedule=schedule,
        controller_gains=K_ref,
        x0=x0,
        budget=budget_l2 * budget_l2,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        eigenspectrum_modes=coverage_config.n_modes,
        config=output_config,
    )
    modes = audit["eigenspectrum"]["epsilon_modes"]
    eigenvalues = audit["eigenspectrum"]["eigenvalues"]
    alpha = jnp.asarray(coverage_config.scale * budget_l2, dtype=jnp.float64)
    xs = []
    xhats = []
    times = []
    epsilons = []
    weights = []
    for mode_idx in range(modes.shape[0]):
        for sign in (-1.0, 1.0):
            epsilon = sign * alpha * modes[mode_idx]
            rollout = rollout_with_kalman_estimator(
                plant,
                K_ref,
                x0,
                epsilon,
                config=output_config,
            )
            xs.append(rollout.x[:-1])
            xhats.append(rollout.x_hat[:-1])
            times.append(jnp.arange(schedule.T, dtype=jnp.int32))
            epsilons.append(epsilon)
            weights.append(jnp.asarray(coverage_config.weight, dtype=jnp.float64))
    coverage_epsilons = jnp.stack(epsilons, axis=0)
    coverage_weights = jnp.stack(weights, axis=0)
    coverage_x = jnp.concatenate(xs, axis=0)
    coverage_xhat = jnp.concatenate(xhats, axis=0)
    coverage_times = jnp.concatenate(times, axis=0)
    state_weights = jnp.repeat(
        coverage_weights / jnp.sum(coverage_weights),
        schedule.T,
        total_repeat_length=coverage_xhat.shape[0],
    )
    coverage_state_weights = state_weights * coverage_config.weight
    coverage_cov = _weighted_covariance(coverage_xhat, state_weights)
    metadata = {
        "enabled": True,
        "n_modes": coverage_config.n_modes,
        "scale": coverage_config.scale,
        "alpha": float(alpha),
        "weight": coverage_config.weight,
        "objective": coverage_config.objective,
        "reference": coverage_config.reference,
        "source": "signed leading exact-audit epsilon eigenmodes of the analytical LQR controller",
        "n_trajectories": int(coverage_epsilons.shape[0]),
        "n_state_samples_for_diagnostics": int(coverage_x.shape[0]),
        "eigenvalues": [float(value) for value in eigenvalues],
        "xhat_coverage": _effective_rank_from_covariance(coverage_cov),
    }
    arrays = {
        "coverage_epsilon_modes": np.asarray(modes),
        "coverage_eigenvalues": np.asarray(eigenvalues),
        "coverage_epsilons": np.asarray(coverage_epsilons),
        "coverage_x": np.asarray(coverage_x),
        "coverage_x_hat": np.asarray(coverage_xhat),
        "coverage_times": np.asarray(coverage_times),
        "coverage_state_weights": np.asarray(coverage_state_weights),
        "coverage_weights": np.asarray(coverage_weights),
    }
    return (
        coverage_epsilons,
        coverage_weights,
        coverage_x,
        coverage_xhat,
        coverage_times,
        coverage_state_weights,
        metadata,
        arrays,
    )


def _observer_error_epsilon_modes(
    *,
    plant: PlantLinearization,
    K_ref: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    output_config: OutputFeedbackConfig,
    n_modes: int,
) -> tuple[Float[Array, "modes T m_w"], Float[Array, " modes"]]:
    """Return disturbance directions ranked by induced Kalman observer-error energy."""

    horizon = K_ref.shape[0]
    n_inputs = horizon * plant.m_w
    basis = jnp.eye(n_inputs, dtype=jnp.float64)

    def one_error(epsilon_flat: Float[Array, " input"]) -> Float[Array, " flat"]:
        epsilon = epsilon_flat.reshape((horizon, plant.m_w))
        x, xhat, _y, _u, _covs = _kalman_estimator_rollout_arrays(
            plant,
            K_ref,
            x0,
            epsilon,
            output_config,
        )
        observer_error = x[:-1] - xhat[:-1]
        return observer_error.reshape(-1)

    observer_error_map = jax.vmap(one_error)(basis).T
    _left, singular_values, right_t = jnp.linalg.svd(observer_error_map, full_matrices=False)
    n_kept = min(int(n_modes), right_t.shape[0])
    modes = right_t[:n_kept].reshape((n_kept, horizon, plant.m_w))
    return modes, singular_values[:n_kept]


def _observer_error_coverage_samples(
    *,
    plant: PlantLinearization,
    schedule: CostSchedule,
    K_ref: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    budget_l2: float,
    output_config: OutputFeedbackConfig,
    coverage_config: ObserverErrorCoverageConfig,
) -> tuple[
    Float[Array, "samples T m_w"],
    Float[Array, " samples"],
    Float[Array, "state_samples n"],
    Float[Array, "state_samples n"],
    Float[Array, " state_samples"],
    Float[Array, " state_samples"],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Create signed coverage samples from leading observer-error disturbance modes."""

    if coverage_config.n_modes <= 0 or coverage_config.weight <= 0.0:
        empty = _empty_coverage_samples(plant)
        return (*empty, {"enabled": False}, {})

    modes, singular_values = _observer_error_epsilon_modes(
        plant=plant,
        K_ref=K_ref,
        x0=x0,
        output_config=output_config,
        n_modes=coverage_config.n_modes,
    )
    alpha = jnp.asarray(coverage_config.scale * budget_l2, dtype=jnp.float64)
    xs = []
    xhats = []
    errors = []
    times = []
    epsilons = []
    weights = []
    for mode_idx in range(modes.shape[0]):
        for sign in (-1.0, 1.0):
            epsilon = sign * alpha * modes[mode_idx]
            rollout = rollout_with_kalman_estimator(
                plant,
                K_ref,
                x0,
                epsilon,
                config=output_config,
            )
            xs.append(rollout.x[:-1])
            xhats.append(rollout.x_hat[:-1])
            errors.append(rollout.x[:-1] - rollout.x_hat[:-1])
            times.append(jnp.arange(schedule.T, dtype=jnp.int32))
            epsilons.append(epsilon)
            weights.append(jnp.asarray(coverage_config.weight, dtype=jnp.float64))
    coverage_epsilons = jnp.stack(epsilons, axis=0)
    coverage_weights = jnp.stack(weights, axis=0)
    coverage_x = jnp.concatenate(xs, axis=0)
    coverage_xhat = jnp.concatenate(xhats, axis=0)
    coverage_errors = jnp.concatenate(errors, axis=0)
    coverage_times = jnp.concatenate(times, axis=0)
    state_weights = jnp.repeat(
        coverage_weights / jnp.sum(coverage_weights),
        schedule.T,
        total_repeat_length=coverage_xhat.shape[0],
    )
    coverage_state_weights = state_weights * coverage_config.weight
    coverage_cov = _weighted_covariance(coverage_errors, state_weights)
    metadata = {
        "enabled": True,
        "n_modes": int(modes.shape[0]),
        "scale": coverage_config.scale,
        "alpha": float(alpha),
        "weight": coverage_config.weight,
        "objective": coverage_config.objective,
        "reference": coverage_config.reference,
        "source": (
            "signed leading right singular vectors of the analytical LQR "
            "disturbance-to-observer-error map"
        ),
        "n_trajectories": int(coverage_epsilons.shape[0]),
        "n_state_samples_for_diagnostics": int(coverage_x.shape[0]),
        "singular_values": [float(value) for value in singular_values],
        "observer_error_coverage": _effective_rank_from_covariance(coverage_cov),
    }
    arrays = {
        "coverage_observer_error_epsilon_modes": np.asarray(modes),
        "coverage_observer_error_singular_values": np.asarray(singular_values),
        "coverage_epsilons": np.asarray(coverage_epsilons),
        "coverage_x": np.asarray(coverage_x),
        "coverage_x_hat": np.asarray(coverage_xhat),
        "coverage_observer_error": np.asarray(coverage_errors),
        "coverage_times": np.asarray(coverage_times),
        "coverage_state_weights": np.asarray(coverage_state_weights),
        "coverage_weights": np.asarray(coverage_weights),
    }
    return (
        coverage_epsilons,
        coverage_weights,
        coverage_x,
        coverage_xhat,
        coverage_times,
        coverage_state_weights,
        metadata,
        arrays,
    )


def _make_parameter_maps(
    condition: RolloutRecoveryCondition,
    K_ref: Float[Array, "T m_u n"],
    state_scales: Float[Array, " n"],
    time_block_scales: Float[Array, "T n"] | None = None,
):
    scales = jnp.ones_like(K_ref, dtype=jnp.float64)
    if condition.use_whitening:
        scales = scales * state_scales.astype(jnp.float64)[None, None, :]
    if condition.use_time_block_preconditioning:
        if time_block_scales is None:
            raise ValueError("time_block_scales is required for time-block preconditioning")
        scales = scales * time_block_scales.astype(jnp.float64)[:, None, :]

    def to_theta(K: Float[Array, "T m_u n"]) -> Float[Array, "T m_u n"]:
        if not condition.use_whitening and not condition.use_time_block_preconditioning:
            return K.astype(jnp.float64)
        return K.astype(jnp.float64) * scales

    def to_K(theta: Float[Array, "T m_u n"]) -> Float[Array, "T m_u n"]:
        if not condition.use_whitening and not condition.use_time_block_preconditioning:
            return theta.astype(jnp.float64)
        return theta.astype(jnp.float64) / scales

    return to_theta, to_K, to_theta(K_ref)


def _gradient_norms(
    value_and_grad_flat,
    theta: np.ndarray,
) -> tuple[float, np.ndarray]:
    value, grad = value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
    _ = value
    grad_np = np.asarray(grad, dtype=np.float64)
    return float(np.linalg.norm(grad_np)), grad_np


def _validate_condition(condition: RolloutRecoveryCondition) -> None:
    valid_optimizers = {"lbfgsb", "adamw", "adamw_then_lbfgsb"}
    if condition.optimizer not in valid_optimizers:
        raise ValueError(
            f"Unknown optimizer={condition.optimizer!r}; expected one of "
            f"{sorted(valid_optimizers)}."
        )
    if condition.optimizer in {"adamw", "adamw_then_lbfgsb"} and condition.maxiter < 1:
        raise ValueError("AdamW conditions require maxiter >= 1.")
    if condition.optimizer in {"adamw", "adamw_then_lbfgsb"} and condition.learning_rate <= 0.0:
        raise ValueError("AdamW conditions require a positive learning_rate.")
    if condition.adam_schedule not in {"fixed", "warmup_cosine"}:
        raise ValueError(
            f"Unknown adam_schedule={condition.adam_schedule!r}; expected 'fixed' or "
            "'warmup_cosine'."
        )
    if condition.adam_schedule == "warmup_cosine" and condition.maxiter < 2:
        raise ValueError("warmup_cosine AdamW conditions require maxiter >= 2.")
    if not 0.0 <= condition.adam_warmup_fraction < 1.0:
        raise ValueError("adam_warmup_fraction must be in [0, 1).")
    if condition.adam_end_lr_fraction < 0.0:
        raise ValueError("adam_end_lr_fraction must be non-negative.")
    if condition.adam_clip_norm is not None and condition.adam_clip_norm <= 0.0:
        raise ValueError("adam_clip_norm must be positive when provided.")
    if condition.polish_maxiter is not None and condition.polish_maxiter < 1:
        raise ValueError("polish_maxiter must be >= 1 when provided.")
    if condition.use_time_block_preconditioning and not condition.use_whitening:
        raise ValueError("Time-block preconditioning requires whitening.")


def _action_mismatch_ratio(
    candidate_u: Float[Array, "T m_u"],
    reference_u: Float[Array, "T m_u"],
    *,
    floor: float,
) -> float:
    mismatch = jnp.sqrt(jnp.mean((candidate_u - reference_u) ** 2))
    reference = jnp.sqrt(jnp.mean(reference_u**2))
    return float(mismatch / jnp.maximum(reference, floor))


def _fit_one_condition(
    *,
    condition: RolloutRecoveryCondition,
    initialization: str,
    initial_K: Float[Array, "T m_u n"],
    plant: PlantLinearization,
    schedule: CostSchedule,
    K_ref: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    coverage_epsilons: Float[Array, "coverage T m_w"],
    coverage_trajectory_weights: Float[Array, " coverage"],
    coverage_x: Float[Array, "coverage n"],
    coverage_xhat: Float[Array, "coverage n"],
    coverage_times: Float[Array, " coverage"],
    coverage_state_weights: Float[Array, " coverage"],
    state_scales: Float[Array, " n"],
    time_block_scales: Float[Array, "T n"],
    bellman_p_next: Float[Array, "T n n"],
    bellman_reference_objective: float,
    x0: Float[Array, " n"],
    riccati_epsilon: Float[Array, "T m_w"],
    budget: float,
    gamma: float,
    lqr_clean_rollout: OutputFeedbackRollout,
    lqr_under_epsilon_rollout: OutputFeedbackRollout,
    lqr_exact_cost: float,
    hinf_exact_cost: float,
    output_config: OutputFeedbackConfig,
) -> RolloutRecoveryFit:
    to_theta, to_K, _K_ref_theta = _make_parameter_maps(
        condition,
        K_ref,
        state_scales,
        time_block_scales,
    )
    theta0 = np.asarray(to_theta(initial_K), dtype=np.float64).reshape(-1)
    shape = K_ref.shape

    def coverage_objective(gains: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        coverage = condition.eigenspectrum_coverage or condition.observer_error_coverage
        if coverage is None:
            return jnp.asarray(0.0, dtype=jnp.float64)
        if coverage.objective == "trajectory":
            return _coverage_trajectory_objective(
                plant,
                schedule,
                gains,
                x0,
                coverage_epsilons,
                coverage_trajectory_weights,
                output_config,
            )
        if coverage.objective == "state":
            return _coverage_state_objective(
                plant,
                schedule,
                gains,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
                output_config,
            )
        raise ValueError(f"Unknown coverage objective={coverage.objective!r}.")

    reference_objective = float(
        output_feedback_clean_objective(
            plant,
            schedule,
            K_ref,
            states,
            weights,
            output_config,
        )
        + coverage_objective(K_ref)
    )

    def objective_theta(theta_tree: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        gains = to_K(theta_tree)
        objective = output_feedback_clean_objective(
            plant,
            schedule,
            gains,
            states,
            weights,
            output_config,
        )
        if (
            condition.eigenspectrum_coverage is not None
            or condition.observer_error_coverage is not None
        ):
            objective = objective + coverage_objective(gains)
        return objective

    def composite_objective_theta(
        theta_tree: Float[Array, "T m_u n"],
        bellman_weight: Float[Array, ""],
    ) -> Float[Array, ""]:
        gains = to_K(theta_tree)
        rollout_ratio = objective_theta(theta_tree) / reference_objective
        bellman_ratio = (
            output_feedback_lqr_bellman_objective(
                plant,
                schedule,
                bellman_p_next,
                gains,
                states,
                weights,
            )
            / bellman_reference_objective
        )
        return rollout_ratio + bellman_weight * bellman_ratio

    @jax.jit
    def clean_value_and_grad_flat(
        theta: Float[Array, " flat"],
    ) -> tuple[Float[Array, ""], Array]:
        theta_tree = theta.reshape(shape)
        value, grads = jax.value_and_grad(objective_theta)(theta_tree)
        return value, grads.reshape(-1)

    @jax.jit
    def composite_value_and_grad_flat(
        theta: Float[Array, " flat"],
        bellman_weight: Float[Array, ""],
    ) -> tuple[Float[Array, ""], Array]:
        theta_tree = theta.reshape(shape)
        value, grads = jax.value_and_grad(composite_objective_theta)(
            theta_tree,
            bellman_weight,
        )
        return value, grads.reshape(-1)

    @jax.jit
    def clean_value_flat(theta: Float[Array, " flat"]) -> Float[Array, ""]:
        return objective_theta(theta.reshape(shape))

    def scipy_value_and_grad_for_weight(bellman_weight: float):
        def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            if not condition.auxiliary_bellman_weights:
                value, grads = clean_value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
                return float(value), np.asarray(grads, dtype=np.float64)
            value, grads = composite_value_and_grad_flat(
                jnp.asarray(theta, dtype=jnp.float64),
                jnp.asarray(bellman_weight, dtype=jnp.float64),
            )
            return float(value), np.asarray(grads, dtype=np.float64)

        return scipy_value_and_grad

    initial_objective = float(
        objective_theta(jnp.asarray(theta0, dtype=jnp.float64).reshape(shape))
    )
    initial_grad_norm, _ = _gradient_norms(clean_value_and_grad_flat, theta0)
    theta_current = theta0
    stage_weights = condition.auxiliary_bellman_weights or (0.0,)
    stage_messages = []
    total_iterations = 0
    total_evaluations = 0
    optimizer_success = True
    projected_gradient_norm = None
    best_objective = initial_objective
    best_iteration: int | None = 0
    def run_lbfgsb_from(
        theta_start: np.ndarray,
        *,
        maxiter: int,
    ) -> tuple[np.ndarray, str, bool, int, int]:
        scipy_result = scipy_opt.minimize(
            scipy_value_and_grad_for_weight(0.0),
            theta_start,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": condition.ftol,
                "gtol": condition.gtol,
                "maxls": condition.maxls,
            },
        )
        return (
            np.asarray(scipy_result.x, dtype=np.float64),
            str(scipy_result.message),
            bool(scipy_result.success),
            int(scipy_result.nit),
            int(scipy_result.nfev),
        )

    if condition.optimizer == "lbfgsb":
        for bellman_weight in stage_weights:
            scipy_result = scipy_opt.minimize(
                scipy_value_and_grad_for_weight(bellman_weight),
                theta_current,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": condition.maxiter,
                    "ftol": condition.ftol,
                    "gtol": condition.gtol,
                    "maxls": condition.maxls,
                },
            )
            theta_current = np.asarray(scipy_result.x, dtype=np.float64)
            total_iterations += int(scipy_result.nit)
            total_evaluations += int(scipy_result.nfev)
            optimizer_success = optimizer_success and bool(scipy_result.success)
            projected_gradient = getattr(scipy_result, "jac", None)
            projected_gradient_norm = (
                None
                if projected_gradient is None
                else float(np.linalg.norm(np.asarray(projected_gradient, dtype=np.float64)))
            )
            clean_value = float(clean_value_flat(jnp.asarray(theta_current, dtype=jnp.float64)))
            if clean_value <= best_objective:
                best_objective = clean_value
                best_iteration = total_iterations
            if condition.auxiliary_bellman_weights:
                stage_messages.append(
                    f"bellman_weight={bellman_weight:g}: {scipy_result.message}"
                )
            else:
                stage_messages.append(str(scipy_result.message))
    else:
        if condition.adam_schedule == "fixed":
            learning_rate = condition.learning_rate
        else:
            warmup_steps = min(
                max(1, int(round(condition.maxiter * condition.adam_warmup_fraction))),
                condition.maxiter - 1,
            )
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=condition.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=condition.maxiter,
                end_value=condition.learning_rate * condition.adam_end_lr_fraction,
            )
        transforms = []
        if condition.adam_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(condition.adam_clip_norm))
        transforms.append(
            optax.adamw(
                learning_rate=learning_rate,
                b1=condition.adam_b1,
                b2=condition.adam_b2,
                eps=condition.adam_eps,
                weight_decay=condition.weight_decay,
            )
        )
        optimizer = optax.chain(*transforms)

        @jax.jit
        def adamw_stage(
            theta: Float[Array, " flat"],
            opt_state: optax.OptState,
            bellman_weight: Float[Array, ""],
            starting_step: Float[Array, ""],
        ):
            best_theta = theta
            best_value = clean_value_flat(theta)
            best_step = starting_step
            all_finite = jnp.asarray(True)

            def step(carry, step_index):
                theta, opt_state, best_theta, best_value, best_step, all_finite = carry
                if condition.auxiliary_bellman_weights:
                    train_value, grads = composite_value_and_grad_flat(theta, bellman_weight)
                else:
                    train_value, grads = clean_value_and_grad_flat(theta)
                updates, opt_state = optimizer.update(grads, opt_state, theta)
                theta = optax.apply_updates(theta, updates)
                clean_value = clean_value_flat(theta)
                finite = jnp.isfinite(train_value) & jnp.isfinite(clean_value)
                finite = finite & jnp.all(jnp.isfinite(grads)) & jnp.all(jnp.isfinite(theta))
                improved = finite & (clean_value < best_value)
                absolute_step = starting_step + step_index + 1
                best_theta = jnp.where(improved, theta, best_theta)
                best_value = jnp.where(improved, clean_value, best_value)
                best_step = jnp.where(improved, absolute_step, best_step)
                all_finite = all_finite & finite
                return (theta, opt_state, best_theta, best_value, best_step, all_finite), None

            steps = jnp.arange(condition.maxiter)
            (theta, opt_state, best_theta, best_value, best_step, all_finite), _ = jax.lax.scan(
                step,
                (theta, opt_state, best_theta, best_value, best_step, all_finite),
                steps,
            )
            return theta, opt_state, best_theta, best_value, best_step, all_finite

        best_theta = jnp.asarray(theta_current, dtype=jnp.float64)
        for bellman_weight in stage_weights:
            opt_state = optimizer.init(jnp.asarray(theta_current, dtype=jnp.float64))
            (
                _theta_terminal,
                _opt_state,
                stage_best_theta,
                stage_best_value,
                stage_best_step,
                all_finite,
            ) = adamw_stage(
                jnp.asarray(theta_current, dtype=jnp.float64),
                opt_state,
                jnp.asarray(bellman_weight, dtype=jnp.float64),
                jnp.asarray(total_iterations, dtype=jnp.int64),
            )
            theta_current = np.asarray(stage_best_theta, dtype=np.float64)
            best_theta = stage_best_theta
            best_objective = float(stage_best_value)
            best_iteration = int(stage_best_step)
            total_iterations += condition.maxiter
            total_evaluations += condition.maxiter
            optimizer_success = optimizer_success and bool(all_finite)
            if condition.auxiliary_bellman_weights:
                stage_messages.append(
                    f"bellman_weight={bellman_weight:g}: AdamW completed "
                    f"{condition.maxiter} full-batch steps"
                )
            else:
                stage_messages.append(
                    f"AdamW completed {condition.maxiter} full-batch steps "
                    f"(lr={condition.learning_rate:g}, schedule={condition.adam_schedule}, "
                    f"clip={condition.adam_clip_norm}, weight_decay={condition.weight_decay:g})"
                )
        theta_current = np.asarray(best_theta, dtype=np.float64)
        if condition.optimizer == "adamw_then_lbfgsb":
            polish_maxiter = condition.polish_maxiter or min(condition.maxiter, 500)
            polish_theta, message, success, nit, nfev = run_lbfgsb_from(
                theta_current,
                maxiter=polish_maxiter,
            )
            total_iterations += nit
            total_evaluations += nfev
            optimizer_success = optimizer_success and success
            clean_value = float(clean_value_flat(jnp.asarray(polish_theta, dtype=jnp.float64)))
            if clean_value <= best_objective:
                theta_current = polish_theta
                best_theta = jnp.asarray(polish_theta, dtype=jnp.float64)
                best_objective = clean_value
                best_iteration = total_iterations
            stage_messages.append(f"L-BFGS-B polish maxiter={polish_maxiter}: {message}")
    theta_final = jnp.asarray(theta_current, dtype=jnp.float64).reshape(shape)
    K_final = to_K(theta_final)
    final_objective = float(objective_theta(theta_final))
    zero_objective = float(objective_theta(to_theta(jnp.zeros_like(K_ref))))
    final_grad_norm, _final_grad = _gradient_norms(
        clean_value_and_grad_flat,
        np.asarray(theta_final, dtype=np.float64).reshape(-1),
    )
    clean = rollout_with_kalman_estimator(plant, K_final, x0, config=output_config)
    under_eps = rollout_with_kalman_estimator(
        plant,
        K_final,
        x0,
        riccati_epsilon,
        config=output_config,
    )
    clean_cost = float(rollout_task_cost(schedule, clean.x, clean.u))
    under_epsilon_cost = output_feedback_cost(schedule, under_eps).total_without_disturbance_penalty
    audit = exact_output_feedback_adversary_audit(
        label=f"{condition.label}_{initialization}",
        plant=plant,
        schedule=schedule,
        controller_gains=K_final,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        config=output_config,
    )
    return RolloutRecoveryFit(
        label=f"{condition.label}__{initialization}",
        condition=condition,
        initialization=initialization,
        K=K_final,
        objective_initial=initial_objective,
        objective_final=final_objective,
        objective_reference=reference_objective,
        objective_zero=zero_objective,
        objective_ratio_to_reference=final_objective / reference_objective,
        gain_relative_error=float(jnp.linalg.norm(K_final - K_ref) / jnp.linalg.norm(K_ref)),
        gradient_norm_initial=initial_grad_norm,
        gradient_norm_final=final_grad_norm,
        projected_gradient_norm_final=projected_gradient_norm,
        best_objective=best_objective,
        best_checkpoint_iteration=best_iteration,
        optimizer_status="; ".join(stage_messages),
        optimizer_success=optimizer_success,
        n_iterations=total_iterations,
        n_function_evaluations=total_evaluations,
        clean_rollout=clean,
        clean_cost=clean_cost,
        clean_action_mismatch_ratio=_action_mismatch_ratio(
            clean.u,
            lqr_clean_rollout.u,
            floor=output_config.denominator_floor,
        ),
        under_epsilon_rollout=under_eps,
        under_epsilon_cost=under_epsilon_cost,
        under_epsilon_cost_ratio_to_lqr=(
            under_epsilon_cost
            / output_feedback_cost(
                schedule, lqr_under_epsilon_rollout
            ).total_without_disturbance_penalty
        ),
        under_epsilon_action_mismatch_ratio=_action_mismatch_ratio(
            under_eps.u,
            lqr_under_epsilon_rollout.u,
            floor=output_config.denominator_floor,
        ),
        exact_l2_cost=audit["cost"].total_without_disturbance_penalty,
        exact_l2_cost_ratio_to_lqr=(
            audit["cost"].total_without_disturbance_penalty / lqr_exact_cost
        ),
        exact_l2_cost_ratio_to_hinf=(
            audit["cost"].total_without_disturbance_penalty / hinf_exact_cost
        ),
        gamma_penalized_feasible=bool(audit["gamma_penalized"]["feasible"]),
        gamma_penalized_lambda_over_gamma_squared=float(
            audit["gamma_penalized"]["max_eigenvalue_over_gamma_squared"]
        ),
    )


def run_output_feedback_rollout_recovery(
    *,
    conditions: tuple[RolloutRecoveryCondition, ...] = DEFAULT_CONDITIONS,
    training_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=500),
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> RolloutRecoveryResult:
    """Run the requested clean output-feedback rollout-recovery matrix."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    K_ref = reference.lqr_solution.K
    x0 = make_cs_output_feedback_initial_state(plant, output_config)
    states, weights = _training_ensemble(plant, training_config, output_config)
    scales = _state_scales(states, weights)
    time_block_scales = _time_block_scales(
        plant,
        schedule.T,
        states,
        weights,
        output_config,
    )
    bellman_p_next = reference.lqr_solution.P[1:].astype(jnp.float64)
    bellman_reference_objective = float(
        output_feedback_lqr_bellman_objective(
            plant,
            schedule,
            bellman_p_next,
            K_ref,
            states,
            weights,
        )
    )
    bellman = train_output_feedback_lqr_bellman_controller(
        reference,
        LinearTrainingConfig(n_steps=200, seed=training_config.seed),
    )
    lqr_clean = rollout_with_kalman_estimator(plant, K_ref, x0, config=output_config)
    covs = robust_estimator_covariances(plant, schedule, gamma_ref.gamma, output_config)
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        output_config,
    )
    robust_policy = robust_estimator_fixed_adversary_policy(
        plant,
        schedule,
        gamma_ref.solution,
        robust_gains,
        covs,
        output_config,
    )
    robust_rollout = rollout_with_robust_estimator_policy(
        plant,
        schedule,
        gamma_ref.solution,
        x0,
        robust_policy,
        gains=robust_gains,
        config=output_config,
    )
    riccati_epsilon = robust_rollout.epsilon
    budget = float(jnp.sum(riccati_epsilon**2))
    lqr_under_eps = rollout_with_kalman_estimator(
        plant,
        K_ref,
        x0,
        riccati_epsilon,
        config=output_config,
    )
    lqr_exact = exact_output_feedback_adversary_audit(
        label="analytical_lqr_kalman",
        plant=plant,
        schedule=schedule,
        controller_gains=K_ref,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma_ref.gamma,
        config=output_config,
    )
    hinf_exact = exact_output_feedback_adversary_audit(
        label="analytical_hinf_robust",
        plant=plant,
        schedule=schedule,
        controller_gains=robust_gains,
        x0=x0,
        budget=budget,
        estimator_kind="robust",
        solution=gamma_ref.solution,
        penalty_gamma=gamma_ref.gamma,
        config=output_config,
    )

    fits = []
    eigenspectrum_coverage_by_label: dict[str, dict[str, Any]] = {}
    observer_error_coverage_by_label: dict[str, dict[str, Any]] = {}
    coverage_arrays: dict[str, np.ndarray] = {}
    for condition in conditions:
        _validate_condition(condition)
        if (
            condition.eigenspectrum_coverage is not None
            and condition.observer_error_coverage is not None
        ):
            raise ValueError(
                f"Condition {condition.label!r} cannot combine eigenspectrum and "
                "observer-error coverage."
            )
        if condition.eigenspectrum_coverage is None and condition.observer_error_coverage is None:
            (
                coverage_epsilons,
                coverage_trajectory_weights,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
            ) = _empty_coverage_samples(plant)
        elif condition.eigenspectrum_coverage is not None:
            (
                coverage_epsilons,
                coverage_trajectory_weights,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
                coverage_metadata,
                condition_arrays,
            ) = _eigenspectrum_coverage_samples(
                plant=plant,
                schedule=schedule,
                K_ref=K_ref,
                x0=x0,
                budget_l2=float(jnp.sqrt(jnp.asarray(budget))),
                gamma=gamma_ref.gamma,
                output_config=output_config,
                coverage_config=condition.eigenspectrum_coverage,
            )
            eigenspectrum_coverage_by_label[condition.label] = coverage_metadata
            for key, value in condition_arrays.items():
                coverage_arrays[f"{condition.label}_{key}"] = value
        else:
            assert condition.observer_error_coverage is not None
            (
                coverage_epsilons,
                coverage_trajectory_weights,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
                coverage_metadata,
                condition_arrays,
            ) = _observer_error_coverage_samples(
                plant=plant,
                schedule=schedule,
                K_ref=K_ref,
                x0=x0,
                budget_l2=float(jnp.sqrt(jnp.asarray(budget))),
                output_config=output_config,
                coverage_config=condition.observer_error_coverage,
            )
            observer_error_coverage_by_label[condition.label] = coverage_metadata
            for key, value in condition_arrays.items():
                coverage_arrays[f"{condition.label}_{key}"] = value
        initial_K_by_name = {
            "scratch": jnp.zeros_like(K_ref),
            "bellman_init": bellman.K,
        }
        for initialization in condition.initializations:
            initial_K = initial_K_by_name[initialization]
            fits.append(
                _fit_one_condition(
                    condition=condition,
                    initialization=initialization,
                    initial_K=initial_K,
                    plant=plant,
                    schedule=schedule,
                    K_ref=K_ref,
                    states=states,
                    weights=weights,
                    coverage_epsilons=coverage_epsilons,
                    coverage_trajectory_weights=coverage_trajectory_weights,
                    coverage_x=coverage_x,
                    coverage_xhat=coverage_xhat,
                    coverage_times=coverage_times,
                    coverage_state_weights=coverage_state_weights,
                    state_scales=scales,
                    time_block_scales=time_block_scales,
                    bellman_p_next=bellman_p_next,
                    bellman_reference_objective=bellman_reference_objective,
                    x0=x0,
                    riccati_epsilon=riccati_epsilon,
                    budget=budget,
                    gamma=gamma_ref.gamma,
                    lqr_clean_rollout=lqr_clean,
                    lqr_under_epsilon_rollout=lqr_under_eps,
                    lqr_exact_cost=lqr_exact["cost"].total_without_disturbance_penalty,
                    hinf_exact_cost=hinf_exact["cost"].total_without_disturbance_penalty,
                    output_config=output_config,
                )
            )

    ensemble_cov = _weighted_covariance(states, weights)
    diagnostics = {
        "training_config": training_config.__dict__,
        "output_config": output_config.__dict__,
        "gamma": gamma_ref.gamma,
        "gamma_factor": gamma_ref.factor,
        "gamma_star": reference.gamma_star,
        "budget": budget,
        "budget_l2": float(jnp.sqrt(jnp.asarray(budget))),
        "state_scales": {
            "min": float(jnp.min(scales)),
            "max": float(jnp.max(scales)),
            "median": float(jnp.median(scales)),
            "condition": float(jnp.max(scales) / jnp.min(scales)),
        },
        "time_block_scales": {
            "min": float(jnp.min(time_block_scales)),
            "max": float(jnp.max(time_block_scales)),
            "median": float(jnp.median(time_block_scales)),
            "condition": float(jnp.max(time_block_scales) / jnp.min(time_block_scales)),
        },
        "bellman_auxiliary": {
            "objective": "clean_rollout_ratio + weight * one_step_lqr_bellman_ratio",
            "reference_objective": bellman_reference_objective,
            "schedules": {
                condition.label: condition.auxiliary_bellman_weights
                for condition in conditions
                if condition.auxiliary_bellman_weights
            },
            "anchor_to_known_controller": False,
        },
        "eigenspectrum_coverage": eigenspectrum_coverage_by_label,
        "observer_error_coverage": observer_error_coverage_by_label,
        "initial_state_ensemble": _effective_rank_from_covariance(ensemble_cov),
        "lqr_clean": {
            "cost": output_feedback_cost(schedule, lqr_clean).total_without_disturbance_penalty,
            "rollout": _rollout_summary(lqr_clean),
            "xhat_coverage": _rollout_xhat_coverage(plant, K_ref, states, weights, output_config),
        },
        "lqr_under_riccati_epsilon": {
            "cost": output_feedback_cost(schedule, lqr_under_eps).total_without_disturbance_penalty,
            "rollout": _rollout_summary(lqr_under_eps),
        },
        "analytical_exact_audits": {
            "lqr": {
                "cost": lqr_exact["cost"].total_without_disturbance_penalty,
                "lambda_over_gamma_squared": lqr_exact["gamma_penalized"][
                    "max_eigenvalue_over_gamma_squared"
                ],
                "gamma_penalized_feasible": lqr_exact["gamma_penalized"]["feasible"],
            },
            "hinf": {
                "cost": hinf_exact["cost"].total_without_disturbance_penalty,
                "lambda_over_gamma_squared": hinf_exact["gamma_penalized"][
                    "max_eigenvalue_over_gamma_squared"
                ],
                "gamma_penalized_feasible": hinf_exact["gamma_penalized"]["feasible"],
            },
        },
    }
    arrays: dict[str, np.ndarray] = {
        "state_scales": np.asarray(scales),
        "time_block_scales": np.asarray(time_block_scales),
        "initial_states": np.asarray(states),
        "initial_state_weights": np.asarray(weights),
        "bellman_initial_K": np.asarray(bellman.K),
        "lqr_reference_K": np.asarray(K_ref),
        "lqr_clean_x": np.asarray(lqr_clean.x),
        "lqr_clean_x_hat": np.asarray(lqr_clean.x_hat),
        "lqr_clean_u": np.asarray(lqr_clean.u),
        "riccati_epsilon": np.asarray(riccati_epsilon),
    }
    for fit in fits:
        key = fit.label
        arrays[f"{key}_K"] = np.asarray(fit.K)
        arrays[f"{key}_clean_x"] = np.asarray(fit.clean_rollout.x)
        arrays[f"{key}_clean_x_hat"] = np.asarray(fit.clean_rollout.x_hat)
        arrays[f"{key}_clean_u"] = np.asarray(fit.clean_rollout.u)
        arrays[f"{key}_under_eps_x"] = np.asarray(fit.under_epsilon_rollout.x)
        arrays[f"{key}_under_eps_x_hat"] = np.asarray(fit.under_epsilon_rollout.x_hat)
        arrays[f"{key}_under_eps_u"] = np.asarray(fit.under_epsilon_rollout.u)
    arrays.update(coverage_arrays)
    return RolloutRecoveryResult(
        issue_id=ISSUE_ID,
        conditions=conditions,
        fits=tuple(fits),
        bellman_initialization_gain_relative_error=bellman.gain_relative_error,
        diagnostics=diagnostics,
        arrays=arrays,
    )


def _fit_summary(fit: RolloutRecoveryFit) -> dict[str, Any]:
    condition = fit.condition.__dict__.copy()
    if fit.condition.eigenspectrum_coverage is not None:
        condition["eigenspectrum_coverage"] = fit.condition.eigenspectrum_coverage.__dict__
    if fit.condition.observer_error_coverage is not None:
        condition["observer_error_coverage"] = fit.condition.observer_error_coverage.__dict__
    return {
        "label": fit.label,
        "condition": condition,
        "initialization": fit.initialization,
        "optimizer_status": fit.optimizer_status,
        "optimizer_success": fit.optimizer_success,
        "n_iterations": fit.n_iterations,
        "n_function_evaluations": fit.n_function_evaluations,
        "objective_initial": fit.objective_initial,
        "objective_final": fit.objective_final,
        "objective_reference": fit.objective_reference,
        "objective_zero": fit.objective_zero,
        "objective_ratio_to_reference": fit.objective_ratio_to_reference,
        "gain_relative_error": fit.gain_relative_error,
        "gradient_norm_initial": fit.gradient_norm_initial,
        "gradient_norm_final": fit.gradient_norm_final,
        "projected_gradient_norm_final": fit.projected_gradient_norm_final,
        "best_objective": fit.best_objective,
        "best_checkpoint_iteration": fit.best_checkpoint_iteration,
        "clean_cost": fit.clean_cost,
        "clean_rollout": _rollout_summary(fit.clean_rollout),
        "clean_action_mismatch_ratio": fit.clean_action_mismatch_ratio,
        "under_epsilon_cost": fit.under_epsilon_cost,
        "under_epsilon_cost_ratio_to_lqr": fit.under_epsilon_cost_ratio_to_lqr,
        "under_epsilon_rollout": _rollout_summary(fit.under_epsilon_rollout),
        "under_epsilon_action_mismatch_ratio": fit.under_epsilon_action_mismatch_ratio,
        "exact_l2_cost": fit.exact_l2_cost,
        "exact_l2_cost_ratio_to_lqr": fit.exact_l2_cost_ratio_to_lqr,
        "exact_l2_cost_ratio_to_hinf": fit.exact_l2_cost_ratio_to_hinf,
        "gamma_penalized_feasible": fit.gamma_penalized_feasible,
        "gamma_penalized_lambda_over_gamma_squared": (
            fit.gamma_penalized_lambda_over_gamma_squared
        ),
    }


def result_summary(
    result: RolloutRecoveryResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return JSON-serializable rollout-recovery summary."""

    return {
        "issue": result.issue_id,
        "umbrella": UMBRELLA_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="output_feedback_rollout_recovery",
        ),
        "related_issues": {
            "output_feedback_lane": OUTPUT_FEEDBACK_LANE_ISSUE_ID,
            "bellman_diagnostics": BELLMAN_DIAGNOSTIC_ISSUE_ID,
            "gamma_sweep": GAMMA_SWEEP_ISSUE_ID,
        },
        "scope": (
            "Clean output-feedback LQR rollout recovery: "
            "clean, stronger optimizer, whitening/scaling, and stronger optimizer "
            "plus whitening/scaling are objective-preserving; block/time "
            "preconditioning is objective-preserving and unscaled before reporting; "
            "Bellman-auxiliary guidance is noncanonical, scratch-only, and annealed "
            "off before final clean-rollout continuation."
        ),
        "non_goals": (
            "No weak Bellman/proximal anchor, no action/gain matching to the known "
            "controller, no coverage perturbations, no robust rollout, and no GRU "
            "training in this materialization."
        ),
        "bellman_initialization_gain_relative_error": (
            result.bellman_initialization_gain_relative_error
        ),
        "diagnostics": result.diagnostics,
        "fits": [_fit_summary(fit) for fit in result.fits],
    }


def _scale_initial_state_config(
    config: LinearTrainingConfig,
    factor: float,
) -> LinearTrainingConfig:
    """Scale synthetic basis/random initial-state coverage while preserving reach state."""

    return replace(
        config,
        basis_scale=config.basis_scale * factor,
        random_state_scale=config.random_state_scale * factor,
    )


def run_initial_state_variability_sweep(
    *,
    scale_factors: tuple[float, ...] = (0.0, 0.3, 1.0, 3.0),
    base_training_config: LinearTrainingConfig = LinearTrainingConfig(),
    conditions: tuple[RolloutRecoveryCondition, ...] = (STRONG_OPTIMIZER_WHITENED,),
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> dict[str, Any]:
    """Run the coupled basis/random synthetic initial-state scale sweep."""

    cells = []
    for factor in scale_factors:
        training_config = _scale_initial_state_config(base_training_config, factor)
        result = run_output_feedback_rollout_recovery(
            conditions=conditions,
            training_config=training_config,
            output_config=output_config,
        )
        summary = result_summary(result)
        cells.append(
            {
                "label": f"initial_state_scale_{factor:g}x",
                "scale_factor": factor,
                "basis_scale": training_config.basis_scale,
                "random_state_scale": training_config.random_state_scale,
                "n_random_states": training_config.n_random_states,
                "reach_weight": training_config.reach_weight,
                "diagnostics": summary["diagnostics"],
                "fits": summary["fits"],
            }
        )
    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "sweep": "coupled_initial_state_variability",
        "scale_factors": scale_factors,
        "base_training_config": base_training_config.__dict__,
        "interpretation": (
            "Coupled sweep over existing synthetic basis/random augmented-state coverage. "
            "The canonical reach state and its weight are preserved; physical start-jitter "
            "is intentionally separate. All rows use strong optimizer plus whitening."
        ),
        "cells": cells,
    }


def eigenspectrum_coverage_conditions(
    *,
    objectives: tuple[str, ...] = ("trajectory", "state"),
    modes: tuple[int, ...] = (1, 4),
    scales: tuple[float, ...] = (0.3, 1.0, 3.0),
    weight: float = 0.1,
) -> tuple[RolloutRecoveryCondition, ...]:
    """Return strong-optimizer-whitened coverage rows for the planned first sweep."""

    conditions = []
    for objective in objectives:
        for n_modes in modes:
            for scale in scales:
                label = f"strong_optimizer_whitened_eigen_{objective}_m{n_modes}_s{scale:g}"
                conditions.append(
                    strong_optimizer_whitened_with_coverage(
                        label,
                        EigenspectrumCoverageConfig(
                            n_modes=n_modes,
                            scale=scale,
                            weight=weight,
                            objective=objective,
                        ),
                    )
                )
    return tuple(conditions)


def observer_error_coverage_conditions(
    *,
    objectives: tuple[str, ...] = ("trajectory", "state"),
    modes: tuple[int, ...] = (1,),
    scales: tuple[float, ...] = (0.3, 1.0),
    weight: float = 0.1,
) -> tuple[RolloutRecoveryCondition, ...]:
    """Return strong-optimizer-whitened rows for observer-error coverage."""

    conditions = []
    for objective in objectives:
        for n_modes in modes:
            for scale in scales:
                label = (
                    f"strong_optimizer_whitened_observer_error_{objective}_m{n_modes}_s{scale:g}"
                )
                conditions.append(
                    strong_optimizer_whitened_with_observer_error_coverage(
                        label,
                        ObserverErrorCoverageConfig(
                            n_modes=n_modes,
                            scale=scale,
                            weight=weight,
                            objective=objective,
                        ),
                    )
                )
    return tuple(conditions)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked rollout-recovery note."""

    rows = [
        "| condition | init | objective ratio | gain rel err | clean cost | "
        "clean action mismatch | under-epsilon ratio | exact L2 ratio | "
        "lambda/gamma^2 | iters | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["fits"]:
        rows.append(
            "| "
            f"{row['condition']['label']} | "
            f"{row['initialization']} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['gain_relative_error']:.8g} | "
            f"{row['clean_cost']:.8g} | "
            f"{row['clean_action_mismatch_ratio']:.8g} | "
            f"{row['under_epsilon_cost_ratio_to_lqr']:.8g} | "
            f"{row['exact_l2_cost_ratio_to_lqr']:.8g} | "
            f"{row['gamma_penalized_lambda_over_gamma_squared']:.8g} | "
            f"{row['n_iterations']} | "
            f"{row['optimizer_status']} |"
        )
    diag = summary["diagnostics"]
    verdict = _rollout_recovery_verdict(summary)
    return f"""# Output-Feedback Rollout Recovery for the Linear Bridge

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

This note extends the first `7a459bb` matrix with two rescue conditions:
objective-preserving block/time preconditioning and noncanonical Bellman-objective
auxiliary guidance. Objective-preserving rows are run from scratch and from a
Bellman-initialized gain; the Bellman-auxiliary row is scratch-only.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

Bellman initialization gain relative error:
`{summary["bellman_initialization_gain_relative_error"]:.8g}`.

Output-feedback certificate gamma factor:
`{diag["gamma_factor"]:.8g}`.

Training-state scale condition:
`{diag["state_scales"]["condition"]:.8g}`.

Time/block preconditioner scale condition:
`{diag["time_block_scales"]["condition"]:.8g}`.

Bellman auxiliary schedule:
`{diag["bellman_auxiliary"]["schedules"]}`.

Initial-state ensemble effective rank:
`{diag["initial_state_ensemble"]["effective_rank_entropy"]:.8g}` entropy /
`{diag["initial_state_ensemble"]["effective_rank_participation"]:.8g}`
participation.

Reference clean LQR cost:
`{diag["lqr_clean"]["cost"]:.8g}`.

Reference LQR under Riccati epsilon cost:
`{diag["lqr_under_riccati_epsilon"]["cost"]:.8g}`.

Analytical exact L2 audit costs:
- LQR: `{diag["analytical_exact_audits"]["lqr"]["cost"]:.8g}`
- H-infinity: `{diag["analytical_exact_audits"]["hinf"]["cost"]:.8g}`

## Run Matrix

{"\n".join(rows)}

## Current Verdict

{verdict}
"""


def _rollout_recovery_verdict(summary: dict[str, Any]) -> str:
    """Return a compact qualitative verdict for the current matrix."""

    fits = summary["fits"]
    scratch_best = min(
        (row for row in fits if row["initialization"] == "scratch"),
        key=lambda row: row["gain_relative_error"],
    )
    bellman_best = min(
        (row for row in fits if row["initialization"] == "bellman_init"),
        key=lambda row: row["gain_relative_error"],
    )
    lines = [
        "This matrix separates discovery from preservation. From-scratch rows test "
        "whether clean rollout can discover the Riccati-like policy; Bellman-init "
        "rows test whether clean rollout preserves it once initialized there.",
        "",
        f"Best from-scratch gain error is `{scratch_best['gain_relative_error']:.8g}` "
        f"({scratch_best['label']}).",
        f"Best Bellman-initialized gain error is `{bellman_best['gain_relative_error']:.8g}` "
        f"({bellman_best['label']}).",
    ]
    if bellman_best["gain_relative_error"] < 1e-2:
        lines.append(
            "Bellman-initialized rollout preserves the analytical policy to a useful "
            "gain tolerance under at least one objective-preserving condition."
        )
    else:
        lines.append(
            "Bellman-initialized rollout does not preserve the analytical policy under "
            "these objective-preserving conditions, which points to clean-rollout "
            "underidentification rather than simple initialization failure."
        )
    if scratch_best["gain_relative_error"] < 1e-2:
        lines.append(
            "At least one from-scratch run also discovers the analytical policy, which "
            "would restore the clean rollout bridge for LQR."
        )
    else:
        lines.append(
            "No from-scratch run in this matrix discovers the analytical policy to the "
            "same tolerance. If Bellman-init preserves but scratch fails, discovery is "
            "the remaining problem; if both fail, the clean objective itself is not "
            "identifying the feedback law."
        )
    return "\n".join(lines)


def write_outputs(
    issue_id: str = ISSUE_ID,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Write tracked rollout-recovery note/manifest and bulk arrays."""

    result = run_output_feedback_rollout_recovery()
    summary = result_summary(result, discretization=discretization, lane=lane)
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "output_feedback_rollout_recovery")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Output-feedback rollout-recovery diagnostics for the Phase 3 linear bridge. "
            "See `notes/output_feedback_rollout_recovery.md`.\n",
            encoding="utf-8",
        )
    npz_path = artifact_dir / "output_feedback_rollout_recovery.npz"
    np.savez_compressed(npz_path, **result.arrays)
    summary["tracked_note"] = f"results/{issue_id}/notes/output_feedback_rollout_recovery.md"
    summary["tracked_manifest"] = (
        f"results/{issue_id}/notes/output_feedback_rollout_recovery_manifest.json"
    )
    summary["artifact_npz"] = (
        f"_artifacts/{issue_id}/output_feedback_rollout_recovery/{npz_path.name}"
    )
    summary["artifact_npz_keys"] = sorted(result.arrays.keys())
    note_path = notes_dir / "output_feedback_rollout_recovery.md"
    manifest_path = notes_dir / "output_feedback_rollout_recovery_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "DEFAULT_CONDITIONS",
    "EigenspectrumCoverageConfig",
    "ObserverErrorCoverageConfig",
    "RolloutRecoveryCondition",
    "RolloutRecoveryFit",
    "RolloutRecoveryResult",
    "STRONG_OPTIMIZER_WHITENED",
    "adamw_optimizer_whitened",
    "eigenspectrum_coverage_conditions",
    "observer_error_coverage_conditions",
    "render_markdown",
    "run_initial_state_variability_sweep",
    "result_summary",
    "run_output_feedback_rollout_recovery",
    "strong_optimizer_whitened_with_observer_error_coverage",
    "write_outputs",
]
