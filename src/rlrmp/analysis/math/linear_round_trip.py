"""Legacy scope note: `write_outputs` is a frozen writer/driver surface. The
math core in this module remains LIVE library code consumed by registered
recipes.

Phase 3 linear same-game round-trip checks for the C&S gate.

This module keeps the first Phase 3 implementation deliberately local and
analytical. It uses the Phase 0 game-card matrices directly, trains
time-varying full-state gains with gradient descent, and audits frozen gains
with the Phase 1 open-loop adversary search. Feedbax GraphSpec execution is
intentionally out of scope for this child issue.
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

from rlrmp.analysis.math.adversary_equivalence import (
    OPEN_LOOP_LEARNING_RATE,
    OPEN_LOOP_RESTARTS,
    OPEN_LOOP_STEP_SWEEP,
    OpenLoopOptimizationConfig,
    OpenLoopOptimizationResult,
    optimize_open_loop_epsilon,
    rollout_arrays_with_open_loop_epsilon,
)
from rlrmp.analysis.math.cs_game_card import (
    INIT_POS,
    PRIMARY_GAMMA_FACTOR,
    TARGET_POS,
    GameCardReference,
    GammaReference,
    materialize_reference,
    reference_summary,
    riccati_worst_case_policy,
    rollout_with_disturbance_policy,
)
from rlrmp.analysis.math.hinf_riccati import (
    ClosedLoopRollout,
    CostSchedule,
    PlantLinearization,
    make_reach_initial_state,
    simulate_closed_loop,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.analysis.math import require_jax_x64
from rlrmp.paths import REPO_ROOT, mkdir_p

ISSUE_ID = "6f5c79e"
UMBRELLA_ID = "43e8728"
GAME_CARD_ISSUE_ID = "cb98e58"
ADVERSARY_EQUIVALENCE_ISSUE_ID = "a7dad8a"


@dataclass(frozen=True)
class LinearOptimizationConfig:
    """Gradient-training configuration for one linear-controller fit."""

    n_steps: int = 2500
    learning_rate: float = 0.3
    seed: int = 0
    basis_scale: float = 0.01
    random_state_scale: float = 0.02
    n_random_states: int = 64
    reach_weight: float = 10.0


@dataclass(frozen=True)
class LinearTrainingResult:
    """Result of one gradient-based linear-controller fit."""

    label: str
    config: LinearOptimizationConfig
    K: Float[Array, "T m_u n"]
    best_objective: float
    final_objective: float
    reference_objective: float
    zero_objective: float
    gain_relative_error: float
    canonical_rollout: ClosedLoopRollout
    optimizer_status: str
    n_iterations: int
    n_function_evaluations: int


@dataclass(frozen=True)
class TeacherFitConfig:
    """Gradient teacher-fit configuration for representational checks."""

    n_steps: int = 2000
    learning_rate: float = 0.1
    target_scale: float = 1000.0


@dataclass(frozen=True)
class TeacherFitResult:
    """Result of fitting a gain tensor to an analytical teacher by descent."""

    label: str
    config: TeacherFitConfig
    K: Float[Array, "T m_u n"]
    best_loss: float
    final_loss: float
    gain_relative_error: float
    canonical_rollout: ClosedLoopRollout


@dataclass(frozen=True)
class ControllerAudit:
    """Frozen-controller metrics against clean and held-out adversary rollouts."""

    label: str
    clean_rollout: ClosedLoopRollout
    clean_cost: float
    delta_v_percent_vs_lqr_reference: float
    gain_relative_error: float
    heldout: OpenLoopOptimizationResult


@dataclass(frozen=True)
class Phase3LinearRoundTripResult:
    """Complete Phase 3 local analytical result bundle."""

    reference: GameCardReference
    gamma_ref: GammaReference
    lqr_training: LinearTrainingResult
    lqr_quasi_newton_training: LinearTrainingResult
    lqr_teacher_fit: TeacherFitResult
    hinf_teacher_fit: TeacherFitResult
    audits: tuple[ControllerAudit, ...]


def canonical_initial_state(plant: PlantLinearization) -> Float[Array, " n"]:
    """Return the Phase 0 reach initial state."""

    return make_reach_initial_state(plant, init_pos=INIT_POS, target_pos=TARGET_POS)


def ensemble_initial_states(
    plant: PlantLinearization,
    config: LinearOptimizationConfig,
) -> tuple[Float[Array, "batch n"], Float[Array, " batch"]]:
    """Build a full-rank deterministic state ensemble for gain-level training.

    A single reach trajectory leaves most gain columns underdetermined. The
    basis states make the clean LQR objective depend on every state coordinate;
    the canonical reach remains overweighted because it is the scientific
    trajectory used by the game card.
    """

    x0 = canonical_initial_state(plant)
    basis = jnp.eye(plant.n, dtype=jnp.float64)
    key = jr.PRNGKey(config.seed)
    random_states = jr.normal(
        key,
        (config.n_random_states, plant.n),
        dtype=jnp.float64,
    )
    states = jnp.concatenate(
        [
            x0[None],
            config.basis_scale * basis,
            -config.basis_scale * basis,
            config.random_state_scale * random_states,
        ],
        axis=0,
    )
    weights = jnp.ones((states.shape[0],), dtype=jnp.float64)
    weights = weights.at[0].set(config.reach_weight)
    return states, weights


def rollout_task_cost(
    schedule: CostSchedule,
    x: Float[Array, "T_plus_1 n"],
    u: Float[Array, "T m_u"],
) -> Float[Array, ""]:
    """Return the quadratic task cost without disturbance penalty."""

    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    return jnp.sum(state_terms) + jnp.sum(control_terms) + terminal


def ensemble_clean_objective(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
) -> Float[Array, ""]:
    """Mean weighted clean rollout cost over an initial-state ensemble."""

    epsilon = jnp.zeros((schedule.T, plant.m_w), dtype=jnp.float64)

    def one_cost(x0: Float[Array, " n"]) -> Float[Array, ""]:
        x, u = rollout_arrays_with_open_loop_epsilon(plant, K, x0, epsilon)
        return rollout_task_cost(schedule, x, u)

    costs = jax.vmap(one_cost)(states)
    return jnp.mean(costs * weights)


def train_lqr_gradient_controller(
    reference: GameCardReference,
    config: LinearOptimizationConfig = LinearOptimizationConfig(),
) -> LinearTrainingResult:
    """Fit a clean LQR linear controller with gradient descent.

    This is intentionally a local optimizer check, not a Riccati solve. It is
    expected to expose whether naive gradient training recovers the analytical
    gain under the exact game-card matrices.
    """

    plant = reference.plant
    schedule = reference.schedule
    states, weights = ensemble_initial_states(plant, config)
    K_ref = reference.lqr_solution.K
    K = jnp.zeros_like(K_ref)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(K)

    def objective(gains: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        return ensemble_clean_objective(plant, schedule, gains, states, weights)

    value_and_grad = jax.value_and_grad(objective)

    @jax.jit
    def run_descent(
        K: Float[Array, "T m_u n"],
        opt_state: optax.OptState,
    ) -> tuple[
        Float[Array, "T m_u n"], Float[Array, "T m_u n"], Float[Array, ""], Float[Array, ""]
    ]:
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

        (K, _opt_state, best_K, best_value), values = jax.lax.scan(
            step,
            (K, opt_state, best_K, best_value),
            None,
            length=config.n_steps,
        )
        return K, best_K, best_value, values[-1]

    _final_K, best_K, best_value, final_value = run_descent(K, opt_state)
    x0 = canonical_initial_state(plant)
    rollout = simulate_closed_loop(plant, best_K, x0, target_pos=TARGET_POS)
    reference_objective = float(objective(K_ref))
    zero_objective = float(objective(jnp.zeros_like(K_ref)))
    gain_error = float(jnp.linalg.norm(best_K - K_ref) / jnp.linalg.norm(K_ref))
    return LinearTrainingResult(
        label="adam_lqr_fit",
        config=config,
        K=best_K,
        best_objective=float(best_value),
        final_objective=float(final_value),
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        gain_relative_error=gain_error,
        canonical_rollout=rollout,
        optimizer_status="completed",
        n_iterations=config.n_steps,
        n_function_evaluations=config.n_steps,
    )


def train_lqr_quasi_newton_controller(
    reference: GameCardReference,
    config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=200),
    *,
    initial_K: Float[Array, "T m_u n"] | None = None,
    label: str = "lbfgsb_lqr_fit",
) -> LinearTrainingResult:
    """Fit a clean LQR linear controller with SciPy L-BFGS-B.

    This is the direct quasi-Newton comparator for the Adam gate. It uses the
    same full-rank objective and can either start from zero or refine an
    existing objective-trained gain tensor.
    """

    plant = reference.plant
    schedule = reference.schedule
    states, weights = ensemble_initial_states(plant, config)
    K_ref = reference.lqr_solution.K
    shape = K_ref.shape
    if initial_K is None:
        theta0 = np.zeros(int(np.prod(shape)), dtype=np.float64)
    else:
        theta0 = np.asarray(initial_K, dtype=np.float64).reshape(-1)

    def objective(gains: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        return ensemble_clean_objective(plant, schedule, gains, states, weights)

    @jax.jit
    def value_and_grad_flat(theta: Float[Array, " flat"]) -> tuple[Float[Array, ""], Array]:
        K = theta.reshape(shape)
        value, grads = jax.value_and_grad(objective)(K)
        return value, grads.reshape(-1)

    def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grads = value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
        return float(value), np.asarray(grads, dtype=np.float64)

    scipy_result = scipy_opt.minimize(
        scipy_value_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": config.n_steps,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    K = jnp.asarray(scipy_result.x, dtype=jnp.float64).reshape(shape)
    x0 = canonical_initial_state(plant)
    rollout = simulate_closed_loop(plant, K, x0, target_pos=TARGET_POS)
    reference_objective = float(objective(K_ref))
    zero_objective = float(objective(jnp.zeros_like(K_ref)))
    final_objective = float(objective(K))
    gain_error = float(jnp.linalg.norm(K - K_ref) / jnp.linalg.norm(K_ref))
    return LinearTrainingResult(
        label=label,
        config=config,
        K=K,
        best_objective=float(min(scipy_result.fun, final_objective)),
        final_objective=final_objective,
        reference_objective=reference_objective,
        zero_objective=zero_objective,
        gain_relative_error=gain_error,
        canonical_rollout=rollout,
        optimizer_status=str(scipy_result.message),
        n_iterations=int(scipy_result.nit),
        n_function_evaluations=int(scipy_result.nfev),
    )


def train_teacher_gain(
    reference: GameCardReference,
    *,
    label: str,
    K_target: Float[Array, "T m_u n"],
    config: TeacherFitConfig = TeacherFitConfig(),
) -> TeacherFitResult:
    """Fit a gain tensor to an analytical teacher by gradient descent.

    This is a representational check, not the minimax objective gate. It tells
    us whether the local parameterization/metric stack can hold the Riccati
    gains when optimization is not the bottleneck.
    """

    target = K_target.astype(jnp.float64) / config.target_scale
    Z = jnp.zeros_like(target)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(Z)

    def objective(z: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        return jnp.mean((z - target) ** 2)

    value_and_grad = jax.value_and_grad(objective)

    @jax.jit
    def run_descent(
        Z: Float[Array, "T m_u n"],
        opt_state: optax.OptState,
    ) -> tuple[
        Float[Array, "T m_u n"], Float[Array, "T m_u n"], Float[Array, ""], Float[Array, ""]
    ]:
        best_Z = Z
        best_value = objective(Z)

        def step(carry, _):
            Z, opt_state, best_Z, best_value = carry
            _value, grads = value_and_grad(Z)
            updates, opt_state = optimizer.update(grads, opt_state, Z)
            Z = optax.apply_updates(Z, updates)
            value = objective(Z)
            improved = value < best_value
            best_Z = jnp.where(improved, Z, best_Z)
            best_value = jnp.minimum(best_value, value)
            return (Z, opt_state, best_Z, best_value), value

        (Z, _opt_state, best_Z, best_value), values = jax.lax.scan(
            step,
            (Z, opt_state, best_Z, best_value),
            None,
            length=config.n_steps,
        )
        return Z, best_Z, best_value, values[-1]

    _final_Z, best_Z, best_value, final_value = run_descent(Z, opt_state)
    K = best_Z * config.target_scale
    x0 = canonical_initial_state(reference.plant)
    rollout = simulate_closed_loop(reference.plant, K, x0, target_pos=TARGET_POS)
    return TeacherFitResult(
        label=label,
        config=config,
        K=K,
        best_loss=float(best_value),
        final_loss=float(final_value),
        gain_relative_error=float(jnp.linalg.norm(K - K_target) / jnp.linalg.norm(K_target)),
        canonical_rollout=rollout,
    )


def audit_controller(
    *,
    label: str,
    reference: GameCardReference,
    K: Float[Array, "T m_u n"],
    K_reference: Float[Array, "T m_u n"],
    budget: float,
    gamma: float,
    step_sweep: tuple[int, ...] = OPEN_LOOP_STEP_SWEEP,
    n_restarts: int = OPEN_LOOP_RESTARTS,
    learning_rate: float = OPEN_LOOP_LEARNING_RATE,
    seed: int = 0,
    initial_candidates: tuple[Float[Array, "T m_w"], ...] = (),
) -> ControllerAudit:
    """Audit a frozen controller with clean metrics and held-out adversary search."""

    plant = reference.plant
    schedule = reference.schedule
    x0 = canonical_initial_state(plant)
    clean = simulate_closed_loop(plant, K, x0, target_pos=TARGET_POS)
    clean_cost = float(rollout_task_cost(schedule, clean.x, clean.u))
    lqr_peak = reference.lqr_rollout.peak_forward_velocity
    delta_v = 100.0 * (clean.peak_forward_velocity - lqr_peak) / lqr_peak

    heldout_results = []
    for idx, n_steps in enumerate(step_sweep):
        heldout_results.append(
            optimize_open_loop_epsilon(
                plant,
                schedule,
                K,
                x0,
                config=OpenLoopOptimizationConfig(
                    budget=budget,
                    n_steps=int(n_steps),
                    n_restarts=n_restarts,
                    learning_rate=learning_rate,
                    seed=seed + 1009 * idx,
                ),
                gamma=gamma,
                initial_candidates=initial_candidates,
            )
        )
    heldout = max(
        heldout_results,
        key=lambda result: result.cost.total_without_disturbance_penalty,
    )
    return ControllerAudit(
        label=label,
        clean_rollout=clean,
        clean_cost=clean_cost,
        delta_v_percent_vs_lqr_reference=float(delta_v),
        gain_relative_error=float(jnp.linalg.norm(K - K_reference) / jnp.linalg.norm(K_reference)),
        heldout=heldout,
    )


def run_phase3_linear_round_trip(
    *,
    training_config: LinearOptimizationConfig = LinearOptimizationConfig(),
    quasi_newton_config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=500),
    heldout_step_sweep: tuple[int, ...] = (50, 200),
    heldout_restarts: int = 4,
) -> Phase3LinearRoundTripResult:
    """Run the local Phase 3 analytical round-trip checks."""

    require_jax_x64("linear round-trip analysis")
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    lqr_training = train_lqr_gradient_controller(reference, training_config)
    lqr_quasi_newton = train_lqr_quasi_newton_controller(
        reference,
        quasi_newton_config,
        initial_K=lqr_training.K,
        label="lbfgsb_after_adam_lqr_fit",
    )
    lqr_teacher = train_teacher_gain(
        reference,
        label="teacher_lqr_fit",
        K_target=reference.lqr_solution.K,
    )
    hinf_teacher = train_teacher_gain(
        reference,
        label="teacher_hinf_fit",
        K_target=gamma_ref.solution.K,
    )
    F = riccati_worst_case_policy(reference.plant, gamma_ref.solution)
    x0 = canonical_initial_state(reference.plant)
    riccati_worst = rollout_with_disturbance_policy(reference.plant, gamma_ref.solution.K, F, x0)
    budget = float(jnp.sum(riccati_worst.epsilon**2))

    audits = (
        audit_controller(
            label="analytical_lqr_reference",
            reference=reference,
            K=reference.lqr_solution.K,
            K_reference=reference.lqr_solution.K,
            budget=budget,
            gamma=gamma_ref.gamma,
            step_sweep=heldout_step_sweep,
            n_restarts=heldout_restarts,
            seed=11,
        ),
        audit_controller(
            label=lqr_training.label,
            reference=reference,
            K=lqr_training.K,
            K_reference=reference.lqr_solution.K,
            budget=budget,
            gamma=gamma_ref.gamma,
            step_sweep=heldout_step_sweep,
            n_restarts=heldout_restarts,
            seed=23,
        ),
        audit_controller(
            label=lqr_quasi_newton.label,
            reference=reference,
            K=lqr_quasi_newton.K,
            K_reference=reference.lqr_solution.K,
            budget=budget,
            gamma=gamma_ref.gamma,
            step_sweep=heldout_step_sweep,
            n_restarts=heldout_restarts,
            seed=31,
        ),
        audit_controller(
            label="teacher_lqr_fit",
            reference=reference,
            K=lqr_teacher.K,
            K_reference=reference.lqr_solution.K,
            budget=budget,
            gamma=gamma_ref.gamma,
            step_sweep=heldout_step_sweep,
            n_restarts=heldout_restarts,
            seed=29,
        ),
        audit_controller(
            label="analytical_hinf_reference",
            reference=reference,
            K=gamma_ref.solution.K,
            K_reference=gamma_ref.solution.K,
            budget=budget,
            gamma=gamma_ref.gamma,
            step_sweep=heldout_step_sweep,
            n_restarts=heldout_restarts,
            seed=37,
            initial_candidates=(riccati_worst.epsilon,),
        ),
        audit_controller(
            label="teacher_hinf_fit",
            reference=reference,
            K=hinf_teacher.K,
            K_reference=gamma_ref.solution.K,
            budget=budget,
            gamma=gamma_ref.gamma,
            step_sweep=heldout_step_sweep,
            n_restarts=heldout_restarts,
            seed=41,
            initial_candidates=(riccati_worst.epsilon,),
        ),
    )
    return Phase3LinearRoundTripResult(
        reference=reference,
        gamma_ref=gamma_ref,
        lqr_training=lqr_training,
        lqr_quasi_newton_training=lqr_quasi_newton,
        lqr_teacher_fit=lqr_teacher,
        hinf_teacher_fit=hinf_teacher,
        audits=audits,
    )


def _heldout_summary(heldout: OpenLoopOptimizationResult) -> dict[str, Any]:
    return {
        "n_steps": heldout.config.n_steps,
        "n_restarts": heldout.config.n_restarts,
        "learning_rate": heldout.config.learning_rate,
        "seed": heldout.config.seed,
        "best_restart_idx": heldout.best_restart_idx,
        "total_cost": heldout.cost.total_without_disturbance_penalty,
        "disturbance_energy": heldout.cost.disturbance_energy,
        "h_infinity_objective": heldout.cost.h_infinity_objective,
        "peak_forward_velocity": heldout.rollout.peak_forward_velocity,
        "time_to_peak_step": heldout.rollout.peak_forward_velocity_idx,
        "terminal_position_error_m": heldout.rollout.terminal_position_error,
        "initial_objectives": list(heldout.initial_objectives),
        "final_objectives": list(heldout.final_objectives),
        "best_objectives": list(heldout.best_objectives),
        "final_energies": list(heldout.final_energies),
        "best_energies": list(heldout.best_energies),
    }


def _linear_training_summary(
    training: LinearTrainingResult,
    reference: GameCardReference,
) -> dict[str, Any]:
    return {
        "label": training.label,
        "best_objective": training.best_objective,
        "final_objective": training.final_objective,
        "reference_objective": training.reference_objective,
        "zero_objective": training.zero_objective,
        "objective_ratio_to_reference": training.best_objective / training.reference_objective,
        "gain_relative_error": training.gain_relative_error,
        "canonical_clean_cost": float(
            rollout_task_cost(
                reference.schedule,
                training.canonical_rollout.x,
                training.canonical_rollout.u,
            )
        ),
        "canonical_peak_forward_velocity": training.canonical_rollout.peak_forward_velocity,
        "canonical_time_to_peak_step": training.canonical_rollout.peak_forward_velocity_idx,
        "canonical_terminal_position_error_m": training.canonical_rollout.terminal_position_error,
        "optimizer_status": training.optimizer_status,
        "n_iterations": training.n_iterations,
        "n_function_evaluations": training.n_function_evaluations,
    }


def result_summary(
    result: Phase3LinearRoundTripResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return a JSON-serializable Phase 3 summary."""

    reference = result.reference
    gamma_ref = result.gamma_ref
    objective_trainings = (result.lqr_training, result.lqr_quasi_newton_training)
    best_objective_training = min(
        objective_trainings,
        key=lambda training: training.best_objective / training.reference_objective,
    )
    objective_ratio_pass = (
        best_objective_training.best_objective / best_objective_training.reference_objective <= 1.01
    )
    gain_recovery_pass = best_objective_training.gain_relative_error <= 0.05
    if objective_ratio_pass and gain_recovery_pass:
        phase3_status = "passed"
    elif objective_ratio_pass:
        phase3_status = "blocked_on_gain_recovery"
    else:
        phase3_status = "blocked_on_optimizer"
    teacher_pass = (
        result.lqr_teacher_fit.gain_relative_error <= 0.01
        and result.hinf_teacher_fit.gain_relative_error <= 0.01
    )
    audits = []
    for audit in result.audits:
        audits.append(
            {
                "label": audit.label,
                "clean_cost": audit.clean_cost,
                "delta_v_percent_vs_lqr_reference": audit.delta_v_percent_vs_lqr_reference,
                "gain_relative_error": audit.gain_relative_error,
                "peak_forward_velocity": audit.clean_rollout.peak_forward_velocity,
                "time_to_peak_step": audit.clean_rollout.peak_forward_velocity_idx,
                "terminal_position_error_m": audit.clean_rollout.terminal_position_error,
                "control_effort": audit.clean_rollout.control_effort,
                "heldout": _heldout_summary(audit.heldout),
            }
        )

    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "game_card_issue": GAME_CARD_ISSUE_ID,
        "adversary_equivalence_issue": ADVERSARY_EQUIVALENCE_ISSUE_ID,
        "regeneration_command": "PYTHONPATH=src python scripts/materialize_linear_round_trip.py",
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="linear_round_trip",
        ),
        "graphspec_execution_conversion_out_of_scope": True,
        "matrix_generalization_out_of_scope": True,
        "phase3_status": phase3_status,
        "interpretation": (
            "The local objective-trained LQR controller met the predeclared pass band."
            if phase3_status == "passed"
            else (
                "Adam-warm-started L-BFGS-B recovered the clean LQR objective/behavior "
                "gate, but the raw gain tensor remains far from the analytical Riccati "
                "gain. Phase 3 is therefore blocked on whether gain-relative-error is "
                "the right certificate or whether a structured/identifiable linear "
                "policy objective is needed."
                if phase3_status == "blocked_on_gain_recovery"
                else (
                    "The analytical replay/audit and teacher-fit representational paths "
                    "are in place, but the tested objective-gradient optimizers for the "
                    "clean LQR gain did not meet the predeclared gain/objective pass band "
                    "from zero initialization."
                )
            )
        ),
        "teacher_fit_status": "passed" if teacher_pass else "failed",
        "predeclared_tolerances": {
            "lqr_objective_ratio": 1.01,
            "gain_relative_error": 0.05,
            "delta_v_absolute_percentage_points": 1.0,
            "terminal_error_m": 1e-4,
        },
        "training_config": result.lqr_training.config.__dict__,
        "quasi_newton_config": result.lqr_quasi_newton_training.config.__dict__,
        "best_objective_training": best_objective_training.label,
        "objective_trainings": {
            result.lqr_training.label: _linear_training_summary(result.lqr_training, reference),
            result.lqr_quasi_newton_training.label: _linear_training_summary(
                result.lqr_quasi_newton_training,
                reference,
            ),
        },
        "teacher_fits": {
            result.lqr_teacher_fit.label: {
                "best_loss": result.lqr_teacher_fit.best_loss,
                "final_loss": result.lqr_teacher_fit.final_loss,
                "gain_relative_error": result.lqr_teacher_fit.gain_relative_error,
                "canonical_clean_cost": float(
                    rollout_task_cost(
                        reference.schedule,
                        result.lqr_teacher_fit.canonical_rollout.x,
                        result.lqr_teacher_fit.canonical_rollout.u,
                    )
                ),
                "canonical_peak_forward_velocity": (
                    result.lqr_teacher_fit.canonical_rollout.peak_forward_velocity
                ),
                "canonical_terminal_position_error_m": (
                    result.lqr_teacher_fit.canonical_rollout.terminal_position_error
                ),
            },
            result.hinf_teacher_fit.label: {
                "best_loss": result.hinf_teacher_fit.best_loss,
                "final_loss": result.hinf_teacher_fit.final_loss,
                "gain_relative_error": result.hinf_teacher_fit.gain_relative_error,
                "canonical_clean_cost": float(
                    rollout_task_cost(
                        reference.schedule,
                        result.hinf_teacher_fit.canonical_rollout.x,
                        result.hinf_teacher_fit.canonical_rollout.u,
                    )
                ),
                "canonical_peak_forward_velocity": (
                    result.hinf_teacher_fit.canonical_rollout.peak_forward_velocity
                ),
                "canonical_terminal_position_error_m": (
                    result.hinf_teacher_fit.canonical_rollout.terminal_position_error
                ),
            },
        },
        "reference": reference_summary(reference),
        "gamma_factor": gamma_ref.factor,
        "gamma": gamma_ref.gamma,
        "audits": audits,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked Phase 3 note."""

    objective_rows = [
        "| optimizer | objective ratio | gain rel err | clean cost | peak forward v | terminal err | iterations | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for training in summary["objective_trainings"].values():
        objective_rows.append(
            "| "
            f"`{training['label']}` | "
            f"{training['objective_ratio_to_reference']:.8g} | "
            f"{training['gain_relative_error']:.8g} | "
            f"{training['canonical_clean_cost']:.8g} | "
            f"{training['canonical_peak_forward_velocity']:.8g} | "
            f"{training['canonical_terminal_position_error_m']:.8g} | "
            f"{training['n_iterations']} | "
            f"{training['optimizer_status']} |"
        )
    audit_rows = [
        "| controller | clean cost | Delta-v vs LQR | gain rel err | held-out cost | held-out steps | terminal err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in summary["audits"]:
        audit_rows.append(
            "| "
            f"`{audit['label']}` | "
            f"{audit['clean_cost']:.8g} | "
            f"{audit['delta_v_percent_vs_lqr_reference']:+.4f}% | "
            f"{audit['gain_relative_error']:.6g} | "
            f"{audit['heldout']['total_cost']:.8g} | "
            f"{audit['heldout']['n_steps']} | "
            f"{audit['terminal_position_error_m']:.6g} |"
        )

    return f"""# Phase 3 Linear Same-Game Round Trip

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

This note records the first local analytical Phase 3 certificate attempt for
the cs2019-to-RNN game-equivalence programme. It intentionally does not perform
the Feedbax GraphSpec execution conversion or the full `63cec06` matrix-analysis
generalization; those remain the next workup after the local certificate.

## Fixed Game

- Game-card issue: `{summary["game_card_issue"]}`.
- Adversary-equivalence issue: `{summary["adversary_equivalence_issue"]}`.
- State: 48D delay-augmented C&S state.
- Disturbance: 8D epsilon through `B_w = [I_8; 0]`.
- Cost: C&S 60-step `(t/T)^6` schedule from Phase 0.
- Primary robust target: `gamma = 1.05 * gamma_star`.

## Local Objective-Training Result

Status: `{summary["phase3_status"]}`.

{summary["interpretation"]}

The clean LQR trainers optimize time-varying full-state gains `K[t]` over a
deterministic full-rank initial-state ensemble. The full-rank ensemble is
necessary because a single reach trajectory can match behavior while leaving
many gain columns underdetermined.

Best objective-trained controller: `{summary["best_objective_training"]}`.

{"\n".join(objective_rows)}

## Teacher-Fit Representational Check

Status: `{summary["teacher_fit_status"]}`.

The teacher-fit check trains the same gain tensor shape by gradient descent
against the analytical gain tensor directly. This is not the minimax objective
gate; it isolates representation and metric plumbing from objective-optimizer
quality.

| teacher fit | gain rel err | clean cost | peak forward v | terminal err |
|---|---:|---:|---:|---:|
| `teacher_lqr_fit` | {summary["teacher_fits"]["teacher_lqr_fit"]["gain_relative_error"]:.8g} | {summary["teacher_fits"]["teacher_lqr_fit"]["canonical_clean_cost"]:.8g} | {summary["teacher_fits"]["teacher_lqr_fit"]["canonical_peak_forward_velocity"]:.8g} | {summary["teacher_fits"]["teacher_lqr_fit"]["canonical_terminal_position_error_m"]:.8g} |
| `teacher_hinf_fit` | {summary["teacher_fits"]["teacher_hinf_fit"]["gain_relative_error"]:.8g} | {summary["teacher_fits"]["teacher_hinf_fit"]["canonical_clean_cost"]:.8g} | {summary["teacher_fits"]["teacher_hinf_fit"]["canonical_peak_forward_velocity"]:.8g} | {summary["teacher_fits"]["teacher_hinf_fit"]["canonical_terminal_position_error_m"]:.8g} |

## Frozen-Controller Audits

{"\n".join(audit_rows)}

Held-out adversary audits use independent projected open-loop epsilon searches
with fresh seeds. Each inner search retains the best-seen objective, not only
the final endpoint, following the Phase 1 `89891ab`/`a7dad8a` lesson.

## Interpretation

This pass should not be treated as a successful Phase 3 exit certificate unless
`phase3_status` is `passed`. A failed clean LQR gain recovery means the local
linear certificate still needs work before GRU same-game interpretation.

The important positive result is narrower: the analytical replay, metric, and
held-out adversary audit surfaces now exist locally and are tied to the exact
Phase 0-2 game. Adam-warm-started L-BFGS-B can recover the clean objective and
canonical behavior, but not the raw analytical gain tensor under the current
full-rank ensemble certificate. The next decision is whether to replace the raw
gain-error gate with a behaviorally equivalent certificate or introduce a more
structured identifiable linear-policy optimization method before attempting the
GRU phase.
"""


def _npz_arrays(result: Phase3LinearRoundTripResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "lqr_reference_K": np.asarray(result.reference.lqr_solution.K),
        "hinf_reference_K": np.asarray(result.gamma_ref.solution.K),
        "adam_lqr_K": np.asarray(result.lqr_training.K),
        "lbfgsb_after_adam_lqr_K": np.asarray(result.lqr_quasi_newton_training.K),
        "teacher_lqr_K": np.asarray(result.lqr_teacher_fit.K),
        "teacher_hinf_K": np.asarray(result.hinf_teacher_fit.K),
        "adam_lqr_x": np.asarray(result.lqr_training.canonical_rollout.x),
        "adam_lqr_u": np.asarray(result.lqr_training.canonical_rollout.u),
        "lbfgsb_after_adam_lqr_x": np.asarray(result.lqr_quasi_newton_training.canonical_rollout.x),
        "lbfgsb_after_adam_lqr_u": np.asarray(result.lqr_quasi_newton_training.canonical_rollout.u),
        "teacher_lqr_x": np.asarray(result.lqr_teacher_fit.canonical_rollout.x),
        "teacher_lqr_u": np.asarray(result.lqr_teacher_fit.canonical_rollout.u),
        "teacher_hinf_x": np.asarray(result.hinf_teacher_fit.canonical_rollout.x),
        "teacher_hinf_u": np.asarray(result.hinf_teacher_fit.canonical_rollout.u),
    }
    for audit in result.audits:
        arrays[f"{audit.label}_clean_x"] = np.asarray(audit.clean_rollout.x)
        arrays[f"{audit.label}_clean_u"] = np.asarray(audit.clean_rollout.u)
        arrays[f"{audit.label}_heldout_x"] = np.asarray(audit.heldout.rollout.x)
        arrays[f"{audit.label}_heldout_u"] = np.asarray(audit.heldout.rollout.u)
        arrays[f"{audit.label}_heldout_epsilon"] = np.asarray(audit.heldout.epsilon)
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

    require_jax_x64("linear round-trip materialization")
    result = run_phase3_linear_round_trip()
    summary = result_summary(result, discretization=discretization, lane=lane)
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "linear_round_trip")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Phase 3 linear same-game round-trip artifacts for the cs2019-to-RNN "
            "game-equivalence programme. See `notes/linear_round_trip.md` for "
            "the tracked local certificate attempt.\n",
            encoding="utf-8",
        )

    npz_path = artifact_dir / "linear_round_trip.npz"
    np.savez_compressed(npz_path, **_npz_arrays(result))
    summary["artifact_npz"] = f"_artifacts/{issue_id}/linear_round_trip/{npz_path.name}"
    summary["artifact_npz_keys"] = sorted(_npz_arrays(result).keys())
    summary["tracked_note"] = f"results/{issue_id}/notes/linear_round_trip.md"
    summary["tracked_manifest"] = f"results/{issue_id}/notes/linear_round_trip_manifest.json"

    note_path = notes_dir / "linear_round_trip.md"
    manifest_path = notes_dir / "linear_round_trip_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ADVERSARY_EQUIVALENCE_ISSUE_ID",
    "GAME_CARD_ISSUE_ID",
    "ISSUE_ID",
    "UMBRELLA_ID",
    "ControllerAudit",
    "LinearOptimizationConfig",
    "LinearTrainingResult",
    "Phase3LinearRoundTripResult",
    "TeacherFitConfig",
    "TeacherFitResult",
    "audit_controller",
    "canonical_initial_state",
    "ensemble_clean_objective",
    "ensemble_initial_states",
    "render_markdown",
    "result_summary",
    "rollout_task_cost",
    "run_phase3_linear_round_trip",
    "train_lqr_gradient_controller",
    "train_lqr_quasi_newton_controller",
    "write_outputs",
]
