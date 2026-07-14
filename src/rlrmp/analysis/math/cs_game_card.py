"""Materialize the C&S released-code-aligned H-infinity analytical game card.

This module is intentionally narrow: it builds the canonical Phase 0 game for
issue ``cb98e58`` / umbrella ``43e8728`` and computes the analytical reference
objects that downstream training code must match. It does not depend on
feedbax runtime state, so tests can protect the mathematical contract before
any simulator-side parity work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.math.hinf_riccati import (
    ClosedLoopRollout,
    CostSchedule,
    Discretization,
    PlantLinearization,
    RiccatiSolution,
    apply_delay_distribution_to_schedule,
    cs_eq15_cost_schedule,
    cs_faithful_pointmass,
    find_gamma_star,
    make_reach_initial_state,
    simulate_closed_loop,
    solve_hinf_riccati,
    solve_lqr,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
ISSUE_ID = "cb98e58"
UMBRELLA_ID = "43e8728"
_ANALYSIS_PRESET = load_analysis_parameter_preset("cs_game_card").parameters
# Full-state deterministic C&S speed-matching reference. This is not the
# default for output-feedback robustness certificates; see
# OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR below.
PRIMARY_GAMMA_FACTOR = float(_ANALYSIS_PRESET["primary_gamma_factor"])
# Working output-feedback robustness target selected from the gamma sweep on
# 97604a8. Keep output-feedback Phase 1/3 diagnostics tied to this named value
# so rerunning the sweep requires updating one contract instead of silently
# falling back to the full-state speed-matching gamma.
OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR = float(
    _ANALYSIS_PRESET["output_feedback_certificate_gamma_factor"]
)
OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID = "97604a8"
DIAGNOSTIC_GAMMA_FACTOR = float(_ANALYSIS_PRESET["diagnostic_gamma_factor"])
DEFAULT_GAMMA_FACTORS = tuple(_ANALYSIS_PRESET["default_gamma_factors"])
INIT_POS = np.asarray(_ANALYSIS_PRESET["initial_position_m"], dtype=np.float64)
TARGET_POS = np.asarray(_ANALYSIS_PRESET["target_position_m"], dtype=np.float64)


@dataclass(frozen=True)
class CostBreakdown:
    """Scalar quadratic cost decomposition for one rollout."""

    state_stage: float
    control_stage: float
    terminal_state: float
    total_without_disturbance_penalty: float
    disturbance_energy: float = 0.0
    h_infinity_objective: float | None = None


@dataclass(frozen=True)
class WorstCaseRollout:
    """Closed-loop rollout under the Riccati state-dependent disturbance."""

    x: Float[Array, "T_plus_1 n"]
    u: Float[Array, "T m_u"]
    epsilon: Float[Array, "T m_w"]


@dataclass(frozen=True)
class GammaReference:
    """Analytical reference at one gamma multiplier."""

    factor: float
    gamma: float
    solution: RiccatiSolution
    nominal_rollout: ClosedLoopRollout
    worst_case_policy: Float[Array, "T m_w n"]
    epsilon_on_nominal: Float[Array, "T m_w"]
    worst_case_rollout: WorstCaseRollout
    nominal_cost: CostBreakdown
    worst_case_cost: CostBreakdown


@dataclass(frozen=True)
class GameCardReference:
    """Full analytical reference bundle for the Phase 0 game card."""

    plant: PlantLinearization
    schedule: CostSchedule
    gamma_star: float
    lqr_solution: RiccatiSolution
    lqr_rollout: ClosedLoopRollout
    lqr_cost: CostBreakdown
    gamma_references: tuple[GammaReference, ...]


def build_canonical_game(
    discretization: Discretization = "euler",
) -> tuple[PlantLinearization, CostSchedule]:
    """Build the canonical C&S 2019 analytical game.

    Canonical released-code fidelity uses forward Euler discretization, as in
    the ModelDB MATLAB implementation. ``discretization="zoh"`` is retained as
    a named higher-order sensitivity variant, not the canonical C&S path.

    The disturbance channel is the physically supported C&S channel on the
    delay-augmented state:

    ``z[t+1] = A_aug z[t] + B_aug u[t] + B_w epsilon[t]``

    where ``B_w`` has shape ``(48, 8)``, with an identity on the current
    physical 8-state block and zeros on the five lag blocks. It is not
    ``I_48``.
    """

    plant = cs_faithful_pointmass(discretization=discretization)
    schedule_phys = cs_eq15_cost_schedule(n_steps=60, alpha_1=1.0, state_dim=8)
    schedule = apply_delay_distribution_to_schedule(
        schedule_phys,
        delay_steps=5,
        n_phys=8,
    )
    return plant, schedule


def build_no_integrator_game(
    discretization: Discretization = "euler",
) -> tuple[PlantLinearization, CostSchedule]:
    """Build the 6D physical-state C&S comparator without disturbance integrators.

    This is not the canonical C&S 2019 game. It preserves the point-mass,
    force-filter, five-step delay augmentation, and Eq. 15 position/velocity/
    force costs while omitting the two disturbance-integrator coordinates from
    every physical delay block.
    """

    plant = cs_faithful_pointmass(
        disturbance_integrator=False,
        delay_steps=5,
        discretization=discretization,
    )
    schedule_phys = cs_eq15_cost_schedule(n_steps=60, alpha_1=1.0, state_dim=6)
    schedule = apply_delay_distribution_to_schedule(
        schedule_phys,
        delay_steps=5,
        n_phys=6,
    )
    return plant, schedule


def build_zoh_sensitivity_game() -> tuple[PlantLinearization, CostSchedule]:
    """Build the named ZOH sensitivity variant of the C&S card."""

    return build_canonical_game(discretization="zoh")


def assert_physical_selector_bw(plant: PlantLinearization) -> None:
    """Raise if canonical ``B_w`` is not ``[I_8; 0]``."""

    if plant.n != 48 or plant.m_w != 8:
        raise ValueError(
            f"Expected canonical plant shape (n=48, m_w=8); got {plant.n=}, {plant.m_w=}"
        )
    top = plant.Bw[:8, :]
    lag = plant.Bw[8:, :]
    if not bool(jnp.allclose(top, jnp.eye(8, dtype=jnp.float64), atol=1e-14)):
        raise ValueError("Canonical C&S B_w top physical block is not I_8.")
    if not bool(jnp.allclose(lag, 0.0, atol=1e-14)):
        raise ValueError("Canonical C&S B_w lag block is not zero.")


def _factor_key(factor: float) -> str:
    text = f"{factor:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _rollout_cost(
    schedule: CostSchedule,
    rollout: ClosedLoopRollout | WorstCaseRollout,
    *,
    gamma: float | None = None,
) -> CostBreakdown:
    x = rollout.x.astype(jnp.float64)
    u = rollout.u.astype(jnp.float64)
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    state_stage = float(jnp.sum(state_terms))
    control_stage = float(jnp.sum(control_terms))
    terminal_state = float(terminal)
    total = state_stage + control_stage + terminal_state

    disturbance_energy = 0.0
    h_inf_objective = None
    if isinstance(rollout, WorstCaseRollout):
        disturbance_energy = float(jnp.sum(rollout.epsilon**2))
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


def riccati_worst_case_policy(
    plant: PlantLinearization,
    solution: RiccatiSolution,
) -> Float[Array, "T m_w n"]:
    """Return the Riccati feedback disturbance policy ``epsilon_t = F_t x_t``.

    For the discrete game with stage disturbance penalty
    ``-gamma^2 epsilon_t^T epsilon_t`` and dynamics
    ``x[t+1] = A x[t] + B u[t] + B_w epsilon[t]``, the maximizing disturbance
    after substituting ``u_t = -K_t x_t`` is:

    ``F_t = (gamma^2 I - B_w^T P[t+1] B_w)^-1 B_w^T P[t+1] (A - B K_t)``.

    This is a state-dependent closed-loop policy. An open-loop epsilon sequence
    is only a realization of this policy along a specified trajectory.
    """

    if not solution.admissible:
        raise ValueError("Cannot build a worst-case policy for an inadmissible solution.")
    if not jnp.isfinite(solution.gamma):
        raise ValueError("LQR has no finite-gamma worst-case epsilon policy.")

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)
    m_w = plant.m_w
    eye_w = jnp.eye(m_w, dtype=jnp.float64)

    policies = []
    gamma2 = solution.gamma * solution.gamma
    for t in range(solution.K.shape[0]):
        p_next = solution.P[t + 1]
        a_cl = A - B @ solution.K[t]
        lhs = gamma2 * eye_w - Bw.T @ p_next @ Bw
        rhs = Bw.T @ p_next @ a_cl
        policies.append(jnp.linalg.solve(lhs, rhs))
    return jnp.stack(policies, axis=0)


def rollout_with_disturbance_policy(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    F: Float[Array, "T m_w n"],
    x0: Array,
) -> WorstCaseRollout:
    """Roll the closed loop with ``u_t = -K_t x_t`` and ``epsilon_t = F_t x_t``."""

    x_seq = [x0.astype(jnp.float64)]
    u_seq: list[jnp.ndarray] = []
    eps_seq: list[jnp.ndarray] = []

    for t in range(K.shape[0]):
        x_t = x_seq[-1]
        u_t = -K[t] @ x_t
        eps_t = F[t] @ x_t
        x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t
        u_seq.append(u_t)
        eps_seq.append(eps_t)
        x_seq.append(x_next)

    return WorstCaseRollout(
        x=jnp.stack(x_seq, axis=0),
        u=jnp.stack(u_seq, axis=0),
        epsilon=jnp.stack(eps_seq, axis=0),
    )


def materialize_reference(
    gamma_factors: tuple[float, ...] = DEFAULT_GAMMA_FACTORS,
) -> GameCardReference:
    """Compute the canonical analytical reference bundle."""

    plant, schedule = build_canonical_game()
    assert_physical_selector_bw(plant)
    gamma_star = find_gamma_star(plant, schedule)
    x0 = make_reach_initial_state(plant, init_pos=INIT_POS, target_pos=TARGET_POS)

    lqr_solution = solve_lqr(plant, schedule)
    lqr_rollout = simulate_closed_loop(plant, lqr_solution.K, x0, target_pos=TARGET_POS)
    lqr_cost = _rollout_cost(schedule, lqr_rollout)

    refs = []
    for factor in gamma_factors:
        gamma = float(factor * gamma_star)
        solution = solve_hinf_riccati(plant, schedule, gamma)
        if not solution.admissible:
            raise RuntimeError(f"H-infinity solution inadmissible at factor={factor}.")
        nominal = simulate_closed_loop(plant, solution.K, x0, target_pos=TARGET_POS)
        F = riccati_worst_case_policy(plant, solution)
        eps_nominal = jnp.einsum("tmn,tn->tm", F, nominal.x[:-1])
        worst_rollout = rollout_with_disturbance_policy(plant, solution.K, F, x0)
        refs.append(
            GammaReference(
                factor=float(factor),
                gamma=gamma,
                solution=solution,
                nominal_rollout=nominal,
                worst_case_policy=F,
                epsilon_on_nominal=eps_nominal,
                worst_case_rollout=worst_rollout,
                nominal_cost=_rollout_cost(schedule, nominal),
                worst_case_cost=_rollout_cost(schedule, worst_rollout, gamma=gamma),
            )
        )

    return GameCardReference(
        plant=plant,
        schedule=schedule,
        gamma_star=float(gamma_star),
        lqr_solution=lqr_solution,
        lqr_rollout=lqr_rollout,
        lqr_cost=lqr_cost,
        gamma_references=tuple(refs),
    )


def reference_summary(
    reference: GameCardReference,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return a JSON-serializable summary of the reference bundle."""

    lqr = reference.lqr_rollout
    lqr_peak = lqr.peak_forward_velocity
    frontier = []
    for ref in reference.gamma_references:
        nominal = ref.nominal_rollout
        delta_v = 100.0 * (nominal.peak_forward_velocity - lqr_peak) / lqr_peak
        frontier.append(
            {
                "factor": ref.factor,
                "factor_key": _factor_key(ref.factor),
                "gamma": ref.gamma,
                "delta_v_percent": float(delta_v),
                "hinf_peak_forward_velocity": nominal.peak_forward_velocity,
                "time_to_peak_step": nominal.peak_forward_velocity_idx,
                "terminal_position_error_m": nominal.terminal_position_error,
                "nominal_total_cost": ref.nominal_cost.total_without_disturbance_penalty,
                "epsilon_on_nominal_energy": float(jnp.sum(ref.epsilon_on_nominal**2)),
                "epsilon_on_nominal_l2": float(jnp.linalg.norm(ref.epsilon_on_nominal)),
                "closed_loop_epsilon_energy": ref.worst_case_cost.disturbance_energy,
                "closed_loop_epsilon_l2": float(jnp.sqrt(ref.worst_case_cost.disturbance_energy)),
                "worst_case_total_cost_without_penalty": (
                    ref.worst_case_cost.total_without_disturbance_penalty
                ),
                "worst_case_h_infinity_objective": ref.worst_case_cost.h_infinity_objective,
            }
        )

    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "regeneration_command": "PYTHONPATH=src python scripts/materialize_analytical_game_card.py",
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="analytical_game_card",
        ),
        "primary_gamma_factor": PRIMARY_GAMMA_FACTOR,
        "output_feedback_certificate_gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "output_feedback_gamma_selection_issue": OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
        "diagnostic_gamma_factor": DIAGNOSTIC_GAMMA_FACTOR,
        "plant": {
            "name": "cs_faithful_pointmass",
            "discretization": reference.plant.discretization,
            "mass": 1.0,
            "damping": 0.1,
            "tau": 0.066,
            "dt": reference.plant.dt,
            "physical_state_dim": 8,
            "delay_steps": 5,
            "state_dim": reference.plant.n,
            "disturbance_dim": reference.plant.m_w,
            "state_order": [
                "px",
                "py",
                "vx",
                "vy",
                "fx",
                "fy",
                "eps_x_int",
                "eps_y_int",
                "lag1_px",
                "...",
                "lag5_eps_y_int",
            ],
            "bw_shape": list(reference.plant.Bw.shape),
            "bw_contract": "top physical 8x8 block is identity; lag rows are zero",
        },
        "task": {
            "horizon_steps": reference.schedule.T,
            "duration_s": reference.schedule.T * reference.plant.dt,
            "init_pos_m": [0.0, 0.0],
            "target_pos_m": [0.15, 0.0],
            "hold_free": True,
            "single_reach": True,
        },
        "cost": {
            "schedule": "C&S Eq. 15 physical 8-state schedule with 5-step delay distribution",
            "position_weight": "fact_t * 1e6",
            "velocity_weight": "fact_t * 1e5",
            "force_and_integrator_weight": "1.0",
            "fact_t": "((t + 1) / T)^6, capped at 1",
            "R": "I_2",
        },
        "epsilon_metric": {
            "norm": "sum_t ||epsilon_t||_2^2 over the 8 physical epsilon coordinates",
            "dt_scaled": False,
            "note": (
                "Gamma is an H-infinity attenuation parameter, not an epsilon budget. "
                "Any PGD budget must be derived explicitly from a realized Riccati "
                "epsilon sequence and treated as an open-loop surrogate budget."
            ),
        },
        "gamma_star": reference.gamma_star,
        "lqr": {
            "peak_forward_velocity": lqr.peak_forward_velocity,
            "time_to_peak_step": lqr.peak_forward_velocity_idx,
            "terminal_position_error_m": lqr.terminal_position_error,
            "total_cost": reference.lqr_cost.total_without_disturbance_penalty,
        },
        "frontier": frontier,
    }


__all__ = [
    "DEFAULT_GAMMA_FACTORS",
    "DIAGNOSTIC_GAMMA_FACTOR",
    "ISSUE_ID",
    "OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR",
    "OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID",
    "PRIMARY_GAMMA_FACTOR",
    "UMBRELLA_ID",
    "GameCardReference",
    "GammaReference",
    "WorstCaseRollout",
    "assert_physical_selector_bw",
    "build_canonical_game",
    "build_no_integrator_game",
    "build_zoh_sensitivity_game",
    "materialize_reference",
    "reference_summary",
    "riccati_worst_case_policy",
    "rollout_with_disturbance_policy",
]
