"""Legacy scope note: `write_outputs` is a frozen writer/driver surface. The
math core in this module remains LIVE library code consumed by registered
recipes.

Phase 1 adversary-equivalence analysis for the C&S game card.

This module compares two disturbance objects for the fixed analytical game:

- the Riccati-implied state-feedback disturbance ``epsilon_t = F_t x_t``;
- an open-loop epsilon sequence optimized under the same discrete L2 budget.

It intentionally stays on the analytical side of the codebase. Feedbax training
changes should consume the decision this module produces, not define it.
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

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.math.cs_game_card import (
    ISSUE_ID as GAME_CARD_ISSUE_ID,
    PRIMARY_GAMMA_FACTOR,
    GammaReference,
    GameCardReference,
    WorstCaseRollout,
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
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "a7dad8a"
UMBRELLA_ID = "43e8728"
_ANALYSIS_PRESET = load_analysis_parameter_preset("adversary_equivalence").parameters
OPEN_LOOP_STEP_SWEEP = tuple(_ANALYSIS_PRESET["open_loop_step_sweep"])
OPEN_LOOP_RESTARTS = int(_ANALYSIS_PRESET["open_loop_restarts"])
OPEN_LOOP_LEARNING_RATE = float(_ANALYSIS_PRESET["open_loop_learning_rate"])
PGD_CONVERGENCE_TOL = 1e-3
RESTART_STABILITY_TOL = 2e-3
EQUIVALENCE_REL_TOL = 1e-2
INIT_POS = np.asarray(_ANALYSIS_PRESET["initial_position_m"], dtype=np.float64)
TARGET_POS = np.asarray(_ANALYSIS_PRESET["target_position_m"], dtype=np.float64)


@dataclass(frozen=True)
class RolloutCost:
    """Quadratic cost and disturbance-energy summary for one rollout."""

    state_stage: float
    control_stage: float
    terminal_state: float
    total_without_disturbance_penalty: float
    disturbance_energy: float
    h_infinity_objective: float


@dataclass(frozen=True)
class OpenLoopOptimizationConfig:
    """Hyperparameters for open-loop epsilon optimization."""

    budget: float
    n_steps: int
    n_restarts: int = OPEN_LOOP_RESTARTS
    learning_rate: float = OPEN_LOOP_LEARNING_RATE
    seed: int = 0


@dataclass(frozen=True)
class OpenLoopOptimizationResult:
    """Best open-loop epsilon sequence found for one optimizer configuration."""

    config: OpenLoopOptimizationConfig
    epsilon: Float[Array, "T m_w"]
    rollout: ClosedLoopRollout
    cost: RolloutCost
    initial_objectives: tuple[float, ...]
    final_objectives: tuple[float, ...]
    best_objectives: tuple[float, ...]
    final_energies: tuple[float, ...]
    best_energies: tuple[float, ...]
    best_restart_idx: int


@dataclass(frozen=True)
class AdversaryEquivalenceResult:
    """Full Phase 1 comparison at one gamma factor."""

    gamma_factor: float
    gamma: float
    budget: float
    riccati_policy: Float[Array, "T m_w n"]
    riccati_rollout: WorstCaseRollout
    riccati_cost: RolloutCost
    open_loop_results: tuple[OpenLoopOptimizationResult, ...]


def project_l2_ball(
    epsilon: Float[Array, "T m_w"],
    radius: float | Float[Array, ""],
) -> Float[Array, "T m_w"]:
    """Project an epsilon sequence onto a rollout-level L2 ball."""

    radius_arr = jnp.asarray(radius, dtype=epsilon.dtype)
    norm = jnp.linalg.norm(epsilon)
    scale = jnp.minimum(1.0, radius_arr / (norm + jnp.asarray(1e-30, epsilon.dtype)))
    return epsilon * scale


def rollout_arrays_with_open_loop_epsilon(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"],
) -> tuple[Float[Array, "T_plus_1 n"], Float[Array, "T m_u"]]:
    """Roll ``u_t = -K_t x_t`` under a fixed open-loop epsilon sequence."""

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)
    K = K.astype(jnp.float64)
    epsilon = epsilon.astype(jnp.float64)

    def step(x_t: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
        K_t, eps_t = inputs
        u_t = -K_t @ x_t
        x_next = A @ x_t + B @ u_t + Bw @ eps_t
        return x_next, (x_next, u_t)

    _, (x_tail, u_seq) = jax.lax.scan(step, x0.astype(jnp.float64), (K, epsilon))
    x_seq = jnp.concatenate([x0[None].astype(jnp.float64), x_tail], axis=0)
    return x_seq, u_seq


def quadratic_rollout_cost(
    schedule: CostSchedule,
    x: Float[Array, "T_plus_1 n"],
    u: Float[Array, "T m_u"],
    epsilon: Float[Array, "T m_w"],
    *,
    gamma: float,
) -> RolloutCost:
    """Compute the finite-horizon quadratic cost used by the game card."""

    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal_state = x[-1] @ schedule.Q_f @ x[-1]
    state_stage = float(jnp.sum(state_terms))
    control_stage = float(jnp.sum(control_terms))
    terminal = float(terminal_state)
    total = state_stage + control_stage + terminal
    energy = float(jnp.sum(epsilon**2))
    return RolloutCost(
        state_stage=state_stage,
        control_stage=control_stage,
        terminal_state=terminal,
        total_without_disturbance_penalty=total,
        disturbance_energy=energy,
        h_infinity_objective=total - float(gamma * gamma) * energy,
    )


def _objective_value(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"],
) -> Float[Array, ""]:
    x, u = rollout_arrays_with_open_loop_epsilon(plant, K, x0, epsilon)
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    return jnp.sum(state_terms) + jnp.sum(control_terms) + terminal


def _random_epsilon_on_ball(
    key: Array,
    shape: tuple[int, int],
    radius: float,
) -> Float[Array, "T m_w"]:
    eps = jr.normal(key, shape, dtype=jnp.float64)
    return project_l2_ball(eps, radius)


def optimize_open_loop_epsilon(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    *,
    config: OpenLoopOptimizationConfig,
    gamma: float,
    initial_candidates: tuple[Float[Array, "T m_w"], ...] = (),
) -> OpenLoopOptimizationResult:
    """Optimize a fixed epsilon sequence under a rollout-level L2 budget.

    The ascent objective is the task cost without the H-infinity
    ``-gamma^2 ||epsilon||^2`` penalty because the energy is enforced by
    projection. The penalized objective is still reported after optimization for
    comparison with the Riccati game.
    """

    radius = float(jnp.sqrt(jnp.asarray(config.budget, dtype=jnp.float64)))
    shape = (K.shape[0], plant.m_w)
    n_random = max(0, config.n_restarts - len(initial_candidates))
    keys = jr.split(jr.PRNGKey(config.seed), n_random)
    starts = [
        project_l2_ball(candidate.astype(jnp.float64), radius) for candidate in initial_candidates
    ]
    starts.extend(_random_epsilon_on_ball(key, shape, radius) for key in keys)
    if not starts:
        starts.append(jnp.zeros(shape, dtype=jnp.float64))

    optimizer = optax.adam(config.learning_rate)
    value_and_grad = jax.value_and_grad(lambda eps: _objective_value(plant, schedule, K, x0, eps))

    @jax.jit
    def run_projected_ascent(
        start: Float[Array, "T m_w"],
    ) -> tuple[Float[Array, "T m_w"], Float[Array, "T m_w"], Float[Array, ""]]:
        eps = start
        opt_state = optimizer.init(eps)
        best_eps = eps
        best_value = _objective_value(plant, schedule, K, x0, eps)

        def step(carry, _):
            eps, opt_state, best_eps, best_value = carry
            _value, grads = value_and_grad(eps)
            updates, opt_state = optimizer.update(-grads, opt_state, eps)
            eps = optax.apply_updates(eps, updates)
            eps = project_l2_ball(eps, radius)
            value = _objective_value(plant, schedule, K, x0, eps)
            improved = value > best_value
            best_eps = jnp.where(improved, eps, best_eps)
            best_value = jnp.maximum(value, best_value)
            return (eps, opt_state, best_eps, best_value), None

        (eps, _opt_state, best_eps, best_value), _ = jax.lax.scan(
            step,
            (eps, opt_state, best_eps, best_value),
            None,
            length=config.n_steps,
        )
        return best_eps, eps, best_value

    best_epsilons = []
    initial_objectives = []
    final_objectives = []
    best_objectives = []
    final_energies = []
    best_energies = []
    for start in starts:
        initial_objectives.append(float(_objective_value(plant, schedule, K, x0, start)))
        best_eps, final_eps, best_value = run_projected_ascent(start)
        best_epsilons.append(best_eps)
        final_objectives.append(float(_objective_value(plant, schedule, K, x0, final_eps)))
        best_objectives.append(float(best_value))
        final_energies.append(float(jnp.sum(final_eps**2)))
        best_energies.append(float(jnp.sum(best_eps**2)))

    best_idx = int(jnp.argmax(jnp.asarray(best_objectives)))
    best_epsilon = best_epsilons[best_idx]
    rollout = simulate_closed_loop(plant, K, x0, target_pos=TARGET_POS, w=best_epsilon)
    cost = quadratic_rollout_cost(
        schedule,
        rollout.x,
        rollout.u,
        best_epsilon,
        gamma=gamma,
    )
    return OpenLoopOptimizationResult(
        config=config,
        epsilon=best_epsilon,
        rollout=rollout,
        cost=cost,
        initial_objectives=tuple(initial_objectives),
        final_objectives=tuple(final_objectives),
        best_objectives=tuple(best_objectives),
        final_energies=tuple(final_energies),
        best_energies=tuple(best_energies),
        best_restart_idx=best_idx,
    )


def analyze_adversary_equivalence(
    *,
    gamma_factor: float = PRIMARY_GAMMA_FACTOR,
    step_sweep: tuple[int, ...] = OPEN_LOOP_STEP_SWEEP,
    n_restarts: int = OPEN_LOOP_RESTARTS,
    learning_rate: float = OPEN_LOOP_LEARNING_RATE,
    seed: int = 0,
) -> AdversaryEquivalenceResult:
    """Run the Phase 1 analytical adversary-equivalence comparison."""

    reference = materialize_reference(gamma_factors=(gamma_factor,))
    gamma_ref = reference.gamma_references[0]
    if gamma_ref.factor != gamma_factor:
        raise RuntimeError("Materialized gamma reference does not match requested factor.")
    return analyze_reference_adversary_equivalence(
        reference,
        gamma_ref,
        step_sweep=step_sweep,
        n_restarts=n_restarts,
        learning_rate=learning_rate,
        seed=seed,
    )


def analyze_reference_adversary_equivalence(
    reference: GameCardReference,
    gamma_ref: GammaReference,
    *,
    step_sweep: tuple[int, ...] = OPEN_LOOP_STEP_SWEEP,
    n_restarts: int = OPEN_LOOP_RESTARTS,
    learning_rate: float = OPEN_LOOP_LEARNING_RATE,
    seed: int = 0,
) -> AdversaryEquivalenceResult:
    """Compare adversary objects for an already materialized game-card reference."""

    plant = reference.plant
    schedule = reference.schedule
    x0 = make_reach_initial_state(plant, init_pos=INIT_POS, target_pos=TARGET_POS)
    K = gamma_ref.solution.K
    F = riccati_worst_case_policy(plant, gamma_ref.solution)
    riccati_rollout = rollout_with_disturbance_policy(plant, K, F, x0)
    riccati_cost = quadratic_rollout_cost(
        schedule,
        riccati_rollout.x,
        riccati_rollout.u,
        riccati_rollout.epsilon,
        gamma=gamma_ref.gamma,
    )
    budget = riccati_cost.disturbance_energy

    results = []
    for idx, n_steps in enumerate(step_sweep):
        config = OpenLoopOptimizationConfig(
            budget=budget,
            n_steps=int(n_steps),
            n_restarts=n_restarts,
            learning_rate=learning_rate,
            seed=seed + 1009 * idx,
        )
        results.append(
            optimize_open_loop_epsilon(
                plant,
                schedule,
                K,
                x0,
                config=config,
                gamma=gamma_ref.gamma,
                initial_candidates=(riccati_rollout.epsilon,),
            )
        )

    return AdversaryEquivalenceResult(
        gamma_factor=gamma_ref.factor,
        gamma=gamma_ref.gamma,
        budget=budget,
        riccati_policy=F,
        riccati_rollout=riccati_rollout,
        riccati_cost=riccati_cost,
        open_loop_results=tuple(results),
    )


def _cost_dict(cost: RolloutCost) -> dict[str, float]:
    return {
        "state_stage": cost.state_stage,
        "control_stage": cost.control_stage,
        "terminal_state": cost.terminal_state,
        "total_without_disturbance_penalty": cost.total_without_disturbance_penalty,
        "disturbance_energy": cost.disturbance_energy,
        "h_infinity_objective": cost.h_infinity_objective,
    }


def result_summary(
    result: AdversaryEquivalenceResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return a JSON-serializable summary of one Phase 1 comparison."""

    riccati_total = result.riccati_cost.total_without_disturbance_penalty
    open_loop = []
    for opt in result.open_loop_results:
        total = opt.cost.total_without_disturbance_penalty
        open_loop.append(
            {
                "n_steps": opt.config.n_steps,
                "n_restarts": opt.config.n_restarts,
                "learning_rate": opt.config.learning_rate,
                "seed": opt.config.seed,
                "best_restart_idx": opt.best_restart_idx,
                "initial_objectives": list(opt.initial_objectives),
                "final_objectives": list(opt.final_objectives),
                "best_objectives": list(opt.best_objectives),
                "final_energies": list(opt.final_energies),
                "best_energies": list(opt.best_energies),
                "best_cost": _cost_dict(opt.cost),
                "best_peak_forward_velocity": opt.rollout.peak_forward_velocity,
                "best_time_to_peak_step": opt.rollout.peak_forward_velocity_idx,
                "best_terminal_position_error_m": opt.rollout.terminal_position_error,
                "total_cost_minus_riccati": total - riccati_total,
                "total_cost_ratio_to_riccati": total / riccati_total,
                "epsilon_l2_distance_to_riccati": float(
                    jnp.linalg.norm(opt.epsilon - result.riccati_rollout.epsilon)
                ),
            }
        )

    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "game_card_issue": GAME_CARD_ISSUE_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="adversary_equivalence",
        ),
        "primary_gamma_factor": PRIMARY_GAMMA_FACTOR,
        "gamma_factor": result.gamma_factor,
        "gamma": result.gamma,
        "budget": result.budget,
        "budget_l2": float(jnp.sqrt(jnp.asarray(result.budget))),
        "predeclared_tolerances": {
            "pgd_convergence_relative_improvement": PGD_CONVERGENCE_TOL,
            "restart_stability_top3_relative_span": RESTART_STABILITY_TOL,
            "equivalence_relative_tolerance": EQUIVALENCE_REL_TOL,
            "deterministic_replay_rtol": 1e-9,
            "deterministic_replay_atol": 1e-11,
        },
        "riccati_feedback": {
            "cost": _cost_dict(result.riccati_cost),
            "peak_forward_velocity": _rollout_peak_forward_velocity(result.riccati_rollout),
            "time_to_peak_step": _rollout_time_to_peak(result.riccati_rollout),
            "terminal_position_error_m": _rollout_terminal_position_error(result.riccati_rollout),
        },
        "open_loop": open_loop,
    }


def _rollout_peak_forward_velocity(rollout: WorstCaseRollout) -> float:
    vel = rollout.x[:, 2:4]
    line_dir = jnp.array([1.0, 0.0], dtype=jnp.float64)
    return float(jnp.max(vel @ line_dir))


def _rollout_time_to_peak(rollout: WorstCaseRollout) -> int:
    vel = rollout.x[:, 2:4]
    line_dir = jnp.array([1.0, 0.0], dtype=jnp.float64)
    return int(jnp.argmax(vel @ line_dir))


def _rollout_terminal_position_error(rollout: WorstCaseRollout) -> float:
    return float(jnp.linalg.norm(rollout.x[-1, 0:2]))


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a tracked Phase 1 note."""

    ric = summary["riccati_feedback"]
    rows = [
        "| PGD steps | restarts | best cost | ratio to Riccati | energy | epsilon L2 distance |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["open_loop"]:
        rows.append(
            "| "
            f"{row['n_steps']} | "
            f"{row['n_restarts']} | "
            f"{row['best_cost']['total_without_disturbance_penalty']:.8g} | "
            f"{row['total_cost_ratio_to_riccati']:.8g} | "
            f"{row['best_cost']['disturbance_energy']:.8g} | "
            f"{row['epsilon_l2_distance_to_riccati']:.8g} |"
        )

    return f"""# Phase 1 Adversary Equivalence

Issue: `{ISSUE_ID}`. Umbrella: `{UMBRELLA_ID}`. Game card: `{GAME_CARD_ISSUE_ID}`.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

This note compares the C&S-style deterministic Riccati state-dependent
disturbance against an open-loop epsilon surrogate under the Phase 0 game-card
budget.

## Fixed Contract

- Gamma factor: `{summary["gamma_factor"]}`.
- Gamma: `{summary["gamma"]:.8g}`.
- Budget: `sum_t ||epsilon_t||^2 = {summary["budget"]:.8g}`.
- Budget L2 radius: `{summary["budget_l2"]:.8g}`.
- Disturbance channel: `B_w = [I_8; 0]`; epsilon is 8D and enters only the
  current physical state.
- Epsilon metric: unweighted discrete rollout sum, no extra `dt` scaling.

## Riccati Feedback Arm

- Total cost without disturbance penalty:
  `{ric["cost"]["total_without_disturbance_penalty"]:.8g}`.
- Disturbance energy: `{ric["cost"]["disturbance_energy"]:.8g}`.
- H-infinity objective: `{ric["cost"]["h_infinity_objective"]:.8g}`.
- Peak forward velocity: `{ric["peak_forward_velocity"]:.8g}`.
- Time to peak: step `{ric["time_to_peak_step"]}`.
- Terminal position error: `{ric["terminal_position_error_m"]:.8g}`.

## Open-Loop Surrogate Arm

{"\n".join(rows)}

The open-loop arm maximizes the finite-horizon task cost under a projected
rollout-level L2 budget. The H-infinity penalty is not part of the ascent
objective because the energy is constrained directly; it is reported as a
diagnostic objective after each optimized rollout. Each restart retains the best
sequence seen over the whole projected-ascent path, so the Riccati-realized
epsilon incumbent cannot be discarded by a nonmonotone optimizer step.

## Predeclared Gates

- PGD convergence: best-cost relative improvement between the two longest
  sweeps should be below `{summary["predeclared_tolerances"]["pgd_convergence_relative_improvement"]}`.
- Restart stability: the top-three restart objectives should span less than
  `{summary["predeclared_tolerances"]["restart_stability_top3_relative_span"]}` relative.
- Equivalence claim: open-loop and Riccati costs/trajectory metrics should be
  within `{summary["predeclared_tolerances"]["equivalence_relative_tolerance"]}` relative, with deterministic replay protected by
  `rtol={summary["predeclared_tolerances"]["deterministic_replay_rtol"]}` and
  `atol={summary["predeclared_tolerances"]["deterministic_replay_atol"]}`.

## Interpretation

Open-loop optimization is initialized with the Riccati realized epsilon sequence
as one restart, plus independent random restarts projected to the same L2
budget. Matching or exceeding the Riccati realized sequence here establishes
trajectory-level replay equivalence for this fixed game card and initial
condition. It does not make the open-loop epsilon object equivalent to a
state-dependent feedback adversary across off-trajectory states or training-time
policy changes.
"""


def _npz_arrays(result: AdversaryEquivalenceResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "riccati_F_policy": np.asarray(result.riccati_policy),
        "riccati_x": np.asarray(result.riccati_rollout.x),
        "riccati_u": np.asarray(result.riccati_rollout.u),
        "riccati_epsilon": np.asarray(result.riccati_rollout.epsilon),
    }
    for opt in result.open_loop_results:
        key = f"open_loop_{opt.config.n_steps}"
        arrays[f"{key}_x"] = np.asarray(opt.rollout.x)
        arrays[f"{key}_u"] = np.asarray(opt.rollout.u)
        arrays[f"{key}_epsilon"] = np.asarray(opt.epsilon)
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

    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    result = analyze_reference_adversary_equivalence(reference, reference.gamma_references[0])
    summary = {
        **result_summary(result, discretization=discretization, lane=lane),
        "game_card_summary": reference_summary(
            reference,
            discretization=discretization,
            lane=lane,
        ),
    }

    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "adversary_equivalence")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Phase 1 adversary-equivalence artifacts for the cs2019-to-RNN "
            "game-equivalence programme. See `notes/adversary_equivalence.md` "
            "for the tracked comparison summary.\n",
            encoding="utf-8",
        )

    npz_path = artifact_dir / "adversary_equivalence.npz"
    np.savez_compressed(npz_path, **_npz_arrays(result))
    summary["artifact_npz"] = f"_artifacts/{issue_id}/adversary_equivalence/{npz_path.name}"
    summary["artifact_npz_keys"] = sorted(_npz_arrays(result).keys())
    summary["tracked_note"] = f"results/{issue_id}/notes/adversary_equivalence.md"
    summary["tracked_manifest"] = f"results/{issue_id}/notes/adversary_equivalence_manifest.json"

    note_path = notes_dir / "adversary_equivalence.md"
    manifest_path = notes_dir / "adversary_equivalence_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "AdversaryEquivalenceResult",
    "OpenLoopOptimizationConfig",
    "OpenLoopOptimizationResult",
    "RolloutCost",
    "analyze_adversary_equivalence",
    "analyze_reference_adversary_equivalence",
    "optimize_open_loop_epsilon",
    "project_l2_ball",
    "quadratic_rollout_cost",
    "render_markdown",
    "result_summary",
    "rollout_arrays_with_open_loop_epsilon",
    "write_outputs",
]
