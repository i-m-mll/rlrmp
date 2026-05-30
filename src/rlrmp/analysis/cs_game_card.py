"""Materialize the C&S released-code-aligned H-infinity analytical game card.

This module is intentionally narrow: it builds the canonical Phase 0 game for
issue ``cb98e58`` / umbrella ``43e8728`` and computes the analytical reference
objects that downstream training code must match. It does not depend on
feedbax runtime state, so tests can protect the mathematical contract before
any simulator-side parity work begins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rlrmp.analysis.hinf_riccati import (
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
from rlrmp.analysis.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "cb98e58"
UMBRELLA_ID = "43e8728"
# Full-state deterministic C&S speed-matching reference. This is not the
# default for output-feedback robustness certificates; see
# OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR below.
PRIMARY_GAMMA_FACTOR = 1.05
# Working output-feedback robustness target selected from the gamma sweep on
# 97604a8. Keep output-feedback Phase 1/3 diagnostics tied to this named value
# so rerunning the sweep requires updating one contract instead of silently
# falling back to the full-state speed-matching gamma.
OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR = 1.4
OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID = "97604a8"
DIAGNOSTIC_GAMMA_FACTOR = 1.5
DEFAULT_GAMMA_FACTORS = (
    1.001,
    PRIMARY_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    DIAGNOSTIC_GAMMA_FACTOR,
    2.0,
    3.0,
)
INIT_POS = jnp.array([0.0, 0.0], dtype=jnp.float64)
TARGET_POS = jnp.array([0.15, 0.0], dtype=jnp.float64)


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


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked game-card note."""

    frontier = summary["frontier"]
    primary = next(row for row in frontier if row["factor"] == PRIMARY_GAMMA_FACTOR)
    output_feedback = next(
        row for row in frontier if row["factor"] == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    diagnostic = next(row for row in frontier if row["factor"] == DIAGNOSTIC_GAMMA_FACTOR)

    rows = [
        "| gamma factor | gamma | Delta-v fwd | peak fwd v | t_peak | "
        "terminal error | closed-loop epsilon L2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frontier:
        rows.append(
            "| "
            f"{row['factor']:.4g} | "
            f"{_fmt(row['gamma'])} | "
            f"{row['delta_v_percent']:+.4f}% | "
            f"{row['hinf_peak_forward_velocity']:.6f} | "
            f"{row['time_to_peak_step']} | "
            f"{row['terminal_position_error_m']:.6g} | "
            f"{row['closed_loop_epsilon_l2']:.6g} |"
        )

    return f"""# Phase 0 Analytical Game Card

Issue: `{ISSUE_ID}`. Umbrella: `{UMBRELLA_ID}`.

This note is the auditable C&S released-code-aligned H-infinity target for the
first cs2019-to-RNN game-equivalence gate. It fixes the analytical game that
later feedbax and trained-controller work must match.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

## Game Definition

- Plant: `cs_faithful_pointmass()`.
- Discretization: `{summary["plant"]["discretization"]}`. The canonical
  released-code path is forward Euler; ZOH is a named sensitivity variant.
- State: 8 physical states plus 5 full-state lag blocks, total `n = 48`.
- Physical state order: `[px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]`.
- Delay state order: `[x_t, x_(t-1), x_(t-2), x_(t-3), x_(t-4), x_(t-5)]`,
  each block using the physical state order above.
- Disturbance channel: `B_w` has shape `(48, 8)`.
- Contract: `B_w[:8, :] = I_8` and `B_w[8:, :] = 0`. The adversary perturbs
  the current physical state only; it does not write directly into delay lags.
- Dynamics convention: `z[t+1] = A_aug z[t] + B_aug u[t] + B_w epsilon[t]`.
- Discrete epsilon convention: epsilon is added in the discrete update; no
  extra `dt` factor is applied.
- Task: hold-free 15 cm forward reach from `[0, 0]` to `[0.15, 0]`.
- Horizon: 60 steps at `dt = 0.01 s` (`0.6 s` total).
- Observation/information structure for the analytical target: full augmented
  state feedback.

## Cost Schedule

The cost is C&S Eq. 15 on the physical 8-state schedule, distributed over the
5-step delay chain with `apply_delay_distribution_to_schedule`.

- Position diagonal: `fact_t * 1e6`.
- Velocity diagonal: `fact_t * 1e5`.
- Force and disturbance-integrator diagonals: `1.0`.
- `fact_t = ((t + 1) / T)^6`, capped at `1`.
- Control cost: `R_t = I_2`.

This resolves the Phase 0 part of blocker `6ec6b19`: the first gate uses this
fixed C&S schedule, not an alpha sweep.

## Gamma And Epsilon

- `gamma_star = {summary["gamma_star"]:.6f}`.
- Full-state C&S speed-matching target: `gamma = 1.05 * gamma_star`, giving
  Delta-v `{primary["delta_v_percent"]:+.4f}%`.
- Output-feedback robustness diagnostics use
  `gamma = {OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR:.4g} * gamma_star`, giving
  full-state Delta-v `{output_feedback["delta_v_percent"]:+.4f}%` at the same
  factor. This value is selected from the output-feedback gamma sweep on
  `{OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID}` and must be updated there, not
  by reusing the full-state speed-matching factor.
- Conservative diagnostic point: `gamma = 1.5 * gamma_star`, giving
  Delta-v `{diagnostic["delta_v_percent"]:+.4f}%`.

Gamma is not an epsilon budget. It is the H-infinity attenuation/penalty
parameter. If a full-state open-loop PGD adversary needs a budget, the
game-card mapping for the full-state speed-matching target is:

```text
gamma_design = 1.05 * gamma_star
E_train = sum_t ||epsilon_realized_t||_2^2
```

where `epsilon_realized_t` is the sequence generated by the Riccati
state-dependent worst-case disturbance policy along the specified closed-loop
trajectory. This provides a budget anchor for an open-loop surrogate; it does
not prove the open-loop surrogate is equivalent to the closed-loop H-infinity
game. This resolves the Phase 0 part of blocker `1ad3c16`.

For the primary `1.05 * gamma_star` target:

- `E_train = {primary["closed_loop_epsilon_energy"]:.8g}`.
- `sqrt(E_train) = {primary["closed_loop_epsilon_l2"]:.8g}`.

## Riccati Versus Open-Loop Adversary Objects

The Riccati game defines a feedback disturbance policy:

```text
epsilon_t = F_t x_t
F_t = (gamma^2 I - B_w^T P[t+1] B_w)^-1 B_w^T P[t+1] (A - B K_t)
```

An open-loop epsilon sequence is only a realization of this policy along a
particular trajectory. Downstream PGD training may optimize an open-loop
sequence with the same norm, but that sequence is not automatically the same
formal object as the Riccati adversary. This resolves the Phase 0 definition
needed by `020a65b`; simulator parity and adversary implementation still belong
to later phases.

## Analytical Frontier

LQR baseline:

- Peak forward velocity: `{summary["lqr"]["peak_forward_velocity"]:.6f} m/s`.
- Time to peak: step `{summary["lqr"]["time_to_peak_step"]}`.
- Terminal position error: `{summary["lqr"]["terminal_position_error_m"]:.6g} m`.

{"\n".join(rows)}

## Generated Bundle

Regenerate with:

```bash
{summary["regeneration_command"]}
```

Tracked manifest:
`results/{ISSUE_ID}/notes/analytical_game_card_manifest.json`.

Bulk arrays:
`_artifacts/{ISSUE_ID}/analytical_game_card/canonical_reference.npz`.

The `.npz` bundle includes LQR and H-infinity gains, nominal trajectories,
Riccati worst-case feedback policies `F_t`, epsilon sequences induced on the
nominal trajectory, and epsilon sequences from the closed-loop Riccati
worst-case rollout.
"""


def _npz_arrays(reference: GameCardReference) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "plant_A": np.asarray(reference.plant.A),
        "plant_B": np.asarray(reference.plant.B),
        "plant_Bw": np.asarray(reference.plant.Bw),
        "schedule_Q": np.asarray(reference.schedule.Q),
        "schedule_R": np.asarray(reference.schedule.R),
        "schedule_Q_f": np.asarray(reference.schedule.Q_f),
        "lqr_K": np.asarray(reference.lqr_solution.K),
        "lqr_P": np.asarray(reference.lqr_solution.P),
        "lqr_x_nominal": np.asarray(reference.lqr_rollout.x),
        "lqr_u_nominal": np.asarray(reference.lqr_rollout.u),
    }
    for ref in reference.gamma_references:
        key = _factor_key(ref.factor)
        arrays[f"hinf_{key}_K"] = np.asarray(ref.solution.K)
        arrays[f"hinf_{key}_P"] = np.asarray(ref.solution.P)
        arrays[f"hinf_{key}_x_nominal"] = np.asarray(ref.nominal_rollout.x)
        arrays[f"hinf_{key}_u_nominal"] = np.asarray(ref.nominal_rollout.u)
        arrays[f"hinf_{key}_F_worst_policy"] = np.asarray(ref.worst_case_policy)
        arrays[f"hinf_{key}_epsilon_on_nominal"] = np.asarray(ref.epsilon_on_nominal)
        arrays[f"hinf_{key}_x_worst_case"] = np.asarray(ref.worst_case_rollout.x)
        arrays[f"hinf_{key}_u_worst_case"] = np.asarray(ref.worst_case_rollout.u)
        arrays[f"hinf_{key}_epsilon_worst_case"] = np.asarray(ref.worst_case_rollout.epsilon)
    return arrays


def write_outputs(
    issue_id: str = ISSUE_ID,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Write tracked and untracked game-card outputs."""

    reference = materialize_reference()
    summary = reference_summary(reference, discretization=discretization, lane=lane)
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "analytical_game_card")

    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Phase 0 analytical game-card artifacts for the cs2019-to-RNN "
            "game-equivalence programme. See `notes/analytical_game_card.md` "
            "for the tracked C&S released-code-aligned reference target.\n",
            encoding="utf-8",
        )

    manifest_path = notes_dir / "analytical_game_card_manifest.json"
    note_path = notes_dir / "analytical_game_card.md"
    npz_path = artifact_dir / "canonical_reference.npz"

    np.savez_compressed(npz_path, **_npz_arrays(reference))

    manifest = {
        **summary,
        "tracked_note": f"results/{issue_id}/notes/analytical_game_card.md",
        "artifact_npz": f"_artifacts/{issue_id}/analytical_game_card/canonical_reference.npz",
        "artifact_npz_keys": sorted(_npz_arrays(reference).keys()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    return manifest


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
    "build_zoh_sensitivity_game",
    "materialize_reference",
    "reference_summary",
    "render_markdown",
    "riccati_worst_case_policy",
    "rollout_with_disturbance_policy",
    "write_outputs",
]
