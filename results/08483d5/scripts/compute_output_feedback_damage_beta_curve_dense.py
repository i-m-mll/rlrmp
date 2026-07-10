from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import stochastic_policy_rollout

from rlrmp.analysis.math.cs_game_card import TARGET_POS, build_no_integrator_game
from rlrmp.analysis.math.cs_released_simulation import (
    default_cs_noise_covariances,
    sample_forward_noise_draws,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_estimator_joint_matrices,
    robust_output_feedback_gains,
    rollout_with_robust_estimator,
    rollout_with_robust_estimator_policy,
)


jax.config.update("jax_enable_x64", True)

OUT_DIR = Path("results/08483d5/notes")
FIGURE_DIR = Path("_artifacts/08483d5/figures/output_feedback_damage_beta_curve_dense")
JSON_PATH = OUT_DIR / "output_feedback_damage_beta_curve_dense.json"
CSV_PATH = OUT_DIR / "output_feedback_damage_beta_curve_dense.csv"
PNG_PATH = FIGURE_DIR / "deterministic_damage_logy.png"
MD_PATH = OUT_DIR / "output_feedback_damage_beta_curve_dense.md"
STOCHASTIC_SEED = 376023
REFERENCE_1P4_DET = 3704.963
REFERENCE_1P4_NOISY = 6131.691
REFERENCE_1P05_DET = 447.8904023668665
REFERENCE_1P05_NOISY = 1911.8930971469426


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _cost_from_arrays(schedule, x, u, epsilon, gamma: float | None) -> dict[str, float | None]:
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    state_stage = _float(jnp.sum(state_terms))
    control_stage = _float(jnp.sum(control_terms))
    terminal_state = _float(terminal)
    total = state_stage + control_stage + terminal_state
    disturbance_energy = _float(jnp.sum(epsilon**2))
    h_inf_objective = None
    if gamma is not None:
        h_inf_objective = total - float(gamma * gamma) * disturbance_energy
    return {
        "state_stage": state_stage,
        "control_stage": control_stage,
        "terminal_state": terminal_state,
        "total_without_disturbance_penalty": total,
        "disturbance_energy": disturbance_energy,
        "h_infinity_objective": h_inf_objective,
    }


def _rollout_summary(plant, x, u, epsilon, schedule, gamma: float | None) -> dict[str, Any]:
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos_abs = pos + TARGET_POS[None, :]
    terminal_error = jnp.linalg.norm(pos_abs[-1] - TARGET_POS)
    cost = _cost_from_arrays(schedule, x, u, epsilon, gamma)
    return {
        "cost": cost,
        "peak_forward_velocity_m_s": _float(jnp.max(forward)),
        "peak_forward_velocity_idx": int(jnp.argmax(forward)),
        "terminal_position_error_m": _float(terminal_error),
        "control_effort_sum_u2": _float(jnp.sum(u**2)),
        "disturbance_l2": float(np.sqrt(max(cost["disturbance_energy"], 0.0))),
    }


_stochastic_policy_rollout = stochastic_policy_rollout


def _rollout_damage(plant, schedule, solution, x0, gains, policy, config) -> dict[str, Any]:
    clean_det = rollout_with_robust_estimator(
        plant, schedule, solution, x0, gains=gains, config=config
    )
    adv_det = rollout_with_robust_estimator_policy(
        plant, schedule, solution, x0, policy, gains=gains, config=config
    )
    clean_det_cost = output_feedback_cost(schedule, clean_det, gamma=solution.gamma)
    adv_det_cost = output_feedback_cost(schedule, adv_det, gamma=solution.gamma)

    covariances = default_cs_noise_covariances(plant, config)
    draws = sample_forward_noise_draws(
        jr.PRNGKey(STOCHASTIC_SEED),
        T=schedule.T,
        covariances=covariances,
    )
    clean_noisy = _stochastic_policy_rollout(
        plant,
        schedule,
        solution,
        x0,
        draws,
        covariances,
        gains,
        policy,
        adversarial=False,
        config=config,
    )
    adv_noisy = _stochastic_policy_rollout(
        plant,
        schedule,
        solution,
        x0,
        draws,
        covariances,
        gains,
        policy,
        adversarial=True,
        config=config,
    )
    clean_noisy_summary = _rollout_summary(
        plant, clean_noisy.x, clean_noisy.u_applied, clean_noisy.adversary_epsilon, schedule, solution.gamma
    )
    adv_noisy_summary = _rollout_summary(
        plant, adv_noisy.x, adv_noisy.u_applied, adv_noisy.adversary_epsilon, schedule, solution.gamma
    )
    return {
        "deterministic": {
            "clean_cost": float(clean_det_cost.total_without_disturbance_penalty),
            "adversarial_cost": float(adv_det_cost.total_without_disturbance_penalty),
            "damage": float(
                adv_det_cost.total_without_disturbance_penalty
                - clean_det_cost.total_without_disturbance_penalty
            ),
            "disturbance_energy": float(adv_det_cost.disturbance_energy),
        },
        "noisy": {
            "clean_cost": clean_noisy_summary["cost"]["total_without_disturbance_penalty"],
            "adversarial_cost": adv_noisy_summary["cost"]["total_without_disturbance_penalty"],
            "damage": float(
                adv_noisy_summary["cost"]["total_without_disturbance_penalty"]
                - clean_noisy_summary["cost"]["total_without_disturbance_penalty"]
            ),
            "disturbance_energy": adv_noisy_summary["cost"]["disturbance_energy"],
        },
    }


def _stage_cost_matrices(schedule, gains) -> tuple[np.ndarray, np.ndarray]:
    n = int(schedule.Q.shape[-1])
    zeros = np.zeros((n, n), dtype=np.float64)
    stage = []
    for t in range(int(schedule.T)):
        K_t = np.asarray(gains[t], dtype=np.float64)
        Q_t = np.asarray(schedule.Q[t], dtype=np.float64)
        R_t = np.asarray(schedule.R[t], dtype=np.float64)
        stage.append(np.block([[Q_t, zeros], [zeros, K_t.T @ R_t @ K_t]]))
    terminal = np.block([[np.asarray(schedule.Q_f, dtype=np.float64), zeros], [zeros, zeros]])
    return np.asarray(stage, dtype=np.float64), np.asarray(terminal, dtype=np.float64)


def _value_recursion(
    matrices: np.ndarray,
    offsets: np.ndarray,
    stage_costs: np.ndarray,
    terminal_cost: np.ndarray,
    z0: np.ndarray,
) -> float:
    z_dim = z0.shape[0]
    P = terminal_cost.copy()
    q = np.zeros((z_dim,), dtype=np.float64)
    r = 0.0
    for t in range(matrices.shape[0] - 1, -1, -1):
        M_t = matrices[t]
        b_t = offsets[t]
        Pb_q = P @ b_t + q
        P_t = stage_costs[t] + M_t.T @ P @ M_t
        q_t = M_t.T @ Pb_q
        r_t = float(b_t @ P @ b_t + 2.0 * q @ b_t + r)
        P = 0.5 * (P_t + P_t.T)
        q = q_t
        r = r_t
    return float(z0 @ P @ z0 + 2.0 * q @ z0 + r)


def _deterministic_matrices(
    A_joint: np.ndarray,
    G_joint: np.ndarray,
    policy: np.ndarray,
    *,
    adversarial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    T = A_joint.shape[0]
    z_dim = A_joint.shape[1]
    m_w = G_joint.shape[1]
    F = policy if adversarial else np.zeros((T, m_w, z_dim), dtype=np.float64)
    M = np.asarray(A_joint, dtype=np.float64) + np.einsum("zw,twk->tzk", G_joint, F)
    b = np.zeros((T, z_dim), dtype=np.float64)
    return M, b


def _known_noise_matrices(
    plant,
    schedule,
    solution,
    config,
    A_joint: np.ndarray,
    G_joint: np.ndarray,
    gains: np.ndarray,
    policy: np.ndarray,
    *,
    adversarial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    covariances = default_cs_noise_covariances(plant, config)
    draws = sample_forward_noise_draws(
        jr.PRNGKey(STOCHASTIC_SEED),
        T=int(schedule.T),
        covariances=covariances,
    )
    T = int(schedule.T)
    n = int(plant.n)
    z_dim = 2 * n
    m_w = int(plant.m_w)
    H = np.asarray(delayed_observation_matrix(plant, config), dtype=np.float64)
    covs = np.asarray(robust_estimator_covariances(plant, schedule, solution.gamma, config))
    inv_gamma2 = 1.0 / float(solution.gamma * solution.gamma)
    F = policy if adversarial else np.zeros((T, m_w, z_dim), dtype=np.float64)
    M = np.asarray(A_joint, dtype=np.float64) + np.einsum("zw,twk->tzk", G_joint, F)
    b = np.zeros((T, z_dim), dtype=np.float64)
    signal_tensor = np.asarray(covariances.signal_dependent_state, dtype=np.float64)
    standard = np.asarray(draws.signal_dependent_standard, dtype=np.float64)
    sensory = np.asarray(draws.sensory, dtype=np.float64)
    motor = np.asarray(draws.motor, dtype=np.float64)
    process = np.asarray(draws.process, dtype=np.float64)
    for t in range(T):
        D_t = np.einsum("j,nmj->nm", standard[t], signal_tensor)
        M[t, :n, n:] += -D_t @ gains[t]
        Sigma = covs[t]
        precision = np.linalg.inv(Sigma) + H.T @ H - inv_gamma2 * np.asarray(schedule.Q[t])
        middle = np.linalg.inv(precision)
        b[t, :n] = motor[t] + process[t]
        b[t, n:] = np.asarray(plant.A, dtype=np.float64) @ middle @ H.T @ sensory[t]
    return M, b


def _recursion_damage(plant, schedule, solution, x0, gains, policy, config) -> dict[str, Any]:
    z0 = np.concatenate([np.asarray(x0, dtype=np.float64), np.asarray(x0, dtype=np.float64)])
    covs = robust_estimator_covariances(plant, schedule, solution.gamma, config)
    A_joint, G_joint = robust_estimator_joint_matrices(
        plant, schedule, solution, jnp.asarray(gains), covs, config
    )
    A_joint = np.asarray(A_joint, dtype=np.float64)
    G_joint = np.asarray(G_joint, dtype=np.float64)
    gains_np = np.asarray(gains, dtype=np.float64)
    policy_np = np.asarray(policy, dtype=np.float64)
    stage_costs, terminal_cost = _stage_cost_matrices(schedule, gains_np)

    det_clean_M, det_clean_b = _deterministic_matrices(
        A_joint, G_joint, policy_np, adversarial=False
    )
    det_adv_M, det_adv_b = _deterministic_matrices(
        A_joint, G_joint, policy_np, adversarial=True
    )
    noisy_clean_M, noisy_clean_b = _known_noise_matrices(
        plant,
        schedule,
        solution,
        config,
        A_joint,
        G_joint,
        gains_np,
        policy_np,
        adversarial=False,
    )
    noisy_adv_M, noisy_adv_b = _known_noise_matrices(
        plant,
        schedule,
        solution,
        config,
        A_joint,
        G_joint,
        gains_np,
        policy_np,
        adversarial=True,
    )
    det_clean = _value_recursion(det_clean_M, det_clean_b, stage_costs, terminal_cost, z0)
    det_adv = _value_recursion(det_adv_M, det_adv_b, stage_costs, terminal_cost, z0)
    noisy_clean = _value_recursion(noisy_clean_M, noisy_clean_b, stage_costs, terminal_cost, z0)
    noisy_adv = _value_recursion(noisy_adv_M, noisy_adv_b, stage_costs, terminal_cost, z0)
    return {
        "deterministic": {
            "clean_cost": det_clean,
            "adversarial_cost": det_adv,
            "damage": det_adv - det_clean,
        },
        "noisy": {
            "clean_cost": noisy_clean,
            "adversarial_cost": noisy_adv,
            "damage": noisy_adv - noisy_clean,
        },
    }


def _beta_grid() -> list[float]:
    coarse = [1.001] + [round(x, 2) for x in np.arange(1.01, 2.0001, 0.01)]
    dense = [round(1.330 + 0.001 * idx, 3) for idx in range(31)]
    return sorted(set(coarse + dense))


def _build_policy(plant, schedule, gamma_star: float, beta: float, config: OutputFeedbackConfig):
    gamma = float(beta * gamma_star)
    solution = solve_hinf_riccati(plant, schedule, gamma)
    if not solution.admissible:
        raise RuntimeError(f"beta={beta} gamma={gamma} is not admissible")
    x0 = make_cs_output_feedback_initial_state(plant, config)
    covs = robust_estimator_covariances(plant, schedule, solution.gamma, config)
    gains = robust_output_feedback_gains(plant, schedule, solution, covs, config)
    policy = robust_estimator_fixed_adversary_policy(
        plant, schedule, solution, gains, covs, config
    )
    return solution, x0, gains, policy


def _plot(rows: list[dict[str, Any]]) -> None:
    beta = np.asarray([row["beta"] for row in rows], dtype=np.float64)
    det = np.asarray([row["deterministic_damage"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    blue = "#1f6f8b"
    ax.plot(beta, det, marker="o", color=blue, linewidth=1.6, markersize=2.4)
    ax.set_xlabel("beta = gamma / gamma_star")
    ax.set_ylabel("deterministic damage: adversarial cost - clean cost (log scale)")
    ax.set_yscale("log")
    ax.set_title("6D output-feedback H-infinity deterministic damage vs beta")
    ax.grid(True, which="both", alpha=0.28)
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


def _write_outputs(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "beta",
                "gamma",
                "deterministic_clean_cost",
                "deterministic_adversarial_cost",
                "deterministic_damage",
                "noisy_clean_cost",
                "noisy_adversarial_cost",
                "noisy_damage",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _plot(rows)
    head = {row["beta"]: row for row in rows}
    lines = [
        "# 6D output-feedback H-infinity damage beta curves",
        "",
        "## Method",
        "",
        "- Source convention: same recursion/rollout convention as the beta 1.05 "
        "and beta 1.4 scripts on this issue branch.",
        "- Controller: 6D no-integrator output-feedback H-infinity analytical teacher.",
        "- State: 36D delay-augmented state, six 6D physical blocks "
        "`[x, y, vx, vy, force_x, force_y]`; no disturbance integrators.",
        "- Damage: task cost increase only, `Q/R/Qf` cost under the H-infinity "
        "controller's fixed optimal adversary policy `F_t`, minus the same "
        "controller/estimator with `F_t = 0`; no `gamma^2 ||epsilon||^2` "
        "subtraction in the reported damage.",
        "- Deterministic convention: no forward-simulation noise.",
        f"- Noisy convention: paired single-seed nominal C&S noise with seed `{STOCHASTIC_SEED}`; "
        "clean and adversarial conditions share sensory, motor, process, and "
        "signal-dependent standard draws.",
        "- Beta-specific gamma, Riccati solution, estimator covariance, controller gains, "
        "and adversary policy were recomputed live for every beta.",
        "",
        "## Benchmark",
        "",
        f"- Beta grid: {len(rows)} points. It includes beta `1.001`, every `0.01` "
        "from `1.01` through `2.00`, and every `0.001` from `1.330` through "
        "`1.360`.",
        f"- Benchmark betas: {payload['benchmark']['betas']}.",
        f"- Max rollout-vs-recursion damage absolute difference: "
        f"{payload['benchmark']['max_damage_abs_diff']:.12g}.",
        f"- Mean rollout time per beta: {payload['benchmark']['mean_rollout_seconds']:.6g} s.",
        f"- Mean recursion time per beta: {payload['benchmark']['mean_recursion_seconds']:.6g} s.",
        f"- Method selected for full grid: `{payload['method_selected']}`.",
        "",
        "## Sanity checks",
        "",
        f"- beta=1.05 deterministic: {head[1.05]['deterministic_damage']:.12g} "
        f"(prior {REFERENCE_1P05_DET:.12g}).",
        f"- beta=1.05 noisy: {head[1.05]['noisy_damage']:.12g} "
        f"(prior {REFERENCE_1P05_NOISY:.12g}).",
        f"- beta=1.40 deterministic: {head[1.4]['deterministic_damage']:.12g} "
        f"(prior about {REFERENCE_1P4_DET:.12g}).",
        f"- beta=1.40 noisy: {head[1.4]['noisy_damage']:.12g} "
        f"(prior about {REFERENCE_1P4_NOISY:.12g}).",
        "",
        "## Headline values",
        "",
        "| beta | deterministic damage | noisy damage |",
        "|---:|---:|---:|",
    ]
    for beta in (1.001, 1.01, 1.05, 1.4, 2.0):
        row = head[beta]
        lines.append(
            f"| {beta:.3f} | {row['deterministic_damage']:.12g} | "
            f"{row['noisy_damage']:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Uncertainties",
            "",
            "- The noisy curve is one paired nominal-noise draw, not a Monte Carlo expectation "
            "over many seeds. It is faithful to the prior beta scripts' convention, but "
            "does not quantify sampling variability.",
            "- I did not use historical cap/radius/trust-region constants; no training or "
            "PGD safety cap enters this analytical recursion.",
            "",
            "## Commands",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --no-sync python "
            "results/08483d5/scripts/compute_output_feedback_damage_beta_curve_dense.py",
            "```",
            "",
        ]
    )
    MD_PATH.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = OutputFeedbackConfig(n_phys=6)
    plant, schedule = build_no_integrator_game()
    gamma_star = float(find_gamma_star(plant, schedule))
    betas = _beta_grid()
    benchmark_betas = [1.05, 1.4, 2.0]
    rows = []
    benchmark = []
    for beta in betas:
        solution, x0, gains, policy = _build_policy(plant, schedule, gamma_star, beta, config)
        t0 = time.perf_counter()
        recursion = _recursion_damage(plant, schedule, solution, x0, gains, policy, config)
        recursion_seconds = time.perf_counter() - t0
        rollout = None
        rollout_seconds = None
        if beta in benchmark_betas:
            t1 = time.perf_counter()
            rollout = _rollout_damage(plant, schedule, solution, x0, gains, policy, config)
            rollout_seconds = time.perf_counter() - t1
            benchmark.append(
                {
                    "beta": beta,
                    "rollout_seconds": rollout_seconds,
                    "recursion_seconds": recursion_seconds,
                    "deterministic_damage_rollout": rollout["deterministic"]["damage"],
                    "deterministic_damage_recursion": recursion["deterministic"]["damage"],
                    "deterministic_damage_abs_diff": abs(
                        rollout["deterministic"]["damage"]
                        - recursion["deterministic"]["damage"]
                    ),
                    "noisy_damage_rollout": rollout["noisy"]["damage"],
                    "noisy_damage_recursion": recursion["noisy"]["damage"],
                    "noisy_damage_abs_diff": abs(
                        rollout["noisy"]["damage"] - recursion["noisy"]["damage"]
                    ),
                }
            )
        rows.append(
            {
                "beta": beta,
                "gamma": float(solution.gamma),
                "deterministic_clean_cost": recursion["deterministic"]["clean_cost"],
                "deterministic_adversarial_cost": recursion["deterministic"][
                    "adversarial_cost"
                ],
                "deterministic_damage": recursion["deterministic"]["damage"],
                "noisy_clean_cost": recursion["noisy"]["clean_cost"],
                "noisy_adversarial_cost": recursion["noisy"]["adversarial_cost"],
                "noisy_damage": recursion["noisy"]["damage"],
                "recursion_seconds": recursion_seconds,
                "rollout_benchmark": rollout,
                "rollout_seconds": rollout_seconds,
            }
        )
        print(
            json.dumps(
                {
                    "beta": beta,
                    "deterministic_damage": rows[-1]["deterministic_damage"],
                    "noisy_damage": rows[-1]["noisy_damage"],
                    "recursion_seconds": recursion_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    max_damage_abs_diff = max(
        max(item["deterministic_damage_abs_diff"], item["noisy_damage_abs_diff"])
        for item in benchmark
    )
    method_selected = (
        "affine_value_recursion"
        if max_damage_abs_diff <= 1e-6
        else "recursion_preserved_with_rollout_benchmark_mismatch"
    )
    payload = {
        "schema_version": "rlrmp.damage_beta_curves.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "repo": "/Users/mll/Main/10 Projects/10 PhD/rlrmp",
            "reference_branch": "feature/08483d5-adaptive-soft-adversary",
            "reference_scripts": [
                "results/08483d5/scripts/compute_output_feedback_rollout_damage.py",
                "results/08483d5/scripts/compute_output_feedback_recursion_damage.py",
                "results/08483d5/scripts/compute_output_feedback_rollout_damage_beta1p05.py",
                "results/08483d5/scripts/compute_output_feedback_recursion_damage_beta1p05.py",
            ],
        },
        "contract": {
            "gamma_star": gamma_star,
            "beta_grid": betas,
            "stochastic_seed": STOCHASTIC_SEED,
            "state_basis": (
                "36D delay-augmented 6D physical state; no disturbance integrators; "
                "observation is delayed 6D physical block"
            ),
            "damage_definition": (
                "Expected/task cost increase from fixed optimal adversary policy F "
                "relative to F=0 under the same controller and same deterministic/noisy condition"
            ),
            "noise_convention": (
                "deterministic noise off; noisy is paired single-seed nominal C&S forward noise"
            ),
        },
        "method_selected": method_selected,
        "benchmark": {
            "betas": benchmark_betas,
            "rows": benchmark,
            "max_damage_abs_diff": max_damage_abs_diff,
            "mean_rollout_seconds": float(np.mean([item["rollout_seconds"] for item in benchmark])),
            "mean_recursion_seconds": float(
                np.mean([item["recursion_seconds"] for item in benchmark])
            ),
        },
        "sanity_checks": {
            "beta_1p05": {
                "deterministic_prior": REFERENCE_1P05_DET,
                "noisy_prior": REFERENCE_1P05_NOISY,
            },
            "beta_1p4": {
                "deterministic_prior_approx": REFERENCE_1P4_DET,
                "noisy_prior_approx": REFERENCE_1P4_NOISY,
            },
        },
        "rows": rows,
        "outputs": {
            "json": str(JSON_PATH),
            "csv": str(CSV_PATH),
            "png": str(PNG_PATH),
            "markdown": str(MD_PATH),
        },
    }
    _write_outputs(payload)
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
