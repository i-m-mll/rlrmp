from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import numpy as np

from compute_output_feedback_recursion_damage import main as _run_recursion_damage

from rlrmp.analysis.math.cs_game_card import (
    TARGET_POS,
)
from rlrmp.analysis.math.cs_released_simulation import (
    default_cs_noise_covariances,
    sample_forward_noise_draws,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    robust_estimator_covariances,
)


jax.config.update("jax_enable_x64", True)

OUT_DIR = Path("results/08483d5/notes")
ROLLOUT_JSON_PATH = OUT_DIR / "output_feedback_damage_estimate_beta1p05.json"
JSON_PATH = OUT_DIR / "output_feedback_recursion_damage_comparison_beta1p05.json"
MD_PATH = OUT_DIR / "output_feedback_recursion_damage_comparison_beta1p05.md"
BETA_GAMMA_FACTOR = 1.05
TEACHER_PACKAGE = Path("_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz")
STOCHASTIC_SEED = 376023


def _as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _stage_cost_matrices(schedule, gains) -> tuple[np.ndarray, np.ndarray]:
    n = int(schedule.Q.shape[-1])
    zeros = np.zeros((n, n), dtype=np.float64)
    stage = []
    for t in range(int(schedule.T)):
        K_t = np.asarray(gains[t], dtype=np.float64)
        Q_t = np.asarray(schedule.Q[t], dtype=np.float64)
        R_t = np.asarray(schedule.R[t], dtype=np.float64)
        stage.append(np.block([[Q_t, zeros], [zeros, K_t.T @ R_t @ K_t]]))
    terminal = np.block(
        [
            [np.asarray(schedule.Q_f, dtype=np.float64), zeros],
            [zeros, zeros],
        ]
    )
    return np.asarray(stage, dtype=np.float64), np.asarray(terminal, dtype=np.float64)


def _value_recursion(
    matrices: np.ndarray,
    offsets: np.ndarray,
    stage_costs: np.ndarray,
    terminal_cost: np.ndarray,
    z0: np.ndarray,
) -> dict[str, Any]:
    """Evaluate fixed-policy task cost by backward affine cost-to-go recursion."""

    z_dim = z0.shape[0]
    P = terminal_cost.copy()
    q = np.zeros((z_dim,), dtype=np.float64)
    r = 0.0
    diagnostics = []
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
        diagnostics.append(
            {
                "t": int(t),
                "max_abs_P_asymmetry_after_symmetrize": float(np.max(np.abs(P - P.T))),
                "offset_l2": float(np.linalg.norm(b_t)),
            }
        )
    total = float(z0 @ P @ z0 + 2.0 * q @ z0 + r)
    return {
        "total_without_disturbance_penalty": total,
        "P0_max_abs": float(np.max(np.abs(P))),
        "q0_l2": float(np.linalg.norm(q)),
        "r0": r,
        "backward_diagnostics_first_steps": list(reversed(diagnostics))[:3],
        "backward_diagnostics_last_steps": list(reversed(diagnostics))[-3:],
    }


def _forward_components(
    plant,
    schedule,
    gains: np.ndarray,
    policy: np.ndarray,
    matrices: np.ndarray,
    offsets: np.ndarray,
    z0: np.ndarray,
) -> dict[str, Any]:
    z = z0.copy()
    x_seq = [z[: plant.n].copy()]
    u_seq = []
    eps_seq = []
    for t in range(int(schedule.T)):
        xhat_t = z[plant.n :]
        u_t = -gains[t] @ xhat_t
        eps_t = policy[t] @ z
        u_seq.append(u_t)
        eps_seq.append(eps_t)
        z = matrices[t] @ z + offsets[t]
        x_seq.append(z[: plant.n].copy())
    x = np.asarray(x_seq, dtype=np.float64)
    u = np.asarray(u_seq, dtype=np.float64)
    epsilon = np.asarray(eps_seq, dtype=np.float64)
    state_terms = np.einsum("ti,tij,tj->t", x[:-1], np.asarray(schedule.Q), x[:-1])
    control_terms = np.einsum("ti,tij,tj->t", u, np.asarray(schedule.R), u)
    terminal = float(x[-1] @ np.asarray(schedule.Q_f) @ x[-1])
    state_stage = float(np.sum(state_terms))
    control_stage = float(np.sum(control_terms))
    total = state_stage + control_stage + terminal
    disturbance_energy = float(np.sum(epsilon**2))
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ np.array([1.0, 0.0], dtype=np.float64)
    pos_abs = pos + np.asarray(TARGET_POS)[None, :]
    return {
        "state_stage": state_stage,
        "control_stage": control_stage,
        "terminal_state": terminal,
        "total_without_disturbance_penalty": total,
        "disturbance_energy": disturbance_energy,
        "disturbance_l2": float(np.sqrt(max(disturbance_energy, 0.0))),
        "max_abs_epsilon": float(np.max(np.abs(epsilon))),
        "peak_forward_velocity_m_s": float(np.max(forward)),
        "peak_forward_velocity_idx": int(np.argmax(forward)),
        "terminal_position_error_m": float(
            np.linalg.norm(pos_abs[-1] - np.asarray(TARGET_POS))
        ),
    }


def _deterministic_matrices(
    A_joint: np.ndarray,
    G_joint: np.ndarray,
    policy: np.ndarray,
    *,
    adversarial: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = A_joint.shape[0]
    z_dim = A_joint.shape[1]
    m_w = G_joint.shape[1]
    F = policy if adversarial else np.zeros((T, m_w, z_dim), dtype=np.float64)
    M = np.asarray(A_joint, dtype=np.float64) + np.einsum("zw,twk->tzk", G_joint, F)
    b = np.zeros((T, z_dim), dtype=np.float64)
    return M, b, F


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
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
    noise_summary = {
        "sensory_energy": float(np.sum(sensory**2)),
        "motor_state_energy": float(np.sum(motor**2)),
        "process_state_energy": float(np.sum(process**2)),
        "signal_dependent_standard_energy": float(np.sum(standard**2)),
    }
    return M, b, F, noise_summary


def _compare_scalar(observed: float, expected: float) -> dict[str, float]:
    abs_diff = float(observed - expected)
    denom = max(abs(float(expected)), 1e-300)
    return {
        "rollout": float(expected),
        "recursion": float(observed),
        "abs_diff": abs(abs_diff),
        "signed_diff": abs_diff,
        "relative_diff": abs(abs_diff) / denom,
    }


def _mode_result(
    mode: str,
    plant,
    schedule,
    gains: np.ndarray,
    stage_costs: np.ndarray,
    terminal_cost: np.ndarray,
    z0: np.ndarray,
    rollout_payload: dict[str, Any],
    clean_M: np.ndarray,
    clean_b: np.ndarray,
    clean_F: np.ndarray,
    adversarial_M: np.ndarray,
    adversarial_b: np.ndarray,
    adversarial_F: np.ndarray,
    noise_summary: dict[str, float] | None = None,
) -> dict[str, Any]:
    clean_value = _value_recursion(clean_M, clean_b, stage_costs, terminal_cost, z0)
    adversarial_value = _value_recursion(
        adversarial_M, adversarial_b, stage_costs, terminal_cost, z0
    )
    clean_forward = _forward_components(
        plant, schedule, gains, clean_F, clean_M, clean_b, z0
    )
    adversarial_forward = _forward_components(
        plant, schedule, gains, adversarial_F, adversarial_M, adversarial_b, z0
    )
    clean_cost = clean_value["total_without_disturbance_penalty"]
    adversarial_cost = adversarial_value["total_without_disturbance_penalty"]
    paired_damage = adversarial_cost - clean_cost
    rollout_clean = rollout_payload["clean"]["cost"]["total_without_disturbance_penalty"]
    rollout_adversarial = rollout_payload["adversarial"]["cost"][
        "total_without_disturbance_penalty"
    ]
    rollout_damage = rollout_payload["paired_damage"]
    return {
        "mode": mode,
        "clean": {
            "value_recursion": clean_value,
            "forward_components": clean_forward,
            "cost_comparison": _compare_scalar(clean_cost, rollout_clean),
            "forward_vs_value_abs_diff": abs(
                clean_forward["total_without_disturbance_penalty"] - clean_cost
            ),
        },
        "adversarial": {
            "value_recursion": adversarial_value,
            "forward_components": adversarial_forward,
            "cost_comparison": _compare_scalar(adversarial_cost, rollout_adversarial),
            "forward_vs_value_abs_diff": abs(
                adversarial_forward["total_without_disturbance_penalty"] - adversarial_cost
            ),
            "disturbance_energy_comparison": _compare_scalar(
                adversarial_forward["disturbance_energy"],
                rollout_payload["adversarial"]["cost"]["disturbance_energy"],
            ),
        },
        "paired_damage": {
            "rollout": float(rollout_damage),
            "recursion": float(paired_damage),
            "abs_diff": abs(float(paired_damage - rollout_damage)),
            "signed_diff": float(paired_damage - rollout_damage),
            "relative_diff": abs(float(paired_damage - rollout_damage))
            / max(abs(float(rollout_damage)), 1e-300),
        },
        "noise_summary": noise_summary,
    }


def _teacher_package_check(plant, schedule, solution, x0, gains, covs) -> dict[str, Any]:
    if not TEACHER_PACKAGE.exists():
        return {"status": "missing", "path": str(TEACHER_PACKAGE)}
    expected = {
        "plant_A": np.asarray(plant.A),
        "plant_B": np.asarray(plant.B),
        "plant_Bw": np.asarray(plant.Bw),
        "schedule_Q": np.asarray(schedule.Q),
        "schedule_R": np.asarray(schedule.R),
        "schedule_Q_f": np.asarray(schedule.Q_f),
        "x0": np.asarray(x0),
        "gamma_star": np.asarray(find_gamma_star(plant, schedule)),
        "observation_matrix": np.asarray(delayed_observation_matrix(plant, OutputFeedbackConfig(n_phys=6))),
    }
    checks = {}
    with np.load(TEACHER_PACKAGE, allow_pickle=False) as data:
        for key, value in expected.items():
            observed = np.asarray(data[key])
            diff = np.max(np.abs(observed - value))
            checks[key] = {
                "max_abs_error": float(diff),
                "allclose_rtol_atol_1e-9": bool(np.allclose(observed, value, rtol=1e-9, atol=1e-9)),
            }
    return {
        "status": "checked_beta_invariant_geometry_only",
        "path": str(TEACHER_PACKAGE),
        "note": (
            "The stored package is the beta-1.4 teacher. For beta 1.05, only the shared "
            "plant/schedule/x0/gamma_star/observation geometry is compared; gamma, Riccati "
            "P, estimator covariances, controller gains, and adversary policy are recomputed live."
        ),
        "recomputed_beta_specific_shapes": {
            "hinf_controller_gains": list(gains.shape),
            "hinf_estimator_covariances": list(covs.shape),
            "hinf_P": list(solution.P.shape),
        },
        "checks": checks,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Output-feedback analytical damage recursion comparison at beta 1.05",
        "",
        "## Contract",
        "",
        "- Reference: 6D no-integrator output-feedback H-infinity teacher.",
        f"- Gamma: {payload['contract']['gamma']:.12g} = "
        f"{payload['contract']['gamma_factor']:.12g} * gamma_star "
        f"({payload['contract']['gamma_star']:.12g}).",
        "- Joint state: `z = [x, xhat]`; clean condition uses `F = 0`; "
        "adversarial condition uses `epsilon_t = F_t z_t`.",
        "- Reported damage is task cost only: Q/R/Qf state/control/terminal cost, "
        "with no `gamma^2 ||epsilon||^2` subtraction.",
        "- Deterministic recursion is exact quadratic cost-to-go. The noisy "
        "single-seed recursion uses the same seed and implements the known "
        "time-varying signal-dependent multiplicative term plus affine sensory, "
        "motor, and process noise offsets.",
        "",
        "## Headline Comparison",
        "",
        "| Mode | Quantity | Rollout | Recursion | Abs diff | Relative diff |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode, result in payload["comparisons"].items():
        for quantity, block in (
            ("clean_cost", result["clean"]["cost_comparison"]),
            ("adversarial_cost", result["adversarial"]["cost_comparison"]),
            ("paired_damage", result["paired_damage"]),
        ):
            lines.append(
                f"| {mode} | {quantity} | {block['rollout']:.12g} | "
                f"{block['recursion']:.12g} | {block['abs_diff']:.12g} | "
                f"{block['relative_diff']:.12g} |"
            )
    lines.extend(
        [
            "",
            "## Disturbance Energy",
            "",
            "| Mode | Rollout | Recursion | Abs diff | Relative diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode, result in payload["comparisons"].items():
        block = result["adversarial"]["disturbance_energy_comparison"]
        lines.append(
            f"| {mode} | {block['rollout']:.12g} | {block['recursion']:.12g} | "
            f"{block['abs_diff']:.12g} | {block['relative_diff']:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"- Match status: `{payload['assessment']['match_status']}`.",
            f"- Second iteration recommended: `{payload['assessment']['second_iteration_recommended']}`.",
            f"- Reason: {payload['assessment']['reason']}",
            "",
            "## Verification Notes",
            "",
            f"- Teacher package check: `{payload['teacher_package_verification']['status']}`.",
            f"- Maximum absolute mismatch observed across headline values: "
            f"{payload['assessment']['max_headline_abs_diff']:.12g}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the shared recursion comparison for the beta=1.05 payload."""

    _run_recursion_damage(
        rollout_json_path=ROLLOUT_JSON_PATH,
        json_path=JSON_PATH,
        markdown_path=MD_PATH,
        default_gamma_factor=BETA_GAMMA_FACTOR,
        script_path="results/08483d5/scripts/compute_output_feedback_recursion_damage_beta1p05.py",
    )


if __name__ == "__main__":
    main()
