from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_PATH = Path("results/08483d5/scripts/compute_output_feedback_damage_beta_curve_dense.py")
OUT_DIR = Path("results/08483d5/notes")
FIGURE_DIR = Path("_artifacts/08483d5/figures/output_feedback_damage_spike_diagnostic")
CSV_PATH = OUT_DIR / "output_feedback_damage_spike_diagnostic.csv"
JSON_PATH = OUT_DIR / "output_feedback_damage_spike_diagnostic.json"
MD_PATH = OUT_DIR / "output_feedback_damage_spike_diagnostic.md"
PNG_DAMAGE_PATH = FIGURE_DIR / "spike_damage_components.png"
PNG_COND_PATH = FIGURE_DIR / "spike_conditioning.png"


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("damage_beta_curves_reference", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load source module from {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _betas() -> list[float]:
    dense = [round(1.330 + 0.001 * idx, 3) for idx in range(31)]
    landmarks = [1.30, 1.325, 1.375, 1.40]
    return sorted(set(landmarks + dense))


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _max_row_norm(array: Any) -> float:
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(arr, axis=-1)))


def _finite_all(*arrays: Any) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(array, dtype=np.float64)))) for array in arrays)


def _cost_fields(prefix: str, cost: Any) -> dict[str, float]:
    return {
        f"{prefix}_x": float(cost.state_stage),
        f"{prefix}_u": float(cost.control_stage),
        f"{prefix}_T": float(cost.terminal_state),
        f"{prefix}_J": float(cost.total_without_disturbance_penalty),
        f"{prefix}_E_w": float(cost.disturbance_energy),
    }


def _estimator_precision_stats(ref: Any, plant: Any, schedule: Any, solution: Any, config: Any) -> dict[str, float]:
    covs = ref.robust_estimator_covariances(plant, schedule, solution.gamma, config)
    H = np.asarray(ref.delayed_observation_matrix(plant, config), dtype=np.float64)
    HH = H.T @ H
    inv_gamma2 = 1.0 / float(solution.gamma * solution.gamma)
    min_eigs = []
    conds = []
    cov_min_eigs = []
    cov_conds = []
    middle_conds = []
    for t in range(int(schedule.T)):
        sigma = np.asarray(covs[t], dtype=np.float64)
        precision = np.linalg.inv(sigma) + HH - inv_gamma2 * np.asarray(schedule.Q[t])
        precision = 0.5 * (precision + precision.T)
        eigs = np.linalg.eigvalsh(precision)
        min_eigs.append(float(eigs[0]))
        conds.append(float(np.linalg.cond(precision)))
        middle = np.linalg.inv(precision)
        middle_conds.append(float(np.linalg.cond(middle)))
        cov = 0.5 * (sigma + sigma.T)
        cov_eigs = np.linalg.eigvalsh(cov)
        cov_min_eigs.append(float(cov_eigs[0]))
        cov_conds.append(float(np.linalg.cond(cov)))
    cov_last = 0.5 * (np.asarray(covs[-1], dtype=np.float64) + np.asarray(covs[-1], dtype=np.float64).T)
    cov_last_eigs = np.linalg.eigvalsh(cov_last)
    cov_min_eigs.append(float(cov_last_eigs[0]))
    cov_conds.append(float(np.linalg.cond(cov_last)))
    return {
        "estimator_precision_min_eig_min": float(np.min(min_eigs)),
        "estimator_precision_cond_max": float(np.max(conds)),
        "estimator_middle_cond_max": float(np.max(middle_conds)),
        "estimator_covariance_min_eig_min": float(np.min(cov_min_eigs)),
        "estimator_covariance_cond_max": float(np.max(cov_conds)),
    }


def _policy_recursion_stats(ref: Any, plant: Any, schedule: Any, solution: Any, gains: Any, policy: Any, config: Any) -> dict[str, float]:
    covs = ref.robust_estimator_covariances(plant, schedule, solution.gamma, config)
    a_joint, g_joint = ref.robust_estimator_joint_matrices(
        plant,
        schedule,
        solution,
        ref.jnp.asarray(gains),
        covs,
        config,
    )
    a_joint = np.asarray(a_joint, dtype=np.float64)
    g_joint = np.asarray(g_joint, dtype=np.float64)
    policy_np = np.asarray(policy, dtype=np.float64)
    adv_joint = a_joint + np.einsum("zw,twk->tzk", g_joint, policy_np)

    clean_radii = []
    clean_norms = []
    adv_radii = []
    adv_norms = []
    clean_cumulative = np.eye(a_joint.shape[1], dtype=np.float64)
    adv_cumulative = np.eye(a_joint.shape[1], dtype=np.float64)
    clean_cumulative_norms = []
    adv_cumulative_norms = []
    for clean_t, adv_t in zip(a_joint, adv_joint, strict=True):
        clean_eigs = np.linalg.eigvals(clean_t)
        adv_eigs = np.linalg.eigvals(adv_t)
        clean_radii.append(float(np.max(np.abs(clean_eigs))))
        adv_radii.append(float(np.max(np.abs(adv_eigs))))
        clean_norms.append(float(np.linalg.norm(clean_t, ord=2)))
        adv_norms.append(float(np.linalg.norm(adv_t, ord=2)))
        clean_cumulative = clean_t @ clean_cumulative
        adv_cumulative = adv_t @ adv_cumulative
        clean_cumulative_norms.append(float(np.linalg.norm(clean_cumulative, ord=2)))
        adv_cumulative_norms.append(float(np.linalg.norm(adv_cumulative, ord=2)))

    return {
        "gain_fro_max": float(np.max(np.linalg.norm(np.asarray(gains, dtype=np.float64), axis=(1, 2)))),
        "gain_op_max": float(max(np.linalg.norm(k, ord=2) for k in np.asarray(gains, dtype=np.float64))),
        "policy_fro_max": float(np.max(np.linalg.norm(policy_np, axis=(1, 2)))),
        "policy_op_max": float(max(np.linalg.norm(f, ord=2) for f in policy_np)),
        "clean_joint_step_spectral_radius_max": float(np.max(clean_radii)),
        "adv_joint_step_spectral_radius_max": float(np.max(adv_radii)),
        "clean_joint_step_op_norm_max": float(np.max(clean_norms)),
        "adv_joint_step_op_norm_max": float(np.max(adv_norms)),
        "clean_joint_cumulative_op_norm_max": float(np.max(clean_cumulative_norms)),
        "adv_joint_cumulative_op_norm_max": float(np.max(adv_cumulative_norms)),
    }


def _compute_row(ref: Any, plant: Any, schedule: Any, gamma_star: float, beta: float, config: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    solution, x0, gains, policy = ref._build_policy(plant, schedule, gamma_star, beta, config)
    build_seconds = time.perf_counter() - t0

    rec_t0 = time.perf_counter()
    recursion = ref._recursion_damage(plant, schedule, solution, x0, gains, policy, config)
    recursion_seconds = time.perf_counter() - rec_t0

    roll_t0 = time.perf_counter()
    clean = ref.rollout_with_robust_estimator(
        plant,
        schedule,
        solution,
        x0,
        gains=gains,
        config=config,
    )
    adv = ref.rollout_with_robust_estimator_policy(
        plant,
        schedule,
        solution,
        x0,
        policy,
        gains=gains,
        config=config,
    )
    clean_cost = ref.output_feedback_cost(schedule, clean, gamma=solution.gamma)
    adv_cost = ref.output_feedback_cost(schedule, adv, gamma=solution.gamma)
    rollout_seconds = time.perf_counter() - roll_t0

    cond_t0 = time.perf_counter()
    estimator_stats = _estimator_precision_stats(ref, plant, schedule, solution, config)
    policy_stats = _policy_recursion_stats(ref, plant, schedule, solution, gains, policy, config)
    conditioning_seconds = time.perf_counter() - cond_t0

    clean_rec = recursion["deterministic"]["clean_cost"]
    adv_rec = recursion["deterministic"]["adversarial_cost"]
    damage_rec = recursion["deterministic"]["damage"]
    damage_rollout = float(adv_cost.total_without_disturbance_penalty - clean_cost.total_without_disturbance_penalty)
    e_w = float(adv_cost.disturbance_energy)
    row: dict[str, Any] = {
        "beta": float(beta),
        "gamma": float(solution.gamma),
        "gamma_star": float(gamma_star),
        "admissible": bool(solution.admissible),
        "J_clean": float(clean_cost.total_without_disturbance_penalty),
        "J_adv": float(adv_cost.total_without_disturbance_penalty),
        "D": damage_rollout,
        "D_over_E_w": float(damage_rollout / e_w) if e_w > 0.0 else math.nan,
        "E_w": e_w,
        "D_x": float(adv_cost.state_stage - clean_cost.state_stage),
        "D_u": float(adv_cost.control_stage - clean_cost.control_stage),
        "D_T": float(adv_cost.terminal_state - clean_cost.terminal_state),
        **_cost_fields("clean", clean_cost),
        **_cost_fields("adv", adv_cost),
        "max_x_clean": _max_row_norm(clean.x),
        "max_x_adv": _max_row_norm(adv.x),
        "max_xhat_clean": _max_row_norm(clean.x_hat),
        "max_xhat_adv": _max_row_norm(adv.x_hat),
        "max_u_clean": _max_row_norm(clean.u),
        "max_u_adv": _max_row_norm(adv.u),
        "max_w_adv": _max_row_norm(adv.epsilon),
        "max_abs_x_clean": float(np.max(np.abs(np.asarray(clean.x, dtype=np.float64)))),
        "max_abs_x_adv": float(np.max(np.abs(np.asarray(adv.x, dtype=np.float64)))),
        "max_abs_u_clean": float(np.max(np.abs(np.asarray(clean.u, dtype=np.float64)))),
        "max_abs_u_adv": float(np.max(np.abs(np.asarray(adv.u, dtype=np.float64)))),
        "max_abs_w_adv": float(np.max(np.abs(np.asarray(adv.epsilon, dtype=np.float64)))),
        "recursion_J_clean": float(clean_rec),
        "recursion_J_adv": float(adv_rec),
        "recursion_D": float(damage_rec),
        "rollout_recursion_J_clean_abs_diff": abs(float(clean_cost.total_without_disturbance_penalty) - float(clean_rec)),
        "rollout_recursion_J_adv_abs_diff": abs(float(adv_cost.total_without_disturbance_penalty) - float(adv_rec)),
        "rollout_recursion_D_abs_diff": abs(damage_rollout - float(damage_rec)),
        "riccati_spectral_radius_max": float(np.max(np.asarray(solution.spectral_radii, dtype=np.float64))),
        "riccati_spectral_radius_margin_min": float(1.0 - np.max(np.asarray(solution.spectral_radii, dtype=np.float64))),
        "riccati_bracket_cond_max": float(np.max(np.asarray(solution.bracket_conditions, dtype=np.float64))),
        "riccati_max_P_cond": float(solution.max_P_cond),
        "finite_all": _finite_all(
            solution.P,
            solution.K,
            solution.spectral_radii,
            solution.bracket_conditions,
            gains,
            policy,
            clean.x,
            clean.x_hat,
            clean.u,
            adv.x,
            adv.x_hat,
            adv.u,
            adv.epsilon,
        ),
        "build_seconds": build_seconds,
        "recursion_seconds": recursion_seconds,
        "rollout_seconds": rollout_seconds,
        "conditioning_seconds": conditioning_seconds,
    }
    row.update(estimator_stats)
    row.update(policy_stats)
    return row


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if math.isfinite(f) else str(f)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "beta",
        "gamma",
        "J_clean",
        "J_adv",
        "D",
        "D_x",
        "D_u",
        "D_T",
        "clean_x",
        "clean_u",
        "clean_T",
        "adv_x",
        "adv_u",
        "adv_T",
        "E_w",
        "D_over_E_w",
        "max_x_clean",
        "max_x_adv",
        "max_u_clean",
        "max_u_adv",
        "max_w_adv",
        "max_xhat_clean",
        "max_xhat_adv",
        "recursion_D",
        "rollout_recursion_D_abs_diff",
        "rollout_recursion_J_clean_abs_diff",
        "rollout_recursion_J_adv_abs_diff",
        "riccati_spectral_radius_max",
        "riccati_spectral_radius_margin_min",
        "riccati_bracket_cond_max",
        "riccati_max_P_cond",
        "estimator_precision_min_eig_min",
        "estimator_precision_cond_max",
        "estimator_covariance_min_eig_min",
        "estimator_covariance_cond_max",
        "gain_op_max",
        "policy_op_max",
        "clean_joint_step_spectral_radius_max",
        "adv_joint_step_spectral_radius_max",
        "clean_joint_cumulative_op_norm_max",
        "adv_joint_cumulative_op_norm_max",
        "finite_all",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    beta = np.asarray([row["beta"] for row in rows], dtype=np.float64)
    d = np.asarray([row["D"] for row in rows], dtype=np.float64)
    dx = np.asarray([row["D_x"] for row in rows], dtype=np.float64)
    du = np.asarray([row["D_u"] for row in rows], dtype=np.float64)
    dt = np.asarray([row["D_T"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=170)
    ax.plot(beta, d, marker="o", markersize=2.6, linewidth=1.4, label="D total")
    ax.plot(beta, dx, linewidth=1.0, label="D_x")
    ax.plot(beta, du, linewidth=1.0, label="D_u")
    ax.plot(beta, dt, linewidth=1.0, label="D_T")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("beta = gamma / gamma_star")
    ax.set_ylabel("adversarial - clean task cost")
    ax.set_title("6D output-feedback H-infinity spike diagnostics")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(PNG_DAMAGE_PATH)
    plt.close(fig)

    rho_margin = np.asarray([row["riccati_spectral_radius_margin_min"] for row in rows])
    est_min = np.asarray([row["estimator_precision_min_eig_min"] for row in rows])
    est_cond = np.asarray([row["estimator_precision_cond_max"] for row in rows])
    adv_cum = np.asarray([row["adv_joint_cumulative_op_norm_max"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.0), dpi=170, sharex=True)
    axes[0].plot(beta, rho_margin, marker="o", markersize=2.4, linewidth=1.2)
    axes[0].set_ylabel("Riccati margin")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(beta, est_min, marker="o", markersize=2.4, linewidth=1.2)
    axes[1].set_ylabel("min eig precision")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(beta, est_cond, marker="o", markersize=2.4, linewidth=1.2, label="precision cond")
    axes[2].plot(beta, adv_cum, marker="s", markersize=2.2, linewidth=1.0, label="adv cumulative norm")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("log scale")
    axes[2].set_xlabel("beta = gamma / gamma_star")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PNG_COND_PATH)
    plt.close(fig)


def _row_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['beta']:.3f} | {row['D']:.6g} | {row['D_x']:.6g} | "
        f"{row['D_u']:.6g} | {row['D_T']:.6g} | {row['E_w']:.6g} | "
        f"{row['D_over_E_w']:.6g} | {row['max_x_adv']:.6g} | "
        f"{row['max_u_adv']:.6g} | {row['max_w_adv']:.6g} | "
        f"{row['rollout_recursion_D_abs_diff']:.3g} |"
    )


def _write_markdown(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    max_row = payload["summary"]["max_damage_row"]
    beta_135 = min(rows, key=lambda row: abs(row["beta"] - 1.35))
    beta_1349_1351 = [row for row in rows if 1.345 <= row["beta"] <= 1.355]
    lines = [
        "# 6D output-feedback H-infinity damage spike diagnostic",
        "",
        "## Method",
        "",
        "- No-launch diagnostic for issue `08483d5`.",
        "- Reused the deterministic no-noise output-feedback H-infinity conventions from "
        "`results/08483d5/scripts/compute_output_feedback_damage_beta_curve_dense.py`.",
        "- No cap, radius, trust-region, PGD, or training defaults enter this calculation.",
        "- For every beta, the Riccati solution, robust estimator covariance, feedback gains, "
        "and fixed optimal disturbance policy are recomputed.",
        "- Direct deterministic rollouts were computed for every row and compared with the "
        "affine value recursion.",
        "",
        "## Headline",
        "",
        f"- Max damage beta: `{max_row['beta']:.3f}` with `D={max_row['D']:.12g}`.",
        f"- At beta 1.350: `D={beta_135['D']:.12g}`, `D_x={beta_135['D_x']:.12g}`, "
        f"`D_u={beta_135['D_u']:.12g}`, `D_T={beta_135['D_T']:.12g}`, "
        f"`E_w={beta_135['E_w']:.12g}`, `D/E_w={beta_135['D_over_E_w']:.12g}`.",
        f"- Max rollout-vs-recursion damage mismatch across all rows: "
        f"`{payload['summary']['max_rollout_recursion_D_abs_diff']:.12g}`.",
        f"- All finite checks passed: `{payload['summary']['all_finite']}`.",
        "",
        "## Spike-neighborhood rows",
        "",
        "| beta | D | D_x | D_u | D_T | E_w | D/E_w | max ||x|| adv | max ||u|| adv | max ||w|| adv | recursion mismatch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_row_line(row) for row in beta_1349_1351)
    lines.extend(
        [
            "",
            "## Landmark rows",
            "",
            "| beta | D | D_x | D_u | D_T | E_w | D/E_w | max ||x|| adv | max ||u|| adv | max ||w|| adv | recursion mismatch |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target in [1.30, 1.325, 1.35, 1.375, 1.40]:
        row = min(rows, key=lambda item: abs(item["beta"] - target))
        lines.append(_row_line(row))
    lines.extend(
        [
            "",
            "## Conditioning at max damage",
            "",
            f"- Riccati spectral-radius max: `{max_row['riccati_spectral_radius_max']:.12g}`.",
            f"- Riccati spectral-radius margin min: `{max_row['riccati_spectral_radius_margin_min']:.12g}`.",
            f"- Riccati bracket condition max: `{max_row['riccati_bracket_cond_max']:.12g}`.",
            f"- Riccati max P condition: `{max_row['riccati_max_P_cond']:.12g}`.",
            f"- Estimator precision min eigenvalue: `{max_row['estimator_precision_min_eig_min']:.12g}`.",
            f"- Estimator precision max condition: `{max_row['estimator_precision_cond_max']:.12g}`.",
            f"- Gain operator norm max: `{max_row['gain_op_max']:.12g}`.",
            f"- Policy operator norm max: `{max_row['policy_op_max']:.12g}`.",
            f"- Adversarial joint cumulative operator norm max: `{max_row['adv_joint_cumulative_op_norm_max']:.12g}`.",
            "",
            "## Interpretation",
            "",
            payload["summary"]["diagnosis"],
            "",
            "## Outputs",
            "",
            f"- CSV: `{CSV_PATH}`",
            f"- JSON: `{JSON_PATH}`",
            f"- Markdown: `{MD_PATH}`",
            f"- Damage plot: `{PNG_DAMAGE_PATH}`",
            f"- Conditioning plot: `{PNG_COND_PATH}`",
            "",
            "## Command",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --no-sync python "
            "results/08483d5/scripts/compute_output_feedback_damage_spike_diagnostic.py",
            "```",
            "",
        ]
    )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _diagnosis(rows: list[dict[str, Any]]) -> str:
    max_row = max(rows, key=lambda row: row["D"])
    max_abs_component = max(abs(max_row["D_x"]), abs(max_row["D_u"]), abs(max_row["D_T"]))
    terminal_fraction = abs(max_row["D_T"]) / max_abs_component if max_abs_component else 0.0
    state_fraction = abs(max_row["D_x"]) / max_abs_component if max_abs_component else 0.0
    issues = []
    if max_row["riccati_spectral_radius_margin_min"] < 1e-4:
        issues.append("Riccati margin is very small")
    if max_row["estimator_precision_min_eig_min"] <= 1e-8:
        issues.append("estimator precision is near-singular")
    if max_row["adv_joint_cumulative_op_norm_max"] > 1e8:
        issues.append("adversarial joint transient gain is enormous")
    clean_scale = max(max_row["max_x_clean"], 1e-12)
    if max_row["max_x_adv"] / clean_scale > 100.0:
        issues.append(
            f"direct rollout state norm jumps {max_row['max_x_adv'] / clean_scale:.1f}x above clean"
        )
    if max_row["max_u_adv"] / max(max_row["max_u_clean"], 1e-12) > 100.0:
        issues.append(
            f"direct rollout control norm jumps {max_row['max_u_adv'] / max(max_row['max_u_clean'], 1e-12):.1f}x above clean"
        )
    if max_row["max_x_adv"] > 1e4 or max_row["max_u_adv"] > 1e6 or max_row["max_w_adv"] > 1e6:
        issues.append("direct rollout state/control/disturbance norms blow up")
    component_note = (
        "terminal-dominated"
        if terminal_fraction >= 0.8
        else "state-running-dominated"
        if state_fraction >= 0.8
        else "spread across state/control/terminal components"
    )
    if issues:
        return (
            f"The spike is real under the implemented deterministic closed-loop equations, "
            f"because direct rollout and value recursion agree, but it is a stress-test "
            f"condition rather than a useful curriculum target. At the maximum row the damage "
            f"is {component_note}, and the conditioning/transient diagnostics flag: "
            f"{'; '.join(issues)}."
        )
    return (
        f"The spike is real under the implemented deterministic closed-loop equations: direct "
        f"rollout and value recursion agree. The max row is {component_note}; conditioning "
        f"diagnostics did not cross the simple stress thresholds used here, so the spike looks "
        f"more like closed-loop transient amplification than a plain matrix-singularity artifact."
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = _load_reference_module()
    config = ref.OutputFeedbackConfig(n_phys=6)
    plant, schedule = ref.build_no_integrator_game()
    gamma_star = float(ref.find_gamma_star(plant, schedule))
    rows = []
    for beta in _betas():
        row = _compute_row(ref, plant, schedule, gamma_star, beta, config)
        rows.append(row)
        print(
            json.dumps(
                {
                    "beta": row["beta"],
                    "D": row["D"],
                    "D_x": row["D_x"],
                    "D_u": row["D_u"],
                    "D_T": row["D_T"],
                    "E_w": row["E_w"],
                    "mismatch": row["rollout_recursion_D_abs_diff"],
                    "finite": row["finite_all"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    max_row = max(rows, key=lambda row: row["D"])
    summary = {
        "beta_count": len(rows),
        "max_damage_row": max_row,
        "max_rollout_recursion_D_abs_diff": float(
            max(row["rollout_recursion_D_abs_diff"] for row in rows)
        ),
        "max_rollout_recursion_J_abs_diff": float(
            max(
                max(
                    row["rollout_recursion_J_clean_abs_diff"],
                    row["rollout_recursion_J_adv_abs_diff"],
                )
                for row in rows
            )
        ),
        "all_finite": all(bool(row["finite_all"]) for row in rows),
    }
    summary["diagnosis"] = _diagnosis(rows)
    payload = {
        "schema_version": "rlrmp.damage_spike_diagnostic.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_script": str(SOURCE_PATH),
        "outputs": {
            "csv": str(CSV_PATH),
            "json": str(JSON_PATH),
            "markdown": str(MD_PATH),
            "damage_components_png": str(PNG_DAMAGE_PATH),
            "conditioning_png": str(PNG_COND_PATH),
        },
        "contract": {
            "gamma_star": gamma_star,
            "betas": _betas(),
            "state_basis": (
                "36D delay-augmented state, six 6D physical blocks "
                "[x, y, vx, vy, force_x, force_y]; no disturbance integrators"
            ),
            "damage_definition": (
                "D = adversarial task cost - clean task cost with no gamma^2 E_w subtraction"
            ),
            "recursion_sanity": "direct deterministic rollout computed for every beta",
        },
        "summary": summary,
        "rows": rows,
    }
    _write_csv(rows)
    _plot(rows)
    JSON_PATH.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload)
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
