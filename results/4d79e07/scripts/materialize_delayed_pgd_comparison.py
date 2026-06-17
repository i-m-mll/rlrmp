"""Materialize delayed-reach PGD vs non-PGD comparison notes.

This issue-local driver consumes the compact manifests emitted by the standard
GRU post-run materializers and writes a tracked Markdown/JSON comparison plus
an ignored CSV table of class-binned perturbation-response deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p


EXPERIMENT = "4d79e07"
PGD_RUN_ID = "delayed_movement_bank_pgd_clip5"
BASELINE_EXPERIMENT = "6c36536"
BASELINE_RUN_ID = "delayed_movement_bank"
OUTPUT_TAG = "delayed_movement_bank_pgd_clip5_final_checkpoints_calibrated"
BASELINE_OUTPUT_TAG = "delayed_movement_bank_final_checkpoints_calibrated"
COMPARISON_TOPIC = "delayed_movement_bank_pgd_clip5_vs_delayed_movement_bank"


@dataclass(frozen=True)
class PathBundle:
    """Resolved paths used by the materializer."""

    pgd_run_spec: Path
    baseline_run_spec: Path
    training_diagnostics_npz: Path
    pgd_standard_manifest: Path
    baseline_standard_manifest: Path
    pgd_evaluation_manifest: Path
    pgd_postrun_manifest: Path
    pgd_perturbation_detail: Path
    baseline_perturbation_detail: Path
    pgd_velocity_no_catch: Path
    pgd_velocity_catch: Path
    baseline_velocity_no_catch: Path
    baseline_velocity_catch: Path
    pgd_figure_summary: Path
    comparison_csv: Path
    comparison_manifest: Path
    comparison_note: Path
    perturbation_bulk_dir: Path
    perturbation_raw_dir: Path


def _paths(repo_root: Path) -> PathBundle:
    return PathBundle(
        pgd_run_spec=repo_root / "results" / EXPERIMENT / "runs" / f"{PGD_RUN_ID}.json",
        baseline_run_spec=repo_root
        / "results"
        / BASELINE_EXPERIMENT
        / "runs"
        / f"{BASELINE_RUN_ID}.json",
        training_diagnostics_npz=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "runs"
        / PGD_RUN_ID
        / "artifacts"
        / "training_diagnostics.npz",
        pgd_standard_manifest=repo_root
        / "results"
        / EXPERIMENT
        / "notes"
        / f"gru_standard_certificates_{OUTPUT_TAG}_manifest.json",
        baseline_standard_manifest=repo_root
        / "results"
        / BASELINE_EXPERIMENT
        / "notes"
        / f"gru_standard_certificates_{BASELINE_OUTPUT_TAG}_manifest.json",
        pgd_evaluation_manifest=repo_root
        / "results"
        / EXPERIMENT
        / "notes"
        / f"gru_evaluation_diagnostics_{OUTPUT_TAG}.json",
        pgd_postrun_manifest=repo_root
        / "results"
        / EXPERIMENT
        / "notes"
        / f"gru_postrun_materialization_{OUTPUT_TAG}.json",
        pgd_perturbation_detail=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "perturbation_response"
        / f"gru_{OUTPUT_TAG}"
        / f"gru_perturbation_response_{OUTPUT_TAG}_manifest_detail.json",
        baseline_perturbation_detail=repo_root
        / "_artifacts"
        / BASELINE_EXPERIMENT
        / "perturbation_response"
        / f"gru_{BASELINE_OUTPUT_TAG}"
        / f"gru_perturbation_response_{BASELINE_OUTPUT_TAG}_manifest_detail.json",
        pgd_velocity_no_catch=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "figures"
        / "delayed_movement_bank_pgd_clip5_velocity_profiles"
        / "no_catch"
        / "velocity_profile_summary.json",
        pgd_velocity_catch=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "figures"
        / "delayed_movement_bank_pgd_clip5_velocity_profiles"
        / "catch"
        / "velocity_profile_summary.json",
        baseline_velocity_no_catch=repo_root
        / "_artifacts"
        / BASELINE_EXPERIMENT
        / "figures"
        / "delayed_movement_bank_velocity_profiles"
        / "no_catch"
        / "velocity_profile_summary.json",
        baseline_velocity_catch=repo_root
        / "_artifacts"
        / BASELINE_EXPERIMENT
        / "figures"
        / "delayed_movement_bank_velocity_profiles"
        / "catch"
        / "velocity_profile_summary.json",
        pgd_figure_summary=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "figures"
        / f"gru_postrun_{OUTPUT_TAG}"
        / "figure_summary.json",
        comparison_csv=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "comparisons"
        / COMPARISON_TOPIC
        / "perturbation_class_comparison.csv",
        comparison_manifest=repo_root
        / "results"
        / EXPERIMENT
        / "notes"
        / f"{COMPARISON_TOPIC}_manifest.json",
        comparison_note=repo_root
        / "results"
        / EXPERIMENT
        / "notes"
        / f"{COMPARISON_TOPIC}.md",
        perturbation_bulk_dir=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "perturbation_response"
        / f"gru_{OUTPUT_TAG}",
        perturbation_raw_dir=repo_root
        / "_artifacts"
        / EXPERIMENT
        / "perturbation_response"
        / f"gru_{OUTPUT_TAG}"
        / PGD_RUN_ID,
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _finite_values(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _last_finite_stats(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim == 1:
        rows = arr[:, None]
    else:
        rows = arr.reshape(arr.shape[0], -1)
    for idx in range(rows.shape[0] - 1, -1, -1):
        values = _finite_values(rows[idx])
        if values.size:
            return {
                "batch_index": int(np.asarray(data["batch_index"])[idx]),
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return {
        "batch_index": None,
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
    }


def _training_diagnostics(path: Path, expected_batches: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        latest_batch = int(np.asarray(data["batch_index"])[-1])
        summary = {
            "path": path,
            "completed_batches": latest_batch + 1,
            "expected_batches": expected_batches,
            "ok": latest_batch + 1 >= expected_batches,
            "latest_batch_index": latest_batch,
            "train_loss_total": _last_finite_stats(data, "train_loss__total"),
            "validation_loss_total": _last_finite_stats(data, "validation_loss__total"),
            "optimizer_clipping_fraction": _last_finite_stats(
                data,
                "optimizer_clipping_fraction",
            ),
            "pgd_inner_objective_best": _last_finite_stats(
                data,
                "pgd_broad_epsilon_inner_objective_best",
            ),
            "pgd_inner_objective_improvement": _last_finite_stats(
                data,
                "pgd_broad_epsilon_inner_objective_improvement",
            ),
            "pgd_inner_objective_final_endpoint_gap": _last_finite_stats(
                data,
                "pgd_broad_epsilon_inner_objective_final_endpoint_gap",
            ),
            "pgd_boundary_fraction": _last_finite_stats(
                data,
                "pgd_broad_epsilon_boundary_fraction",
            ),
            "pgd_radius_ratio_mean": _last_finite_stats(
                data,
                "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean",
            ),
            "pgd_n_steps": _last_finite_stats(data, "pgd_broad_epsilon_n_steps"),
            "pgd_step_fraction": _last_finite_stats(
                data,
                "pgd_broad_epsilon_step_size_fraction_of_l2_radius",
            ),
        }
    return summary


def _mean_metric(group: MappingLike, name: str) -> float | None:
    metric = group.get("metrics", {}).get(name, {})
    value = metric.get("mean")
    return _finite_float(value)


def _ratio(group: MappingLike, name: str) -> float | None:
    value = group.get(name, {}).get("ratio_of_means")
    return _finite_float(value)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _ratio_change(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


MappingLike = dict[str, Any]


def _perturbation_groups(manifest: MappingLike, run_id: str) -> MappingLike:
    return (
        manifest["runs"][run_id]
        .get("robust_response_summary", {})
        .get("class_summary", {})
        .get("groups", {})
    )


def _status_counts(manifest: MappingLike, run_id: str) -> MappingLike:
    return manifest["runs"][run_id].get("status_counts", {})


def _comparison_rows(
    pgd_manifest: MappingLike,
    baseline_manifest: MappingLike,
) -> list[dict[str, Any]]:
    pgd_groups = _perturbation_groups(pgd_manifest, PGD_RUN_ID)
    baseline_groups = _perturbation_groups(baseline_manifest, BASELINE_RUN_ID)
    rows: list[dict[str, Any]] = []
    for group_name in sorted(set(pgd_groups) & set(baseline_groups)):
        pgd_group = pgd_groups[group_name]
        baseline_group = baseline_groups[group_name]
        row = {
            "class": group_name,
            "pgd_rows": pgd_group.get("status_counts", {}).get("evaluated", 0),
            "baseline_rows": baseline_group.get("status_counts", {}).get("evaluated", 0),
            "pgd_delta_action_mean": _mean_metric(pgd_group, "delta_action_norm"),
            "baseline_delta_action_mean": _mean_metric(
                baseline_group,
                "delta_action_norm",
            ),
            "pgd_max_delta_position_m": _mean_metric(
                pgd_group,
                "delta_position_response_m.max",
            ),
            "baseline_max_delta_position_m": _mean_metric(
                baseline_group,
                "delta_position_response_m.max",
            ),
            "pgd_auc_delta_position_m_s": _mean_metric(
                pgd_group,
                "delta_position_response_m.auc",
            ),
            "baseline_auc_delta_position_m_s": _mean_metric(
                baseline_group,
                "delta_position_response_m.auc",
            ),
            "pgd_full_qrf_delta_cost": _mean_metric(
                pgd_group,
                "extra_full_qrf_delta_cost_total",
            ),
            "baseline_full_qrf_delta_cost": _mean_metric(
                baseline_group,
                "extra_full_qrf_delta_cost_total",
            ),
            "pgd_extlqg_ratio": _ratio(pgd_group, "gru_extlqg_delta_cost_ratio"),
            "baseline_extlqg_ratio": _ratio(
                baseline_group,
                "gru_extlqg_delta_cost_ratio",
            ),
            "pgd_robust_ratio": _ratio(
                pgd_group,
                "gru_robust_analytical_delta_cost_ratio",
            ),
            "baseline_robust_ratio": _ratio(
                baseline_group,
                "gru_robust_analytical_delta_cost_ratio",
            ),
        }
        row["delta_action_pgd_over_baseline"] = _ratio_change(
            row["pgd_delta_action_mean"],
            row["baseline_delta_action_mean"],
        )
        row["max_position_pgd_over_baseline"] = _ratio_change(
            row["pgd_max_delta_position_m"],
            row["baseline_max_delta_position_m"],
        )
        row["full_qrf_cost_pgd_over_baseline"] = _ratio_change(
            row["pgd_full_qrf_delta_cost"],
            row["baseline_full_qrf_delta_cost"],
        )
        row["extlqg_ratio_pgd_over_baseline"] = _ratio_change(
            row["pgd_extlqg_ratio"],
            row["baseline_extlqg_ratio"],
        )
        row["robust_ratio_pgd_over_baseline"] = _ratio_change(
            row["pgd_robust_ratio"],
            row["baseline_robust_ratio"],
        )
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    mkdir_p(path.parent)
    fieldnames = [
        "class",
        "pgd_rows",
        "baseline_rows",
        "pgd_delta_action_mean",
        "baseline_delta_action_mean",
        "delta_action_pgd_over_baseline",
        "pgd_max_delta_position_m",
        "baseline_max_delta_position_m",
        "max_position_pgd_over_baseline",
        "pgd_auc_delta_position_m_s",
        "baseline_auc_delta_position_m_s",
        "pgd_full_qrf_delta_cost",
        "baseline_full_qrf_delta_cost",
        "full_qrf_cost_pgd_over_baseline",
        "pgd_extlqg_ratio",
        "baseline_extlqg_ratio",
        "extlqg_ratio_pgd_over_baseline",
        "pgd_robust_ratio",
        "baseline_robust_ratio",
        "robust_ratio_pgd_over_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _profile_summary(path: Path) -> MappingLike:
    return _read_json(path).get("profile", {})


def _standard_summary(manifest: MappingLike) -> dict[str, Any]:
    row = manifest["rows"][0]
    failure = manifest.get("failure_decomposition", {}).get("rows", [{}])[0]
    components = {item["name"]: item for item in row.get("certificate_components", [])}
    action_component = components.get("state_weighted_action_mismatch", {})
    action_summary = action_component.get("summary", {})
    return {
        "status": row.get("status"),
        "state_weighted_action_mismatch": action_summary.get(
            "aggregate_mismatch_ratio",
        ),
        "closed_loop_transition_mismatch": row.get("closed_loop_transition_mismatch"),
        "value_policy_gap": row.get("value_policy_gap"),
        "bellman_hessian_residual": row.get("bellman_hessian_residual"),
        "classification": failure.get("classification", {}).get("classification"),
        "blockers": manifest.get("summary", {}).get("blockers", []),
    }


def _behavior_summary(manifest: MappingLike) -> dict[str, Any]:
    behavior = manifest["runs"][PGD_RUN_ID]["behavior"]
    return {
        "endpoint_error_m": behavior.get("endpoint_error_m", {}),
        "overshoot_m": behavior.get("overshoot_m", {}),
        "command_norm": behavior.get("command_norm", {}),
        "command_jerk_norm": behavior.get("command_jerk_norm", {}),
        "first_five_step_command_norm": behavior.get("first_five_step_command_norm", {}),
    }


def _figure_velocity_summary(manifest: MappingLike) -> dict[str, Any]:
    velocity = manifest.get("velocity_profiles", {})
    return {
        "pgd_peak_forward_velocity_m_s": velocity.get("peak_forward_velocity_m_s"),
        "pgd_time_of_peak_forward_velocity_s": velocity.get(
            "time_of_peak_forward_velocity_s",
        ),
        "references": velocity.get("references", {}),
    }


def _tree_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _artifact_size_summary(paths: PathBundle, existing: MappingLike | None) -> dict[str, Any]:
    current = {
        "artifacts_4d79e07_bytes": _tree_size_bytes(
            paths.perturbation_bulk_dir.parents[1],
        ),
        "perturbation_bulk_bytes": _tree_size_bytes(paths.perturbation_bulk_dir),
        "perturbation_raw_npz_count": len(list(paths.perturbation_raw_dir.glob("*.npz"))),
        "perturbation_raw_npz_bytes": sum(
            path.stat().st_size for path in paths.perturbation_raw_dir.glob("*.npz")
        )
        if paths.perturbation_raw_dir.exists()
        else 0,
    }
    current["artifacts_4d79e07_human"] = _human_size(current["artifacts_4d79e07_bytes"])
    current["perturbation_bulk_human"] = _human_size(current["perturbation_bulk_bytes"])
    current["perturbation_raw_npz_human"] = _human_size(
        current["perturbation_raw_npz_bytes"],
    )
    before = None
    if existing:
        before = existing.get("artifact_size_cleanup", {}).get("before_cleanup")
    if before is None and current["perturbation_raw_npz_count"]:
        before = current
    return {
        "before_cleanup": before,
        "current": current,
        "cleanup_completed": bool(
            before and before.get("perturbation_raw_npz_count", 0) > 0
            and current["perturbation_raw_npz_count"] == 0
        ),
    }


def _format_float(value: Any, digits: int = 4) -> str:
    value = _finite_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def _format_percent_change(numerator: float | None, denominator: float | None) -> str:
    ratio = _ratio_change(numerator, denominator)
    if ratio is None:
        return "NA"
    return f"{(ratio - 1.0) * 100.0:+.1f}%"


def _metric_mean(metric: MappingLike) -> float | None:
    return _finite_float(metric.get("mean"))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _render_markdown(manifest: MappingLike) -> str:
    run_contract = manifest["run_contract"]
    training = manifest["training_diagnostics"]
    behavior = manifest["behavior_summary"]
    standard = manifest["standard_certificate"]
    velocity = manifest["velocity_profiles"]
    perturbation_rows = manifest["perturbation_comparison"]["rows"]
    size_summary = manifest["artifact_size_cleanup"]
    residuals = manifest["residual_blockers"]

    lines: list[str] = [
        "# Delayed movement-bank PGD clip5 materialization",
        "",
        "This note materializes the completed PGD delayed-reach row against the",
        "non-PGD delayed movement-bank baseline on the same final-checkpoint,",
        "calibrated perturbation-response lens.",
        "",
        "## Run contract",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["field", "PGD row", "baseline"],
            [
                ["run", PGD_RUN_ID, BASELINE_RUN_ID],
                ["issue", EXPERIMENT, BASELINE_EXPERIMENT],
                ["PGD enabled", str(run_contract["pgd_enabled"]), "False"],
                ["PGD support", run_contract["pgd_support"], "not used"],
                ["budget scale", _format_float(run_contract["budget_scale"], 8), "1.0"],
                ["effective 15 cm L2 radius", _format_float(run_contract["effective_l2_radius_15cm"], 8), "0.00123243"],
                ["inner steps", str(run_contract["inner_steps"]), "not used"],
                ["step fraction", _format_float(run_contract["step_fraction"], 4), "not used"],
                ["gradient clip", _format_float(run_contract["gradient_clip_norm"], 4), "none"],
                ["LR schedule", run_contract["lr_schedule"], "delayed_cosine"],
                ["warmup / alpha", run_contract["warmup_alpha"], "0 / 1.0"],
                ["initial H0 encoder", "disabled", "disabled"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Training diagnostics",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["metric", "value"],
            [
                ["completed batches", f"{training['completed_batches']} / {training['expected_batches']}"],
                ["diagnostics ok", str(training["ok"])],
                ["final train loss mean", _format_float(training["train_loss_total"]["mean"], 6)],
                ["final validation loss mean", _format_float(training["validation_loss_total"]["mean"], 6)],
                ["PGD inner objective best mean", _format_float(training["pgd_inner_objective_best"]["mean"], 6)],
                ["PGD inner improvement mean", _format_float(training["pgd_inner_objective_improvement"]["mean"], 6)],
                ["PGD final endpoint gap mean", _format_float(training["pgd_inner_objective_final_endpoint_gap"]["mean"], 6)],
                ["PGD boundary fraction mean", _format_float(training["pgd_boundary_fraction"]["mean"], 6)],
                ["PGD radius-ratio mean", _format_float(training["pgd_radius_ratio_mean"]["mean"], 6)],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Delayed kinematics and hold checks",
            "",
        ]
    )
    profile_rows = []
    for bank in ("no_catch", "catch"):
        pgd_profile = velocity[bank]["pgd"]
        baseline_profile = velocity[bank]["baseline"]
        profile_rows.append(
            [
                bank,
                _format_float(pgd_profile["peak_mean_forward_velocity_m_s"], 5),
                _format_float(baseline_profile["peak_mean_forward_velocity_m_s"], 5),
                _format_percent_change(
                    pgd_profile["peak_mean_forward_velocity_m_s"],
                    baseline_profile["peak_mean_forward_velocity_m_s"],
                ),
                _format_float(pgd_profile["time_of_peak_mean_forward_velocity_s"], 4),
                _format_float(baseline_profile["time_of_peak_mean_forward_velocity_s"], 4),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "bank",
                "PGD peak velocity m/s",
                "baseline peak velocity m/s",
                "PGD change",
                "PGD peak time s",
                "baseline peak time s",
            ],
            profile_rows,
        )
    )
    lines.extend(
        [
            "",
            "Behavior sidecars on the PGD row: endpoint error mean "
            f"{_format_float(_metric_mean(behavior['endpoint_error_m']), 5)} m, "
            f"overshoot mean {_format_float(_metric_mean(behavior['overshoot_m']), 5)} m, "
            f"first-five-step command norm mean "
            f"{_format_float(_metric_mean(behavior['first_five_step_command_norm']), 5)}.",
            "",
            "## Standard certificate",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "row",
                "status",
                "state-weighted action mismatch",
                "classification",
                "transition/value/Bellman",
            ],
            [
                [
                    "PGD",
                    standard["pgd"]["status"],
                    _format_float(standard["pgd"]["state_weighted_action_mismatch"], 6),
                    standard["pgd"]["classification"],
                    "not_applicable / not_applicable / not_applicable",
                ],
                [
                    "baseline",
                    standard["baseline"]["status"],
                    _format_float(standard["baseline"]["state_weighted_action_mismatch"], 6),
                    standard["baseline"]["classification"],
                    "not_applicable / not_applicable / not_applicable",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "The shared blocker remains the 6D GRU feedback contract versus the",
            "current 8D analytical output-feedback response-map contract.",
            "",
            "## Perturbation response comparison",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "class",
                "PGD cost",
                "baseline cost",
                "PGD/base cost",
                "PGD max dx",
                "base max dx",
                "PGD/extLQG",
                "base/extLQG",
            ],
            [
                [
                    row["class"],
                    _format_float(row["pgd_full_qrf_delta_cost"], 5),
                    _format_float(row["baseline_full_qrf_delta_cost"], 5),
                    _format_float(row["full_qrf_cost_pgd_over_baseline"], 4),
                    _format_float(row["pgd_max_delta_position_m"], 5),
                    _format_float(row["baseline_max_delta_position_m"], 5),
                    _format_float(row["pgd_extlqg_ratio"], 5),
                    _format_float(row["baseline_extlqg_ratio"], 5),
                ]
                for row in perturbation_rows
            ],
        )
    )
    csv_path = manifest["outputs"]["perturbation_class_comparison_csv"]
    lines.extend(
        [
            "",
            f"Full class comparison CSV: `{csv_path}`.",
            "",
            "## Interpretation",
            "",
            "- Movement-only PGD made the fixed-bank no-catch reach faster than the",
            "  non-PGD delayed baseline, while catch/hold trials stayed flat at zero",
            "  forward velocity on the fixed bank.",
            "- PGD reduced full-Q/R/Q_f perturbation delta cost across every",
            "  comparable perturbation class in this bank, while action-response",
            "  norms increased on several command, observation, and sensory classes.",
            "- The standard empirical/nonlinear certificate remains a partial blocked",
            "  certificate, and the PGD row has a modestly larger clean-rollout action",
            "  mismatch than the non-PGD delayed baseline.",
            "",
            "## Artifact size cleanup",
            "",
        ]
    )
    before = size_summary.get("before_cleanup") or {}
    current = size_summary["current"]
    lines.extend(
        _markdown_table(
            ["scope", "before", "current"],
            [
                [
                    "all 4d79e07 artifacts",
                    before.get("artifacts_4d79e07_human", "not recorded"),
                    current["artifacts_4d79e07_human"],
                ],
                [
                    "PGD perturbation bulk",
                    before.get("perturbation_bulk_human", "not recorded"),
                    current["perturbation_bulk_human"],
                ],
                [
                    "raw perturbation NPZ caches",
                    f"{before.get('perturbation_raw_npz_count', 'not recorded')} files / "
                    f"{before.get('perturbation_raw_npz_human', 'not recorded')}",
                    f"{current['perturbation_raw_npz_count']} files / "
                    f"{current['perturbation_raw_npz_human']}",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Residual blockers",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in residuals)
    lines.append("")
    return "\n".join(lines)


def materialize(repo_root: Path) -> dict[str, Any]:
    paths = _paths(repo_root)
    pgd_spec = _read_json(paths.pgd_run_spec)
    baseline_spec = _read_json(paths.baseline_run_spec)
    pgd_standard = _read_json(paths.pgd_standard_manifest)
    baseline_standard = _read_json(paths.baseline_standard_manifest)
    pgd_evaluation = _read_json(paths.pgd_evaluation_manifest)
    pgd_perturbation = _read_json(paths.pgd_perturbation_detail)
    baseline_perturbation = _read_json(paths.baseline_perturbation_detail)
    pgd_figure = _read_json(paths.pgd_figure_summary)

    comparison_rows = _comparison_rows(pgd_perturbation, baseline_perturbation)
    _write_csv(comparison_rows, paths.comparison_csv)

    existing = _read_json(paths.comparison_manifest) if paths.comparison_manifest.exists() else None
    hps = pgd_spec["hps"]
    pgd_training = hps["broad_epsilon_pgd_training"]
    optimizer = pgd_spec["optimizer"]
    expected_batches = int(pgd_spec["n_train_batches"])
    artifact_size_cleanup = _artifact_size_summary(paths, existing)
    manifest: dict[str, Any] = {
        "schema_version": "rlrmp.delayed_pgd_materialization_summary.v1",
        "issue": EXPERIMENT,
        "baseline_issue": BASELINE_EXPERIMENT,
        "run_id": PGD_RUN_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "run_contract": {
            "pgd_enabled": bool(pgd_training["enabled"]),
            "pgd_support": "movement_epoch_only",
            "budget_scale": pgd_training["budget_scale"],
            "effective_l2_radius_15cm": pgd_training["budget_contract"][
                "effective_l2_radius_15cm"
            ],
            "inner_steps": pgd_training["inner_maximizer"]["n_steps"],
            "step_fraction": pgd_training["inner_maximizer"][
                "step_size_fraction_of_l2_radius"
            ],
            "gradient_clip_norm": optimizer["gradient_clip_norm"],
            "lr_schedule": optimizer["schedule"],
            "warmup_alpha": (
                f"{optimizer['constant_lr_iterations']} / "
                f"{optimizer['cosine_annealing_alpha']}"
            ),
            "initial_hidden_encoder_enabled": bool(
                pgd_spec["training_summary"]["initial_hidden_encoder"]["enabled"]
            ),
            "baseline_pgd_enabled": bool(
                baseline_spec["hps"]["broad_epsilon_pgd_training"]["enabled"]
            ),
        },
        "training_diagnostics": _training_diagnostics(
            paths.training_diagnostics_npz,
            expected_batches,
        ),
        "behavior_summary": _behavior_summary(pgd_evaluation),
        "velocity_profiles": {
            "no_catch": {
                "pgd": _profile_summary(paths.pgd_velocity_no_catch),
                "baseline": _profile_summary(paths.baseline_velocity_no_catch),
            },
            "catch": {
                "pgd": _profile_summary(paths.pgd_velocity_catch),
                "baseline": _profile_summary(paths.baseline_velocity_catch),
            },
            "postrun_repeated_validation": _figure_velocity_summary(pgd_figure),
        },
        "standard_certificate": {
            "pgd": _standard_summary(pgd_standard),
            "baseline": _standard_summary(baseline_standard),
        },
        "perturbation_comparison": {
            "pgd_status_counts": _status_counts(pgd_perturbation, PGD_RUN_ID),
            "baseline_status_counts": _status_counts(
                baseline_perturbation,
                BASELINE_RUN_ID,
            ),
            "rows": comparison_rows,
        },
        "artifact_size_cleanup": artifact_size_cleanup,
        "outputs": {
            "note": _repo_rel(paths.comparison_note, repo_root),
            "manifest": _repo_rel(paths.comparison_manifest, repo_root),
            "perturbation_class_comparison_csv": _repo_rel(
                paths.comparison_csv,
                repo_root,
            ),
        },
        "source_manifests": {
            "pgd_run_spec": _repo_rel(paths.pgd_run_spec, repo_root),
            "baseline_run_spec": _repo_rel(paths.baseline_run_spec, repo_root),
            "pgd_standard_manifest": _repo_rel(paths.pgd_standard_manifest, repo_root),
            "baseline_standard_manifest": _repo_rel(
                paths.baseline_standard_manifest,
                repo_root,
            ),
            "pgd_evaluation_manifest": _repo_rel(paths.pgd_evaluation_manifest, repo_root),
            "pgd_postrun_manifest": _repo_rel(paths.pgd_postrun_manifest, repo_root),
            "pgd_perturbation_detail": _repo_rel(paths.pgd_perturbation_detail, repo_root),
            "baseline_perturbation_detail": _repo_rel(
                paths.baseline_perturbation_detail,
                repo_root,
            ),
        },
        "residual_blockers": [
            "Objective comparator was skipped for this final-checkpoint lens; the existing comparator expects validation-selected checkpoints.",
            "Map decomposition was skipped by the upstream materializer because the delayed-bank rollout arrays broadcast as (1, 60, 2, 240) versus requested (80, 90, 2, 240).",
            "Feedback ablation was skipped because calibrated force/filter feedback rows require a controller-feedback scale manifest wiring path.",
            "Standard response-map components remain blocked by the 6D GRU feedback versus 8D analytical output-feedback contract.",
        ],
    }

    # Convert Path objects embedded in diagnostics to repo-relative strings.
    manifest["training_diagnostics"]["path"] = _repo_rel(
        manifest["training_diagnostics"]["path"],
        repo_root,
    )

    mkdir_p(paths.comparison_manifest.parent)
    paths.comparison_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_marked_section(
        paths.comparison_note,
        "materialization_summary",
        _render_markdown(manifest),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Defaults to rlrmp.paths.REPO_ROOT.",
    )
    args = parser.parse_args()
    manifest = materialize(args.repo_root.resolve())
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
