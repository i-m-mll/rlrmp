"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

SISU-conditioned perturbation-class comparison diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    CheckpointSelectionMode,
    default_cs_perturbation_bank,
    evaluate_run_perturbation_bank,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import resolve_run_inputs
from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import set_sisu_condition
from rlrmp.paths import REPO_ROOT, mkdir_p


SCHEMA_VERSION = "rlrmp.sisu_perturbation_class_comparison.v1"
_ANALYSIS_PRESET = load_analysis_parameter_preset("sisu_perturbation_comparison").parameters
DEFAULT_SISU_LEVELS = tuple(_ANALYSIS_PRESET["sisu_levels"])
DEFAULT_N_ROLLOUT_TRIALS = int(_ANALYSIS_PRESET["n_rollout_trials"])
DEFAULT_OUTPUT_STEM = "sisu_perturbation_class_comparison_targetfix"

METRIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "delta_action_norm",
        "slug": "mean_delta_action",
        "label": "Mean delta action",
        "better": "lower",
        "role": "response_magnitude",
    },
    {
        "key": "delta_position_response_m.max",
        "slug": "max_delta_x_m",
        "label": "Max delta x (m)",
        "better": "lower",
        "role": "response_magnitude",
    },
    {
        "key": "delta_position_response_m.auc",
        "slug": "auc_delta_x_m_s",
        "label": "AUC delta x (m*s)",
        "better": "lower",
        "role": "response_magnitude",
    },
    {
        "key": "delta_endpoint_error_m",
        "slug": "mean_endpoint_delta_m",
        "label": "Mean endpoint delta (m)",
        "better": "diagnostic_signed_closer_to_zero",
        "role": "signed_endpoint_delta",
    },
    {
        "key": "delta_terminal_speed_m_s",
        "slug": "mean_terminal_speed_delta_m_s",
        "label": "Mean terminal-speed delta (m/s)",
        "better": "diagnostic_signed_closer_to_zero",
        "role": "signed_terminal_speed_delta",
    },
    {
        "key": "extra_full_qrf_delta_cost_total",
        "slug": "mean_full_qrf_delta_cost",
        "label": "Mean full-Q/R/Qf delta cost",
        "better": "lower",
        "role": "cost_delta",
    },
)


def materialize_sisu_perturbation_comparison(
    *,
    source_experiment: str,
    result_experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    sisu_levels: Sequence[float] = DEFAULT_SISU_LEVELS,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    bank_mode: str = "calibrated",
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_scale_manifest_path: Path | None = None,
    preferred_checkpoint_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate and summarize SISU 1-vs-0 perturbation-class responses.

    The evaluator reruns only local perturbation-bank rollouts and passes
    ``write_bulk_arrays=False`` so raw rollout arrays are not materialized.
    """

    if tuple(float(level) for level in sisu_levels) != DEFAULT_SISU_LEVELS:
        raise ValueError("this comparison currently expects SISU levels 0.0 and 1.0")

    repo_root = repo_root.resolve()
    bank = default_cs_perturbation_bank(
        mode=bank_mode,  # type: ignore[arg-type]
        calibration_level=calibration_level,
        calibration_reach=calibration_reach,
        feedback_scale_manifest_path=feedback_scale_manifest_path,
    )
    runs = resolve_run_inputs(
        experiment=source_experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    scratch_bulk_dir = (
        repo_root / "_artifacts" / result_experiment / output_stem / "no_raw_rollout_arrays"
    )
    mkdir_p(scratch_bulk_dir)
    evaluated: dict[str, dict[str, Any]] = {}
    for run in runs:
        by_sisu: dict[str, Any] = {}
        for sisu in sisu_levels:
            by_sisu[_sisu_key(float(sisu))] = evaluate_run_perturbation_bank(
                run,
                source_experiment=source_experiment,
                bank=bank,
                n_rollout_trials=n_rollout_trials,
                write_bulk_arrays=False,
                bulk_dir=scratch_bulk_dir / _sisu_key(float(sisu)),
                trial_spec_transform=lambda trials, value=float(sisu): set_sisu_condition(
                    trials,
                    value,
                ),
                preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
                checkpoint_selection_mode=checkpoint_selection_mode,
                repo_root=repo_root,
            )
        evaluated[run.run_id] = by_sisu

    manifest = build_comparison_manifest(
        source_experiment=source_experiment,
        result_experiment=result_experiment,
        bank=bank,
        run_summaries=evaluated,
        sisu_levels=sisu_levels,
        n_rollout_trials=n_rollout_trials,
        output_stem=output_stem,
        repo_root=repo_root,
        bank_mode=bank_mode,
        calibration_level=calibration_level,
        calibration_reach=calibration_reach,
        feedback_scale_manifest_path=feedback_scale_manifest_path,
        preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=checkpoint_selection_mode,
    )
    notes_dir = repo_root / "results" / result_experiment / "notes"
    mkdir_p(notes_dir)
    json_path = notes_dir / f"{output_stem}.json"
    md_path = notes_dir / f"{output_stem}.md"
    regeneration_spec_path = notes_dir / f"{output_stem}_regeneration_spec.json"
    manifest["outputs"] = {
        "json": repo_relative(json_path, repo_root=repo_root),
        "markdown": repo_relative(md_path, repo_root=repo_root),
        "regeneration_spec": repo_relative(regeneration_spec_path, repo_root=repo_root),
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(manifest))
    write_regeneration_spec(
        spec_path=regeneration_spec_path,
        diagnostic_name="sisu_perturbation_class_comparison",
        materializer=(
            "rlrmp.analysis.pipelines.sisu_perturbation_comparison."
            "materialize_sisu_perturbation_comparison"
        ),
        command=None,
        parameters={
            "source_experiment": source_experiment,
            "result_experiment": result_experiment,
            "run_ids": list(run_ids),
            "labels": None if labels is None else list(labels),
            "n_rollout_trials": n_rollout_trials,
            "sisu_levels": list(sisu_levels),
            "output_stem": output_stem,
            "bank_mode": bank_mode,
            "calibration_level": calibration_level,
            "calibration_reach": calibration_reach,
            "feedback_scale_manifest_path": (
                None
                if feedback_scale_manifest_path is None
                else repo_relative(feedback_scale_manifest_path, repo_root=repo_root)
            ),
            "preferred_checkpoint_manifest_path": (
                None
                if preferred_checkpoint_manifest_path is None
                else repo_relative(preferred_checkpoint_manifest_path, repo_root=repo_root)
            ),
            "checkpoint_selection_mode": checkpoint_selection_mode,
            "write_bulk_arrays": False,
        },
        inputs=[{"role": "run_spec", "path": run.run_spec_path} for run in runs]
        + [{"role": "run_artifact_dir", "path": run.artifact_dir} for run in runs]
        + (
            []
            if feedback_scale_manifest_path is None
            else [
                {"role": "controller_feedback_scale_manifest", "path": feedback_scale_manifest_path}
            ]
        )
        + (
            []
            if preferred_checkpoint_manifest_path is None
            else [{"role": "checkpoint_manifest", "path": preferred_checkpoint_manifest_path}]
        ),
        outputs=[
            {"role": "comparison_json", "path": json_path},
            {"role": "comparison_markdown", "path": md_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/sisu_perturbation_comparison.py",
            "src/rlrmp/analysis/pipelines/gru_perturbation_bank.py",
            "src/rlrmp/analysis/pipelines/sisu_spectrum_diagnostics.py",
        ],
        notes=[
            "Local rerollouts only; remote training is not part of this materializer.",
            "Raw perturbation-response rollout arrays are intentionally not written.",
        ],
        repo_root=repo_root,
    )
    return manifest


def build_comparison_manifest(
    *,
    source_experiment: str,
    result_experiment: str,
    bank: Mapping[str, Any],
    run_summaries: Mapping[str, Mapping[str, Any]],
    sisu_levels: Sequence[float],
    n_rollout_trials: int,
    output_stem: str,
    repo_root: Path,
    bank_mode: str,
    calibration_level: str | Sequence[str] | None,
    calibration_reach: str | float | None,
    feedback_scale_manifest_path: Path | None,
    preferred_checkpoint_manifest_path: Path | None,
    checkpoint_selection_mode: str,
) -> dict[str, Any]:
    """Build a compact JSON manifest from per-SISU perturbation summaries."""

    runs: dict[str, Any] = {}
    for run_id, levels in run_summaries.items():
        low = levels[_sisu_key(0.0)]
        high = levels[_sisu_key(1.0)]
        class_comparisons = compare_summary_groups(
            low["robust_response_summary"]["class_summary"]["groups"],
            high["robust_response_summary"]["class_summary"]["groups"],
        )
        timing_comparisons = compare_summary_groups(
            low["robust_response_summary"]["timing_cell_summary"]["groups"],
            high["robust_response_summary"]["timing_cell_summary"]["groups"],
        )
        runs[run_id] = {
            "label": high.get("label", run_id),
            "status_counts_by_sisu": {
                key: value.get("status_counts", {}) for key, value in levels.items()
            },
            "checkpoint_selection_by_sisu": {
                key: value.get("checkpoint_selection", []) for key, value in levels.items()
            },
            "class_comparison": class_comparisons,
            "timing_cell_comparison": timing_comparisons,
            "headline": summarize_headline(class_comparisons),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "source_experiment": source_experiment,
        "output_stem": output_stem,
        "sisu_levels": [float(level) for level in sisu_levels],
        "n_rollout_trials_per_replicate": int(n_rollout_trials),
        "checkpoint_policy": "validation_selected_per_replicate",
        "bank": {
            "schema_version": bank.get("schema_version"),
            "bank_id": bank.get("bank_id"),
            "mode": bank_mode,
            "calibration_level": calibration_level,
            "calibration_reach": calibration_reach,
            "n_perturbation_rows": len(bank.get("perturbations", [])),
        },
        "input_contract": (
            "SISU is set by changing the scalar trial input to 0.0 or 1.0 before "
            "the standard perturbation bank is applied. All other validation-bank "
            "inputs and checkpoint selections are held fixed."
        ),
        "materialization_policy": {
            "rerollouts_needed": True,
            "paired_rerollouts_performed": True,
            "reason": (
                "Existing targetfix perturbation summaries were materialized only at "
                "the default validation SISU input of 1.0; SISU=0 perturbation "
                "responses were not present in the compact summaries. The final "
                "comparison reruns both SISU=0 and SISU=1 locally through the same "
                "evaluator for paired apples-to-apples summaries."
            ),
            "remote_training": "not_run",
            "raw_rollout_arrays_written": False,
        },
        "metric_policy": {
            "lower_is_better": [spec["slug"] for spec in METRIC_SPECS if spec["better"] == "lower"],
            "diagnostic_signed": [
                spec["slug"]
                for spec in METRIC_SPECS
                if spec["better"].startswith("diagnostic_signed")
            ],
            "ratio_meaning": (
                "SISU1/SISU0 ratio < 1 means the SISU=1 conditioned controller had "
                "a smaller perturbation response on that metric; > 1 means larger."
            ),
            "difference_meaning": (
                "Differences are SISU1 minus SISU0. For signed endpoint and "
                "terminal-speed deltas, sign is diagnostic; closer to zero is "
                "usually preferable."
            ),
        },
        "provenance": {
            "feedback_scale_manifest_path": (
                None
                if feedback_scale_manifest_path is None
                else repo_relative(feedback_scale_manifest_path, repo_root=repo_root)
            ),
            "preferred_checkpoint_manifest_path": (
                None
                if preferred_checkpoint_manifest_path is None
                else repo_relative(preferred_checkpoint_manifest_path, repo_root=repo_root)
            ),
            "checkpoint_selection_mode": checkpoint_selection_mode,
        },
        "runs": runs,
    }


def compare_summary_groups(
    low_groups: Mapping[str, Mapping[str, Any]],
    high_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare SISU=1 and SISU=0 grouped perturbation summaries."""

    comparisons = {}
    for key in sorted(set(low_groups) | set(high_groups)):
        low = low_groups.get(key, {})
        high = high_groups.get(key, {})
        comparisons[key] = compare_group(key, low, high)
    return comparisons


def compare_group(
    group_key: str,
    low: Mapping[str, Any],
    high: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one class or timing-cell group."""

    low_metrics = low.get("metrics", {})
    high_metrics = high.get("metrics", {})
    metrics = {}
    for spec in METRIC_SPECS:
        low_value = metric_mean(low_metrics, spec["key"])
        high_value = metric_mean(high_metrics, spec["key"])
        metrics[spec["slug"]] = {
            "label": spec["label"],
            "metric_key": spec["key"],
            "role": spec["role"],
            "better": spec["better"],
            "sisu_0": low_value,
            "sisu_1": high_value,
            "delta_1_minus_0": _delta(high_value, low_value),
            "ratio_1_over_0": _ratio(high_value, low_value),
        }
    return {
        "group": group_key,
        "rows_sisu_0": low.get("n_rows"),
        "rows_sisu_1": high.get("n_rows"),
        "status_counts_sisu_0": low.get("status_counts", {}),
        "status_counts_sisu_1": high.get("status_counts", {}),
        "amplitudes_sisu_0": low.get("amplitudes", []),
        "amplitudes_sisu_1": high.get("amplitudes", []),
        "metrics": metrics,
        "notes": _group_notes(low, high),
    }


def metric_mean(metrics: Mapping[str, Any], dotted_key: str) -> float | None:
    """Read a class-summary metric mean from flat or nested metric summaries."""

    direct = metrics.get(dotted_key)
    if isinstance(direct, Mapping) and direct.get("mean") is not None:
        return float(direct["mean"])
    current: Any = metrics
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, Mapping) or current.get("mean") is None:
        return None
    return float(current["mean"])


def summarize_headline(class_comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact counts over class-level SISU1/SISU0 comparison ratios."""

    cost = _ratio_counts(class_comparisons, "mean_full_qrf_delta_cost")
    max_dx = _ratio_counts(class_comparisons, "max_delta_x_m")
    action = _ratio_counts(class_comparisons, "mean_delta_action")
    return {
        "full_qrf_delta_cost": cost,
        "max_delta_x_m": max_dx,
        "mean_delta_action": action,
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render the SISU perturbation-class comparison report."""

    lines = [
        "# SISU Perturbation-Class Robustification Comparison",
        "",
        f"Issue: `{manifest['issue']}`. Source experiment: `{manifest['source_experiment']}`.",
        "",
        (
            "This report compares SISU=1 against SISU=0 within each trained "
            "targetfix model on the calibrated 020a65b-style perturbation bank. "
            "It is discovery-trained robustness evidence, not teacher/distillation "
            "behavior, not trial-history adaptation, and not a formal H-infinity "
            "equivalence claim."
        ),
        "",
        "## Interpretation",
        "",
        (
            "- For response magnitudes and full-Q/R/Qf delta cost, lower is better; "
            "a SISU1/SISU0 ratio below 1 is an improvement."
        ),
        (
            "- Endpoint and terminal-speed deltas are signed diagnostics; the "
            "SISU1-SISU0 difference shows direction, and values closer to zero are "
            "usually preferable."
        ),
        (
            "- Existing targetfix perturbation summaries were sufficient for the "
            "SISU=1 side only, but this materialization reran both SISU=0 and "
            "SISU=1 locally through the same evaluator for a paired comparison. "
            "No remote training and no raw rollout arrays were written."
        ),
        "",
        "## Bank",
        "",
        f"- Bank id: `{manifest['bank']['bank_id']}`",
        f"- Bank mode: `{manifest['bank']['mode']}`",
        f"- Perturbation rows: {manifest['bank']['n_perturbation_rows']}",
        f"- Rollout trials per replicate: {manifest['n_rollout_trials_per_replicate']}",
        "",
    ]
    for run_id, run in _ordered_runs(manifest["runs"]):
        lines.extend(_render_run(run_id, run))
    return "\n".join(lines).rstrip() + "\n"


def _ordered_runs(runs: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return rows in the phase-spec order when labels identify rows A/B."""

    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        run_id, run = item
        label = str(run.get("label", run_id)).lower() if isinstance(run, Mapping) else run_id
        if "raw strong" in label or "gamma-1.05" in label:
            return (0, label)
        if "effective" in label and "020a65b" in label:
            return (1, label)
        return (2, label)

    return sorted(runs.items(), key=sort_key)


def _render_run(run_id: str, run: Mapping[str, Any]) -> list[str]:
    lines = [
        f"## {run['label']}",
        "",
        f"Run: `{run_id}`",
        "",
        "### Metric Glossary",
        "",
        "- Ratios are `SISU=1 / SISU=0`; values below 1 mean the high-SISU "
        "condition had the smaller perturbation response.",
        "- `Mean delta action ratio`: mean command-change norm under perturbation.",
        "- `Max delta x ratio`: peak hand-position response magnitude in meters.",
        "- `AUC delta x ratio`: time-integrated hand-position response magnitude.",
        "- `Cost SISU=0`, `Cost SISU=1`, `Cost ratio`, and `Cost diff`: "
        "post-hoc full-Q/R/Q_f perturbation delta cost, with `diff = SISU1 - SISU0`.",
        "- Signed diagnostics are separated because endpoint and terminal-speed "
        "deltas are directional sidecars, not simple lower-is-better ratios.",
        "",
        "### Headline",
        "",
        "| Metric | Class groups with ratio < 1 | ratio = 1 | ratio > 1 | unavailable |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Full-Q/R/Qf delta cost", "full_qrf_delta_cost"),
        ("Max delta x", "max_delta_x_m"),
        ("Mean delta action", "mean_delta_action"),
    ):
        row = run["headline"][key]
        lines.append(
            f"| {label} | {row['improved']} | {row['equal']} | "
            f"{row['worse']} | {row['not_available']} |"
        )
    lines.extend(
        [
            "",
            "### Class-Binned Summary",
            "",
            "| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | "
            "AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | "
            "Cost diff | Notes |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for class_key, row in run["class_comparison"].items():
        lines.append(_render_comparison_row(class_key, row))
    lines.extend(
        [
            "",
            "#### Signed Diagnostics",
            "",
            "| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | "
            "Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | "
            "terminal diff |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for class_key, row in run["class_comparison"].items():
        lines.append(_render_signed_diagnostic_row(class_key, row))
    lines.extend(
        [
            "",
            "### Timing-Cell Summary",
            "",
            "| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | "
            "Full-Q/R/Qf cost ratio | cost diff | Notes |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell_key, row in run["timing_cell_comparison"].items():
        metrics = row["metrics"]
        lines.append(
            "| "
            f"`{cell_key}` | "
            f"{_rows_text(row)} | "
            f"{_fmt_ratio(metrics['mean_delta_action'])} | "
            f"{_fmt_ratio(metrics['max_delta_x_m'])} | "
            f"{_fmt_ratio(metrics['auc_delta_x_m_s'])} | "
            f"{_fmt_ratio(metrics['mean_full_qrf_delta_cost'])} | "
            f"{_fmt_float(metrics['mean_full_qrf_delta_cost']['delta_1_minus_0'])} | "
            f"{_fmt_notes(row.get('notes', []))} |"
        )
    lines.append("")
    return lines


def _render_comparison_row(group_key: str, row: Mapping[str, Any]) -> str:
    metrics = row["metrics"]
    action = metrics["mean_delta_action"]
    max_dx = metrics["max_delta_x_m"]
    auc_dx = metrics["auc_delta_x_m_s"]
    cost = metrics["mean_full_qrf_delta_cost"]
    return (
        "| "
        f"`{group_key}` | "
        f"{_rows_text(row)} | "
        f"{_status_text(row)} | "
        f"{_fmt_ratio(action)} | "
        f"{_fmt_ratio(max_dx)} | "
        f"{_fmt_ratio(auc_dx)} | "
        f"{_fmt_float(cost['sisu_0'])} | "
        f"{_fmt_float(cost['sisu_1'])} | "
        f"{_fmt_ratio(cost)} | "
        f"{_fmt_float(cost['delta_1_minus_0'])} | "
        f"{_fmt_notes(row.get('notes', []))} |"
    )


def _render_signed_diagnostic_row(group_key: str, row: Mapping[str, Any]) -> str:
    metrics = row["metrics"]
    endpoint = metrics["mean_endpoint_delta_m"]
    terminal = metrics["mean_terminal_speed_delta_m_s"]
    return (
        "| "
        f"`{group_key}` | "
        f"{_fmt_float(endpoint['sisu_0'])} | "
        f"{_fmt_float(endpoint['sisu_1'])} | "
        f"{_fmt_float(endpoint['delta_1_minus_0'])} | "
        f"{_fmt_float(terminal['sisu_0'])} | "
        f"{_fmt_float(terminal['sisu_1'])} | "
        f"{_fmt_float(terminal['delta_1_minus_0'])} |"
    )


def _ratio_counts(
    comparisons: Mapping[str, Mapping[str, Any]],
    metric_slug: str,
) -> dict[str, int]:
    counts = {"improved": 0, "equal": 0, "worse": 0, "not_available": 0}
    for row in comparisons.values():
        ratio = row.get("metrics", {}).get(metric_slug, {}).get("ratio_1_over_0")
        if ratio is None or not np.isfinite(float(ratio)):
            counts["not_available"] += 1
        elif abs(float(ratio) - 1.0) <= 1e-12:
            counts["equal"] += 1
        elif float(ratio) < 1.0:
            counts["improved"] += 1
        else:
            counts["worse"] += 1
    return counts


def _delta(high: float | None, low: float | None) -> float | None:
    if high is None or low is None:
        return None
    return float(high - low)


def _ratio(high: float | None, low: float | None) -> float | None:
    if high is None or low is None or abs(low) <= 1e-12:
        return None
    return float(high / low)


def _group_notes(low: Mapping[str, Any], high: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    for prefix, row in (("sisu_0", low), ("sisu_1", high)):
        for warning in row.get("denominator_warnings", []) or []:
            notes.append(f"{prefix}:{warning}")
        for reason_map_name in (
            "not_applicable_reasons",
            "extlqg_not_applicable_reasons",
            "robust_analytical_not_applicable_reasons",
        ):
            reasons = row.get(reason_map_name, {})
            if isinstance(reasons, Mapping) and reasons:
                notes.append(f"{prefix}:{reason_map_name}={len(reasons)}")
    return sorted(set(notes))


def _sisu_key(value: float) -> str:
    return f"sisu_{value:g}"


def _rows_text(row: Mapping[str, Any]) -> str:
    low = row.get("rows_sisu_0")
    high = row.get("rows_sisu_1")
    return str(low) if low == high else f"{low}/{high}"


def _status_text(row: Mapping[str, Any]) -> str:
    low = row.get("status_counts_sisu_0", {})
    high = row.get("status_counts_sisu_1", {})
    low_text = _format_status_counts(low)
    high_text = _format_status_counts(high)
    return low_text if low_text == high_text else f"0:{low_text}; 1:{high_text}"


def _format_status_counts(counts: Any) -> str:
    if not isinstance(counts, Mapping) or not counts:
        return "unknown"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.6g}"


def _fmt_ratio(metric: Mapping[str, Any]) -> str:
    return _fmt_float(metric.get("ratio_1_over_0"))


def _fmt_notes(notes: Any) -> str:
    if not notes:
        return "none"
    return "; ".join(str(note) for note in notes)


__all__ = [
    "DEFAULT_N_ROLLOUT_TRIALS",
    "DEFAULT_OUTPUT_STEM",
    "DEFAULT_SISU_LEVELS",
    "METRIC_SPECS",
    "SCHEMA_VERSION",
    "build_comparison_manifest",
    "compare_group",
    "compare_summary_groups",
    "materialize_sisu_perturbation_comparison",
    "metric_mean",
    "render_markdown",
    "summarize_headline",
]
