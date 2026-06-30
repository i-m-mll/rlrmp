"""Materialize beta 1.4 feedback, robustness, and stabilization summary tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from materialize_beta1p4_stabilization_diagnostics import (
    DEFAULT_ROWS as STABILIZATION_ROWS,
)
from materialize_beta1p4_stabilization_diagnostics import (
    materialize_stabilization_diagnostics,
)
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    materialize_validation_selected_checkpoint_manifest,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
    materialize_gru_evaluation_diagnostics,
)
from rlrmp.analysis.pipelines.gru_feedback_ablation import materialize_gru_feedback_ablation
from rlrmp.analysis.pipelines.gru_perturbation_bank import materialize_gru_perturbation_response
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE = "b413bb0"
BASELINE_EXPERIMENT = "33b0dcb"
BASELINE_RUN = "h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64"
OUTPUT_TAG = "beta1p4_moderate_validation_selected_diagnostics"
SUMMARY_TAG = "beta1p4_feedback_robustness_summary"
MARKER = "beta1p4_feedback_robustness_summary"
SCHEMA_VERSION = "rlrmp.b413bb0.beta1p4_feedback_robustness_summary.v1"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
PERTURBATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "perturbation_response"
FEEDBACK_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "feedback_ablation"
SUMMARY_JSON = NOTES_DIR / f"{SUMMARY_TAG}.json"
SUMMARY_MD = NOTES_DIR / f"{SUMMARY_TAG}.md"


@dataclass(frozen=True)
class RowSource:
    """One source run included in the beta 1.4 comparison."""

    row_key: str
    source_experiment: str
    run_id: str
    label: str
    training_condition: str


ROWS: tuple[RowSource, ...] = (
    RowSource(
        row_key="baseline_no_pgd_h0_const_band16",
        source_experiment=BASELINE_EXPERIMENT,
        run_id=BASELINE_RUN,
        label="no-PGD H0 const_band16",
        training_condition="no-PGD H0 6D open-loop moderate const_band16 baseline",
    ),
    RowSource(
        row_key="direct_epsilon",
        source_experiment=ISSUE,
        run_id="direct_epsilon",
        label="direct epsilon",
        training_condition="beta 1.4 direct-epsilon PGD",
    ),
    RowSource(
        row_key="linear_no_bias",
        source_experiment=ISSUE,
        run_id="linear_no_bias",
        label="linear no bias",
        training_condition="beta 1.4 finite linear no-bias adversary",
    ),
    RowSource(
        row_key="affine",
        source_experiment=ISSUE,
        run_id="affine",
        label="affine",
        training_condition="beta 1.4 finite affine adversary",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate existing sidecars.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    summary = materialize_summary(force=args.force)
    print(
        json.dumps(
            {
                "summary_json": repo_rel(SUMMARY_JSON),
                "summary_markdown": repo_rel(SUMMARY_MD),
                "rows": [row["row"] for row in summary["rows"]],
                "source_outputs": summary["source_outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def materialize_summary(*, force: bool = False) -> dict[str, Any]:
    """Materialize diagnostics for b413 rows plus the 33b baseline."""

    mkdir_p(NOTES_DIR)
    mkdir_p(PERTURBATION_BULK_DIR)
    mkdir_p(FEEDBACK_BULK_DIR)
    manifests = materialize_component_manifests(force=force)
    stabilization = materialize_stabilization_diagnostics(
        rows=STABILIZATION_ROWS,
        force=force,
    )
    rows = [
        table_row(
            row,
            manifests=manifests,
            stabilization=stabilization,
        )
        for row in ROWS
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "scope": (
            "beta 1.4 b413 direct_epsilon, linear_no_bias, and affine rows "
            "compared with the 33b0dcb H0 no-PGD const_band16 baseline"
        ),
        "row_order": [row.row_key for row in ROWS],
        "baseline_row": "baseline_no_pgd_h0_const_band16",
        "rows": rows,
        "comparisons_vs_baseline": comparisons_vs_baseline(rows),
        "aggregation_contract": aggregation_contract(),
        "source_outputs": source_outputs(manifests, stabilization),
    }
    write_compact_json(SUMMARY_JSON, summary)
    update_marked_section(SUMMARY_MD, MARKER, render_markdown(summary))
    return summary


def materialize_component_manifests(*, force: bool) -> dict[str, dict[str, Any]]:
    """Run or load the evaluation, perturbation, and feedback sidecars."""

    outputs: dict[str, dict[str, Any]] = {}
    for source_experiment, rows in rows_by_source().items():
        source_key = "b413" if source_experiment == ISSUE else "baseline_33b0dcb"
        run_ids = tuple(row.run_id for row in rows)
        labels = tuple(row.label for row in rows)
        paths = component_paths(source_key)
        checkpoint = load_or_materialize(
            paths["checkpoint"],
            force=force,
            materializer=lambda: materialize_validation_selected_checkpoint_manifest(
                experiment=source_experiment,
                run_ids=run_ids,
                output_path=paths["checkpoint"],
                repo_root=REPO_ROOT,
            ),
        )
        evaluation = load_or_materialize(
            paths["evaluation"],
            force=force,
            materializer=lambda: materialize_gru_evaluation_diagnostics(
                experiment=source_experiment,
                run_ids=run_ids,
                labels=labels,
                output_path=paths["evaluation"],
                bulk_dir=paths["evaluation_bulk_dir"],
                n_rollout_trials=64,
                write_bulk_arrays=False,
                regeneration_spec_path=paths["evaluation_regeneration_spec"],
                repo_root=REPO_ROOT,
            ),
        )
        perturbation = load_or_materialize(
            paths["perturbation"],
            force=force,
            materializer=lambda: materialize_gru_perturbation_response(
                source_experiment=source_experiment,
                result_experiment=ISSUE,
                run_ids=run_ids,
                labels=labels,
                n_rollout_trials=64,
                output_path=paths["perturbation"],
                note_path=paths["perturbation_note"],
                bulk_dir=paths["perturbation_bulk_dir"],
                regeneration_spec_path=paths["perturbation_regeneration_spec"],
                bank_mode="calibrated",
                calibration_level="moderate",
                calibration_reach=0.15,
                feedback_scale_manifest_path=paths["evaluation"],
                extlqg_physical_dim=6,
                write_bulk_arrays=False,
                repo_root=REPO_ROOT,
            ),
        )
        feedback = load_or_materialize(
            paths["feedback"],
            force=force,
            materializer=lambda: materialize_gru_feedback_ablation(
                source_experiment=source_experiment,
                result_experiment=ISSUE,
                scope="beta1p4_moderate_feedback_ablation",
                run_ids=run_ids,
                labels=labels,
                n_rollout_trials=64,
                bank_mode="calibrated",
                calibration_level="moderate",
                calibration_reach=0.15,
                feedback_selection_level="moderate",
                feedback_scale_manifest_path=paths["evaluation"],
                output_path=paths["feedback"],
                note_path=paths["feedback_note"],
                bulk_dir=paths["feedback_bulk_dir"],
                regeneration_spec_path=paths["feedback_regeneration_spec"],
                repo_root=REPO_ROOT,
            ),
        )
        outputs[source_key] = {
            "source_experiment": source_experiment,
            "checkpoint": checkpoint,
            "evaluation": evaluation,
            "perturbation": perturbation,
            "feedback": feedback,
            "paths": {key: repo_rel(path) for key, path in paths.items()},
            "perturbation_detail": load_repo_json(
                perturbation["bulk_detail_manifest"]["path"]
            ),
        }
    return outputs


def rows_by_source() -> dict[str, list[RowSource]]:
    """Group row sources by source experiment."""

    grouped: dict[str, list[RowSource]] = defaultdict(list)
    for row in ROWS:
        grouped[row.source_experiment].append(row)
    return dict(grouped)


def component_paths(source_key: str) -> dict[str, Path]:
    """Return output paths for one source-experiment diagnostic group."""

    prefix = f"{OUTPUT_TAG}_{source_key}"
    return {
        "checkpoint": NOTES_DIR / f"{prefix}_checkpoints.json",
        "evaluation": NOTES_DIR / f"{prefix}_gru_evaluation_diagnostics.json",
        "evaluation_regeneration_spec": NOTES_DIR
        / f"{prefix}_gru_evaluation_diagnostics_regeneration_spec.json",
        "evaluation_bulk_dir": PERTURBATION_BULK_DIR / OUTPUT_TAG / source_key / "evaluation",
        "perturbation": NOTES_DIR / f"{prefix}_gru_perturbation_response_manifest.json",
        "perturbation_note": NOTES_DIR / f"{prefix}_gru_perturbation_response.md",
        "perturbation_regeneration_spec": NOTES_DIR
        / f"{prefix}_gru_perturbation_response_manifest_regeneration_spec.json",
        "perturbation_bulk_dir": PERTURBATION_BULK_DIR / OUTPUT_TAG / source_key,
        "feedback": NOTES_DIR / f"{prefix}_gru_feedback_ablation_manifest.json",
        "feedback_note": NOTES_DIR / f"{prefix}_gru_feedback_ablation.md",
        "feedback_regeneration_spec": NOTES_DIR
        / f"{prefix}_gru_feedback_ablation_manifest_regeneration_spec.json",
        "feedback_bulk_dir": FEEDBACK_BULK_DIR / OUTPUT_TAG / source_key,
    }


def load_or_materialize(
    path: Path,
    *,
    force: bool,
    materializer: Any,
) -> dict[str, Any]:
    """Load an existing JSON output unless regeneration was requested."""

    if path.exists() and not force:
        return load_json(path)
    return materializer()


def table_row(
    row: RowSource,
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    stabilization: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the compact summary row."""

    source_key = "b413" if row.source_experiment == ISSUE else "baseline_33b0dcb"
    group = manifests[source_key]
    evaluation = group["evaluation"]["runs"][row.run_id]
    perturbation_detail = group["perturbation_detail"]["runs"][row.run_id]
    feedback = group["feedback"]["runs"][row.run_id]
    stabilization_row = {
        str(item["row"]): item for item in stabilization["rows"]
    }[row.row_key]
    response_groups = perturbation_detail["robust_response_summary"]["class_summary"]["groups"]
    feedback_components = feedback["feedback_pass_audit"]["components"]
    return {
        "row": row.row_key,
        "source_experiment": row.source_experiment,
        "source_run_id": row.run_id,
        "training_condition": row.training_condition,
        "peak_velocity_m_s": float(
            evaluation["behavior"]["velocity_profile"][
                "mean_profile_peak_forward_velocity_m_s"
            ]
        ),
        "sensory_deviation_max_mm": group_metric(
            response_groups,
            metric="delta_position_response_m.max",
            sensory=True,
            scale=1000.0,
        ),
        "sensory_deviation_auc_mm_s": group_metric(
            response_groups,
            metric="delta_position_response_m.auc",
            sensory=True,
            scale=1000.0,
        ),
        "non_sensory_deviation_max_mm": group_metric(
            response_groups,
            metric="delta_position_response_m.max",
            sensory=False,
            scale=1000.0,
        ),
        "non_sensory_deviation_auc_mm_s": group_metric(
            response_groups,
            metric="delta_position_response_m.auc",
            sensory=False,
            scale=1000.0,
        ),
        "peak_dx_over_open_loop": perturbation_ratio_metric(
            perturbation_detail,
            "closed_loop_peak_dx_over_open_loop_peak_dx",
        ),
        "auc_dx_over_open_loop": perturbation_ratio_metric(
            perturbation_detail,
            "closed_loop_auc_dx_over_open_loop_auc_dx",
        ),
        "endpoint_delta_over_reach_length": group_metric(
            response_groups,
            metric="attenuation_metrics.endpoint_delta_over_reach_length",
            sensory=False,
            scale=1.0,
        ),
        "feedback_delta_action_norm": float(
            feedback["interpretation"]["max_feedback_delta_action_norm_mean"]
        ),
        "feedback_ablation_dependence_index": float(
            feedback_components["feedback_ablation_dependence"]["ablation_dependence_index"]
        ),
        "feedback_audit_status": feedback["feedback_pass_audit"]["status"],
        "feedback_interpretation": feedback["interpretation"]["label"],
        "stabilization_feedback_auc_mm_s": maybe_float(stabilization_row["feedback_auc_mm_s"]),
        "stabilization_mechanical_auc_mm_s": maybe_float(
            stabilization_row["mechanical_auc_mm_s"]
        ),
        "stabilization_command_auc_mm_s": maybe_float(
            stabilization_row["command_input_auc_mm_s"]
        ),
        "stabilization_process_force_auc_mm_s": maybe_float(
            stabilization_row["process_force_auc_mm_s"]
        ),
        "stabilization_feedback_peak_mm": maybe_float(stabilization_row["feedback_peak_mm"]),
        "stabilization_mechanical_peak_mm": maybe_float(
            stabilization_row["mechanical_peak_mm"]
        ),
        "checkpoint_policy": evaluation["checkpoint_policy"],
        "source_paths": {
            "run_spec": evaluation["run_spec_path"],
            "artifact_dir": evaluation["artifact_dir"],
        },
    }


def group_metric(
    groups: Mapping[str, Any],
    *,
    metric: str,
    sensory: bool,
    scale: float,
) -> float | None:
    """Return sensory value or non-sensory unweighted group mean."""

    values = []
    for group_key, group in groups.items():
        channel = str(group_key).split("/", maxsplit=1)[0]
        is_sensory = channel == "sensory_feedback"
        if sensory != is_sensory:
            continue
        if channel == "target_stream":
            continue
        payload = group.get("metrics", {}).get(metric, {})
        if payload.get("status") == "available" and payload.get("mean") is not None:
            values.append(float(payload["mean"]) * float(scale))
    if not values:
        return None
    return float(sum(values) / len(values))


def maybe_float(value: Any) -> float | None:
    """Return a float for numeric values, otherwise None."""

    if isinstance(value, int | float):
        return float(value)
    return None


def perturbation_ratio_metric(perturbation_run: Mapping[str, Any], metric_name: str) -> float | None:
    """Mean an available non-sensory attenuation ratio across perturbation rows."""

    values = []
    for item in perturbation_run["perturbations"]:
        if item.get("status") != "evaluated":
            continue
        channel = str(item.get("channel") or item.get("perturbation", {}).get("channel"))
        if channel in {"sensory_feedback", "target_stream"}:
            continue
        metric = item.get("metrics", {}).get("attenuation_metrics", {}).get(metric_name, {})
        if metric.get("status") == "available" and metric.get("mean") is not None:
            values.append(float(metric["mean"]))
    if not values:
        return None
    return float(sum(values) / len(values))


def comparisons_vs_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deltas against the no-PGD H0 const_band16 baseline."""

    by_row = {str(row["row"]): row for row in rows}
    baseline = by_row["baseline_no_pgd_h0_const_band16"]
    numeric_fields = [
        key
        for key, value in baseline.items()
        if isinstance(value, int | float) or value is None
    ]
    comparisons: dict[str, Any] = {}
    for row in rows:
        row_key = str(row["row"])
        if row_key == "baseline_no_pgd_h0_const_band16":
            continue
        deltas = {}
        for field in numeric_fields:
            value = row.get(field)
            reference = baseline.get(field)
            if isinstance(value, int | float) and isinstance(reference, int | float):
                deltas[field] = float(value) - float(reference)
        comparisons[row_key] = deltas
    return comparisons


def source_outputs(
    manifests: Mapping[str, Mapping[str, Any]],
    stabilization: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact output provenance."""

    outputs = {
        "summary_json": repo_rel(SUMMARY_JSON),
        "summary_markdown": repo_rel(SUMMARY_MD),
        "stabilization": stabilization["outputs"],
    }
    for source_key, group in manifests.items():
        outputs[source_key] = group["paths"]
        outputs[source_key]["perturbation_detail"] = group["perturbation"][
            "bulk_detail_manifest"
        ]["path"]
        outputs[source_key]["feedback_detail"] = group["feedback"][
            "bulk_detail_manifest"
        ]["path"]
    return outputs


def aggregation_contract() -> dict[str, str]:
    """Metric definitions for future regeneration."""

    return {
        "sensory_deviation_max_mm": (
            "sensory_feedback class mean delta_position_response_m.max, converted to mm"
        ),
        "sensory_deviation_auc_mm_s": (
            "sensory_feedback class mean delta_position_response_m.auc, converted to mm*s"
        ),
        "non_sensory_deviation_max_mm": (
            "unweighted mean over non-sensory and non-target class means for "
            "delta_position_response_m.max, converted to mm"
        ),
        "non_sensory_deviation_auc_mm_s": (
            "unweighted mean over non-sensory and non-target class means for "
            "delta_position_response_m.auc, converted to mm*s"
        ),
        "feedback_delta_action_norm": (
            "feedback-ablation interpretation max_feedback_delta_action_norm_mean"
        ),
        "feedback_ablation_dependence_index": (
            "feedback_pass_audit.components.feedback_ablation_dependence."
            "ablation_dependence_index"
        ),
        "stabilization_*": (
            "endpoint stabilization-task probe summaries from the c92 stabilization "
            "contract, evaluated locally on these checkpoints"
        ),
        "peak_dx_over_open_loop": (
            "mean non-sensory closed_loop_peak_dx_over_open_loop_peak_dx over "
            "evaluated reach-context perturbation rows"
        ),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the requested summary tables."""

    lines = [
        "# Beta 1.4 Feedback and Robustness Summary",
        "",
        "This summary compares the three completed b413 beta 1.4 rows with the "
        "existing 33b0dcb H0 no-PGD `const_band16` baseline. All values use "
        "validation-selected checkpoints and the calibrated moderate perturbation bank.",
        "",
        "## Sensory and Non-Sensory Deviations",
        "",
        "| Row | Sensory max (mm) | Sensory AUC (mm*s) | Non-sensory max (mm) | "
        "Non-sensory AUC (mm*s) | Peak dx/OL | AUC dx/OL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            f"`{row['row']}` | {fmt(row['sensory_deviation_max_mm'])} | "
            f"{fmt(row['sensory_deviation_auc_mm_s'])} | "
            f"{fmt(row['non_sensory_deviation_max_mm'])} | "
            f"{fmt(row['non_sensory_deviation_auc_mm_s'])} | "
            f"{fmt(row['peak_dx_over_open_loop'])} | "
            f"{fmt(row['auc_dx_over_open_loop'])} |"
        )
    lines.extend(
        [
            "",
            "## Feedback and Stabilization",
            "",
            "| Row | Feedback delta action | Ablation dependence | Feedback audit | "
            "Stab feedback AUC | Stab mechanical AUC | Stab command AUC | "
            "Stab process-force AUC |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| "
            f"`{row['row']}` | {fmt(row['feedback_delta_action_norm'])} | "
            f"{fmt(row['feedback_ablation_dependence_index'])} | "
            f"`{row['feedback_audit_status']}` | "
            f"{fmt(row['stabilization_feedback_auc_mm_s'])} | "
            f"{fmt(row['stabilization_mechanical_auc_mm_s'])} | "
            f"{fmt(row['stabilization_command_auc_mm_s'])} | "
            f"{fmt(row['stabilization_process_force_auc_mm_s'])} |"
        )
    lines.extend(
        [
            "",
            "## Deltas Against Baseline",
            "",
            "Negative deviation/AUC deltas mean lower response than the baseline.",
            "",
            "| Row | Sensory AUC delta | Non-sensory AUC delta | Feedback AUC delta | "
            "Mechanical AUC delta | Feedback delta-action delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row_key, delta in summary["comparisons_vs_baseline"].items():
        lines.append(
            "| "
            f"`{row_key}` | {fmt(delta.get('sensory_deviation_auc_mm_s'))} | "
            f"{fmt(delta.get('non_sensory_deviation_auc_mm_s'))} | "
            f"{fmt(delta.get('stabilization_feedback_auc_mm_s'))} | "
            f"{fmt(delta.get('stabilization_mechanical_auc_mm_s'))} | "
            f"{fmt(delta.get('feedback_delta_action_norm'))} |"
        )
    lines.extend(
        [
            "",
            "## Output Manifests",
            "",
            f"- Summary JSON: `{summary['source_outputs']['summary_json']}`",
            f"- Stabilization detail: `{summary['source_outputs']['stabilization']['detail_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    """Format scalar table values."""

    if value is None:
        return "not available"
    if isinstance(value, int | float):
        return f"{float(value):.4g}"
    return str(value)


def load_repo_json(path: str) -> dict[str, Any]:
    """Load a JSON object from a repo-relative path."""

    return load_json(REPO_ROOT / path)


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    """Return a repo-relative path string."""

    repo_root = Path(REPO_ROOT)
    path = Path(path)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.relative_to(repo_root))


if __name__ == "__main__":
    main()
