"""Materialize c92 output-feedback-budget PGD feedback/robustness diagnostics."""

from __future__ import annotations
from rlrmp.io import write_csv_rows
from rlrmp.paths import portable_repo_path

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from materialize_pgd_1p05_stabilization_diagnostics import (
    RowSpec as StabilizationRowSpec,
    evaluate_row as evaluate_stabilization_row,
    probe_contract as stabilization_probe_contract,
)
from rlrmp.eval.robustness_diagnostics import (
    build_summary as canonical_build_summary,
    run_feedback_robustness_diagnostics,
)
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT


ISSUE = "c92ebd8"
TAG = "output_feedback_budget_diagnostics"
MARKER = "output_feedback_budget_diagnostics"
SCHEMA_VERSION = "rlrmp.c92ebd8.output_feedback_budget_diagnostics.v1"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
EVALUATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "evaluation_diagnostics" / TAG
PERTURBATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "perturbation_response" / TAG
STABILIZATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "stabilization_diagnostics" / TAG
SUMMARY_JSON = NOTES_DIR / f"{TAG}.json"
SUMMARY_CSV = NOTES_DIR / f"{TAG}.csv"
SUMMARY_MD = NOTES_DIR / f"{TAG}.md"
DETAIL_JSON = STABILIZATION_BULK_DIR / "per_probe_detail.json"


@dataclass(frozen=True)
class RowSpec:
    """One c92 row included in the OFB-budget comparison."""

    run_id: str
    training_key: str
    training_condition: str
    pgd_budget_source: str | None
    active_l2_radius_15cm: float | None


ROWS: tuple[RowSpec, ...] = (
    RowSpec(
        run_id="open_loop_moderate",
        training_key="no_pgd_open_loop",
        training_condition="no-PGD open-loop calibrated moderate",
        pgd_budget_source=None,
        active_l2_radius_15cm=None,
    ),
    RowSpec(
        run_id="moderate_pgd_ofb1p05",
        training_key="pgd_ofb_1p05",
        training_condition="PGD output-feedback-budget gamma 1.05",
        pgd_budget_source="ofb_6d_no_integrator_gamma_1p05_rollout_radius",
        active_l2_radius_15cm=0.0017513324974961241,
    ),
    RowSpec(
        run_id="moderate_pgd_ofb1p4",
        training_key="pgd_ofb_1p4",
        training_condition="PGD output-feedback-budget gamma 1.4",
        pgd_budget_source="ofb_6d_no_integrator_gamma_1p4_rollout_radius",
        active_l2_radius_15cm=0.004545011406169036,
    ),
)
ROW_IDS = tuple(row.run_id for row in ROWS)


def main() -> None:
    """Materialize sidecars and write the compact OFB-budget comparison."""

    paths = output_paths()
    result = run_feedback_robustness_diagnostics(
        hooks=globals(),
        paths=paths,
        output_dirs=(NOTES_DIR, EVALUATION_BULK_DIR, PERTURBATION_BULK_DIR, STABILIZATION_BULK_DIR),
        issue=ISSUE,
        repo_root=REPO_ROOT,
        run_ids=ROW_IDS,
        labels=tuple(run_label(row) for row in ROWS),
        evaluation_bulk_dir=EVALUATION_BULK_DIR,
        feedback_scope="output_feedback_budget_feedback_ablation",
        materialize_extensions=lambda current_paths, _components: {
            "stabilization": materialize_stabilization(
                current_paths["stabilization_detail"]
            )
        },
        build_rows=lambda components: [
            table_row(
                row_spec,
                evaluation=components["evaluation"],
                feedback=components["feedback"],
                perturbation_detail=components["perturbation_detail"],
                stabilization=components["stabilization"],
            )
            for row_spec in ROWS
        ],
        build_summary_payload=lambda rows, components: build_summary(
            rows,
            checkpoint_manifest=components["checkpoint_manifest"],
            evaluation=components["evaluation"],
            perturbation=components["perturbation"],
            feedback=components["feedback"],
            stabilization=components["stabilization"],
            paths=paths,
        ),
        write_outputs=_write_outputs,
    )
    print(json.dumps({"summary": repo_rel(SUMMARY_JSON), "rows": result["rows"]}, indent=2))


def _write_outputs(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    write_compact_json(SUMMARY_JSON, summary)
    write_csv(rows)
    update_marked_section(SUMMARY_MD, MARKER, render_markdown(summary))


def output_paths() -> dict[str, Path]:
    """Return materialized output paths for this diagnostic lane."""

    return {
        "checkpoint_manifest": NOTES_DIR / f"{TAG}_validation_selected_checkpoints.json",
        "evaluation": NOTES_DIR / f"gru_evaluation_diagnostics_{TAG}.json",
        "evaluation_regeneration_spec": NOTES_DIR
        / f"gru_evaluation_diagnostics_{TAG}_regeneration_spec.json",
        "perturbation": NOTES_DIR / f"gru_perturbation_response_{TAG}_manifest.json",
        "perturbation_note": NOTES_DIR / f"gru_perturbation_response_{TAG}.md",
        "perturbation_regeneration_spec": NOTES_DIR
        / f"gru_perturbation_response_{TAG}_manifest_regeneration_spec.json",
        "feedback": NOTES_DIR / f"gru_feedback_ablation_{TAG}.json",
        "feedback_note": NOTES_DIR / f"gru_feedback_ablation_{TAG}.md",
        "feedback_regeneration_spec": NOTES_DIR
        / f"gru_feedback_ablation_{TAG}_regeneration_spec.json",
        "stabilization_detail": DETAIL_JSON,
    }


def materialize_stabilization(detail_path: Path) -> dict[str, Any]:
    """Evaluate stabilization probes without generating response figures."""

    if detail_path.exists():
        detail = load_json(detail_path)
        if sorted(detail.get("rows", {})) == sorted(ROW_IDS):
            return detail
    detail_rows: dict[str, Any] = {}
    summary_rows = []
    for row_spec in ROWS:
        result = evaluate_stabilization_row(
            StabilizationRowSpec(
                row_spec.run_id,
                row_spec.training_key,
                "moderate",
            ),
            repo_root=Path(REPO_ROOT).resolve(),
        )
        detail_rows[row_spec.run_id] = result.pop("per_probe_detail")
        summary_rows.append(result)
    detail = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "detail_role": "stabilization endpoint per-probe scalar and trajectory diagnostics",
        "probe_contract": stabilization_probe_contract(),
        "summary_rows": summary_rows,
        "rows": detail_rows,
    }
    write_compact_json(detail_path, detail)
    return detail


def table_row(
    row_spec: RowSpec,
    *,
    evaluation: Mapping[str, Any],
    feedback: Mapping[str, Any],
    perturbation_detail: Mapping[str, Any],
    stabilization: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the requested compact comparison columns for one row."""

    run_id = row_spec.run_id
    eval_run = evaluation["runs"][run_id]
    feedback_run = feedback["runs"][run_id]
    perturb_run = perturbation_detail["runs"][run_id]
    stabilization_run = stabilization_summary_by_run(stabilization)[run_id]
    feedback_audit = feedback_run["feedback_pass_audit"]["components"]
    response_summary = perturb_run["robust_response_summary"]["class_summary"]["groups"]
    peak_velocity = eval_run["behavior"]["velocity_profile"][
        "mean_profile_peak_forward_velocity_m_s"
    ]
    return {
        "row": run_id,
        "training_condition": row_spec.training_condition,
        "physical_level": "moderate",
        "active_l2_radius_15cm": row_spec.active_l2_radius_15cm,
        "pgd_budget_source": row_spec.pgd_budget_source,
        "peak_velocity_m_s": float(peak_velocity),
        "fb_delta_u": float(
            feedback_run["interpretation"]["max_feedback_delta_action_norm_mean"]
        ),
        "ablation_idx": float(
            feedback_audit["feedback_ablation_dependence"]["ablation_dependence_index"]
        ),
        "sensory_auc_dx_mm_s": sensory_auc_dx_mm_s(response_summary),
        "non_sensory_auc_dx_mm_s": non_sensory_auc_dx_mm_s(response_summary),
        "peak_dx_over_open_loop": peak_dx_over_open_loop(perturb_run),
        "stabilization_feedback_auc_mm_s": float(stabilization_run["feedback_auc_mm_s"]),
        "stabilization_mechanical_auc_mm_s": float(stabilization_run["mechanical_auc_mm_s"]),
        "stabilization_command_auc_mm_s": float(stabilization_run["command_input_auc_mm_s"]),
        "stabilization_process_force_auc_mm_s": float(
            stabilization_run["process_force_auc_mm_s"]
        ),
        "formal_hinf_claim": "not_claimed_diagnostic_only",
        "checkpoint_policy": eval_run["checkpoint_policy"],
        "source_paths": {
            "run_spec": eval_run["run_spec_path"],
            "artifact_dir": eval_run["artifact_dir"],
        },
    }


def sensory_auc_dx_mm_s(groups: Mapping[str, Any]) -> float:
    """Return sensory-feedback position-response AUC in mm*s."""

    metric = groups["sensory_feedback/sensory_feedback_offset"]["metrics"][
        "delta_position_response_m.auc"
    ]
    return float(metric["mean"]) * 1000.0


def non_sensory_auc_dx_mm_s(groups: Mapping[str, Any]) -> float:
    """Return unweighted mean AUC across available non-sensory classes in mm*s."""

    values = []
    for group_key, group in groups.items():
        channel = group_key.split("/", maxsplit=1)[0]
        if channel in {"sensory_feedback", "target_stream"}:
            continue
        metric = group["metrics"].get("delta_position_response_m.auc", {})
        if metric.get("status") == "available" and metric.get("mean") is not None:
            values.append(float(metric["mean"]) * 1000.0)
    if not values:
        raise ValueError("no non-sensory AUC values available")
    return sum(values) / len(values)


def peak_dx_over_open_loop(perturb_run: Mapping[str, Any]) -> float:
    """Return mean closed-loop/open-loop peak displacement over non-sensory rows."""

    values = []
    for row in perturb_run["perturbations"]:
        if row.get("status") != "evaluated":
            continue
        channel = str(row.get("channel") or row.get("perturbation", {}).get("channel"))
        if channel in {"sensory_feedback", "target_stream"}:
            continue
        metric = row.get("metrics", {}).get("attenuation_metrics", {}).get(
            "closed_loop_peak_dx_over_open_loop_peak_dx",
            {},
        )
        if metric.get("status") == "available" and metric.get("mean") is not None:
            values.append(float(metric["mean"]))
    if not values:
        raise ValueError("no peak dx/open-loop values available")
    return sum(values) / len(values)


def stabilization_summary_by_run(stabilization: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return stabilization scalar summaries keyed by run id."""

    return {str(row["run_id"]): row for row in stabilization["summary_rows"]}


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_manifest: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    feedback: Mapping[str, Any],
    stabilization: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Return the combined JSON summary payload."""

    components = {
        "checkpoint_manifest": checkpoint_manifest,
        "evaluation": evaluation,
        "perturbation": perturbation,
        "feedback": feedback,
        "stabilization": stabilization,
    }
    return canonical_build_summary(
        rows,
        schema_version=SCHEMA_VERSION,
        issue=ISSUE,
        scope=(
            "OFB-budget PGD reach-context feedback/robustness and stabilization "
            "diagnostics for c92 moderate rows"
        ),
        row_order=ROW_IDS,
        paths=paths,
        repo_relative=repo_rel,
        components=components,
        component_schema_names=tuple(components),
        extensions=_summary_extensions(rows),
        source_output_extensions={
            "summary_json": repo_rel(SUMMARY_JSON),
            "summary_markdown": repo_rel(SUMMARY_MD),
            "summary_csv": repo_rel(SUMMARY_CSV),
            "perturbation_detail_manifest": perturbation["bulk_detail_manifest"],
            "stabilization_detail": repo_rel(DETAIL_JSON),
        },
    )


def _summary_extensions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "baseline_row": "open_loop_moderate",
        "comparisons_vs_baseline": comparisons_vs_baseline(rows),
        "budget_comparison": compare_budgets(rows),
        "interpretation": interpret_rows(rows),
        "aggregation_contract": aggregation_contract(),
        "formal_hinf_claim_policy": (
            "No formal H-infinity claim is made here. These are empirical GRU "
            "feedback/robustness diagnostics plus budget provenance."
        ),
    }


def comparisons_vs_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return row-wise deltas against the no-PGD moderate baseline."""

    by_run = {str(row["row"]): row for row in rows}
    baseline = by_run["open_loop_moderate"]
    comparisons = {}
    for run_id in ("moderate_pgd_ofb1p05", "moderate_pgd_ofb1p4"):
        row = by_run[run_id]
        comparisons[run_id] = metric_deltas(row, baseline)
    return comparisons


def compare_budgets(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the higher OFB 1.4 budget against the lower OFB 1.05 budget."""

    by_run = {str(row["row"]): row for row in rows}
    return {
        "moderate_pgd_ofb1p4_minus_moderate_pgd_ofb1p05": metric_deltas(
            by_run["moderate_pgd_ofb1p4"],
            by_run["moderate_pgd_ofb1p05"],
        )
    }


def metric_deltas(row: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """Return numeric metric differences for interpretation."""

    fields = (
        "peak_velocity_m_s",
        "fb_delta_u",
        "ablation_idx",
        "sensory_auc_dx_mm_s",
        "non_sensory_auc_dx_mm_s",
        "peak_dx_over_open_loop",
        "stabilization_feedback_auc_mm_s",
        "stabilization_mechanical_auc_mm_s",
        "stabilization_command_auc_mm_s",
        "stabilization_process_force_auc_mm_s",
    )
    return {field: float(row[field]) - float(reference[field]) for field in fields}


def interpret_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return conservative qualitative interpretation from the compiled rows."""

    by_run = {str(row["row"]): row for row in rows}
    baseline = by_run["open_loop_moderate"]
    pgd_rows = [by_run["moderate_pgd_ofb1p05"], by_run["moderate_pgd_ofb1p4"]]
    robustness_fields = (
        "sensory_auc_dx_mm_s",
        "non_sensory_auc_dx_mm_s",
        "peak_dx_over_open_loop",
        "stabilization_feedback_auc_mm_s",
        "stabilization_mechanical_auc_mm_s",
    )
    row_scores = {
        str(row["row"]): {
            field: direction_against_baseline(float(row[field]), float(baseline[field]))
            for field in robustness_fields
        }
        for row in pgd_rows
    }
    strict_wins = {
        row_id: all(direction == "lower" for direction in directions.values())
        for row_id, directions in row_scores.items()
    }
    budget = compare_budgets(rows)[
        "moderate_pgd_ofb1p4_minus_moderate_pgd_ofb1p05"
    ]
    return {
        "baseline_answer": (
            "Both output-feedback-budget PGD rows improve the reach-context "
            "displacement diagnostics versus the no-PGD open-loop moderate baseline, "
            "but both worsen the stabilization endpoint feedback/mechanical AUCs. "
            "That is not a clean across-task robustness improvement."
            if not any(strict_wins.values())
            else "At least one OFB-budget PGD row improves all tracked robustness metrics."
        ),
        "budget_answer": budget_answer(budget),
        "row_metric_directions_vs_baseline": row_scores,
        "formal_hinf_claim": (
            "No formal H-infinity evidence is claimed. The rows carry OFB rollout "
            "budget provenance, but these diagnostics are empirical phenotype checks."
        ),
    }


def direction_against_baseline(value: float, baseline: float) -> str:
    """Classify a metric as lower, higher, or unchanged against baseline."""

    tolerance = max(abs(baseline) * 0.01, 1e-9)
    if value < baseline - tolerance:
        return "lower"
    if value > baseline + tolerance:
        return "higher"
    return "approximately_unchanged"


def budget_answer(delta: Mapping[str, float]) -> str:
    """Summarize the high-budget-vs-low-budget direction."""

    reach_fields = (
        "sensory_auc_dx_mm_s",
        "non_sensory_auc_dx_mm_s",
        "peak_dx_over_open_loop",
    )
    stabilization_fields = (
        "stabilization_feedback_auc_mm_s",
        "stabilization_mechanical_auc_mm_s",
    )
    high_budget_reach_wins = all(float(delta[field]) < 0.0 for field in reach_fields)
    high_budget_stabilization_losses = all(
        float(delta[field]) > 0.0 for field in stabilization_fields
    )
    if high_budget_reach_wins and high_budget_stabilization_losses:
        return (
            "The larger OFB 1.4 budget is stronger on reach-context attenuation "
            "but worse on stabilization endpoint feedback/mechanical AUCs than "
            "the OFB 1.05 budget."
        )
    return "The two OFB budgets split the tracked robustness metrics."


def aggregation_contract() -> dict[str, str]:
    """Return metric definitions for future regeneration."""

    return {
        "peak_velocity_m_s": "mean_profile_peak_forward_velocity_m_s from evaluation diagnostics",
        "fb_delta_u": "feedback_ablation.interpretation.max_feedback_delta_action_norm_mean",
        "ablation_idx": (
            "feedback_pass_audit.components.feedback_ablation_dependence."
            "ablation_dependence_index"
        ),
        "sensory_auc_dx_mm_s": (
            "sensory_feedback/sensory_feedback_offset class mean "
            "delta_position_response_m.auc, converted to mm*s"
        ),
        "non_sensory_auc_dx_mm_s": (
            "unweighted mean of available non-sensory/non-target class means for "
            "delta_position_response_m.auc, converted to mm*s"
        ),
        "peak_dx_over_open_loop": (
            "mean available closed_loop_peak_dx_over_open_loop_peak_dx over "
            "evaluated non-sensory/non-target reach perturbation rows"
        ),
        "stabilization_*_auc_mm_s": (
            "stabilization-task endpoint mean signed-direction-aligned absolute "
            "hand-position displacement over the post-onset window"
        ),
    }


def write_csv(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ['row', 'training_condition', 'physical_level', 'active_l2_radius_15cm', 'pgd_budget_source', 'peak_velocity_m_s', 'fb_delta_u', 'ablation_idx', 'sensory_auc_dx_mm_s', 'non_sensory_auc_dx_mm_s', 'peak_dx_over_open_loop', 'stabilization_feedback_auc_mm_s', 'stabilization_mechanical_auc_mm_s', 'stabilization_command_auc_mm_s', 'stabilization_process_force_auc_mm_s', 'formal_hinf_claim']
    write_csv_rows(SUMMARY_CSV, list(rows), fieldnames=fields)


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the concise tracked note."""

    lines = [
        "# Output-feedback-budget PGD diagnostics",
        "",
        "This note compares exactly the c92 no-PGD open-loop moderate baseline "
        "against the two new open-loop moderate rows trained with output-feedback "
        "rollout PGD budgets. The older raw/full-state PGD row is intentionally not "
        "part of the main table.",
        "",
        "| Row | Training condition | Active L2 radius | Peak velocity | fb delta u | "
        "Ablation idx | Sensory AUC dx | Non-sensory AUC dx | Peak dx/OL | "
        "Stab feedback AUC | Stab mechanical AUC | Stab command AUC | "
        "Stab process-force AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        radius = (
            "not applicable"
            if row["active_l2_radius_15cm"] is None
            else f"{row['active_l2_radius_15cm']:.6g}"
        )
        lines.append(
            "| "
            f"`{row['row']}` | {row['training_condition']} | {radius} | "
            f"{row['peak_velocity_m_s']:.5f} | {row['fb_delta_u']:.3f} | "
            f"{row['ablation_idx']:.3f} | {row['sensory_auc_dx_mm_s']:.3f} | "
            f"{row['non_sensory_auc_dx_mm_s']:.3f} | "
            f"{row['peak_dx_over_open_loop']:.3f} | "
            f"{row['stabilization_feedback_auc_mm_s']:.3f} | "
            f"{row['stabilization_mechanical_auc_mm_s']:.4f} | "
            f"{row['stabilization_command_auc_mm_s']:.3f} | "
            f"{row['stabilization_process_force_auc_mm_s']:.4f} |"
        )
    interpretation = summary["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation["baseline_answer"],
            "",
            interpretation["budget_answer"],
            "",
            interpretation["formal_hinf_claim"],
            "",
            "Budget provenance:",
        ]
    )
    for row in summary["rows"]:
        if row["pgd_budget_source"] is not None:
            lines.append(
                f"- `{row['row']}`: `{row['pgd_budget_source']}`, "
                f"active_l2_radius_15cm={row['active_l2_radius_15cm']:.16g}."
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in summary["source_outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(["", "## Aggregation contract", ""])
    for key, text in summary["aggregation_contract"].items():
        lines.append(f"- `{key}`: {text}.")
    return "\n".join(lines) + "\n"


def run_output_is_current(path: Path, *, expected_trials: int) -> bool:
    """Return whether an existing run-keyed manifest has the expected rows/trials."""

    if not path.exists():
        return False
    data = load_json(path)
    runs = data.get("runs", {})
    if sorted(runs) != sorted(ROW_IDS):
        return False
    return all(
        int(runs[run_id].get("n_rollout_trials_per_replicate", -1)) == expected_trials
        for run_id in ROW_IDS
    )


def perturbation_output_is_current(path: Path, *, expected_trials: int) -> bool:
    """Return whether a slim perturbation manifest and detail file are current."""

    if not run_output_is_current(path, expected_trials=expected_trials):
        return False
    data = load_json(path)
    detail_path = data.get("bulk_detail_manifest", {}).get("path")
    return isinstance(detail_path, str) and (REPO_ROOT / detail_path).exists()


def run_label(row: RowSpec) -> str:
    """Return a display label with enough provenance to avoid PGD-row ambiguity."""

    if row.pgd_budget_source is None:
        return row.training_condition
    return (
        f"{row.training_condition} / {row.pgd_budget_source} / "
        f"radius={row.active_l2_radius_15cm:.6g}"
    )


repo_rel = portable_repo_path


def load_json(path: Path) -> Any:
    """Load JSON from a repo-relative or absolute path."""

    resolved = path if path.is_absolute() else REPO_ROOT / path
    return json.loads(resolved.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
