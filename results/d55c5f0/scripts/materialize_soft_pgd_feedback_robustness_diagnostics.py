"""Materialize d55 soft-PGD feedback-bank robustness diagnostics."""

from __future__ import annotations
from rlrmp.io import write_csv_rows

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from _soft_pgd_materializer_common import (
    ISSUE,
    SOFT_ROWS,
    SOFT_RUN_IDS,
    assert_soft_inputs_ready,
    load_c92_module,
    repo_rel,
)
from rlrmp.eval.robustness_diagnostics import (
    build_summary as canonical_build_summary,
    run_feedback_robustness_diagnostics,
)
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT


TAG = "soft_pgd_feedback_robustness_diagnostics"
MARKER = TAG
SCHEMA_VERSION = "rlrmp.d55c5f0.soft_pgd_feedback_robustness_diagnostics.v1"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
EVALUATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "evaluation_diagnostics" / TAG
PERTURBATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "perturbation_response" / TAG
STABILIZATION_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "stabilization_diagnostics" / TAG
SUMMARY_JSON = NOTES_DIR / f"{TAG}.json"
SUMMARY_CSV = NOTES_DIR / f"{TAG}.csv"
SUMMARY_MD = NOTES_DIR / f"{TAG}.md"
DETAIL_JSON = STABILIZATION_BULK_DIR / "per_probe_detail.json"


def main() -> None:
    """Materialize sidecars and write the compact d55 soft-PGD comparison."""

    assert_soft_inputs_ready()
    base, ofb = load_reference_materializers()
    patch_reference_materializers(base, ofb)
    paths = ofb.output_paths()
    result = run_feedback_robustness_diagnostics(
        hooks=ofb,
        paths=paths,
        output_dirs=(NOTES_DIR, EVALUATION_BULK_DIR, PERTURBATION_BULK_DIR, STABILIZATION_BULK_DIR),
        issue=ISSUE,
        repo_root=REPO_ROOT,
        run_ids=SOFT_RUN_IDS,
        labels=tuple(row.label for row in SOFT_ROWS),
        evaluation_bulk_dir=EVALUATION_BULK_DIR,
        feedback_scope="soft_pgd_feedback_robustness_ablation",
        materialize_extensions=lambda current_paths, _components: {
            "stabilization": ofb.materialize_stabilization(
                current_paths["stabilization_detail"]
            )
        },
        build_rows=lambda components: [
            ofb.table_row(
                row_spec,
                evaluation=components["evaluation"],
                feedback=components["feedback"],
                perturbation_detail=components["perturbation_detail"],
                stabilization=components["stabilization"],
            )
            for row_spec in ofb.ROWS
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
        write_outputs=lambda summary, rows: _write_outputs(summary, rows),
    )
    print(json.dumps({"summary": repo_rel(SUMMARY_JSON), "rows": result["rows"]}, indent=2))


def _write_outputs(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    write_compact_json(SUMMARY_JSON, summary)
    write_csv(rows)
    update_marked_section(SUMMARY_MD, MARKER, render_markdown(summary))


def load_reference_materializers() -> tuple[Any, Any]:
    """Load c92 reference modules, preloading their local dependency names."""

    base = load_c92_module(
        "materialize_pgd_1p05_stabilization_diagnostics",
        "materialize_pgd_1p05_stabilization_diagnostics.py",
    )
    ofb = load_c92_module(
        "c92_ofb_budget_feedback_robustness_diagnostics",
        "materialize_ofb_budget_feedback_robustness_diagnostics.py",
    )
    return base, ofb


def patch_reference_materializers(base: Any, ofb: Any) -> None:
    """Point reused c92 logic at d55 rows and d55 output paths."""

    base.ISSUE = ISSUE
    ofb.ISSUE = ISSUE
    ofb.TAG = TAG
    ofb.MARKER = MARKER
    ofb.SCHEMA_VERSION = SCHEMA_VERSION
    ofb.NOTES_DIR = NOTES_DIR
    ofb.EVALUATION_BULK_DIR = EVALUATION_BULK_DIR
    ofb.PERTURBATION_BULK_DIR = PERTURBATION_BULK_DIR
    ofb.STABILIZATION_BULK_DIR = STABILIZATION_BULK_DIR
    ofb.SUMMARY_JSON = SUMMARY_JSON
    ofb.SUMMARY_CSV = SUMMARY_CSV
    ofb.SUMMARY_MD = SUMMARY_MD
    ofb.DETAIL_JSON = DETAIL_JSON
    ofb.ROWS = tuple(
        ofb.RowSpec(
            run_id=row.run_id,
            training_key=row.training_key,
            training_condition=row.label,
            pgd_budget_source=f"soft_pgd_gamma_factor_{row.gamma_factor:g}",
            active_l2_radius_15cm=None,
        )
        for row in SOFT_ROWS
    )
    ofb.ROW_IDS = SOFT_RUN_IDS


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
            "Soft-constraint PGD reach-context feedback/robustness and "
            "stabilization diagnostics for the d55 first-batch rows"
        ),
        row_order=SOFT_RUN_IDS,
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
        "soft_gamma_comparisons": adjacent_soft_gamma_comparisons(rows),
        "interpretation_contract": {
            "status": "diagnostic_materializer_only",
            "note": (
                "This script aggregates empirical diagnostics. It does not claim "
                "a formal H-infinity certificate or choose a winning soft scale."
            ),
        },
        "aggregation_contract": aggregation_contract(),
        "formal_hinf_claim_policy": (
            "No formal H-infinity claim is made here. These are empirical GRU "
            "feedback/robustness diagnostics plus soft-gamma provenance."
        ),
    }


def adjacent_soft_gamma_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return adjacent row deltas across the soft-gamma scale."""

    by_run = {str(row["row"]): row for row in rows}
    return {
        "soft_pgd_ofb1p4_minus_soft_pgd_ofb1p05": metric_deltas(
            by_run["soft_pgd_ofb1p4"],
            by_run["soft_pgd_ofb1p05"],
        ),
        "soft_pgd_ofb1p8_minus_soft_pgd_ofb1p4": metric_deltas(
            by_run["soft_pgd_ofb1p8"],
            by_run["soft_pgd_ofb1p4"],
        ),
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
        "# Soft-PGD Feedback Robustness Diagnostics",
        "",
        "This note compares only the d55 first-batch soft-constraint PGD rows. "
        "It intentionally does not read or overwrite c92 OFB-budget outputs.",
        "",
        "| Row | Gamma factor | Peak velocity | fb delta u | Ablation idx | "
        "Sensory AUC dx | Non-sensory AUC dx | Peak dx/OL | Stab feedback AUC | "
        "Stab mechanical AUC | Stab command AUC | Stab process-force AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gamma_by_run = {row.run_id: row.gamma_factor for row in SOFT_ROWS}
    for row in summary["rows"]:
        lines.append(
            "| "
            f"`{row['row']}` | {gamma_by_run[str(row['row'])]:.2f} | "
            f"{row['peak_velocity_m_s']:.5f} | {row['fb_delta_u']:.3f} | "
            f"{row['ablation_idx']:.3f} | {row['sensory_auc_dx_mm_s']:.3f} | "
            f"{row['non_sensory_auc_dx_mm_s']:.3f} | "
            f"{row['peak_dx_over_open_loop']:.3f} | "
            f"{row['stabilization_feedback_auc_mm_s']:.3f} | "
            f"{row['stabilization_mechanical_auc_mm_s']:.4f} | "
            f"{row['stabilization_command_auc_mm_s']:.3f} | "
            f"{row['stabilization_process_force_auc_mm_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Soft-gamma deltas",
            "",
        ]
    )
    for name, deltas in summary["soft_gamma_comparisons"].items():
        lines.append(f"- `{name}`:")
        for key, value in deltas.items():
            lines.append(f"  - `{key}`: {value:.6g}")
    lines.extend(["", "## Outputs", ""])
    for key, path in summary["source_outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(["", "## Aggregation contract", ""])
    for key, text in summary["aggregation_contract"].items():
        lines.append(f"- `{key}`: {text}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
