"""Materialize PGD 1.05 reach-context feedback and robustness diagnostics."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    materialize_validation_selected_checkpoint_manifest,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
    materialize_gru_evaluation_diagnostics,
)
from rlrmp.analysis.pipelines.gru_feedback_ablation import materialize_gru_feedback_ablation
from rlrmp.analysis.pipelines.gru_perturbation_bank import materialize_gru_perturbation_response
from rlrmp.analysis.pipelines.hinf_phenotype_sidecar import (
    build_hinf_phenotype_sidecar,
    load_hinf_phenotype_sources,
    write_hinf_phenotype_sidecar,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE = "c92ebd8"
TAG = "pgd_1p05_reach_context_diagnostics"
ROWS = (
    "open_loop_small",
    "open_loop_moderate",
    "open_loop_stress",
    "small",
    "moderate",
    "stress",
)
PHYSICAL_LEVELS = {
    "open_loop_small": "small",
    "open_loop_moderate": "moderate",
    "open_loop_stress": "stress",
    "small": "small",
    "moderate": "moderate",
    "stress": "stress",
}
TRAINING_CONDITIONS = {
    "open_loop_small": "no-PGD open-loop calibrated",
    "open_loop_moderate": "no-PGD open-loop calibrated",
    "open_loop_stress": "no-PGD open-loop calibrated",
    "small": "PGD 1.05 open-loop calibrated",
    "moderate": "PGD 1.05 open-loop calibrated",
    "stress": "PGD 1.05 open-loop calibrated",
}
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / TAG
SUMMARY_JSON = NOTES_DIR / f"{TAG}.json"
SUMMARY_MD = NOTES_DIR / f"{TAG}.md"
SUMMARY_CSV = NOTES_DIR / f"{TAG}.csv"


def main() -> None:
    """Materialize component sidecars and write the compact comparison table."""

    mkdir_p(NOTES_DIR)
    mkdir_p(BULK_DIR)
    paths = output_paths()
    labels = tuple(run_label(run_id) for run_id in ROWS)

    checkpoint_manifest = (
        load_json(paths["checkpoint_manifest"])
        if paths["checkpoint_manifest"].exists()
        else materialize_validation_selected_checkpoint_manifest(
            experiment=ISSUE,
            run_ids=ROWS,
            output_path=paths["checkpoint_manifest"],
            repo_root=REPO_ROOT,
        )
    )
    evaluation = (
        load_json(paths["evaluation"])
        if paths["evaluation"].exists()
        else materialize_gru_evaluation_diagnostics(
            experiment=ISSUE,
            run_ids=ROWS,
            labels=labels,
            output_path=paths["evaluation"],
            bulk_dir=BULK_DIR / "evaluation_diagnostics",
            n_rollout_trials=64,
            write_bulk_arrays=False,
            regeneration_spec_path=paths["evaluation_regeneration_spec"],
            repo_root=REPO_ROOT,
        )
    )
    perturbation = (
        load_json(paths["perturbation"])
        if perturbation_output_is_current(paths["perturbation"], expected_trials=64)
        else materialize_gru_perturbation_response(
            source_experiment=ISSUE,
            result_experiment=ISSUE,
            run_ids=ROWS,
            labels=labels,
            n_rollout_trials=64,
            output_path=paths["perturbation"],
            note_path=paths["perturbation_note"],
            bulk_dir=BULK_DIR / "perturbation_response",
            regeneration_spec_path=paths["perturbation_regeneration_spec"],
            bank_mode="calibrated",
            calibration_level="moderate",
            calibration_reach=0.15,
            feedback_scale_manifest_path=paths["evaluation"],
            extlqg_physical_dim=6,
            write_bulk_arrays=False,
            repo_root=REPO_ROOT,
        )
    )
    feedback = (
        load_json(paths["feedback"])
        if run_output_is_current(paths["feedback"], expected_trials=64)
        else materialize_gru_feedback_ablation(
            source_experiment=ISSUE,
            result_experiment=ISSUE,
            scope="pgd_1p05_reach_context_feedback_ablation",
            run_ids=ROWS,
            labels=labels,
            n_rollout_trials=64,
            bank_mode="calibrated",
            calibration_level="moderate",
            calibration_reach=0.15,
            feedback_selection_level="moderate",
            feedback_scale_manifest_path=paths["evaluation"],
            output_path=paths["feedback"],
            note_path=paths["feedback_note"],
            regeneration_spec_path=paths["feedback_regeneration_spec"],
            repo_root=REPO_ROOT,
        )
    )
    sidecar = materialize_hinf_sidecar(paths)
    detail = load_json(Path(perturbation["bulk_detail_manifest"]["path"]))
    rows = [
        table_row(
            run_id,
            evaluation=evaluation,
            feedback=feedback,
            perturbation_detail=detail,
            phenotype_sidecar=sidecar,
        )
        for run_id in ROWS
    ]
    summary = build_summary(
        rows,
        checkpoint_manifest=checkpoint_manifest,
        paths=paths,
        evaluation=evaluation,
        perturbation=perturbation,
        feedback=feedback,
        sidecar=sidecar,
    )
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(rows)
    SUMMARY_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_JSON), "rows": rows}, indent=2))


def output_paths() -> dict[str, Path]:
    """Return all materialized output paths for this diagnostic lane."""

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
        "phenotype": NOTES_DIR / f"hinf_phenotype_sidecar_{TAG}.json",
        "phenotype_note": NOTES_DIR / f"hinf_phenotype_sidecar_{TAG}.md",
        "phenotype_regeneration_spec": NOTES_DIR
        / f"hinf_phenotype_sidecar_{TAG}_regeneration_spec.json",
    }


def materialize_hinf_sidecar(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Build the diagnostic-only H-infinity phenotype sidecar for this subset."""

    sources = load_hinf_phenotype_sources(
        {
            "evaluation_diagnostics": paths["evaluation"],
            "perturbation_response": paths["perturbation"],
            "feedback_ablation": paths["feedback"],
        }
    )
    sidecar = build_hinf_phenotype_sidecar(
        sources=sources,
        issue=ISSUE,
        scope="pgd_1p05_reach_context_interpretive",
        paired_run_ids={
            "open_loop_small": "small",
            "open_loop_moderate": "moderate",
            "open_loop_stress": "stress",
        },
    )
    write_hinf_phenotype_sidecar(
        sidecar,
        json_path=paths["phenotype"],
        markdown_path=paths["phenotype_note"],
        regeneration_spec_path=paths["phenotype_regeneration_spec"],
        repo_root=REPO_ROOT,
    )
    return sidecar


def run_output_is_current(path: Path, *, expected_trials: int) -> bool:
    """Return whether an existing run-keyed manifest has the expected rows/trials."""

    if not path.exists():
        return False
    data = load_json(path)
    runs = data.get("runs", {})
    if sorted(runs) != sorted(ROWS):
        return False
    return all(
        int(runs[run_id].get("n_rollout_trials_per_replicate", -1)) == expected_trials
        for run_id in ROWS
    )


def perturbation_output_is_current(path: Path, *, expected_trials: int) -> bool:
    """Return whether a slim perturbation manifest and detail file are current."""

    if not run_output_is_current(path, expected_trials=expected_trials):
        return False
    data = load_json(path)
    detail_path = data.get("bulk_detail_manifest", {}).get("path")
    return isinstance(detail_path, str) and (REPO_ROOT / detail_path).exists()


def table_row(
    run_id: str,
    *,
    evaluation: Mapping[str, Any],
    feedback: Mapping[str, Any],
    perturbation_detail: Mapping[str, Any],
    phenotype_sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract one compact comparison-table row from materialized outputs."""

    eval_run = evaluation["runs"][run_id]
    feedback_run = feedback["runs"][run_id]
    perturb_run = perturbation_detail["runs"][run_id]
    feedback_audit = feedback_run["feedback_pass_audit"]["components"]
    response_summary = perturb_run["robust_response_summary"]["class_summary"]["groups"]
    peak_velocity = eval_run["behavior"]["velocity_profile"][
        "mean_profile_peak_forward_velocity_m_s"
    ]
    return {
        "row": run_id,
        "training_condition": TRAINING_CONDITIONS[run_id],
        "physical_level": PHYSICAL_LEVELS[run_id],
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
        "formal_hinf_claim": formal_hinf_status(run_id, phenotype_sidecar),
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


def formal_hinf_status(run_id: str, sidecar: Mapping[str, Any]) -> str:
    """Return the formal H-infinity claim status for a run."""

    for row in sidecar.get("rows", ()):
        if row.get("run_id") == run_id:
            return str(row.get("formal_hinf_claim", {}).get("status", "unknown"))
    return "missing"


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    evaluation: Mapping[str, Any],
    perturbation: Mapping[str, Any],
    feedback: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the JSON summary payload."""

    return {
        "schema_version": "rlrmp.c92ebd8.pgd_1p05_reach_context_diagnostics.v1",
        "issue": ISSUE,
        "scope": "PGD 1.05 reach-context perturbation, feedback, robustness diagnostics",
        "rows": list(rows),
        "row_order": list(ROWS),
        "aggregation_contract": {
            "peak_velocity_m_s": (
                "mean_profile_peak_forward_velocity_m_s from evaluation diagnostics"
            ),
            "fb_delta_u": (
                "feedback_ablation.interpretation.max_feedback_delta_action_norm_mean"
            ),
            "ablation_idx": (
                "feedback_pass_audit.components.feedback_ablation_dependence."
                "ablation_dependence_index"
            ),
            "sensory_auc_dx_mm_s": (
                "sensory_feedback/sensory_feedback_offset class mean "
                "delta_position_response_m.auc, converted to mm*s"
            ),
            "non_sensory_auc_dx_mm_s": (
                "unweighted mean of available non-sensory/non-target class means "
                "for delta_position_response_m.auc, converted to mm*s"
            ),
            "peak_dx_over_open_loop": (
                "mean available attenuation_metrics."
                "closed_loop_peak_dx_over_open_loop_peak_dx over evaluated "
                "non-sensory/non-target perturbation rows"
            ),
        },
        "interpretation": interpret_rows(rows),
        "formal_hinf_claim_policy": sidecar["interpretation_contract"],
        "source_outputs": {
            key: repo_rel(path) for key, path in paths.items()
        }
        | {
            "summary_json": repo_rel(SUMMARY_JSON),
            "summary_markdown": repo_rel(SUMMARY_MD),
            "summary_csv": repo_rel(SUMMARY_CSV),
            "perturbation_detail_manifest": perturbation["bulk_detail_manifest"],
        },
        "component_schemas": {
            "checkpoint_manifest": checkpoint_manifest.get("schema_version"),
            "evaluation": evaluation.get("schema_version"),
            "perturbation": perturbation.get("schema_version"),
            "feedback": feedback.get("schema_version"),
            "phenotype_sidecar": sidecar.get("schema_version"),
        },
    }


def interpret_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return conservative qualitative interpretation from the compiled table."""

    by_run = {row["row"]: row for row in rows}
    comparisons = []
    for level in ("small", "moderate", "stress"):
        baseline = by_run[f"open_loop_{level}"]
        pgd = by_run[level]
        comparisons.append(
            {
                "physical_level": level,
                "pgd_minus_no_pgd": {
                    "peak_velocity_m_s": (
                        pgd["peak_velocity_m_s"] - baseline["peak_velocity_m_s"]
                    ),
                    "fb_delta_u": pgd["fb_delta_u"] - baseline["fb_delta_u"],
                    "ablation_idx": pgd["ablation_idx"] - baseline["ablation_idx"],
                    "sensory_auc_dx_mm_s": (
                        pgd["sensory_auc_dx_mm_s"] - baseline["sensory_auc_dx_mm_s"]
                    ),
                    "non_sensory_auc_dx_mm_s": (
                        pgd["non_sensory_auc_dx_mm_s"]
                        - baseline["non_sensory_auc_dx_mm_s"]
                    ),
                    "peak_dx_over_open_loop": (
                        pgd["peak_dx_over_open_loop"]
                        - baseline["peak_dx_over_open_loop"]
                    ),
                },
            }
        )
    robustness_wins = [
        cmp
        for cmp in comparisons
        if cmp["pgd_minus_no_pgd"]["non_sensory_auc_dx_mm_s"] < 0
        and cmp["pgd_minus_no_pgd"]["peak_dx_over_open_loop"] < 0
    ]
    return {
        "qualitative_control_gain": (
            "PGD 1.05 does not read as a simple global control-gain increase: "
            "nominal peak velocity stays close to the no-PGD rows, while feedback "
            "dependence and perturbation-response metrics move unevenly by physical "
            "level."
        ),
        "robustness_answer": (
            "PGD 1.05 does not consistently increase robustness beyond the calibrated "
            "no-PGD open-loop rows on these reach-context diagnostics."
            if len(robustness_wins) < len(comparisons)
            else "PGD 1.05 improves both non-sensory AUC and peak dx/OL at every level."
        ),
        "formal_hinf_claim": (
            "No formal H-infinity claim is made; the phenotype sidecar is "
            "diagnostic-only unless a standard certificate passes."
        ),
        "level_comparisons": comparisons,
    }


def write_csv(rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the compact comparison table as CSV."""

    fields = [
        "row",
        "training_condition",
        "physical_level",
        "peak_velocity_m_s",
        "fb_delta_u",
        "ablation_idx",
        "sensory_auc_dx_mm_s",
        "non_sensory_auc_dx_mm_s",
        "peak_dx_over_open_loop",
        "formal_hinf_claim",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the Markdown interpretation note."""

    lines = [
        "# PGD 1.05 reach-context diagnostics",
        "",
        "Rows compare calibrated no-PGD open-loop comparators against PGD 1.05 rows "
        "at the same physical perturbation level. Values are computed from the "
        "materialized sidecars listed below, not copied from the issue comment.",
        "",
        "| Row | Training condition | Physical level | Peak velocity | fb Δu | "
        "Ablation idx | Sensory AUC Δx | Non-sensory AUC Δx | Peak dx/OL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            f"`{row['row']}` | {row['training_condition']} | `{row['physical_level']}` | "
            f"{row['peak_velocity_m_s']:.5f} | {row['fb_delta_u']:.3f} | "
            f"{row['ablation_idx']:.3f} | {row['sensory_auc_dx_mm_s']:.3f} | "
            f"{row['non_sensory_auc_dx_mm_s']:.3f} | "
            f"{row['peak_dx_over_open_loop']:.3f} |"
        )
    interpretation = summary["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation["qualitative_control_gain"],
            "",
            interpretation["robustness_answer"],
            "",
            interpretation["formal_hinf_claim"],
            "",
            "## Source outputs",
            "",
        ]
    )
    for key, path in summary["source_outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Aggregation contract",
            "",
        ]
    )
    for key, text in summary["aggregation_contract"].items():
        lines.append(f"- `{key}`: {text}.")
    return "\n".join(lines) + "\n"


def run_label(run_id: str) -> str:
    """Return a display label that keeps PGD and no-PGD rows unambiguous."""

    return f"{TRAINING_CONDITIONS[run_id]} / {PHYSICAL_LEVELS[run_id]}"


def repo_rel(path: Path) -> str:
    """Return a repo-relative path."""

    return str(path.relative_to(REPO_ROOT))


def load_json(path: Path) -> Any:
    """Load JSON from a repo-relative or absolute path."""

    resolved = path if path.is_absolute() else REPO_ROOT / path
    return json.loads(resolved.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
