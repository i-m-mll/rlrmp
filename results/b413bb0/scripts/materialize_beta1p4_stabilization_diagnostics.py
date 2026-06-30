"""Materialize beta 1.4 stabilization-task endpoint diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE = "b413bb0"
TAG = "beta1p4_stabilization_diagnostics"
MARKER = "beta1p4_stabilization_diagnostics"
SCHEMA_VERSION = "rlrmp.b413bb0.beta1p4_stabilization_diagnostics.v1"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "stabilization_diagnostics" / TAG
DETAIL_JSON = BULK_DIR / "per_probe_detail.json"
SUMMARY_JSON = NOTES_DIR / f"{TAG}.json"
SUMMARY_MD = NOTES_DIR / f"{TAG}.md"


@dataclass(frozen=True)
class StabilizationRowSource:
    """One trained row to evaluate on endpoint stabilization probes."""

    row_key: str
    source_experiment: str
    run_id: str
    training_key: str
    training_condition: str
    physical_level: str = "moderate"


DEFAULT_ROWS: tuple[StabilizationRowSource, ...] = (
    StabilizationRowSource(
        row_key="baseline_no_pgd_h0_const_band16",
        source_experiment="33b0dcb",
        run_id="h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64",
        training_key="baseline_no_pgd_h0_const_band16",
        training_condition="no-PGD H0 6D open-loop moderate const_band16 baseline",
    ),
    StabilizationRowSource(
        row_key="direct_epsilon",
        source_experiment=ISSUE,
        run_id="direct_epsilon",
        training_key="beta1p4_direct_epsilon",
        training_condition="beta 1.4 direct-epsilon PGD",
    ),
    StabilizationRowSource(
        row_key="linear_no_bias",
        source_experiment=ISSUE,
        run_id="linear_no_bias",
        training_key="beta1p4_linear_no_bias",
        training_condition="beta 1.4 finite linear no-bias adversary",
    ),
    StabilizationRowSource(
        row_key="affine",
        source_experiment=ISSUE,
        run_id="affine",
        training_key="beta1p4_affine",
        training_condition="beta 1.4 finite affine adversary",
    ),
)


def main() -> None:
    """CLI entry point."""

    summary = materialize_stabilization_diagnostics()
    print(
        {
            "summary_json": repo_rel(SUMMARY_JSON),
            "summary_markdown": repo_rel(SUMMARY_MD),
            "detail_json": repo_rel(DETAIL_JSON),
            "rows": [row["row"] for row in summary["rows"]],
        }
    )


def materialize_stabilization_diagnostics(
    rows: Sequence[StabilizationRowSource] = DEFAULT_ROWS,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate endpoint stabilization probes and write b413-local outputs."""

    mkdir_p(NOTES_DIR)
    mkdir_p(BULK_DIR)
    row_keys = tuple(row.row_key for row in rows)
    if DETAIL_JSON.exists() and SUMMARY_JSON.exists() and not force:
        summary = load_json(SUMMARY_JSON)
        if tuple(summary.get("row_order", ())) == row_keys:
            return summary

    helper = load_c92_stabilization_helper()
    detail_rows: dict[str, Any] = {}
    summary_rows = []
    for row in rows:
        helper.ISSUE = row.source_experiment
        result = evaluate_row_allowing_missing_families(
            helper,
            helper.RowSpec(row.run_id, row.training_key, row.physical_level),
            repo_root=Path(REPO_ROOT).resolve(),
        )
        per_probe_detail = result.pop("per_probe_detail")
        result = {
            "row": row.row_key,
            "source_experiment": row.source_experiment,
            "source_run_id": row.run_id,
            "training_condition": row.training_condition,
            **result,
        }
        detail_rows[row.row_key] = {
            "source_experiment": row.source_experiment,
            "source_run_id": row.run_id,
            "per_probe_detail": per_probe_detail,
        }
        summary_rows.append(result)

    detail = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "detail_role": "stabilization endpoint per-probe scalar and trajectory diagnostics",
        "probe_contract": helper.probe_contract(),
        "row_order": list(row_keys),
        "rows": detail_rows,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "scope": (
            "beta 1.4 b413 rows plus the 33b0dcb H0 no-PGD const_band16 baseline "
            "evaluated on stabilization-task endpoint probes"
        ),
        "row_order": list(row_keys),
        "baseline_row": "baseline_no_pgd_h0_const_band16",
        "probe_contract": helper.probe_contract(),
        "rows": summary_rows,
        "comparisons_vs_baseline": comparisons_vs_baseline(summary_rows),
        "outputs": {
            "summary_json": repo_rel(SUMMARY_JSON),
            "summary_markdown": repo_rel(SUMMARY_MD),
            "detail_json": repo_rel(DETAIL_JSON),
        },
    }
    write_compact_json(DETAIL_JSON, detail)
    write_compact_json(SUMMARY_JSON, summary)
    update_marked_section(SUMMARY_MD, MARKER, render_markdown(summary))
    return summary


def load_c92_stabilization_helper() -> Any:
    """Load the existing c92 stabilization probe implementation by file path."""

    helper_path = (
        REPO_ROOT
        / "results"
        / "c92ebd8"
        / "scripts"
        / "materialize_pgd_1p05_stabilization_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "c92_pgd_1p05_stabilization_diagnostics",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load stabilization helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_row_allowing_missing_families(
    helper: Any,
    row_spec: Any,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate one row while preserving unsupported stabilization probes as missing."""

    try:
        return helper.evaluate_row(row_spec, repo_root=repo_root)
    except KeyError as exc:
        if exc.args != ("process_epsilon_force_state_xy",):
            raise

    run = helper.resolve_run_inputs(
        experiment=helper.ISSUE,
        run_ids=[row_spec.run_id],
        labels=[row_spec.run_id],
        repo_root=repo_root,
    )[0]
    hps = helper.dict_to_namespace(
        helper.normalize_gru_hps(run.run_spec["hps"]),
        to_type=helper.TreeNamespace,
    )
    seed = int(run.run_spec.get("seed", 42))
    pair = helper.setup_task_model_pair(hps, key=helper.jr.PRNGKey(seed))
    n_replicates = int(hps.model.n_replicates)
    model, checkpoint_selection = helper.load_validation_selected_checkpoint_model(
        experiment=helper.ISSUE,
        run_id=run.run_id,
        run_spec=run.run_spec,
        checkpoint_selection_mode="sparse_history",
        repo_root=repo_root,
    )
    base_trials = helper.repeat_single_validation_trial(
        pair.task.validation_trials,
        helper.DEFAULT_N_ROLLOUT_TRIALS,
    )
    steady_trials, timing = helper.make_steady_state_trial_specs(
        base_trials,
        delayed=False,
        target_position=helper.np.asarray(
            helper._target_position(run, base_trials),
            dtype=helper.np.float64,
        ),
        pulse_duration_steps=helper.DEFAULT_PULSE_DURATION_STEPS,
        min_post_onset_steps=helper.DEFAULT_POST_ONSET_FIGURE_STEPS,
    )
    steady_trials = helper.pad_feedback_offset_inputs(
        steady_trials,
        expected_feedback_dim=helper._expected_feedback_dim_from_hps(hps),
    )
    steady_trials = helper.zero_disturbance_payload(steady_trials)
    feedback_dim = helper._feedback_dim(steady_trials)
    probes = helper.build_probes(
        feedback_dim=feedback_dim,
        pulse_start=int(timing["pulse_start_step"]),
        pulse_duration=int(timing["pulse_duration_steps"]),
    )
    base = helper._evaluate_model_on_trial_specs(
        model=model,
        task=pair.task,
        trial_specs=steady_trials,
        n_replicates=n_replicates,
        seed=0,
    )
    details = []
    for probe in probes:
        adapter = helper.apply_perturbation_to_trial_specs(steady_trials, probe.row, model=model)
        if adapter.status != "evaluated":
            details.append(
                {
                    "perturbation_id": probe.perturbation_id,
                    "group": probe.group,
                    "family": probe.family,
                    "status": adapter.status,
                    "reason": adapter.reason,
                    "adapter": adapter.to_json(),
                }
            )
            continue
        perturbed = helper._evaluate_model_on_trial_specs(
            model=adapter.model if adapter.model is not None else model,
            task=pair.task,
            trial_specs=adapter.trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        details.append(
            helper.summarize_probe(
                probe=probe,
                base=base,
                perturbed=perturbed,
                pulse_start=int(timing["pulse_start_step"]),
            )
            | {"status": "evaluated", "adapter": adapter.to_json()}
        )
    family_summary = helper.summarize_by_family(details)
    group_summary = helper.summarize_by_group(details)
    feedback_group = group_summary.get("feedback", {})
    mechanical_group = group_summary.get("mechanical", {})
    command_family = family_summary.get("command_input_pulse", {})
    process_family = family_summary.get("process_epsilon_force_state_xy", {})
    return {
        "run_id": row_spec.run_id,
        "training": row_spec.training,
        "physical_level": row_spec.physical_level,
        "run_spec_path": helper.repo_relative(run.run_spec_path, repo_root),
        "artifact_dir": helper.repo_relative(run.artifact_dir, repo_root),
        "checkpoint_selection_summary": helper.checkpoint_selection_summary(
            checkpoint_selection
        ),
        "response_label": helper.response_label(
            helper.washin_diagnostics(base, pulse_start=timing["pulse_start_step"])
        ),
        "dt_s": float(base.dt),
        "timing": timing,
        "n_replicates": int(base.command.shape[0]),
        "n_rollout_trials_per_replicate": int(base.command.shape[1]),
        "feedback_dim": int(feedback_dim),
        "washin": helper.washin_diagnostics(base, pulse_start=timing["pulse_start_step"]),
        "feedback_auc_mm_s": feedback_group.get("auc_displacement_mm_s_mean"),
        "mechanical_auc_mm_s": mechanical_group.get("auc_displacement_mm_s_mean"),
        "command_input_auc_mm_s": command_family.get("auc_displacement_mm_s_mean"),
        "process_force_auc_mm_s": process_family.get("auc_displacement_mm_s_mean"),
        "feedback_peak_mm": feedback_group.get("peak_displacement_mm_mean"),
        "mechanical_peak_mm": mechanical_group.get("peak_displacement_mm_mean"),
        "family_summary": family_summary,
        "missing_families": [
            family
            for family in ("process_epsilon_force_state_xy",)
            if family not in family_summary
        ],
        "per_probe_detail": details,
    }


def comparisons_vs_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return numeric deltas and ratios for each beta row against baseline."""

    by_row = {str(row["row"]): row for row in rows}
    baseline = by_row["baseline_no_pgd_h0_const_band16"]
    fields = (
        "feedback_auc_mm_s",
        "mechanical_auc_mm_s",
        "command_input_auc_mm_s",
        "process_force_auc_mm_s",
        "feedback_peak_mm",
        "mechanical_peak_mm",
    )
    comparisons: dict[str, Any] = {}
    for row_key, row in by_row.items():
        if row_key == "baseline_no_pgd_h0_const_band16":
            continue
        comparisons[row_key] = {
            f"{field}_delta": maybe_delta(row.get(field), baseline.get(field))
            for field in fields
        } | {
            f"{field}_ratio": maybe_ratio(row.get(field), baseline.get(field))
            for field in fields
        }
    return comparisons


def ratio(value: float, reference: float) -> float | None:
    """Return value/reference with a zero guard."""

    if abs(reference) < 1e-12:
        return None
    return float(value / reference)


def maybe_delta(value: Any, reference: Any) -> float | None:
    """Return value-reference when both values are numeric."""

    if not isinstance(value, int | float) or not isinstance(reference, int | float):
        return None
    return float(value) - float(reference)


def maybe_ratio(value: Any, reference: Any) -> float | None:
    """Return value/reference when both values are numeric."""

    if not isinstance(value, int | float) or not isinstance(reference, int | float):
        return None
    return ratio(float(value), float(reference))


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact table note."""

    lines = [
        "# Beta 1.4 Stabilization Diagnostics",
        "",
        "Endpoint stabilization probes reuse the c92 probe contract. AUC values are "
        "mean signed-direction-aligned absolute hand-position displacement after "
        "probe onset in `mm*s`.",
        "",
        "| Row | Source | Training condition | Feedback AUC | Mechanical AUC | "
        "Command AUC | Process-force AUC | Feedback peak | Mechanical peak |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            f"`{row['row']}` | `{row['source_experiment']}` | "
            f"{row['training_condition']} | "
            f"{row['feedback_auc_mm_s']:.4g} | "
            f"{row['mechanical_auc_mm_s']:.4g} | "
            f"{row['command_input_auc_mm_s']:.4g} | "
            f"{fmt(row['process_force_auc_mm_s'])} | "
            f"{row['feedback_peak_mm']:.4g} | "
            f"{row['mechanical_peak_mm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparisons",
            "",
            "Negative AUC deltas mean lower endpoint displacement than the no-PGD H0 "
            "const_band16 baseline.",
            "",
            "| Row | Feedback AUC delta | Mechanical AUC delta | Command AUC delta | "
            "Process-force AUC delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row_key, delta in summary["comparisons_vs_baseline"].items():
        lines.append(
            "| "
            f"`{row_key}` | {fmt(delta['feedback_auc_mm_s_delta'])} | "
            f"{fmt(delta['mechanical_auc_mm_s_delta'])} | "
            f"{fmt(delta['command_input_auc_mm_s_delta'])} | "
            f"{fmt(delta['process_force_auc_mm_s_delta'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    """Format scalar table values."""

    if value is None:
        return "not available"
    if isinstance(value, int | float):
        return f"{float(value):.4g}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    import json

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
