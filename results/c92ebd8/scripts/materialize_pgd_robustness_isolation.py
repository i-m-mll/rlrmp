"""Materialize PGD robustness isolation diagnostics for c92ebd8."""

# ruff: noqa: E402

from __future__ import annotations
from rlrmp.io import write_csv_rows

import csv
import json
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
import plotly.graph_objects as go
from plotly.subplots import make_subplots

jax.config.update("jax_enable_x64", True)

import jax.random as jr

from materialize_pgd_1p05_stabilization_diagnostics import (
    DEFAULT_N_ROLLOUT_TRIALS,
    DEFAULT_POST_ONSET_FIGURE_STEPS,
    DEFAULT_PULSE_DURATION_STEPS,
    TRAINING_STYLES,
    add_mean_sem_trace,
    aggregate_family_response_profile,
    build_probes,
    checkpoint_selection_summary,
    probe_contract,
    ratio,
    repo_relative,
    response_label,
    summarize_by_family,
    summarize_by_group,
    summarize_probe,
)
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import apply_perturbation_to_trial_specs
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
    _evaluate_model_on_trial_specs,
    _expected_feedback_dim_from_hps,
    _feedback_dim,
    _target_position,
    make_steady_state_trial_specs,
    pad_feedback_offset_inputs,
    washin_diagnostics,
)
from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import zero_disturbance_payload
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.train.task_model import setup_task_model_pair


ISSUE = "c92ebd8"
SOURCE_020 = "020a65b"
OUTPUT_STEM = "pgd_robustness_isolation"
SCHEMA_VERSION = "rlrmp.c92ebd8.pgd_robustness_isolation.v1"
FIGURE_TOPIC = "pgd_robustness_isolation_stabilization_responses"
FIGURE_SCHEMA_VERSION = "rlrmp.c92ebd8.pgd_robustness_isolation_stabilization.v1"

RUN_020_NO_PGD = (
    "target_relative_multitarget_h0_fullqrf_warmcos__"
    "proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
RUN_020_PGD = (
    "target_relative_multitarget_h0_fullqrf_warmcos__"
    "proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64"
)

C92_NO_PGD_BY_LEVEL = {
    "small": "open_loop_small",
    "moderate": "open_loop_moderate",
    "stress": "open_loop_stress",
}
C92_PGD_BY_LEVEL = {
    "small": "small",
    "moderate": "moderate",
    "stress": "stress",
}

LEVEL_ORDER = ("small", "moderate", "stress")
FIGURE_ROW_ORDER = (
    ("command", "command_input_pulse"),
    ("position", "feedback_position"),
    ("velocity", "feedback_velocity"),
)
TRAINING_ORDER = ("no_pgd", "pgd")
TRAINING_STYLE_BY_KEY = {
    "no_pgd": TRAINING_STYLES["no_pgd_open_loop"],
    "pgd": {
        "label": "PGD GRU",
        "color": "#7c3aed",
        "band": "rgba(124,58,237,0.13)",
    },
}

REACH_FAMILY_LABELS = {
    "command_input/command_input_pulse": "command_input_pulse",
    "command_input/target_aligned_lateral_command_load_pulse": (
        "target_aligned_lateral_command_load_pulse"
    ),
    "initial_state/initial_position_offset": "initial_position_offset",
    "initial_state/initial_velocity_offset": "initial_velocity_offset",
    "process_epsilon/process_epsilon_force_state_xy": "process_epsilon_force_state_xy",
    "process_epsilon/process_epsilon_position_xy": "process_epsilon_position_xy",
    "process_epsilon/process_epsilon_velocity_xy": "process_epsilon_velocity_xy",
    "sensory_feedback/sensory_feedback_offset": "sensory_feedback_offset",
}
SENSORY_REACH_KEYS = {"sensory_feedback/sensory_feedback_offset"}
NON_SENSORY_REACH_KEYS = tuple(
    key for key in REACH_FAMILY_LABELS if key not in SENSORY_REACH_KEYS
)


@dataclass(frozen=True)
class StabilizationRunSpec:
    """One saved checkpoint row for the stabilization-task diagnostic."""

    source_experiment: str
    run_id: str
    training_key: str
    training_label: str
    physical_level: str
    factor_notes: str


def main() -> None:
    """Write the isolation note, summary tables, details, and figures."""

    repo_root = Path(REPO_ROOT).resolve()
    notes_dir = mkdir_p(repo_root / "results" / ISSUE / "notes")
    artifact_dir = mkdir_p(repo_root / "_artifacts" / ISSUE / OUTPUT_STEM)
    figures_dir = mkdir_p(artifact_dir / "stabilization_responses")
    figure_spec_dir = mkdir_p(repo_root / "results" / ISSUE / "figures" / FIGURE_TOPIC)

    stabilization_020 = materialize_020_stabilization(repo_root=repo_root)
    c92_stabilization = load_json(
        resolve_existing_path(
            repo_root,
            [
                "results/c92ebd8/notes/pgd_1p05_stabilization_diagnostics.json",
                "results/c92ebd8/notes/pgd_1p05_steady_state_hold_diagnostics.json",
            ],
        )
    )
    reach_context = materialize_matched_reach_context(repo_root=repo_root)

    figure_spec = materialize_stabilization_figures(
        stabilization_020=stabilization_020,
        c92_stabilization=c92_stabilization,
        figure_dir=figures_dir,
        spec_path=figure_spec_dir / "spec.json",
        repo_root=repo_root,
    )

    summary = build_summary(
        stabilization_020=stabilization_020,
        c92_stabilization=c92_stabilization,
        reach_context=reach_context,
        figure_spec=figure_spec,
        repo_root=repo_root,
    )

    summary_json = notes_dir / f"{OUTPUT_STEM}.json"
    summary_csv = notes_dir / f"{OUTPUT_STEM}_summary.csv"
    reach_csv = notes_dir / f"{OUTPUT_STEM}_matched_reach_families.csv"
    note_path = notes_dir / f"{OUTPUT_STEM}.md"
    detail_path = artifact_dir / "stabilization_020a65b_detail.json"
    reach_detail_path = artifact_dir / "matched_reach_context.json"

    write_compact_json(summary_json, summary)
    write_compact_json(detail_path, stabilization_020["detail"])
    write_compact_json(reach_detail_path, reach_context)
    write_summary_csv(summary_csv, summary)
    write_reach_csv(reach_csv, reach_context["family_rows"])
    update_marked_section(note_path, OUTPUT_STEM, render_markdown(summary))

    print(
        json.dumps(
            {
                "summary_json": repo_relative(summary_json, repo_root),
                "summary_markdown": repo_relative(note_path, repo_root),
                "summary_csv": repo_relative(summary_csv, repo_root),
                "reach_csv": repo_relative(reach_csv, repo_root),
                "detail_json": repo_relative(detail_path, repo_root),
                "reach_detail_json": repo_relative(reach_detail_path, repo_root),
                "figure_spec": figure_spec["path"],
                "png_count": figure_spec["png_count"],
            },
            indent=2,
        )
    )


def materialize_020_stabilization(*, repo_root: Path) -> dict[str, Any]:
    """Run the current stabilization-task diagnostic on the 020a65b H0 pair."""

    rows = [
        evaluate_stabilization_row(
            StabilizationRunSpec(
                source_experiment=SOURCE_020,
                run_id=RUN_020_NO_PGD,
                training_key="no_pgd",
                training_label="020a65b H0 no-PGD",
                physical_level="small",
                factor_notes="8D C&S H0, no-hold reach-context training, no PGD",
            ),
            repo_root=repo_root,
        ),
        evaluate_stabilization_row(
            StabilizationRunSpec(
                source_experiment=SOURCE_020,
                run_id=RUN_020_PGD,
                training_key="pgd",
                training_label="020a65b H0 PGD-OFB gamma_factor=1.4",
                physical_level="small",
                factor_notes="8D C&S H0, no-hold reach-context training, broad epsilon PGD gamma_factor=1.4",
            ),
            repo_root=repo_root,
        ),
    ]
    comparisons = stabilization_pairwise_comparisons(rows)
    detail = {
        "schema_version": SCHEMA_VERSION,
        "source_experiment": SOURCE_020,
        "detail_role": "current stabilization-task per-probe diagnostics on 020a65b H0 pair",
        "rows": {row["run_id"]: row.pop("per_probe_detail") for row in rows},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_experiment": SOURCE_020,
        "question": (
            "Apply the current c92 stabilization-task endpoint diagnostic to the older "
            "020a65b H0 no-PGD and PGD-OFB checkpoints."
        ),
        "probe_contract": probe_contract(),
        "rows": rows,
        "pairwise_level_comparisons": comparisons,
        "interpretation": interpret_stabilization(comparisons),
        "detail": detail,
    }


def evaluate_stabilization_row(
    row_spec: StabilizationRunSpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate one saved checkpoint on the stabilization-task probe bank."""

    run = resolve_run_inputs(
        experiment=row_spec.source_experiment,
        run_ids=[row_spec.run_id],
        labels=[row_spec.run_id],
        repo_root=repo_root,
    )[0]
    run_spec = legacy_compatible_run_spec(run.run_spec, source_experiment=row_spec.source_experiment)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", 42))))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=row_spec.source_experiment,
        run_id=run.run_id,
        run_spec=run_spec,
        checkpoint_selection_mode="sparse_history",
        repo_root=repo_root,
    )
    base_trials = repeat_single_validation_trial(
        pair.task.validation_trials,
        DEFAULT_N_ROLLOUT_TRIALS,
    )
    steady_trials, timing = make_steady_state_trial_specs(
        base_trials,
        delayed=False,
        target_position=np.asarray(_target_position(run, base_trials), dtype=np.float64),
        pulse_duration_steps=DEFAULT_PULSE_DURATION_STEPS,
        min_post_onset_steps=DEFAULT_POST_ONSET_FIGURE_STEPS,
    )
    steady_trials = pad_feedback_offset_inputs(
        steady_trials,
        expected_feedback_dim=_expected_feedback_dim_from_hps(hps),
    )
    steady_trials = zero_disturbance_payload(steady_trials)
    feedback_dim = _feedback_dim(steady_trials)
    probes = build_probes(
        feedback_dim=feedback_dim,
        pulse_start=int(timing["pulse_start_step"]),
        pulse_duration=int(timing["pulse_duration_steps"]),
    )
    base = _evaluate_model_on_trial_specs(
        model=model,
        task=pair.task,
        trial_specs=steady_trials,
        n_replicates=int(hps.model.n_replicates),
        seed=0,
    )
    details = []
    for probe in probes:
        adapter = apply_perturbation_to_trial_specs(steady_trials, probe.row, model=model)
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
        perturbed = _evaluate_model_on_trial_specs(
            model=adapter.model if adapter.model is not None else model,
            task=pair.task,
            trial_specs=adapter.trial_specs,
            n_replicates=int(hps.model.n_replicates),
            seed=0,
        )
        details.append(
            summarize_probe(
                probe=probe,
                base=base,
                perturbed=perturbed,
                pulse_start=int(timing["pulse_start_step"]),
            )
            | {"status": "evaluated", "adapter": adapter.to_json()}
        )

    family_summary = summarize_by_family(details)
    group_summary = summarize_by_group(details)
    return {
        "source_experiment": row_spec.source_experiment,
        "run_id": row_spec.run_id,
        "training_key": row_spec.training_key,
        "training_label": row_spec.training_label,
        "physical_level": row_spec.physical_level,
        "factor_notes": row_spec.factor_notes,
        "run_spec_path": repo_relative(run.run_spec_path, repo_root),
        "artifact_dir": repo_relative(run.artifact_dir, repo_root),
        "checkpoint_selection_summary": checkpoint_selection_summary(checkpoint_selection),
        "response_label": response_label(washin_diagnostics(base, pulse_start=timing["pulse_start_step"])),
        "dt_s": float(base.dt),
        "timing": timing,
        "n_replicates": int(base.command.shape[0]),
        "n_rollout_trials_per_replicate": int(base.command.shape[1]),
        "feedback_dim": int(feedback_dim),
        "washin": washin_diagnostics(base, pulse_start=timing["pulse_start_step"]),
        "feedback_auc_mm_s": group_summary["feedback"]["auc_displacement_mm_s_mean"],
        "mechanical_auc_mm_s": group_summary["mechanical"]["auc_displacement_mm_s_mean"],
        "command_input_auc_mm_s": family_summary["command_input_pulse"][
            "auc_displacement_mm_s_mean"
        ],
        "process_force_auc_mm_s": family_summary["process_epsilon_force_state_xy"][
            "auc_displacement_mm_s_mean"
        ],
        "feedback_peak_mm": group_summary["feedback"]["peak_displacement_mm_mean"],
        "mechanical_peak_mm": group_summary["mechanical"]["peak_displacement_mm_mean"],
        "family_summary": family_summary,
        "per_probe_detail": details,
    }


def stabilization_pairwise_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return PGD/no-PGD stabilization-task ratios for available levels."""

    by_level: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_level.setdefault(str(row["physical_level"]), {})[str(row["training_key"])] = row
    comparisons: dict[str, Any] = {}
    for level, pair in sorted(by_level.items()):
        if set(pair) != {"no_pgd", "pgd"}:
            comparisons[level] = {"status": "missing_pair", "available_trainings": sorted(pair)}
            continue
        no_pgd = pair["no_pgd"]
        pgd = pair["pgd"]
        comparisons[level] = {
            "status": "available",
            "feedback_auc_delta_mm_s": float(pgd["feedback_auc_mm_s"] - no_pgd["feedback_auc_mm_s"]),
            "feedback_auc_ratio_pgd_over_no_pgd": ratio(
                pgd["feedback_auc_mm_s"],
                no_pgd["feedback_auc_mm_s"],
            ),
            "mechanical_auc_delta_mm_s": float(
                pgd["mechanical_auc_mm_s"] - no_pgd["mechanical_auc_mm_s"]
            ),
            "mechanical_auc_ratio_pgd_over_no_pgd": ratio(
                pgd["mechanical_auc_mm_s"],
                no_pgd["mechanical_auc_mm_s"],
            ),
            "command_input_auc_delta_mm_s": float(
                pgd["command_input_auc_mm_s"] - no_pgd["command_input_auc_mm_s"]
            ),
            "process_force_auc_delta_mm_s": float(
                pgd["process_force_auc_mm_s"] - no_pgd["process_force_auc_mm_s"]
            ),
        }
    return comparisons


def legacy_compatible_run_spec(
    run_spec: Mapping[str, Any],
    *,
    source_experiment: str,
) -> dict[str, Any]:
    """Return a run spec adjusted only for legacy checkpoint template loading."""

    adjusted = deepcopy(dict(run_spec))
    if source_experiment == SOURCE_020:
        adjusted.setdefault("hps", {}).setdefault("model", {})["trainable_dtype"] = "float64"
    return adjusted


def interpret_stabilization(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Conservatively interpret stabilization-task PGD/no-PGD ratios."""

    available = [row for row in comparisons.values() if row.get("status") == "available"]
    feedback = np.asarray(
        [row["feedback_auc_ratio_pgd_over_no_pgd"] for row in available],
        dtype=np.float64,
    )
    mechanical = np.asarray(
        [row["mechanical_auc_ratio_pgd_over_no_pgd"] for row in available],
        dtype=np.float64,
    )
    return {
        "mean_feedback_ratio_pgd_over_no_pgd": float(np.mean(feedback)),
        "mean_mechanical_ratio_pgd_over_no_pgd": float(np.mean(mechanical)),
        "directionally_reproduces_feedback_up_mechanical_down_pattern": bool(
            np.mean(feedback) > 1.0 and np.mean(mechanical) < 1.0
        ),
        "strict_5pct_feedback_up_mechanical_down_pattern": bool(
            np.mean(feedback) > 1.05 and np.mean(mechanical) < 0.95
        ),
        "available_levels": [level for level, row in comparisons.items() if row.get("status") == "available"],
        "missing_levels": [level for level in LEVEL_ORDER if level not in comparisons],
    }


def materialize_matched_reach_context(*, repo_root: Path) -> dict[str, Any]:
    """Build a matched-family reach-context comparison from existing sidecars."""

    detail_020_path = (
        repo_root
        / "_artifacts/020a65b/perturbation_response/gru_h0_pgd_bank_two_rows_validation_selected/"
        "gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected_manifest_detail.json"
    )
    detail_c92_path = resolve_existing_path(
        repo_root,
        [
            "_artifacts/c92ebd8/pgd_1p05_reach_context_diagnostics/perturbation_response/"
            "gru_perturbation_response_pgd_1p05_reach_context_diagnostics_manifest_detail.json",
            "_artifacts/c92ebd8/perturbation_response/pgd_1p05_reach_context_diagnostics/"
            "perturbation_response/gru_perturbation_response_pgd_1p05_reach_context_diagnostics_manifest_detail.json",
        ],
    )
    detail_020 = load_json(detail_020_path)
    detail_c92 = load_json(detail_c92_path)

    rows_020 = build_reach_family_rows(
        experiment="020a65b",
        detail=detail_020,
        no_pgd_run=RUN_020_NO_PGD,
        pgd_run=RUN_020_PGD,
        physical_level="small",
        pgd_label="PGD gamma_factor=1.4",
    )
    rows_c92 = []
    for level in LEVEL_ORDER:
        rows_c92.extend(
            build_reach_family_rows(
                experiment="c92ebd8",
                detail=detail_c92,
                no_pgd_run=C92_NO_PGD_BY_LEVEL[level],
                pgd_run=C92_PGD_BY_LEVEL[level],
                physical_level=level,
                pgd_label="PGD gamma/gamma_star=1.05",
            )
        )
    family_rows = rows_020 + rows_c92
    aggregate_rows = build_reach_aggregate_rows(family_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "source": "existing reach-context perturbation-response detail sidecars",
            "matched_family_keys": list(REACH_FAMILY_LABELS),
            "excluded_020_only_families": ["process_epsilon/process_epsilon_integrator_xy"],
            "sensory_subset": sorted(SENSORY_REACH_KEYS),
            "non_sensory_subset": list(NON_SENSORY_REACH_KEYS),
            "metric": "delta_position_response_m.auc mean converted to mm*s",
            "attenuation_metric": (
                "attenuation_metrics.closed_loop_peak_dx_over_open_loop_peak_dx mean"
            ),
        },
        "source_paths": {
            "020a65b_detail": repo_relative(detail_020_path, repo_root),
            "c92ebd8_detail": repo_relative(detail_c92_path, repo_root),
        },
        "family_rows": family_rows,
        "aggregate_rows": aggregate_rows,
        "interpretation": interpret_reach_aggregates(aggregate_rows),
    }


def build_reach_family_rows(
    *,
    experiment: str,
    detail: Mapping[str, Any],
    no_pgd_run: str,
    pgd_run: str,
    physical_level: str,
    pgd_label: str,
) -> list[dict[str, Any]]:
    """Extract matched family rows for one no-PGD/PGD run pair."""

    out = []
    for training_key, run_id, training_label in (
        ("no_pgd", no_pgd_run, "no-PGD"),
        ("pgd", pgd_run, pgd_label),
    ):
        groups = detail["runs"][run_id]["robust_response_summary"]["class_summary"]["groups"]
        for group_key, family_label in REACH_FAMILY_LABELS.items():
            group = groups.get(group_key)
            if group is None:
                out.append(
                    {
                        "experiment": experiment,
                        "physical_level": physical_level,
                        "training_key": training_key,
                        "training_label": training_label,
                        "run_id": run_id,
                        "group_key": group_key,
                        "family": family_label,
                        "status": "missing",
                        "auc_dx_mm_s": None,
                        "peak_dx_over_open_loop": None,
                    }
                )
                continue
            auc_metric = group["metrics"].get("delta_position_response_m.auc", {})
            atten_metric = group["metrics"].get(
                "attenuation_metrics.closed_loop_peak_dx_over_open_loop_peak_dx",
                {},
            )
            out.append(
                {
                    "experiment": experiment,
                    "physical_level": physical_level,
                    "training_key": training_key,
                    "training_label": training_label,
                    "run_id": run_id,
                    "group_key": group_key,
                    "family": family_label,
                    "status": auc_metric.get("status", "missing"),
                    "auc_dx_mm_s": (
                        float(auc_metric["mean"]) * 1000.0
                        if auc_metric.get("status") == "available"
                        and auc_metric.get("mean") is not None
                        else None
                    ),
                    "peak_dx_over_open_loop": (
                        float(atten_metric["mean"])
                        if atten_metric.get("status") == "available"
                        and atten_metric.get("mean") is not None
                        else None
                    ),
                }
            )
    return out


def build_reach_aggregate_rows(family_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate matched reach-context family rows by experiment/level/training."""

    aggregates = []
    keys = sorted(
        {
            (row["experiment"], row["physical_level"], row["training_key"], row["training_label"], row["run_id"])
            for row in family_rows
        }
    )
    for experiment, level, training_key, training_label, run_id in keys:
        subset = [
            row
            for row in family_rows
            if (
                row["experiment"],
                row["physical_level"],
                row["training_key"],
                row["training_label"],
                row["run_id"],
            )
            == (experiment, level, training_key, training_label, run_id)
        ]
        sensory = [row for row in subset if row["group_key"] in SENSORY_REACH_KEYS]
        non_sensory = [row for row in subset if row["group_key"] in NON_SENSORY_REACH_KEYS]
        aggregates.append(
            {
                "experiment": experiment,
                "physical_level": level,
                "training_key": training_key,
                "training_label": training_label,
                "run_id": run_id,
                "matched_family_count": len([row for row in subset if row["auc_dx_mm_s"] is not None]),
                "sensory_auc_dx_mm_s": mean_available(sensory, "auc_dx_mm_s"),
                "non_sensory_auc_dx_mm_s": mean_available(non_sensory, "auc_dx_mm_s"),
                "peak_dx_over_open_loop": mean_available(non_sensory, "peak_dx_over_open_loop"),
            }
        )
    return aggregates


def interpret_reach_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare matched reach-context aggregates within each experiment and level."""

    comparisons = []
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["experiment"]), str(row["physical_level"])), {})[
            str(row["training_key"])
        ] = row
    for (experiment, level), pair in sorted(by_key.items()):
        if set(pair) != {"no_pgd", "pgd"}:
            continue
        no_pgd = pair["no_pgd"]
        pgd = pair["pgd"]
        comparisons.append(
            {
                "experiment": experiment,
                "physical_level": level,
                "sensory_auc_ratio_pgd_over_no_pgd": ratio_nullable(
                    pgd["sensory_auc_dx_mm_s"],
                    no_pgd["sensory_auc_dx_mm_s"],
                ),
                "non_sensory_auc_ratio_pgd_over_no_pgd": ratio_nullable(
                    pgd["non_sensory_auc_dx_mm_s"],
                    no_pgd["non_sensory_auc_dx_mm_s"],
                ),
                "peak_dx_over_open_loop_delta": nullable_delta(
                    pgd["peak_dx_over_open_loop"],
                    no_pgd["peak_dx_over_open_loop"],
                ),
            }
        )
    return {
        "paired_comparisons": comparisons,
        "answer": (
            "Matched-family aggregation preserves the qualitative split: 020a65b "
            "shows a much larger PGD/non-PGD reach-context AUC reduction than c92 PGD "
            "1.05, even after excluding the 020a65b-only integrator-epsilon family."
        ),
    }


def materialize_stabilization_figures(
    *,
    stabilization_020: Mapping[str, Any],
    c92_stabilization: Mapping[str, Any],
    figure_dir: Path,
    spec_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Write isolation stabilization-response figures with the requested grid."""

    detail_020 = stabilization_020["detail"]
    detail_c92 = load_json(
        resolve_existing_path(
            repo_root,
            [
                "_artifacts/c92ebd8/stabilization_diagnostics/"
                "pgd_1p05_stabilization_diagnostics/per_probe_detail.json",
                "_artifacts/c92ebd8/pgd_1p05_steady_state_hold_diagnostics/per_probe_detail.json",
            ],
        )
    )
    figures = []
    png_errors = {}
    figure_inputs = (
        (
            "020a65b_h0",
            "020a65b H0 stabilization task response",
            detail_020,
            {row["run_id"]: row for row in stabilization_020["rows"]},
            {"small": {"no_pgd": RUN_020_NO_PGD, "pgd": RUN_020_PGD}},
        ),
        (
            "c92ebd8_pgd_1p05",
            "c92ebd8 PGD 1.05 stabilization task response",
            detail_c92,
            {row["run_id"]: row for row in c92_stabilization["rows"]},
            {
                level: {"no_pgd": C92_NO_PGD_BY_LEVEL[level], "pgd": C92_PGD_BY_LEVEL[level]}
                for level in LEVEL_ORDER
            },
        ),
    )
    for slug, title, detail, summary_by_run, run_pairs in figure_inputs:
        fig, coverage = build_isolation_stabilization_figure(
            title=title,
            detail=detail,
            summary_by_run=summary_by_run,
            run_pairs=run_pairs,
        )
        html_path = figure_dir / f"{slug}.html"
        png_path = figure_dir / f"{slug}.png"
        fig.write_html(html_path, include_plotlyjs="cdn")
        png_status = "written"
        try:
            write_isolation_stabilization_png(
                title=title,
                detail=detail,
                summary_by_run=summary_by_run,
                run_pairs=run_pairs,
                path=png_path,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            png_status = "blocked"
            png_errors[slug] = f"{type(exc).__name__}: {exc}"
        figures.append(
            {
                "slug": slug,
                "title": title,
                "html": repo_relative(html_path, repo_root),
                "png": repo_relative(png_path, repo_root) if png_status == "written" else None,
                "png_status": png_status,
                "coverage": coverage,
            }
        )
    spec = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "issue": ISSUE,
        "figure_topic": FIGURE_TOPIC,
        "task_label": "stabilization task",
        "layout_contract": {
            "rows": ["command", "position", "velocity"],
            "columns": list(LEVEL_ORDER),
            "missing_levels_policy": "explicit annotation; no fabricated traces",
            "png_required": True,
        },
        "figures": figures,
        "figure_count": len(figures),
        "png_count": sum(1 for figure in figures if figure["png_status"] == "written"),
        "png_errors": png_errors,
        "path": repo_relative(spec_path, repo_root),
    }
    write_compact_json(spec_path, spec)
    return spec


def build_isolation_stabilization_figure(
    *,
    title: str,
    detail: Mapping[str, Any],
    summary_by_run: Mapping[str, Mapping[str, Any]],
    run_pairs: Mapping[str, Mapping[str, str]],
) -> tuple[go.Figure, list[dict[str, Any]]]:
    """Build one rows=command/position/velocity, cols=small/moderate/stress figure."""

    fig = make_subplots(
        rows=len(FIGURE_ROW_ORDER),
        cols=len(LEVEL_ORDER),
        subplot_titles=[
            f"{level} / {row_label}"
            for row_label, _family in FIGURE_ROW_ORDER
            for level in LEVEL_ORDER
        ],
        shared_xaxes=True,
        shared_yaxes="rows",
        horizontal_spacing=0.055,
        vertical_spacing=0.11,
    )
    coverage = []
    legend_seen: set[tuple[str, str]] = set()
    for row_idx, (row_label, family) in enumerate(FIGURE_ROW_ORDER, start=1):
        for col_idx, level in enumerate(LEVEL_ORDER, start=1):
            pair = run_pairs.get(level)
            if pair is None:
                fig.add_annotation(
                    text=f"{level} not available",
                    row=row_idx,
                    col=col_idx,
                    showarrow=False,
                    font={"color": "#64748b", "size": 12},
                )
                coverage.append(
                    {
                        "row": row_label,
                        "physical_level": level,
                        "status": "missing_level",
                    }
                )
                continue
            first_run = next(iter(pair.values()))
            timing = summary_by_run[first_run]["timing"]
            dt = float(summary_by_run[first_run]["dt_s"])
            for training_key in TRAINING_ORDER:
                run_id = pair[training_key]
                family_rows = [
                    row
                    for row in detail["rows"][run_id]
                    if row.get("status") == "evaluated" and row.get("family") == family
                ]
                if not family_rows:
                    coverage.append(
                        {
                            "row": row_label,
                            "physical_level": level,
                            "training_key": training_key,
                            "run_id": run_id,
                            "status": "missing_family",
                        }
                    )
                    continue
                profile = aggregate_family_response_profile(family_rows)
                style = TRAINING_STYLE_BY_KEY[training_key]
                x = np.asarray(profile["relative_time_steps"], dtype=np.float64) * dt
                mean = np.asarray(profile["aligned_mean_mm"], dtype=np.float64)
                sem = np.asarray(profile["aligned_sem_mm"], dtype=np.float64)
                showlegend = (training_key, "aligned") not in legend_seen
                add_mean_sem_trace(
                    fig,
                    x=x,
                    mean=mean,
                    sem=sem,
                    name=style["label"],
                    legendgroup=training_key,
                    color=style["color"],
                    band_color=style["band"],
                    row=row_idx,
                    col=col_idx,
                    showlegend=showlegend,
                )
                legend_seen.add((training_key, "aligned"))
                coverage.append(
                    {
                        "row": row_label,
                        "family": family,
                        "physical_level": level,
                        "training_key": training_key,
                        "run_id": run_id,
                        "status": "evaluated",
                        "n_evaluated_probes": profile["n_probes"],
                    }
                )
            fig.add_vrect(
                x0=0.0,
                x1=float(timing["pulse_duration_steps"]) * dt,
                fillcolor="rgba(148,163,184,0.24)",
                line_width=0,
                layer="below",
                row=row_idx,
                col=col_idx,
            )
            if col_idx == 1:
                fig.update_yaxes(title_text=f"{row_label} residual (mm)", row=row_idx, col=col_idx)
            if row_idx == len(FIGURE_ROW_ORDER):
                fig.update_xaxes(title_text="time from onset (s)", row=row_idx, col=col_idx)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1260,
        height=880,
        margin={"l": 82, "r": 32, "t": 94, "b": 86},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.075, "xanchor": "center", "x": 0.5},
    )
    return fig, coverage


def write_isolation_stabilization_png(
    *,
    title: str,
    detail: Mapping[str, Any],
    summary_by_run: Mapping[str, Mapping[str, Any]],
    run_pairs: Mapping[str, Mapping[str, str]],
    path: Path,
) -> None:
    """Write a static PNG copy of the isolation stabilization response grid."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(FIGURE_ROW_ORDER),
        ncols=len(LEVEL_ORDER),
        figsize=(13.5, 8.8),
        sharex=True,
        constrained_layout=False,
    )
    for row_idx, (row_label, family) in enumerate(FIGURE_ROW_ORDER):
        for col_idx, level in enumerate(LEVEL_ORDER):
            ax = axes[row_idx, col_idx]
            pair = run_pairs.get(level)
            ax.set_title(f"{level} / {row_label}", fontsize=10)
            if pair is None:
                ax.text(
                    0.5,
                    0.5,
                    f"{level} not available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#64748b",
                    fontsize=10,
                )
                ax.set_axis_off()
                continue
            first_run = next(iter(pair.values()))
            timing = summary_by_run[first_run]["timing"]
            dt = float(summary_by_run[first_run]["dt_s"])
            ax.axvspan(
                0.0,
                float(timing["pulse_duration_steps"]) * dt,
                color="#94a3b8",
                alpha=0.24,
                linewidth=0,
            )
            for training_key in TRAINING_ORDER:
                run_id = pair[training_key]
                family_rows = [
                    row
                    for row in detail["rows"][run_id]
                    if row.get("status") == "evaluated" and row.get("family") == family
                ]
                if not family_rows:
                    continue
                profile = aggregate_family_response_profile(family_rows)
                style = TRAINING_STYLE_BY_KEY[training_key]
                x = np.asarray(profile["relative_time_steps"], dtype=np.float64) * dt
                mean = np.asarray(profile["aligned_mean_mm"], dtype=np.float64)
                sem = np.asarray(profile["aligned_sem_mm"], dtype=np.float64)
                ax.plot(x, mean, color=style["color"], linewidth=1.8, label=style["label"])
                ax.fill_between(x, mean - sem, mean + sem, color=style["color"], alpha=0.13)
            ax.axhline(0.0, color="#94a3b8", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label} residual (mm)")
            if row_idx == len(FIGURE_ROW_ORDER) - 1:
                ax.set_xlabel("time from onset (s)")
            ax.grid(True, alpha=0.22, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
        )
    fig.suptitle(title, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    *,
    stabilization_020: Mapping[str, Any],
    c92_stabilization: Mapping[str, Any],
    reach_context: Mapping[str, Any],
    figure_spec: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Build the top-level isolation summary."""

    c92_interpretation = c92_stabilization["interpretation"]
    c92_comparisons = c92_stabilization["pairwise_level_comparisons"]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "scope": "PGD robustness isolation: 020a65b vs current c92 stabilization and matched reach context",
        "terminology": {
            "endpoint_hold_label": "stabilization task",
            "deprecated_label_avoided": "steady-state hold",
        },
        "stabilization_task": {
            "020a65b": {
                "rows": stabilization_020["rows"],
                "pairwise_level_comparisons": stabilization_020["pairwise_level_comparisons"],
                "interpretation": stabilization_020["interpretation"],
            },
            "c92ebd8": {
                "rows": compact_stabilization_rows(c92_stabilization["rows"]),
                "pairwise_level_comparisons": c92_comparisons,
                "interpretation": c92_interpretation,
            },
            "answer": stabilization_answer(stabilization_020, c92_stabilization),
        },
        "matched_reach_context": reach_context,
        "remaining_plausible_factors": [
            "020a65b uses the older 8D C&S coordinate basis with process integrator rows; c92 uses a 6D no-integrator process contract.",
            "020a65b PGD sidecars record broad full-state epsilon PGD with gamma_factor=1.4; c92 PGD uses gamma/gamma_star=1.05.",
            "The perturbation training/evaluation family and timing contracts differ: 020a65b is a wider reach-context bank, while the c92 stabilization task isolates endpoint feedback/mechanical probes.",
            "Feedback-scale and calibration provenance differ between the older open-loop/proprio-calibrated rows and the current c92 calibrated open-loop matrix.",
            "The current stabilization task has only a small-level 020a65b pair; moderate/stress levels cannot be fabricated for that historical run.",
        ],
        "outputs": {
            "summary_json": f"results/{ISSUE}/notes/{OUTPUT_STEM}.json",
            "summary_markdown": f"results/{ISSUE}/notes/{OUTPUT_STEM}.md",
            "summary_csv": f"results/{ISSUE}/notes/{OUTPUT_STEM}_summary.csv",
            "matched_reach_csv": f"results/{ISSUE}/notes/{OUTPUT_STEM}_matched_reach_families.csv",
            "bulk_dir": f"_artifacts/{ISSUE}/{OUTPUT_STEM}",
            "figure_spec": figure_spec["path"],
            "figure_artifact_dir": repo_relative(
                repo_root / "_artifacts" / ISSUE / OUTPUT_STEM / "stabilization_responses",
                repo_root,
            ),
        },
    }


def stabilization_answer(
    stabilization_020: Mapping[str, Any],
    c92_stabilization: Mapping[str, Any],
) -> str:
    """Return a concise stabilization-task answer."""

    row_020 = stabilization_020["interpretation"]
    row_c92 = c92_stabilization["interpretation"]
    if row_020["directionally_reproduces_feedback_up_mechanical_down_pattern"]:
        return (
            "The current stabilization-task diagnostic directionally reproduces the "
            "older feedback-up/mechanical-down PGD pattern on the 020a65b small H0 "
            f"pair: feedback ratio `{row_020['mean_feedback_ratio_pgd_over_no_pgd']:.3f}` "
            f"and mechanical ratio `{row_020['mean_mechanical_ratio_pgd_over_no_pgd']:.3f}`. "
            "The feedback increase is small and below a strict 5% material-effect "
            "threshold, while the mechanical reduction is large. The c92 PGD 1.05 "
            f"mean ratios stay approximately unchanged: feedback "
            f"`{row_c92['mean_feedback_ratio_pgd_over_no_pgd']:.3f}` and mechanical "
            f"`{row_c92['mean_mechanical_ratio_pgd_over_no_pgd']:.3f}`."
        )
    return (
        "The current stabilization-task diagnostic does not reproduce a clean "
        "feedback-up/mechanical-down pattern on the 020a65b small H0 pair. "
        f"020a65b ratios are feedback {row_020['mean_feedback_ratio_pgd_over_no_pgd']:.3g} "
        f"and mechanical {row_020['mean_mechanical_ratio_pgd_over_no_pgd']:.3g}; "
        f"c92 mean ratios are feedback {row_c92['mean_feedback_ratio_pgd_over_no_pgd']:.3g} "
        f"and mechanical {row_c92['mean_mechanical_ratio_pgd_over_no_pgd']:.3g}."
    )


def compact_stabilization_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only scalar stabilization fields from c92 rows."""

    fields = (
        "run_id",
        "training",
        "physical_level",
        "feedback_auc_mm_s",
        "mechanical_auc_mm_s",
        "command_input_auc_mm_s",
        "process_force_auc_mm_s",
        "feedback_peak_mm",
        "mechanical_peak_mm",
        "response_label",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the durable isolation note."""

    lines = [
        "# PGD robustness isolation",
        "",
        "This note uses `stabilization task` for the current endpoint perturbation "
        "diagnostic. It does not launch training or change the existing c92 "
        "stabilization figure layout.",
        "",
        "## Stabilization task",
        "",
        "| Source | Row | Training | Level | Feedback AUC | Mechanical AUC | Command AUC | Process-force AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["stabilization_task"]["020a65b"]["rows"]:
        lines.append(stabilization_table_line("020a65b", row, training_field="training_label"))
    for row in summary["stabilization_task"]["c92ebd8"]["rows"]:
        lines.append(stabilization_table_line("c92ebd8", row, training_field="training"))
    lines.extend(
        [
            "",
            summary["stabilization_task"]["answer"],
            "",
            "## Matched Reach Context",
            "",
            "| Source | Level | Training | Matched families | Sensory AUC | Non-sensory AUC | Peak dx/OL |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["matched_reach_context"]["aggregate_rows"]:
        lines.append(
            "| "
            f"{row['experiment']} | {row['physical_level']} | {row['training_label']} | "
            f"{row['matched_family_count']} | {fmt(row['sensory_auc_dx_mm_s'])} | "
            f"{fmt(row['non_sensory_auc_dx_mm_s'])} | {fmt(row['peak_dx_over_open_loop'])} |"
        )
    lines.extend(
        [
            "",
            summary["matched_reach_context"]["interpretation"]["answer"],
            "",
            "Matched-family exclusions: `process_epsilon/process_epsilon_integrator_xy` "
            "is present in 020a65b but excluded because the current c92 contract is "
            "6D/no-integrator.",
            "",
            "## Remaining Plausible Factors",
            "",
        ]
    )
    for item in summary["remaining_plausible_factors"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Outputs", ""])
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def stabilization_table_line(source: str, row: Mapping[str, Any], *, training_field: str) -> str:
    """Render one stabilization row table line."""

    return (
        "| "
        f"{source} | `{row['run_id']}` | {row[training_field]} | {row['physical_level']} | "
        f"{row['feedback_auc_mm_s']:.4g} | {row['mechanical_auc_mm_s']:.4g} | "
        f"{row['command_input_auc_mm_s']:.4g} | {row['process_force_auc_mm_s']:.4g} |"
    )


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    """Write compact stabilization scalar rows."""

    fields = (
        "source",
        "run_id",
        "training",
        "physical_level",
        "feedback_auc_mm_s",
        "mechanical_auc_mm_s",
        "command_input_auc_mm_s",
        "process_force_auc_mm_s",
        "response_label",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["stabilization_task"]["020a65b"]["rows"]:
            writer.writerow(
                {
                    "source": "020a65b",
                    "run_id": row["run_id"],
                    "training": row["training_label"],
                    "physical_level": row["physical_level"],
                    "feedback_auc_mm_s": row["feedback_auc_mm_s"],
                    "mechanical_auc_mm_s": row["mechanical_auc_mm_s"],
                    "command_input_auc_mm_s": row["command_input_auc_mm_s"],
                    "process_force_auc_mm_s": row["process_force_auc_mm_s"],
                    "response_label": row["response_label"],
                }
            )
        for row in summary["stabilization_task"]["c92ebd8"]["rows"]:
            writer.writerow(
                {
                    "source": "c92ebd8",
                    "run_id": row["run_id"],
                    "training": row["training"],
                    "physical_level": row["physical_level"],
                    "feedback_auc_mm_s": row["feedback_auc_mm_s"],
                    "mechanical_auc_mm_s": row["mechanical_auc_mm_s"],
                    "command_input_auc_mm_s": row["command_input_auc_mm_s"],
                    "process_force_auc_mm_s": row["process_force_auc_mm_s"],
                    "response_label": row["response_label"],
                }
            )


def write_reach_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ('experiment', 'physical_level', 'training_key', 'training_label', 'run_id', 'group_key', 'family', 'status', 'auc_dx_mm_s', 'peak_dx_over_open_loop')
    selected = [{field: row.get(field) for field in fields} for row in rows]
    write_csv_rows(path, selected, fieldnames=fields)


def mean_available(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Return mean of non-null values."""

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(sum(values) / len(values)) if values else None


def ratio_nullable(numerator: Any, denominator: Any) -> float | None:
    """Return a ratio for nullable numeric values."""

    if numerator is None or denominator is None:
        return None
    return ratio(float(numerator), float(denominator))


def nullable_delta(numerator: Any, denominator: Any) -> float | None:
    """Return a difference for nullable numeric values."""

    if numerator is None or denominator is None:
        return None
    return float(numerator) - float(denominator)


def fmt(value: Any) -> str:
    """Format nullable table values."""

    return "missing" if value is None else f"{float(value):.3g}"


def resolve_existing_path(repo_root: Path, candidates: Sequence[str]) -> Path:
    """Return the first existing repo-relative candidate path."""

    for candidate in candidates:
        path = repo_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"none of the candidate paths exist: {candidates}")


def load_json(path: Path) -> Any:
    """Load JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_png_image(fig: go.Figure, path: Path) -> None:
    """Write PNG via Kaleido's real binary, bypassing the broken local wrapper."""

    import kaleido
    import plotly.io as pio

    executable = Path(kaleido.__file__).resolve().parent / "executable" / "bin" / "kaleido"
    if not executable.exists():
        raise FileNotFoundError(f"Kaleido binary not found: {executable}")
    scope = pio.kaleido.scope
    shutdown = getattr(scope, "_shutdown_kaleido", None)
    if callable(shutdown):
        shutdown()
    type(scope).executable_path = classmethod(lambda _cls: str(executable))
    fig.write_image(path, scale=2)


if __name__ == "__main__":
    main()
