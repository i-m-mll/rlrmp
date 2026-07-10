"""Materialize c92 PGD 1.05 stabilization-task perturbation diagnostics."""

from __future__ import annotations
from rlrmp.viz.traces import add_band_trace as canonical_add_band_trace
from rlrmp.io import write_csv_rows

import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
    DEFAULT_FORCE_FILTER_SCALE,
    DEFAULT_N_ROLLOUT_TRIALS,
    DEFAULT_POSITION_SCALE_M,
    DEFAULT_PRE_ONSET_FIGURE_STEPS,
    DEFAULT_POST_ONSET_FIGURE_STEPS,
    DEFAULT_PULSE_DURATION_STEPS,
    DEFAULT_VELOCITY_SCALE_M_S,
    default_feedback_perturbations,
)
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.eval.robustness_diagnostics import (
    evaluate_stabilization_row as canonical_evaluate_stabilization_row,
)


ISSUE = "c92ebd8"
MARKER = "pgd_1p05_stabilization_diagnostics"
OUTPUT_STEM = "pgd_1p05_stabilization_diagnostics"
SCHEMA_VERSION = "rlrmp.c92ebd8.pgd_1p05_stabilization_diagnostics.v2"
FIGURE_TOPIC = "pgd_1p05_stabilization_perturbation_responses"
FIGURE_MARKER = "pgd_1p05_stabilization_perturbation_responses"
FIGURE_SCHEMA_VERSION = "rlrmp.c92ebd8.pgd_1p05_stabilization_responses.v2"
LEVEL_ORDER = ("small", "moderate", "stress")
PERTURBATION_FAMILY_ORDER = (
    "command_input_pulse",
    "feedback_position",
    "feedback_velocity",
    "feedback_force_filter",
    "process_epsilon_force_state_xy",
)
RESPONSE_VARIABLE_ORDER = ("command", "position", "velocity")
FAMILY_LABELS = {
    "feedback_position": "Position feedback offset",
    "feedback_velocity": "Velocity feedback offset",
    "feedback_force_filter": "Force/filter feedback offset",
    "command_input_pulse": "Command-input pulse",
    "process_epsilon_force_state_xy": "Process-epsilon force-state pulse",
}
RESPONSE_VARIABLE_SPECS = {
    "command": {
        "label": "Command",
        "aligned_key": "aligned_command_window_profile",
        "orthogonal_key": "orthogonal_command_window_profile",
        "scale": 1.0,
        "unit": "command units",
        "axis_title": "command residual",
    },
    "position": {
        "label": "Position",
        "aligned_key": "aligned_position_window_profile_m",
        "orthogonal_key": "orthogonal_position_window_profile_m",
        "scale": 1000.0,
        "unit": "mm",
        "axis_title": "position residual (mm)",
    },
    "velocity": {
        "label": "Velocity",
        "aligned_key": "aligned_velocity_window_profile_m_s",
        "orthogonal_key": "orthogonal_velocity_window_profile_m_s",
        "scale": 1000.0,
        "unit": "mm/s",
        "axis_title": "velocity residual (mm/s)",
    },
}
TRAINING_STYLES = {
    "no_pgd_open_loop": {
        "label": "No-PGD open-loop calibrated GRU",
        "color": "#2563eb",
        "band": "rgba(37,99,235,0.13)",
    },
    "pgd_1p05": {
        "label": "PGD 1.05 GRU",
        "color": "#7c3aed",
        "band": "rgba(124,58,237,0.13)",
    },
}


@dataclass(frozen=True)
class RowSpec:
    """One trained row to evaluate."""

    run_id: str
    training: str
    physical_level: str


@dataclass(frozen=True)
class ProbeSpec:
    """One steady-state diagnostic perturbation probe."""

    perturbation_id: str
    group: str
    family: str
    row: Mapping[str, Any]
    direction: tuple[float, float]
    amplitude: float
    units: str


ROWS: tuple[RowSpec, ...] = (
    RowSpec("open_loop_small", "no_pgd_open_loop", "small"),
    RowSpec("open_loop_moderate", "no_pgd_open_loop", "moderate"),
    RowSpec("open_loop_stress", "no_pgd_open_loop", "stress"),
    RowSpec("small", "pgd_1p05", "small"),
    RowSpec("moderate", "pgd_1p05", "moderate"),
    RowSpec("stress", "pgd_1p05", "stress"),
)


def main() -> None:
    """Run the diagnostic and write compact tracked outputs."""

    repo_root = Path(REPO_ROOT).resolve()
    summary = materialize(repo_root=repo_root)
    notes_dir = mkdir_p(repo_root / "results" / ISSUE / "notes")
    detail_dir = mkdir_p(
        repo_root / "_artifacts" / ISSUE / "stabilization_diagnostics" / OUTPUT_STEM
    )
    json_path = notes_dir / f"{OUTPUT_STEM}.json"
    csv_path = notes_dir / f"{OUTPUT_STEM}.csv"
    md_path = notes_dir / f"{OUTPUT_STEM}.md"
    detail_path = detail_dir / "per_probe_detail.json"

    detail_rows = {row["run_id"]: row.pop("per_probe_detail") for row in summary["rows"]}
    detail = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "detail_role": "per-probe scalar and trajectory diagnostics",
        "rows": detail_rows,
    }
    summary["outputs"] = {
        "summary_json": repo_relative(json_path, repo_root),
        "summary_csv": repo_relative(csv_path, repo_root),
        "summary_markdown": repo_relative(md_path, repo_root),
        "detail_json": repo_relative(detail_path, repo_root),
    }
    figure_outputs = materialize_stabilization_response_figures(
        detail=detail,
        summary=summary,
        repo_root=repo_root,
    )
    summary["outputs"].update(figure_outputs["outputs"])
    detail["outputs"] = summary["outputs"]
    write_compact_json(json_path, summary)
    write_compact_json(detail_path, detail)
    write_csv(csv_path, summary["rows"])
    update_marked_section(md_path, MARKER, render_markdown(summary))


def materialize(*, repo_root: Path) -> dict[str, Any]:
    """Evaluate all requested c92 rows on stabilization-task endpoint probes."""

    rows = []
    for row_spec in ROWS:
        rows.append(evaluate_row(row_spec, repo_root=repo_root))
    comparisons = pairwise_level_comparisons(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "source_experiment": ISSUE,
        "question": (
            "On a stabilization task anchored at the target endpoint, compare PGD 1.05 "
            "rows against their "
            "no-PGD open-loop calibrated counterparts for feedback and non-feedback "
            "mechanical perturbation displacement AUC."
        ),
        "probe_contract": probe_contract(),
        "rows": rows,
        "pairwise_level_comparisons": comparisons,
        "interpretation": interpret(comparisons),
    }


def evaluate_row(row_spec: RowSpec, *, repo_root: Path) -> dict[str, Any]:
    """Evaluate one trained row."""

    return canonical_evaluate_stabilization_row(
        row_spec,
        repo_root=repo_root,
        hooks=globals(),
        source_experiment=ISSUE,
        row_metadata=lambda row: {
            "run_id": row.run_id,
            "training": row.training,
            "physical_level": row.physical_level,
        },
    )


def build_probes(*, feedback_dim: int, pulse_start: int, pulse_duration: int) -> tuple[ProbeSpec, ...]:
    """Return feedback and mechanical stabilization-task endpoint probes."""

    probes: list[ProbeSpec] = []
    for perturbation in default_feedback_perturbations(
        feedback_dim=feedback_dim,
        position_scale_m=DEFAULT_POSITION_SCALE_M,
        velocity_scale_m_s=DEFAULT_VELOCITY_SCALE_M_S,
        force_filter_scale=DEFAULT_FORCE_FILTER_SCALE,
    ):
        row = perturbation.to_bank_row(
            feedback_dim=feedback_dim,
            pulse_start=pulse_start,
            pulse_duration=pulse_duration,
        )
        probes.append(
            ProbeSpec(
                perturbation_id=perturbation.perturbation_id,
                group="feedback",
                family=f"feedback_{perturbation.family}",
                row=row,
                direction=perturbation.direction,
                amplitude=perturbation.amplitude,
                units=perturbation.units,
            )
        )
    for family, channel, amplitude, units, axes in (
        ("command_input_pulse", "command_input", 1.0, "N", ("x", "y")),
        ("process_epsilon_force_state_xy", "process_epsilon", 0.01, "epsilon", ("x", "y")),
    ):
        for axis in axes:
            for sign in (-1, 1):
                direction = (float(sign), 0.0) if axis == "x" else (0.0, float(sign))
                row = mechanical_row(
                    family=family,
                    channel=channel,
                    axis=axis,
                    sign=sign,
                    amplitude=amplitude,
                    units=units,
                    pulse_start=pulse_start,
                    pulse_duration=pulse_duration,
                )
                probes.append(
                    ProbeSpec(
                        perturbation_id=str(row["perturbation_id"]),
                        group="mechanical",
                        family=family,
                        row=row,
                        direction=direction,
                        amplitude=amplitude,
                        units=units,
                    )
                )
    return tuple(probes)


def mechanical_row(
    *,
    family: str,
    channel: str,
    axis: str,
    sign: int,
    amplitude: float,
    units: str,
    pulse_start: int,
    pulse_duration: int,
) -> dict[str, Any]:
    """Return one steady-state mechanical perturbation row."""

    timing = {
        "epoch": "steady_state_endpoint",
        "start_time_index": int(pulse_start),
        "duration_steps": int(pulse_duration),
        "timing_bin": "steady_state_endpoint",
        "timing_bin_role": "stabilization_task_endpoint_mechanical_probe",
    }
    if family == "process_epsilon_force_state_xy":
        epsilon_index = 4 if axis == "x" else 5
        return {
            "perturbation_id": f"steady_state_{family}__{axis}_{sign_label(sign)}",
            "channel": channel,
            "family": family,
            "semantic_family": "non_feedback_mechanical_process_force",
            "amplitude": float(amplitude),
            "units": units,
            "axis": axis,
            "basis": "cs_lss_process_epsilon_current_physical_block",
            "sign": int(sign),
            "timing": timing,
            "timing_bin": "steady_state_endpoint",
            "adapter": "task_trial_spec.inputs['epsilon']",
            "epsilon_component": f"force_state_{axis}",
            "epsilon_index": int(epsilon_index),
            "description": (
                "Stabilization-task endpoint process-epsilon pulse on the current "
                "physical force-state component."
            ),
        }
    return {
        "perturbation_id": f"steady_state_{family}__{axis}_{sign_label(sign)}",
        "channel": channel,
        "family": family,
        "semantic_family": "non_feedback_mechanical_command_input",
        "amplitude": float(amplitude),
        "units": units,
        "axis": axis,
        "basis": "command_cartesian_force_xy",
        "sign": int(sign),
        "timing": timing,
        "timing_bin": "steady_state_endpoint",
        "adapter": "feedbax.additive_channel_adapter.command_input",
        "description": (
            "Stabilization-task endpoint pulse at the post-controller command/force port."
        ),
    }


def summarize_probe(
    *,
    probe: ProbeSpec,
    base: Any,
    perturbed: Any,
    pulse_start: int,
) -> dict[str, Any]:
    """Summarize one perturbation response as displacement AUC."""

    direction = np.asarray(probe.direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    delta_position = perturbed.position - base.position
    delta_command = perturbed.command - base.command
    delta_velocity = perturbed.velocity - base.velocity
    orthogonal_direction = right_handed_orthogonal_direction(direction)
    aligned_position = np.tensordot(delta_position, direction, axes=([-1], [0]))
    orthogonal_position = np.tensordot(delta_position, orthogonal_direction, axes=([-1], [0]))
    aligned_velocity = np.tensordot(delta_velocity, direction, axes=([-1], [0]))
    orthogonal_velocity = np.tensordot(delta_velocity, orthogonal_direction, axes=([-1], [0]))
    aligned_command = np.tensordot(delta_command, direction, axes=([-1], [0]))
    orthogonal_command = np.tensordot(delta_command, orthogonal_direction, axes=([-1], [0]))
    displacement_norm = np.linalg.norm(delta_position, axis=-1)
    action_norm = np.linalg.norm(delta_command, axis=-1)
    aligned_response = aligned_position[:, :, pulse_start:]
    norm_response = displacement_norm[:, :, pulse_start:]
    action_response = action_norm[:, :, pulse_start:]
    auc_by_trial = np.sum(np.abs(aligned_response), axis=-1) * float(base.dt) * 1000.0
    norm_auc_by_trial = np.sum(norm_response, axis=-1) * float(base.dt) * 1000.0
    aligned_position_window, relative_steps = mean_onset_window(
        aligned_position,
        pulse_start=pulse_start,
    )
    orthogonal_position_window, _ = mean_onset_window(
        orthogonal_position,
        pulse_start=pulse_start,
    )
    aligned_velocity_window, _ = mean_onset_window(
        aligned_velocity,
        pulse_start=pulse_start,
    )
    orthogonal_velocity_window, _ = mean_onset_window(
        orthogonal_velocity,
        pulse_start=pulse_start,
    )
    aligned_command_window, _ = mean_onset_window(
        aligned_command,
        pulse_start=pulse_start,
    )
    orthogonal_command_window, _ = mean_onset_window(
        orthogonal_command,
        pulse_start=pulse_start,
    )
    return {
        "perturbation_id": probe.perturbation_id,
        "group": probe.group,
        "family": probe.family,
        "channel": probe.row["channel"],
        "direction": [float(value) for value in direction],
        "projection_basis": {
            "aligned_direction": [float(value) for value in direction],
            "orthogonal_direction": [float(value) for value in orthogonal_direction],
            "orthogonal_convention": "right_handed_plus_90_degrees_xy",
        },
        "amplitude": float(probe.amplitude),
        "units": probe.units,
        "relative_time_steps": [int(value) for value in relative_steps],
        "aligned_position_window_profile_m": [
            float(value) for value in aligned_position_window
        ],
        "orthogonal_position_window_profile_m": [
            float(value) for value in orthogonal_position_window
        ],
        "aligned_velocity_window_profile_m_s": [
            float(value) for value in aligned_velocity_window
        ],
        "orthogonal_velocity_window_profile_m_s": [
            float(value) for value in orthogonal_velocity_window
        ],
        "aligned_command_window_profile": [float(value) for value in aligned_command_window],
        "orthogonal_command_window_profile": [
            float(value) for value in orthogonal_command_window
        ],
        "metrics": {
            "auc_aligned_displacement_mm_s": float(np.mean(auc_by_trial)),
            "auc_displacement_norm_mm_s": float(np.mean(norm_auc_by_trial)),
            "peak_aligned_displacement_mm": float(
                np.mean(np.max(np.abs(aligned_response), axis=-1)) * 1000.0
            ),
            "peak_displacement_norm_mm": float(
                np.mean(np.max(norm_response, axis=-1)) * 1000.0
            ),
            "peak_action_norm": float(np.mean(np.max(action_response, axis=-1))),
        },
    }


def summarize_by_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate evaluated probes by family."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "evaluated":
            grouped[str(row["family"])].append(row)
    return {family: summarize_probe_group(items) for family, items in sorted(grouped.items())}


def summarize_by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate evaluated probes by high-level diagnostic group."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "evaluated":
            grouped[str(row["group"])].append(row)
    return {group: summarize_probe_group(items) for group, items in sorted(grouped.items())}


def summarize_probe_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return scalar summaries for a set of probe rows."""

    auc = np.asarray(
        [row["metrics"]["auc_aligned_displacement_mm_s"] for row in rows],
        dtype=np.float64,
    )
    norm_auc = np.asarray(
        [row["metrics"]["auc_displacement_norm_mm_s"] for row in rows],
        dtype=np.float64,
    )
    peak = np.asarray(
        [row["metrics"]["peak_aligned_displacement_mm"] for row in rows],
        dtype=np.float64,
    )
    return {
        "n_probes": int(len(rows)),
        "auc_displacement_mm_s_mean": float(np.mean(auc)),
        "auc_displacement_mm_s_sem": sem(auc),
        "auc_displacement_norm_mm_s_mean": float(np.mean(norm_auc)),
        "peak_displacement_mm_mean": float(np.mean(peak)),
        "peak_displacement_mm_sem": sem(peak),
    }


def pairwise_level_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare PGD against the no-PGD open-loop row at each physical level."""

    by_level = {str(row["physical_level"]): {} for row in rows}
    for row in rows:
        by_level[str(row["physical_level"])][str(row["training"])] = row
    comparisons: dict[str, Any] = {}
    for level, pair in sorted(by_level.items()):
        no_pgd = pair["no_pgd_open_loop"]
        pgd = pair["pgd_1p05"]
        comparisons[level] = {
            "feedback_auc_delta_mm_s": float(
                pgd["feedback_auc_mm_s"] - no_pgd["feedback_auc_mm_s"]
            ),
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


def interpret(comparisons: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Generate a compact qualitative interpretation from level-paired deltas."""

    feedback_ratios = np.asarray(
        [payload["feedback_auc_ratio_pgd_over_no_pgd"] for payload in comparisons.values()],
        dtype=np.float64,
    )
    mechanical_ratios = np.asarray(
        [payload["mechanical_auc_ratio_pgd_over_no_pgd"] for payload in comparisons.values()],
        dtype=np.float64,
    )
    feedback_direction = direction_label(float(np.mean(feedback_ratios)))
    mechanical_direction = direction_label(float(np.mean(mechanical_ratios)))
    if feedback_direction == "reduced" and mechanical_direction == "reduced":
        locus = "both feedback-channel and non-feedback/mechanical"
    elif feedback_direction == "reduced":
        locus = "mostly feedback-channel"
    elif mechanical_direction == "reduced":
        locus = "mostly non-feedback/mechanical"
    else:
        locus = "neither"
    return {
        "mean_feedback_ratio_pgd_over_no_pgd": float(np.mean(feedback_ratios)),
        "mean_mechanical_ratio_pgd_over_no_pgd": float(np.mean(mechanical_ratios)),
        "feedback_effect": feedback_direction,
        "mechanical_effect": mechanical_direction,
        "qualitative_locus": locus,
    }


def direction_label(mean_ratio: float) -> str:
    """Classify a mean PGD/no-PGD ratio."""

    if mean_ratio < 0.95:
        return "reduced"
    if mean_ratio > 1.05:
        return "increased"
    return "approximately_unchanged"


def response_label(wash: Mapping[str, Any]) -> str:
    """Use the 87424a4 response-label convention."""

    command = wash["network_output_drift"]["max"]
    hidden = wash["hidden_state_drift"]["max"]
    plant = wash["plant_state_drift"]["max"]
    return (
        "steady_state_response"
        if max(float(command), float(hidden), float(plant)) < 1e-4
        else "washin_endpoint_response"
    )


def checkpoint_selection_summary(selections: Sequence[Any]) -> dict[str, Any]:
    """Summarize selected checkpoints without listing all local paths."""

    rows = [selection.to_json(repo_root=Path(REPO_ROOT).resolve()) for selection in selections]
    batches = [row.get("checkpoint_batch") for row in rows if row.get("checkpoint_batch") is not None]
    sources = sorted({str(row.get("selection_source")) for row in rows if row.get("selection_source")})
    return {
        "n_replicates": len(rows),
        "selection_sources": sources,
        "checkpoint_batch_min": None if not batches else int(min(batches)),
        "checkpoint_batch_max": None if not batches else int(max(batches)),
    }


def probe_contract() -> dict[str, Any]:
    """Return the diagnostic probe contract."""

    return {
        "task_context": "stabilization task endpoint response",
        "fanout_policy": (
            "prefix-equivalent deterministic batched trials, matching the 87424a4 "
            "contract because the eval API has no supported hidden-state fork hook"
        ),
        "pulse_duration_steps": DEFAULT_PULSE_DURATION_STEPS,
        "post_onset_steps_requested": DEFAULT_POST_ONSET_FIGURE_STEPS,
        "n_rollout_trials_per_replicate": DEFAULT_N_ROLLOUT_TRIALS,
        "feedback_probe": (
            "mean over position, velocity, and force/filter sensory-feedback offset "
            "families from the 87424a4 bank"
        ),
        "mechanical_probe": (
            "mean over non-feedback command-input and process-epsilon force-state "
            "pulse families at the same stabilization-task endpoint onset"
        ),
        "auc_metric": (
            "mean signed-direction-aligned absolute hand-position displacement over the "
            "post-onset window, in mm*s"
        ),
        "fixed_probe_scales": {
            "feedback_position_m": DEFAULT_POSITION_SCALE_M,
            "feedback_velocity_m_s": DEFAULT_VELOCITY_SCALE_M_S,
            "feedback_force_filter_model_units": DEFAULT_FORCE_FILTER_SCALE,
            "command_input_n": 1.0,
            "process_epsilon_force_state": 0.01,
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ('run_id', 'training', 'physical_level', 'feedback_auc_mm_s', 'mechanical_auc_mm_s', 'command_input_auc_mm_s', 'process_force_auc_mm_s', 'feedback_peak_mm', 'mechanical_peak_mm', 'response_label')
    write_csv_rows(path, list(rows), fieldnames=columns)


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown note."""

    lines = [
        "# PGD 1.05 Stabilization Task Diagnostics",
        "",
        "This diagnostic reruns stabilization-task endpoint perturbation probes, not "
        "the reach-context perturbation-profile bank. AUC values are mean "
        "signed-direction-aligned absolute hand-position displacement over the "
        "post-onset window in `mm*s`.",
        "",
        "| Row | Training | Level | Feedback AUC | Mechanical AUC | Command AUC | Process-force AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['training']}` | {row['physical_level']} | "
            f"{row['feedback_auc_mm_s']:.4g} | {row['mechanical_auc_mm_s']:.4g} | "
            f"{row['command_input_auc_mm_s']:.4g} | {row['process_force_auc_mm_s']:.4g} |"
        )
    interpretation = summary["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Mean PGD/no-PGD feedback AUC ratio: "
                f"`{interpretation['mean_feedback_ratio_pgd_over_no_pgd']:.3g}` "
                f"({interpretation['feedback_effect']})."
            ),
            (
                f"Mean PGD/no-PGD mechanical AUC ratio: "
                f"`{interpretation['mean_mechanical_ratio_pgd_over_no_pgd']:.3g}` "
                f"({interpretation['mechanical_effect']})."
            ),
            (
                "Qualitative locus: "
                f"`{interpretation['qualitative_locus']}`."
            ),
            "",
            "Definitions: feedback AUC averages the position, velocity, and force/filter "
            "false-feedback offset families. Mechanical AUC averages non-feedback "
            "`command_input_pulse` and `process_epsilon_force_state_xy` pulses.",
            "",
            "Caveat: as in issue 87424a4, rows are labeled as wash-in endpoint responses "
            "unless the strict drift threshold is met; the API uses deterministic "
            "prefix-equivalent fan-out rather than a literal hidden-state snapshot fork.",
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_stabilization_response_figures(
    *,
    detail: Mapping[str, Any],
    summary: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Write stabilization-task perturbation response figures and tracked metadata."""

    figure_spec_dir = mkdir_p(repo_root / "results" / ISSUE / "figures" / FIGURE_TOPIC)
    figure_artifact_dir = mkdir_p(repo_root / "_artifacts" / ISSUE / "figures" / FIGURE_TOPIC)
    note_path = repo_root / "results" / ISSUE / "notes" / f"{FIGURE_TOPIC}.md"
    spec_path = figure_spec_dir / "spec.json"
    summary_by_run = {str(row["run_id"]): row for row in summary["rows"]}
    figure_specs = []
    png_errors: dict[str, str] = {}

    for family in PERTURBATION_FAMILY_ORDER:
        fig, coverage, event_markers = build_stabilization_response_family_figure(
            family=family,
            detail=detail,
            summary_by_run=summary_by_run,
        )
        html_path = figure_artifact_dir / f"{family}.html"
        png_path = figure_artifact_dir / f"{family}.png"
        png_status, png_renderer = write_figure_outputs(
            fig=fig,
            html_path=html_path,
            png_path=png_path,
            png_errors=png_errors,
            error_key=family,
        )
        figure_specs.append(
            {
                "role": "perturbation_family",
                "family": family,
                "title": FAMILY_LABELS[family],
                "figure_kind": "stabilization_task_perturbation_family_response_grid",
                "layout": {
                    "rows": 3,
                    "cols": 3,
                    "row_axis": "response_variable",
                    "row_order": list(RESPONSE_VARIABLE_ORDER),
                    "col_axis": "physical_level",
                    "col_order": list(LEVEL_ORDER),
                },
                "html": repo_relative(html_path, repo_root),
                "png": repo_relative(png_path, repo_root) if png_status == "written" else None,
                "png_status": png_status,
                "png_renderer": png_renderer,
                "png_bytes": png_size(png_path) if png_status == "written" else None,
                "coverage": coverage,
                "perturbation_event_markers": event_markers,
            }
        )

    validate_figure_specs(figure_specs=figure_specs)
    spec = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "issue": ISSUE,
        "source_diagnostic": OUTPUT_STEM,
        "figure_topic": FIGURE_TOPIC,
        "figure_count": len(figure_specs),
        "html_count": len(figure_specs),
        "png_count": sum(1 for figure in figure_specs if figure["png_status"] == "written"),
        "task_label": "stabilization task",
        "plot_contract": {
            "task_context": "stabilization task endpoint perturbation response",
            "figure_axis": "perturbation_family",
            "figure_order": list(PERTURBATION_FAMILY_ORDER),
            "grid": "three response-variable rows by three physical-level columns",
            "row_axis": "response_variable",
            "row_order": list(RESPONSE_VARIABLE_ORDER),
            "col_axis": "physical_level",
            "col_order": list(LEVEL_ORDER),
            "durable_isolation_figure_contract": (
                "future stabilization-task isolation figures compare the same paired "
                "training rows with one 3x3 response-variable-by-level grid per "
                "perturbation family unless a later tracked spec explicitly supersedes it"
            ),
            "paired_rows": {
                "small": ["open_loop_small", "small"],
                "moderate": ["open_loop_moderate", "moderate"],
                "stress": ["open_loop_stress", "stress"],
            },
            "probe_timing": "single steady_state_endpoint pulse timing",
            "trajectory_policy": (
                "command, hand-position, and hand-velocity traces are direct "
                "perturbation residual profiles relative to the unperturbed "
                "stabilization rollout; no separate trajectory-vs-residual figure "
                "split is used"
            ),
            "y_axis": {
                key: RESPONSE_VARIABLE_SPECS[key]["axis_title"]
                for key in RESPONSE_VARIABLE_ORDER
            },
            "orthogonal_trace": (
                "lower-emphasis signed projection onto the +90-degree right-handed "
                "orthogonal direction"
            ),
            "uncertainty_band": "SEM across the four signed x/y probes in each family",
            "trace_source": (
                "directly from per_probe_detail aligned/orthogonal command, position, "
                "and velocity window profiles"
            ),
            "perturbation_event_marker": {
                "display_preference": "shaded vertical onset-to-offset band",
                "x_axis_reference": "seconds relative to perturbation onset",
                "onset_source": (
                    "detail.rows[*].adapter.adapter_provenance.relative_start_time_index "
                    "and summary.rows[*].timing.pulse_start_step"
                ),
                "duration_source": (
                    "detail.rows[*].adapter.adapter_provenance.duration_steps and "
                    "summary.rows[*].timing.pulse_duration_steps"
                ),
                "fallback_when_duration_missing": "vertical onset line at x=0",
            },
        },
        "figures": figure_specs,
        "png_errors": png_errors,
    }
    write_compact_json(spec_path, spec)
    update_marked_section(note_path, FIGURE_MARKER, render_figure_note(spec))
    return {
        "outputs": {
            "stabilization_response_figure_spec": repo_relative(spec_path, repo_root),
            "stabilization_response_figure_note": repo_relative(note_path, repo_root),
            "stabilization_response_figure_dir": repo_relative(figure_artifact_dir, repo_root),
            "stabilization_response_htmls": [figure["html"] for figure in figure_specs],
            "stabilization_response_pngs": [
                figure["png"] for figure in figure_specs if figure["png"] is not None
            ],
        },
        "spec": spec,
    }


def write_figure_outputs(
    *,
    fig: go.Figure,
    html_path: Path,
    png_path: Path,
    png_errors: dict[str, str],
    error_key: str,
) -> tuple[str, str | None]:
    """Write one HTML/PNG figure pair and return the PNG status."""

    fig.write_html(html_path, include_plotlyjs=True)
    try:
        write_png_image(fig, png_path)
    except (ValueError, RuntimeError, OSError) as exc:
        try:
            write_png_from_html(html_path=html_path, png_path=png_path)
        except (RuntimeError, OSError, subprocess.SubprocessError) as chrome_exc:
            if png_path.exists() and png_path.stat().st_size > 0:
                return "written", "existing_png_after_export_block"
            png_errors[error_key] = (
                f"kaleido {type(exc).__name__}: {exc}; "
                f"chrome {type(chrome_exc).__name__}: {chrome_exc}"
            )
            return "blocked", None
        return "written", "chrome_headless_html_screenshot_after_kaleido_block"
    return "written", "kaleido"


def build_stabilization_response_family_figure(
    *,
    family: str,
    detail: Mapping[str, Any],
    summary_by_run: Mapping[str, Mapping[str, Any]],
) -> tuple[go.Figure, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one 3x3 response-variable-by-level perturbation-family figure."""

    subplot_titles = [
        f"{RESPONSE_VARIABLE_SPECS[response]['label']} - {level}"
        for response in RESPONSE_VARIABLE_ORDER
        for level in LEVEL_ORDER
    ]
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.055,
        vertical_spacing=0.09,
    )
    coverage = []
    event_markers: list[dict[str, Any]] = []
    legend_seen: set[tuple[str, str]] = set()
    for row_index, response_variable in enumerate(RESPONSE_VARIABLE_ORDER, start=1):
        response_spec = RESPONSE_VARIABLE_SPECS[response_variable]
        for col_index, level in enumerate(LEVEL_ORDER, start=1):
            run_pair = {
                "no_pgd_open_loop": f"open_loop_{level}",
                "pgd_1p05": level,
            }
            timing = summary_by_run[run_pair["no_pgd_open_loop"]]["timing"]
            dt = float(summary_by_run[run_pair["no_pgd_open_loop"]]["dt_s"])
            for training, run_id in run_pair.items():
                family_rows = [
                    row
                    for row in detail["rows"][run_id]
                    if row.get("status") == "evaluated" and row.get("family") == family
                ]
                profile = aggregate_family_response_profile(
                    family_rows,
                    response_variable=response_variable,
                )
                style = TRAINING_STYLES[training]
                x = np.asarray(profile["relative_time_steps"], dtype=np.float64) * dt
                aligned = np.asarray(profile["aligned_mean"], dtype=np.float64)
                aligned_sem = np.asarray(profile["aligned_sem"], dtype=np.float64)
                orthogonal = np.asarray(profile["orthogonal_mean"], dtype=np.float64)
                label = style["label"]
                show_aligned = (training, "aligned") not in legend_seen
                show_orthogonal = (training, "orthogonal") not in legend_seen
                add_mean_sem_trace(
                    fig,
                    x=x,
                    mean=aligned,
                    sem=aligned_sem,
                    name=f"{label} aligned",
                    legendgroup=f"{training}-aligned",
                    color=style["color"],
                    band_color=style["band"],
                    row=row_index,
                    col=col_index,
                    showlegend=show_aligned,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=orthogonal,
                        mode="lines",
                        name=f"{label} orthogonal",
                        legendgroup=f"{training}-orthogonal",
                        showlegend=show_orthogonal,
                        line={"color": style["color"], "width": 1.35, "dash": "dot"},
                        opacity=0.72,
                    ),
                    row=row_index,
                    col=col_index,
                )
                legend_seen.add((training, "aligned"))
                legend_seen.add((training, "orthogonal"))
                coverage.append(
                    {
                        "family": family,
                        "response_variable": response_variable,
                        "physical_level": level,
                        "training": training,
                        "run_id": run_id,
                        "n_evaluated_probes": int(profile["n_probes"]),
                        "perturbation_ids": profile["perturbation_ids"],
                        "profile_source": profile["profile_source"],
                        "unit": profile["unit"],
                    }
                )
            marker = infer_perturbation_event_marker(
                family_rows=[
                    row
                    for run_id in run_pair.values()
                    for row in detail["rows"][run_id]
                    if row.get("status") == "evaluated" and row.get("family") == family
                ],
                summary_timing=timing,
                dt=dt,
            )
            add_perturbation_event_marker(fig, marker=marker, row=row_index, col=col_index)
            event_markers.append(
                {
                    "family": family,
                    "response_variable": response_variable,
                    "physical_level": level,
                    **marker,
                }
            )
            if col_index == 1:
                fig.update_yaxes(
                    title_text=response_spec["axis_title"],
                    row=row_index,
                    col=col_index,
                )
            if row_index == len(RESPONSE_VARIABLE_ORDER):
                fig.update_xaxes(
                    title_text="time from perturbation onset (s)",
                    row=row_index,
                    col=col_index,
                )
    fig.update_layout(
        title=f"c92 PGD 1.05 stabilization task response: {FAMILY_LABELS[family]}",
        template="plotly_white",
        width=1320,
        height=920,
        margin={"l": 78, "r": 28, "t": 96, "b": 96},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    return fig, coverage, event_markers


def aggregate_family_response_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    response_variable: str,
) -> dict[str, Any]:
    """Aggregate signed x/y probe response profiles for one family and row."""

    if not rows:
        raise ValueError("family response figure requires at least one evaluated probe")
    response_spec = RESPONSE_VARIABLE_SPECS[response_variable]
    relative_steps = rows[0]["relative_time_steps"]
    aligned = np.asarray(
        [row[response_spec["aligned_key"]] for row in rows],
        dtype=np.float64,
    ) * float(response_spec["scale"])
    orthogonal = np.asarray(
        [row[response_spec["orthogonal_key"]] for row in rows],
        dtype=np.float64,
    ) * float(response_spec["scale"])
    return {
        "n_probes": int(len(rows)),
        "perturbation_ids": [str(row["perturbation_id"]) for row in rows],
        "profile_source": {
            "aligned": str(response_spec["aligned_key"]),
            "orthogonal": str(response_spec["orthogonal_key"]),
        },
        "unit": str(response_spec["unit"]),
        "relative_time_steps": [int(value) for value in relative_steps],
        "aligned_mean": [float(value) for value in np.mean(aligned, axis=0)],
        "aligned_sem": [float(value) for value in sem_profile(aligned)],
        "orthogonal_mean": [float(value) for value in np.mean(orthogonal, axis=0)],
        "orthogonal_sem": [float(value) for value in sem_profile(orthogonal)],
    }


def infer_perturbation_event_marker(
    *,
    family_rows: Sequence[Mapping[str, Any]],
    summary_timing: Mapping[str, Any],
    dt: float,
) -> dict[str, Any]:
    """Infer onset and duration display metadata for one subplot."""

    provenance_rows = [
        row.get("adapter", {}).get("adapter_provenance", {})
        for row in family_rows
        if isinstance(row.get("adapter"), Mapping)
    ]
    detail_onsets = {
        int(value)
        for provenance in provenance_rows
        for value in (provenance.get("relative_start_time_index"),)
        if value is not None
    }
    detail_durations = {
        int(value)
        for provenance in provenance_rows
        for value in (provenance.get("duration_steps"),)
        if value is not None
    }
    summary_onset = int(summary_timing["pulse_start_step"])
    onset_step = next(iter(detail_onsets)) if len(detail_onsets) == 1 else summary_onset
    onset_source = (
        "detail.adapter_provenance.relative_start_time_index"
        if len(detail_onsets) == 1
        else "summary.timing.pulse_start_step"
    )
    summary_duration = summary_timing.get("pulse_duration_steps")
    duration_steps = None
    duration_source = None
    if len(detail_durations) == 1:
        duration_steps = next(iter(detail_durations))
        duration_source = "detail.adapter_provenance.duration_steps"
    elif summary_duration is not None:
        duration_steps = int(summary_duration)
        duration_source = "summary.timing.pulse_duration_steps"
    display = "duration_band" if duration_steps is not None else "onset_line"
    return {
        "display": display,
        "x_axis_reference": "seconds relative to perturbation onset",
        "onset_step": int(onset_step),
        "onset_source": onset_source,
        "duration_steps": None if duration_steps is None else int(duration_steps),
        "duration_source": duration_source,
        "display_x0_s": 0.0,
        "display_x1_s": None if duration_steps is None else float(duration_steps) * float(dt),
        "dt_s": float(dt),
    }


def add_perturbation_event_marker(
    fig: go.Figure,
    *,
    marker: Mapping[str, Any],
    row: int,
    col: int,
) -> None:
    """Add a shaded perturbation-duration band, or an onset line if duration is missing."""

    if marker["display"] == "duration_band" and marker["display_x1_s"] is not None:
        fig.add_vrect(
            x0=float(marker["display_x0_s"]),
            x1=float(marker["display_x1_s"]),
            fillcolor="rgba(148,163,184,0.25)",
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )
        return
    fig.add_vline(
        x=float(marker["display_x0_s"]),
        line={"color": "rgba(100,116,139,0.75)", "width": 1.2, "dash": "dash"},
        row=row,
        col=col,
    )


def add_mean_sem_trace(
    fig: go.Figure, *, x: np.ndarray, mean: np.ndarray, sem: np.ndarray, name: str,
    legendgroup: str, color: str, band_color: str, row: int, col: int, showlegend: bool,
) -> None:
    """Add a mean trace plus SEM band."""
    canonical_add_band_trace(
        fig, x=x, mean=mean, spread=sem, name=name, legendgroup=legendgroup,
        color=color, band_fill_color=band_color, band_line_color=color, band_mode="lines",
        row=row, col=col, showlegend=showlegend, line_width=2.2,
    )


def write_png_image(fig: go.Figure, path: Path) -> None:
    """Write a PNG, working around Kaleido 0.2's unquoted path wrapper."""

    import kaleido
    import plotly.io as pio

    direct_binary = Path(kaleido.__file__).resolve().parent / "executable" / "bin" / "kaleido"
    if direct_binary.exists():
        pio.kaleido.scope.__class__.executable_path = classmethod(
            lambda cls: str(direct_binary)
        )
    fig.write_image(path, scale=2)


def write_png_from_html(*, html_path: Path, png_path: Path) -> None:
    """Render a Plotly HTML figure to PNG using a local headless browser."""

    chrome_path = find_headless_chrome()
    if chrome_path is None:
        raise RuntimeError("no local Chrome/Edge executable found for HTML PNG fallback")
    result = subprocess.run(
        [
            str(chrome_path),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--virtual-time-budget=3000",
            "--window-size=1360,980",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise RuntimeError("headless browser completed without a nonempty PNG")


def find_headless_chrome() -> Path | None:
    """Return a local Chromium-family executable when available."""

    for candidate in (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
    ):
        if candidate.exists():
            return candidate
    return None


def validate_figure_specs(
    *,
    figure_specs: Sequence[Mapping[str, Any]],
) -> None:
    """Assert expected family, response-variable, level, run, and event coverage."""

    if len(figure_specs) != len(PERTURBATION_FAMILY_ORDER):
        raise ValueError(f"expected {len(PERTURBATION_FAMILY_ORDER)} figures, got {len(figure_specs)}")
    if [str(figure["family"]) for figure in figure_specs] != list(PERTURBATION_FAMILY_ORDER):
        raise ValueError("perturbation-family figure order mismatch")
    expected_trainings = {"no_pgd_open_loop", "pgd_1p05"}
    for figure in figure_specs:
        layout = figure["layout"]
        if (layout["rows"], layout["cols"]) != (3, 3):
            raise ValueError(f"{figure['family']} must be 3x3, got {layout}")
        if list(layout["row_order"]) != list(RESPONSE_VARIABLE_ORDER):
            raise ValueError(f"{figure['family']} response row order mismatch")
        if list(layout["col_order"]) != list(LEVEL_ORDER):
            raise ValueError(f"{figure['family']} level column order mismatch")
        coverage = list(figure["coverage"])
        if len(coverage) != 18:
            raise ValueError(f"{figure['family']} should cover 18 traces, got {len(coverage)}")
        seen: dict[tuple[str, str], set[str]] = {
            (response, level): set()
            for response in RESPONSE_VARIABLE_ORDER
            for level in LEVEL_ORDER
        }
        for row in coverage:
            key = (str(row["response_variable"]), str(row["physical_level"]))
            if key not in seen:
                raise ValueError(f"unexpected coverage key for {figure['family']}: {key}")
            seen[key].add(str(row["training"]))
            if int(row["n_evaluated_probes"]) != 4:
                raise ValueError(
                    f"{figure['family']} {row['run_id']} should have 4 probes, "
                    f"got {row['n_evaluated_probes']}"
                )
        for key, trainings in seen.items():
            if trainings != expected_trainings:
                raise ValueError(f"{figure['family']} {key} coverage mismatch: {trainings}")
        markers = list(figure["perturbation_event_markers"])
        if len(markers) != 9:
            raise ValueError(f"{figure['family']} should have 9 event markers, got {len(markers)}")
        for marker in markers:
            if marker["display"] != "duration_band":
                raise ValueError(f"{figure['family']} expected duration bands, got {marker}")
            if marker["duration_steps"] is None:
                raise ValueError(f"{figure['family']} missing perturbation duration")


def render_figure_note(spec: Mapping[str, Any]) -> str:
    """Render the stabilization response figure note."""

    lines = [
        "# PGD 1.05 Stabilization Task Perturbation Responses",
        "",
        "- Scope: stabilization-task endpoint perturbation response figures.",
        "- Figure family: one figure per perturbation family "
        f"({', '.join(f'`{family}`' for family in PERTURBATION_FAMILY_ORDER)}).",
        "- Per-figure layout: 3 rows of response variables (`command`, `position`, "
        "`velocity`) by 3 physical-level columns (`small`, `moderate`, `stress`).",
        "- Contract: later stabilization-task isolation figures should use the same "
        "perturbation-family figure set and response-variable-by-level layout unless "
        "a later tracked spec supersedes it.",
        "- Row pairing: each subplot overlays the no-PGD open-loop calibrated row with "
        "its PGD 1.05 counterpart.",
        "- Timing: single `steady_state_endpoint` pulse timing; no early/mid/late split.",
        "- Perturbation marker: shaded vertical band from perturbation onset to offset; "
        "duration comes from adapter provenance with summary timing as fallback.",
        "- Trace contract: signed direction-aligned response residual with a lower-emphasis "
        "orthogonal companion trace. Command, hand-position, and hand-velocity traces "
        "come directly from the diagnostic detail payload and are residuals relative "
        "to the unperturbed endpoint rollout.",
        f"- Figure spec: `results/{ISSUE}/figures/{FIGURE_TOPIC}/spec.json`.",
        f"- Figure count: `{spec['figure_count']}` HTML, `{spec['png_count']}` PNG.",
        "",
        "| Perturbation family | Layout | Event marker | HTML | PNG |",
        "|---|---|---|---|---:|",
    ]
    for figure in spec["figures"]:
        layout = figure["layout"]
        layout_label = f"{layout['rows']}x{layout['cols']}"
        marker_displays = sorted(
            {
                f"{marker['display']} ({marker['duration_steps']} steps)"
                for marker in figure["perturbation_event_markers"]
            }
        )
        marker_label = ", ".join(marker_displays)
        lines.append(
            f"| `{figure['family']}` | {layout_label} | {marker_label} | "
            f"`{figure['html']}` | `{figure['png_status']}` |"
        )
    if spec.get("png_errors"):
        lines.extend(["", "PNG export blockers:"])
        for family, error in spec["png_errors"].items():
            lines.append(f"- `{family}`: {error}")
    return "\n".join(lines) + "\n"


def mean_onset_window(
    aligned_values: np.ndarray,
    *,
    pulse_start: int,
    pre_steps: int = DEFAULT_PRE_ONSET_FIGURE_STEPS,
    post_steps: int = DEFAULT_POST_ONSET_FIGURE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trial/replicate mean in a pre-onset and recovery window."""

    start = max(int(pulse_start) - int(pre_steps), 0)
    stop = min(int(pulse_start) + int(post_steps), aligned_values.shape[2])
    window = aligned_values[:, :, start:stop]
    relative_steps = np.arange(start, stop, dtype=int) - int(pulse_start)
    return np.mean(window, axis=(0, 1)), relative_steps


def right_handed_orthogonal_direction(direction: np.ndarray) -> np.ndarray:
    """Return the +90 degree right-handed x-y rotation of ``direction``."""

    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return np.asarray([-unit[1], unit[0]], dtype=np.float64)


def sem_profile(values: np.ndarray) -> np.ndarray:
    """Return SEM across profile rows."""

    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def repo_relative(path: Path, repo_root: Path) -> str:
    """Return a repo-relative path string."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        pass
    for marker in ("_artifacts", "results"):
        if marker in path.parts:
            idx = path.parts.index(marker)
            return str(Path(*path.parts[idx:]))
    return str(path)


def png_size(path: Path) -> int:
    """Return a PNG byte count for generated figure specs."""

    return int(path.stat().st_size)


def sign_label(sign: int) -> str:
    """Return a sign label."""

    return "pos" if sign > 0 else "neg"


def sem(values: np.ndarray) -> float:
    """Return standard error of the mean."""

    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio with NaN for zero denominators."""

    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


if __name__ == "__main__":
    main()
