#!/usr/bin/env python
"""Materialize beta 1.4 nominal velocity profiles split by GRU replicate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
from feedbax.plot import save_figure

from materialize_beta1p4_nominal_velocity_profiles import (
    COMPARATOR_COLORS,
    DT,
    ISSUE,
    RUN_ROWS,
    build_robust_output_feedback_6d_context,
    forward_velocity_profile,
    json_safe,
    repo_rel,
    write_png_output,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import evaluate_run_rollouts
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    resolve_run_inputs,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import _build_extlqg_comparator_context
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT
from rlrmp.viz import profile_comparison_grid


TOPIC = "beta1p4_nominal_velocity_profiles_by_replicate"
NOTE_PATH = REPO_ROOT / "results" / ISSUE / "notes" / f"{TOPIC}.md"
SPEC_PATH = REPO_ROOT / "results" / ISSUE / "figures" / TOPIC / "spec.json"
HTML_PATH = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC / "figure.html"
PNG_PATH = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC / "figure.png"

REPLICATE_COLORS = (
    "#2563eb",
    "#7c3aed",
    "#f97316",
    "#059669",
    "#e11d48",
    "#0891b2",
    "#a16207",
    "#4f46e5",
)


def main() -> None:
    """CLI entry point."""

    extlqg_context = _build_extlqg_comparator_context(physical_dim=6)
    robust_context = build_robust_output_feedback_6d_context()
    run_profiles = evaluate_requested_run_profiles_by_replicate()
    output = materialize_figure(
        run_profiles=run_profiles,
        extlqg_context=extlqg_context,
        robust_context=robust_context,
    )
    write_note(output)
    print(json.dumps(output, indent=2, sort_keys=True))


def evaluate_requested_run_profiles_by_replicate() -> dict[str, np.ndarray]:
    """Evaluate final-checkpoint nominal velocity as replicate x time curves."""

    runs = resolve_run_inputs(
        experiment=ISSUE,
        run_ids=[str(row["run_id"]) for row in RUN_ROWS],
        labels=None,
        repo_root=REPO_ROOT,
    )
    profiles: dict[str, np.ndarray] = {}
    for run in runs:
        evaluation, _model = evaluate_run_rollouts(
            run,
            experiment=ISSUE,
            n_rollout_trials=DEFAULT_N_ROLLOUT_TRIALS,
            use_validation_selected_checkpoints=False,
            repo_root=REPO_ROOT,
        )
        profiles[run.run_id] = forward_velocity_profiles_by_replicate(evaluation.velocity)
    return profiles


def forward_velocity_profiles_by_replicate(velocity: Any) -> np.ndarray:
    """Return replicate x time forward velocity curves averaged over trials."""

    array = np.asarray(velocity, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 2:
        raise ValueError(f"expected velocity with trailing shape (time, 2), got {array.shape}")
    if array.ndim == 3:
        return array[..., 0]
    forward = array[..., 0]
    sample_axes = tuple(range(1, forward.ndim - 1))
    if not sample_axes:
        return forward
    return np.nanmean(forward, axis=sample_axes)


def materialize_figure(
    *,
    run_profiles: Mapping[str, np.ndarray],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and save the beta 1.4 by-replicate nominal velocity figure."""

    fig = profile_comparison_grid(
        n_panels=len(RUN_ROWS),
        rows=len(RUN_ROWS),
        cols=1,
        subplot_titles=[str(row["label"]) for row in RUN_ROWS],
        vertical_spacing=0.075,
    )
    ext_profile = forward_velocity_profile(extlqg_context["base_evaluation"].velocity)
    robust_profile = forward_velocity_profile(robust_context["velocity"])

    for row_index, row_spec in enumerate(RUN_ROWS, start=1):
        row_profiles = np.asarray(run_profiles[str(row_spec["run_id"])], dtype=np.float64)
        add_replicate_lines(
            fig,
            row_profiles,
            row=row_index,
            col=1,
            name_prefix=f"{row_spec['label']} GRU",
        )
        add_line(
            fig,
            ext_profile,
            row=row_index,
            col=1,
            name="6D extLQG",
            color=COMPARATOR_COLORS["extlqg6d"],
            dash="dash",
            showlegend=row_index == 1,
            width=2.8,
        )
        add_line(
            fig,
            robust_profile,
            row=row_index,
            col=1,
            name="6D output-feedback H-infinity",
            color=COMPARATOR_COLORS["robust_output_feedback6d"],
            dash="dot",
            showlegend=row_index == 1,
            width=2.8,
        )
        fig.update_yaxes(title_text="m/s", row=row_index, col=1)
    fig.update_xaxes(title_text="time from movement onset (s)", row=len(RUN_ROWS), col=1)
    fig.update_layout(
        title="b413 beta 1.4 nominal forward velocity profiles by replicate",
        template="plotly_white",
        width=1040,
        height=820,
        legend_title_text="profile",
        margin={"l": 78, "r": 24, "t": 90, "b": 70},
    )

    spec = build_spec(robust_context=robust_context, run_profiles=run_profiles)
    saved = save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=ISSUE,
        topic=TOPIC,
        extra_packages=["rlrmp"],
    )
    png_export = write_png_output(fig=fig, html_path=HTML_PATH, png_path=PNG_PATH)
    append_png_export_to_spec(png_export)
    return {
        "status": "materialized",
        "topic": TOPIC,
        "save_result": json_safe(saved),
        "spec": repo_rel(SPEC_PATH),
        "html": repo_rel(HTML_PATH),
        "png": repo_rel(PNG_PATH) if png_export["status"] == "written" else None,
        "png_export": png_export,
        "rows": row_summaries(run_profiles),
        "analytical_comparator_contract": spec["analytical_comparator_contract"],
    }


def add_replicate_lines(
    fig: go.Figure,
    profiles: np.ndarray,
    *,
    row: int,
    col: int,
    name_prefix: str,
) -> None:
    """Add one visible line per GRU replicate."""

    if profiles.ndim != 2:
        raise ValueError(f"expected replicate x time profiles, got {profiles.shape}")
    for replicate_index, profile in enumerate(profiles):
        color = REPLICATE_COLORS[replicate_index % len(REPLICATE_COLORS)]
        add_line(
            fig,
            profile,
            row=row,
            col=col,
            name=f"{name_prefix} rep {replicate_index}",
            color=color,
            dash="solid",
            showlegend=True,
            width=1.9,
        )


def add_line(
    fig: go.Figure,
    profile: np.ndarray,
    *,
    row: int,
    col: int,
    name: str,
    color: str,
    dash: str,
    showlegend: bool,
    width: float = 2.1,
) -> None:
    """Add a single profile line to a subplot."""

    line_profile = np.asarray(profile, dtype=np.float64)
    if line_profile.ndim == 2:
        line_profile = np.nanmean(line_profile, axis=0)
    if line_profile.ndim != 1:
        raise ValueError(f"expected a 1D profile line, got shape {line_profile.shape}")
    time = np.arange(line_profile.shape[0], dtype=np.float64) * DT
    fig.add_trace(
        go.Scatter(
            x=time,
            y=line_profile,
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            line={"color": color, "dash": dash, "width": width},
        ),
        row=row,
        col=col,
    )


def build_spec(
    *,
    robust_context: Mapping[str, Any],
    run_profiles: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return the tracked by-replicate figure specification."""

    return {
        "schema_version": "rlrmp.b413_beta1p4_nominal_velocity_profiles_by_replicate.v1",
        "issue": ISSUE,
        "figure_kind": "beta1p4_nominal_forward_velocity_profiles_by_replicate",
        "analytical_comparator_contract": {
            "extlqg": {
                "label": "6D extLQG",
                "state_dim": 36,
                "physical_dim": 6,
                "disturbance_integrators_exposed": False,
                "source": (
                    "rlrmp.analysis.pipelines.gru_perturbation_bank."
                    "_build_extlqg_comparator_context(physical_dim=6)"
                ),
                "game_contract": "6D no-integrator C&S comparator",
            },
            "output_feedback_hinf": robust_context["contract"],
        },
        "inputs": [
            {
                "role": "run_spec",
                "path": f"results/{ISSUE}/runs/{row['run_id']}.json",
                "label": row["label"],
                "adversary_mechanism": row["adversary_mechanism"],
                "artifact_dir": f"_artifacts/{ISSUE}/runs/{row['run_id']}",
            }
            for row in RUN_ROWS
        ],
        "transform": [
            {
                "name": "final_checkpoint_forward_velocity_profile_by_replicate",
                "kwargs": {
                    "row_labels": {str(row["run_id"]): row["label"] for row in RUN_ROWS},
                    "n_rollout_trials_per_replicate": DEFAULT_N_ROLLOUT_TRIALS,
                    "use_validation_selected_checkpoints": False,
                    "gru_curve_contract": (
                        "one trace per replicate; each trace is the mean forward "
                        "velocity over the fixed validation rollout bank"
                    ),
                    "analytical_comparators": [
                        "6d_output_feedback_extlqg",
                        "6d_output_feedback_hinf",
                    ],
                },
            }
        ],
        "replicate_curve_summary": row_summaries(run_profiles),
        "plot_kwargs": {
            "grid_helper": "rlrmp.viz.profile_comparison_grid",
            "shared_yaxes": "all",
            "rows": len(RUN_ROWS),
            "physical_state_dim": 6,
            "state_dim": 36,
            "disturbance_integrators_exposed": False,
        },
        "outputs": {
            "html": repo_rel(HTML_PATH),
            "png": repo_rel(PNG_PATH),
        },
    }


def row_summaries(run_profiles: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Return compact per-row replicate metadata for the spec and CLI output."""

    summaries: list[dict[str, Any]] = []
    for row in RUN_ROWS:
        profiles = np.asarray(run_profiles[str(row["run_id"])], dtype=np.float64)
        peak_by_replicate = np.nanmax(profiles, axis=1)
        summaries.append(
            {
                "run_id": row["run_id"],
                "label": row["label"],
                "adversary_mechanism": row["adversary_mechanism"],
                "n_replicates": int(profiles.shape[0]),
                "n_time_steps": int(profiles.shape[1]),
                "peak_forward_velocity_m_s_by_replicate": [
                    float(value) for value in peak_by_replicate
                ],
                "peak_forward_velocity_m_s_range": [
                    float(np.nanmin(peak_by_replicate)),
                    float(np.nanmax(peak_by_replicate)),
                ],
            }
        )
    return summaries


def append_png_export_to_spec(png_export: Mapping[str, Any]) -> None:
    """Append PNG export status to this figure's tracked spec."""

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec["png_export"] = dict(png_export)
    write_compact_json(SPEC_PATH, spec)


def write_note(output: Mapping[str, Any]) -> None:
    """Write a concise regenerable note for the by-replicate profile figure."""

    lines = [
        "# Beta 1.4 Nominal Velocity Profiles by Replicate",
        "",
        "- Scope: nominal velocity profiles only for the completed b413 beta 1.4 rows.",
        "- Rows: `direct_epsilon`, `linear_no_bias`, and `affine`.",
        "- GRU traces: one line per trained replicate, averaged over the fixed "
        f"{DEFAULT_N_ROLLOUT_TRIALS}-trial validation rollout bank.",
        "- Checkpoint policy: final trained checkpoint for all GRU traces.",
        "- Analytical comparators: 6D no-integrator extLQG and 6D no-integrator "
        "output-feedback H-infinity.",
        f"- Figure spec: `{output['spec']}`.",
        f"- HTML artifact: `{output['html']}`.",
        f"- PNG artifact: `{output['png']}`.",
        f"- PNG renderer: `{output['png_export']['renderer']}`.",
        "",
        "| row | n replicates | peak forward velocity range (m/s) |",
        "|---|---:|---:|",
    ]
    for row in output["rows"]:
        peak_range = row["peak_forward_velocity_m_s_range"]
        lines.append(
            f"| `{row['run_id']}` | {row['n_replicates']} | "
            f"{peak_range[0]:.6g} - {peak_range[1]:.6g} |"
        )
    lines.append("")
    update_marked_section(NOTE_PATH, TOPIC, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
