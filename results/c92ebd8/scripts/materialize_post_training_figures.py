#!/usr/bin/env python
"""Materialize c92 post-training figures and notes."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from feedbax.plot import save_figure

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    simulate_robust_released_forward,
    zero_forward_noise_draws,
    zero_noise_covariances,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    _build_extlqg_comparator_context,
    _simulate_extlqg_perturbed,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.viz import profile_comparison_grid


ISSUE = "c92ebd8"
DT = 0.01
REACH_LENGTH_M = 0.15
OUTPUT_TAG = "validation_selected_moderate"
RUNS_DIR = REPO_ROOT / "results" / ISSUE / "runs"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
PERT_MANIFEST = NOTES_DIR / f"gru_perturbation_response_{OUTPUT_TAG}_manifest.json"
EVAL_MANIFEST = NOTES_DIR / f"gru_evaluation_diagnostics_{OUTPUT_TAG}.json"
FEEDBACK_ABLATION = NOTES_DIR / f"gru_feedback_ablation_{OUTPUT_TAG}.json"
PHENOTYPE_SIDECAR = NOTES_DIR / f"hinf_phenotype_sidecar_{OUTPUT_TAG}.json"
POSTRUN_MANIFEST = NOTES_DIR / f"gru_postrun_materialization_{OUTPUT_TAG}.json"
FIGURE_ROOT = REPO_ROOT / "results" / ISSUE / "figures"
PERT_FIGURE_TOPIC = "moderate_perturbation_profiles"
PERT_FIGURE_SPEC = FIGURE_ROOT / PERT_FIGURE_TOPIC / "spec.json"
PERT_FIGURE_BULK = REPO_ROOT / "_artifacts" / ISSUE / "figures" / PERT_FIGURE_TOPIC
NOMINAL_TOPIC = "nominal_velocity_profiles"
NOTE_PATH = NOTES_DIR / "post_training_analysis.md"

RUN_ORDER = (
    "open_loop_small",
    "open_loop_moderate",
    "open_loop_stress",
    "closed_loop_small",
    "closed_loop_moderate",
    "closed_loop_stress",
    "closed_loop_cmd_lateral_small",
    "closed_loop_cmd_lateral_moderate",
    "closed_loop_cmd_lateral_stress",
)

RUN_LABELS = {run_id: run_id for run_id in RUN_ORDER}

SOURCE_COLORS = {
    "gru": "#2563eb",
    "extlqg6d": "#c2410c",
    "robust_output_feedback6d": "#15803d",
}
COORD_DASH = {"orthogonal": "solid", "along": "dot"}
QUANTITY_SPECS = (
    ("command", "Command", "N"),
    ("position", "Position", "m"),
    ("velocity", "Velocity", "m/s"),
)
TIMING_ORDER = {
    "movement_onset": 0,
    "initial_condition": 1,
    "early": 10,
    "early_visible": 11,
    "mid": 20,
    "mid_visible": 21,
    "late": 30,
    "late_visible": 31,
    "none": 99,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-perturbation-profiles", action="store_true")
    parser.add_argument("--skip-nominal-velocity", action="store_true")
    args = parser.parse_args()

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    extlqg_context = _build_extlqg_comparator_context(physical_dim=6)
    robust_context = build_robust_output_feedback_6d_context()

    if not args.skip_perturbation_profiles:
        outputs["moderate_perturbation_profiles"] = materialize_moderate_profiles(
            extlqg_context=extlqg_context,
        )
    elif PERT_FIGURE_SPEC.exists():
        spec = read_json(PERT_FIGURE_SPEC)
        outputs["moderate_perturbation_profiles"] = {
            "status": "materialized",
            "spec": repo_rel(PERT_FIGURE_SPEC),
            "figure_count": int(spec.get("figure_count", 0)),
            "bulk_dir": repo_rel(PERT_FIGURE_BULK),
        }
    if not args.skip_nominal_velocity:
        outputs["nominal_velocity_profiles"] = materialize_nominal_velocity_profiles(
            extlqg_context=extlqg_context,
            robust_context=robust_context,
        )
    elif (FIGURE_ROOT / NOMINAL_TOPIC / "spec.json").exists():
        outputs["nominal_velocity_profiles"] = {
            "status": "materialized",
            "spec": f"results/{ISSUE}/figures/{NOMINAL_TOPIC}/spec.json",
            "html": f"results/{ISSUE}/figures/{NOMINAL_TOPIC}/figure.html",
        }
    write_note(outputs)
    print(json.dumps(outputs, indent=2, sort_keys=True))


def materialize_moderate_profiles(*, extlqg_context: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_json(PERT_MANIFEST)
    detail_manifest = load_detail_manifest(manifest)
    PERT_FIGURE_BULK.mkdir(parents=True, exist_ok=True)
    figure_specs = []
    for run_id in RUN_ORDER:
        run = detail_manifest["runs"][run_id]
        rows = [
            row
            for row in run["perturbations"]
            if row.get("status") == "evaluated"
            and row.get("perturbation", {}).get("level_name") == "moderate"
        ]
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[figure_group_key(row)].append(row)
        for group_key in sorted(groups):
            group_rows = groups[group_key]
            title = f"{RUN_LABELS[run_id]}: {figure_title(group_rows[0])}"
            for figure_kind in ("trajectory", "residual"):
                fig = build_family_figure(
                    group_rows,
                    run=run,
                    extlqg_context=extlqg_context,
                    figure_kind=figure_kind,
                    title=f"{title}: moderate {figure_kind}",
                )
                path = (
                    PERT_FIGURE_BULK
                    / safe_slug(run_id)
                    / f"{safe_slug(group_key)}__{figure_kind}.html"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(path, include_plotlyjs="cdn")
                figure_specs.append(
                    {
                        "run_id": run_id,
                        "run_label": RUN_LABELS[run_id],
                        "group_key": group_key,
                        "title": title,
                        "figure_kind": figure_kind,
                        "n_rows": len(group_rows),
                        "html": repo_rel(path),
                        "extlqg_available_rows": sum(
                            1
                            for row in group_rows
                            if row.get("extlqg_comparator", {}).get("status") == "available"
                        ),
                    }
                )
    spec = {
        "schema_version": "rlrmp.c92_moderate_perturbation_profiles.v1",
        "issue": ISSUE,
        "physical_level": "moderate",
        "source_manifest": repo_rel(PERT_MANIFEST),
        "bulk_detail_manifest": manifest.get("bulk_detail_manifest"),
        "plot_contract": moderate_profile_plot_contract(),
        "figure_count": len(figure_specs),
        "figures": figure_specs,
    }
    PERT_FIGURE_SPEC.parent.mkdir(parents=True, exist_ok=True)
    PERT_FIGURE_SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return {
        "status": "materialized",
        "spec": repo_rel(PERT_FIGURE_SPEC),
        "figure_count": len(figure_specs),
        "bulk_dir": repo_rel(PERT_FIGURE_BULK),
    }


def build_family_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    figure_kind: Literal["trajectory", "residual"],
    title: str,
) -> go.Figure:
    timing_bins = timing_bins_for_rows(rows)
    n_cols = len(timing_bins)
    subplot_titles = [
        f"{quantity_label}: {timing_label}"
        for quantity_label, _quantity_name, _unit in QUANTITY_SPECS
        for timing_label in timing_bins
    ]
    fig = profile_comparison_grid(
        n_panels=len(QUANTITY_SPECS) * n_cols,
        rows=len(QUANTITY_SPECS),
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes="rows",
        vertical_spacing=0.08,
        horizontal_spacing=0.045,
    )
    legend_seen: set[tuple[str, str, str]] = set()
    for col_index, timing_label in enumerate(timing_bins, start=1):
        timing_rows = [row for row in rows if row_timing_label(row) == timing_label]
        timing = representative_timing(timing_rows)
        if timing is not None:
            x0, x1 = perturbation_interval_bounds(timing)
            for row_index in range(1, len(QUANTITY_SPECS) + 1):
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="rgba(234,179,8,0.18)",
                    line={"color": "rgba(120,80,0,0.55)", "width": 1, "dash": "dot"},
                    layer="below",
                    row=row_index,
                    col=col_index,
                    exclude_empty_subplots=False,
                )
        traces = collect_traces_for_rows(
            timing_rows,
            run=run,
            extlqg_context=extlqg_context,
            figure_kind=figure_kind,
        )
        for row_index, (quantity, _quantity_name, unit) in enumerate(QUANTITY_SPECS, start=1):
            for source in ("gru", "extlqg6d"):
                variants = ("clean", "perturbed") if figure_kind == "trajectory" else ("residual",)
                for variant in variants:
                    for coord in ("orthogonal", "along"):
                        samples = traces.get((source, variant, quantity, coord))
                        if samples is None or samples.size == 0:
                            continue
                        add_profile_trace(
                            fig,
                            samples,
                            source=source,
                            variant=variant,
                            quantity=quantity,
                            coord=coord,
                            figure_kind=figure_kind,
                            row=row_index,
                            col=col_index,
                            showlegend=(source, variant, coord) not in legend_seen,
                        )
                        legend_seen.add((source, variant, coord))
            fig.update_yaxes(
                title_text=axis_unit(quantity, figure_kind=figure_kind, native_unit=unit),
                row=row_index,
                col=1,
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=max(980, 280 * n_cols),
        height=840,
        legend_title_text="Source / trace",
        margin={"l": 70, "r": 24, "t": 96, "b": 70},
    )
    for col_index in range(1, n_cols + 1):
        fig.update_xaxes(title_text="time from movement onset (s)", row=len(QUANTITY_SPECS), col=col_index)
    return fig


def collect_traces_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    figure_kind: Literal["trajectory", "residual"],
) -> dict[tuple[str, str, str, str], np.ndarray]:
    trace_samples: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        sign = int(row.get("sign") or row.get("perturbation", {}).get("sign") or 1)
        path_text = run.get("bulk_files", {}).get(row["perturbation_id"])
        if path_text:
            with np.load(REPO_ROOT / path_text) as arrays:
                append_source_samples(
                    trace_samples,
                    arrays,
                    source="gru",
                    sign=sign,
                    figure_kind=figure_kind,
                )
        if row.get("extlqg_comparator", {}).get("status") == "available":
            try:
                ext_arrays = simulate_extlqg_arrays(row["perturbation"], extlqg_context)
            except ValueError:
                ext_arrays = None
            if ext_arrays is not None:
                append_source_samples(
                    trace_samples,
                    ext_arrays,
                    source="extlqg6d",
                    sign=sign,
                    figure_kind=figure_kind,
                )
    return {
        key: np.concatenate(samples, axis=0)
        for key, samples in trace_samples.items()
        if samples
    }


def append_source_samples(
    trace_samples: dict[tuple[str, str, str, str], list[np.ndarray]],
    arrays: Mapping[str, np.ndarray],
    *,
    source: str,
    sign: int,
    figure_kind: Literal["trajectory", "residual"],
) -> None:
    directions, orthogonals, base_start = clean_reach_basis(arrays["base_position"])
    position_base = as_samples(arrays["base_position"]) - base_start[:, None, :]
    position_delta = as_samples(arrays["delta_position"])
    velocity_base = as_samples(arrays["base_velocity"])
    velocity_delta = as_samples(arrays["delta_velocity"])
    command_base = as_samples(arrays["base_action"])
    command_delta = as_samples(arrays["delta_action"])
    vectors = {
        "position": (position_base, position_delta),
        "velocity": (velocity_base, velocity_delta),
        "command": (command_base, command_delta),
    }
    coord_vectors = {"along": directions, "orthogonal": orthogonals}
    for quantity, (base, delta) in vectors.items():
        for coord, basis in coord_vectors.items():
            base_projected = project_samples(base, basis)
            residual_projected = float(sign) * project_samples(delta, basis)
            if figure_kind == "trajectory":
                trace_samples[(source, "clean", quantity, coord)].append(base_projected)
                trace_samples[(source, "perturbed", quantity, coord)].append(
                    base_projected + residual_projected
                )
            else:
                trace_samples[(source, "residual", quantity, coord)].append(residual_projected)


def simulate_extlqg_arrays(
    perturbation: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    base = context["base_evaluation"]
    perturbed, _initial_state, _adapter = _simulate_extlqg_perturbed(
        perturbation,
        context=context,
    )
    return {
        "base_position": np.asarray(base.position, dtype=np.float64),
        "delta_position": np.asarray(perturbed.position - base.position, dtype=np.float64),
        "base_velocity": np.asarray(base.velocity, dtype=np.float64),
        "delta_velocity": np.asarray(perturbed.velocity - base.velocity, dtype=np.float64),
        "base_action": np.asarray(base.command, dtype=np.float64),
        "delta_action": np.asarray(perturbed.command - base.command, dtype=np.float64),
    }


def materialize_nominal_velocity_profiles(
    *,
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = read_json(EVAL_MANIFEST)
    robust_contract = robust_context["contract"]
    fig = profile_comparison_grid(
        n_panels=len(RUN_ORDER),
        rows=len(RUN_ORDER),
        cols=1,
        subplot_titles=[RUN_LABELS[run_id] for run_id in RUN_ORDER],
        vertical_spacing=0.018,
    )
    ext_profile = forward_velocity_profile(extlqg_context["base_evaluation"].velocity)
    robust_profile = forward_velocity_profile(robust_context["velocity"])
    for row_index, run_id in enumerate(RUN_ORDER, start=1):
        run = manifest["runs"][run_id]
        with np.load(REPO_ROOT / run["bulk_arrays"]["path"]) as arrays:
            gru_samples = forward_velocity_profile(arrays["velocity"])
        add_mean_band(
            fig,
            gru_samples,
            row=row_index,
            col=1,
            name="GRU nominal",
            color=SOURCE_COLORS["gru"],
            showlegend=row_index == 1,
        )
        add_line(
            fig,
            ext_profile,
            row=row_index,
            col=1,
            name="6D extLQG",
            color=SOURCE_COLORS["extlqg6d"],
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
            color=SOURCE_COLORS["robust_output_feedback6d"],
            dash="dot",
            showlegend=row_index == 1,
            width=2.8,
        )
        fig.update_yaxes(title_text="m/s", row=row_index, col=1)
    fig.update_xaxes(title_text="time from movement onset (s)", row=len(RUN_ORDER), col=1)
    fig.update_layout(
        title="c92 nominal forward velocity profiles",
        template="plotly_white",
        width=1040,
        height=1500,
        legend_title_text="profile",
        margin={"l": 78, "r": 24, "t": 90, "b": 70},
    )
    spec = {
        "schema_version": "rlrmp.c92_nominal_velocity_profiles.v1",
        "issue": ISSUE,
        "figure_kind": "nominal_forward_velocity_profile_comparison",
        "analytical_comparator_contract": {
            "extlqg": {
                "label": "6D extLQG",
                "state_dim": 36,
                "physical_dim": 6,
                "disturbance_integrators_exposed": False,
                "source": "rlrmp.analysis.pipelines.gru_perturbation_bank._build_extlqg_comparator_context(physical_dim=6)",
            },
            "output_feedback_hinf": robust_contract,
        },
        "inputs": (
            [{"path": repo_rel(EVAL_MANIFEST)}]
            + [{"path": f"results/{ISSUE}/runs/{run_id}.json"} for run_id in RUN_ORDER]
        ),
        "transform": [
            {
                "name": "forward_velocity_profile_mean_band",
                "kwargs": {
                    "trained_rows": list(RUN_ORDER),
                    "analytical_comparators": [
                        "6d_output_feedback_extlqg",
                        "6d_output_feedback_hinf",
                    ],
                },
            }
        ],
        "plot_kwargs": {
            "shared_yaxes": "all",
            "rows": len(RUN_ORDER),
            "physical_state_dim": 6,
            "state_dim": 36,
            "disturbance_integrators_exposed": False,
        },
    }
    saved = save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=ISSUE,
        topic=NOMINAL_TOPIC,
        extra_packages=["rlrmp"],
    )
    return {
        "status": "materialized",
        "topic": NOMINAL_TOPIC,
        "save_result": json_safe(saved),
        "spec": f"results/{ISSUE}/figures/{NOMINAL_TOPIC}/spec.json",
        "html": f"results/{ISSUE}/figures/{NOMINAL_TOPIC}/figure.html",
        "analytical_comparator_contract": {
            "output_feedback_hinf": robust_contract,
        },
    }


def build_robust_output_feedback_6d_context() -> dict[str, Any]:
    """Build the requested 6D no-integrator output-feedback H-infinity nominal path."""

    plant, schedule = build_no_integrator_game()
    config = OutputFeedbackConfig(n_phys=6)
    gamma_star = find_gamma_star(plant, schedule)
    solution = solve_hinf_riccati(
        plant,
        schedule,
        OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR * gamma_star,
    )
    covariances = robust_estimator_covariances(
        plant,
        schedule,
        solution.gamma,
        config,
    )
    gains = robust_output_feedback_gains(
        plant,
        schedule,
        solution,
        covariances,
        config,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    rollout = simulate_robust_released_forward(
        plant,
        schedule,
        solution,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        gains=gains,
        config=config,
    )
    velocity = np.asarray(rollout.x[1:, 2:4], dtype=np.float64)
    command = np.asarray(rollout.u_command, dtype=np.float64)
    contract = {
        "label": "6D output-feedback H-infinity",
        "state_dim": int(plant.n),
        "physical_dim": int(config.n_phys),
        "disturbance_dim": int(plant.m_w),
        "control_dim": int(plant.m_u),
        "delay_steps": int(config.delay_steps),
        "disturbance_integrators_exposed": False,
        "game_source": "rlrmp.analysis.math.cs_game_card.build_no_integrator_game",
        "config": "rlrmp.analysis.math.output_feedback.OutputFeedbackConfig(n_phys=6)",
        "gamma_factor": float(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR),
        "gamma_star": float(gamma_star),
        "gamma": float(solution.gamma),
        "admissible": bool(solution.admissible),
        "nominal_profile_convention": "zero-noise released-forward rollout, x[1:] velocity to match 60-step GRU diagnostics",
    }
    if contract["state_dim"] != 36 or contract["physical_dim"] != 6:
        raise ValueError(f"unexpected 6D H-infinity contract: {contract}")
    return {
        "velocity": velocity[None, None, :, :],
        "command": command[None, None, :, :],
        "contract": contract,
    }


def forward_velocity_profile(velocity: Any) -> np.ndarray:
    samples = as_samples(np.asarray(velocity, dtype=np.float64))
    return samples[..., 0]


def add_mean_band(
    fig: go.Figure,
    samples: np.ndarray,
    *,
    row: int,
    col: int,
    name: str,
    color: str,
    showlegend: bool,
) -> None:
    mean, low, high = mean_band(samples)
    time = np.arange(mean.shape[0], dtype=np.float64) * DT
    fig.add_trace(
        go.Scatter(
            x=time,
            y=high,
            mode="lines",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=low,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.12)",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )
    add_line(
        fig,
        mean,
        row=row,
        col=col,
        name=name,
        color=color,
        dash="solid",
        showlegend=showlegend,
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


def add_profile_trace(
    fig: go.Figure,
    samples: np.ndarray,
    *,
    source: str,
    variant: str,
    quantity: str,
    coord: str,
    figure_kind: Literal["trajectory", "residual"],
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    samples = scale_profile_samples(samples, quantity=quantity, figure_kind=figure_kind)
    mean, low, high = mean_band(samples)
    time = np.arange(mean.shape[0], dtype=np.float64) * DT
    color = SOURCE_COLORS[source]
    dash = COORD_DASH[coord]
    label = f"{source_label(source)} {variant} {coord}"
    if samples.shape[0] > 1:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=high,
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"{source}-{variant}-{coord}",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=low,
                mode="lines",
                fill="tonexty",
                fillcolor=band_color(source),
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"{source}-{variant}-{coord}",
            ),
            row=row,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=mean,
            mode="lines",
            name=label,
            legendgroup=f"{source}-{variant}-{coord}",
            showlegend=showlegend,
            line={"color": color, "dash": dash, "width": 1.25 if variant == "clean" else 2.25},
            opacity=0.42 if variant == "clean" else 0.95,
        ),
        row=row,
        col=col,
    )


def clean_reach_basis(base_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = as_samples(base_position)
    start = base[:, 0, :]
    target = base[:, -1, :]
    displacement = target - start
    length = np.linalg.norm(displacement, axis=-1, keepdims=True)
    fallback = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (base.shape[0], 1))
    direction = np.divide(displacement, length, out=fallback.copy(), where=length > 1e-9)
    orthogonal = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)
    return direction, orthogonal, start


def as_samples(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 2:
        raise ValueError(f"expected array with trailing shape (time, 2), got {array.shape}")
    return array.reshape((-1, array.shape[-2], 2))


def project_samples(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("sti,si->st", as_samples(values), basis)


def scale_profile_samples(
    samples: np.ndarray,
    *,
    quantity: str,
    figure_kind: Literal["trajectory", "residual"],
) -> np.ndarray:
    if figure_kind != "residual":
        return samples
    if quantity == "position":
        return 100.0 * samples / REACH_LENGTH_M
    if quantity == "velocity":
        return 100.0 * samples / nominal_extlqg_peak_velocity()
    return samples


@lru_cache(maxsize=1)
def nominal_extlqg_peak_velocity() -> float:
    context = _build_extlqg_comparator_context(physical_dim=6)
    speed = np.linalg.norm(np.asarray(context["base_evaluation"].velocity), axis=-1)
    peak = float(np.nanmax(speed))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"nominal extLQG peak velocity must be positive; got {peak}")
    return peak


def axis_unit(
    quantity: str,
    *,
    figure_kind: Literal["trajectory", "residual"],
    native_unit: str,
) -> str:
    if figure_kind == "residual" and quantity in {"position", "velocity"}:
        return "%"
    return native_unit


def mean_band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    if array.shape[0] <= 1:
        return mean, mean, mean
    return mean, np.nanpercentile(array, 10.0, axis=0), np.nanpercentile(array, 90.0, axis=0)


def figure_group_key(row: Mapping[str, Any]) -> str:
    perturbation = row.get("perturbation", {})
    family = str(row.get("family") or perturbation.get("family"))
    channel = str(row.get("channel") or perturbation.get("channel"))
    provenance = perturbation.get("channel_provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    parts = [channel, family]
    for key in ("feedback_quantity", "target_relative_axis_role"):
        value = perturbation.get(key) or provenance.get(key)
        if value is not None:
            parts.append(str(value))
    epsilon_component = perturbation.get("epsilon_component")
    if epsilon_component is not None:
        parts.append(str(epsilon_component))
    return "__".join(parts)


def figure_title(row: Mapping[str, Any]) -> str:
    return figure_group_key(row).replace("__", " / ").replace("_", " ")


def timing_bins_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({row_timing_label(row) for row in rows}, key=timing_sort_key) or ["none"]


def row_timing_label(row: Mapping[str, Any]) -> str:
    timing = row.get("timing") or row.get("perturbation", {}).get("timing") or {}
    return str(row.get("timing_bin") or timing.get("timing_bin") or "movement_onset")


def timing_sort_key(label: str) -> tuple[int, str]:
    return (TIMING_ORDER.get(label, 50), label)


def representative_timing(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        timing = row.get("timing") or row.get("perturbation", {}).get("timing")
        if isinstance(timing, Mapping):
            return timing
    return None


def perturbation_interval_bounds(timing: Mapping[str, Any]) -> tuple[float, float]:
    """Return sample-interval bounds for a timed perturbation band."""

    start = int(timing.get("start_time_index", timing.get("time_index", 0)))
    duration = int(timing.get("duration_steps", 1))
    if duration < 1:
        raise ValueError(f"perturbation duration must be positive; got {duration}")
    left = max(0.0, (float(start) - 0.5) * DT)
    right = (float(start + duration) - 0.5) * DT
    return left, max(right, left + DT)


def source_label(source: str) -> str:
    if source == "extlqg6d":
        return "6D extLQG"
    if source == "robust_output_feedback6d":
        return "6D output-feedback H-infinity"
    return "GRU"


def band_color(source: str) -> str:
    if source == "extlqg6d":
        return "rgba(194,65,12,0.10)"
    if source == "robust_output_feedback6d":
        return "rgba(21,128,61,0.10)"
    return "rgba(37,99,235,0.10)"


def write_note(outputs: Mapping[str, Any]) -> None:
    lines = [
        "# c92 Post-Training Analysis",
        "",
        "- Scope: nine no-PGD calibrated perturbation rows after 12000/12000 batches.",
        "- Physical level for perturbation profiles: `moderate`.",
        "- Analytical comparators: 6D no-integrator extLQG and output-feedback H-infinity.",
        "- Robustness phenotype outputs are interpretive diagnostics, not formal H-infinity certificates.",
        "",
    ]
    if "moderate_perturbation_profiles" in outputs:
        profile = outputs["moderate_perturbation_profiles"]
        lines.extend(
            [
                "## Moderate Perturbation Profiles",
                "",
                f"- Status: `{profile['status']}`.",
                f"- Figure spec: `{profile['spec']}`.",
                f"- HTML render directory: `{profile['bulk_dir']}`.",
                f"- Figure count: `{profile['figure_count']}`.",
                "",
            ]
        )
    if "nominal_velocity_profiles" in outputs:
        nominal = outputs["nominal_velocity_profiles"]
        hinf_contract = nominal.get("analytical_comparator_contract", {}).get(
            "output_feedback_hinf",
            {},
        )
        lines.extend(
            [
                "## Nominal Velocity Profiles",
                "",
                f"- Status: `{nominal['status']}`.",
                f"- Figure spec: `{nominal['spec']}`.",
                f"- Navigable HTML link: `{nominal['html']}`.",
                "- H-infinity comparator: 6D no-integrator output-feedback path "
                f"(`state_dim={hinf_contract.get('state_dim')}`, "
                f"`physical_dim={hinf_contract.get('physical_dim')}`, "
                f"`disturbance_integrators_exposed={str(hinf_contract.get('disturbance_integrators_exposed')).lower()}`).",
                "",
            ]
        )
    lines.extend(
        [
            "## Diagnostic Inputs",
            "",
            f"- Evaluation diagnostics: `{repo_rel(EVAL_MANIFEST)}`.",
            f"- Feedback-quality diagnostics: `{repo_rel(FEEDBACK_ABLATION)}`.",
            f"- Perturbation response manifest: `{repo_rel(PERT_MANIFEST)}`.",
            f"- Robustness phenotype sidecar: `{repo_rel(PHENOTYPE_SIDECAR)}`.",
            f"- Post-run materialization manifest: `{repo_rel(POSTRUN_MANIFEST)}`.",
            "",
        ]
    )
    update_marked_section(NOTE_PATH, "c92_post_training_analysis", "\n".join(lines) + "\n")


def load_detail_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path_text = manifest.get("bulk_detail_manifest")
    if isinstance(path_text, Mapping):
        path_text = path_text["path"]
    if path_text is None:
        return dict(manifest)
    return read_json(REPO_ROOT / str(path_text))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def moderate_profile_plot_contract() -> dict[str, Any]:
    return {
        "grid_helper": "rlrmp.viz.profile_comparison_grid",
        "shared_yaxes": "rows",
        "residual_scaling": {
            "position": "percent of fixed 0.15 m reach length",
            "velocity": "percent of nominal 6D extLQG peak speed",
            "command": "native command units",
        },
        "sign_handling": (
            "perturbation sign is multiplied into residuals; trajectory figures "
            "plot clean + sign-aligned residual so + and - rows do not cancel"
        ),
        "band": "central 80% interval across replicate/trial/component/sign rows",
        "timing_band": (
            "trace x values are sample centers (sample i at i*dt); perturbation "
            "shading uses sample-interval edges [start_i-0.5, start_i+duration-0.5]*dt, "
            "clamped at zero for initial-condition rows"
        ),
        "comparator_contract": {
            "extlqg": {
                "state_dim": 36,
                "physical_dim": 6,
                "disturbance_integrators_exposed": False,
                "source": "rlrmp.analysis.pipelines.gru_perturbation_bank._build_extlqg_comparator_context(physical_dim=6)",
            }
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return repo_rel(value)
        except ValueError:
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
