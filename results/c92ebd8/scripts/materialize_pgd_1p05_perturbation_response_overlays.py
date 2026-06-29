#!/usr/bin/env python
"""Materialize c92 PGD 1.05 perturbation-response overlay figures."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

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
    _evaluation_from_extlqg_rollout,
    _simulate_extlqg_perturbed,
    _simulate_robust_output_feedback_perturbed,
    materialize_gru_perturbation_response,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.viz import profile_comparison_grid


ISSUE = "c92ebd8"
SOURCE_TOPIC = "moderate_perturbation_profiles"
TOPIC = "pgd_1p05_moderate_perturbation_response_overlays"
DT = 0.01
REACH_LENGTH_M = 0.15
SOURCE_SPEC = REPO_ROOT / "results" / ISSUE / "figures" / SOURCE_TOPIC / "spec.json"
SOURCE_DETAIL = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE
    / "perturbation_response"
    / "gru_validation_selected_moderate"
    / "gru_perturbation_response_validation_selected_moderate_manifest_detail.json"
)
FIGURE_SPEC = REPO_ROOT / "results" / ISSUE / "figures" / TOPIC / "spec.json"
NOTE_PATH = REPO_ROOT / "results" / ISSUE / "notes" / f"{TOPIC}.md"
BULK_ROOT = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC
PGD_BULK_ROOT = BULK_ROOT / "pgd_perturbation_response"
PGD_MANIFEST = BULK_ROOT / "pgd_perturbation_response_manifest.json"
PGD_NOTE = BULK_ROOT / "pgd_perturbation_response.md"
PGD_REGENERATION_SPEC = BULK_ROOT / "pgd_perturbation_response_manifest_regeneration_spec.json"
PGD_EVALUATION_MANIFEST = (
    REPO_ROOT / "results" / ISSUE / "notes" / "gru_evaluation_diagnostics_pgd_1p05_reach_context_diagnostics.json"
)

NO_PGD_RUNS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
PGD_RUN_BY_NO_PGD = {
    "open_loop_small": "small",
    "open_loop_moderate": "moderate",
    "open_loop_stress": "stress",
}
RUN_LABELS = {
    "open_loop_small": "small",
    "open_loop_moderate": "moderate",
    "open_loop_stress": "stress",
}
PGD_LABELS = {
    "small": "PGD 1.05 small",
    "moderate": "PGD 1.05 moderate",
    "stress": "PGD 1.05 stress",
}

SOURCE_COLORS = {
    "no_pgd_gru": "#2563eb",
    "extlqg6d": "#c2410c",
    "pgd_gru": "#7c3aed",
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
    """CLI entry point."""

    args = build_parser().parse_args()
    source_spec = read_json(SOURCE_SPEC)
    source_detail = read_json(SOURCE_DETAIL)
    pgd_detail = ensure_pgd_bulk_arrays(force=args.force_pgd_bulk)
    extlqg_context = _build_extlqg_comparator_context(physical_dim=6)
    robust_context = build_robust_output_feedback_6d_context()
    output = materialize_overlay_figures(
        source_spec=source_spec,
        source_detail=source_detail,
        pgd_detail=pgd_detail,
        extlqg_context=extlqg_context,
        robust_context=robust_context,
        limit=args.limit,
    )
    write_note(output)
    print(json.dumps(output, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-pgd-bulk",
        action="store_true",
        help="Regenerate figure-owned PGD perturbation-response bulk arrays.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug limit on number of residual figures to render.",
    )
    return parser


def ensure_pgd_bulk_arrays(*, force: bool = False) -> dict[str, Any]:
    """Return PGD perturbation-response detail with bulk arrays available."""

    if not force and PGD_MANIFEST.exists():
        manifest = read_json(PGD_MANIFEST)
        detail_ref = manifest.get("bulk_detail_manifest", {})
        detail_path = REPO_ROOT / detail_ref.get("path", "")
        if detail_path.exists():
            detail = read_json(detail_path)
            if pgd_detail_has_bulk_arrays(detail):
                return detail

    mkdir_p(BULK_ROOT)
    manifest = materialize_gru_perturbation_response(
        source_experiment=ISSUE,
        result_experiment=ISSUE,
        run_ids=tuple(PGD_RUN_BY_NO_PGD.values()),
        labels=tuple(PGD_LABELS[run_id] for run_id in PGD_RUN_BY_NO_PGD.values()),
        n_rollout_trials=64,
        output_path=PGD_MANIFEST,
        note_path=PGD_NOTE,
        bulk_dir=PGD_BULK_ROOT,
        regeneration_spec_path=PGD_REGENERATION_SPEC,
        bank_mode="calibrated",
        calibration_level="moderate",
        calibration_reach=REACH_LENGTH_M,
        feedback_scale_manifest_path=PGD_EVALUATION_MANIFEST,
        extlqg_physical_dim=6,
        write_bulk_arrays=True,
        repo_root=REPO_ROOT,
    )
    detail_path = REPO_ROOT / manifest["bulk_detail_manifest"]["path"]
    return read_json(detail_path)


def pgd_detail_has_bulk_arrays(detail: Mapping[str, Any]) -> bool:
    """Return whether every requested PGD row has evaluated row bulk arrays."""

    for pgd_run in PGD_RUN_BY_NO_PGD.values():
        run = detail.get("runs", {}).get(pgd_run, {})
        bulk_files = run.get("bulk_files", {})
        evaluated = [
            row
            for row in run.get("perturbations", [])
            if row.get("status") == "evaluated"
            and row.get("perturbation", {}).get("level_name") == "moderate"
        ]
        if not evaluated or len(bulk_files) < len(evaluated):
            return False
    return True


def materialize_overlay_figures(
    *,
    source_spec: Mapping[str, Any],
    source_detail: Mapping[str, Any],
    pgd_detail: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    limit: int | None = None,
) -> dict[str, Any]:
    """Render residual overlay figures for the three open-loop no-PGD rows."""

    figure_specs = []
    robust_unavailable: dict[str, int] = defaultdict(int)
    source_figures = [
        figure
        for figure in source_spec.get("figures", [])
        if figure.get("run_id") in NO_PGD_RUNS and figure.get("figure_kind") == "residual"
    ]
    if limit is not None:
        source_figures = source_figures[:limit]

    for source_figure in source_figures:
        no_pgd_run_id = str(source_figure["run_id"])
        pgd_run_id = PGD_RUN_BY_NO_PGD[no_pgd_run_id]
        group_key = str(source_figure["group_key"])
        no_rows = group_rows(source_detail["runs"][no_pgd_run_id], group_key)
        pgd_rows = group_rows(pgd_detail["runs"][pgd_run_id], group_key)
        if not no_rows:
            raise ValueError(f"source group {group_key!r} has no rows for {no_pgd_run_id}")
        if not pgd_rows:
            raise ValueError(f"PGD group {group_key!r} has no rows for {pgd_run_id}")
        fig, figure_robust_unavailable = build_overlay_figure(
            no_rows,
            pgd_rows=pgd_rows,
            no_pgd_run=source_detail["runs"][no_pgd_run_id],
            pgd_run=pgd_detail["runs"][pgd_run_id],
            extlqg_context=extlqg_context,
            robust_context=robust_context,
            title=(
                f"{RUN_LABELS[no_pgd_run_id]}: {figure_title(no_rows[0])}: "
                "moderate residual PGD/H-inf overlay"
            ),
        )
        for key, count in figure_robust_unavailable.items():
            robust_unavailable[key] += count
        path = BULK_ROOT / no_pgd_run_id / f"{safe_slug(group_key)}__residual_overlay.html"
        mkdir_p(path.parent)
        fig.write_html(path, include_plotlyjs="cdn")
        figure_specs.append(
            {
                "source_run_id": no_pgd_run_id,
                "pgd_run_id": pgd_run_id,
                "physical_level": RUN_LABELS[no_pgd_run_id],
                "group_key": group_key,
                "title": source_figure.get("title"),
                "figure_kind": "residual_overlay",
                "source_html": source_figure.get("html"),
                "html": repo_rel(path),
                "n_source_rows": len(no_rows),
                "n_pgd_rows": len(pgd_rows),
                "robust_output_feedback_unavailable": dict(figure_robust_unavailable),
            }
        )

    spec = {
        "schema_version": "rlrmp.c92_pgd_1p05_perturbation_response_overlays.v1",
        "issue": ISSUE,
        "source_figure_topic": SOURCE_TOPIC,
        "figure_kind": "moderate_residual_perturbation_response_overlay",
        "source_spec": repo_rel(SOURCE_SPEC),
        "source_detail_manifest": repo_rel(SOURCE_DETAIL),
        "pgd_detail_manifest": PGD_MANIFEST_DETAIL_REL(),
        "plot_contract": overlay_plot_contract(robust_context),
        "figure_count": len(figure_specs),
        "figures": figure_specs,
        "robust_output_feedback_unavailable": dict(robust_unavailable),
    }
    mkdir_p(FIGURE_SPEC.parent)
    FIGURE_SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return {
        "status": "materialized",
        "topic": TOPIC,
        "spec": repo_rel(FIGURE_SPEC),
        "bulk_dir": repo_rel(BULK_ROOT),
        "figure_count": len(figure_specs),
        "robust_output_feedback_unavailable": dict(robust_unavailable),
    }


def PGD_MANIFEST_DETAIL_REL() -> str:
    """Return the repository-relative PGD bulk detail manifest path."""

    manifest = read_json(PGD_MANIFEST)
    return str(manifest["bulk_detail_manifest"]["path"])


def build_overlay_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    pgd_rows: Sequence[Mapping[str, Any]],
    no_pgd_run: Mapping[str, Any],
    pgd_run: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    title: str,
) -> tuple[go.Figure, dict[str, int]]:
    """Build one residual overlay figure."""

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
    legend_seen: set[tuple[str, str]] = set()
    robust_unavailable: dict[str, int] = defaultdict(int)
    for col_index, timing_label in enumerate(timing_bins, start=1):
        timing_rows = [row for row in rows if row_timing_label(row) == timing_label]
        timing_pgd_rows = [row for row in pgd_rows if row_timing_label(row) == timing_label]
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
        traces, unavailable = collect_overlay_traces(
            timing_rows,
            pgd_rows=timing_pgd_rows,
            no_pgd_run=no_pgd_run,
            pgd_run=pgd_run,
            extlqg_context=extlqg_context,
            robust_context=robust_context,
        )
        for key, count in unavailable.items():
            robust_unavailable[key] += count
        for row_index, (quantity, _quantity_name, unit) in enumerate(QUANTITY_SPECS, start=1):
            for source in ("no_pgd_gru", "extlqg6d", "pgd_gru", "robust_output_feedback6d"):
                for coord in ("orthogonal", "along"):
                    samples = traces.get((source, quantity, coord))
                    if samples is None or samples.size == 0:
                        continue
                    add_residual_trace(
                        fig,
                        samples,
                        source=source,
                        quantity=quantity,
                        coord=coord,
                        row=row_index,
                        col=col_index,
                        showlegend=(source, coord) not in legend_seen,
                    )
                    legend_seen.add((source, coord))
            fig.update_yaxes(
                title_text=residual_axis_unit(quantity, native_unit=unit),
                row=row_index,
                col=1,
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=max(980, 280 * n_cols),
        height=840,
        legend_title_text="Source / component",
        margin={"l": 70, "r": 24, "t": 96, "b": 70},
    )
    for col_index in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="time from movement onset (s)",
            row=len(QUANTITY_SPECS),
            col=col_index,
        )
    return fig, dict(robust_unavailable)


def collect_overlay_traces(
    rows: Sequence[Mapping[str, Any]],
    *,
    pgd_rows: Sequence[Mapping[str, Any]],
    no_pgd_run: Mapping[str, Any],
    pgd_run: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[str, int]]:
    """Collect no-PGD, extLQG, PGD, and 6D H-inf residual samples."""

    trace_samples: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    unavailable: dict[str, int] = defaultdict(int)
    pgd_by_id = {str(row["perturbation_id"]): row for row in pgd_rows}
    for row in rows:
        perturbation_id = str(row["perturbation_id"])
        sign = int(row.get("sign") or row.get("perturbation", {}).get("sign") or 1)
        append_run_bulk_samples(
            trace_samples,
            row,
            run=no_pgd_run,
            source="no_pgd_gru",
            sign=sign,
        )
        pgd_row = pgd_by_id.get(perturbation_id)
        if pgd_row is not None:
            append_run_bulk_samples(
                trace_samples,
                pgd_row,
                run=pgd_run,
                source="pgd_gru",
                sign=sign,
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
                )
        robust_arrays, reason = simulate_robust_output_feedback_arrays(
            row["perturbation"],
            robust_context,
        )
        if robust_arrays is None:
            unavailable[reason] += 1
        else:
            append_source_samples(
                trace_samples,
                robust_arrays,
                source="robust_output_feedback6d",
                sign=sign,
            )
    return (
        {
            key: np.concatenate(samples, axis=0)
            for key, samples in trace_samples.items()
            if samples
        },
        dict(unavailable),
    )


def append_run_bulk_samples(
    trace_samples: dict[tuple[str, str, str], list[np.ndarray]],
    row: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    source: str,
    sign: int,
) -> None:
    """Append residual samples from a run bulk NPZ."""

    path_text = run.get("bulk_files", {}).get(row["perturbation_id"])
    if not path_text:
        raise ValueError(f"missing bulk file for {source} row {row['perturbation_id']}")
    with np.load(REPO_ROOT / path_text) as arrays:
        append_source_samples(trace_samples, arrays, source=source, sign=sign)


def append_source_samples(
    trace_samples: dict[tuple[str, str, str], list[np.ndarray]],
    arrays: Mapping[str, np.ndarray],
    *,
    source: str,
    sign: int,
) -> None:
    """Append sign-aligned residual projections for one source."""

    directions, orthogonals, base_start = clean_reach_basis(arrays["base_position"])
    position_delta = as_samples(arrays["delta_position"])
    velocity_delta = as_samples(arrays["delta_velocity"])
    command_delta = as_samples(arrays["delta_action"])
    vectors = {
        "position": position_delta,
        "velocity": velocity_delta,
        "command": command_delta,
    }
    coord_vectors = {"along": directions, "orthogonal": orthogonals}
    _ = base_start
    for quantity, delta in vectors.items():
        for coord, basis in coord_vectors.items():
            residual_projected = float(sign) * project_samples(delta, basis)
            trace_samples[(source, quantity, coord)].append(residual_projected)


def simulate_extlqg_arrays(
    perturbation: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return deterministic 6D extLQG arrays for one perturbation."""

    base = context["base_evaluation"]
    perturbed, _initial_state, _adapter = _simulate_extlqg_perturbed(
        perturbation,
        context=context,
    )
    return evaluation_pair_arrays(base, perturbed)


def simulate_robust_output_feedback_arrays(
    perturbation: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray] | None, str]:
    """Return deterministic 6D H-inf arrays when the perturbation is supported."""

    if str(perturbation.get("channel")) not in {"initial_state", "command_input", "process_epsilon"}:
        return None, f"unsupported_channel:{perturbation.get('channel')}"
    try:
        base = context["base_evaluation"]
        perturbed, _initial_state, _adapter = _simulate_robust_output_feedback_perturbed(
            perturbation,
            context=context,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        return None, f"blocked:{type(exc).__name__}"
    return evaluation_pair_arrays(base, perturbed), ""


def evaluation_pair_arrays(base: Any, perturbed: Any) -> dict[str, np.ndarray]:
    """Convert two rollout evaluations to the bulk array schema."""

    return {
        "base_position": np.asarray(base.position, dtype=np.float64),
        "delta_position": np.asarray(perturbed.position - base.position, dtype=np.float64),
        "base_velocity": np.asarray(base.velocity, dtype=np.float64),
        "delta_velocity": np.asarray(perturbed.velocity - base.velocity, dtype=np.float64),
        "base_action": np.asarray(base.command, dtype=np.float64),
        "delta_action": np.asarray(perturbed.command - base.command, dtype=np.float64),
    }


def build_robust_output_feedback_6d_context() -> dict[str, Any]:
    """Build deterministic 6D no-integrator output-feedback H-infinity context."""

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
    base_rollout = simulate_robust_released_forward(
        plant,
        schedule,
        solution,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        gains=gains,
        config=config,
    )
    if int(plant.n) != 36 or int(config.n_phys) != 6:
        raise ValueError(f"unexpected H-inf context dimensions: plant.n={plant.n}, n_phys={config.n_phys}")
    return {
        "plant": plant,
        "schedule": schedule,
        "config": config,
        "solution": solution,
        "gains": gains,
        "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "gamma": float(solution.gamma),
        "gamma_star": float(gamma_star),
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": _evaluation_from_extlqg_rollout(base_rollout, initial_state=x0),
        "contract": {
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
        },
    }


def clean_reach_basis(base_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return along/orthogonal reach bases for sample trajectories."""

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
    """Flatten leading dimensions into samples for time-by-xy arrays."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 2:
        raise ValueError(f"expected array with trailing shape (time, 2), got {array.shape}")
    return array.reshape((-1, array.shape[-2], 2))


def project_samples(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project sample trajectories onto per-sample basis vectors."""

    return np.einsum("sti,si->st", as_samples(values), basis)


def add_residual_trace(
    fig: go.Figure,
    samples: np.ndarray,
    *,
    source: str,
    quantity: str,
    coord: str,
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    """Add one residual component trace with mean and central 80 percent band."""

    scaled = scale_residual_samples(samples, quantity=quantity)
    mean, low, high = mean_band(scaled)
    time = np.arange(mean.shape[0], dtype=np.float64) * DT
    color = SOURCE_COLORS[source]
    legendgroup = f"{source}-residual-{coord}"
    if scaled.shape[0] > 1:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=high,
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=legendgroup,
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
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=mean,
            mode="lines",
            name=f"{source_label(source)} {coord}",
            legendgroup=legendgroup,
            showlegend=showlegend,
            line={"color": color, "dash": COORD_DASH[coord], "width": 2.2},
            opacity=0.95,
        ),
        row=row,
        col=col,
    )


def scale_residual_samples(samples: np.ndarray, *, quantity: str) -> np.ndarray:
    """Scale residual samples to the existing profile convention."""

    if quantity == "position":
        return 100.0 * samples / REACH_LENGTH_M
    if quantity == "velocity":
        return 100.0 * samples / nominal_extlqg_peak_velocity()
    return samples


def nominal_extlqg_peak_velocity() -> float:
    """Return nominal 6D extLQG peak speed for residual velocity scaling."""

    context = _build_extlqg_comparator_context(physical_dim=6)
    speed = np.linalg.norm(np.asarray(context["base_evaluation"].velocity), axis=-1)
    peak = float(np.nanmax(speed))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"nominal extLQG peak velocity must be positive; got {peak}")
    return peak


def mean_band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and central 80 percent interval over sample rows."""

    array = np.asarray(samples, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    if array.shape[0] <= 1:
        return mean, mean, mean
    return mean, np.nanpercentile(array, 10.0, axis=0), np.nanpercentile(array, 90.0, axis=0)


def residual_axis_unit(quantity: str, *, native_unit: str) -> str:
    """Return residual axis unit."""

    if quantity in {"position", "velocity"}:
        return "%"
    return native_unit


def source_label(source: str) -> str:
    """Return a plot label for a source key."""

    labels = {
        "no_pgd_gru": "No-PGD GRU",
        "pgd_gru": "PGD 1.05 GRU",
        "extlqg6d": "6D extLQG",
        "robust_output_feedback6d": "6D output-feedback H-infinity",
    }
    return labels[source]


def band_color(source: str) -> str:
    """Return transparent band color for a source key."""

    colors = {
        "no_pgd_gru": "rgba(37,99,235,0.10)",
        "pgd_gru": "rgba(124,58,237,0.10)",
        "extlqg6d": "rgba(194,65,12,0.10)",
        "robust_output_feedback6d": "rgba(21,128,61,0.10)",
    }
    return colors[source]


def group_rows(run: Mapping[str, Any], group_key: str) -> list[Mapping[str, Any]]:
    """Return evaluated moderate rows matching one figure group."""

    return [
        row
        for row in run.get("perturbations", [])
        if row.get("status") == "evaluated"
        and row.get("perturbation", {}).get("level_name") == "moderate"
        and figure_group_key(row) == group_key
    ]


def figure_group_key(row: Mapping[str, Any]) -> str:
    """Return the existing c92 perturbation-profile grouping key."""

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
    """Return a compact title fragment for a group row."""

    return figure_group_key(row).replace("__", " / ").replace("_", " ")


def timing_bins_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return sorted timing bins present in a group."""

    return sorted({row_timing_label(row) for row in rows}, key=timing_sort_key) or ["none"]


def row_timing_label(row: Mapping[str, Any]) -> str:
    """Return the row timing label."""

    timing = row.get("timing") or row.get("perturbation", {}).get("timing") or {}
    return str(row.get("timing_bin") or timing.get("timing_bin") or "movement_onset")


def timing_sort_key(label: str) -> tuple[int, str]:
    """Return sort key for timing labels."""

    return (TIMING_ORDER.get(label, 50), label)


def representative_timing(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return representative timing metadata for a set of rows."""

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


def overlay_plot_contract(robust_context: Mapping[str, Any]) -> dict[str, Any]:
    """Return the plot contract for the tracked spec."""

    return {
        "source_topic": SOURCE_TOPIC,
        "figure_kind": "residual_only_overlay",
        "grid_helper": "rlrmp.viz.profile_comparison_grid",
        "shared_yaxes": "rows",
        "residual_scaling": {
            "position": "percent of fixed 0.15 m reach length",
            "velocity": "percent of nominal 6D extLQG peak speed",
            "command": "native command units",
        },
        "trace_sources": {
            "first_color": "No-PGD GRU",
            "second_color": "6D extLQG",
            "third_color": "PGD 1.05 GRU",
            "fourth_color": "6D output-feedback H-infinity",
        },
        "component_traces": ["orthogonal", "along"],
        "h_infinity_contract": robust_context["contract"],
        "h_infinity_supported_channels": [
            "initial_state",
            "command_input",
            "process_epsilon",
        ],
        "h_infinity_unsupported_channels": [
            "sensory_feedback",
            "delayed_observation",
            "target_stream",
        ],
    }


def write_note(output: Mapping[str, Any]) -> None:
    """Write a concise note for the overlay figure family."""

    lines = [
        "# PGD 1.05 Perturbation-Response Overlays",
        "",
        "- Scope: residual perturbation-response figures only.",
        f"- Source figure topic: `{SOURCE_TOPIC}`.",
        "- Rows: `open_loop_small`, `open_loop_moderate`, and `open_loop_stress`, "
        "paired with PGD rows `small`, `moderate`, and `stress`.",
        "- Added traces: PGD 1.05 GRU along/orthogonal residuals and 6D "
        "output-feedback H-infinity along/orthogonal residuals where the H-inf "
        "replay supports the perturbation channel.",
        f"- Figure spec: `{output['spec']}`.",
        f"- HTML render directory: `{output['bulk_dir']}`.",
        f"- Figure count: `{output['figure_count']}`.",
        "",
    ]
    if output.get("robust_output_feedback_unavailable"):
        lines.extend(
            [
                "H-infinity trace caveat: sensory-feedback rows remain unsupported by "
                "the 6D robust released-forward replay helper, so those figures carry "
                "No-PGD, extLQG, and PGD traces only.",
                "",
            ]
        )
    update_marked_section(NOTE_PATH, TOPIC, "\n".join(lines) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    """Return a repository-relative path."""

    return str(path.relative_to(REPO_ROOT))


def safe_slug(text: str) -> str:
    """Return a filesystem-safe slug."""

    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


if __name__ == "__main__":
    main()
