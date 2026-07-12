"""Shared analytical-profile and figure assembly helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jax.random as jr
import numpy as np
import plotly.graph_objects as go

from rlrmp.analysis.math.trial_alignment import (
    align_trials,
    pooled_trial_mean_with_band,
    replicate_mean_curves,
)
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
    simulate_robust_released_forward,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati, solve_lqr
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)
from rlrmp.viz.profile_grids import profile_comparison_grid
from rlrmp.viz.colors import hex_to_rgba
from rlrmp.viz.traces import add_band_trace, add_reference_trace


_VELOCITY_COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")


class VelocityFigureProfile(Protocol):
    """Structural input contract for fixed-bank velocity figures."""

    label: str
    bank_kind: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    replicate_mean: np.ndarray
    replicate_std: np.ndarray
    n_replicates: int


def write_velocity_figure(
    profiles: VelocityFigureProfile | Sequence[VelocityFigureProfile],
    *,
    output_dir: Path,
    references: Sequence[Any],
    title: str,
) -> Path:
    """Write pooled fixed-bank profiles using single- or multi-run presentation."""

    return _write_velocity_profiles(
        profiles, output_dir=output_dir, references=references, title=title, by_replicate=False
    )


def write_velocity_by_replicate_figure(
    profiles: VelocityFigureProfile | Sequence[VelocityFigureProfile],
    *,
    output_dir: Path,
    references: Sequence[Any],
    title: str,
) -> Path:
    """Write replicate-resolved fixed-bank velocity profiles."""

    return _write_velocity_profiles(
        profiles, output_dir=output_dir, references=references, title=title, by_replicate=True
    )


def _write_velocity_profiles(
    profiles: VelocityFigureProfile | Sequence[VelocityFigureProfile],
    *,
    output_dir: Path,
    references: Sequence[Any],
    title: str,
    by_replicate: bool,
) -> Path:
    is_sequence = isinstance(profiles, Sequence)
    rows = tuple(profiles) if is_sequence else (profiles,)
    if not rows:
        raise ValueError("At least one profile is required")
    bank_kind = rows[0].bank_kind
    suffix = " by replicate" if by_replicate else ""
    fig = profile_comparison_grid(
        n_panels=len(rows),
        subplot_titles=(
            [profile.label for profile in rows]
            if is_sequence
            else [f"{rows[0].label}{suffix} ({bank_kind})"]
        ),
        vertical_spacing=0.025 if is_sequence else 0.04,
    )
    for row, profile in enumerate(rows, start=1):
        for index in range(profile.n_replicates if by_replicate else 1):
            mean = profile.replicate_mean[index] if by_replicate else profile.mean
            std = profile.replicate_std[index] if by_replicate else profile.std
            color = (
                (*_VELOCITY_COLORS, "#be123c")[index % 7]
                if by_replicate
                else _VELOCITY_COLORS[(row - 1) % 6]
            )
            name = f"replicate {index}" if by_replicate else profile.label
            legendgroup = f"replicate-{index}" if by_replicate else "gru"
            if not by_replicate and is_sequence:
                legendgroup = f"run-{getattr(profile, 'experiment')}-{getattr(profile, 'run_id')}"
            add_band_trace(
                fig,
                x=profile.time_s,
                mean=mean,
                std=std,
                row=row,
                color=color,
                name=name,
                legendgroup=legendgroup,
                showlegend=row == 1 if by_replicate else True,
                fill_alpha=0.10 if by_replicate else 0.16,
                line_width=1.8 if by_replicate else 2.4,
            )
        for reference in references:
            add_reference_trace(fig, reference=reference, row=row, showlegend=row == 1)
        fig.add_vline(
            x=0.0,
            line={"color": "black", "dash": "dash", "width": 1},
            row=row,
            col=1,
        )

    layout: dict[str, Any] = dict(
        title=title.format(bank_kind=bank_kind),
        width=(1020 if by_replicate else 980) if is_sequence else (940 if by_replicate else 900),
        height=(max(560, 280 * len(rows)) if by_replicate else max(520, 260 * len(rows)))
        if is_sequence
        else (560 if by_replicate else 520),
        margin={"l": 72, "r": 24, "t": 76, "b": 76 if by_replicate else 72},
        hovermode="x unified",
    )
    if by_replicate:
        layout["legend"] = {"groupclick": "togglegroup"}
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=len(rows), col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    path = output_dir / f"forward_velocity_profiles{suffix.replace(' ', '_')}_stochastic.html"
    fig.write_html(path)
    return path


def build_forward_velocity_figure(
    cell_kms: Mapping[str, Mapping[str, Any]],
    *,
    labels: Sequence[str],
    display_names: Mapping[str, str],
    colors: Mapping[str, str],
    trace_mode: str,
    title: str,
    width: int,
    height_per_cell: int,
    vertical_spacing: float,
    dt: float = 0.01,
) -> go.Figure:
    """Build a go-cue-aligned multi-cell forward-velocity figure."""

    labels_present = [label for label in labels if label in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()
    if trace_mode not in {"pooled", "replicate"}:
        raise ValueError(f"unknown trace mode: {trace_mode}")

    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[display_names[label] for label in labels_present],
        vertical_spacing=vertical_spacing,
    )
    for row, label in enumerate(labels_present, start=1):
        velocity = cell_kms[label]["forward_vel_profile"]
        aligned, center = align_trials(velocity, cell_kms[label]["go_idx"])
        color = colors[label]
        if trace_mode == "pooled":
            mean, lower, upper, window = pooled_trial_mean_with_band(aligned, band="sd")
            time = ((np.arange(aligned.shape[-1]) - center) * dt)[window]
            _add_pooled_band(
                fig,
                time=time,
                mean=mean,
                lower=lower,
                upper=upper,
                color=color,
                name=display_names[label],
                row=row,
            )
        else:
            curves, window = replicate_mean_curves(aligned)
            time = ((np.arange(aligned.shape[-1]) - center) * dt)[window]
            _add_replicate_curves(fig, time, curves, color=color, row=row)
        fig.add_vline(
            x=0.0,
            line={"color": "black", "dash": "dash", "width": 1},
            row=row,
            col=1,
        )

    _finish_multi_cell_figure(
        fig,
        n_cells=n_cells,
        title=title,
        width=width,
        height_per_cell=height_per_cell,
        yaxis_title="Fwd vel (m/s)",
    )
    return fig


def build_hold_drift_figure(
    cell_kms: Mapping[str, Mapping[str, Any]],
    *,
    labels: Sequence[str],
    display_names: Mapping[str, str],
    colors: Mapping[str, str],
    trace_mode: str,
    title: str,
    width: int,
    height_per_cell: int,
    vertical_spacing: float,
    pre_go_window_steps: int | None = None,
    dt: float = 0.01,
) -> go.Figure:
    """Build a go-cue-aligned multi-cell pre-go position-drift figure."""

    labels_present = [label for label in labels if label in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()
    if trace_mode not in {"pooled", "replicate"}:
        raise ValueError(f"unknown trace mode: {trace_mode}")

    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[display_names[label] for label in labels_present],
        vertical_spacing=vertical_spacing,
    )
    for row, label in enumerate(labels_present, start=1):
        position = cell_kms[label]["pos_forward_profile"]
        aligned, center = align_trials(position, cell_kms[label]["go_idx"])
        time = np.arange(aligned.shape[-1]) - center
        color = colors[label]
        if trace_mode == "pooled":
            mean, lower, upper, window = pooled_trial_mean_with_band(aligned, band="sd")
            time = (time * dt)[window]
            keep = time <= 0.0
            if pre_go_window_steps is not None:
                keep &= time >= -pre_go_window_steps * dt
            _add_pooled_band(
                fig,
                time=time[keep],
                mean=mean[keep] * 1000.0,
                lower=lower[keep] * 1000.0,
                upper=upper[keep] * 1000.0,
                color=color,
                name=display_names[label],
                row=row,
            )
        else:
            curves, window = replicate_mean_curves(aligned)
            time = (time * dt)[window]
            keep = time <= 0.0
            if pre_go_window_steps is not None:
                keep &= time >= -pre_go_window_steps * dt
            _add_replicate_curves(
                fig,
                time[keep],
                curves[:, keep] * 1000.0,
                color=color,
                row=row,
            )
        fig.add_hline(
            y=0,
            line={"color": "grey", "dash": "dot", "width": 1},
            row=row,
            col=1,
        )
        if pre_go_window_steps is not None:
            fig.add_vline(
                x=-pre_go_window_steps * dt,
                line={"color": "red", "dash": "dot", "width": 1},
                row=row,
                col=1,
            )

    _finish_multi_cell_figure(
        fig,
        n_cells=n_cells,
        title=title,
        width=width,
        height_per_cell=height_per_cell,
        yaxis_title="Fwd pos (mm)",
    )
    return fig


def _add_pooled_band(
    fig: go.Figure,
    *,
    time: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    name: str,
    row: int,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=time,
            y=upper,
            mode="lines",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=lower,
            mode="lines",
            line={"color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor=hex_to_rgba(color, 0.25),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=mean,
            mode="lines",
            line={"color": color, "width": 2},
            name=name,
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def _add_replicate_curves(
    fig: go.Figure,
    time: np.ndarray,
    curves: np.ndarray,
    *,
    color: str,
    row: int,
) -> None:
    for replicate, curve in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=curve,
                mode="lines",
                name=f"Rep {replicate}",
                line={"color": hex_to_rgba(color, 0.7), "width": 1.5},
                showlegend=row == 1,
                legendgroup=f"rep{replicate}",
            ),
            row=row,
            col=1,
        )


def _finish_multi_cell_figure(
    fig: go.Figure,
    *,
    n_cells: int,
    title: str,
    width: int,
    height_per_cell: int,
    yaxis_title: str,
) -> None:
    fig.update_layout(
        title=title,
        width=width,
        height=height_per_cell * n_cells + 100,
        margin={"l": 70, "r": 60, "t": 80, "b": 60},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text=yaxis_title, row=row, col=1)


@dataclass(frozen=True)
class AnalyticalVelocityProfile:
    """One analytical nominal forward-velocity profile."""

    label: str
    kind: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    source: str
    line_color: str
    line_dash: str
    terminal_position_error_m: float
    endpoint_spread_m: float
    parity_status: str
    row: str = "analytical"
    checkpoint: str | None = None
    run_spec: str | None = None

    @property
    def peak_forward_velocity_m_s(self) -> float:
        """Return the peak mean forward velocity."""

        return float(np.nanmax(self.mean))

    @property
    def time_of_peak_forward_velocity_s(self) -> float:
        """Return the time of peak mean forward velocity."""

        return float(self.time_s[int(np.nanargmax(self.mean))])


def materialize_analytical_profiles(
    *, n_samples: int, seed: int = 376023
) -> tuple[AnalyticalVelocityProfile, AnalyticalVelocityProfile]:
    """Build 6D extLQG and H-infinity nominal profiles with common random draws."""

    plant, schedule = build_no_integrator_game()
    config = OutputFeedbackConfig(n_phys=6)
    gamma_star = find_gamma_star(plant, schedule)
    lqr_solution = solve_lqr(plant, schedule)
    hinf_solution = solve_hinf_riccati(
        plant,
        schedule,
        OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR * gamma_star,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    covariances = default_cs_noise_covariances(plant, config)
    extlqg_path = build_extlqg_comparator_path(
        plant,
        lqr_solution.K,
        covariances,
        schedule=schedule,
        config=config,
    )
    robust_covariances = robust_estimator_covariances(
        plant,
        schedule,
        hinf_solution.gamma,
        config,
    )
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        hinf_solution,
        robust_covariances,
        config,
    )

    extlqg_rollouts = []
    hinf_rollouts = []
    for key in jr.split(jr.PRNGKey(seed), n_samples):
        draws = sample_forward_noise_draws(key, T=schedule.T, covariances=covariances)
        extlqg_rollouts.append(
            simulate_lqg_released_forward(
                plant,
                extlqg_path.controller_gains,
                x0,
                draws=draws,
                covariances=covariances,
                estimator_gains=extlqg_path.estimator_gains,
                config=config,
            )
        )
        hinf_rollouts.append(
            simulate_robust_released_forward(
                plant,
                schedule,
                hinf_solution,
                x0,
                draws=draws,
                covariances=covariances,
                gains=robust_gains,
                config=config,
            )
        )

    vel_lo, _vel_hi = plant.vel_slice
    extlqg_forward = np.stack(
        [np.asarray(rollout.x[:, vel_lo], dtype=np.float64) for rollout in extlqg_rollouts]
    )
    hinf_forward = np.stack(
        [np.asarray(rollout.x[:, vel_lo], dtype=np.float64) for rollout in hinf_rollouts]
    )
    time_s = np.arange(schedule.T + 1, dtype=np.float64) * float(plant.dt)
    return (
        AnalyticalVelocityProfile(
            label="6D analytical extLQG nominal",
            kind="analytical_extlqg_6d_output_feedback",
            time_s=time_s,
            mean=np.mean(extlqg_forward, axis=0),
            std=np.std(extlqg_forward, axis=0),
            n_samples=n_samples,
            source="rlrmp.analysis.math.cs_released_simulation",
            line_color="#111827",
            line_dash="dash",
            terminal_position_error_m=float(
                np.mean([rollout.terminal_position_error for rollout in extlqg_rollouts])
            ),
            endpoint_spread_m=float(
                np.std([rollout.terminal_position_error for rollout in extlqg_rollouts])
            ),
            parity_status=extlqg_path.parity_status,
        ),
        AnalyticalVelocityProfile(
            label="6D output-feedback H-infinity nominal",
            kind="analytical_hinf_6d_output_feedback",
            time_s=time_s,
            mean=np.mean(hinf_forward, axis=0),
            std=np.std(hinf_forward, axis=0),
            n_samples=n_samples,
            source="rlrmp.analysis.math.cs_released_simulation",
            line_color="#dc2626",
            line_dash="dot",
            terminal_position_error_m=float(
                np.mean([rollout.terminal_position_error for rollout in hinf_rollouts])
            ),
            endpoint_spread_m=float(
                np.std([rollout.terminal_position_error for rollout in hinf_rollouts])
            ),
            parity_status=(
                "6D no-integrator output-feedback robust estimator/controller; "
                f"gamma_factor={OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR:g}"
            ),
        ),
    )


def build_profile_family_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    quantity_specs: Sequence[tuple[str, str, str]],
    timing_bins_for_rows: Callable[[Sequence[Mapping[str, Any]]], Sequence[str]],
    row_timing_label: Callable[[Mapping[str, Any]], str],
    representative_timing: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any] | None],
    perturbation_interval_bounds: Callable[[Mapping[str, Any]], tuple[float, float]],
    collect_traces: Callable[[Sequence[Mapping[str, Any]]], Mapping[tuple[Any, ...], Any]],
    trace_key: Callable[[str, str, str, str], tuple[Any, ...]],
    add_trace: Callable[..., None],
    sources: Sequence[str],
    variants: Sequence[str],
    axis_unit: Callable[[str, str], str],
    figure_kind: str,
    title: str,
    width_min: int,
    width_per_column: int,
    height: int,
    legend_title: str = "Source / trace",
    vertical_spacing: float = 0.08,
    horizontal_spacing: float = 0.045,
    margin: Mapping[str, int] | None = None,
) -> Any:
    """Build the shared perturbation-profile quantity-by-timing grid."""

    timing_bins = tuple(timing_bins_for_rows(rows))
    n_cols = len(timing_bins)
    subplot_titles = [
        f"{quantity_label}: {timing_label}"
        for _quantity, quantity_label, _unit in quantity_specs
        for timing_label in timing_bins
    ]
    fig = profile_comparison_grid(
        n_panels=len(quantity_specs) * n_cols,
        rows=len(quantity_specs),
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_yaxes="rows",
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    legend_seen: set[tuple[str, str, str]] = set()
    for col_index, timing_label in enumerate(timing_bins, start=1):
        timing_rows = [row for row in rows if row_timing_label(row) == timing_label]
        timing = representative_timing(timing_rows)
        if timing is not None:
            x0, x1 = perturbation_interval_bounds(timing)
            for row_index in range(1, len(quantity_specs) + 1):
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
        traces = collect_traces(timing_rows)
        for row_index, (quantity, _quantity_label, unit) in enumerate(
            quantity_specs,
            start=1,
        ):
            for source in sources:
                for variant in variants:
                    for coord in ("orthogonal", "along"):
                        samples = traces.get(trace_key(source, variant, quantity, coord))
                        if samples is None or samples.size == 0:
                            continue
                        legend_key = (source, variant, coord)
                        add_trace(
                            fig,
                            samples,
                            source=source,
                            variant=variant,
                            quantity=quantity,
                            coord=coord,
                            row=row_index,
                            col=col_index,
                            showlegend=legend_key not in legend_seen,
                        )
                        legend_seen.add(legend_key)
            fig.update_yaxes(
                title_text=axis_unit(quantity, unit),
                row=row_index,
                col=1,
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=max(width_min, width_per_column * n_cols),
        height=height,
        legend_title_text=legend_title,
        margin=dict(margin or {"l": 70, "r": 24, "t": 96, "b": 70}),
    )
    for col_index in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="time from movement onset (s)",
            row=len(quantity_specs),
            col=col_index,
        )
    return fig


def build_stabilization_family_figure(
    *,
    response_variables: Sequence[str],
    columns: Sequence[Any],
    response_label: Callable[[str], str],
    column_label: Callable[[Any], str],
    response_axis_title: Callable[[str], str],
    render_cell: Callable[[Any, str, Any, int, int, set[tuple[str, str]], dict], None],
    title: str,
    width: int,
    horizontal_spacing: float,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a stabilization response grid while delegating cell semantics."""

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=len(response_variables),
        cols=len(columns),
        subplot_titles=[
            f"{response_label(response)} - {column_label(column)}"
            for response in response_variables
            for column in columns
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=0.09,
    )
    coverage: list[dict[str, Any]] = []
    event_markers: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    outputs = {
        "coverage": coverage,
        "event_markers": event_markers,
        "unavailable": unavailable,
    }
    legend_seen: set[tuple[str, str]] = set()
    cache: dict[Any, Any] = {}
    for row_index, response in enumerate(response_variables, start=1):
        for col_index, column in enumerate(columns, start=1):
            render_cell(
                fig,
                response,
                column,
                row_index,
                col_index,
                legend_seen,
                {**outputs, "cache": cache},
            )
            if col_index == 1:
                fig.update_yaxes(
                    title_text=response_axis_title(response),
                    row=row_index,
                    col=col_index,
                )
            if row_index == len(response_variables):
                fig.update_xaxes(
                    title_text="time from perturbation onset (s)",
                    row=row_index,
                    col=col_index,
                )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=width,
        height=900,
        margin={"l": 78, "r": 28, "t": 96, "b": 112},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.09,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    return fig, coverage, event_markers, unavailable


def build_stabilization_response_family_figure(
    *,
    family: str,
    response_variables: Sequence[str],
    columns: Sequence[Any],
    cell_context: Callable[[str, Any], Mapping[str, Any]],
    analytical_sources: Sequence[str],
    analytical_profile: Callable[..., Mapping[str, Any]],
    add_profile_traces: Callable[..., None],
    coverage_row: Callable[..., dict[str, Any]],
    add_unsupported_annotation: Callable[..., None],
    infer_event_marker: Callable[..., Mapping[str, Any]],
    add_event_marker: Callable[..., None],
    response_label: Callable[[str], str],
    column_label: Callable[[Any], str],
    response_axis_title: Callable[[str], str],
    title: str,
    width: int,
    horizontal_spacing: float,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a complete learned-versus-analytical stabilization family grid."""

    def render_cell(fig, response, column, row, col, legend_seen, outputs):
        context = cell_context(response, column)

        def emit_profile(source, run_id, profile, status):
            add_profile_traces(
                fig=fig,
                profile=profile,
                source=source,
                dt=context["dt"],
                row=row,
                col=col,
                legend_seen=legend_seen,
            )
            outputs["coverage"].append(
                coverage_row(
                    source=source,
                    family=family,
                    response_variable=response,
                    run_id=run_id,
                    profile=profile,
                    analytical_status=status,
                    **context["coverage_metadata"],
                )
            )

        for learned in context["learned"]:
            emit_profile(
                learned["source"], learned["run_id"], learned["profile"], "not_applicable"
            )
        for source in analytical_sources:
            cache_key = (*context["cache_prefix"], source, family, response)
            if cache_key not in outputs["cache"]:
                outputs["cache"][cache_key] = analytical_profile(
                    source=source,
                    family=family,
                    response_variable=response,
                    baseline_rows=context["baseline_rows"],
                    timing=context["timing"],
                )
            result = outputs["cache"][cache_key]
            if result["status"] == "available":
                emit_profile(source, None, result["profile"], "available")
            else:
                outputs["unavailable"].append(
                    {
                        "source": source, "family": family, "response_variable": response,
                        **context["identity_metadata"],
                        "status": result["status"], "reason": result["reason"],
                    }
                )
                if source == "robust_output_feedback6d":
                    add_unsupported_annotation(
                        fig=fig, text="H-inf replay unsupported", row=row, col=col
                    )
        marker = infer_event_marker(
            family_rows=context["baseline_rows"],
            summary_timing=context["timing"],
            dt=context["dt"],
        )
        add_event_marker(fig=fig, marker=marker, row=row, col=col)
        outputs["event_markers"].append(
            {"family": family, "response_variable": response,
             **context["identity_metadata"], **marker}
        )

    return build_stabilization_family_figure(
        response_variables=response_variables,
        columns=columns,
        response_label=response_label,
        column_label=column_label,
        response_axis_title=response_axis_title,
        render_cell=render_cell,
        title=title,
        width=width,
        horizontal_spacing=horizontal_spacing,
    )
