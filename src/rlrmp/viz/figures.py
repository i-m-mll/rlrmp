"""Shared analytical-profile and figure assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax.random as jr
import numpy as np

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
from rlrmp.viz.traces import add_profile_line


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


def build_nominal_velocity_spec(
    *,
    schema_version: str,
    issue: str,
    figure_kind: str,
    robust_contract: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    transform_name: str,
    transform_kwargs: Mapping[str, Any],
    rows: int,
    outputs: Mapping[str, str],
    cols: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared nominal-velocity figure-spec contract."""

    plot_kwargs = {
        "grid_helper": "rlrmp.viz.profile_comparison_grid",
        "shared_yaxes": "all",
        "rows": rows,
        "physical_state_dim": 6,
        "state_dim": 36,
        "disturbance_integrators_exposed": False,
    }
    if cols is not None:
        plot_kwargs["cols"] = cols
    return {
        "schema_version": schema_version,
        "issue": issue,
        "figure_kind": figure_kind,
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
            "output_feedback_hinf": dict(robust_contract),
        },
        "inputs": [dict(item) for item in inputs],
        "transform": [{"name": transform_name, "kwargs": dict(transform_kwargs)}],
        **dict(extra or {}),
        "plot_kwargs": plot_kwargs,
        "outputs": dict(outputs),
    }


def build_nominal_profile_figure(
    *,
    rows: Sequence[Mapping[str, Any]],
    add_run_profiles: Callable[[Any, Mapping[str, Any], int], None],
    ext_profile: np.ndarray,
    robust_profile: np.ndarray,
    comparator_colors: Mapping[str, str],
    title: str,
    height: int,
    vertical_spacing: float = 0.075,
) -> Any:
    """Build a shared-y-axis nominal profile comparison grid."""

    fig = profile_comparison_grid(
        n_panels=len(rows),
        rows=len(rows),
        cols=1,
        subplot_titles=[str(row["label"]) for row in rows],
        vertical_spacing=vertical_spacing,
    )
    for row_index, row_spec in enumerate(rows, start=1):
        add_run_profiles(fig, row_spec, row_index)
        add_profile_line(
            fig,
            ext_profile,
            row=row_index,
            col=1,
            name="6D extLQG",
            color=comparator_colors["extlqg6d"],
            dash="dash",
            showlegend=row_index == 1,
            width=2.8,
        )
        add_profile_line(
            fig,
            robust_profile,
            row=row_index,
            col=1,
            name="6D output-feedback H-infinity",
            color=comparator_colors["robust_output_feedback6d"],
            dash="dot",
            showlegend=row_index == 1,
            width=2.8,
        )
        fig.update_yaxes(title_text="m/s", row=row_index, col=1)
    fig.update_xaxes(title_text="time from movement onset (s)", row=len(rows), col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1040,
        height=height,
        legend_title_text="profile",
        margin={"l": 78, "r": 24, "t": 90, "b": 70},
    )
    return fig


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
