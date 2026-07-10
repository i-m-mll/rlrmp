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
) -> Any:
    """Build a shared-y-axis nominal profile comparison grid."""

    fig = profile_comparison_grid(
        n_panels=len(rows),
        rows=len(rows),
        cols=1,
        subplot_titles=[str(row["label"]) for row in rows],
        vertical_spacing=0.075,
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
