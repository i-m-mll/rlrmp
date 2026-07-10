#!/usr/bin/env python
"""Materialize PGD 1.05 nominal velocity profiles for c92."""

from __future__ import annotations
from rlrmp.viz.traces import add_sample_band
from rlrmp.eval.robustness_diagnostics import (
    build_robust_output_feedback_6d_context as _build_robust_output_feedback_6d_context,
)
from rlrmp.io import json_ready
from rlrmp.paths import portable_repo_path
from rlrmp.viz.traces import add_profile_line

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
from feedbax.plot import save_figure

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import evaluate_run_rollouts
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    resolve_run_inputs,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    _build_extlqg_comparator_context,
    _evaluation_from_extlqg_rollout,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.viz import profile_comparison_grid


ISSUE = "c92ebd8"
TOPIC = "pgd_1p05_nominal_velocity_profiles"
NOTE_PATH = REPO_ROOT / "results" / ISSUE / "notes" / f"{TOPIC}.md"
DT = 0.01
PHYSICAL_LEVELS = ("small", "moderate", "stress")
PGD_RUNS = {
    "small": "small",
    "moderate": "moderate",
    "stress": "stress",
}
NO_PGD_RUNS = {
    "small": "open_loop_small",
    "moderate": "open_loop_moderate",
    "stress": "open_loop_stress",
}
SOURCE_COLORS = {
    "pgd": "#2563eb",
    "no_pgd": "#f97316",
    "extlqg6d": "#c2410c",
    "robust_output_feedback6d": "#15803d",
}


def main() -> None:
    """CLI entry point."""

    extlqg_context = _build_extlqg_comparator_context(physical_dim=6)
    robust_context = build_robust_output_feedback_6d_context()
    run_profiles = evaluate_requested_run_profiles()
    output = materialize_figure(
        run_profiles=run_profiles,
        extlqg_context=extlqg_context,
        robust_context=robust_context,
    )
    write_note(output)
    print(json.dumps(output, indent=2, sort_keys=True))


def evaluate_requested_run_profiles() -> dict[str, np.ndarray]:
    """Evaluate final-checkpoint nominal forward velocity for the requested rows."""

    run_ids = tuple(PGD_RUNS.values()) + tuple(NO_PGD_RUNS.values())
    runs = resolve_run_inputs(
        experiment=ISSUE,
        run_ids=run_ids,
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
        profiles[run.run_id] = forward_velocity_profile(evaluation.velocity)
    return profiles


def materialize_figure(
    *,
    run_profiles: Mapping[str, np.ndarray],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and save the PGD-vs-no-PGD nominal velocity profile figure."""

    fig = profile_comparison_grid(
        n_panels=len(PHYSICAL_LEVELS),
        rows=len(PHYSICAL_LEVELS),
        cols=1,
        subplot_titles=[level for level in PHYSICAL_LEVELS],
        vertical_spacing=0.075,
    )
    ext_profile = forward_velocity_profile(extlqg_context["base_evaluation"].velocity)
    robust_profile = forward_velocity_profile(robust_context["velocity"])

    for row_index, level in enumerate(PHYSICAL_LEVELS, start=1):
        add_mean_band(
            fig,
            run_profiles[PGD_RUNS[level]],
            row=row_index,
            col=1,
            name="PGD 1.05 GRU",
            color=SOURCE_COLORS["pgd"],
            showlegend=row_index == 1,
        )
        add_mean_band(
            fig,
            run_profiles[NO_PGD_RUNS[level]],
            row=row_index,
            col=1,
            name="No-PGD GRU",
            color=SOURCE_COLORS["no_pgd"],
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
    fig.update_xaxes(title_text="time from movement onset (s)", row=len(PHYSICAL_LEVELS), col=1)
    fig.update_layout(
        title="c92 PGD 1.05 nominal forward velocity profiles",
        template="plotly_white",
        width=1040,
        height=760,
        legend_title_text="profile",
        margin={"l": 78, "r": 24, "t": 90, "b": 70},
    )

    spec = {
        "schema_version": "rlrmp.c92_pgd_1p05_nominal_velocity_profiles.v1",
        "issue": ISSUE,
        "figure_kind": "pgd_1p05_nominal_forward_velocity_profile_comparison",
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
            },
            "output_feedback_hinf": robust_context["contract"],
        },
        "inputs": [
            {"role": "run_spec", "path": f"results/{ISSUE}/runs/{run_id}.json"}
            for run_id in tuple(PGD_RUNS.values()) + tuple(NO_PGD_RUNS.values())
        ],
        "transform": [
            {
                "name": "final_checkpoint_forward_velocity_profile_mean_band",
                "kwargs": {
                    "physical_levels": list(PHYSICAL_LEVELS),
                    "pgd_runs": PGD_RUNS,
                    "no_pgd_runs": NO_PGD_RUNS,
                    "n_rollout_trials": DEFAULT_N_ROLLOUT_TRIALS,
                    "analytical_comparators": [
                        "6d_output_feedback_extlqg",
                        "6d_output_feedback_hinf",
                    ],
                },
            }
        ],
        "plot_kwargs": {
            "grid_helper": "rlrmp.viz.profile_comparison_grid",
            "shared_yaxes": "all",
            "rows": len(PHYSICAL_LEVELS),
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
        topic=TOPIC,
        extra_packages=["rlrmp"],
    )
    return {
        "status": "materialized",
        "topic": TOPIC,
        "save_result": json_safe(saved),
        "spec": f"results/{ISSUE}/figures/{TOPIC}/spec.json",
        "html": f"results/{ISSUE}/figures/{TOPIC}/figure.html",
        "bulk_html": f"_artifacts/{ISSUE}/figures/{TOPIC}/figure.html",
        "run_pairs": [
            {
                "physical_level": level,
                "pgd_run": PGD_RUNS[level],
                "no_pgd_run": NO_PGD_RUNS[level],
            }
            for level in PHYSICAL_LEVELS
        ],
        "analytical_comparator_contract": {
            "extlqg": spec["analytical_comparator_contract"]["extlqg"],
            "output_feedback_hinf": robust_context["contract"],
        },
    }


def build_robust_output_feedback_6d_context() -> dict[str, Any]:
    """Build this script's canonical robust output-feedback context."""

    return _build_robust_output_feedback_6d_context(
        evaluation_from_rollout=_evaluation_from_extlqg_rollout,
        gamma_factor=OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    )


def forward_velocity_profile(velocity: Any) -> np.ndarray:
    """Return samples x time forward velocity profiles."""

    samples = as_samples(np.asarray(velocity, dtype=np.float64))
    return samples[..., 0]


def as_samples(values: np.ndarray) -> np.ndarray:
    """Flatten leading dimensions into a sample axis for time-by-xy arrays."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 2:
        raise ValueError(f"expected array with trailing shape (time, 2), got {array.shape}")
    return array.reshape((-1, array.shape[-2], 2))


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
    """Add this script's central sample band."""

    add_sample_band(
        fig,
        samples,
        reducer=mean_band,
        row=row,
        col=col,
        name=name,
        color=color,
        showlegend=showlegend,
        dt=DT,
    )


add_line = add_profile_line


def mean_band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and central 80 percent interval over sample rows."""

    array = np.asarray(samples, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    if array.shape[0] <= 1:
        return mean, mean, mean
    return mean, np.nanpercentile(array, 10.0, axis=0), np.nanpercentile(array, 90.0, axis=0)


def band_color(color: str) -> str:
    """Return a transparent fill color matching the supported line palette."""

    if color == SOURCE_COLORS["pgd"]:
        return "rgba(37,99,235,0.12)"
    if color == SOURCE_COLORS["no_pgd"]:
        return "rgba(249,115,22,0.12)"
    return "rgba(0,0,0,0.08)"


def write_note(output: Mapping[str, Any]) -> None:
    """Write a concise regenerable note for the nominal profile figure."""

    lines = [
        "# PGD 1.05 Nominal Velocity Profiles",
        "",
        "- Scope: nominal velocity profiles only for the PGD 1.05 mini-matrix.",
        "- Rows: `small`, `moderate`, and `stress`, each paired with its no-PGD open-loop row.",
        "- Checkpoint policy: final trained checkpoint for all six GRU traces.",
        "- Analytical comparators: 6D no-integrator extLQG and 6D no-integrator "
        "output-feedback H-infinity.",
        f"- Figure spec: `{output['spec']}`.",
        f"- Navigable HTML link: `{output['html']}`.",
        "",
    ]
    update_marked_section(NOTE_PATH, TOPIC, "\n".join(lines) + "\n")


repo_rel = portable_repo_path


json_safe = json_ready


if __name__ == "__main__":
    main()
