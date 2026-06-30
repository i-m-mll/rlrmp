"""Materialize the ae9f30f nominal velocity overlay figure."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go
from feedbax.plot import save_figure
from plotly.subplots import make_subplots

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import cs_output_feedback_reference_profiles
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE = "ae9f30f"
TOPIC = "nominal_velocity_overlay"
TRIALS_TAG = "gru_soft_lambda_wave1_trained_final"
ROWS = {
    "direct_epsilon": ("direct_epsilon_b1p05", "direct_epsilon_b1p4"),
    "linear_no_bias": ("linear_no_bias_b1p05",),
}
STOPPED_ROW = "linear_no_bias_b1p4"


def _row_profile(row: str) -> dict[str, Any]:
    path = REPO_ROOT / "_artifacts" / ISSUE / "evaluation_diagnostics" / TRIALS_TAG / f"{row}.npz"
    data = np.load(path)
    velocity = np.asarray(data["velocity"], dtype=np.float64)
    forward = velocity[..., 0].reshape(-1, velocity.shape[-2])
    # Feedbax Task.eval_trials returns post-transition samples only. Reinsert
    # the true initial velocity so this plot uses the same state-indexed
    # convention as the analytical x[0:T+1] reference traces.
    initial_forward = np.zeros((forward.shape[0], 1), dtype=np.float64)
    forward = np.concatenate([initial_forward, forward], axis=1)
    mean = np.mean(forward, axis=0)
    std = np.std(forward, axis=0)
    time_s = np.arange(mean.shape[0], dtype=np.float64) * 0.01
    peak_idx = int(np.argmax(mean))
    return {
        "row": row,
        "time_s": time_s,
        "mean": mean,
        "std": std,
        "n_samples": int(forward.shape[0]),
        "peak_forward_velocity_m_s": float(mean[peak_idx]),
        "time_of_peak_s": float(time_s[peak_idx]),
    }


def _hinf_reference_profile() -> dict[str, Any]:
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x = np.asarray(gamma_ref.nominal_rollout.x, dtype=np.float64)
    vel_lo, _ = reference.plant.vel_slice
    forward = x[:, vel_lo]
    time_s = np.arange(forward.shape[0], dtype=np.float64) * float(reference.plant.dt)
    peak_idx = int(np.argmax(forward))
    return {
        "label": "C&S H-infinity nominal 6D",
        "time_s": time_s,
        "mean": forward,
        "std": np.zeros_like(forward),
        "peak_forward_velocity_m_s": float(forward[peak_idx]),
        "time_of_peak_s": float(time_s[peak_idx]),
        "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    }


def _extlqg_reference_profile() -> dict[str, Any]:
    references = cs_output_feedback_reference_profiles(n_samples=40)
    reference = references[0]
    return {
        "label": "C&S extLQG/output-feedback 8D",
        "time_s": np.asarray(reference.time_s, dtype=np.float64),
        "mean": np.asarray(reference.forward_velocity, dtype=np.float64),
        "std": np.asarray(reference.forward_velocity_std, dtype=np.float64),
        "peak_forward_velocity_m_s": float(reference.peak_forward_velocity_m_s),
        "time_of_peak_s": float(reference.time_of_peak_forward_velocity_s),
        "n_samples": int(reference.n_samples),
        "parity_status": reference.parity_status,
    }


def _add_profile(
    fig: go.Figure,
    *,
    profile: dict[str, Any],
    row: int,
    color: str,
    name: str,
    dash: str | None = None,
    band_alpha: float = 0.14,
) -> None:
    time_s = profile["time_s"]
    mean = profile["mean"]
    std = profile["std"]
    if np.any(std):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_s, time_s[::-1]]),
                y=np.concatenate([mean + std, (mean - std)[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, band_alpha),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name=f"{name} +/- 1 SD",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=mean,
            mode="lines",
            line={"color": color, "width": 2, **({"dash": dash} if dash else {})},
            name=name,
            showlegend=True,
        ),
        row=row,
        col=1,
    )


def _rgba(hex_color: str, alpha: float) -> str:
    raw = hex_color.lstrip("#")
    r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def main() -> None:
    extlqg = _extlqg_reference_profile()
    hinf = _hinf_reference_profile()
    profiles = {row: _row_profile(row) for rows in ROWS.values() for row in rows}
    summary: dict[str, Any] = {
        "schema_version": "rlrmp.ae9f30f.nominal_velocity_overlay.v2",
        "issue": ISSUE,
        "topic": TOPIC,
        "time_index_convention": {
            "name": "state_indexed",
            "sample_0": "true trial initial velocity",
            "gru_source": "Task.eval_trials strips the prepended initial state; this script reinserts the zero initial forward velocity before plotting.",
            "analytical_source": "full nominal rollout state history x[0:T+1]",
        },
        "rows": {},
        "references": {
            extlqg["label"]: {
                "peak_forward_velocity_m_s": extlqg["peak_forward_velocity_m_s"],
                "time_of_peak_s": extlqg["time_of_peak_s"],
                "n_samples": extlqg["n_samples"],
                "parity_status": extlqg["parity_status"],
            },
            hinf["label"]: {
                "peak_forward_velocity_m_s": hinf["peak_forward_velocity_m_s"],
                "time_of_peak_s": hinf["time_of_peak_s"],
                "gamma_factor": hinf["gamma_factor"],
            },
        },
        "stopped_context": {
            STOPPED_ROW: "omitted from trained-model profile panel; stopped at batch 999/1000 gate",
        },
    }

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=("direct_epsilon", "linear_no_bias"),
        vertical_spacing=0.08,
    )
    colors = {
        "direct_epsilon_b1p05": "#2563eb",
        "direct_epsilon_b1p4": "#0f766e",
        "linear_no_bias_b1p05": "#7c3aed",
    }
    for idx, mechanism in enumerate(("direct_epsilon", "linear_no_bias"), start=1):
        for profile in (extlqg, hinf):
            _add_profile(
                fig,
                profile=profile,
                row=idx,
                color="#111827" if "extLQG" in profile["label"] else "#dc2626",
                name=profile["label"],
                dash="dash" if "extLQG" in profile["label"] else "dot",
                band_alpha=0.08,
            )
        for row in ROWS[mechanism]:
            profile = profiles[row]
            _add_profile(fig, profile=profile, row=idx, color=colors[row], name=row)
            summary["rows"][row] = {
                "mechanism_panel": mechanism,
                "n_samples": profile["n_samples"],
                "peak_forward_velocity_m_s": profile["peak_forward_velocity_m_s"],
                "time_of_peak_s": profile["time_of_peak_s"],
            }
    fig.update_yaxes(title_text="forward velocity (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="forward velocity (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        title="Nominal forward-velocity profiles, final checkpoints",
        legend_title_text="trace",
        width=980,
        height=760,
    )

    spec = {
        "figure_kind": "nominal_velocity_overlay",
        "inputs": [
            {"path": f"_artifacts/{ISSUE}/evaluation_diagnostics/{TRIALS_TAG}/{row}.npz"}
            for row in summary["rows"]
        ]
        + [{"path": f"results/{ISSUE}/runs/{row}.json"} for row in summary["rows"]]
        + [{"path": f"results/{ISSUE}/runs/{STOPPED_ROW}.json"}],
        "transform": [{"name": "nominal_velocity_overlay", "kwargs": {}}],
        "plot_kwargs": {
            "issue": ISSUE,
            "topic": TOPIC,
            "evaluation_npz_dir": f"_artifacts/{ISSUE}/evaluation_diagnostics/{TRIALS_TAG}",
            "stopped_context": f"results/{ISSUE}/runs/{STOPPED_ROW}.json",
            "time_index_convention": "state_indexed_with_reinserted_initial_gru_velocity",
        },
        "plot": {
            "shared_yaxes": "all",
            "panels": list(ROWS),
            "references": list(summary["references"]),
        },
    }
    save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=ISSUE,
        topic=TOPIC,
        extra_packages=["rlrmp"],
    )
    summary_dir = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC
    mkdir_p(summary_dir)
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
