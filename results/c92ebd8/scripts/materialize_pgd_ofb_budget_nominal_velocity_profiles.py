#!/usr/bin/env python
"""Materialize output-feedback-budget PGD nominal velocity profiles for c92."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from feedbax.plot import save_figure

from materialize_pgd_1p05_nominal_velocity_profiles import (
    build_robust_output_feedback_6d_context,
    forward_velocity_profile,
    json_safe,
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


ISSUE = "c92ebd8"
TOPIC = "pgd_ofb_budget_moderate_nominal_velocity_profiles"
NOTE_PATH = REPO_ROOT / "results" / ISSUE / "notes" / f"{TOPIC}.md"
SPEC_PATH = REPO_ROOT / "results" / ISSUE / "figures" / TOPIC / "spec.json"
HTML_PATH = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC / "figure.html"
PNG_PATH = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC / "figure.png"
DT = 0.01

RUN_ROWS = (
    {
        "run_id": "open_loop_moderate",
        "label": "No-PGD moderate",
        "legend": "No-PGD moderate GRU",
        "color": "#f97316",
        "training": "no_pgd_open_loop",
        "pgd_budget": None,
    },
    {
        "run_id": "moderate_pgd_ofb1p05",
        "label": "OFB gamma 1.05 budget",
        "legend": "OFB gamma 1.05 budget GRU",
        "color": "#2563eb",
        "training": "pgd_output_feedback_budget",
        "pgd_budget": "ofb_6d_no_integrator_gamma_1p05_rollout_radius",
    },
    {
        "run_id": "moderate_pgd_ofb1p4",
        "label": "OFB gamma 1.4 budget",
        "legend": "OFB gamma 1.4 budget GRU",
        "color": "#7c3aed",
        "training": "pgd_output_feedback_budget",
        "pgd_budget": "ofb_6d_no_integrator_gamma_1p4_rollout_radius",
    },
)

COMPARATOR_COLORS = {
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
        profiles[run.run_id] = forward_velocity_profile(evaluation.velocity)
    return profiles


def materialize_figure(
    *,
    run_profiles: Mapping[str, np.ndarray],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and save the output-feedback-budget nominal velocity figure."""

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
        add_mean_band(
            fig,
            run_profiles[str(row_spec["run_id"])],
            row=row_index,
            col=1,
            name=str(row_spec["legend"]),
            color=str(row_spec["color"]),
            showlegend=True,
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
        title="c92 output-feedback-budget PGD nominal forward velocity profiles",
        template="plotly_white",
        width=1040,
        height=760,
        legend_title_text="profile",
        margin={"l": 78, "r": 24, "t": 90, "b": 70},
    )

    spec = build_spec(robust_context=robust_context)
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
        "rows": [
            {
                "run_id": row["run_id"],
                "label": row["label"],
                "training": row["training"],
                "pgd_budget": row["pgd_budget"],
            }
            for row in RUN_ROWS
        ],
        "analytical_comparator_contract": spec["analytical_comparator_contract"],
    }


def build_spec(*, robust_context: Mapping[str, Any]) -> dict[str, Any]:
    """Return the tracked figure specification."""

    return {
        "schema_version": "rlrmp.c92_pgd_ofb_budget_nominal_velocity_profiles.v1",
        "issue": ISSUE,
        "figure_kind": "pgd_output_feedback_budget_nominal_forward_velocity_profiles",
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
                "training": row["training"],
                "pgd_budget": row["pgd_budget"],
            }
            for row in RUN_ROWS
        ],
        "transform": [
            {
                "name": "final_checkpoint_forward_velocity_profile_mean_band",
                "kwargs": {
                    "row_labels": {str(row["run_id"]): row["label"] for row in RUN_ROWS},
                    "n_rollout_trials": DEFAULT_N_ROLLOUT_TRIALS,
                    "use_validation_selected_checkpoints": False,
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
    """Add a mean line and central 80 percent band."""

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
            fillcolor=band_color(color),
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


def mean_band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and central 80 percent interval over sample rows."""

    array = np.asarray(samples, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    if array.shape[0] <= 1:
        return mean, mean, mean
    return mean, np.nanpercentile(array, 10.0, axis=0), np.nanpercentile(array, 90.0, axis=0)


def band_color(color: str) -> str:
    """Return a transparent fill color for a line color."""

    hex_color = color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},0.12)"


def write_png_output(*, fig: go.Figure, html_path: Path, png_path: Path) -> dict[str, Any]:
    """Write a PNG copy, falling back to headless Chrome when Kaleido is blocked."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_png_image(fig, png_path)
    except (ValueError, RuntimeError, OSError) as exc:
        try:
            write_png_from_html(html_path=html_path, png_path=png_path)
        except (RuntimeError, OSError, subprocess.SubprocessError) as chrome_exc:
            if png_path.exists() and png_path.stat().st_size > 0:
                return {
                    "status": "written",
                    "renderer": "existing_png_after_export_block",
                    "path": repo_rel(png_path),
                    "bytes": png_path.stat().st_size,
                    "errors": {
                        "kaleido": f"{type(exc).__name__}: {exc}",
                        "chrome": f"{type(chrome_exc).__name__}: {chrome_exc}",
                    },
                }
            return {
                "status": "blocked",
                "renderer": None,
                "path": repo_rel(png_path),
                "bytes": 0,
                "errors": {
                    "kaleido": f"{type(exc).__name__}: {exc}",
                    "chrome": f"{type(chrome_exc).__name__}: {chrome_exc}",
                },
            }
        return {
            "status": "written",
            "renderer": "chrome_headless_html_screenshot_after_kaleido_block",
            "path": repo_rel(png_path),
            "bytes": png_path.stat().st_size,
            "errors": {"kaleido": f"{type(exc).__name__}: {exc}"},
        }
    return {
        "status": "written",
        "renderer": "kaleido",
        "path": repo_rel(png_path),
        "bytes": png_path.stat().st_size,
        "errors": {},
    }


def write_png_image(fig: go.Figure, path: Path) -> None:
    """Write a PNG, working around Kaleido path handling on local installs."""

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


def append_png_export_to_spec(png_export: Mapping[str, Any]) -> None:
    """Append PNG export status to the routed tracked spec."""

    spec = json.loads(SPEC_PATH.read_text())
    spec["png_export"] = dict(png_export)
    write_compact_json(SPEC_PATH, spec)


def write_note(output: Mapping[str, Any]) -> None:
    """Write a concise regenerable note for the nominal profile figure."""

    lines = [
        "# Output-Feedback-Budget PGD Nominal Velocity Profiles",
        "",
        "- Scope: nominal velocity profiles only for the two output-feedback-budget "
        "PGD rows and the matched no-PGD moderate row.",
        "- Rows: `open_loop_moderate` (No-PGD moderate), `moderate_pgd_ofb1p05` "
        "(OFB gamma 1.05 budget), and `moderate_pgd_ofb1p4` "
        "(OFB gamma 1.4 budget).",
        "- Checkpoint policy: final trained checkpoint for all GRU traces.",
        "- Analytical comparators: 6D no-integrator extLQG and 6D no-integrator "
        "output-feedback H-infinity.",
        f"- Figure spec: `{output['spec']}`.",
        f"- HTML artifact: `{output['html']}`.",
        f"- PNG artifact: `{output['png']}`.",
        f"- PNG renderer: `{output['png_export']['renderer']}`.",
        "",
    ]
    update_marked_section(NOTE_PATH, TOPIC, "\n".join(lines) + "\n")


def repo_rel(path: Path) -> str:
    """Return a repository-relative path."""

    return str(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
