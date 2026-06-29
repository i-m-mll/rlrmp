#!/usr/bin/env python
"""Materialize c92 no-PGD perturbation profiles with PGD/H-infinity overlays."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from materialize_post_training_figures import (
    COORD_DASH,
    DT,
    ISSUE,
    QUANTITY_SPECS,
    append_source_samples,
    axis_unit,
    figure_group_key,
    figure_title,
    load_detail_manifest,
    mean_band,
    moderate_profile_plot_contract,
    perturbation_interval_bounds,
    read_json,
    repo_rel,
    representative_timing,
    safe_slug,
    scale_profile_samples,
    simulate_extlqg_arrays,
    timing_bins_for_rows,
    row_timing_label,
)
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
    _simulate_robust_output_feedback_perturbed,
    materialize_gru_perturbation_response,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.viz import profile_comparison_grid


SOURCE_TOPIC = "moderate_perturbation_profiles"
OVERLAY_TOPIC = "pgd_1p05_moderate_perturbation_profiles_overlay"
OUTPUT_TAG = "pgd_1p05_moderate_profile_overlay"
SOURCE_TAG = "validation_selected_moderate"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
FIGURE_ROOT = REPO_ROOT / "results" / ISSUE / "figures"
SOURCE_SPEC = FIGURE_ROOT / SOURCE_TOPIC / "spec.json"
SOURCE_MANIFEST = NOTES_DIR / f"gru_perturbation_response_{SOURCE_TAG}_manifest.json"
PGD_MANIFEST = NOTES_DIR / f"gru_perturbation_response_{OUTPUT_TAG}_manifest.json"
PGD_NOTE = NOTES_DIR / f"gru_perturbation_response_{OUTPUT_TAG}.md"
PGD_REGEN_SPEC = NOTES_DIR / f"gru_perturbation_response_{OUTPUT_TAG}_manifest_regeneration_spec.json"
FEEDBACK_SCALE_MANIFEST = NOTES_DIR / "gru_evaluation_diagnostics_pgd_1p05_reach_context_diagnostics.json"
OVERLAY_SPEC = FIGURE_ROOT / OVERLAY_TOPIC / "spec.json"
OVERLAY_BULK = REPO_ROOT / "_artifacts" / ISSUE / "figures" / OVERLAY_TOPIC
PGD_BULK = REPO_ROOT / "_artifacts" / ISSUE / OVERLAY_TOPIC / "perturbation_response"
NOTE_PATH = NOTES_DIR / f"{OVERLAY_TOPIC}.md"

PAIRINGS = {
    "open_loop_small": "small",
    "open_loop_moderate": "moderate",
    "open_loop_stress": "stress",
}
SOURCE_COLORS = {
    "no_pgd_gru": "#2563eb",
    "extlqg6d": "#c2410c",
    "pgd_gru": "#15803d",
    "robust_output_feedback6d": "#7c3aed",
}
SOURCE_ORDER = ("no_pgd_gru", "extlqg6d", "pgd_gru", "robust_output_feedback6d")
ROBUST_SUPPORTED_CHANNELS = {"initial_state", "command_input", "process_epsilon"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pgd-eval",
        action="store_true",
        help="Require an existing PGD perturbation-response detail manifest.",
    )
    args = parser.parse_args()

    ensure_inputs()
    if not args.skip_pgd_eval:
        materialize_pgd_bulk_arrays()
    pgd_manifest = read_json(PGD_MANIFEST)
    pgd_detail = load_detail_manifest(pgd_manifest)
    source_manifest = read_json(SOURCE_MANIFEST)
    source_detail = load_detail_manifest(source_manifest)

    extlqg_context = _build_extlqg_comparator_context(physical_dim=6)
    robust_context = build_robust_output_feedback_6d_context()
    spec = materialize_overlay_figures(
        source_manifest=source_manifest,
        source_detail=source_detail,
        pgd_manifest=pgd_manifest,
        pgd_detail=pgd_detail,
        extlqg_context=extlqg_context,
        robust_context=robust_context,
    )
    write_note(spec)
    print(
        json.dumps(
            {
                "status": "materialized",
                "topic": OVERLAY_TOPIC,
                "spec": repo_rel(OVERLAY_SPEC),
                "figure_count": spec["figure_count"],
                "bulk_dir": repo_rel(OVERLAY_BULK),
            },
            indent=2,
            sort_keys=True,
        )
    )


def ensure_inputs() -> None:
    missing = [
        path
        for path in (SOURCE_SPEC, SOURCE_MANIFEST, FEEDBACK_SCALE_MANIFEST)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing required source inputs: " + ", ".join(repo_rel(path) for path in missing)
        )


def materialize_pgd_bulk_arrays() -> None:
    materialize_gru_perturbation_response(
        source_experiment=ISSUE,
        result_experiment=ISSUE,
        run_ids=tuple(PAIRINGS.values()),
        labels=tuple(PAIRINGS.values()),
        n_rollout_trials=64,
        output_path=PGD_MANIFEST,
        note_path=PGD_NOTE,
        bulk_dir=PGD_BULK,
        regeneration_spec_path=PGD_REGEN_SPEC,
        bank_mode="calibrated",
        calibration_level="moderate",
        calibration_reach=0.15,
        feedback_scale_manifest_path=FEEDBACK_SCALE_MANIFEST,
        extlqg_physical_dim=6,
        write_bulk_arrays=True,
        repo_root=REPO_ROOT,
    )


def materialize_overlay_figures(
    *,
    source_manifest: Mapping[str, Any],
    source_detail: Mapping[str, Any],
    pgd_manifest: Mapping[str, Any],
    pgd_detail: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> dict[str, Any]:
    OVERLAY_BULK.mkdir(parents=True, exist_ok=True)
    figure_specs = []
    missing_pgd_rows = []
    for open_loop_run, pgd_run_id in PAIRINGS.items():
        source_run = source_detail["runs"][open_loop_run]
        pgd_run = pgd_detail["runs"][pgd_run_id]
        pgd_by_perturbation_id = {
            str(row["perturbation_id"]): row
            for row in pgd_run["perturbations"]
            if row.get("status") == "evaluated"
            and row.get("perturbation", {}).get("level_name") == "moderate"
        }
        rows = [
            row
            for row in source_run["perturbations"]
            if row.get("status") == "evaluated"
            and row.get("perturbation", {}).get("level_name") == "moderate"
        ]
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[figure_group_key(row)].append(row)
        for group_key in sorted(groups):
            group_rows = groups[group_key]
            title = f"{open_loop_run} vs {pgd_run_id}: {figure_title(group_rows[0])}"
            for figure_kind in ("trajectory", "residual"):
                trace_summary = TraceSummary()
                fig = build_overlay_figure(
                    group_rows,
                    source_run=source_run,
                    pgd_run=pgd_run,
                    pgd_by_perturbation_id=pgd_by_perturbation_id,
                    extlqg_context=extlqg_context,
                    robust_context=robust_context,
                    figure_kind=figure_kind,
                    title=f"{title}: moderate {figure_kind}",
                    trace_summary=trace_summary,
                )
                path = (
                    OVERLAY_BULK
                    / safe_slug(open_loop_run)
                    / f"{safe_slug(group_key)}__{figure_kind}.html"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(path, include_plotlyjs="cdn")
                missing_ids = sorted(
                    str(row["perturbation_id"])
                    for row in group_rows
                    if str(row["perturbation_id"]) not in pgd_by_perturbation_id
                )
                missing_pgd_rows.extend(
                    {
                        "open_loop_run_id": open_loop_run,
                        "pgd_run_id": pgd_run_id,
                        "perturbation_id": perturbation_id,
                    }
                    for perturbation_id in missing_ids
                )
                figure_specs.append(
                    {
                        "open_loop_run_id": open_loop_run,
                        "pgd_run_id": pgd_run_id,
                        "group_key": group_key,
                        "title": title,
                        "figure_kind": figure_kind,
                        "n_rows": len(group_rows),
                        "html": repo_rel(path),
                        "source_counts": trace_summary.source_counts,
                        "pgd_missing_rows": missing_ids,
                        "robust_hinf_unsupported_rows": trace_summary.robust_unsupported,
                    }
                )
    spec = {
        "schema_version": "rlrmp.c92_pgd_1p05_moderate_perturbation_profiles_overlay.v1",
        "issue": ISSUE,
        "physical_level": "moderate",
        "source_topic": SOURCE_TOPIC,
        "source_figure_spec": repo_rel(SOURCE_SPEC),
        "source_manifest": repo_rel(SOURCE_MANIFEST),
        "source_bulk_detail_manifest": source_manifest.get("bulk_detail_manifest"),
        "pgd_manifest": repo_rel(PGD_MANIFEST),
        "pgd_bulk_detail_manifest": pgd_manifest.get("bulk_detail_manifest"),
        "run_pairings": PAIRINGS,
        "figure_count": len(figure_specs),
        "plot_contract": overlay_plot_contract(robust_context),
        "missing_pgd_row_count": len(missing_pgd_rows),
        "missing_pgd_rows": missing_pgd_rows[:50],
        "figures": figure_specs,
    }
    OVERLAY_SPEC.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY_SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def build_overlay_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_run: Mapping[str, Any],
    pgd_run: Mapping[str, Any],
    pgd_by_perturbation_id: Mapping[str, Mapping[str, Any]],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    figure_kind: Literal["trajectory", "residual"],
    title: str,
    trace_summary: "TraceSummary",
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
        traces = collect_overlay_traces(
            timing_rows,
            source_run=source_run,
            pgd_run=pgd_run,
            pgd_by_perturbation_id=pgd_by_perturbation_id,
            extlqg_context=extlqg_context,
            robust_context=robust_context,
            figure_kind=figure_kind,
            trace_summary=trace_summary,
        )
        for row_index, (quantity, _quantity_name, unit) in enumerate(QUANTITY_SPECS, start=1):
            for source in SOURCE_ORDER:
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
        width=max(1080, 300 * n_cols),
        height=860,
        legend_title_text="Source / trace",
        margin={"l": 72, "r": 24, "t": 100, "b": 70},
    )
    for col_index in range(1, n_cols + 1):
        fig.update_xaxes(title_text="time from movement onset (s)", row=len(QUANTITY_SPECS), col=col_index)
    return fig


def collect_overlay_traces(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_run: Mapping[str, Any],
    pgd_run: Mapping[str, Any],
    pgd_by_perturbation_id: Mapping[str, Mapping[str, Any]],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    figure_kind: Literal["trajectory", "residual"],
    trace_summary: "TraceSummary",
) -> dict[tuple[str, str, str, str], np.ndarray]:
    trace_samples: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        sign = int(row.get("sign") or row.get("perturbation", {}).get("sign") or 1)
        perturbation_id = str(row["perturbation_id"])
        source_path = source_run.get("bulk_files", {}).get(perturbation_id)
        if source_path:
            with np.load(REPO_ROOT / source_path) as arrays:
                append_source_samples(
                    trace_samples,
                    arrays,
                    source="no_pgd_gru",
                    sign=sign,
                    figure_kind=figure_kind,
                )
                trace_summary.increment("no_pgd_gru")
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
                trace_summary.increment("extlqg6d")
        pgd_row = pgd_by_perturbation_id.get(perturbation_id)
        pgd_path = pgd_run.get("bulk_files", {}).get(perturbation_id) if pgd_row else None
        if pgd_path:
            with np.load(REPO_ROOT / pgd_path) as arrays:
                append_source_samples(
                    trace_samples,
                    arrays,
                    source="pgd_gru",
                    sign=sign,
                    figure_kind=figure_kind,
                )
                trace_summary.increment("pgd_gru")
        if robust_supported(row):
            try:
                robust_arrays = simulate_robust_arrays(row["perturbation"], robust_context)
            except ValueError:
                robust_arrays = None
            if robust_arrays is not None:
                append_source_samples(
                    trace_samples,
                    robust_arrays,
                    source="robust_output_feedback6d",
                    sign=sign,
                    figure_kind=figure_kind,
                )
                trace_summary.increment("robust_output_feedback6d")
        else:
            trace_summary.robust_unsupported.append(
                {
                    "perturbation_id": perturbation_id,
                    "channel": str(row.get("channel") or row.get("perturbation", {}).get("channel")),
                }
            )
    return {
        key: np.concatenate(samples, axis=0)
        for key, samples in trace_samples.items()
        if samples
    }


def robust_supported(row: Mapping[str, Any]) -> bool:
    channel = str(row.get("channel") or row.get("perturbation", {}).get("channel"))
    return channel in ROBUST_SUPPORTED_CHANNELS


def simulate_robust_arrays(
    perturbation: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    base = context["base_evaluation"]
    perturbed, _initial_state, _adapter = _simulate_robust_output_feedback_perturbed(
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


def build_robust_output_feedback_6d_context() -> dict[str, Any]:
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
    }
    if contract["state_dim"] != 36 or contract["physical_dim"] != 6:
        raise ValueError(f"unexpected 6D H-infinity contract: {contract}")
    return {
        "plant": plant,
        "schedule": schedule,
        "config": config,
        "solution": solution,
        "gains": gains,
        "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "gamma": solution.gamma,
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": _evaluation_from_extlqg_rollout(base_rollout, initial_state=x0),
        "contract": contract,
    }


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
    label = f"{source_label(source)} {variant} {coord}"
    legend_group = f"{source}-{variant}-{coord}"
    if samples.shape[0] > 1:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=high,
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=legend_group,
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
                legendgroup=legend_group,
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
            legendgroup=legend_group,
            showlegend=showlegend,
            line={
                "color": color,
                "dash": COORD_DASH[coord],
                "width": 1.25 if variant == "clean" else 2.25,
            },
            opacity=0.42 if variant == "clean" else 0.95,
        ),
        row=row,
        col=col,
    )


def source_label(source: str) -> str:
    return {
        "no_pgd_gru": "No-PGD GRU",
        "extlqg6d": "6D extLQG",
        "pgd_gru": "PGD 1.05 GRU",
        "robust_output_feedback6d": "6D output-feedback H-infinity",
    }[source]


def band_color(source: str) -> str:
    return {
        "no_pgd_gru": "rgba(37,99,235,0.10)",
        "extlqg6d": "rgba(194,65,12,0.10)",
        "pgd_gru": "rgba(21,128,61,0.10)",
        "robust_output_feedback6d": "rgba(124,58,237,0.10)",
    }[source]


def overlay_plot_contract(robust_context: Mapping[str, Any]) -> dict[str, Any]:
    contract = moderate_profile_plot_contract()
    contract["source_traces"] = {
        source: {
            "label": source_label(source),
            "color": SOURCE_COLORS[source],
        }
        for source in SOURCE_ORDER
    }
    contract["comparator_contract"]["output_feedback_hinf"] = robust_context["contract"]
    contract["comparator_contract"]["output_feedback_hinf"]["supported_perturbation_channels"] = sorted(
        ROBUST_SUPPORTED_CHANNELS
    )
    contract["comparator_contract"]["output_feedback_hinf"]["unsupported_perturbation_channels"] = (
        "sensory_feedback, delayed_observation, and target_stream are not replayed by "
        "the current robust analytical perturbation API."
    )
    return contract


def write_note(spec: Mapping[str, Any]) -> None:
    unsupported_rows = sum(
        len(figure["robust_hinf_unsupported_rows"])
        for figure in spec["figures"]
        if figure["figure_kind"] == "residual"
    )
    lines = [
        "# PGD 1.05 Moderate Perturbation Profile Overlay",
        "",
        f"- Source figure family identified: `{SOURCE_TOPIC}`.",
        f"- New overlay topic: `{OVERLAY_TOPIC}`.",
        f"- Figure spec: `{repo_rel(OVERLAY_SPEC)}`.",
        f"- HTML render directory: `{repo_rel(OVERLAY_BULK)}`.",
        f"- Figure count: `{spec['figure_count']}`.",
        "- Rows: `open_loop_small` vs `small`, `open_loop_moderate` vs `moderate`, "
        "`open_loop_stress` vs `stress`.",
        "- Traces: no-PGD GRU, 6D extLQG, PGD 1.05 GRU, and 6D output-feedback H-infinity.",
        "- H-infinity perturbation traces are rendered for analytical replay-supported channels "
        "(`initial_state`, `command_input`, `process_epsilon`). Sensory-feedback, delayed-observation, "
        f"and target-stream rows are marked unsupported in the spec (`{unsupported_rows}` residual-row "
        "entries across rendered figures).",
        "",
    ]
    update_marked_section(NOTE_PATH, OVERLAY_TOPIC, "\n".join(lines) + "\n")


class TraceSummary:
    def __init__(self) -> None:
        self.source_counts = {source: 0 for source in SOURCE_ORDER}
        self.robust_unsupported: list[dict[str, str]] = []

    def increment(self, source: str) -> None:
        self.source_counts[source] += 1


if __name__ == "__main__":
    main()
