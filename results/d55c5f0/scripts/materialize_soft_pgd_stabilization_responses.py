"""Materialize d55 soft-PGD stabilization perturbation response figures."""

from __future__ import annotations
from rlrmp.viz.colors import band_color as canonical_band_color

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from _soft_pgd_materializer_common import (
    ISSUE,
    SOFT_ROWS,
    SOFT_RUN_IDS,
    assert_soft_inputs_ready,
    load_c92_module,
    repo_rel,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    _simulate_extlqg_perturbed,
    _simulate_robust_output_feedback_perturbed,
)
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.viz.figures import (
    build_stabilization_response_family_figure as canonical_build_family_figure,
)


TOPIC = "soft_pgd_stabilization_perturbation_responses"
MARKER = TOPIC
SCHEMA_VERSION = "rlrmp.d55c5f0.soft_pgd_stabilization_responses.v1"
DETAIL_SCHEMA_VERSION = "rlrmp.d55c5f0.soft_pgd_stabilization_detail.v1"
ROBUST_SUPPORTED_CHANNELS = {"command_input", "process_epsilon"}
FEEDBACK_QUANTITY_BY_FAMILY = {
    "feedback_position": "position",
    "feedback_velocity": "velocity",
    "feedback_force_filter": "force_filter",
}


band_color = canonical_band_color


SOURCE_STYLES = {
    row.run_id: {
        "label": row.legend,
        "color": row.color,
        "band": band_color(row.color, alpha=0.10),
    }
    for row in SOFT_ROWS
} | {
    "extlqg6d": {
        "label": "6D extLQG",
        "color": "#c2410c",
        "band": "rgba(194,65,12,0.08)",
    },
    "robust_output_feedback6d": {
        "label": "6D output-feedback H-infinity",
        "color": "#15803d",
        "band": "rgba(21,128,61,0.08)",
    },
}
ANALYTICAL_SOURCES = ("extlqg6d", "robust_output_feedback6d")


def main() -> None:
    """Run local materialization and write the requested d55 outputs."""

    assert_soft_inputs_ready()
    base, overlay = load_reference_materializers()
    patch_reference_materializers(base)
    repo_root = Path(REPO_ROOT).resolve()
    summary = materialize_row_detail(base=base, repo_root=repo_root)
    detail = split_detail(summary=summary)
    output_dirs = output_directories(repo_root)

    extlqg_context = overlay._build_extlqg_comparator_context(physical_dim=6)
    robust_context = overlay.build_robust_output_feedback_6d_context()
    figures = materialize_figures(
        base=base,
        detail=detail,
        summary=summary,
        extlqg_context=extlqg_context,
        robust_context=robust_context,
        repo_root=repo_root,
        figure_dir=output_dirs["figure_dir"],
    )
    spec = build_spec(
        base=base,
        summary=summary,
        figures=figures,
        robust_context=robust_context,
        figure_dir=output_dirs["figure_dir"],
        detail_path=output_dirs["detail_path"],
    )

    write_compact_json(output_dirs["detail_path"], detail)
    write_compact_json(output_dirs["summary_path"], summary)
    write_compact_json(output_dirs["spec_path"], spec)
    update_marked_section(output_dirs["note_path"], MARKER, render_note(spec))
    validate_spec(base=base, spec=spec)
    print(json.dumps(spec, indent=2, sort_keys=True))


def load_reference_materializers() -> tuple[Any, Any]:
    """Load c92 reference modules, preloading their local dependency names."""

    base = load_c92_module(
        "materialize_pgd_1p05_stabilization_diagnostics",
        "materialize_pgd_1p05_stabilization_diagnostics.py",
    )
    overlay = load_c92_module(
        "materialize_pgd_1p05_perturbation_response_overlays",
        "materialize_pgd_1p05_perturbation_response_overlays.py",
    )
    return base, overlay


def patch_reference_materializers(base: Any) -> None:
    """Point reused c92 stabilization evaluation at d55."""

    base.ISSUE = ISSUE


def output_directories(repo_root: Path) -> dict[str, Path]:
    """Return and create scoped output locations."""

    figure_spec_dir = mkdir_p(repo_root / "results" / ISSUE / "figures" / TOPIC)
    figure_dir = mkdir_p(repo_root / "_artifacts" / ISSUE / "figures" / TOPIC)
    detail_dir = mkdir_p(repo_root / "_artifacts" / ISSUE / "stabilization_diagnostics" / TOPIC)
    notes_dir = mkdir_p(repo_root / "results" / ISSUE / "notes")
    return {
        "figure_dir": figure_dir,
        "detail_dir": detail_dir,
        "spec_path": figure_spec_dir / "spec.json",
        "note_path": notes_dir / f"{TOPIC}.md",
        "detail_path": detail_dir / "per_probe_detail.json",
        "summary_path": detail_dir / "summary.json",
    }


def materialize_row_detail(*, base: Any, repo_root: Path) -> dict[str, Any]:
    """Evaluate the three d55 soft-PGD rows on stabilization probes."""

    rows = [
        base.evaluate_row(
            base.RowSpec(row.run_id, row.training_key, "moderate"),
            repo_root=repo_root,
        )
        for row in SOFT_ROWS
    ]
    return {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "issue": ISSUE,
        "source_experiment": ISSUE,
        "task_label": "stabilization task endpoint response",
        "rows_requested": list(SOFT_RUN_IDS),
        "probe_contract": base.probe_contract(),
        "rows": rows,
    }


def split_detail(*, summary: dict[str, Any]) -> dict[str, Any]:
    """Move per-probe rows into the bulk detail payload."""

    detail_rows = {}
    for row in summary["rows"]:
        detail_rows[str(row["run_id"])] = row.pop("per_probe_detail")
    return {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "issue": ISSUE,
        "detail_role": "per-probe scalar and trajectory diagnostics",
        "rows": detail_rows,
    }


def materialize_figures(
    *,
    base: Any,
    detail: Mapping[str, Any],
    summary: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    repo_root: Path,
    figure_dir: Path,
) -> list[dict[str, Any]]:
    """Write one HTML/PNG figure per perturbation family."""

    summary_by_run = {str(row["run_id"]): row for row in summary["rows"]}
    figure_specs = []
    png_errors: dict[str, str] = {}
    for family in base.PERTURBATION_FAMILY_ORDER:
        fig, coverage, event_markers, unavailable = build_family_figure(
            base=base,
            family=family,
            detail=detail,
            summary_by_run=summary_by_run,
            extlqg_context=extlqg_context,
            robust_context=robust_context,
        )
        html_path = figure_dir / f"{family}.html"
        png_path = figure_dir / f"{family}.png"
        png_status, png_renderer = base.write_figure_outputs(
            fig=fig,
            html_path=html_path,
            png_path=png_path,
            png_errors=png_errors,
            error_key=family,
        )
        figure_specs.append(
            {
                "role": "perturbation_family",
                "family": family,
                "title": base.FAMILY_LABELS[family],
                "figure_kind": "stabilization_task_soft_pgd_response_grid",
                "layout": {
                    "rows": 3,
                    "cols": 3,
                    "row_axis": "response_state",
                    "row_order": list(base.RESPONSE_VARIABLE_ORDER),
                    "col_axis": "soft_pgd_row",
                    "col_order": list(SOFT_RUN_IDS),
                },
                "html": repo_rel(html_path),
                "png": repo_rel(png_path) if png_status == "written" else None,
                "png_status": png_status,
                "png_renderer": png_renderer,
                "png_bytes": base.png_size(png_path) if png_status == "written" else None,
                "coverage": coverage,
                "perturbation_event_markers": event_markers,
                "analytical_unavailable": unavailable,
            }
        )
    if png_errors:
        for figure in figure_specs:
            figure["png_errors"] = png_errors
    return figure_specs


def build_family_figure(
    *,
    base: Any,
    family: str,
    detail: Mapping[str, Any],
    summary_by_run: Mapping[str, Mapping[str, Any]],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
) -> tuple[go.Figure, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one response-state-by-soft-row grid."""

    def cell_context(response_variable: str, soft_row: Any) -> dict[str, Any]:
        timing = summary_by_run[soft_row.run_id]["timing"]
        dt = float(summary_by_run[soft_row.run_id]["dt_s"])
        baseline_rows = family_rows(detail=detail, run_id=soft_row.run_id, family=family)
        profile = base.aggregate_family_response_profile(
            baseline_rows, response_variable=response_variable
        )
        return {
            "timing": timing, "dt": dt,
            "baseline_rows": baseline_rows,
            "learned": [{"source": soft_row.run_id, "run_id": soft_row.run_id, "profile": profile}],
            "cache_prefix": (soft_row.run_id,),
            "coverage_metadata": {"column": soft_row.run_id},
            "identity_metadata": {"soft_row": soft_row.run_id},
        }

    return canonical_build_family_figure(
        family=family,
        response_variables=base.RESPONSE_VARIABLE_ORDER,
        columns=SOFT_ROWS,
        cell_context=cell_context,
        analytical_sources=ANALYTICAL_SOURCES,
        analytical_profile=lambda **kwargs: analytical_profile(
            base=base, extlqg_context=extlqg_context, robust_context=robust_context, **kwargs
        ),
        add_profile_traces=lambda **kwargs: add_profile_traces(base, **kwargs),
        coverage_row=coverage_row,
        add_unsupported_annotation=add_unsupported_annotation,
        infer_event_marker=base.infer_perturbation_event_marker,
        add_event_marker=base.add_perturbation_event_marker,
        response_label=lambda response: base.RESPONSE_VARIABLE_SPECS[response]["label"],
        column_label=lambda row: row.label,
        response_axis_title=lambda response: base.RESPONSE_VARIABLE_SPECS[response]["axis_title"],
        title=f"d55 soft-PGD stabilization task response: {base.FAMILY_LABELS[family]}",
        width=1420,
        horizontal_spacing=0.055,
    )


def family_rows(*, detail: Mapping[str, Any], run_id: str, family: str) -> list[Mapping[str, Any]]:
    """Return evaluated detail rows for one run/family."""

    return [
        row
        for row in detail["rows"][run_id]
        if row.get("status") == "evaluated" and row.get("family") == family
    ]


def analytical_profile(
    *,
    base: Any,
    source: str,
    family: str,
    response_variable: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an aggregate analytical response profile for one source."""

    if source == "robust_output_feedback6d" and not robust_family_supported(baseline_rows):
        return {
            "status": "unsupported",
            "reason": "robust output-feedback replay supports command_input and process_epsilon only",
        }
    detail_rows = []
    blocked: list[str] = []
    for row in baseline_rows:
        perturbation = analytical_perturbation_from_detail(row, timing=timing)
        try:
            if source == "extlqg6d":
                context = extlqg_context
                perturbed, _initial_state, _adapter = _simulate_extlqg_perturbed(
                    perturbation,
                    context=context,
                )
            else:
                context = robust_context
                perturbed, _initial_state, _adapter = _simulate_robust_output_feedback_perturbed(
                    perturbation,
                    context=context,
                )
        except (KeyError, ValueError, RuntimeError) as exc:
            blocked.append(f"{row['perturbation_id']}:{type(exc).__name__}:{exc}")
            continue
        probe = base.ProbeSpec(
            perturbation_id=str(row["perturbation_id"]),
            group=str(row["group"]),
            family=str(row["family"]),
            row=perturbation,
            direction=tuple(float(value) for value in row["direction"]),
            amplitude=float(row["amplitude"]),
            units=str(row["units"]),
        )
        detail_rows.append(
            base.summarize_probe(
                probe=probe,
                base=context["base_evaluation"],
                perturbed=perturbed,
                pulse_start=int(timing["pulse_start_step"]),
            )
            | {"status": "evaluated", "adapter": {"status": "analytical", "source": source}}
        )
    if not detail_rows:
        return {
            "status": "blocked",
            "reason": "; ".join(blocked) if blocked else "no analytical rows evaluated",
        }
    return {
        "status": "available",
        "profile": base.aggregate_family_response_profile(
            detail_rows,
            response_variable=response_variable,
        ),
        "blocked_rows": blocked,
    }


def robust_family_supported(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether all rows are on robust replay-supported channels."""

    return all(str(row.get("channel")) in ROBUST_SUPPORTED_CHANNELS for row in rows)


def analytical_perturbation_from_detail(
    row: Mapping[str, Any],
    *,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a stabilization detail row to the analytical perturbation schema."""

    provenance = row.get("adapter", {}).get("adapter_provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    direction = np.asarray(row["direction"], dtype=np.float64)
    axis = "x" if abs(float(direction[0])) >= abs(float(direction[1])) else "y"
    sign = 1 if float(direction[0 if axis == "x" else 1]) >= 0.0 else -1
    start = int(provenance.get("relative_start_time_index", timing["pulse_start_step"]))
    duration = int(provenance.get("duration_steps", timing["pulse_duration_steps"]))
    perturbation = {
        "perturbation_id": str(row["perturbation_id"]),
        "channel": str(row["channel"]),
        "amplitude": float(row["amplitude"]),
        "units": str(row["units"]),
        "axis": axis,
        "sign": sign,
        "timing": {
            "start_time_index": start,
            "duration_steps": duration,
            "timing_bin": "steady_state_endpoint",
        },
    }
    family = str(row["family"])
    if str(row["channel"]) == "sensory_feedback":
        perturbation["family"] = "sensory_feedback_offset"
        perturbation["feedback_quantity"] = FEEDBACK_QUANTITY_BY_FAMILY[family]
        feedback_index = provenance.get("axis_index")
        if feedback_index is not None:
            perturbation["feedback_payload_index"] = int(feedback_index)
        if family == "feedback_force_filter":
            perturbation["force_filter_feedback_only"] = True
    elif str(row["channel"]) == "process_epsilon":
        perturbation["family"] = family
        epsilon_index = provenance.get("epsilon_index")
        if epsilon_index is not None:
            perturbation["epsilon_index"] = int(epsilon_index)
    else:
        perturbation["family"] = family
    return perturbation


def add_profile_traces(
    base: Any,
    fig: go.Figure,
    *,
    profile: Mapping[str, Any],
    source: str,
    dt: float,
    row: int,
    col: int,
    legend_seen: set[tuple[str, str]],
) -> None:
    """Add aligned and orthogonal mean/SEM traces for one source profile."""

    style = SOURCE_STYLES[source]
    x = np.asarray(profile["relative_time_steps"], dtype=np.float64) * float(dt)
    aligned = np.asarray(profile["aligned_mean"], dtype=np.float64)
    aligned_sem = np.asarray(profile["aligned_sem"], dtype=np.float64)
    orthogonal = np.asarray(profile["orthogonal_mean"], dtype=np.float64)
    orthogonal_sem = np.asarray(profile["orthogonal_sem"], dtype=np.float64)
    for component, mean, sem, dash, opacity in (
        ("aligned", aligned, aligned_sem, "solid", 0.96),
        ("orthogonal", orthogonal, orthogonal_sem, "dot", 0.68),
    ):
        legend_key = (source, component)
        legendgroup = f"{source}-{component}"
        base.add_mean_sem_trace(
            fig,
            x=x,
            mean=mean,
            sem=sem,
            name=f"{style['label']} {component}",
            legendgroup=legendgroup,
            color=style["color"],
            band_color=style["band"],
            row=row,
            col=col,
            showlegend=legend_key not in legend_seen,
        )
        fig.data[-1].line.dash = dash
        fig.data[-1].opacity = opacity
        legend_seen.add(legend_key)


def add_unsupported_annotation(fig: go.Figure, *, text: str, row: int, col: int) -> None:
    """Add a compact in-panel unsupported-comparator marker."""

    fig.add_annotation(
        x=0.5,
        y=0.96,
        xref="x domain",
        yref="y domain",
        text=text,
        showarrow=False,
        font={"size": 10, "color": "#475569"},
        bgcolor="rgba(248,250,252,0.82)",
        bordercolor="rgba(148,163,184,0.55)",
        borderwidth=1,
        row=row,
        col=col,
    )


def coverage_row(
    *,
    source: str,
    family: str,
    response_variable: str,
    column: str,
    run_id: str | None,
    profile: Mapping[str, Any],
    analytical_status: str,
) -> dict[str, Any]:
    """Return one source coverage entry."""

    return {
        "family": family,
        "response_variable": response_variable,
        "soft_row": column,
        "source": source,
        "source_label": SOURCE_STYLES[source]["label"],
        "run_id": run_id,
        "analytical_status": analytical_status,
        "n_evaluated_probes": int(profile["n_probes"]),
        "perturbation_ids": list(profile["perturbation_ids"]),
        "profile_source": dict(profile["profile_source"]),
        "unit": str(profile["unit"]),
    }


def build_spec(
    *,
    base: Any,
    summary: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]],
    robust_context: Mapping[str, Any],
    figure_dir: Path,
    detail_path: Path,
) -> dict[str, Any]:
    """Build the tracked figure spec."""

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "figure_topic": TOPIC,
        "figure_count": len(figures),
        "html_count": len(figures),
        "png_count": sum(1 for figure in figures if figure["png_status"] == "written"),
        "task_label": "stabilization task",
        "inputs": [
            {
                "role": "run_spec",
                "run_id": str(row["run_id"]),
                "path": str(row["run_spec_path"]),
            }
            for row in summary["rows"]
        ],
        "bulk_detail": repo_rel(detail_path),
        "bulk_figure_dir": repo_rel(figure_dir),
        "plot_contract": {
            "task_context": "stabilization task endpoint perturbation response",
            "figure_axis": "perturbation_family",
            "figure_order": list(base.PERTURBATION_FAMILY_ORDER),
            "grid": "three response-state rows by three soft-PGD row columns",
            "row_axis": "response_state",
            "row_order": list(base.RESPONSE_VARIABLE_ORDER),
            "col_axis": "soft_pgd_row",
            "col_order": list(SOFT_RUN_IDS),
            "response_states_are_not_perturbation_types": True,
            "perturbation_families_are_separate_figures": True,
            "trace_sources": {key: value["label"] for key, value in SOURCE_STYLES.items()},
            "analytical_comparator_contract": {
                "extlqg": {
                    "label": "6D extLQG",
                    "state_dim": 36,
                    "physical_dim": 6,
                    "disturbance_integrators_exposed": False,
                    "supported_perturbation_channels": [
                        "command_input",
                        "process_epsilon",
                        "sensory_feedback",
                    ],
                    "source": (
                        "rlrmp.analysis.pipelines.gru_perturbation_bank."
                        "_build_extlqg_comparator_context(physical_dim=6)"
                    ),
                },
                "output_feedback_hinf": robust_context["contract"]
                | {
                    "supported_perturbation_channels": sorted(ROBUST_SUPPORTED_CHANNELS),
                    "unsupported_perturbation_channels": ["sensory_feedback"],
                    "unsupported_policy": (
                        "sensory-feedback stabilization subplots are annotated and "
                        "listed in analytical_unavailable; no H-infinity trace is faked"
                    ),
                },
            },
            "perturbation_event_marker": {
                "display_preference": "shaded vertical onset-to-offset band",
                "x_axis_reference": "seconds relative to perturbation onset",
                "onset_source": (
                    "detail.rows[*].adapter.adapter_provenance.relative_start_time_index "
                    "with summary timing fallback"
                ),
                "duration_source": (
                    "detail.rows[*].adapter.adapter_provenance.duration_steps "
                    "with summary timing fallback"
                ),
                "fallback_when_duration_missing": "vertical onset line at x=0",
            },
            "orthogonal_trace": (
                "lower-emphasis signed projection onto the +90-degree right-handed "
                "orthogonal direction"
            ),
            "uncertainty_band": (
                "SEM across the four signed x/y probes for GRU traces; analytical "
                "traces aggregate the same four probe definitions"
            ),
        },
        "figures": list(figures),
    }


def validate_spec(*, base: Any, spec: Mapping[str, Any]) -> None:
    """Assert the requested figure and comparator coverage."""

    figures = list(spec["figures"])
    if len(figures) != len(base.PERTURBATION_FAMILY_ORDER):
        raise ValueError(f"expected 5 figures, got {len(figures)}")
    if [figure["family"] for figure in figures] != list(base.PERTURBATION_FAMILY_ORDER):
        raise ValueError("perturbation family order mismatch")
    for figure in figures:
        layout = figure["layout"]
        if (layout["rows"], layout["cols"]) != (3, 3):
            raise ValueError(f"{figure['family']} should be 3x3, got {layout}")
        if list(layout["row_order"]) != list(base.RESPONSE_VARIABLE_ORDER):
            raise ValueError(f"{figure['family']} response row order mismatch")
        if list(layout["col_order"]) != list(SOFT_RUN_IDS):
            raise ValueError(f"{figure['family']} soft-row column order mismatch")
        common_sources = {"extlqg6d"}
        if figure["family"] in {"command_input_pulse", "process_epsilon_force_state_xy"}:
            common_sources.add("robust_output_feedback6d")
        seen = defaultdict(set)
        for row in figure["coverage"]:
            seen[(row["response_variable"], row["soft_row"])].add(row["source"])
            if int(row["n_evaluated_probes"]) != 4:
                raise ValueError(
                    f"{figure['family']} {row['source']} should have 4 probes, "
                    f"got {row['n_evaluated_probes']}"
                )
        for response in base.RESPONSE_VARIABLE_ORDER:
            for soft_row in SOFT_RUN_IDS:
                key = (response, soft_row)
                expected_sources = set(common_sources)
                expected_sources.add(soft_row)
                if seen[key] != expected_sources:
                    raise ValueError(
                        f"{figure['family']} {key} source mismatch: {sorted(seen[key])}"
                    )
        markers = list(figure["perturbation_event_markers"])
        if len(markers) != 9:
            raise ValueError(f"{figure['family']} should have 9 event markers, got {len(markers)}")
        for marker in markers:
            if marker["display"] != "duration_band":
                raise ValueError(f"{figure['family']} missing duration band: {marker}")


def render_note(spec: Mapping[str, Any]) -> str:
    """Render the tracked Markdown note."""

    unsupported = sum(
        1
        for figure in spec["figures"]
        for row in figure.get("analytical_unavailable", [])
        if row["source"] == "robust_output_feedback6d"
    )
    lines = [
        "# Soft-PGD Stabilization Perturbation Responses",
        "",
        "- Scope: stabilization-task endpoint perturbation response figures only.",
        "- Rows: `soft_pgd_ofb1p05`, `soft_pgd_ofb1p4`, and `soft_pgd_ofb1p8`.",
        "- Figure family: one figure per perturbation family "
        f"({', '.join(f'`{family}`' for family in spec['plot_contract']['figure_order'])}).",
        "- Per-figure layout: response-state rows (`command`, `position`, `velocity`) "
        "by soft-PGD row columns.",
        "- Perturbation timing: each subplot uses a shaded onset-to-offset band; "
        "duration comes from adapter provenance with summary timing fallback.",
        "- Analytical comparators: 6D extLQG is rendered for all five families. "
        "6D output-feedback H-infinity is rendered for `command_input_pulse` and "
        "`process_epsilon_force_state_xy`; sensory-feedback families are annotated "
        "as unsupported rather than faked.",
        f"- Tracked spec: `results/{ISSUE}/figures/{TOPIC}/spec.json`.",
        f"- Bulk figures: `{spec['bulk_figure_dir']}`.",
        f"- Bulk detail: `{spec['bulk_detail']}`.",
        f"- Figure count: `{spec['figure_count']}` HTML and `{spec['png_count']}` PNG.",
        "",
        "| Perturbation family | Layout | HTML | PNG | H-inf unsupported panels |",
        "|---|---|---|---|---:|",
    ]
    for figure in spec["figures"]:
        layout = figure["layout"]
        unsupported_count = sum(
            1
            for row in figure.get("analytical_unavailable", [])
            if row["source"] == "robust_output_feedback6d"
        )
        lines.append(
            f"| `{figure['family']}` | {layout['rows']}x{layout['cols']} | "
            f"`{figure['html']}` | `{figure['png']}` | {unsupported_count} |"
        )
    lines.extend(
        [
            "",
            f"Total unsupported H-infinity sensory-feedback subplot entries: `{unsupported}`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
