#!/usr/bin/env python
"""Materialize calibrated perturbation-bank profile figures for issue 3244f1a."""

from __future__ import annotations
from rlrmp.viz.traces import add_reduced_sample_trace

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

from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
    materialize_gru_evaluation_diagnostics,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    _build_extlqg_comparator_context,
    _simulate_extlqg_perturbed,
)
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT
from rlrmp.viz.figures import build_profile_family_figure


ISSUE = "3244f1a"
SOURCE_EXPERIMENT = "33b0dcb"
RUN_ID = "h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64"
RUN_LABEL = "no-PGD const_band16"
N_ROLLOUT_TRIALS = 8
DT = 0.01

NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
SCRIPTS_DIR = REPO_ROOT / "results" / ISSUE / "scripts"
FIGURE_SPEC_ROOT = REPO_ROOT / "results" / ISSUE / "figures" / "final_calibrated_bank_profiles"
FIGURE_BULK_ROOT = REPO_ROOT / "_artifacts" / ISSUE / "figures" / "final_calibrated_bank_profiles"
EVAL_BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "evaluation_diagnostics"
PERT_BULK_ROOT = REPO_ROOT / "_artifacts" / ISSUE / "perturbation_response"

EVAL_MANIFEST = NOTES_DIR / "gru_evaluation_diagnostics_const_band16_validation_selected.json"
PROFILE_NOTE = NOTES_DIR / "final_calibrated_bank_profile_figures.md"
FIGURE_README = FIGURE_SPEC_ROOT / "README.md"

LEVEL_CONFIGS = {
    "small": {
        "display": "small",
        "bank_level": "small",
        "fraction_of_reach": 0.05,
    },
    "medium": {
        "display": "medium",
        "bank_level": "moderate",
        "fraction_of_reach": 0.10,
    },
}

SOURCE_COLORS = {
    "gru": "#2563eb",
    "extlqg6d": "#c2410c",
}
COORD_DASH = {
    "orthogonal": "solid",
    "along": "dot",
}
QUANTITY_SPECS = (
    ("command", "Command", "N"),
    ("position", "Position", "m"),
    ("velocity", "Velocity", "m/s"),
)
TIMING_ORDER = {
    "movement_onset": 0,
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
    parser.add_argument("--skip-materialization", action="store_true")
    parser.add_argument("--n-rollout-trials", type=int, default=N_ROLLOUT_TRIALS)
    parser.add_argument(
        "--level",
        action="append",
        choices=tuple(LEVEL_CONFIGS),
        dest="levels",
        help="Figure/data level to materialize. Defaults to small and medium.",
    )
    args = parser.parse_args()
    level_keys = tuple(args.levels or ("small", "medium"))

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_SPEC_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_BULK_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_materialization or not EVAL_MANIFEST.exists():
        materialize_gru_evaluation_diagnostics(
            experiment=SOURCE_EXPERIMENT,
            run_ids=(RUN_ID,),
            labels=(RUN_LABEL,),
            output_path=EVAL_MANIFEST,
            bulk_dir=EVAL_BULK_DIR,
            n_rollout_trials=args.n_rollout_trials,
            regeneration_spec_path=_regeneration_spec_path(EVAL_MANIFEST),
            repo_root=REPO_ROOT,
        )

    extlqg_context = _build_extlqg_comparator_context(physical_dim=6)
    extlqg_peak_velocity = _extlqg_nominal_peak_velocity(extlqg_context)
    profile_specs = []
    for level_key in level_keys:
        paths = _level_paths(level_key)
        for path in (paths["figure_spec_dir"], paths["figure_bulk_dir"], paths["pert_bulk_dir"]):
            path.mkdir(parents=True, exist_ok=True)

        if args.skip_materialization and paths["pert_manifest"].exists():
            manifest = _read_json(paths["pert_manifest"])
        else:
            raise RuntimeError(
                "direct perturbation manifest and raw-array regeneration is retired; "
                "materialize the registered perturbation evaluation/analysis bundle "
                "through Feedbax custody, then rerun with --skip-materialization"
            )

        profile_spec = materialize_profile_figures(
            manifest,
            level_key=level_key,
            extlqg_context=extlqg_context,
            extlqg_peak_velocity=extlqg_peak_velocity,
        )
        write_compact_json(paths["figure_spec"], profile_spec)
        profile_specs.append(profile_spec)
        print(f"Wrote {len(profile_spec['figures'])} {level_key} profile figure pair(s).")
        print(f"Figure spec: {_repo_rel(paths['figure_spec'])}")

    write_figure_readme(profile_specs)
    write_profile_note(profile_specs)


def materialize_profile_figures(
    manifest: Mapping[str, Any],
    *,
    level_key: str,
    extlqg_context: Mapping[str, Any],
    extlqg_peak_velocity: float,
) -> dict[str, Any]:
    paths = _level_paths(level_key)
    level = LEVEL_CONFIGS[level_key]
    detail_manifest = _load_perturbation_detail_manifest(manifest)
    run = detail_manifest["runs"][RUN_ID]
    rows = [row for row in run["perturbations"] if row.get("status") == "evaluated"]
    extlqg_statuses = [row.get("extlqg_comparator", {}).get("status") for row in rows]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _figure_group_key(row)
        groups[key].append(row)

    figure_specs = []
    for key in sorted(groups):
        group_rows = groups[key]
        title = _figure_title(group_rows[0])
        trajectory = _build_family_figure(
            group_rows,
            run=run,
            extlqg_context=extlqg_context,
            extlqg_peak_velocity=extlqg_peak_velocity,
            figure_kind="trajectory",
            title=f"{title}: {level_key} perturbed traces",
        )
        residual = _build_family_figure(
            group_rows,
            run=run,
            extlqg_context=extlqg_context,
            extlqg_peak_velocity=extlqg_peak_velocity,
            figure_kind="residual",
            title=f"{title}: {level_key} perturbed-minus-clean residuals",
        )
        stem = _safe_slug(key)
        trajectory_path = paths["figure_bulk_dir"] / f"{stem}__trajectory.html"
        residual_path = paths["figure_bulk_dir"] / f"{stem}__residual.html"
        trajectory.write_html(trajectory_path, include_plotlyjs="cdn")
        residual.write_html(residual_path, include_plotlyjs="cdn")
        figure_specs.append(
            {
                "group_key": key,
                "title": title,
                "n_rows": len(group_rows),
                "timing_bins": _timing_bins(group_rows),
                "trajectory_html": _repo_rel(trajectory_path),
                "residual_html": _repo_rel(residual_path),
                "extlqg_available_rows": sum(
                    1
                    for row in group_rows
                    if row.get("extlqg_comparator", {}).get("status") == "available"
                ),
            }
        )
    return {
        "schema_version": "rlrmp.final_calibrated_bank_profile_figures.v2",
        "issue": ISSUE,
        "source_experiment": SOURCE_EXPERIMENT,
        "run_id": RUN_ID,
        "level": level_key,
        "calibration_level": str(level["bank_level"]),
        "calibration_level_display": str(level["display"]),
        "level_fraction_of_reach": float(level["fraction_of_reach"]),
        "bank_manifest": _repo_rel(paths["pert_manifest"]),
        "bulk_detail_manifest": manifest.get("bulk_detail_manifest"),
        "evaluation_manifest": _repo_rel(EVAL_MANIFEST),
        "figure_readme": _repo_rel(FIGURE_README),
        "coverage": {
            "evaluated_gru_rows": len(rows),
            "extlqg_available_rows": sum(status == "available" for status in extlqg_statuses),
            "extlqg_not_applicable_rows": sum(
                status == "not_applicable" for status in extlqg_statuses
            ),
            "extlqg_blocked_rows": sum(status == "blocked" for status in extlqg_statuses),
        },
        "coordinate_basis": {
            "reach_length_m": 0.15,
            "nominal_extlqg_peak_velocity_m_s": float(extlqg_peak_velocity),
            "direction_source": (
                "GRU target_position minus clean baseline start position for each "
                "replicate/trial; analytical extLQG uses clean start-to-end direction "
                "because the comparator rollout is already in the fixed +x target basis"
            ),
            "coordinates": {
                "orthogonal": "solid line; primary for perturbation residual inspection",
                "along": "dotted line",
            },
        },
        "aggregation": {
            "sign_handling": (
                "perturbation sign is multiplied into residuals; trajectory figures "
                "plot clean + sign-aligned residual so + and - rows do not cancel"
            ),
            "band": "central 80% interval across replicate/trial/component/sign rows",
            "timing_display": (
                "timing bins are separate columns with shaded pulse windows and "
                "dotted boundary lines"
            ),
            "residual_scaling": {
                "position": "percent of fixed 0.15 m reach length",
                "velocity": "percent of nominal 6D extLQG peak speed",
                "command": "native command units",
            },
        },
        "figures": figure_specs,
    }


def _build_family_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    extlqg_peak_velocity: float,
    figure_kind: Literal["trajectory", "residual"],
    title: str,
) -> go.Figure:
    """Build a calibrated family figure through the canonical profile grid."""

    return build_profile_family_figure(
        rows,
        quantity_specs=QUANTITY_SPECS,
        timing_bins_for_rows=_timing_bins,
        row_timing_label=_row_timing_label,
        representative_timing=_representative_timing,
        perturbation_interval_bounds=lambda timing: (
            float(timing.get("start_time_index", 0)) * DT,
            (float(timing.get("start_time_index", 0)) + float(timing.get("duration_steps", 1)))
            * DT,
        ),
        collect_traces=lambda timing_rows: _collect_traces_for_rows(
            timing_rows,
            run=run,
            extlqg_context=extlqg_context,
            figure_kind=figure_kind,
        ),
        trace_key=lambda source, variant, quantity, coord: (
            source,
            variant,
            quantity,
            coord,
        ),
        add_trace=lambda fig, samples, **kwargs: _add_profile_trace(
            fig,
            samples,
            figure_kind=figure_kind,
            extlqg_peak_velocity=extlqg_peak_velocity,
            **kwargs,
        ),
        sources=("gru", "extlqg6d"),
        variants=("clean", "perturbed") if figure_kind == "trajectory" else ("residual",),
        axis_unit=lambda quantity, unit: _axis_unit(
            quantity,
            figure_kind=figure_kind,
            native_unit=unit,
        ),
        figure_kind=figure_kind,
        title=title,
        width_min=950,
        width_per_column=270,
        height=840,
    )


def _collect_traces_for_rows(
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
            arrays = dict(np.load(REPO_ROOT / path_text))
            _append_source_samples(
                trace_samples,
                arrays,
                source="gru",
                sign=sign,
                figure_kind=figure_kind,
                target_position=_gru_target_position_samples(),
            )
        if row.get("extlqg_comparator", {}).get("status") == "available":
            try:
                ext_arrays = _simulate_extlqg_arrays(row["perturbation"], extlqg_context)
            except ValueError:
                ext_arrays = None
            if ext_arrays is not None:
                _append_source_samples(
                    trace_samples,
                    ext_arrays,
                    source="extlqg6d",
                    sign=sign,
                    figure_kind=figure_kind,
                    target_position=None,
                )
    return {
        key: np.concatenate(samples, axis=0) for key, samples in trace_samples.items() if samples
    }


def _append_source_samples(
    trace_samples: dict[tuple[str, str, str, str], list[np.ndarray]],
    arrays: Mapping[str, np.ndarray],
    *,
    source: str,
    sign: int,
    figure_kind: Literal["trajectory", "residual"],
    target_position: np.ndarray | None,
) -> None:
    directions, orthogonals, base_start = _clean_reach_basis(
        arrays["base_position"],
        target_position=target_position,
    )
    position_base = _as_samples(arrays["base_position"]) - base_start[:, None, :]
    position_delta = _as_samples(arrays["delta_position"])
    velocity_base = _as_samples(arrays["base_velocity"])
    velocity_delta = _as_samples(arrays["delta_velocity"])
    command_base = _as_samples(arrays["base_action"])
    command_delta = _as_samples(arrays["delta_action"])
    vectors = {
        "position": (position_base, position_delta),
        "velocity": (velocity_base, velocity_delta),
        "command": (command_base, command_delta),
    }
    coord_vectors = {
        "along": directions,
        "orthogonal": orthogonals,
    }
    for quantity, (base, delta) in vectors.items():
        for coord, basis in coord_vectors.items():
            base_projected = _project_samples(base, basis)
            residual_projected = float(sign) * _project_samples(delta, basis)
            if figure_kind == "trajectory":
                trace_samples[(source, "clean", quantity, coord)].append(base_projected)
                trace_samples[(source, "perturbed", quantity, coord)].append(
                    base_projected + residual_projected
                )
            else:
                trace_samples[(source, "residual", quantity, coord)].append(residual_projected)


def _simulate_extlqg_arrays(
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
        "perturbed_position": np.asarray(perturbed.position, dtype=np.float64),
        "delta_position": np.asarray(perturbed.position - base.position, dtype=np.float64),
        "base_velocity": np.asarray(base.velocity, dtype=np.float64),
        "perturbed_velocity": np.asarray(perturbed.velocity, dtype=np.float64),
        "delta_velocity": np.asarray(perturbed.velocity - base.velocity, dtype=np.float64),
        "base_action": np.asarray(base.command, dtype=np.float64),
        "perturbed_action": np.asarray(perturbed.command, dtype=np.float64),
        "delta_action": np.asarray(perturbed.command - base.command, dtype=np.float64),
    }


def _add_profile_trace(
    fig: go.Figure,
    samples: np.ndarray,
    *,
    source: str,
    variant: str,
    quantity: str,
    coord: str,
    figure_kind: Literal["trajectory", "residual"],
    extlqg_peak_velocity: float,
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    samples = _scale_profile_samples(
        samples,
        quantity=quantity,
        figure_kind=figure_kind,
        extlqg_peak_velocity=extlqg_peak_velocity,
    )
    label = f"{_source_label(source)} {variant} {coord}"
    add_reduced_sample_trace(
        fig,
        samples,
        reducer=_mean_band,
        row=row,
        col=col,
        name=label,
        legendgroup=f"{source}-{variant}-{coord}",
        color=SOURCE_COLORS[source],
        band_fill_color=_band_color(source),
        dash=COORD_DASH[coord],
        width=1.25 if variant == "clean" else 2.25,
        opacity=0.42 if variant == "clean" else 0.95,
        showlegend=showlegend,
        dt=DT,
        hovertemplate=f"{label}<br>{quantity}: %{{y:.4g}}<br>time: %{{x:.3f}}s<br>n: %{{customdata}}<extra></extra>",
    )


def _scale_profile_samples(
    samples: np.ndarray,
    *,
    quantity: str,
    figure_kind: Literal["trajectory", "residual"],
    extlqg_peak_velocity: float,
) -> np.ndarray:
    """Scale residual panels into the compact captioned units."""

    if figure_kind != "residual":
        return samples
    if quantity == "position":
        return 100.0 * samples / 0.15
    if quantity == "velocity":
        return 100.0 * samples / float(extlqg_peak_velocity)
    return samples


def _axis_unit(
    quantity: str,
    *,
    figure_kind: Literal["trajectory", "residual"],
    native_unit: str,
) -> str:
    if figure_kind == "residual" and quantity in {"position", "velocity"}:
        return "%"
    return native_unit


def _mean_band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    if array.shape[0] <= 1:
        return mean, mean, mean
    return mean, np.nanpercentile(array, 10.0, axis=0), np.nanpercentile(array, 90.0, axis=0)


def _clean_reach_basis(
    base_position: np.ndarray,
    *,
    target_position: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _as_samples(base_position)
    start = base[:, 0, :]
    if target_position is None:
        target = base[:, -1, :]
    else:
        target_samples = _as_samples(target_position)
        if target_samples.shape[0] == base.shape[0]:
            target = target_samples[:, -1, :]
        elif base.shape[0] % target_samples.shape[0] == 0:
            repeats = base.shape[0] // target_samples.shape[0]
            target = np.tile(target_samples[:, -1, :], (repeats, 1))
        else:
            raise ValueError(
                "target_position sample count must match or divide base_position "
                f"sample count; got {target_samples.shape[0]} and {base.shape[0]}"
            )
    displacement = target - start
    length = np.linalg.norm(displacement, axis=-1, keepdims=True)
    fallback = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (base.shape[0], 1))
    direction = np.divide(
        displacement,
        length,
        out=fallback.copy(),
        where=length > 1e-9,
    )
    orthogonal = np.stack([-direction[:, 1], direction[:, 0]], axis=-1)
    return direction, orthogonal, start


def _project_samples(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("sti,si->st", _as_samples(values), basis)


def _as_samples(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 3 or array.shape[-1] != 2:
        raise ValueError(f"expected array with trailing shape (time, 2), got {array.shape}")
    return array.reshape((-1, array.shape[-2], 2))


@lru_cache(maxsize=1)
def _gru_target_position_samples() -> np.ndarray:
    path = EVAL_BULK_DIR / f"{RUN_ID}.npz"
    with np.load(path) as arrays:
        return np.asarray(arrays["target_position"], dtype=np.float64)


def _figure_group_key(row: Mapping[str, Any]) -> str:
    perturbation = row.get("perturbation", {})
    family = str(row.get("family") or perturbation.get("family"))
    channel = str(row.get("channel") or perturbation.get("channel"))
    provenance = perturbation.get("channel_provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    parts = [channel, family]
    feedback_quantity = perturbation.get("feedback_quantity") or provenance.get("feedback_quantity")
    epsilon_component = perturbation.get("epsilon_component")
    axis_role = provenance.get("target_relative_axis_role")
    if feedback_quantity is not None:
        parts.append(str(feedback_quantity))
    if epsilon_component is not None:
        parts.append(str(epsilon_component))
    if axis_role is not None:
        parts.append(str(axis_role))
    return "__".join(parts)


def _figure_title(row: Mapping[str, Any]) -> str:
    key = _figure_group_key(row)
    return key.replace("__", " / ").replace("_", " ")


def _timing_bins(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = sorted({_row_timing_label(row) for row in rows}, key=_timing_sort_key)
    return labels or ["none"]


def _row_timing_label(row: Mapping[str, Any]) -> str:
    timing = row.get("timing") or row.get("perturbation", {}).get("timing") or {}
    return str(row.get("timing_bin") or timing.get("timing_bin") or "movement_onset")


def _timing_sort_key(label: str) -> tuple[int, str]:
    return (TIMING_ORDER.get(label, 50), label)


def _representative_timing(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        timing = row.get("timing") or row.get("perturbation", {}).get("timing")
        if isinstance(timing, Mapping):
            return timing
    return None


def _source_label(source: str) -> str:
    if source == "extlqg6d":
        return "6D extLQG"
    return "GRU"


def _band_color(source: str) -> str:
    if source == "extlqg6d":
        return "rgba(194,65,12,0.10)"
    return "rgba(37,99,235,0.10)"


def _extlqg_nominal_peak_velocity(extlqg_context: Mapping[str, Any]) -> float:
    base = extlqg_context["base_evaluation"]
    speed = np.linalg.norm(np.asarray(base.velocity, dtype=np.float64), axis=-1)
    peak = float(np.nanmax(speed))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"nominal extLQG peak velocity must be positive; got {peak}")
    return peak


def _level_paths(level_key: str) -> dict[str, Path]:
    level = LEVEL_CONFIGS[level_key]
    bank_level = str(level["bank_level"])
    return {
        "figure_spec_dir": FIGURE_SPEC_ROOT / level_key,
        "figure_bulk_dir": FIGURE_BULK_ROOT / level_key,
        "figure_spec": FIGURE_SPEC_ROOT / level_key / "spec.json",
        "pert_bulk_dir": (
            PERT_BULK_ROOT / f"gru_targetsupport_const_band16_calibrated_{bank_level}"
        ),
        "pert_manifest": (
            NOTES_DIR
            / f"gru_perturbation_response_const_band16_calibrated_{bank_level}_manifest.json"
        ),
        "pert_note": (
            NOTES_DIR / f"gru_perturbation_response_const_band16_calibrated_{bank_level}.md"
        ),
    }


def write_figure_readme(profile_specs: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        "# Final Calibrated Perturbation-Bank Profile Figures",
        "",
        "These figures compare the no-PGD `const_band16` GRU row with the 6D "
        "no-integrator analytical extLQG comparator on calibrated perturbation "
        "banks at fixed 15 cm reach length.",
        "",
        "For residual figures, position rows are shown as percent of the fixed "
        "0.15 m reach length, and velocity rows are shown as percent of the "
        "nominal 6D extLQG peak speed. Command residual rows remain in native "
        "command units. Orthogonal traces are solid; along-reach traces are "
        "dotted. Yellow bands mark the perturbation pulse window.",
        "",
        "| level | calibrated bank level | figure spec |",
        "|---|---|---|",
    ]
    for spec in profile_specs:
        rows.append(
            "| {level} | `{bank_level}` | `{figure_spec}` |".format(
                level=spec["level"],
                bank_level=spec["calibration_level"],
                figure_spec=_repo_rel(_level_paths(str(spec["level"]))["figure_spec"]),
            )
        )
    update_marked_section(FIGURE_README, "caption", "\n".join(rows) + "\n")


def write_profile_note(profile_specs: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        "# Final Calibrated Perturbation-Bank Profile Figures",
        "",
        f"- Source row: `{SOURCE_EXPERIMENT}/{RUN_ID}`.",
        "- Analytical comparator: 6D no-integrator extLQG.",
        "- Banks: calibrated physical `small` and user-facing `medium` "
        "(`moderate` in calibration manifests), fixed reach `0.15 m`.",
        "- Coordinates: orthogonal solid, along-reach dotted.",
        "- Residuals: sign-aligned perturbed-minus-clean profiles.",
        f"- Figure README: `{_repo_rel(FIGURE_README)}`.",
        "",
    ]
    for spec in profile_specs:
        rows.extend(
            [
                f"## {str(spec['level']).title()}",
                "",
                f"- Calibrated bank level: `{spec['calibration_level']}`.",
                f"- Figure spec: `{_repo_rel(_level_paths(str(spec['level']))['figure_spec'])}`.",
                f"- Perturbation manifest: `{spec['bank_manifest']}`.",
                f"- Evaluated GRU rows: `{spec['coverage']['evaluated_gru_rows']}`.",
                f"- 6D extLQG available rows: `{spec['coverage']['extlqg_available_rows']}`.",
                f"- 6D extLQG not-applicable rows: `{spec['coverage']['extlqg_not_applicable_rows']}`.",
                f"- 6D extLQG blocked rows: `{spec['coverage']['extlqg_blocked_rows']}`.",
                "",
                "| group | rows | extLQG rows | trajectory | residual |",
                "|---|---:|---:|---|---|",
            ]
        )
        for figure in spec["figures"]:
            rows.append(
                "| {title} | {n_rows} | {extlqg_available_rows} | `{trajectory}` | `{residual}` |".format(
                    title=figure["title"],
                    n_rows=figure["n_rows"],
                    extlqg_available_rows=figure["extlqg_available_rows"],
                    trajectory=figure["trajectory_html"],
                    residual=figure["residual_html"],
                )
            )
        rows.append("")
    rows.extend(
        [
            "## Units",
            "",
            "- Residual position panels: percent of fixed 0.15 m reach length.",
            "- Residual velocity panels: percent of nominal 6D extLQG peak speed.",
            "- Trajectory panels and residual command panels: native units.",
            "",
        ]
    )
    update_marked_section(PROFILE_NOTE, "final_calibrated_bank_profiles", "\n".join(rows) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_perturbation_detail_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path_text = manifest.get("bulk_detail_manifest")
    if path_text is None:
        return dict(manifest)
    if isinstance(path_text, Mapping):
        path_text = path_text["path"]
    return _read_json(REPO_ROOT / str(path_text))


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def _regeneration_spec_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_regeneration_spec.json")


if __name__ == "__main__":
    main()
