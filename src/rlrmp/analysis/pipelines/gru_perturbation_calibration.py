"""Open-loop physical calibration for the C&S perturbation bank."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.data_products.calibration import (
    CALIBRATION_DEFAULTS_PRODUCT_ROLE,
    CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION,
    NativeConvention,
    ReachCalibrationPoint,
    ReachRelativeLevel,
    TimingCalibrationBin,
    load_perturbation_calibration_defaults,
)
from rlrmp.data_products.envelope import consumed_identity_from_loader
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p

if TYPE_CHECKING:
    from rlrmp.eval.gru_diagnostics import RolloutEvaluation


SCHEMA_VERSION = "rlrmp.perturbation_open_loop_calibration.v2"
DEFAULT_RESULT_EXPERIMENT = "1ad3c16"
DEFAULT_OUTPUT_FILENAME = "perturbation_open_loop_calibration.json"
DEFAULT_REGENERATION_SPEC_FILENAME = "perturbation_open_loop_calibration_regeneration_spec.json"
DEFAULT_NOTE_FILENAME = "perturbation_open_loop_calibration.md"
DEFAULT_BULK_SUBDIR = "perturbation_open_loop_calibration"
DEFAULT_NOMINAL_GRU_BASELINE = {
    "experiment": "5f70333",
    "run_id": "lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64",
    "training_distribution": "single_target_nominal_only_full_Q/R/Q_f",
    "role": (
        "declared nominal-only GRU baseline for later closed-loop GRU calibration; "
        "this materializer defines bins from extLQG nominal-command open-loop replay"
    ),
}
OPEN_LOOP_SUPPORTED_CHANNELS = {"initial_state", "command_input", "process_epsilon"}
TIMED_PLANT_SIDE_FAMILIES = {
    "command_input_pulse",
    "process_epsilon_position_xy",
    "process_epsilon_velocity_xy",
    "process_epsilon_force_state_xy",
    "process_epsilon_integrator_xy",
}


# Generated/adopted calibration values are no longer baked here as source
# constants. Open-loop unit sensitivities, the controller-visible velocity scale,
# and the adopted runtime-default tables are persisted as governed data products
# under results/ea6ccb4/data_products/ and loaded fail-closed by identity via
# rlrmp.data_products.calibration. See issue ea6ccb4.
# The force/filter native scale below is a unit convention (1 N reference offset),
# not generated data, and stays as a source constant.
DEFAULT_CONTROLLER_VISIBLE_FORCE_FILTER_SCALE_N = float(
    load_analysis_parameter_preset("gru_perturbation_calibration").parameters[
        "controller_visible_force_filter_scale_n"
    ]
)


def materialize_perturbation_open_loop_calibration(
    *,
    amplitude_factors: Sequence[float] | None = None,
    reach_points: Sequence[ReachCalibrationPoint] | None = None,
    levels: Sequence[ReachRelativeLevel] | None = None,
    plant_timing_bins: Sequence[TimingCalibrationBin] | None = None,
    controller_visible_timing_bins: Sequence[TimingCalibrationBin] | None = None,
    native_conventions: Sequence[NativeConvention] | None = None,
    result_experiment: str = DEFAULT_RESULT_EXPERIMENT,
    output_path: Path | None = None,
    note_path: Path | None = None,
    regeneration_spec_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize physical effect-size calibration for perturbation amplitudes."""

    from rlrmp.eval.perturbation_bank import (
        _build_extlqg_comparator_context,
        _extlqg_cost_summary,
        default_cs_perturbation_bank,
    )

    defaults_used = (
        amplitude_factors is None
        or reach_points is None
        or levels is None
        or plant_timing_bins is None
        or controller_visible_timing_bins is None
        or native_conventions is None
    )
    if defaults_used:
        defaults = load_perturbation_calibration_defaults()
        amplitude_factors = (
            defaults.amplitude_factors if amplitude_factors is None else amplitude_factors
        )
        reach_points = defaults.reach_calibration_points if reach_points is None else reach_points
        levels = defaults.reach_relative_levels if levels is None else levels
        plant_timing_bins = (
            defaults.plant_timing_bins if plant_timing_bins is None else plant_timing_bins
        )
        controller_visible_timing_bins = (
            defaults.controller_visible_timing_bins
            if controller_visible_timing_bins is None
            else controller_visible_timing_bins
        )
        native_conventions = (
            defaults.native_conventions if native_conventions is None else native_conventions
        )

    output_path = output_path or (
        repo_root / "_artifacts" / result_experiment / DEFAULT_BULK_SUBDIR / DEFAULT_OUTPUT_FILENAME
    )
    note_path = note_path or (
        repo_root / "results" / result_experiment / "notes" / DEFAULT_NOTE_FILENAME
    )
    regeneration_spec_path = regeneration_spec_path or (
        output_path.parent / DEFAULT_REGENERATION_SPEC_FILENAME
    )
    mkdir_p(output_path.parent)
    bank = default_cs_perturbation_bank()
    context = _build_extlqg_comparator_context()
    base = context["base_evaluation"]
    base_cost = _extlqg_cost_summary(base, context["base_initial_state"])
    sensitivities = _family_sensitivities(
        bank["perturbations"],
        plant_timing_bins=plant_timing_bins,
        context=context,
        base=base,
        base_cost=base_cost,
    )
    rows = []
    for reach in reach_points:
        for level in levels:
            target_peak_delta_x_m = float(reach.reach_length_m) * float(level.fraction_of_reach)
            for sensitivity_key, sensitivity in sensitivities.items():
                if sensitivity.get("status") != "available":
                    continue
                amplitude = calibrated_amplitude_from_unit_sensitivity(
                    target_peak_delta_x_m=target_peak_delta_x_m,
                    peak_delta_x_per_unit_m=float(sensitivity["peak_delta_x_per_unit"]),
                )
                scaled = _set_perturbation_amplitude(
                    sensitivity["perturbation"],
                    amplitude=float(amplitude),
                    suffix=(
                        f"{sensitivity_key}__{reach.label}__{level.name}"
                        f"__target_{target_peak_delta_x_m * 1000.0:.3f}mm"
                    ),
                )
                rows.append(
                    _evaluate_calibration_row(
                        scaled,
                        amplitude_factor=float(amplitude)
                        / float(sensitivity["perturbation"]["amplitude"]),
                        reach=reach,
                        level=level,
                        target_peak_delta_x_m=target_peak_delta_x_m,
                        context=context,
                        base=base,
                        base_cost=base_cost,
                        sensitivity=sensitivity,
                    )
                )
    for sensitivity in sensitivities.values():
        if sensitivity.get("status") != "available":
            continue
        rows.append(
            {
                "row_kind": "unit_sensitivity",
                "sensitivity_id": sensitivity["sensitivity_id"],
                "perturbation_id": sensitivity["perturbation_id"],
                "channel": sensitivity["channel"],
                "family": sensitivity["family"],
                "axis": sensitivity.get("axis"),
                "sign": sensitivity.get("sign"),
                "timing_bin": sensitivity.get("timing_bin"),
                "timing": sensitivity.get("timing"),
                "amplitude": 1.0,
                "open_loop_peak_delta_x_per_unit_m": sensitivity["peak_delta_x_per_unit"],
                "open_loop_auc_delta_x_per_unit_m_s": sensitivity["auc_delta_x_per_unit"],
                "selection_rule": sensitivity["selection_rule"],
                "open_loop": sensitivity["open_loop"],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "bank_schema_version": bank["schema_version"],
        "scope": "extlqg_nominal_command_open_loop_physical_effect_calibration",
        "calibration_mode": "reach_relative_peak_delta_x",
        "reach_points": [reach.to_json() for reach in reach_points],
        "level_definitions": [level.to_json() for level in levels],
        "plant_timing_bins": [timing_bin.to_json() for timing_bin in plant_timing_bins],
        "controller_visible_timing_bins": [
            timing_bin.to_json() for timing_bin in controller_visible_timing_bins
        ],
        "native_conventions": [convention.to_json() for convention in native_conventions],
        "legacy_fixed_mm_amplitude_factors": list(amplitude_factors),
        "target_rule": "target_peak_delta_x_m = reach_length_m * fraction_of_reach",
        "selection_rule": (
            "For timed plant-side rows, calibrate one unit sensitivity for each "
            "family x plant timing bin, using x axis, positive sign, and stable "
            "perturbation_id tie-breaks. Initial-condition rows keep t=0 and one "
            "unit sensitivity per family. Reach and severity rows are derived by "
            "scaling those unit sensitivities; they are not independently calibrated."
        ),
        "open_loop_replay_geometry": {
            "nominal_reach_length_m": 0.15,
            "note": (
                "The extLQG comparator helper currently replays nominal commands on "
                "the canonical 0.15 m +x target. The reach-relative calibration "
                "varies the requested physical-effect target by declared reach "
                "length; it does not retrain or rebuild multi-target extLQG rows."
            ),
        },
        "open_loop_reference": {
            "controller": "extLQG nominal command replay",
            "feedback_correction": False,
            "role": (
                "define physical perturbation bins by peak delta x before feedback "
                "correction; closed-loop extLQG metrics are reported at the same "
                "amplitudes only for interpretation"
            ),
        },
        "nominal_gru_baseline_for_later_closed_loop_calibration": DEFAULT_NOMINAL_GRU_BASELINE,
        "consumed_data_identities": _consumed_default_identities() if defaults_used else [],
        "bulk_manifest_path": _repo_relative(output_path, repo_root=repo_root),
        "regeneration_spec_path": repo_relative(regeneration_spec_path, repo_root=repo_root),
        "rows": rows,
        "family_summary": _summarize_reach_relative_rows(rows),
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_marked_section(
        note_path,
        "perturbation_open_loop_calibration",
        render_calibration_markdown(manifest),
    )
    _write_calibration_regeneration_spec(
        spec_path=regeneration_spec_path,
        output_path=output_path,
        note_path=note_path,
        manifest=manifest,
        amplitude_factors=amplitude_factors,
        result_experiment=result_experiment,
        repo_root=repo_root,
    )
    return manifest


def default_amplitude_factors() -> tuple[float, ...]:
    """Return adopted amplitude factors from the governed defaults product."""

    return load_perturbation_calibration_defaults().amplitude_factors


def default_reach_calibration_points() -> tuple[ReachCalibrationPoint, ...]:
    """Return adopted reach calibration points from the governed defaults product."""

    return load_perturbation_calibration_defaults().reach_calibration_points


def default_reach_relative_levels() -> tuple[ReachRelativeLevel, ...]:
    """Return adopted reach-relative levels from the governed defaults product."""

    return load_perturbation_calibration_defaults().reach_relative_levels


def default_plant_timing_bins() -> tuple[TimingCalibrationBin, ...]:
    """Return adopted plant-side timing bins from the governed defaults product."""

    return load_perturbation_calibration_defaults().plant_timing_bins


def default_controller_visible_timing_bins() -> tuple[TimingCalibrationBin, ...]:
    """Return adopted controller-visible timing bins from the governed defaults product."""

    return load_perturbation_calibration_defaults().controller_visible_timing_bins


def default_native_conventions() -> tuple[NativeConvention, ...]:
    """Return adopted native conventions from the governed defaults product."""

    return load_perturbation_calibration_defaults().native_conventions


def calibrated_amplitude_from_unit_sensitivity(
    *,
    target_peak_delta_x_m: float,
    peak_delta_x_per_unit_m: float,
) -> float:
    """Return amplitude derived from a unit open-loop sensitivity."""

    if peak_delta_x_per_unit_m <= 0.0:
        raise ValueError("peak_delta_x_per_unit_m must be positive")
    return float(target_peak_delta_x_m) / float(peak_delta_x_per_unit_m)


def render_calibration_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact calibration summary."""

    lines = [
        "# Perturbation Open-Loop Calibration",
        "",
        f"- Issue: `{manifest['issue']}`",
        f"- Scope: `{manifest['scope']}`",
        "- Open-loop reference: extLQG nominal command replay.",
        "- Closed-loop extLQG is reported at the same amplitudes where supported.",
        "- Calibration mode: reach-relative peak `delta x`, with target peak "
        "`delta x = fraction * reach_length`.",
        "- Unit sensitivities are calibrated by perturbation family and timing bin; "
        "reach/level rows are deterministic scalings from those sensitivities, not "
        "independent calibrations.",
        "- Replay geometry: canonical 0.15 m +x extLQG nominal command replay; "
        "the reach-relative targets vary the requested physical effect size, not "
        "the nominal replay task.",
        "- GRU baseline for later closed-loop calibration: "
        f"`{manifest['nominal_gru_baseline_for_later_closed_loop_calibration']['experiment']}` / "
        f"`{manifest['nominal_gru_baseline_for_later_closed_loop_calibration']['run_id']}` "
        "(single-target nominal-only; documented for later closed-loop comparison, "
        "not retrained here).",
        f"- Bulk row manifest: `{manifest.get('bulk_manifest_path', 'not_materialized')}`",
        f"- Regeneration spec: `{manifest.get('regeneration_spec_path', 'not_materialized')}`",
        "",
        "## Deterministic Config",
        "",
        "Reach lengths:",
        "",
        "| Label | Split | Reach length | Role |",
        "|---|---|---:|---|",
    ]
    for reach in manifest["reach_points"]:
        lines.append(
            f"| `{reach['label']}` | {reach['split']} | "
            f"{_fmt_mm(reach['reach_length_m'])} | {reach['role']} |"
        )
    lines.extend(
        [
            "",
            "Levels:",
            "",
            "| Level | Fraction of reach | Role |",
            "|---|---:|---|",
        ]
    )
    for level in manifest["level_definitions"]:
        lines.append(
            f"| `{level['name']}` | {100.0 * float(level['fraction_of_reach']):.1f}% | "
            f"{level['role']} |"
        )
    lines.extend(
        [
            "",
            "Plant-side timing bins:",
            "",
            "| Bin | Start step | Duration | Role |",
            "|---|---:|---:|---|",
        ]
    )
    for timing_bin in manifest.get("plant_timing_bins", []):
        lines.append(
            f"| `{timing_bin['label']}` | {timing_bin['start_time_index']} | "
            f"{timing_bin['duration_steps']} | {timing_bin['role']} |"
        )
    lines.extend(
        [
            "",
            "Controller-visible/native conventions:",
            "",
            "| Family | Channel | Native rule | Timing rule | Report metric |",
            "|---|---|---|---|---|",
        ]
    )
    for convention in manifest.get("native_conventions", []):
        lines.append(
            f"| `{convention['family']}` | `{convention['channel']}` | "
            f"{convention['native_unit_rule']} | {convention['timing_rule']} | "
            f"{convention['report_metric']} |"
        )
    lines.extend(
        [
            "",
            "## Selected Reach-Relative Amplitudes",
            "",
            "| Reach | Level | Family | Timing bin | Amplitude | Target peak dx | Achieved peak dx | "
            "Achieved % reach | AUC dx | Notes |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in _selected_calibration_rows(manifest["rows"]):
        notes = row.get("warning") or "none"
        lines.append(
            f"| `{row['reach_label']}` | `{row['level_name']}` | `{row['family']}` | "
            f"`{row.get('timing_bin', 'initial_condition')}` | "
            f"{_fmt(row['amplitude'])} | "
            f"{_fmt_mm(row['target_peak_delta_x_m'])} | "
            f"{_fmt_mm(row['achieved_peak_delta_x_m'])} | "
            f"{_fmt_pct(row['achieved_peak_delta_x_fraction_of_reach'])} | "
            f"{_fmt(row.get('achieved_auc_delta_x_m_s'))} | "
            f"{notes} |"
        )
    lines.extend(
        [
            "",
            "## Rerun Command",
            "",
            "```bash",
            "uv run python scripts/materialize_perturbation_open_loop_calibration.py",
            "```",
            "",
            "Bulk per-row data is written under `_artifacts/1ad3c16/...`; keep it out of "
            "`results/`.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _evaluate_calibration_row(
    perturbation: Mapping[str, Any],
    *,
    amplitude_factor: float,
    reach: ReachCalibrationPoint,
    level: ReachRelativeLevel,
    target_peak_delta_x_m: float,
    context: Mapping[str, Any],
    base: RolloutEvaluation,
    base_cost: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    from rlrmp.eval.perturbation_bank import (
        _extlqg_cost_summary,
        _simulate_extlqg_perturbed,
        extlqg_comparator_status,
        summarize_perturbation_response,
    )

    open_loop_eval, initial_state, provenance = _simulate_open_loop_command_replay(
        perturbation,
        context=context,
    )
    open_loop_cost = _extlqg_cost_summary(open_loop_eval, initial_state)
    open_loop_metrics = summarize_perturbation_response(
        base,
        open_loop_eval,
        base_full_qrf_cost=base_cost,
        perturbed_full_qrf_cost=open_loop_cost,
    )
    extlqg_status = extlqg_comparator_status(perturbation, status="not_applicable")
    if perturbation["channel"] != "command_input":
        try:
            ext_eval, ext_initial, ext_provenance = _simulate_extlqg_perturbed(
                perturbation,
                context=context,
            )
            ext_cost = _extlqg_cost_summary(ext_eval, ext_initial)
            extlqg_status = {
                "status": "available",
                "analytical_adapter": ext_provenance,
                "reference_response_metrics": summarize_perturbation_response(
                    base,
                    ext_eval,
                    base_full_qrf_cost=base_cost,
                    perturbed_full_qrf_cost=ext_cost,
                ),
            }
        except (KeyError, ValueError) as exc:
            extlqg_status = {"status": "blocked", "reason": str(exc)}
    peak = _metric_mean(open_loop_metrics, "delta_position_response_m.max")
    auc = _metric_mean(open_loop_metrics, "delta_position_response_m.auc")
    target_error = None if peak is None else float(peak) - float(target_peak_delta_x_m)
    warning = None
    if peak is None:
        warning = "peak delta x unavailable"
    elif abs(target_error) > max(1e-9, 1e-6 * float(target_peak_delta_x_m)):
        warning = "achieved peak differs from target beyond numerical tolerance"
    return {
        "row_kind": "reach_relative_calibrated_amplitude",
        "reach_label": reach.label,
        "reach_split": reach.split,
        "reach_length_m": float(reach.reach_length_m),
        "level_name": level.name,
        "level_fraction_of_reach": float(level.fraction_of_reach),
        "target_peak_delta_x_m": float(target_peak_delta_x_m),
        "achieved_peak_delta_x_m": peak,
        "achieved_peak_delta_x_mm": None if peak is None else float(peak) * 1000.0,
        "achieved_peak_delta_x_fraction_of_reach": (
            None if peak is None else float(peak) / float(reach.reach_length_m)
        ),
        "achieved_auc_delta_x_m_s": auc,
        "target_peak_error_m": target_error,
        "warning": warning,
        "sensitivity_reference": {
            "sensitivity_id": sensitivity["sensitivity_id"],
            "perturbation_id": sensitivity["perturbation_id"],
            "peak_delta_x_per_unit_m": sensitivity["peak_delta_x_per_unit"],
            "auc_delta_x_per_unit_m_s": sensitivity["auc_delta_x_per_unit"],
            "selection_rule": sensitivity["selection_rule"],
        },
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation["family"],
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation["amplitude"],
        "amplitude_factor": amplitude_factor,
        "timing_bin": sensitivity.get("timing_bin"),
        "timing": perturbation.get("timing"),
        "open_loop": {
            "status": "available",
            "adapter": provenance,
            "response_metrics": open_loop_metrics,
        },
        "closed_loop_extlqg": extlqg_status,
    }


def _simulate_open_loop_command_replay(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> tuple[RolloutEvaluation, np.ndarray, dict[str, Any]]:
    """Replay nominal extLQG commands through perturbed plant dynamics."""

    from rlrmp.eval.perturbation_bank import (
        _evaluation_from_extlqg_rollout,
        _extlqg_process_epsilon,
        _perturbed_extlqg_initial_state,
    )

    plant = context["plant"]
    schedule = context["schedule"]
    x0 = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
    command = jnp.asarray(context["base_evaluation"].command[0, 0], dtype=jnp.float64)
    epsilon = jnp.zeros((schedule.T, plant.m_w), dtype=jnp.float64)
    provenance = {
        "adapter": "extlqg_nominal_command_open_loop_replay",
        "feedback_correction": False,
    }
    if perturbation["channel"] == "initial_state":
        x0 = _perturbed_extlqg_initial_state(x0, perturbation)
        provenance["perturbation_adapter"] = "initial_state_offset"
    elif perturbation["channel"] == "process_epsilon":
        epsilon = _extlqg_process_epsilon(perturbation, schedule.T, plant.m_w)
        provenance["perturbation_adapter"] = "process_epsilon_sequence"
    elif perturbation["channel"] == "command_input":
        command = _apply_command_input_to_open_loop_command(command, perturbation)
        provenance["perturbation_adapter"] = "command_input_added_to_replayed_command"
    else:
        raise ValueError(f"open-loop replay does not support {perturbation['channel']!r}")
    xs = [x0]
    for t in range(schedule.T):
        xs.append(plant.A @ xs[-1] + plant.B @ command[t] + plant.Bw @ epsilon[t])
    rollout = type("OpenLoopRollout", (), {})()
    rollout.x = jnp.stack(xs, axis=0)
    rollout.u_command = command
    evaluation = _evaluation_from_extlqg_rollout(rollout, initial_state=x0)
    return evaluation, np.asarray(x0), provenance


def _apply_command_input_to_open_loop_command(
    command: Any,
    perturbation: Mapping[str, Any],
) -> jnp.ndarray:
    result = jnp.array(command, dtype=jnp.float64)
    timing = perturbation.get("timing", {})
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    axis = 0 if perturbation.get("axis") == "x" else 1
    sign = float(perturbation.get("sign", 1))
    amp = float(perturbation["amplitude"])
    end = min(result.shape[0], start + duration)
    return result.at[start:end, axis].add(sign * amp)


def _set_perturbation_amplitude(
    perturbation: Mapping[str, Any],
    *,
    amplitude: float,
    suffix: str,
) -> dict[str, Any]:
    row = dict(perturbation)
    row["base_amplitude"] = float(perturbation["amplitude"])
    row["amplitude"] = float(amplitude)
    row["perturbation_id"] = f"{perturbation['perturbation_id']}__{suffix}"
    return row


def _family_sensitivities(
    perturbations: Sequence[Mapping[str, Any]],
    *,
    plant_timing_bins: Sequence[TimingCalibrationBin],
    context: Mapping[str, Any],
    base: RolloutEvaluation,
    base_cost: Mapping[str, Any],
) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for perturbation in perturbations:
        by_family.setdefault(str(perturbation["family"]), []).append(perturbation)
    sensitivities = {}
    for family, family_rows in sorted(by_family.items()):
        representative = _representative_perturbation(family_rows)
        if representative["channel"] not in OPEN_LOOP_SUPPORTED_CHANNELS:
            sensitivities[family] = {
                "status": "not_applicable",
                "reason": (
                    "open-loop command replay has no feedback/target channel for "
                    f"{representative['channel']!r}"
                ),
                "perturbation": representative,
            }
            continue
        representatives = _calibration_representatives(
            family=family,
            representative=representative,
            plant_timing_bins=plant_timing_bins,
        )
        for sensitivity_id, timing_bin_label, representative in representatives:
            unit = _set_perturbation_amplitude(
                representative,
                amplitude=1.0,
                suffix=f"{sensitivity_id}__unit_sensitivity",
            )
            row = _evaluate_unit_sensitivity_row(
                unit,
                context=context,
                base=base,
                base_cost=base_cost,
            )
            peak = _metric_mean(
                row["open_loop"]["response_metrics"], "delta_position_response_m.max"
            )
            auc = _metric_mean(
                row["open_loop"]["response_metrics"], "delta_position_response_m.auc"
            )
            if peak is None or abs(peak) <= 1e-12:
                sensitivities[sensitivity_id] = {
                    "status": "blocked",
                    "reason": "unit-amplitude open-loop peak delta x was unavailable or near zero",
                    "perturbation": representative,
                    "timing_bin": timing_bin_label,
                    "open_loop": row["open_loop"],
                }
                continue
            sensitivities[sensitivity_id] = {
                "status": "available",
                "sensitivity_id": sensitivity_id,
                "perturbation": representative,
                "perturbation_id": representative["perturbation_id"],
                "channel": representative["channel"],
                "family": family,
                "axis": representative.get("axis"),
                "sign": representative.get("sign"),
                "timing_bin": timing_bin_label,
                "timing": representative.get("timing"),
                "peak_delta_x_per_unit": float(abs(peak)),
                "auc_delta_x_per_unit": auc,
                "open_loop": row["open_loop"],
                "selection_rule": _representative_selection_rule(family),
            }
    return sensitivities


def _calibration_representatives(
    *,
    family: str,
    representative: Mapping[str, Any],
    plant_timing_bins: Sequence[TimingCalibrationBin],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    if family not in TIMED_PLANT_SIDE_FAMILIES:
        return [(f"{family}__initial_condition", "initial_condition", representative)]
    rows = []
    for timing_bin in plant_timing_bins:
        rows.append(
            (
                f"{family}__{timing_bin.label}",
                timing_bin.label,
                _with_timing_bin(representative, timing_bin),
            )
        )
    return rows


def _with_timing_bin(
    perturbation: Mapping[str, Any],
    timing_bin: TimingCalibrationBin,
) -> dict[str, Any]:
    row = dict(perturbation)
    timing = dict(row.get("timing", {}))
    timing["epoch"] = "movement_indexed"
    timing["start_time_index"] = int(timing_bin.start_time_index)
    timing["duration_steps"] = int(timing_bin.duration_steps)
    timing["calibration_timing_bin"] = timing_bin.label
    row["timing"] = timing
    row["calibration_timing_bin"] = timing_bin.label
    row["perturbation_id"] = (
        f"{perturbation['family']}__{timing_bin.label}_t{timing_bin.start_time_index}"
        f"__{perturbation.get('axis', 'axis')}_{_sign_label(int(perturbation.get('sign', 1)))}"
    )
    return row


def _representative_perturbation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            int(row.get("timing", {}).get("start_time_index", 0)),
            0 if row.get("axis") == "x" else 1,
            0 if int(row.get("sign", 1)) > 0 else 1,
            str(row["perturbation_id"]),
        ),
    )


def _representative_selection_rule(family: str) -> str:
    if family in TIMED_PLANT_SIDE_FAMILIES:
        return (
            "family x timing_bin unit sensitivity; x axis, positive sign, perturbation_id tie-break"
        )
    return "initial-condition unit sensitivity; x axis, positive sign, perturbation_id tie-break"


def _sign_label(sign: int) -> str:
    return "pos" if sign > 0 else "neg"


def _evaluate_unit_sensitivity_row(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    base: RolloutEvaluation,
    base_cost: Mapping[str, Any],
) -> dict[str, Any]:
    from rlrmp.eval.perturbation_bank import (
        _extlqg_cost_summary,
        summarize_perturbation_response,
    )

    open_loop_eval, initial_state, provenance = _simulate_open_loop_command_replay(
        perturbation,
        context=context,
    )
    open_loop_cost = _extlqg_cost_summary(open_loop_eval, initial_state)
    open_loop_metrics = summarize_perturbation_response(
        base,
        open_loop_eval,
        base_full_qrf_cost=base_cost,
        perturbed_full_qrf_cost=open_loop_cost,
    )
    return {
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation["family"],
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation["amplitude"],
        "timing": perturbation.get("timing"),
        "open_loop": {
            "status": "available",
            "adapter": provenance,
            "response_metrics": open_loop_metrics,
        },
    }


def _summarize_reach_relative_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("row_kind") != "reach_relative_calibrated_amplitude":
            continue
        by_family.setdefault(str(row["family"]), []).append(row)
    return {
        family: _summarize_reach_relative_family(rows_for_family)
        for family, rows_for_family in sorted(by_family.items())
    }


def _summarize_reach_relative_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("open_loop", {}).get("status") == "available"]
    targets = []
    for row in rows:
        targets.append(
            {
                "perturbation_id": row["perturbation_id"],
                "reach_label": row.get("reach_label"),
                "level_name": row.get("level_name"),
                "amplitude": float(row["amplitude"]),
                "target_peak_delta_x_m": row.get("target_peak_delta_x_m"),
                "achieved_peak_delta_x_m": row.get("achieved_peak_delta_x_m"),
                "achieved_peak_delta_x_fraction_of_reach": row.get(
                    "achieved_peak_delta_x_fraction_of_reach"
                ),
                "axis": row.get("axis"),
                "sign": row.get("sign"),
            }
        )
    return {
        "n_targets": len(rows),
        "n_available": len(available),
        "targets": targets,
        "warnings": sorted({str(row["warning"]) for row in rows if row.get("warning")}),
    }


def _write_calibration_regeneration_spec(
    *,
    spec_path: Path,
    output_path: Path,
    note_path: Path,
    manifest: Mapping[str, Any],
    amplitude_factors: Sequence[float],
    result_experiment: str,
    repo_root: Path,
) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/materialize_perturbation_open_loop_calibration.py",
        "--result-experiment",
        result_experiment,
        "--output-path",
        repo_relative(output_path, repo_root=repo_root),
        "--note-path",
        repo_relative(note_path, repo_root=repo_root),
        "--regeneration-spec-path",
        repo_relative(spec_path, repo_root=repo_root),
    ]
    for factor in amplitude_factors:
        command.extend(["--amplitude-factor", str(float(factor))])
    source_model = dict(DEFAULT_NOMINAL_GRU_BASELINE)
    source_run_ids = [
        str(source_model["run_id"]),
        "extLQG nominal command replay",
    ]
    return write_regeneration_spec(
        spec_path=spec_path,
        diagnostic_name="perturbation_open_loop_calibration",
        materializer=(
            "rlrmp.analysis.pipelines.gru_perturbation_calibration."
            "materialize_perturbation_open_loop_calibration"
        ),
        command=command,
        parameters={
            "result_experiment": result_experiment,
            "amplitude_factors": [float(factor) for factor in amplitude_factors],
            "consumed_data_identities": manifest.get("consumed_data_identities", []),
            "source_model": source_model,
            "source_run_ids": source_run_ids,
            "reach_points": manifest.get("reach_points", []),
            "level_definitions": manifest.get("level_definitions", []),
            "plant_timing_bins": manifest.get("plant_timing_bins", []),
            "controller_visible_timing_bins": manifest.get(
                "controller_visible_timing_bins",
                [],
            ),
        },
        inputs=[
            {
                "role": "perturbation_bank",
                "description": "default_cs_perturbation_bank generated in-process",
                "bank_schema_version": manifest.get("bank_schema_version"),
            },
            {
                "role": "source_model",
                "description": "declared nominal GRU baseline for later closed-loop calibration",
                **source_model,
            },
            {
                "role": "source_run_ids",
                "run_ids": source_run_ids,
            },
        ],
        outputs=[
            {"role": "calibration_bulk_manifest", "path": output_path},
            {"role": "calibration_markdown_note", "path": note_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py",
            "scripts/materialize_perturbation_open_loop_calibration.py",
            "src/rlrmp/analysis/pipelines/diagnostic_provenance.py",
        ],
        notes=[
            (
                "Open-loop calibration replays extLQG nominal commands; "
                "it does not regenerate GRU model outputs."
            ),
            "Nominal GRU baseline metadata is recorded for later closed-loop calibration context.",
        ],
        repo_root=repo_root,
    )


def _consumed_default_identities() -> list[dict[str, str]]:
    from rlrmp.runtime.training_run_specs import add_consumed_data_identity

    spec = add_consumed_data_identity(
        {},
        **consumed_identity_from_loader(
            load_product=load_perturbation_calibration_defaults,
            role=CALIBRATION_DEFAULTS_PRODUCT_ROLE,
            schema=CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION,
        ),
    )
    return list(spec.get("consumed_data_identities", []))


def _metric_mean(metrics: Mapping[str, Any], key: str) -> float | None:
    current: Any = metrics
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, Mapping):
        current = current.get("mean")
    if current is None:
        return None
    value = float(current)
    return value if np.isfinite(value) else None


def _selected_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("row_kind") == "reach_relative_calibrated_amplitude"
        and row.get("open_loop", {}).get("status") == "available"
    ]
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("reach_split")),
            float(row.get("reach_length_m") or 0.0),
            float(row.get("level_fraction_of_reach") or 0.0),
            str(row.get("family")),
        ),
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1e3 or (abs(value) > 0.0 and abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6g}"


def _fmt_mm(value_m: float | None) -> str:
    if value_m is None:
        return "n/a"
    return f"{float(value_m) * 1000.0:.3f} mm"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.3f}%"


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_CONTROLLER_VISIBLE_FORCE_FILTER_SCALE_N",
    "DEFAULT_NOMINAL_GRU_BASELINE",
    "DEFAULT_REGENERATION_SPEC_FILENAME",
    "NativeConvention",
    "ReachCalibrationPoint",
    "ReachRelativeLevel",
    "SCHEMA_VERSION",
    "TimingCalibrationBin",
    "calibrated_amplitude_from_unit_sensitivity",
    "default_amplitude_factors",
    "default_controller_visible_timing_bins",
    "default_native_conventions",
    "default_plant_timing_bins",
    "default_reach_calibration_points",
    "default_reach_relative_levels",
    "materialize_perturbation_open_loop_calibration",
    "render_calibration_markdown",
]
