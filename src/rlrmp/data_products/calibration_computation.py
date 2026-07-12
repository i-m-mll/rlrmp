"""Scientific computation for the governed perturbation calibration product."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.data_products.calibration import (
    CALIBRATION_DEFAULTS_PRODUCT_ROLE,
    CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION,
    NativeConvention,
    ReachCalibrationPoint,
    ReachRelativeLevel,
    TimingCalibrationBin,
    calibrated_amplitude_from_unit_sensitivity,
    load_perturbation_calibration_defaults,
)
from rlrmp.data_products.envelope import consumed_identity_from_loader

if TYPE_CHECKING:
    from rlrmp.eval.gru_diagnostics import RolloutEvaluation


SCHEMA_VERSION = "rlrmp.perturbation_open_loop_calibration.v2"
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


def default_amplitude_factors() -> tuple[float, ...]:
    """Return adopted amplitude factors from the governed defaults product."""

    return load_perturbation_calibration_defaults().amplitude_factors


def calibration_origin_metadata() -> Mapping[str, str]:
    """Return governed provenance for the nominal closed-loop calibration origin."""

    origin = load_analysis_parameter_preset("gru_perturbation_calibration").parameters[
        "origin_baseline"
    ]
    return {str(key): str(value) for key, value in origin.items()}


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


__all__ = [
    "DEFAULT_CONTROLLER_VISIBLE_FORCE_FILTER_SCALE_N",
    "NativeConvention",
    "ReachCalibrationPoint",
    "ReachRelativeLevel",
    "SCHEMA_VERSION",
    "TimingCalibrationBin",
    "calibrated_amplitude_from_unit_sensitivity",
    "calibration_origin_metadata",
    "default_amplitude_factors",
    "default_controller_visible_timing_bins",
    "default_native_conventions",
    "default_plant_timing_bins",
    "default_reach_calibration_points",
    "default_reach_relative_levels",
]
