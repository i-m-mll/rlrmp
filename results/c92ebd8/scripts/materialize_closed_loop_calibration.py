"""Materialize c92ebd8 closed-loop perturbation calibration values."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    _build_extlqg_comparator_context,
    _extlqg_cost_summary,
    _simulate_extlqg_perturbed,
    summarize_perturbation_response,
)
from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
    ReachRelativeLevel,
    TimingCalibrationBin,
    default_controller_visible_timing_bins,
    default_plant_timing_bins,
    default_reach_relative_levels,
)
from rlrmp.io import write_compact_json
from rlrmp.paths import mkdir_p


SCHEMA_VERSION = "rlrmp.c92ebd8.closed_loop_perturbation_calibration.v1"
ISSUE = "c92ebd8"
DEFAULT_REACH_LENGTH_M = 0.15
DEFAULT_OUTPUT_PATH = Path("results") / ISSUE / "notes" / "closed_loop_calibration_table.json"
DEFAULT_REACH_RELATIVE_LEVELS = default_reach_relative_levels()
DEFAULT_PLANT_TIMING_BINS = default_plant_timing_bins()
DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS = default_controller_visible_timing_bins()


def materialize_closed_loop_calibration(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    reach_length_m: float = DEFAULT_REACH_LENGTH_M,
) -> dict[str, Any]:
    """Write the c92 closed-loop calibration table and return its payload."""

    context = _build_extlqg_comparator_context(physical_dim=6)
    base = context["base_evaluation"]
    base_cost = _extlqg_cost_summary(base, context["base_initial_state"])
    unit_sensitivities = _unit_sensitivity_rows(context=context, base=base, base_cost=base_cost)
    rows = _physical_level_rows(
        unit_sensitivities,
        reach_length_m=reach_length_m,
        levels=DEFAULT_REACH_RELATIVE_LEVELS,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "status": "materialized",
        "scope": "closed_loop_6d_extlqg_perturbation_calibration_for_c92_training_rows",
        "source": {
            "comparator": "6D extLQG deterministic released-forward rollout",
            "context_builder": (
                "rlrmp.analysis.pipelines.gru_perturbation_bank."
                "_build_extlqg_comparator_context(physical_dim=6)"
            ),
            "simulation_adapter": (
                "rlrmp.analysis.pipelines.gru_perturbation_bank._simulate_extlqg_perturbed"
            ),
            "base_rollout_noise": "zero_forward_noise_draws and zero_noise_covariances",
            "training_code_status": (
                "table materialized; ordinary run config can select closed-loop versus "
                "open-loop calibration sources by family"
            ),
        },
        "target_rule": {
            "reach_length_m": float(reach_length_m),
            "target_peak_delta_x_m": "reach_length_m * level_fraction_of_reach",
            "amplitude_rule": "amplitude = target_peak_delta_x_m / closed_loop_peak_delta_x_per_unit_m",
            "linearity_assumption": (
                "single-unit deterministic extLQG sensitivity is used as a local "
                "linear scale for reach-relative amplitudes"
            ),
        },
        "physical_levels": [level.to_json() for level in DEFAULT_REACH_RELATIVE_LEVELS],
        "timing_bins": {
            "plant_side": [timing_bin.to_json() for timing_bin in DEFAULT_PLANT_TIMING_BINS],
            "controller_visible": [
                timing_bin.to_json() for timing_bin in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS
            ],
        },
        "calibration_regimes": {
            "open_loop_all": {
                "closed_loop_families": [],
                "open_loop_families": [
                    "initial_position_offset",
                    "initial_velocity_offset",
                    "process_epsilon_*",
                    "command_input_pulse",
                    "sensory_feedback_offset",
                ],
            },
            "closed_loop_sensory": {
                "closed_loop_families": ["sensory_feedback_offset"],
                "open_loop_families": [
                    "initial_position_offset",
                    "initial_velocity_offset",
                    "process_epsilon_*",
                    "command_input_pulse",
                ],
            },
            "closed_loop_sensory_command_lateral": {
                "closed_loop_families": [
                    "sensory_feedback_offset",
                    "command_input_pulse",
                    "target_aligned_lateral_command_load_pulse",
                ],
                "open_loop_families": [
                    "initial_position_offset",
                    "initial_velocity_offset",
                    "process_epsilon_*",
                ],
            },
        },
        "unit_sensitivities": unit_sensitivities,
        "rows": rows,
        "row_summary": _row_summary(rows),
    }
    mkdir_p(output_path.parent)
    write_compact_json(output_path, payload)
    return payload


def _unit_sensitivity_rows(
    *,
    context: Mapping[str, Any],
    base: Any,
    base_cost: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for perturbation in _unit_perturbations():
        perturbed, initial_state, provenance = _simulate_extlqg_perturbed(
            perturbation,
            context=context,
        )
        perturbed_cost = _extlqg_cost_summary(perturbed, initial_state)
        metrics = summarize_perturbation_response(
            base,
            perturbed,
            base_full_qrf_cost=base_cost,
            perturbed_full_qrf_cost=perturbed_cost,
        )
        peak = _metric_mean(metrics, "delta_position_response_m.max")
        auc = _metric_mean(metrics, "delta_position_response_m.auc")
        if peak is None or peak <= 1e-12:
            status = "blocked"
            reason = "unit closed-loop peak delta x was unavailable or near zero"
        else:
            status = "available"
            reason = None
        rows.append(
            {
                "status": status,
                "reason": reason,
                "sensitivity_id": _sensitivity_id(perturbation),
                "family": perturbation["family"],
                "channel": perturbation["channel"],
                "component": perturbation.get("component"),
                "axis": perturbation["axis"],
                "sign": perturbation["sign"],
                "timing_bin": perturbation["timing"]["timing_bin"],
                "timing": perturbation["timing"],
                "units": perturbation["units"],
                "unit_amplitude": 1.0,
                "closed_loop_peak_delta_x_per_unit_m": peak,
                "closed_loop_auc_delta_x_per_unit_m_s": auc,
                "adapter_provenance": provenance,
            }
        )
    return rows


def _physical_level_rows(
    unit_sensitivities: Sequence[Mapping[str, Any]],
    *,
    reach_length_m: float,
    levels: Sequence[ReachRelativeLevel],
) -> list[dict[str, Any]]:
    rows = []
    for sensitivity in unit_sensitivities:
        if sensitivity["status"] != "available":
            continue
        peak_per_unit = float(sensitivity["closed_loop_peak_delta_x_per_unit_m"])
        for level in levels:
            target_peak = float(reach_length_m) * float(level.fraction_of_reach)
            amplitude = target_peak / peak_per_unit
            rows.append(
                {
                    "row_kind": "closed_loop_reach_relative_calibrated_amplitude",
                    "family": sensitivity["family"],
                    "channel": sensitivity["channel"],
                    "component": sensitivity.get("component"),
                    "axis": sensitivity["axis"],
                    "timing_bin": sensitivity["timing_bin"],
                    "physical_level": level.name,
                    "level_fraction_of_reach": float(level.fraction_of_reach),
                    "reach_length_m": float(reach_length_m),
                    "target_peak_delta_x_m": target_peak,
                    "amplitude": amplitude,
                    "units": sensitivity["units"],
                    "sensitivity_id": sensitivity["sensitivity_id"],
                    "closed_loop_peak_delta_x_per_unit_m": peak_per_unit,
                }
            )
    return rows


def _unit_perturbations() -> list[dict[str, Any]]:
    rows = []
    for timing_bin in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS:
        for component, units, axes in (
            ("position", "m", ("x", "y")),
            ("velocity", "m/s", ("vx", "vy")),
        ):
            for axis in axes:
                rows.append(
                    _perturbation(
                        family="sensory_feedback_offset",
                        channel="sensory_feedback",
                        component=component,
                        axis=axis,
                        units=units,
                        timing_bin=timing_bin,
                    )
                )
    for timing_bin in DEFAULT_PLANT_TIMING_BINS:
        for axis in ("x", "y"):
            rows.append(
                _perturbation(
                    family="command_input_pulse",
                    channel="command_input",
                    component="random_force_pulse_cardinal_basis",
                    axis=axis,
                    units="N",
                    timing_bin=timing_bin,
                )
            )
        rows.append(
            _perturbation(
                family="target_aligned_lateral_command_load_pulse",
                channel="command_input",
                component="target_aligned_lateral_load",
                axis="y",
                units="N",
                timing_bin=timing_bin,
            )
        )
    return rows


def _perturbation(
    *,
    family: str,
    channel: str,
    component: str,
    axis: str,
    units: str,
    timing_bin: TimingCalibrationBin,
) -> dict[str, Any]:
    return {
        "perturbation_id": (
            f"{family}__{component}__unit_closed_loop__"
            f"{timing_bin.label}_t{timing_bin.start_time_index}_{axis}_pos"
        ),
        "channel": channel,
        "family": family,
        "component": component,
        "amplitude": 1.0,
        "units": units,
        "axis": axis,
        "basis": (
            "sensory_feedback_named_channel"
            if channel == "sensory_feedback"
            else "command_cartesian_force_xy"
        ),
        "sign": 1,
        "timing": {
            "epoch": (
                "controller_visible" if channel == "sensory_feedback" else "movement_indexed"
            ),
            "start_time_index": int(timing_bin.start_time_index),
            "duration_steps": int(timing_bin.duration_steps),
            "timing_bin": timing_bin.label,
            "timing_bin_role": timing_bin.role,
        },
        "adapter": (
            "feedbax.additive_channel_adapter.sensory_feedback"
            if channel == "sensory_feedback"
            else "feedbax.additive_channel_adapter.command_input"
        ),
        "calibration_role": "closed_loop_6d_unit_sensitivity",
    }


def _sensitivity_id(perturbation: Mapping[str, Any]) -> str:
    return "__".join(
        [
            str(perturbation["family"]),
            str(perturbation.get("component")),
            str(perturbation["timing"]["timing_bin"]),
            str(perturbation["axis"]),
        ]
    )


def _metric_mean(metrics: Mapping[str, Any], dotted_key: str) -> float | None:
    current: Any = metrics
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, Mapping):
        current = current.get("mean")
    if current is None:
        return None
    value = float(current)
    return value if np.isfinite(value) else None


def _row_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for row in rows:
        family = str(row["family"])
        level = str(row["physical_level"])
        by_family[family] = by_family.get(family, 0) + 1
        by_level[level] = by_level.get(level, 0) + 1
    return {
        "count": len(rows),
        "by_family": by_family,
        "by_physical_level": by_level,
    }


def main() -> None:
    payload = materialize_closed_loop_calibration()
    print(json.dumps({"output": str(DEFAULT_OUTPUT_PATH), "rows": len(payload["rows"])}))


if __name__ == "__main__":
    main()
