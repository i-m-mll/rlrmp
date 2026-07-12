"""Tests for governed reach-relative perturbation calibration computation."""

from __future__ import annotations

from rlrmp.data_products.calibration import calibrated_amplitude_from_unit_sensitivity
from rlrmp.data_products.calibration_computation import (
    calibration_origin_metadata,
    default_controller_visible_timing_bins,
    default_native_conventions,
    default_plant_timing_bins,
    default_reach_calibration_points,
    default_reach_relative_levels,
)


def test_default_reach_relative_calibration_config_is_explicit() -> None:
    reaches = {point.label: point for point in default_reach_calibration_points()}
    levels = {level.name: level for level in default_reach_relative_levels()}

    assert reaches["seen_train_0p10"].split == "seen/train"
    assert reaches["seen_train_0p10"].reach_length_m == 0.10
    assert reaches["seen_train_anchor_0p15"].split == "seen/train"
    assert reaches["seen_train_anchor_0p15"].reach_length_m == 0.15
    assert "original_anchor" in reaches["seen_train_anchor_0p15"].role
    assert reaches["heldout_eval_0p12"].split == "held-out/eval"
    assert reaches["heldout_eval_0p12"].reach_length_m == 0.12
    assert reaches["heldout_eval_0p18"].split == "held-out/eval"
    assert reaches["heldout_eval_0p18"].reach_length_m == 0.18

    assert levels["small"].fraction_of_reach == 0.05
    assert levels["moderate"].fraction_of_reach == 0.10
    assert levels["stress"].fraction_of_reach == 0.25


def test_default_timing_and_native_conventions_are_explicit() -> None:
    plant_bins = {timing_bin.label: timing_bin for timing_bin in default_plant_timing_bins()}
    visible_bins = {
        timing_bin.label: timing_bin for timing_bin in default_controller_visible_timing_bins()
    }
    native_conventions = {
        convention.family: convention for convention in default_native_conventions()
    }

    assert [plant_bins[label].start_time_index for label in ("early", "mid", "late")] == [
        5,
        15,
        35,
    ]
    assert all(plant_bins[label].duration_steps == 5 for label in ("early", "mid", "late"))
    assert [
        visible_bins[label].start_time_index
        for label in ("early_visible", "mid_visible", "late_visible")
    ] == [10, 20, 40]

    assert (
        "fractions of reach length"
        in native_conventions["sensory_feedback_offset"].native_unit_rule
    )
    assert (
        "pre-noise delayed-measurement"
        in native_conventions["delayed_observation_offset"].native_unit_rule
    )
    assert "fractions of reach length" in native_conventions["target_stream_jump"].native_unit_rule
    assert native_conventions["true_extra_delay_steps"].native_unit_rule.startswith("integer")


def test_reach_and_level_amplitudes_are_derived_from_unit_sensitivity() -> None:
    unit_sensitivity = 2.5

    small = calibrated_amplitude_from_unit_sensitivity(
        target_peak_delta_x_m=0.10 * 0.05,
        peak_delta_x_per_unit_m=unit_sensitivity,
    )
    moderate = calibrated_amplitude_from_unit_sensitivity(
        target_peak_delta_x_m=0.10 * 0.10,
        peak_delta_x_per_unit_m=unit_sensitivity,
    )
    heldout_moderate = calibrated_amplitude_from_unit_sensitivity(
        target_peak_delta_x_m=0.18 * 0.10,
        peak_delta_x_per_unit_m=unit_sensitivity,
    )

    assert abs(moderate - 2.0 * small) < 1e-12
    assert abs(heldout_moderate - 1.8 * moderate) < 1e-12


def test_nominal_origin_is_governed_preset_metadata() -> None:
    origin = calibration_origin_metadata()

    assert origin["experiment"] == "5f70333"
    assert origin["run_id"].startswith("lss_stabilization_fullqrf")
