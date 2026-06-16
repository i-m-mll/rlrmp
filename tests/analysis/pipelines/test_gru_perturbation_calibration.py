"""Tests for reach-relative perturbation calibration metadata."""

from __future__ import annotations

import json
from pathlib import Path

from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
    DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS,
    DEFAULT_NATIVE_CONVENTIONS,
    DEFAULT_PLANT_TIMING_BINS,
    DEFAULT_REACH_CALIBRATION_POINTS,
    DEFAULT_REACH_RELATIVE_LEVELS,
    _write_calibration_regeneration_spec,
    calibrated_amplitude_from_unit_sensitivity,
    render_calibration_markdown,
)


def test_default_reach_relative_calibration_config_is_explicit() -> None:
    reaches = {point.label: point for point in DEFAULT_REACH_CALIBRATION_POINTS}
    levels = {level.name: level for level in DEFAULT_REACH_RELATIVE_LEVELS}

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
    plant_bins = {timing_bin.label: timing_bin for timing_bin in DEFAULT_PLANT_TIMING_BINS}
    visible_bins = {
        timing_bin.label: timing_bin for timing_bin in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS
    }
    native_conventions = {convention.family: convention for convention in DEFAULT_NATIVE_CONVENTIONS}

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

    assert "fractions of reach length" in native_conventions[
        "sensory_feedback_offset"
    ].native_unit_rule
    assert "pre-noise delayed-measurement" in native_conventions[
        "delayed_observation_offset"
    ].native_unit_rule
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


def test_reach_relative_markdown_reports_targets_and_rerun_command() -> None:
    manifest = {
        "issue": "1ad3c16",
        "scope": "extlqg_nominal_command_open_loop_physical_effect_calibration",
        "nominal_gru_baseline_for_later_closed_loop_calibration": {
            "experiment": "5f70333",
            "run_id": "lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64",
        },
        "bulk_manifest_path": "_artifacts/1ad3c16/perturbation_open_loop_calibration/test.json",
        "regeneration_spec_path": (
            "_artifacts/1ad3c16/perturbation_open_loop_calibration/"
            "perturbation_open_loop_calibration_regeneration_spec.json"
        ),
        "reach_points": [point.to_json() for point in DEFAULT_REACH_CALIBRATION_POINTS[:1]],
        "level_definitions": [level.to_json() for level in DEFAULT_REACH_RELATIVE_LEVELS[:1]],
        "plant_timing_bins": [timing_bin.to_json() for timing_bin in DEFAULT_PLANT_TIMING_BINS],
        "native_conventions": [
            convention.to_json() for convention in DEFAULT_NATIVE_CONVENTIONS[:1]
        ],
        "rows": [
            {
                "row_kind": "reach_relative_calibrated_amplitude",
                "reach_label": "seen_train_0p10",
                "reach_split": "seen/train",
                "reach_length_m": 0.10,
                "level_name": "small",
                "level_fraction_of_reach": 0.05,
                "family": "initial_position_offset",
                "amplitude": 0.005,
                "target_peak_delta_x_m": 0.005,
                "achieved_peak_delta_x_m": 0.005,
                "achieved_peak_delta_x_fraction_of_reach": 0.05,
                "achieved_auc_delta_x_m_s": 0.003,
                "timing_bin": "initial_condition",
                "sensitivity_reference": {
                    "sensitivity_id": "initial_position_offset__initial_condition"
                },
                "warning": None,
                "open_loop": {"status": "available"},
            }
        ],
    }

    markdown = render_calibration_markdown(manifest)

    assert "target peak `delta x = fraction * reach_length`" in markdown
    assert "Unit sensitivities are calibrated by perturbation family and timing bin" in markdown
    assert "| `early` | 5 | 5 | plant_side_open_loop_calibration |" in markdown
    assert "`sensory_feedback_offset`" in markdown
    assert "`seen_train_0p10`" in markdown
    assert "`initial_condition`" in markdown
    assert "5.000 mm" in markdown
    assert "5.000%" in markdown
    assert "uv run python scripts/materialize_perturbation_open_loop_calibration.py" in markdown
    assert "single-target nominal-only; documented for later closed-loop comparison" in markdown
    assert "Regeneration spec:" in markdown


def test_calibration_regeneration_spec_records_source_model_and_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "_artifacts" / "1ad3c16" / "calibration.json"
    note_path = tmp_path / "results" / "1ad3c16" / "notes" / "calibration.md"
    spec_path = tmp_path / "_artifacts" / "1ad3c16" / "calibration_regeneration_spec.json"
    output_path.parent.mkdir(parents=True)
    note_path.parent.mkdir(parents=True)
    output_path.write_text("{}", encoding="utf-8")
    note_path.write_text("# note\n", encoding="utf-8")

    _write_calibration_regeneration_spec(
        spec_path=spec_path,
        output_path=output_path,
        note_path=note_path,
        manifest={
            "bank_schema_version": "bank.v1",
            "reach_points": [DEFAULT_REACH_CALIBRATION_POINTS[0].to_json()],
            "level_definitions": [DEFAULT_REACH_RELATIVE_LEVELS[0].to_json()],
            "plant_timing_bins": [DEFAULT_PLANT_TIMING_BINS[0].to_json()],
            "controller_visible_timing_bins": [DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS[0].to_json()],
        },
        amplitude_factors=(0.1, 1.0),
        result_experiment="1ad3c16",
        repo_root=tmp_path,
    )

    regeneration = json.loads(spec_path.read_text())
    assert regeneration["metadata"]["diagnostic_name"] == "perturbation_open_loop_calibration"
    assert regeneration["parameters"]["source_model"]["run_id"]
    assert "extLQG nominal command replay" in regeneration["parameters"]["source_run_ids"]
    assert {item["role"] for item in regeneration["outputs"]} == {
        "calibration_bulk_manifest",
        "calibration_markdown_note",
    }
