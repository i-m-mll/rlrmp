"""Tests for reach-relative perturbation calibration metadata."""

from __future__ import annotations

from rlrmp.analysis.gru_perturbation_calibration import (
    DEFAULT_REACH_CALIBRATION_POINTS,
    DEFAULT_REACH_RELATIVE_LEVELS,
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


def test_reach_relative_markdown_reports_targets_and_rerun_command() -> None:
    manifest = {
        "issue": "1ad3c16",
        "scope": "extlqg_nominal_command_open_loop_physical_effect_calibration",
        "nominal_gru_baseline_for_later_closed_loop_calibration": {
            "experiment": "5f70333",
            "run_id": "lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64",
        },
        "bulk_manifest_path": "_artifacts/1ad3c16/perturbation_open_loop_calibration/test.json",
        "reach_points": [point.to_json() for point in DEFAULT_REACH_CALIBRATION_POINTS[:1]],
        "level_definitions": [level.to_json() for level in DEFAULT_REACH_RELATIVE_LEVELS[:1]],
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
                "warning": None,
                "open_loop": {"status": "available"},
            }
        ],
    }

    markdown = render_calibration_markdown(manifest)

    assert "target peak `delta x = fraction * reach_length`" in markdown
    assert "`seen_train_0p10`" in markdown
    assert "5.000 mm" in markdown
    assert "5.000%" in markdown
    assert "uv run python scripts/materialize_perturbation_open_loop_calibration.py" in markdown
    assert "single-target nominal-only; documented for later closed-loop comparison" in markdown
