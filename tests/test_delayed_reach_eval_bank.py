"""Tests for reusable delayed-reach evaluation-bank specs."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    DEFAULT_DELAYED_REACH_GO_CUE_STEPS,
    DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
    delayed_reach_eval_bank_spec,
    delayed_reach_fixed_eval_bank_specs,
    delayed_reach_fixed_rescore_bank_spec,
    plan_fixed_bank_checkpoint_rescore,
)


def load_delayed_timing_velocity_materializer():
    """Load the issue-local delayed timing velocity materializer."""

    module_name = "rlrmp_test_delayed_timing_velocity_materializer"
    script_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "40e1911"
        / "scripts"
        / "materialize_delayed_timing_hold_lane_velocity_profiles.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_uniform_delayed_eval_bank_preserves_historical_grid_contract() -> None:
    bank = delayed_reach_eval_bank_spec(
        bank_role="no_catch",
        direction_count=4,
    )

    payload = bank.to_json()

    assert payload["schema_version"] == DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION
    assert payload["kind"] == "no_catch"
    assert payload["catch"] is False
    assert payload["go_cue_min"] == 10
    assert payload["go_cue_max"] == 30
    assert payload["go_cue_steps"] == list(DEFAULT_DELAYED_REACH_GO_CUE_STEPS)
    assert payload["direction_source"] == "uniform_grid"
    assert payload["direction_count"] == 4
    assert payload["requested_direction_count"] == 4
    assert payload["trial_count"] == 21 * 4
    assert payload["movement_horizon_steps"] == 60
    assert payload["reach_length_m"] == DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M
    assert payload["reach_length_source"] == "uniform_grid_default"
    assert payload["duplicate_direction_count"] == 0
    assert payload["source_trial_indices"] == [0, 1, 2, 3]
    assert payload["target_radii_m"] == pytest.approx([0.15, 0.15, 0.15, 0.15])
    assert payload["target_angles_rad"] == pytest.approx(
        [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
    )


def test_validation_target_direction_source_preserves_radii_angles_and_duplicates() -> None:
    bank = delayed_reach_eval_bank_spec(
        bank_role="catch",
        direction_source="validation_targets",
        direction_count=20,
    )

    payload = bank.to_json()

    assert payload["kind"] == "catch"
    assert payload["catch"] is True
    assert payload["direction_source"] == "validation_targets"
    assert payload["direction_source_inferred_from_validation_targets"] is True
    assert payload["trial_count"] == 21 * 20
    assert payload["duplicate_direction_count"] == 10
    assert payload["validation_target_provenance"]["n_target_conditions"] == 20
    assert payload["validation_target_provenance"]["duplicate_direction_metadata"][
        "duplicate_direction_count"
    ] == 10
    assert sorted(set(round(radius, 2) for radius in payload["target_radii_m"])) == [
        0.1,
        0.12,
        0.15,
        0.18,
    ]
    assert payload["target_angles_rad"][:3] == pytest.approx(
        [0.0, 0.0, math.pi / 3.0]
    )
    roles = [
        row["target_role"]
        for row in payload["validation_target_provenance"]["actual_targets"]
    ]
    assert roles[0] == "original_anchor"
    assert "held_out_validation_support" in roles


def test_delayed_reach_fixed_rescore_bank_spec_carries_no_catch_and_catch_payloads() -> None:
    no_catch, catch = delayed_reach_fixed_eval_bank_specs(direction_count=4)
    assert [bank.bank_role for bank in (no_catch, catch)] == ["no_catch", "catch"]
    assert [bank.p_catch_trial for bank in (no_catch, catch)] == [0.0, 1.0]

    fixed_bank = delayed_reach_fixed_rescore_bank_spec(direction_count=4)
    payload = fixed_bank.to_json()

    assert payload["bank_identity"] == "delayed_reach_go_cue_grid_no_catch_catch"
    assert payload["scorer_identity"] == "feedbax_task_loss_mean_over_trials"
    assert payload["n_trials"] == 2 * 21 * 4
    assert payload["validation_role"] == (
        "fixed_delayed_reach_no_catch_catch_rollout_validation"
    )
    assert payload["selection_metric"] == "mean_task_loss_equal_weight_over_declared_banks"
    assert payload["bank_spec"]["selection_source"] == "delayed_reach_fixed_bank_rescore"
    assert payload["bank_spec"]["bank_kinds"] == ["no_catch", "catch"]
    assert [bank["kind"] for bank in payload["bank_spec"]["banks"]] == [
        "no_catch",
        "catch",
    ]


def test_default_direction_count_matches_historical_fixed_bank() -> None:
    bank = delayed_reach_eval_bank_spec(bank_role="no_catch")
    payload = bank.to_json()

    assert DEFAULT_DELAYED_REACH_DIRECTION_COUNT == 20
    assert payload["direction_count"] == 20
    assert payload["trial_count"] == 21 * 20


def test_fixed_bank_projection_direction_uses_intended_target_for_catch() -> None:
    materializer = load_delayed_timing_velocity_materializer()

    direction = materializer.fixed_bank_projection_direction(
        {
            "bank_family": "delayed_reach_fixed_eval_bank",
            "catch": True,
            "direction_count": 4,
            "target_angles_rad": [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi],
        },
        trial_count=8,
    )

    expected = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    assert direction == pytest.approx(expected)


def test_delayed_fixed_rescore_plan_uses_delayed_selection_source(tmp_path) -> None:
    bank = delayed_reach_fixed_rescore_bank_spec(direction_count=4)

    manifest = plan_fixed_bank_checkpoint_rescore(
        experiment="issue123",
        run_ids=("run_a",),
        validation_bank=bank,
        repo_root=tmp_path,
    )

    assert manifest["selection_source"] == "delayed_reach_fixed_bank_rescore"
    assert manifest["validation_bank"]["bank_spec"]["bank_kinds"] == ["no_catch", "catch"]
