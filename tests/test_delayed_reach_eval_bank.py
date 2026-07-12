"""Tests for reusable delayed-reach evaluation-bank specs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlrmp.eval.ensemble import fixed_bank_projection_direction

from rlrmp.eval.checkpoint_selection import (
    CHECKPOINT_SELECTION_BANK_SCHEMA_ID,
    DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    DEFAULT_DELAYED_REACH_GO_CUE_STEPS,
    DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    FEEDBAX_MANIFEST_SCHEMA_VERSION,
    delayed_reach_eval_bank_spec,
    delayed_reach_fixed_eval_bank_specs,
    delayed_reach_fixed_rescore_bank_spec,
    plan_fixed_bank_checkpoint_selection,
)
from rlrmp.train.cs_perturbation_training import (
    TARGET_SUPPORT_PROFILE_020A65B,
    TARGET_SUPPORT_PROFILE_CONST_BAND16,
    target_relative_target_support_config,
)

def test_uniform_delayed_eval_bank_preserves_historical_grid_contract() -> None:
    bank = delayed_reach_eval_bank_spec(
        bank_role="no_catch",
        direction_count=4,
    )

    payload = bank.to_json()

    assert payload["schema_id"] == CHECKPOINT_SELECTION_BANK_SCHEMA_ID
    assert payload["schema_version"] == FEEDBAX_MANIFEST_SCHEMA_VERSION
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


def test_validation_target_direction_source_uses_default_band16_fixed_reach() -> None:
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
    assert payload["duplicate_direction_count"] == 0
    assert payload["validation_target_provenance"]["n_target_conditions"] == 20
    assert (
        payload["validation_target_provenance"]["duplicate_direction_metadata"][
            "duplicate_direction_count"
        ]
        == 0
    )
    assert payload["target_distribution"]["target_support_profile"] == (
        TARGET_SUPPORT_PROFILE_CONST_BAND16
    )
    assert sorted(set(round(radius, 2) for radius in payload["target_radii_m"])) == [0.15]
    assert payload["target_angles_rad"][:3] == pytest.approx([0.0, math.pi / 36.0, math.pi / 18.0])
    roles = [
        row["target_role"] for row in payload["validation_target_provenance"]["actual_targets"]
    ]
    assert roles[0] == "original_anchor"
    assert "seen_training_support" in roles


def test_validation_target_direction_source_preserves_explicit_old_profile() -> None:
    bank = delayed_reach_eval_bank_spec(
        bank_role="catch",
        direction_source="validation_targets",
        direction_count=20,
        target_config=target_relative_target_support_config(
            profile=TARGET_SUPPORT_PROFILE_020A65B,
            enabled=True,
        ),
    )

    payload = bank.to_json()

    assert payload["duplicate_direction_count"] == 10
    assert (
        payload["target_distribution"]["target_support_profile"] == TARGET_SUPPORT_PROFILE_020A65B
    )
    assert sorted(set(round(radius, 2) for radius in payload["target_radii_m"])) == [
        0.1,
        0.12,
        0.15,
        0.18,
    ]
    assert payload["target_angles_rad"][:3] == pytest.approx([0.0, 0.0, math.pi / 3.0])


def test_delayed_reach_fixed_rescore_bank_spec_carries_no_catch_and_catch_payloads() -> None:
    no_catch, catch = delayed_reach_fixed_eval_bank_specs(direction_count=4)
    assert [bank.bank_role for bank in (no_catch, catch)] == ["no_catch", "catch"]
    assert [bank.p_catch_trial for bank in (no_catch, catch)] == [0.0, 1.0]

    fixed_bank = delayed_reach_fixed_rescore_bank_spec(direction_count=4)
    payload = fixed_bank.to_json()

    assert payload["bank_identity"] == "delayed_reach_go_cue_grid_no_catch_catch"
    assert payload["scorer_identity"] == "feedbax_task_loss_mean_over_trials"
    assert payload["n_trials"] == 2 * 21 * 4
    assert payload["validation_role"] == ("fixed_delayed_reach_no_catch_catch_rollout_validation")
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
    direction = fixed_bank_projection_direction(
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


@pytest.mark.parametrize(
    "metadata",
    (
        {"bank_family": "delayed_reach_fixed_eval_bank"},
        {
            "bank_family": "delayed_reach_fixed_eval_bank",
            "target_angles_rad": [0.0],
        },
        {
            "bank_family": "delayed_reach_fixed_eval_bank",
            "direction_count": 4,
        },
        {
            "bank_family": "delayed_reach_fixed_eval_bank",
            "direction_count": 0,
            "target_angles_rad": [],
        },
    ),
)
def test_fixed_bank_projection_direction_preserves_incomplete_metadata_fallback(
    metadata: dict[str, object],
) -> None:
    assert fixed_bank_projection_direction(metadata, trial_count=4) is None


def test_delayed_fixed_rescore_plan_uses_delayed_selection_source(tmp_path) -> None:
    bank = delayed_reach_fixed_rescore_bank_spec(direction_count=4)

    manifest = plan_fixed_bank_checkpoint_selection(
        experiment="issue123",
        run_ids=("run_a",),
        validation_bank=bank,
        repo_root=tmp_path,
    )

    assert manifest.selection_type == "delayed_reach_fixed_bank_rescore"
    assert manifest.bank.metadata["validation_bank"]["bank_spec"]["bank_kinds"] == [
        "no_catch",
        "catch",
    ]
