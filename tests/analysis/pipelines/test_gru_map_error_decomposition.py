"""Tests for GRU observation-action map-error decomposition."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.pipelines.gru_map_error_decomposition import (
    ALIGNED_ACTION_CHANNELS,
    ALIGNED_OBSERVATION_CHANNELS,
    decompose_gru_map_error,
    render_map_error_decomposition_markdown,
)


def test_decomposition_reports_scalar_alignment_and_axis_energies() -> None:
    reference = np.zeros((1, 2, 2, 8))
    reference[..., 0, 0, 0] = 1.0
    reference[..., 1, 1, 5] = 2.0
    candidate = 0.5 * reference
    candidate[..., 1, 0, 6] += 3.0
    covariance = np.eye(8)
    covariance[6, 6] = 4.0

    result = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        input_covariance=covariance,
        input_covariance_metadata={"status": "available", "source": "unit"},
        top_k=2,
    )

    summary = result["summary"]
    assert summary["candidate_reference_norm_ratio"] > 1.0
    assert 0.0 < summary["candidate_reference_cosine"] < 1.0
    assert summary["best_scalar_gain"] == 0.5
    assert summary["best_scalar_residual_ratio"] == 9.0 / 5.0

    by_observation_channel = {
        entry["label"]: entry for entry in result["energy_decomposition"]["by_observation_channel"]
    }
    by_action_channel = {
        entry["label"]: entry for entry in result["energy_decomposition"]["by_action_channel"]
    }
    assert by_observation_channel["vx"]["energy"] == 9.0
    assert by_action_channel["ux"]["energy"] > by_action_channel["uy"]["energy"]
    assert result["top_singular_directions"][0]["covariance_projection"]["status"] == "available"
    assert result["top_singular_directions"][0]["dominant_observation_channel"] == "vx"
    assert result["top_singular_directions"][0]["dominant_action_channel"] == "ux"
    assert "well_excited_residual" in result["decision_rule_annotations"]


def test_decomposition_marks_covariance_projection_not_available() -> None:
    reference = np.ones((2, 3, 2, 12))
    candidate = np.zeros_like(reference)

    result = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        top_k=1,
    )

    projection = result["top_singular_directions"][0]["covariance_projection"]
    assert projection["status"] == "not_available"
    assert projection["missing_input"] == "input_covariance"
    assert "low_norm" in result["decision_rule_annotations"]
    assert "excitation_unknown" in result["decision_rule_annotations"]


def test_reach_alignment_on_axis_matches_raw_basis() -> None:
    reference = np.zeros((1, 1, 2, 4))
    candidate = np.zeros_like(reference)
    candidate[..., 0, 0] = 2.0
    covariance = np.eye(4)

    raw = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        input_covariance=covariance,
        top_k=1,
    )
    aligned = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        alignment_directions=np.array([1.0, 0.0]),
        alignment_frame_source="declared_static_reach_direction",
        target_time_convention="static_target_endpoint_minus_start",
        input_covariance=covariance,
        top_k=1,
    )

    assert aligned["summary"] == raw["summary"]
    assert aligned["energy_decomposition"]["by_observation_channel"][0]["energy"] == 4.0
    assert aligned["energy_decomposition"]["by_action_channel"][0]["energy"] == 4.0
    assert aligned["basis"]["observation_channels"] == list(ALIGNED_OBSERVATION_CHANNELS)
    assert aligned["basis"]["action_channels"] == list(ALIGNED_ACTION_CHANNELS)
    assert aligned["alignment"]["alignment_basis"] == "reach_aligned_parallel_lateral"
    assert aligned["alignment"]["direction_vectors"]["unit_direction"] == [1.0, 0.0]


def test_reference_observation_basis_transform_flips_target_relative_reference() -> None:
    candidate = np.zeros((1, 1, 2, 4))
    reference = np.zeros_like(candidate)
    candidate[..., 0, 0] = 1.0
    reference[..., 0, 0] = -1.0

    raw = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        top_k=1,
    )
    converted = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        reference_observation_from_candidate_transform=-np.eye(4),
        candidate_feedback_basis="target_relative_delayed_feedback",
        reference_feedback_basis="raw_delayed_position_velocity",
        top_k=1,
    )

    assert raw["summary"]["candidate_reference_cosine"] == -1.0
    assert converted["summary"]["candidate_reference_cosine"] == 1.0
    assert converted["summary"]["aggregate_delta_ratio"] == 0.0
    candidate_basis = converted["basis"]["candidate_observation_basis"]
    assert candidate_basis["reference_converted_to_candidate_basis"] is True
    assert candidate_basis["candidate_feedback_basis"] == "target_relative_delayed_feedback"
    assert "target_x - delayed_x" in candidate_basis["sign_convention"]


def test_reach_alignment_90_degree_direction_uses_feedbax_lateral_sign() -> None:
    reference = np.zeros((1, 1, 2, 4))
    candidate = np.zeros_like(reference)
    candidate[..., 0, 0] = 1.0

    result = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        alignment_directions=np.array([0.0, 1.0]),
        top_k=1,
    )

    by_observation_channel = {
        entry["label"]: entry for entry in result["energy_decomposition"]["by_observation_channel"]
    }
    by_action_channel = {
        entry["label"]: entry for entry in result["energy_decomposition"]["by_action_channel"]
    }
    assert by_observation_channel["p_lateral"]["energy"] == 1.0
    assert by_observation_channel["p_parallel"]["energy"] == 0.0
    assert by_action_channel["u_lateral"]["energy"] == 1.0
    assert by_action_channel["u_parallel"]["energy"] == 0.0
    assert result["top_singular_directions"][0]["dominant_observation_channel"] == "p_lateral"
    assert result["top_singular_directions"][0]["dominant_action_channel"] == "u_lateral"
    assert "lateral = cross" in result["alignment"]["sign_convention"]


def test_reach_alignment_supports_condition_wise_directions() -> None:
    reference = np.zeros((2, 1, 2, 4))
    candidate = np.zeros_like(reference)
    candidate[0, 0, 0, 0] = 1.0
    candidate[1, 0, 1, 1] = 1.0

    result = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        alignment_directions=np.array([[1.0, 0.0], [0.0, 1.0]]),
        input_covariance=np.eye(4),
        input_covariance_metadata={"status": "available"},
        top_k=1,
    )

    by_observation_channel = {
        entry["label"]: entry for entry in result["energy_decomposition"]["by_observation_channel"]
    }
    by_action_channel = {
        entry["label"]: entry for entry in result["energy_decomposition"]["by_action_channel"]
    }
    assert by_observation_channel["p_parallel"]["energy"] == 2.0
    assert by_action_channel["u_parallel"]["energy"] == 2.0
    assert result["alignment"]["direction_vectors"]["mode"] == "condition_wise"
    assert result["top_singular_directions"][0]["covariance_projection"]["status"] == (
        "not_available"
    )


def test_raw_behavior_and_metadata_are_preserved_by_default() -> None:
    reference = np.ones((1, 1, 2, 4))
    candidate = np.zeros_like(reference)

    result = decompose_gru_map_error(
        candidate_map=candidate,
        reference_map=reference,
        observation_dim=4,
        top_k=1,
    )

    assert result["basis"]["observation_channels"] == ["px", "py", "vx", "vy"]
    assert result["basis"]["action_channels"] == ["ux", "uy"]
    assert result["alignment"] == {
        "alignment_basis": "raw_cartesian",
        "frame_source": "raw_cartesian_observation_action_basis",
        "target_time_convention": "not_applicable_static_or_raw",
        "sign_convention": "raw ux/uy and px/py/vx/vy Cartesian channels",
        "direction_vectors": None,
        "target_velocity_used": False,
        "moving_target": {"status": "not_applicable"},
    }


def test_moving_target_alignment_is_explicitly_deferred_metadata() -> None:
    result = decompose_gru_map_error(
        candidate_map=np.zeros((1, 1, 2, 4)),
        reference_map=np.ones((1, 1, 2, 4)),
        observation_dim=4,
        alignment_directions=np.array([1.0, 0.0]),
        moving_target_status="deferred",
        target_time_convention="deferred_reference_trajectory_contract_required",
        top_k=1,
    )

    assert result["alignment"]["moving_target"]["status"] == "deferred"
    assert "reference trajectory" in result["alignment"]["moving_target"]["deferred_note"]
    assert result["alignment"]["target_time_convention"] == (
        "deferred_reference_trajectory_contract_required"
    )


def test_markdown_renderer_includes_decision_annotations() -> None:
    decomposition = decompose_gru_map_error(
        candidate_map=np.zeros((1, 1, 2, 4)),
        reference_map=np.ones((1, 1, 2, 4)),
        observation_dim=4,
        top_k=1,
    )
    markdown = render_map_error_decomposition_markdown(
        {
            "issue": "ddf7f43",
            "source_issue": "aacb9ed",
            "rows": [
                {
                    "run_id": "unit__nominal_clean",
                    "decomposition": decomposition,
                }
            ],
        }
    )

    assert "# GRU Map-Error Decomposition" in markdown
    assert "unit__nominal_clean" in markdown
    assert "low_norm" in markdown
