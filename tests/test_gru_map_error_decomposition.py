"""Tests for GRU observation-action map-error decomposition."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.gru_map_error_decomposition import (
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
