"""Focused tests for the 50c260d affine tracker bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import materialize_output_feedback_affine_tracker as materializer  # noqa: E402
from rlrmp.analysis.output_feedback import (  # noqa: E402
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    rollout_with_kalman_estimator,
)
from rlrmp.analysis.output_feedback_affine_tracker import (  # noqa: E402
    baseline_conditions,
    rollout_with_affine_tracker,
    selected_coverage_conditions,
    staged_curriculum_conditions,
)
from rlrmp.analysis.cs_game_card import materialize_reference  # noqa: E402


def test_reference_affine_replay_matches_kalman_rollout() -> None:
    reference = materialize_reference()
    config = OutputFeedbackConfig()
    plant = reference.plant
    K_ref = reference.lqr_solution.K
    x0 = make_cs_output_feedback_initial_state(plant, config)
    kalman = rollout_with_kalman_estimator(plant, K_ref, x0, config=config)

    affine = rollout_with_affine_tracker(
        plant,
        K_ref,
        kalman.u,
        kalman.x_hat,
        x0,
        config=config,
    )

    np.testing.assert_allclose(affine.x, kalman.x, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(affine.x_hat, kalman.x_hat, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(affine.u, kalman.u, atol=1e-10, rtol=1e-10)


def test_row_labels_and_selected_coverage_metadata() -> None:
    legacy_labels = [condition.label for condition in baseline_conditions(maxiter=7)]
    assert legacy_labels == [
        "reference_affine_replay",
        "feedforward_only_k_ref_frozen",
        "gain_only_u_ref_frozen",
        "both_from_scratch",
        "spline_tracker_r20",
    ]

    labels = [condition.label for condition in staged_curriculum_conditions(maxiter=7)]
    assert labels == [
        "affine_clean_scratch_baseline",
        "affine_ff_clean_stage",
        "affine_fb_riccati_eps",
        "affine_joint_riccati_eps",
        "affine_fb_state_eig",
        "affine_joint_state_eig",
        "affine_fb_observer_error",
        "affine_joint_observer_error",
        "affine_fb_mixed",
        "affine_joint_mixed",
        "affine_feedback_action_match_riccati_eps",
        "affine_feedback_action_match_mixed",
    ]
    assert len(labels) == 12
    assert staged_curriculum_conditions(maxiter=7)[2].objective_family == "reward_rollout"
    assert staged_curriculum_conditions(maxiter=7)[2].stage_source_label == "affine_ff_clean_stage"
    assert staged_curriculum_conditions(maxiter=7)[10].is_diagnostic

    coverage = selected_coverage_conditions(maxiter=7)

    assert len(coverage) == 6
    assert coverage[0].label == "both_from_scratch__state_eigenspectrum_m4_s1_w0p1"
    assert coverage[0].training_distribution == "eigenspectrum_state"
    assert coverage[0].eigenspectrum_coverage.n_modes == 4
    assert coverage[1].eigenspectrum_coverage.scale == 3.0
    assert coverage[2].training_distribution == "observer_error_state"
    assert coverage[2].observer_error_coverage.n_modes == 1


def test_materializer_failure_adapter_receives_affine_rows(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_timed_run(**kwargs):
        captured["timed_run_kwargs"] = kwargs
        summary = {
            "fits": [
                {
                    "label": "both_from_scratch",
                    "condition": {
                        "row_kind": "both_from_scratch",
                        "train_feedforward": True,
                        "train_gain": True,
                        "gain_basis_rank": None,
                        "training_distribution": "nominal",
                        "eigenspectrum_coverage": None,
                        "observer_error_coverage": None,
                    },
                }
            ],
            "standard_certificate": {"rows": []},
        }
        arrays = {
            "both_from_scratch_K": np.zeros((2, 1, 2)),
            "both_from_scratch_clean_x_hat": np.zeros((3, 2)),
            "both_from_scratch_under_eps_x_hat": np.zeros((3, 2)),
            "lqr_reference_K": np.zeros((2, 1, 2)),
        }
        return summary, arrays

    def fake_failure_rows_from_manifest_entries(**kwargs):
        captured["failure_kwargs"] = kwargs
        return [{"classification": {"classification": "not_failure"}}]

    monkeypatch.setattr(materializer, "timed_run", fake_timed_run)
    monkeypatch.setattr(
        materializer.failure,
        "failure_rows_from_manifest_entries",
        fake_failure_rows_from_manifest_entries,
    )

    summary, arrays = materializer.materialize(
        maxiter=3,
        include_selected_coverage=False,
        manifest_path=tmp_path / "manifest.json",
    )

    assert summary["failure_decomposition"]["classification_counts"] == {"not_failure": 1}
    assert captured["timed_run_kwargs"]["maxiter"] == 3
    assert captured["failure_kwargs"]["entries"][0]["run_parts"] == (
        "affine_tracker",
        "both_from_scratch",
    )
    assert captured["failure_kwargs"]["entries"][0]["parameters"]["controller_family"] == (
        "affine_tracker"
    )
    assert arrays["both_from_scratch_K"].shape == (2, 1, 2)
