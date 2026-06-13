"""Tests for GRU post-hoc evaluation diagnostics."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.gru_evaluation_diagnostics import (
    RolloutEvaluation,
    compute_gru_gate_arrays,
    summarize_rollout_behavior,
)
from rlrmp.analysis.trial_alignment import TrialTiming


def test_summarize_rollout_behavior_reports_control_and_kinematic_metrics() -> None:
    evaluation = RolloutEvaluation(
        position=np.array([[[[0.05, 0.0], [0.12, 0.0], [0.16, 0.0]]]]),
        velocity=np.array([[[[1.0, 0.0], [0.5, 0.0], [-0.1, 0.0]]]]),
        command=np.array([[[[1.0, 0.0], [2.0, 0.0], [2.5, 0.0]]]]),
        hidden=np.array([[[[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]]]]),
        gru_input=np.zeros((1, 1, 3, 2)),
        initial_position=np.array([[0.0, 0.0]]),
        initial_velocity=np.array([[0.0, 0.0]]),
        target_position=np.array([[[0.15, 0.0], [0.15, 0.0], [0.15, 0.0]]]),
        dt=0.01,
    )

    summary = summarize_rollout_behavior(evaluation)

    assert summary["command_norm"]["max"] == 2.5
    assert summary["first_five_step_command_norm"]["count"] == 3
    assert summary["command_jerk_norm"]["mean"] == 0.5
    assert np.isclose(summary["endpoint_error_m"]["mean"], 0.01)
    assert summary["terminal_speed_m_s"]["mean"] == 0.1
    assert np.isclose(summary["overshoot_m"]["mean"], 0.01)
    assert summary["post_peak_forward_velocity_sign_changes"]["mean"] == 1.0
    assert summary["hidden_state_norm"]["max"] == 5.0


def test_summarize_rollout_behavior_uses_delayed_movement_window() -> None:
    position = np.zeros((1, 1, 8, 2), dtype=np.float64)
    position[0, 0, :, 0] = np.arange(8, dtype=np.float64)
    velocity = np.ones_like(position)
    command = np.ones((1, 1, 8, 2), dtype=np.float64)
    hidden = np.ones((1, 1, 8, 2), dtype=np.float64)
    target = np.zeros((1, 8, 2), dtype=np.float64)
    target[0, :, 0] = 6.0
    evaluation = RolloutEvaluation(
        position=position,
        velocity=velocity,
        command=command,
        hidden=hidden,
        gru_input=np.zeros((1, 1, 8, 2), dtype=np.float64),
        initial_position=np.array([[0.0, 0.0]]),
        initial_velocity=np.array([[0.0, 0.0]]),
        target_position=target,
        dt=0.01,
        timing=TrialTiming(
            is_delayed=True,
            go_index=np.array([2], dtype=np.int64),
            movement_horizon_steps=4,
            n_time_steps=8,
        ),
    )

    summary = summarize_rollout_behavior(evaluation)

    assert summary["time_basis"]["time_basis"] == "go_cue_aligned_canonical_movement_window"
    # Movement window is absolute samples [2, 6), so terminal position is x=5,
    # not the padded tail's x=7.
    assert summary["endpoint_error_m"]["mean"] == 1.0


def test_compute_gru_gate_arrays_reconstructs_equations() -> None:
    class Cell:
        use_bias = True
        weight_ih = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        weight_hh = np.array(
            [
                [0.5],
                [-0.5],
                [0.25],
            ]
        )
        bias = np.array([0.0, 0.0, 0.0])
        bias_n = np.array([0.1])

    gru_input = np.array([[[[0.2, -0.1], [0.0, 0.4]]]])
    hidden = np.array([[[[0.3], [0.2]]]])

    gates = compute_gru_gate_arrays(Cell(), gru_input, hidden)

    assert gates["reset"].shape == (1, 1, 2, 1)
    assert gates["update"].shape == (1, 1, 2, 1)
    assert gates["candidate"].shape == (1, 1, 2, 1)
    assert np.all((gates["reset"] >= 0.0) & (gates["reset"] <= 1.0))
    assert np.all((gates["update"] >= 0.0) & (gates["update"] <= 1.0))
