"""Tests for movement-window GRU/extLQG diagnostics."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.cs_game_card import build_canonical_game
from rlrmp.analysis.gru_movement_window_diagnostic import (
    MovementWindowSpec,
    ReachRollout,
    build_movement_window_diagnostic,
    score_movement_window_full_qrf_cost,
    summarize_direction_conditioned_velocity,
)


def test_direction_conditioned_velocity_groups_reach_axes() -> None:
    velocity = np.zeros((2, 4, 2), dtype=np.float64)
    velocity[0, :, 0] = [0.0, 1.0, 2.0, 1.0]
    velocity[1, :, 1] = [0.0, 0.5, 1.0, 0.5]
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    window = MovementWindowSpec(n_steps=4, dt=0.01)

    summary = summarize_direction_conditioned_velocity(
        velocity=velocity,
        reach_direction=directions,
        window=window,
    )

    assert summary["status"] == "available"
    assert set(summary["by_direction"]) == {"+x", "+y"}
    assert summary["by_direction"]["+x"]["n_samples"] == 1
    assert summary["by_direction"]["+x"]["peak_forward_velocity_m_s"] == 2.0
    assert summary["by_direction"]["+y"]["peak_forward_velocity_m_s"] == 1.0


def test_movement_window_full_qrf_cost_reports_command_term() -> None:
    _plant, schedule = build_canonical_game()
    n_steps = 5
    states = np.zeros((1, int(schedule.T), int(schedule.Q.shape[-1])), dtype=np.float64)
    commands = np.zeros((1, int(schedule.T), int(schedule.R.shape[-1])), dtype=np.float64)
    commands[:, :n_steps, :] = 1.0
    initial = np.zeros((1, int(schedule.Q.shape[-1])), dtype=np.float64)
    window = MovementWindowSpec(n_steps=n_steps)

    cost = score_movement_window_full_qrf_cost(
        states=states,
        commands=commands,
        initial_states=initial,
        target_position=np.zeros((2,), dtype=np.float64),
        window=window,
    )

    assert cost["status"] == "available"
    np.testing.assert_allclose(cost["running_state"]["mean"], 0.0)
    np.testing.assert_allclose(cost["terminal_state"]["mean"], 0.0)
    np.testing.assert_allclose(cost["command_control"]["mean"], 2.0 * n_steps)
    np.testing.assert_allclose(cost["total"]["mean"], 2.0 * n_steps)
    np.testing.assert_allclose(cost["term_sum_delta"]["mean"], 0.0)


def test_build_diagnostic_compares_gru_to_extlqg_reference() -> None:
    _plant, schedule = build_canonical_game()
    n_steps = 4
    state_dim = int(schedule.Q.shape[-1])
    command_dim = int(schedule.R.shape[-1])
    states = np.zeros((1, int(schedule.T), state_dim), dtype=np.float64)
    initial = np.zeros((1, state_dim), dtype=np.float64)
    ref_commands = np.ones((1, int(schedule.T), command_dim), dtype=np.float64)
    gru_commands = ref_commands * 2.0
    ref_velocity = np.ones((1, int(schedule.T), 2), dtype=np.float64)
    gru_velocity = ref_velocity * 2.0
    direction = np.asarray([[1.0, 0.0]], dtype=np.float64)
    window = MovementWindowSpec(n_steps=n_steps)

    diagnostic = build_movement_window_diagnostic(
        [
            ReachRollout(
                label="extLQG",
                role="extlqg",
                velocity=ref_velocity,
                command=ref_commands,
                states=states,
                initial_states=initial,
                target_position=np.zeros((2,), dtype=np.float64),
                reach_direction=direction,
            ),
            ReachRollout(
                label="GRU",
                role="gru",
                velocity=gru_velocity,
                command=gru_commands,
                states=states,
                initial_states=initial,
                target_position=np.zeros((2,), dtype=np.float64),
                reach_direction=direction,
            ),
        ],
        window=window,
    )

    comparison = diagnostic["comparisons"][0]
    assert comparison["candidate_label"] == "GRU"
    assert comparison["reference_label"] == "extLQG"
    direction_comparison = comparison["velocity_by_direction"]["directions"]["+x"]
    assert direction_comparison["peak_forward_velocity_delta_m_s"] == 1.0
    assert direction_comparison["peak_forward_velocity_ratio"] == 2.0
    cost_terms = comparison["full_qrf_cost"]["terms"]
    assert cost_terms["command_control"]["ratio_to_reference"] == 4.0
    assert cost_terms["total"]["ratio_to_reference"] == 4.0
