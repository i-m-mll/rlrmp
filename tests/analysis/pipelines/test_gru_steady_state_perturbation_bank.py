"""Tests for the steady-state GRU perturbation bank."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
    FeedbackPerturbation,
    aggregate_family_profiles,
    build_response_figure,
    default_feedback_perturbations,
    identity_condition,
    make_steady_state_trial_specs,
    pad_feedback_offset_inputs,
    right_handed_orthogonal_direction,
    signed_pair_antisymmetry,
    summarize_feedback_row,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import RolloutEvaluation


class CartesianState(eqx.Module):
    """Tiny target-state test double."""

    pos: jnp.ndarray
    vel: jnp.ndarray | None = None
    force: jnp.ndarray | None = None


class DelayedTaskInputs(eqx.Module):
    """Tiny delayed-task input test double."""

    effector_target: CartesianState
    hold: jnp.ndarray
    target_on: jnp.ndarray


class TrialSpec(eqx.Module):
    """Tiny TaskTrialSpec-like test double."""

    inits: dict[str, jnp.ndarray]
    targets: dict[str, jnp.ndarray]
    inputs: dict[str, object]


def test_default_bank_is_symmetric_and_omits_force_filter_for_4d_feedback() -> None:
    bank_4d = default_feedback_perturbations(feedback_dim=4)
    bank_6d = default_feedback_perturbations(feedback_dim=6)

    assert {row.family for row in bank_4d} == {"position", "velocity"}
    assert {row.family for row in bank_6d} == {"position", "velocity", "force_filter"}
    assert {row.amplitude for row in bank_6d if row.family == "force_filter"} == {10.0}
    for family in {row.family for row in bank_6d}:
        directions = {row.direction for row in bank_6d if row.family == family}
        assert directions == {(1.0, 0.0), (-1.0, -0.0), (0.0, 1.0), (-0.0, -1.0)}


def test_orthogonal_direction_uses_right_handed_plus_90_convention() -> None:
    np.testing.assert_allclose(
        right_handed_orthogonal_direction(np.array([1.0, 0.0])),
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        right_handed_orthogonal_direction(np.array([0.0, 1.0])),
        [-1.0, 0.0],
    )
    np.testing.assert_allclose(
        right_handed_orthogonal_direction(np.array([-1.0, 0.0])),
        [0.0, -1.0],
    )


def test_washin_transform_sets_target_state_and_parametric_pulse_duration() -> None:
    trials = _trial_spec(horizon=12, feedback_dim=6)

    updated, timing = make_steady_state_trial_specs(
        trials,
        delayed=True,
        target_position=np.array([0.15, -0.02]),
        pre_go_steps=3,
        post_go_washin_steps=4,
        pulse_duration_steps=1,
    )

    vector = np.asarray(updated.inits["mechanics.vector"])
    assert timing["pulse_start_step"] == 7
    assert timing["pulse_duration_steps"] == 1
    np.testing.assert_allclose(vector[..., 0:2], [[0.15, -0.02]])
    np.testing.assert_allclose(vector[..., 8:10], [[0.15, -0.02]])
    np.testing.assert_allclose(vector[..., 2:8], 0.0)
    controller_input = np.asarray(updated.inputs["input"])
    np.testing.assert_allclose(controller_input[0, :3, 0], 0.0)
    np.testing.assert_allclose(controller_input[0, 3:, 0], 1.0)
    task = updated.inputs["task"]
    np.testing.assert_allclose(np.asarray(task.hold)[0, :3, 0], 1.0)
    np.testing.assert_allclose(np.asarray(task.hold)[0, 3:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(updated.inputs["epsilon"]), 0.0)
    np.testing.assert_allclose(
        np.asarray(updated.inputs["target"])[0, :, :],
        np.tile([0.15, -0.02], (12, 1)),
    )


def test_pulse_start_preserves_default_post_onset_window_when_possible() -> None:
    trials = _trial_spec(horizon=80, feedback_dim=6)

    _, timing = make_steady_state_trial_specs(
        trials,
        delayed=False,
        target_position=np.array([0.0, 0.0]),
        post_go_washin_steps=70,
    )

    assert timing["pulse_start_step_requested"] == 70
    assert timing["pulse_start_step"] == 30
    assert timing["post_onset_steps_available"] == 50


def test_undelayed_washin_extends_short_hold_trials_to_keep_recovery_window() -> None:
    trials = _trial_spec(horizon=60, feedback_dim=6)

    updated, timing = make_steady_state_trial_specs(
        trials,
        delayed=False,
        target_position=np.array([0.0, 0.0]),
        post_go_washin_steps=30,
        min_post_onset_steps=50,
    )

    assert timing["pulse_start_step_requested"] == 30
    assert timing["pulse_start_step_requested_meaning"].startswith("legacy compatibility")
    assert timing["pulse_start_step_horizon_clamped"] == 30
    assert timing["pulse_start_step"] == 30
    assert timing["post_go_washin_steps_actual"] == 30
    assert timing["post_onset_steps_available"] == 50
    assert timing["horizon_steps"] == 80
    assert timing["horizon_extension"]["extended"] is True
    assert timing["horizon_extension"]["original_horizon_steps"] == 60
    assert np.asarray(updated.inputs["target"]).shape[1] == 80
    assert np.asarray(updated.inputs["input"]).shape[1] == 79
    assert np.asarray(updated.inputs["epsilon"]).shape[1] == 79
    assert np.asarray(updated.inputs["task"].hold).shape[1] == 79


def test_delayed_washin_keeps_pre_go_plus_post_go_hold_timing() -> None:
    trials = _trial_spec(horizon=90, feedback_dim=6)

    _, timing = make_steady_state_trial_specs(
        trials,
        delayed=True,
        target_position=np.array([0.0, 0.0]),
        pre_go_steps=10,
        post_go_washin_steps=30,
        min_post_onset_steps=50,
    )

    assert timing["pre_go_steps"] == 10
    assert timing["post_go_washin_steps_actual"] == 30
    assert timing["pulse_start_step_requested"] == 40
    assert timing["pulse_start_step_horizon_clamped"] == 40
    assert timing["pulse_start_step"] == 40
    assert timing["post_onset_steps_available"] == 50
    assert timing["horizon_extension"]["extended"] is False


def test_direction_aligned_aggregation_and_antisymmetry() -> None:
    rows = [
        _row("position", (1.0, 0.0), [1.0, 0.5]),
        _row("position", (-1.0, 0.0), [0.9, 0.4]),
        _row("position", (0.0, 1.0), [1.1, 0.6]),
        _row("position", (0.0, -1.0), [1.0, 0.5]),
    ]

    summary = aggregate_family_profiles(rows)
    antisymmetry = signed_pair_antisymmetry(rows)

    np.testing.assert_allclose(summary["position"]["aligned_output_profile_mean"], [1.0, 0.5])
    np.testing.assert_allclose(summary["position"]["orthogonal_output_profile_mean"], [0.1, 0.05])
    assert summary["position"]["n_rows"] == 4
    assert antisymmetry["status"] == "available"
    assert antisymmetry["mean_aligned_pair_difference_ratio"] < 0.08


def test_family_aggregation_preserves_onset_window_rows() -> None:
    rows = [
        _row(
            "velocity",
            (1.0, 0.0),
            [1.0, 0.5],
            relative_steps=[-1, 0, 1],
            output_window=[0.0, 1.0, 0.5],
            orthogonal_window=[0.0, 0.1, 0.2],
            position_window=[0.0, 0.1, 0.2],
            orthogonal_position_window=[0.0, -0.1, -0.2],
            velocity_window=[0.0, 0.3, 0.1],
            orthogonal_velocity_window=[0.0, -0.3, -0.1],
        ),
        _row(
            "velocity",
            (-1.0, 0.0),
            [1.0, 0.5],
            relative_steps=[-1, 0, 1],
            output_window=[0.0, 1.2, 0.7],
            orthogonal_window=[0.0, 0.3, 0.4],
            position_window=[0.0, 0.2, 0.4],
            orthogonal_position_window=[0.0, -0.2, -0.4],
            velocity_window=[0.0, 0.4, 0.2],
            orthogonal_velocity_window=[0.0, -0.4, -0.2],
        ),
    ]

    summary = aggregate_family_profiles(rows)["velocity"]

    assert summary["relative_time_steps"] == [-1, 0, 1]
    np.testing.assert_allclose(summary["aligned_output_window_profile_mean"], [0.0, 1.1, 0.6])
    np.testing.assert_allclose(
        summary["orthogonal_output_window_profile_mean"],
        [0.0, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        summary["orthogonal_output_window_profile_sem"],
        [0.0, 0.1, 0.1],
    )
    np.testing.assert_allclose(summary["aligned_position_window_profile_mean"], [0.0, 0.15, 0.3])
    np.testing.assert_allclose(
        summary["orthogonal_position_window_profile_mean"],
        [0.0, -0.15, -0.3],
    )
    np.testing.assert_allclose(summary["aligned_velocity_window_profile_mean"], [0.0, 0.35, 0.15])
    np.testing.assert_allclose(
        summary["orthogonal_velocity_window_profile_mean"],
        [0.0, -0.35, -0.15],
    )


def test_summarize_feedback_row_stores_orthogonal_plant_projections() -> None:
    base = _rollout_evaluation()
    perturbed = _rollout_evaluation(
        command_delta=[[[[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]]]],
        position_delta=[[[[0.0, 0.0], [3.0, 30.0], [4.0, 40.0]]]],
        velocity_delta=[[[[0.0, 0.0], [5.0, 50.0], [6.0, 60.0]]]],
    )
    perturbation = FeedbackPerturbation(
        perturbation_id="position_x_pos",
        family="position",
        feedback_indices=(0, 1),
        direction=(1.0, 0.0),
        amplitude=0.1,
        units="m",
        sign=1,
    )

    row = summarize_feedback_row(
        perturbation=perturbation,
        base=base,
        perturbed=perturbed,
        pulse_start=1,
    )

    np.testing.assert_allclose(row["orthogonal_output_profile"], [10.0, 20.0])
    np.testing.assert_allclose(row["orthogonal_position_profile"], [30.0, 40.0])
    np.testing.assert_allclose(row["orthogonal_velocity_profile"], [50.0, 60.0])
    np.testing.assert_allclose(row["orthogonal_position_window_profile"], [0.0, 30.0, 40.0])
    np.testing.assert_allclose(row["orthogonal_velocity_window_profile"], [0.0, 50.0, 60.0])
    assert row["projection_basis"]["orthogonal_convention"] == "right_handed_plus_90_degrees_xy"


def test_response_figure_adds_solid_lower_emphasis_orthogonal_traces_to_all_rows() -> None:
    conditions = {
        "a": {
            "label": "A",
            "family_summary": {"position": _family_summary()},
        },
        "b": {
            "label": "B",
            "family_summary": {"position": _family_summary()},
        },
    }

    figure = build_response_figure(
        comparison_title="test",
        conditions=conditions,
        dt=0.01,
        pulse_duration_steps=5,
    )

    legend_traces = [trace for trace in figure.data if trace.showlegend]
    assert [trace.name for trace in legend_traces] == [
        "A aligned",
        "A orthogonal",
        "B aligned",
        "B orthogonal",
    ]
    orthogonal_traces = [trace for trace in figure.data if trace.name == "A orthogonal"]
    assert len(orthogonal_traces) == 3
    for trace in orthogonal_traces:
        assert trace.line.dash == "solid"
        assert trace.line.width < 2.1
        assert "0.6" in trace.line.color


def test_feedback_offset_padding_preserves_existing_components() -> None:
    trials = _trial_spec(horizon=12, feedback_dim=4)

    padded = pad_feedback_offset_inputs(trials, expected_feedback_dim=6)

    payload = np.asarray(padded.inputs["perturbation_training.sensory_feedback"])
    assert payload.shape == (1, 12, 6)
    np.testing.assert_allclose(payload[..., :4], 0.0)
    np.testing.assert_allclose(payload[..., 4:], 0.0)


def test_identity_condition_accepts_run_override() -> None:
    condition = identity_condition("pgd", "PGD", run_id="run-b")

    assert condition.condition_id == "pgd"
    assert condition.run_id == "run-b"


def _trial_spec(*, horizon: int, feedback_dim: int) -> TrialSpec:
    state = CartesianState(
        pos=jnp.zeros((1, horizon, 2)),
        vel=jnp.zeros((1, horizon - 1, 2)),
    )
    task = DelayedTaskInputs(
        effector_target=state,
        hold=jnp.zeros((1, horizon - 1, 1)),
        target_on=jnp.zeros((1, horizon - 1, 1)),
    )
    return TrialSpec(
        inits={"mechanics.vector": jnp.ones((1, 16))},
        targets={},
        inputs={
            "epsilon": jnp.ones((1, horizon - 1, 8)),
            "input": jnp.ones((1, horizon - 1, 2)),
            "target": jnp.zeros((1, horizon, 2)),
            "task": task,
            "perturbation_training.sensory_feedback": jnp.zeros((1, horizon, feedback_dim)),
        },
    )


def _row(
    family: str,
    direction: tuple[float, float],
    profile: list[float],
    *,
    relative_steps: list[int] | None = None,
    output_window: list[float] | None = None,
    orthogonal_profile: list[float] | None = None,
    orthogonal_window: list[float] | None = None,
    position_window: list[float] | None = None,
    orthogonal_position_window: list[float] | None = None,
    velocity_window: list[float] | None = None,
    orthogonal_velocity_window: list[float] | None = None,
) -> dict[str, object]:
    row = {
        "status": "evaluated",
        "family": family,
        "direction": direction,
        "aligned_output_profile": profile,
        "orthogonal_output_profile": (
            orthogonal_profile
            if orthogonal_profile is not None
            else [0.1 * value for value in profile]
        ),
        "aligned_position_profile": position_window or profile,
        "orthogonal_position_profile": orthogonal_position_window
        or [0.01 * value for value in profile],
        "aligned_velocity_profile": velocity_window or profile,
        "orthogonal_velocity_profile": orthogonal_velocity_window
        or [0.02 * value for value in profile],
        "metrics": {
            "peak_output_response": max(abs(value) for value in profile),
            "peak_orthogonal_output_response": max(
                abs(value)
                for value in (
                    orthogonal_profile
                    if orthogonal_profile is not None
                    else [0.1 * value for value in profile]
                )
            ),
            "output_auc_impulse": sum(abs(value) for value in profile),
            "orthogonal_output_auc_impulse": sum(
                abs(value)
                for value in (
                    orthogonal_profile
                    if orthogonal_profile is not None
                    else [0.1 * value for value in profile]
                )
            ),
            "terminal_residual": abs(profile[-1]),
        },
    }
    if relative_steps is not None:
        row["relative_time_steps"] = relative_steps
    if output_window is not None:
        row["aligned_output_window_profile"] = output_window
    if orthogonal_window is not None:
        row["orthogonal_output_window_profile"] = orthogonal_window
    if position_window is not None:
        row["aligned_position_window_profile"] = position_window
    if orthogonal_position_window is not None:
        row["orthogonal_position_window_profile"] = orthogonal_position_window
    if velocity_window is not None:
        row["aligned_velocity_window_profile"] = velocity_window
    if orthogonal_velocity_window is not None:
        row["orthogonal_velocity_window_profile"] = orthogonal_velocity_window
    return row


def _family_summary() -> dict[str, object]:
    return {
        "relative_time_steps": [-1, 0, 1],
        "aligned_output_window_profile_mean": [0.0, 1.0, 0.5],
        "aligned_output_window_profile_sem": [0.0, 0.1, 0.05],
        "orthogonal_output_window_profile_mean": [0.0, 0.25, 0.1],
        "orthogonal_output_window_profile_sem": [0.0, 0.02, 0.01],
        "aligned_position_window_profile_mean": [0.0, 0.1, 0.2],
        "aligned_position_window_profile_sem": [0.0, 0.01, 0.02],
        "orthogonal_position_window_profile_mean": [0.0, -0.1, -0.2],
        "orthogonal_position_window_profile_sem": [0.0, 0.01, 0.02],
        "aligned_velocity_window_profile_mean": [0.0, 0.2, 0.1],
        "aligned_velocity_window_profile_sem": [0.0, 0.02, 0.01],
        "orthogonal_velocity_window_profile_mean": [0.0, -0.2, -0.1],
        "orthogonal_velocity_window_profile_sem": [0.0, 0.02, 0.01],
    }


def _rollout_evaluation(
    *,
    command_delta: list[list[list[list[float]]]] | None = None,
    position_delta: list[list[list[list[float]]]] | None = None,
    velocity_delta: list[list[list[list[float]]]] | None = None,
) -> RolloutEvaluation:
    zeros_2d = np.zeros((1, 1, 3, 2), dtype=float)
    zeros_hidden = np.zeros((1, 1, 3, 4), dtype=float)
    command = np.asarray(command_delta, dtype=float) if command_delta is not None else zeros_2d
    position = np.asarray(position_delta, dtype=float) if position_delta is not None else zeros_2d
    velocity = np.asarray(velocity_delta, dtype=float) if velocity_delta is not None else zeros_2d
    return RolloutEvaluation(
        position=position,
        velocity=velocity,
        command=command,
        hidden=zeros_hidden,
        gru_input=zeros_hidden,
        initial_position=np.zeros((1, 2), dtype=float),
        initial_velocity=np.zeros((1, 2), dtype=float),
        target_position=np.zeros(2, dtype=float),
        dt=0.01,
    )
