"""Tests for the steady-state GRU perturbation bank."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
    aggregate_family_profiles,
    build_response_figure,
    default_feedback_perturbations,
    identity_condition,
    make_steady_state_trial_specs,
    pad_feedback_offset_inputs,
    right_handed_orthogonal_direction,
    signed_pair_antisymmetry,
)


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
            velocity_window=[0.0, 0.3, 0.1],
        ),
        _row(
            "velocity",
            (-1.0, 0.0),
            [1.0, 0.5],
            relative_steps=[-1, 0, 1],
            output_window=[0.0, 1.2, 0.7],
            orthogonal_window=[0.0, 0.3, 0.4],
            position_window=[0.0, 0.2, 0.4],
            velocity_window=[0.0, 0.4, 0.2],
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
    np.testing.assert_allclose(summary["aligned_velocity_window_profile_mean"], [0.0, 0.35, 0.15])


def test_response_figure_adds_lower_emphasis_orthogonal_output_traces() -> None:
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
    orthogonal_trace = next(trace for trace in figure.data if trace.name == "A orthogonal")
    assert orthogonal_trace.line.dash == "dot"
    assert "0.6" in orthogonal_trace.line.color


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
    velocity_window: list[float] | None = None,
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
    if velocity_window is not None:
        row["aligned_velocity_window_profile"] = velocity_window
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
        "aligned_velocity_window_profile_mean": [0.0, 0.2, 0.1],
        "aligned_velocity_window_profile_sem": [0.0, 0.02, 0.01],
    }
