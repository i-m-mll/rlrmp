"""Tests for GRU post-hoc evaluation diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
from jax import Array as JaxArray
import numpy as np

from rlrmp.analysis.pipelines._selected_eval_rollouts import SelectedEvalRolloutProduct
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
    RolloutEvaluation,
    compute_gru_gate_arrays,
    summarize_controller_feedback_scales,
    summarize_rollout_behavior,
)


def test_jax_rollout_product_materializes_legacy_rollout_with_parity() -> None:
    states = SimpleNamespace(
        mechanics=SimpleNamespace(
            effector=SimpleNamespace(
                pos=jnp.asarray([[[[1.0, 2.0]]]], dtype=jnp.float64),
                vel=jnp.asarray([[[[0.1, 0.2]]]], dtype=jnp.float64),
            ),
            vector=jnp.ones((1, 1, 1, 4), dtype=jnp.float64),
        ),
        net=SimpleNamespace(
            output=jnp.asarray([[[[0.3, 0.4]]]], dtype=jnp.float64),
            hidden=jnp.asarray([[[[0.5, 0.6]]]], dtype=jnp.float64),
            input=jnp.asarray([[[[0.7, 0.8]]]], dtype=jnp.float64),
        ),
        sensory=SimpleNamespace(output=jnp.asarray([[[[0.9, 1.0]]]], dtype=jnp.float64)),
    )
    trial_specs = SimpleNamespace(
        inits={"effector": SimpleNamespace(pos=jnp.asarray([[0.0, 0.0]]), vel=jnp.asarray([[0.0, 0.0]]))},
        inputs={
            "effector_target": SimpleNamespace(pos=jnp.asarray([[[1.0, 1.0]]], dtype=jnp.float64))
        },
    )

    product = SelectedEvalRolloutProduct.from_states(
        states,
        trial_specs,
        dt=0.01,
        include_mechanics_vector=True,
        include_feedback=True,
    )
    legacy = product.to_rollout_evaluation(RolloutEvaluation)

    assert isinstance(product.position, JaxArray)
    assert isinstance(product.feedback, JaxArray)
    assert isinstance(product.mechanics_vector, JaxArray)
    assert isinstance(legacy.position, np.ndarray)
    np.testing.assert_allclose(legacy.position, np.asarray(product.position))
    np.testing.assert_allclose(legacy.command, np.asarray(product.command))
    np.testing.assert_allclose(legacy.mechanics_vector, np.asarray(product.mechanics_vector))


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


def test_summarize_rollout_behavior_accepts_jax_backed_rollout_with_numpy_parity() -> None:
    numpy_evaluation = RolloutEvaluation(
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
    jax_evaluation = RolloutEvaluation(
        **{
            field: jnp.asarray(getattr(numpy_evaluation, field), dtype=jnp.float64)
            for field in (
                "position",
                "velocity",
                "command",
                "hidden",
                "gru_input",
                "initial_position",
                "initial_velocity",
                "target_position",
            )
        },
        dt=numpy_evaluation.dt,
    )

    assert summarize_rollout_behavior(jax_evaluation) == summarize_rollout_behavior(
        numpy_evaluation
    )



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

    assert hasattr(gates["reset"], "block_until_ready")
    assert gates["reset"].shape == (1, 1, 2, 1)
    assert gates["update"].shape == (1, 1, 2, 1)
    assert gates["candidate"].shape == (1, 1, 2, 1)
    assert np.all((gates["reset"] >= 0.0) & (gates["reset"] <= 1.0))
    assert np.all((gates["update"] >= 0.0) & (gates["update"] <= 1.0))


def test_summarize_controller_feedback_scales_uses_trailing_feedback_channels() -> None:
    gru_input = np.zeros((1, 2, 3, 7), dtype=np.float64)
    gru_input[..., -6:] = np.array(
        [
            [
                [1.0, 0.0, 3.0, 4.0, 5.0, 12.0],
                [2.0, 0.0, 0.0, 6.0, 8.0, 15.0],
                [4.0, 3.0, 5.0, 12.0, 7.0, 24.0],
            ],
            [
                [0.0, 1.0, 8.0, 15.0, 9.0, 40.0],
                [0.0, 2.0, 7.0, 24.0, 11.0, 60.0],
                [3.0, 4.0, 20.0, 21.0, 13.0, 84.0],
            ],
        ]
    )
    evaluation = RolloutEvaluation(
        position=np.zeros((1, 2, 3, 2)),
        velocity=np.zeros((1, 2, 3, 2)),
        command=np.zeros((1, 2, 3, 2)),
        hidden=np.zeros((1, 2, 3, 2)),
        gru_input=gru_input,
        initial_position=np.zeros((2, 2)),
        initial_velocity=np.zeros((2, 2)),
        target_position=np.zeros((2, 3, 2)),
        dt=0.01,
    )

    summary = summarize_controller_feedback_scales(
        evaluation,
        run_id="run_a",
        checkpoint_policy="validation_selected_per_replicate",
    )

    assert summary["status"] == "available"
    assert summary["run_id"] == "run_a"
    assert summary["feedback_dim"] == 6
    assert summary["feedback_start_index"] == 1
    assert summary["feedback_basis"] == "target_relative_delayed_feedback_plus_force_filter"
    assert summary["components"]["position"]["gru_input_indices"] == [1, 2]
    assert summary["components"]["velocity"]["gru_input_indices"] == [3, 4]
    assert summary["components"]["force_filter"]["gru_input_indices"] == [5, 6]
    np.testing.assert_allclose(
        summary["components"]["force_filter"]["reference_scale"],
        np.quantile(np.linalg.norm(gru_input[..., -2:], axis=-1).reshape(-1), 0.95),
    )


def test_controller_feedback_scales_accepts_jax_inputs() -> None:
    gru_input = jnp.zeros((1, 1, 2, 4), dtype=jnp.float64)
    gru_input = gru_input.at[..., -4:].set(
        jnp.asarray([[[[3.0, 4.0, 5.0, 12.0], [6.0, 8.0, 8.0, 15.0]]]])
    )
    evaluation = RolloutEvaluation(
        position=np.zeros((1, 1, 2, 2)),
        velocity=np.zeros((1, 1, 2, 2)),
        command=np.zeros((1, 1, 2, 2)),
        hidden=np.zeros((1, 1, 2, 2)),
        gru_input=gru_input,
        initial_position=np.zeros((1, 2)),
        initial_velocity=np.zeros((1, 2)),
        target_position=np.zeros((1, 2, 2)),
        dt=0.01,
    )

    summary = summarize_controller_feedback_scales(evaluation)

    assert summary["status"] == "available"
    assert summary["feedback_basis"] == "target_relative_delayed_feedback"
    np.testing.assert_allclose(summary["components"]["position"]["p95_norm"], 9.75)
