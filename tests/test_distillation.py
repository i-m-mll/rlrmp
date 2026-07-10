"""Tests for guided distillation losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rlrmp.train.distillation_native.losses import (
    DistillationLossWeights,
    batched_directional_jvps,
    cs_h0_distillation_config,
    guided_distillation_loss,
    input_output_jvp_matching_loss,
)


def test_input_output_jvp_loss_matches_dense_jacobian_on_toy_map() -> None:
    feedback = jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10.0
    actions = jnp.arange(3, dtype=jnp.float32).reshape(3, 1) / 5.0
    n_input = feedback.size + actions.size
    n_output = actions.size
    student_matrix = jnp.arange(n_output * n_input, dtype=jnp.float32).reshape(
        n_output,
        n_input,
    )
    teacher_matrix = student_matrix * 0.5 + 0.25
    feedback_directions = jnp.arange(5 * feedback.size, dtype=jnp.float32).reshape(
        5,
        *feedback.shape,
    )
    action_directions = jnp.arange(5 * actions.size, dtype=jnp.float32).reshape(
        5,
        *actions.shape,
    )

    def student_policy(obs, act):
        x = jnp.concatenate([jnp.ravel(obs), jnp.ravel(act)])
        return (student_matrix @ x).reshape(actions.shape)

    def teacher_policy(obs, act):
        x = jnp.concatenate([jnp.ravel(obs), jnp.ravel(act)])
        return (teacher_matrix @ x).reshape(actions.shape)

    actual = input_output_jvp_matching_loss(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        feedback_history=feedback,
        action_history=actions,
        feedback_directions=feedback_directions,
        action_directions=action_directions,
    )

    dense_input_dirs = jnp.concatenate(
        [
            feedback_directions.reshape(5, -1),
            action_directions.reshape(5, -1),
        ],
        axis=-1,
    )
    dense_jvp_diff = dense_input_dirs @ (student_matrix - teacher_matrix).T
    expected = jnp.mean(jnp.square(dense_jvp_diff.reshape(5, *actions.shape)))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_directional_jvp_loss_is_jittable_and_batches_directions() -> None:
    feedback = jnp.ones((2, 3, 2), dtype=jnp.float32)
    actions = jnp.zeros((2, 3, 1), dtype=jnp.float32)
    feedback_directions = jnp.ones((4, *feedback.shape), dtype=jnp.float32) * 0.2
    action_directions = jnp.ones((4, *actions.shape), dtype=jnp.float32) * 0.1

    def student_policy(obs, act):
        return jnp.tanh(0.3 * obs[..., :1] + 0.2 * act)

    def teacher_policy(obs, act):
        return jnp.tanh(0.1 * obs[..., :1] - 0.4 * act)

    jvps = batched_directional_jvps(
        student_policy,
        feedback,
        actions,
        feedback_directions,
        action_directions,
    )

    assert jvps.shape == (4, *actions.shape)

    loss_fn = jax.jit(
        lambda obs, act, obs_dirs, act_dirs: input_output_jvp_matching_loss(
            student_policy=student_policy,
            teacher_policy=teacher_policy,
            feedback_history=obs,
            action_history=act,
            feedback_directions=obs_dirs,
            action_directions=act_dirs,
        )
    )
    loss = loss_fn(feedback, actions, feedback_directions, action_directions)

    assert jnp.isfinite(loss)
    assert float(loss) > 0.0


def test_guided_loss_runs_through_cs_h0_distillation_surface() -> None:
    feedback = jnp.ones((2, 4, 6), dtype=jnp.float32)
    actions = jnp.zeros((2, 4, 2), dtype=jnp.float32)
    perturbation_feedback = feedback.at[..., 0].add(0.1)
    perturbation_actions = actions.at[..., 1].add(0.05)
    feedback_directions = jnp.ones((3, *feedback.shape), dtype=jnp.float32) * 0.01
    action_directions = jnp.ones((3, *actions.shape), dtype=jnp.float32) * 0.02
    config = cs_h0_distillation_config(
        weights=DistillationLossWeights(
            clean_action=1.0,
            perturbation_response=1.0,
            input_output_jvp=0.5,
            student_forced_rollout_anchor=0.25,
        ),
        n_jvp_directions=3,
    )

    def student_policy(obs, act):
        return 0.3 * obs[..., :2] + 0.2 * act

    def teacher_policy(obs, act):
        return 0.1 * obs[..., :2] - 0.1 * act

    result = guided_distillation_loss(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        feedback_history=feedback,
        action_history=actions,
        config=config,
        perturbation_feedback_history=perturbation_feedback,
        perturbation_action_history=perturbation_actions,
        feedback_directions=feedback_directions,
        action_directions=action_directions,
        student_forced_rollout=jnp.ones((2, 4, 2), dtype=jnp.float32),
        rollout_anchor=jnp.zeros((2, 4, 2), dtype=jnp.float32),
    )

    assert set(result.components) == {
        "clean_action",
        "input_output_jvp",
        "perturbation_response",
        "student_forced_rollout_anchor",
    }
    assert config.summary()["hidden_state_supervision"] is False
    assert (
        config.summary()["feedback_basis"] == "target_relative_delayed_feedback_plus_force_filter"
    )
    assert config.summary()["n_jvp_directions"] == 3
    assert jnp.isfinite(result.total)
    assert float(result.total) > 0.0
