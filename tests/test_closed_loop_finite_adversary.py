"""Tests for finite closed-loop adversary policy primitives."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rlrmp.train.closed_loop_finite_adversary import (
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    FiniteAffinePolicy,
    FiniteLinearNoBiasPolicy,
    finite_policy_contract,
    target_centered_full_state_features,
)


def test_linear_no_bias_policy_is_zero_at_zero_centered_features() -> None:
    gains = jnp.ones((3, 2, 4), dtype=jnp.float32)
    policy = FiniteLinearNoBiasPolicy(gains)
    features = jnp.zeros((5, 3, 4), dtype=jnp.float32)

    epsilon = policy(features)

    np.testing.assert_allclose(epsilon, 0.0)
    assert policy.metadata.policy_class == LINEAR_NO_BIAS_POLICY
    assert policy.metadata.shared_across_trials_in_batch is True
    assert policy.metadata.has_bias is False
    assert finite_policy_contract(policy.metadata)["metadata"]["zero_feature_behavior"] == (
        "zero_epsilon"
    )


def test_affine_policy_can_emit_nonzero_bias_at_zero_features() -> None:
    gains = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    bias = jnp.asarray([[0.1, -0.2, 0.3], [0.0, 0.4, -0.5]], dtype=jnp.float32)
    policy = FiniteAffinePolicy(gains, bias)
    features = jnp.zeros((7, 2, 4), dtype=jnp.float32)

    epsilon = policy(features)

    np.testing.assert_allclose(epsilon, np.broadcast_to(np.asarray(bias), (7, 2, 3)))
    assert policy.metadata.policy_class == AFFINE_POLICY
    assert policy.metadata.has_bias is True
    assert finite_policy_contract(policy.metadata)["metadata"]["zero_feature_behavior"] == (
        "bias_epsilon"
    )


def test_finite_policy_uses_live_features_at_each_rollout_time() -> None:
    gains = jnp.zeros((3, 1, 2), dtype=jnp.float32)
    gains = gains.at[0, 0, 0].set(1.0)
    gains = gains.at[1, 0, 0].set(2.0)
    gains = gains.at[2, 0, 0].set(3.0)
    policy = FiniteLinearNoBiasPolicy(gains)
    clean_like_features = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    live_features = clean_like_features.at[0, 1, 0].set(5.0)
    live_features = live_features.at[1, 2, 0].set(-2.0)

    clean_epsilon = policy(clean_like_features)
    live_epsilon = policy(live_features)

    np.testing.assert_allclose(clean_epsilon, 0.0)
    np.testing.assert_allclose(np.asarray(live_epsilon[0, :, 0]), [0.0, 10.0, 0.0])
    np.testing.assert_allclose(np.asarray(live_epsilon[1, :, 0]), [0.0, 0.0, -6.0])
    assert policy.metadata.live_feature_source == "live_perturbed_rollout_state"


def test_target_centered_full_state_features_zero_at_target_centered_state() -> None:
    target = jnp.asarray([0.15, -0.05], dtype=jnp.float32)
    mechanics = jnp.zeros((4, 2, 16), dtype=jnp.float32)
    mechanics = mechanics.at[..., 0:2].set(target)
    mechanics = mechanics.at[..., 8:10].set(target)

    features = target_centered_full_state_features(mechanics, target_position=target)

    np.testing.assert_allclose(features, 0.0)


def test_target_centered_full_state_features_supports_no_integrator_blocks() -> None:
    target = jnp.asarray([0.12, 0.03], dtype=jnp.float32)
    mechanics = jnp.zeros((2, 4, 12), dtype=jnp.float32)
    mechanics = mechanics.at[..., 0:2].set(target)
    mechanics = mechanics.at[..., 6:8].set(target)
    mechanics = mechanics.at[..., 2].set(1.5)
    mechanics = mechanics.at[..., 8].set(-2.0)

    features = target_centered_full_state_features(mechanics, target_position=target)

    np.testing.assert_allclose(np.asarray(features[..., 0:2]), 0.0)
    np.testing.assert_allclose(np.asarray(features[..., 6:8]), 0.0)
    np.testing.assert_allclose(np.asarray(features[..., 2]), 1.5)
    np.testing.assert_allclose(np.asarray(features[..., 8]), -2.0)


def test_target_centered_full_state_features_broadcasts_batched_targets() -> None:
    target = jnp.asarray([[0.12, 0.03], [0.05, -0.02]], dtype=jnp.float32)
    mechanics = jnp.zeros((2, 4, 12), dtype=jnp.float32)
    mechanics = mechanics.at[0, ..., 0:2].set(target[0])
    mechanics = mechanics.at[0, ..., 6:8].set(target[0])
    mechanics = mechanics.at[1, ..., 0:2].set(target[1])
    mechanics = mechanics.at[1, ..., 6:8].set(target[1])

    features = target_centered_full_state_features(mechanics, target_position=target)

    np.testing.assert_allclose(np.asarray(features[..., 0:2]), 0.0)
    np.testing.assert_allclose(np.asarray(features[..., 6:8]), 0.0)
