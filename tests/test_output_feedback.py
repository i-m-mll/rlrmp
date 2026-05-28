"""Contract tests for the C&S output-feedback estimator lane."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import PRIMARY_GAMMA_FACTOR, materialize_reference
from rlrmp.analysis.hinf_riccati import simulate_closed_loop
from rlrmp.analysis.output_feedback import (
    INIT_POS,
    TARGET_POS,
    OutputFeedbackConfig,
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator,
    rollout_with_robust_estimator_policy,
)


def test_output_feedback_initial_history_matches_cs_kron() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    H = delayed_observation_matrix(reference.plant)
    expected_phys = jnp.zeros((8,), dtype=jnp.float64).at[:2].set(INIT_POS - TARGET_POS)

    assert x0.shape == (48,)
    assert jnp.allclose(x0.reshape(6, 8), expected_phys[None, :])
    assert jnp.allclose(H @ x0, expected_phys)


def test_delayed_observation_matrix_selects_last_physical_block() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    H = delayed_observation_matrix(reference.plant)

    assert H.shape == (8, 48)
    assert jnp.allclose(H[:, :40], 0.0)
    assert jnp.allclose(H[:, 40:48], jnp.eye(8))


def test_kalman_estimator_clean_lqr_matches_true_state_rollout() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    true_state = simulate_closed_loop(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        target_pos=TARGET_POS,
    )
    output_feedback = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
    )

    assert jnp.allclose(output_feedback.x, true_state.x, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(output_feedback.x_hat, output_feedback.x, rtol=1e-10, atol=1e-10)


def test_robust_estimator_defaults_to_matlab_compatible_indexing() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant)

    matlab_compatible = rollout_with_robust_estimator(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
    )
    corrected = rollout_with_robust_estimator(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        config=OutputFeedbackConfig(use_matlab_persistent_m_index=False),
    )

    assert matlab_compatible.peak_forward_velocity < 1.0
    assert jnp.sum(corrected.u**2) > 10.0 * jnp.sum(matlab_compatible.u**2)


def test_joint_estimator_adversary_policy_is_finite_and_uses_two_state_blocks() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    covs = robust_estimator_covariances(
        reference.plant,
        reference.schedule,
        gamma_ref.gamma,
    )
    gains = robust_output_feedback_gains(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        covs,
    )
    policy = robust_estimator_fixed_adversary_policy(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        gains,
        covs,
    )
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    rollout = rollout_with_robust_estimator_policy(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        policy,
        gains=gains,
    )

    assert policy.shape == (reference.schedule.T, reference.plant.m_w, 2 * reference.plant.n)
    assert jnp.all(jnp.isfinite(policy))
    assert rollout.epsilon.shape == (reference.schedule.T, reference.plant.m_w)
    assert float(jnp.sum(rollout.epsilon**2)) > 0.0
