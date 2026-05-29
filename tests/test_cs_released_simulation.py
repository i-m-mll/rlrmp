"""Tests for the C&S released-code stochastic simulation lane."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from rlrmp.analysis.cs_game_card import PRIMARY_GAMMA_FACTOR, materialize_reference
from rlrmp.analysis.cs_released_simulation import (
    CSNoiseCovariances,
    FixedStepPerturbation,
    build_extlqg_comparator_path,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
    simulate_robust_released_forward,
    simulate_shared_noise_lqg_vs_robust,
    zero_forward_noise_draws,
    zero_noise_covariances,
)
from rlrmp.analysis.output_feedback import (
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator,
)


def test_seeded_forward_noise_draws_are_reproducible() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    T = reference.schedule.T
    covariances = CSNoiseCovariances(
        sensory=jnp.eye(8, dtype=jnp.float64) * 1e-6,
        motor=jnp.eye(plant.m_u, dtype=jnp.float64) * 1e-7,
        process=jnp.eye(plant.n, dtype=jnp.float64) * 1e-8,
        signal_dependent_control=jnp.eye(plant.m_u, dtype=jnp.float64) * 0.05,
    )

    draws_a = sample_forward_noise_draws(jr.PRNGKey(0), T=T, covariances=covariances)
    draws_b = sample_forward_noise_draws(jr.PRNGKey(0), T=T, covariances=covariances)
    draws_c = sample_forward_noise_draws(jr.PRNGKey(1), T=T, covariances=covariances)

    assert jnp.allclose(draws_a.sensory, draws_b.sensory)
    assert jnp.allclose(draws_a.motor, draws_b.motor)
    assert jnp.allclose(draws_a.process, draws_b.process)
    assert jnp.allclose(draws_a.signal_dependent, draws_b.signal_dependent)
    assert not jnp.allclose(draws_a.sensory, draws_c.sensory)


def test_lqg_released_zero_noise_matches_deterministic_kalman_rollout() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    x0 = make_cs_output_feedback_initial_state(plant)
    draws = zero_forward_noise_draws(T=reference.schedule.T, plant=plant)
    covariances = zero_noise_covariances(plant)

    stochastic = simulate_lqg_released_forward(
        plant,
        reference.lqr_solution.K,
        x0,
        draws=draws,
        covariances=covariances,
    )
    deterministic = rollout_with_kalman_estimator(
        plant,
        reference.lqr_solution.K,
        x0,
    )

    assert jnp.allclose(stochastic.x, deterministic.x, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.x_hat, deterministic.x_hat, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.y, deterministic.y, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.u_command, deterministic.u, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.u_applied, deterministic.u, rtol=1e-10, atol=1e-10)


def test_robust_released_zero_noise_matches_deterministic_estimator_rollout() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(plant)
    draws = zero_forward_noise_draws(T=reference.schedule.T, plant=plant)
    covariances = zero_noise_covariances(plant)
    estimator_covariances = robust_estimator_covariances(
        plant,
        reference.schedule,
        gamma_ref.gamma,
    )
    gains = robust_output_feedback_gains(
        plant,
        reference.schedule,
        gamma_ref.solution,
        estimator_covariances,
    )

    stochastic = simulate_robust_released_forward(
        plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        draws=draws,
        covariances=covariances,
        gains=gains,
    )
    deterministic = rollout_with_robust_estimator(
        plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        gains=gains,
    )

    assert jnp.allclose(stochastic.x, deterministic.x, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.x_hat, deterministic.x_hat, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.y, deterministic.y, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.u_command, deterministic.u, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(stochastic.u_applied, deterministic.u, rtol=1e-10, atol=1e-10)


def test_shared_noise_comparison_reuses_draws_across_lqg_and_robust_arms() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    x0 = make_cs_output_feedback_initial_state(plant)
    gamma_ref = reference.gamma_references[0]
    covariances = CSNoiseCovariances(
        sensory=jnp.eye(8, dtype=jnp.float64) * 1e-7,
        motor=jnp.eye(plant.m_u, dtype=jnp.float64) * 1e-8,
        process=jnp.eye(plant.n, dtype=jnp.float64) * 1e-9,
        signal_dependent_control=jnp.eye(plant.m_u, dtype=jnp.float64) * 0.02,
    )

    comparison = simulate_shared_noise_lqg_vs_robust(
        jr.PRNGKey(11),
        plant=plant,
        schedule=reference.schedule,
        lqg_gains=reference.lqr_solution.K,
        robust_solution=gamma_ref.solution,
        x0=x0,
        covariances=covariances,
    )

    assert jnp.allclose(comparison.lqg.sensory_noise, comparison.draws.sensory)
    assert jnp.allclose(comparison.robust.sensory_noise, comparison.draws.sensory)
    assert jnp.allclose(comparison.lqg.motor_noise, comparison.robust.motor_noise)
    assert jnp.allclose(
        comparison.lqg.process_noise,
        comparison.robust.process_noise,
    )
    assert comparison.lqg.x.shape == comparison.robust.x.shape == (
        reference.schedule.T + 1,
        plant.n,
    )


def test_fixed_step_perturbation_hook_adds_state_impulse() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    T = reference.schedule.T
    x0 = jnp.zeros((plant.n,), dtype=jnp.float64)
    zero_gains = jnp.zeros_like(reference.lqr_solution.K)
    value = jnp.zeros((plant.n,), dtype=jnp.float64).at[plant.vel_slice[0]].set(0.25)

    baseline = simulate_lqg_released_forward(
        plant,
        zero_gains,
        x0,
        draws=zero_forward_noise_draws(T=T, plant=plant),
        covariances=zero_noise_covariances(plant),
    )
    perturbed = simulate_lqg_released_forward(
        plant,
        zero_gains,
        x0,
        draws=zero_forward_noise_draws(T=T, plant=plant),
        covariances=zero_noise_covariances(plant),
        perturbation=FixedStepPerturbation(step=2, value=value),
    )

    assert jnp.allclose(perturbed.perturbations[2], value)
    assert jnp.allclose(perturbed.x[3] - baseline.x[3], value)


def test_extlqg_comparator_path_tracks_matlab_chain_and_shapes() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    covariances = zero_noise_covariances(plant)
    comparator = build_extlqg_comparator_path(
        plant,
        reference.lqr_solution.K,
        covariances,
    )

    assert comparator.function_chain == ("extLQG", "computeOFC", "computeExtKalman")
    assert comparator.controller_gains.shape == reference.lqr_solution.K.shape
    assert comparator.estimator_gains.shape == (reference.schedule.T, plant.n, 8)
    assert comparator.state_covariances.shape == (reference.schedule.T + 1, plant.n, plant.n)
    assert comparator.noise_covariances is covariances
    assert "full MATLAB fixed-point iteration" in comparator.parity_status
