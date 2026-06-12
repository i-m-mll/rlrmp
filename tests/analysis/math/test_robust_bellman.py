"""Tests for robust Bellman diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.math.cs_game_card import materialize_reference
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig, ensemble_initial_states
from rlrmp.analysis.math.robust_bellman import (
    deterministic_inner_max_margin,
    deterministic_robust_bellman_objective,
    flattened_epsilon_penalized_objective,
    information_state_bellman_matrices,
    information_state_exact_inner_bellman_objective,
    information_state_feasibility_margin,
    information_state_persistent_index_bellman_matrices,
    train_deterministic_numerical_minmax_bellman,
    train_deterministic_robust_bellman,
    train_output_feedback_flattened_epsilon_exact_inner,
    train_output_feedback_information_state_exact_inner_bellman,
    train_output_feedback_information_state_exact_inner_persistent_index,
    train_output_feedback_information_state_numerical_minmax_bellman,
    train_output_feedback_joint_robust_bellman,
)
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)


def test_deterministic_robust_bellman_prefers_reference_to_zero() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    gamma_ref = reference.gamma_references[0]
    states, weights = ensemble_initial_states(
        reference.plant,
        LinearTrainingConfig(n_random_states=4),
    )
    ref_objective = deterministic_robust_bellman_objective(
        reference.plant,
        reference.schedule,
        gamma_ref.solution.P[1:],
        gamma_ref.solution.K,
        states,
        weights,
        gamma_ref.gamma,
    )
    zero_objective = deterministic_robust_bellman_objective(
        reference.plant,
        reference.schedule,
        gamma_ref.solution.P[1:],
        jnp.zeros_like(gamma_ref.solution.K),
        states,
        weights,
        gamma_ref.gamma,
    )

    assert (
        deterministic_inner_max_margin(reference.plant, gamma_ref.solution.P[1:], gamma_ref.gamma)
        > 0.0
    )
    assert ref_objective < zero_objective


def test_deterministic_robust_bellman_training_recovers_reference_smoke() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_deterministic_robust_bellman(
        reference,
        gamma_factor=1.4,
        config=LinearTrainingConfig(n_steps=250, n_random_states=16),
    )

    assert result.objective_ratio_to_reference < 1.000001
    assert result.gain_relative_error < 1e-3


def test_deterministic_numerical_minmax_bellman_recovers_reference_smoke() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    fits = train_deterministic_numerical_minmax_bellman(
        reference,
        gamma_factor=1.4,
        time_indices=(10,),
        config=LinearTrainingConfig(n_steps=80, n_random_states=8),
    )

    assert fits[0].objective_ratio_to_reference < 1.000001
    assert fits[0].gain_relative_error < 1e-3


def test_output_feedback_information_state_numerical_minmax_recovers_formal_target() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_output_feedback_information_state_numerical_minmax_bellman(
        reference,
        gamma_factor=1.4,
        time_indices=(10,),
        config=LinearTrainingConfig(n_steps=80, n_random_states=8),
    )

    assert result["target"] == "formal_time_indexed_information_state"
    assert result["recovers_formal_target"]
    assert result["max_gain_relative_error"] < 2e-2
    assert result["min_feasibility_margin"] > 0.0
    assert result["cs_persistent_index_gain_relative_error"] > 1e-2


def test_output_feedback_information_state_exact_inner_recovers_formal_smoke() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_output_feedback_information_state_exact_inner_bellman(
        reference,
        gamma_factor=1.4,
        time_indices=(10,),
        config=LinearTrainingConfig(n_steps=80, n_random_states=8),
    )

    row = result["fits"][0]
    assert result["target"] == "formal_time_indexed_information_state_exact_hidden_state_inner"
    assert row["feasible"]
    assert row["feasibility_margin"] > 0.0
    assert row["objective_ratio_to_reference"] < 1.000001
    assert row["gain_relative_error"] < 2e-2


def test_output_feedback_information_state_persistent_index_recovers_code_target() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_output_feedback_information_state_exact_inner_persistent_index(
        reference,
        gamma_factor=1.4,
        time_indices=(10,),
        config=LinearTrainingConfig(n_steps=80, n_random_states=8),
    )

    row = result["fits"][0]
    assert result["target"] == "cs_code_fidelity_persistent_index_exact_hidden_state_inner"
    assert row["feasible"]
    assert row["feasibility_margin"] > 0.0
    assert row["objective_ratio_to_reference"] < 1.000001
    assert row["gain_error_to_persistent_target"] < 2e-2
    assert row["gain_error_to_formal_target"] > 1e-2


def test_persistent_index_bellman_blocks_preserve_shapes_and_finite_margin() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    gamma_ref = reference.gamma_references[0]
    covs = robust_estimator_covariances(
        reference.plant,
        reference.schedule,
        gamma_ref.gamma,
        OutputFeedbackConfig(use_matlab_persistent_m_index=True),
    )
    L, N, M_u = information_state_persistent_index_bellman_matrices(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
    )

    assert L.shape == reference.schedule.Q.shape
    assert N.shape == (reference.schedule.T, reference.plant.n, reference.plant.m_u)
    assert M_u.shape == reference.schedule.R.shape
    assert information_state_feasibility_margin(L[10], covs[10], gamma_ref.gamma) > 0.0


def test_information_state_exact_inner_reports_margin_failure_condition() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    gamma_ref = reference.gamma_references[0]
    covs = robust_estimator_covariances(
        reference.plant,
        reference.schedule,
        gamma_ref.gamma,
        OutputFeedbackConfig(use_matlab_persistent_m_index=False),
    )
    L, N, M_u = information_state_bellman_matrices(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
    )
    states, weights = ensemble_initial_states(
        reference.plant,
        LinearTrainingConfig(n_random_states=4),
    )
    gains = robust_output_feedback_gains(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        covs,
        OutputFeedbackConfig(use_matlab_persistent_m_index=False),
    )

    assert information_state_feasibility_margin(L[10], covs[10], gamma_ref.gamma) > 0.0
    assert information_state_feasibility_margin(L[10], covs[10], gamma_ref.gamma * 1e-6) < 0.0
    value = information_state_exact_inner_bellman_objective(
        L[10],
        N[10],
        M_u[10],
        covs[10],
        gains[10],
        states,
        weights,
        gamma_ref.gamma,
    )
    assert jnp.isfinite(value)


def test_flattened_epsilon_objective_reports_finite_margin_for_formal_target() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    gamma_ref = reference.gamma_references[0]
    config = OutputFeedbackConfig(use_matlab_persistent_m_index=False)
    covs = robust_estimator_covariances(
        reference.plant,
        reference.schedule,
        gamma_ref.gamma,
        config,
    )
    gains = robust_output_feedback_gains(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        covs,
        config,
    )
    objective, margin, ratio = flattened_epsilon_penalized_objective(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        covs,
        gains,
        make_cs_output_feedback_initial_state(reference.plant, config),
        config,
    )

    assert jnp.isfinite(objective)
    assert margin > 0.0
    assert 0.0 < ratio < 1.0


def test_flattened_epsilon_exact_inner_training_smoke() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_output_feedback_flattened_epsilon_exact_inner(
        reference,
        gamma_factor=1.4,
        config=LinearTrainingConfig(n_steps=1),
    )

    assert result.feasible
    assert result.reference_feasibility_margin > 0.0
    assert result.reference_lambda_over_gamma_squared < 1.0
    assert result.objective_ratio_to_reference is not None


def test_output_feedback_joint_robust_bellman_training_recovers_reference_smoke() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_output_feedback_joint_robust_bellman(
        reference,
        gamma_factor=1.4,
        config=LinearTrainingConfig(n_steps=160, n_random_states=8),
    )

    assert result["objective_ratio_to_reference"] < 1.000001
    assert result["gain_relative_error"] > 0.1
