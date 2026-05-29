"""Tests for robust Bellman diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.linear_round_trip import LinearTrainingConfig, ensemble_initial_states
from rlrmp.analysis.robust_bellman import (
    deterministic_inner_max_margin,
    deterministic_robust_bellman_objective,
    train_deterministic_numerical_minmax_bellman,
    train_output_feedback_information_state_numerical_minmax_bellman,
    train_output_feedback_joint_robust_bellman,
    train_deterministic_robust_bellman,
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


def test_output_feedback_joint_robust_bellman_training_recovers_reference_smoke() -> None:
    reference = materialize_reference(gamma_factors=(1.4,))
    result = train_output_feedback_joint_robust_bellman(
        reference,
        gamma_factor=1.4,
        config=LinearTrainingConfig(n_steps=160, n_random_states=8),
    )

    assert result["objective_ratio_to_reference"] < 1.000001
    assert result["gain_relative_error"] > 0.1
