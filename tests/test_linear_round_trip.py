"""Contract tests for Phase 3 linear round-trip helpers."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import PRIMARY_GAMMA_FACTOR, materialize_reference
from rlrmp.analysis.linear_round_trip import (
    LinearTrainingConfig,
    ensemble_clean_objective,
    ensemble_initial_states,
    result_summary,
    rollout_task_cost,
    run_phase3_linear_round_trip,
    train_lqr_quasi_newton_controller,
)


def test_ensemble_initial_states_are_full_rank_and_overweight_reach() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    config = LinearTrainingConfig(n_random_states=2)

    states, weights = ensemble_initial_states(reference.plant, config)

    assert states.shape == (1 + 2 * reference.plant.n + 2, reference.plant.n)
    assert weights[0] == config.reach_weight
    assert jnp.linalg.matrix_rank(states[1 : 1 + reference.plant.n]) == reference.plant.n


def test_reference_lqr_beats_zero_gain_on_full_rank_training_objective() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    config = LinearTrainingConfig(n_random_states=4)
    states, weights = ensemble_initial_states(reference.plant, config)

    reference_objective = ensemble_clean_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.K,
        states,
        weights,
    )
    zero_objective = ensemble_clean_objective(
        reference.plant,
        reference.schedule,
        jnp.zeros_like(reference.lqr_solution.K),
        states,
        weights,
    )

    assert reference_objective < zero_objective


def test_rollout_task_cost_matches_reference_lqr_cost() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))

    cost = rollout_task_cost(
        reference.schedule,
        reference.lqr_rollout.x,
        reference.lqr_rollout.u,
    )

    assert cost == reference.lqr_cost.total_without_disturbance_penalty


def test_phase3_smoke_reports_optimizer_status() -> None:
    result = run_phase3_linear_round_trip(
        training_config=LinearTrainingConfig(n_steps=3, n_random_states=2),
        quasi_newton_config=LinearTrainingConfig(n_steps=1, n_random_states=2),
        heldout_step_sweep=(0,),
        heldout_restarts=1,
    )
    summary = result_summary(result)

    assert summary["issue"] == "6f5c79e"
    assert summary["graphspec_execution_conversion_out_of_scope"] is True
    assert summary["matrix_generalization_out_of_scope"] is True
    assert summary["phase3_status"] in {
        "passed",
        "blocked_on_gain_recovery",
        "blocked_on_optimizer",
    }
    assert summary["best_objective_training"] in {
        "adam_lqr_fit",
        "lbfgsb_after_adam_lqr_fit",
    }
    assert summary["teacher_fit_status"] in {"passed", "failed"}
    assert set(summary["objective_trainings"]) == {
        "adam_lqr_fit",
        "lbfgsb_after_adam_lqr_fit",
    }
    assert {audit["label"] for audit in summary["audits"]} == {
        "analytical_lqr_reference",
        "adam_lqr_fit",
        "lbfgsb_after_adam_lqr_fit",
        "teacher_lqr_fit",
        "analytical_hinf_reference",
        "teacher_hinf_fit",
    }


def test_quasi_newton_smoke_runs_same_objective_contract() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))

    result = train_lqr_quasi_newton_controller(
        reference,
        LinearTrainingConfig(n_steps=1, n_random_states=2),
    )

    assert result.label == "lbfgsb_lqr_fit"
    assert result.zero_objective > result.reference_objective
    assert result.best_objective <= result.zero_objective
    assert result.n_function_evaluations >= 1
