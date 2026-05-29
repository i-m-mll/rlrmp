"""Contract tests for 7a459bb output-feedback rollout recovery helpers."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import PRIMARY_GAMMA_FACTOR, materialize_reference
from rlrmp.analysis.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    OutputFeedbackRollout,
    make_cs_output_feedback_initial_state,
    output_feedback_clean_objective,
    rollout_with_kalman_estimator,
)
from rlrmp.analysis.output_feedback_rollout_recovery import (
    RolloutRecoveryCondition,
    _make_parameter_maps,
    _state_scales,
    _time_block_scales,
    _training_ensemble,
    result_summary,
    run_output_feedback_rollout_recovery,
)


def test_whitened_parameterization_preserves_clean_objective_at_same_gain() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    training_config = LinearTrainingConfig(n_random_states=4)
    output_config = OutputFeedbackConfig()
    states, weights = _training_ensemble(reference.plant, training_config, output_config)
    scales = _state_scales(states, weights)
    condition = RolloutRecoveryCondition(label="whitened", use_whitening=True)
    to_theta, to_K, _theta_ref = _make_parameter_maps(
        condition,
        reference.lqr_solution.K,
        scales,
    )
    recovered = to_K(to_theta(reference.lqr_solution.K))
    original_objective = output_feedback_clean_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.K,
        states,
        weights,
        output_config,
    )
    recovered_objective = output_feedback_clean_objective(
        reference.plant,
        reference.schedule,
        recovered,
        states,
        weights,
        output_config,
    )

    assert jnp.allclose(recovered, reference.lqr_solution.K)
    assert jnp.allclose(original_objective, recovered_objective)


def test_time_block_parameterization_preserves_clean_objective_at_same_gain() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    training_config = LinearTrainingConfig(n_random_states=4)
    output_config = OutputFeedbackConfig()
    states, weights = _training_ensemble(reference.plant, training_config, output_config)
    scales = _state_scales(states, weights)
    time_block_scales = _time_block_scales(
        reference.plant,
        reference.schedule.T,
        states,
        weights,
        output_config,
    )
    condition = RolloutRecoveryCondition(
        label="block_time",
        use_whitening=True,
        use_time_block_preconditioning=True,
    )
    to_theta, to_K, _theta_ref = _make_parameter_maps(
        condition,
        reference.lqr_solution.K,
        scales,
        time_block_scales,
    )
    recovered = to_K(to_theta(reference.lqr_solution.K))
    original_objective = output_feedback_clean_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.K,
        states,
        weights,
        output_config,
    )
    recovered_objective = output_feedback_clean_objective(
        reference.plant,
        reference.schedule,
        recovered,
        states,
        weights,
        output_config,
    )

    assert jnp.allclose(recovered, reference.lqr_solution.K)
    assert jnp.allclose(original_objective, recovered_objective)


def test_rollout_recovery_smoke_emits_scratch_and_bellman_rows() -> None:
    result = run_output_feedback_rollout_recovery(
        conditions=(RolloutRecoveryCondition(label="smoke", maxiter=1),),
        training_config=LinearTrainingConfig(n_random_states=4),
    )
    summary = result_summary(result)
    labels = {row["label"] for row in summary["fits"]}

    assert labels == {"smoke__scratch", "smoke__bellman_init"}
    assert len(result.fits) == 2
    assert all(isinstance(fit.clean_rollout, OutputFeedbackRollout) for fit in result.fits)
    assert all("exact_l2_cost_ratio_to_lqr" in row for row in summary["fits"])


def test_bellman_auxiliary_condition_emits_scratch_only_with_schedule() -> None:
    condition = RolloutRecoveryCondition(
        label="aux_smoke",
        use_whitening=True,
        maxiter=1,
        initializations=("scratch",),
        auxiliary_bellman_weights=(0.1, 0.0),
    )
    result = run_output_feedback_rollout_recovery(
        conditions=(condition,),
        training_config=LinearTrainingConfig(n_random_states=4),
    )
    summary = result_summary(result)

    assert [row["label"] for row in summary["fits"]] == ["aux_smoke__scratch"]
    assert summary["diagnostics"]["bellman_auxiliary"]["schedules"] == {"aux_smoke": (0.1, 0.0)}
    assert "bellman_weight=0.1" in summary["fits"][0]["optimizer_status"]
    assert "bellman_weight=0" in summary["fits"][0]["optimizer_status"]


def test_rollout_recovery_clean_rollout_has_estimator_fields() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    rollout = rollout_with_kalman_estimator(reference.plant, reference.lqr_solution.K, x0)

    assert isinstance(rollout, OutputFeedbackRollout)
    assert rollout.x_hat.shape == rollout.x.shape
    assert rollout.y.shape[0] == reference.schedule.T
    assert rollout.estimator_covariances.shape[0] == reference.schedule.T + 1
