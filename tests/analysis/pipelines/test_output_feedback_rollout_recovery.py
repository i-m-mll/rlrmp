"""Contract tests for 7a459bb output-feedback rollout recovery helpers."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    PRIMARY_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    OutputFeedbackRollout,
    make_cs_output_feedback_initial_state,
    output_feedback_clean_objective,
    rollout_with_kalman_estimator,
)
from rlrmp.eval.output_feedback_rollout_recovery import (
    EigenspectrumCoverageConfig,
    ObserverErrorCoverageConfig,
    RolloutRecoveryCondition,
    _eigenspectrum_coverage_samples,
    _coverage_state_objective,
    _make_parameter_maps,
    _observer_error_coverage_samples,
    _scale_initial_state_config,
    eigenspectrum_coverage_conditions,
    observer_error_coverage_conditions,
    _state_scales,
    _time_block_scales,
    _training_ensemble,
    adamw_optimizer_whitened,
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
    assert summary["diagnostics"]["gamma_factor"] == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
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


def test_adamw_condition_uses_whitened_full_batch_objective_and_reports_best() -> None:
    condition = adamw_optimizer_whitened(
        label="adamw_smoke",
        learning_rate=1e-3,
        maxiter=2,
        initializations=("scratch",),
    )
    result = run_output_feedback_rollout_recovery(
        conditions=(condition,),
        training_config=LinearTrainingConfig(n_random_states=4),
    )
    summary = result_summary(result)
    fit = summary["fits"][0]

    assert fit["label"] == "adamw_smoke__scratch"
    assert fit["condition"]["optimizer"] == "adamw"
    assert fit["condition"]["use_whitening"] is True
    assert fit["n_iterations"] == 2
    assert fit["n_function_evaluations"] == 2
    assert fit["optimizer_success"] is True
    assert fit["best_objective"] <= fit["objective_initial"]
    assert fit["best_checkpoint_iteration"] is not None
    assert "AdamW completed 2 full-batch steps" in fit["optimizer_status"]
    assert "adamw_smoke__scratch_K" in result.arrays


def test_adamw_polish_condition_reports_both_optimizer_stages() -> None:
    condition = adamw_optimizer_whitened(
        label="adamw_polish_smoke",
        optimizer="adamw_then_lbfgsb",
        learning_rate=1e-3,
        adam_schedule="warmup_cosine",
        adam_clip_norm=10.0,
        maxiter=2,
        polish_maxiter=1,
        initializations=("scratch",),
    )
    result = run_output_feedback_rollout_recovery(
        conditions=(condition,),
        training_config=LinearTrainingConfig(n_random_states=4),
    )
    fit = result_summary(result)["fits"][0]

    assert fit["condition"]["optimizer"] == "adamw_then_lbfgsb"
    assert fit["condition"]["adam_schedule"] == "warmup_cosine"
    assert fit["n_iterations"] >= 2
    assert fit["n_function_evaluations"] >= 2
    assert "AdamW completed 2 full-batch steps" in fit["optimizer_status"]
    assert "L-BFGS-B polish maxiter=1" in fit["optimizer_status"]


def test_eigenspectrum_coverage_samples_are_time_indexed_signed_pairs() -> None:
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    (
        epsilons,
        trajectory_weights,
        coverage_x,
        coverage_xhat,
        times,
        state_weights,
        metadata,
        arrays,
    ) = _eigenspectrum_coverage_samples(
        plant=reference.plant,
        schedule=reference.schedule,
        K_ref=reference.lqr_solution.K,
        x0=x0,
        budget_l2=1.0,
        gamma=gamma_ref.gamma,
        output_config=OutputFeedbackConfig(),
        coverage_config=EigenspectrumCoverageConfig(n_modes=1, scale=0.1, weight=0.05),
    )

    assert metadata["enabled"] is True
    assert metadata["n_trajectories"] == 2
    assert metadata["n_state_samples_for_diagnostics"] == 2 * reference.schedule.T
    assert epsilons.shape == (2, reference.schedule.T, reference.plant.m_w)
    assert trajectory_weights.shape == (2,)
    assert coverage_x.shape == coverage_xhat.shape == (2 * reference.schedule.T, reference.plant.n)
    assert times.shape == state_weights.shape == (2 * reference.schedule.T,)
    assert arrays["coverage_epsilon_modes"].shape == (1, reference.schedule.T, reference.plant.m_w)


def test_observer_error_coverage_samples_are_time_indexed_signed_pairs() -> None:
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    (
        epsilons,
        trajectory_weights,
        coverage_x,
        coverage_xhat,
        times,
        state_weights,
        metadata,
        arrays,
    ) = _observer_error_coverage_samples(
        plant=reference.plant,
        schedule=reference.schedule,
        K_ref=reference.lqr_solution.K,
        x0=x0,
        budget_l2=1.0,
        output_config=OutputFeedbackConfig(),
        coverage_config=ObserverErrorCoverageConfig(n_modes=1, scale=0.1, weight=0.05),
    )

    assert metadata["enabled"] is True
    assert metadata["n_trajectories"] == 2
    assert metadata["n_state_samples_for_diagnostics"] == 2 * reference.schedule.T
    assert epsilons.shape == (2, reference.schedule.T, reference.plant.m_w)
    assert trajectory_weights.shape == (2,)
    assert coverage_x.shape == coverage_xhat.shape == (2 * reference.schedule.T, reference.plant.n)
    assert times.shape == state_weights.shape == (2 * reference.schedule.T,)
    assert arrays["coverage_observer_error_epsilon_modes"].shape == (
        1,
        reference.schedule.T,
        reference.plant.m_w,
    )
    assert arrays["coverage_observer_error"].shape == coverage_x.shape


def test_coverage_state_objective_uses_clean_objective_for_zero_time_overlap() -> None:
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    states = jnp.stack([x0, 0.5 * x0])
    weights = jnp.asarray([2.0, 0.5], dtype=jnp.float64)
    output_config = OutputFeedbackConfig()

    coverage_objective = _coverage_state_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.K,
        states,
        states,
        jnp.zeros((states.shape[0],), dtype=jnp.int32),
        weights,
        output_config,
    )
    normalized_clean_weights = weights / jnp.sum(weights) * states.shape[0]
    clean_objective = output_feedback_clean_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.K,
        states,
        normalized_clean_weights,
        output_config,
    )

    assert jnp.allclose(coverage_objective, clean_objective)


def test_initial_state_scale_sweep_preserves_reach_weight() -> None:
    base = LinearTrainingConfig(basis_scale=0.01, random_state_scale=0.02, reach_weight=10.0)
    scaled = _scale_initial_state_config(base, 0.3)

    assert scaled.basis_scale == 0.003
    assert scaled.random_state_scale == 0.006
    assert scaled.n_random_states == base.n_random_states
    assert scaled.reach_weight == base.reach_weight


def test_eigenspectrum_conditions_use_strong_optimizer_whitening() -> None:
    conditions = eigenspectrum_coverage_conditions(
        objectives=("trajectory", "state"),
        modes=(1,),
        scales=(0.3,),
        weight=0.1,
    )

    assert [condition.eigenspectrum_coverage.objective for condition in conditions] == [
        "trajectory",
        "state",
    ]
    assert all(condition.use_whitening for condition in conditions)
    assert all(condition.maxiter == 2000 for condition in conditions)
    assert all(condition.initializations == ("scratch",) for condition in conditions)


def test_observer_error_conditions_use_strong_optimizer_whitening() -> None:
    conditions = observer_error_coverage_conditions(
        objectives=("trajectory", "state"),
        modes=(1,),
        scales=(0.3,),
        weight=0.1,
    )

    assert [condition.observer_error_coverage.objective for condition in conditions] == [
        "trajectory",
        "state",
    ]
    assert all(condition.use_whitening for condition in conditions)
    assert all(condition.maxiter == 2000 for condition in conditions)
    assert all(condition.initializations == ("scratch",) for condition in conditions)


def test_rollout_recovery_clean_rollout_has_estimator_fields() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    rollout = rollout_with_kalman_estimator(reference.plant, reference.lqr_solution.K, x0)

    assert isinstance(rollout, OutputFeedbackRollout)
    assert rollout.x_hat.shape == rollout.x.shape
    assert rollout.y.shape[0] == reference.schedule.T
    assert rollout.estimator_covariances.shape[0] == reference.schedule.T + 1
