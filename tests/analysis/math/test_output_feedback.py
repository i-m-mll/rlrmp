"""Contract tests for the C&S output-feedback estimator lane."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    PRIMARY_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.hinf_riccati import simulate_closed_loop
from rlrmp.analysis.math.output_feedback import (
    INIT_POS,
    TARGET_POS,
    OutputFeedbackConfig,
    analyze_output_feedback_gamma_sweep,
    delayed_observation_matrix,
    exact_output_feedback_adversary_audit,
    gamma_sweep_summary,
    kalman_estimator_joint_matrices,
    make_cs_output_feedback_initial_state,
    measurement_covariance,
    output_feedback_cost,
    output_feedback_lqr_bellman_objective,
    position_velocity_observation_config,
    robust_estimator_covariances,
    robust_output_feedback_feasibility_diagnostics,
    robust_estimator_fixed_adversary_policy,
    robust_estimator_joint_matrices,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator,
    rollout_with_robust_estimator_policy,
    train_output_feedback_lqr_bellman_controller,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig, ensemble_initial_states


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


def test_position_velocity_observation_matrix_selects_oldest_4d_feedback_block() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    config = position_velocity_observation_config(reference.plant)
    H = delayed_observation_matrix(reference.plant, config)
    R_obs = measurement_covariance(reference.plant, config)

    assert config.observed_physical_indices == (0, 1, 2, 3)
    assert H.shape == (4, 48)
    assert R_obs.shape == (4, 4)
    assert jnp.allclose(H[:, :40], 0.0)
    assert jnp.allclose(H[:, 40:44], jnp.eye(4))
    assert jnp.allclose(H[:, 44:48], 0.0)


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


def test_kalman_joint_matrices_reproduce_rollout_for_nonzero_epsilon() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    epsilon = jnp.ones((reference.schedule.T, reference.plant.m_w), dtype=jnp.float64) * 1e-5
    rollout = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        epsilon,
    )
    A_joint, G_joint = kalman_estimator_joint_matrices(
        reference.plant,
        reference.lqr_solution.K,
    )
    z = jnp.concatenate([x0, x0], axis=0)
    zs = [z]
    for t in range(reference.schedule.T):
        z = A_joint[t] @ z + G_joint @ epsilon[t]
        zs.append(z)
    z_stack = jnp.stack(zs, axis=0)

    assert jnp.allclose(z_stack[:, : reference.plant.n], rollout.x)
    assert jnp.allclose(z_stack[:, reference.plant.n :], rollout.x_hat)


def test_robust_joint_matrices_reproduce_rollout_for_nonzero_epsilon() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant)
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
    epsilon = jnp.ones((reference.schedule.T, reference.plant.m_w), dtype=jnp.float64) * 1e-5
    rollout = rollout_with_robust_estimator(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        x0,
        epsilon,
        gains=gains,
    )
    A_joint, G_joint = robust_estimator_joint_matrices(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        gains,
        covs,
    )
    z = jnp.concatenate([x0, x0], axis=0)
    zs = [z]
    for t in range(reference.schedule.T):
        z = A_joint[t] @ z + G_joint @ epsilon[t]
        zs.append(z)
    z_stack = jnp.stack(zs, axis=0)

    assert jnp.allclose(z_stack[:, : reference.plant.n], rollout.x)
    assert jnp.allclose(z_stack[:, reference.plant.n :], rollout.x_hat)


def test_exact_output_feedback_audit_matches_rollout_and_uses_budget() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    audit = exact_output_feedback_adversary_audit(
        label="lqr_smoke",
        plant=reference.plant,
        schedule=reference.schedule,
        controller_gains=reference.lqr_solution.K,
        x0=x0,
        budget=1e-8,
        estimator_kind="kalman",
    )
    rollout_cost = output_feedback_cost(reference.schedule, audit["rollout"])

    assert audit["boundary_active"]
    assert jnp.isclose(audit["epsilon_energy"], 1e-8, rtol=1e-8, atol=1e-12)
    assert abs(audit["quadratic_total"] - rollout_cost.total_without_disturbance_penalty) < 1e-6


def test_exact_output_feedback_audit_reports_gamma_penalized_status() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant)
    audit = exact_output_feedback_adversary_audit(
        label="lqr_smoke",
        plant=reference.plant,
        schedule=reference.schedule,
        controller_gains=reference.lqr_solution.K,
        x0=x0,
        budget=1e-8,
        estimator_kind="kalman",
        penalty_gamma=gamma_ref.gamma,
    )

    penalized = audit["gamma_penalized"]
    assert penalized["gamma"] == gamma_ref.gamma
    assert penalized["gamma_squared"] > 0.0
    assert penalized["max_eigenvalue_over_gamma_squared"] > 0.0
    assert penalized["feasible"] != penalized["unbounded"]


def test_robust_output_feedback_feasibility_diagnostics_are_finite() -> None:
    reference = materialize_reference(gamma_factors=(1.5,))
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
    diagnostics = robust_output_feedback_feasibility_diagnostics(
        reference.plant,
        reference.schedule,
        gamma_ref.solution,
        gains,
        covs,
    )

    assert jnp.isfinite(jnp.asarray(diagnostics["estimator_precision_min_eig"]))
    assert jnp.isfinite(jnp.asarray(diagnostics["gain_correction_min_eig"]))
    assert jnp.isfinite(jnp.asarray(diagnostics["fixed_policy_lhs_min_eig"]))


def test_output_feedback_gamma_sweep_smoke() -> None:
    sweep = analyze_output_feedback_gamma_sweep(gamma_factors=(1.5,))
    summary = gamma_sweep_summary(sweep)
    row = summary["rows"][0]

    assert "arrays" not in summary
    assert row["status"] == "ok"
    assert row["gamma_factor"] == 1.5
    assert "robust_lambda_over_gamma_squared" in row
    assert row["exact_fixed_controller_audits"][0]["gamma_penalized"]["gamma"] == row["gamma"]


def test_output_feedback_default_gamma_is_sweep_selected() -> None:
    from rlrmp.analysis.math.output_feedback import (  # local import keeps smoke tests cheap.
        analyze_phase0b_output_feedback,
    )

    result = analyze_phase0b_output_feedback()

    assert result["gamma_ref"].factor == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR


def test_output_feedback_bellman_objective_prefers_reference_to_zero() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    K_ref = reference.lqr_solution.K
    states, weights = ensemble_initial_states(
        reference.plant,
        LinearTrainingConfig(n_random_states=4),
    )
    reference_objective = output_feedback_lqr_bellman_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.P[1:],
        K_ref,
        states,
        weights,
    )
    zero_objective = output_feedback_lqr_bellman_objective(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.P[1:],
        jnp.zeros_like(K_ref),
        states,
        weights,
    )

    assert reference_objective < zero_objective


def test_output_feedback_bellman_training_recovers_lqr_gain_smoke() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    result = train_output_feedback_lqr_bellman_controller(reference)

    assert result.label == "of_bellman_lbfgsb_lqr_fit"
    assert result.best_objective / result.reference_objective < 1.000001
    assert result.gain_relative_error < 1e-3
