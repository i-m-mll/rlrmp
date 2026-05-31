"""Contract tests for 87edaae smooth time-basis coverage plumbing."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.output_feedback import OutputFeedbackRollout
from rlrmp.analysis.output_feedback_time_constrained import (
    TimeBasisCondition,
    TimeBasisFit,
    _fit_summary,
    r20_observer_error_state_coverage_conditions,
    r20_state_coverage_conditions,
    r20_state_eigenspectrum_coverage_conditions,
    r12_observer_error_state_coverage_conditions,
    r12_state_coverage_conditions,
    r12_state_eigenspectrum_coverage_conditions,
)


def test_r12_state_coverage_helpers_emit_planned_row_set() -> None:
    eigenspectrum = r12_state_eigenspectrum_coverage_conditions()
    observer_error = r12_observer_error_state_coverage_conditions()
    conditions = r12_state_coverage_conditions()

    assert len(eigenspectrum) == 6
    assert len(observer_error) == 2
    assert conditions == eigenspectrum + observer_error
    assert {
        (condition.eigenspectrum_coverage.n_modes, condition.eigenspectrum_coverage.scale)
        for condition in eigenspectrum
    } == {(1, 0.3), (1, 1.0), (1, 3.0), (4, 0.3), (4, 1.0), (4, 3.0)}
    assert {
        (condition.observer_error_coverage.n_modes, condition.observer_error_coverage.scale)
        for condition in observer_error
    } == {(1, 0.3), (1, 1.0)}
    assert all(condition.rank == 12 for condition in conditions)
    assert all(condition.initialization == "scratch" for condition in conditions)
    assert all(condition.optimizer == "adamw_then_lbfgsb" for condition in conditions)
    assert all(condition.use_whitening for condition in conditions)
    assert all(condition.eigenspectrum_coverage.objective == "state" for condition in eigenspectrum)
    assert all(
        condition.observer_error_coverage.objective == "state" for condition in observer_error
    )
    assert all(condition.eigenspectrum_coverage.weight == 0.1 for condition in eigenspectrum)
    assert all(condition.observer_error_coverage.weight == 0.1 for condition in observer_error)


def test_r20_state_coverage_helpers_emit_focused_row_set() -> None:
    eigenspectrum = r20_state_eigenspectrum_coverage_conditions()
    observer_error = r20_observer_error_state_coverage_conditions()
    conditions = r20_state_coverage_conditions()

    assert len(eigenspectrum) == 2
    assert len(observer_error) == 1
    assert conditions == eigenspectrum + observer_error
    assert {
        (condition.eigenspectrum_coverage.n_modes, condition.eigenspectrum_coverage.scale)
        for condition in eigenspectrum
    } == {(4, 1.0), (4, 3.0)}
    assert {
        (condition.observer_error_coverage.n_modes, condition.observer_error_coverage.scale)
        for condition in observer_error
    } == {(1, 0.3)}
    assert all(condition.rank == 20 for condition in conditions)
    assert all(condition.initialization == "scratch" for condition in conditions)
    assert all(condition.optimizer == "adamw_then_lbfgsb" for condition in conditions)
    assert all(condition.eigenspectrum_coverage.objective == "state" for condition in eigenspectrum)
    assert all(
        condition.observer_error_coverage.objective == "state" for condition in observer_error
    )
    assert all(condition.eigenspectrum_coverage.weight == 0.1 for condition in eigenspectrum)
    assert all(condition.observer_error_coverage.weight == 0.1 for condition in observer_error)


def test_plain_time_basis_condition_keeps_legacy_label() -> None:
    condition = TimeBasisCondition(
        rank=12,
        initialization="scratch",
        optimizer="adamw_then_lbfgsb",
        learning_rate=1e-2,
    )

    assert condition.label == "spline_r12__scratch_adamw_then_lbfgsb_lr_0p01"


def test_fit_summary_serializes_nested_coverage_config() -> None:
    condition = r12_state_eigenspectrum_coverage_conditions(modes=(4,), scales=(3.0,))[0]
    fit = TimeBasisFit(
        condition=condition,
        theta=jnp.zeros((12, 1, 2)),
        K=jnp.zeros((3, 1, 2)),
        objective_initial=2.0,
        objective_final=1.5,
        objective_reference=1.0,
        objective_zero=3.0,
        objective_ratio_to_reference=1.5,
        gain_relative_error=0.2,
        gradient_norm_initial=4.0,
        gradient_norm_final=0.5,
        best_objective=1.4,
        best_checkpoint_iteration=2,
        optimizer_status="ok",
        optimizer_success=True,
        n_iterations=3,
        n_function_evaluations=4,
        clean_rollout=_rollout(),
        clean_cost=1.0,
        clean_action_mismatch_ratio=0.1,
        under_epsilon_rollout=_rollout(),
        under_epsilon_cost=1.1,
        under_epsilon_cost_ratio_to_lqr=1.2,
        under_epsilon_action_mismatch_ratio=0.3,
        exact_l2_cost=1.3,
        exact_l2_cost_ratio_to_lqr=1.4,
        exact_l2_cost_ratio_to_hinf=1.5,
        gamma_penalized_feasible=True,
        gamma_penalized_lambda_over_gamma_squared=0.9,
    )

    summary = _fit_summary(fit)

    assert summary["condition"]["eigenspectrum_coverage"] == {
        "n_modes": 4,
        "scale": 3.0,
        "weight": 0.1,
        "objective": "state",
        "reference": "lqr_exact_budget_l2",
    }
    assert "observer_error_coverage" not in summary["condition"]


def _rollout() -> OutputFeedbackRollout:
    return OutputFeedbackRollout(
        x=jnp.zeros((4, 2)),
        x_hat=jnp.zeros((4, 2)),
        y=jnp.zeros((3, 1)),
        u=jnp.zeros((3, 1)),
        epsilon=jnp.zeros((3, 1)),
        estimator_covariances=jnp.zeros((4, 2, 2)),
        peak_forward_velocity=0.0,
        peak_forward_velocity_idx=0,
        terminal_position_error=0.0,
        control_effort=0.0,
    )
