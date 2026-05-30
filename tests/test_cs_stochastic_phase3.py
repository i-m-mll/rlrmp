"""Tests for the Phase 3 released-code stochastic evaluation lane."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    PRIMARY_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.cs_stochastic_phase3 import (
    Phase3ControllerSpec,
    Phase3StochasticConfig,
    result_summary,
    run_phase3_stochastic_evaluation,
)


def _small_controller_specs() -> tuple[Phase3ControllerSpec, ...]:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    K_ref = reference.lqr_solution.K
    return (
        Phase3ControllerSpec(
            label="analytical_lqr_reference",
            source="analytical_lqr_reference",
            K=K_ref,
            deterministic_gain_relative_error=0.0,
            deterministic_objective_ratio_to_reference=1.0,
        ),
        Phase3ControllerSpec(
            label="scratch_smoke",
            source="deterministic_phase3_scratch_fit",
            K=jnp.zeros_like(K_ref),
            deterministic_gain_relative_error=1.0,
            deterministic_objective_ratio_to_reference=None,
        ),
        Phase3ControllerSpec(
            label="preservation_smoke",
            source="deterministic_phase3_preservation_init_fit",
            K=K_ref,
            deterministic_gain_relative_error=0.0,
            deterministic_objective_ratio_to_reference=1.0,
        ),
    )


def test_phase3_stochastic_evaluation_is_reproducible() -> None:
    config = Phase3StochasticConfig(n_trials=3, seed=17)
    first = run_phase3_stochastic_evaluation(
        config=config,
        controllers=_small_controller_specs(),
    )
    second = run_phase3_stochastic_evaluation(
        config=config,
        controllers=_small_controller_specs(),
    )
    first_summary = result_summary(first)
    second_summary = result_summary(second)

    assert first_summary["evaluations"] == second_summary["evaluations"]
    assert first_summary["evaluations"][0]["action_mismatch_to_reference_mean"] == 0.0
    assert first_summary["evaluations"][1]["action_mismatch_to_reference_mean"] > 0.0


def test_phase3_stochastic_manifest_marks_lane_and_no_bellman_parity_claim() -> None:
    result = run_phase3_stochastic_evaluation(
        config=Phase3StochasticConfig(n_trials=2, seed=3),
        controllers=_small_controller_specs(),
    )
    summary = result_summary(result)

    assert summary["rerun_metadata"]["discretization"] == "euler"
    assert summary["rerun_metadata"]["lane"] == "released_stochastic"
    assert (
        summary["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert summary["claims"]["bellman_stochastic_parity"] is False
    assert "stochastic Bellman objective" in summary["claims"]["note"]
    assert "Bellman parity claim" in summary["non_goals"]


def test_phase3_stochastic_result_reports_required_metrics() -> None:
    result = run_phase3_stochastic_evaluation(
        config=Phase3StochasticConfig(n_trials=2, seed=5),
        controllers=_small_controller_specs(),
    )
    row = result_summary(result)["evaluations"][0]

    for key in (
        "cost_mean",
        "cost_std",
        "cost_ratio_to_reference_mean",
        "peak_forward_velocity_mean",
        "terminal_error_mean",
        "action_mismatch_to_reference_mean",
        "deterministic_exact_l2_cost_ratio_to_lqr",
        "deterministic_lambda_over_gamma_squared",
        "deterministic_gamma_penalized_feasible",
    ):
        assert key in row

    assert row["deterministic_exact_l2_cost_ratio_to_lqr"] > 0.0
    assert row["deterministic_lambda_over_gamma_squared"] > 0.0
