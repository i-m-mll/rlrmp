"""Contract tests for d01c35a linear equivalence certificate helpers."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.math.cs_game_card import PRIMARY_GAMMA_FACTOR, materialize_reference
from rlrmp.analysis.math.linear_equivalence_certificate import (
    CertificateConfig,
    policy_evaluation_matrices,
    result_summary,
    run_linear_equivalence_certificate,
)
from rlrmp.analysis.math.linear_round_trip import LinearOptimizationConfig


def test_policy_evaluation_matches_lqr_value_matrices() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))

    P = policy_evaluation_matrices(
        reference.plant,
        reference.schedule,
        reference.lqr_solution.K,
    )

    assert P.shape == reference.lqr_solution.P.shape
    assert jnp.allclose(P, reference.lqr_solution.P, rtol=1e-8, atol=1e-8)


def test_certificate_smoke_reports_adam_and_quasi_newton() -> None:
    result = run_linear_equivalence_certificate(
        config=CertificateConfig(n_validation_random_states=2),
        training_config=LinearOptimizationConfig(n_steps=3, n_random_states=2),
        quasi_newton_config=LinearOptimizationConfig(n_steps=1, n_random_states=2),
        heldout_step_sweep=(0,),
        heldout_restarts=1,
    )
    summary = result_summary(result)

    assert summary["issue"] == "d01c35a"
    assert summary["phase3_issue"] == "6f5c79e"
    assert summary["overall_status"] in {
        "blocked_not_disturbance_equivalent",
        "passed",
    }
    assert {controller["label"] for controller in summary["controllers"]} == {
        "adam_lqr_fit",
        "lbfgsb_after_adam_lqr_fit",
    }
    for controller in summary["controllers"]:
        assert controller["distribution_metrics"]
        assert controller["reference_gradient_norm"] < 1e-6
