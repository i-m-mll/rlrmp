"""Tests for the Phase 1 released stochastic C&S evaluation lane."""

from __future__ import annotations

import json

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import PRIMARY_GAMMA_FACTOR, materialize_reference
from rlrmp.analysis.cs_released_simulation import (
    default_cs_noise_covariances,
    sample_forward_noise_draws,
)
from rlrmp.analysis.cs_stochastic_phase1 import (
    analyze_phase1_stochastic,
    result_summary,
    simulate_full_state_released_forward,
)
from rlrmp.analysis.output_feedback import make_cs_output_feedback_initial_state


def test_phase1_stochastic_shapes_and_shared_noise() -> None:
    result = analyze_phase1_stochastic(seeds=(0, 1))
    trial = result.trials[0]
    T = trial.output_feedback_lqg.x.shape[0] - 1
    n = trial.output_feedback_lqg.x.shape[1]

    assert len(result.trials) == 2
    assert trial.full_state_lqr.x.shape == (T + 1, n)
    assert trial.full_state_hinf.u_applied.shape == trial.output_feedback_hinf.u_applied.shape
    assert jnp.allclose(
        trial.output_feedback_lqg.sensory_noise,
        trial.output_feedback_hinf.sensory_noise,
    )
    assert jnp.allclose(
        trial.output_feedback_lqg.motor_noise,
        trial.output_feedback_hinf.motor_noise,
    )
    assert jnp.allclose(trial.full_state_lqr.motor_noise, trial.output_feedback_lqg.motor_noise)
    assert jnp.allclose(
        trial.full_state_lqr.signal_dependent_standard,
        trial.output_feedback_lqg.signal_dependent_standard,
    )


def test_full_state_stochastic_rollout_is_reproducible_for_fixed_draws() -> None:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    plant = reference.plant
    x0 = make_cs_output_feedback_initial_state(plant)
    covariances = default_cs_noise_covariances(
        plant,
        motor_covariance_scale=1e-10,
        process_covariance_scale=1.0,
        signal_dependent_scale=0.02,
    )
    draws = sample_forward_noise_draws(
        jnp.asarray([0, 123], dtype=jnp.uint32),
        T=reference.schedule.T,
        covariances=covariances,
    )

    rollout_a = simulate_full_state_released_forward(
        plant,
        reference.lqr_solution.K,
        x0,
        draws=draws,
        covariances=covariances,
    )
    rollout_b = simulate_full_state_released_forward(
        plant,
        reference.lqr_solution.K,
        x0,
        draws=draws,
        covariances=covariances,
    )

    assert jnp.allclose(rollout_a.x, rollout_b.x)
    assert jnp.allclose(rollout_a.u_applied, rollout_b.u_applied)


def test_phase1_stochastic_manifest_metadata_has_no_bellman_claim() -> None:
    result = analyze_phase1_stochastic(seeds=(3,))
    summary = result_summary(result)

    assert summary["rerun_metadata"]["lane"] == "released_stochastic"
    assert summary["no_bellman_claim"] is True
    assert "Bellman" in summary["bellman_claim"]
    assert "fixed_point" in summary["extlqg_comparator"]["parity_status"]
    assert summary["arms"]["output_feedback_lqg_extlqg"]["estimator_rms_error"]["mean"] is not None
    assert summary["arms"]["full_state_lqr"]["estimator_rms_error"]["mean"] is None
    assert set(summary["deterministic_certificate_sidecar"]) == {
        "full_state_lqr",
        "full_state_hinf",
        "output_feedback_lqg_extlqg",
        "output_feedback_hinf",
    }
    assert (
        summary["deterministic_certificate_sidecar"]["full_state_hinf"]["lambda_over_gamma_squared"]
        < 1.0
    )
    json.dumps(summary)
