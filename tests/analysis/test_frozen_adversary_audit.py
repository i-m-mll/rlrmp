"""Tests for frozen finite-adversary audit helpers."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.frozen_adversary_audit import (
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    dense_shared_metric_from_blocks,
    energy_from_per_trial_metric_blocks,
    energy_from_shared_metric_blocks,
    finite_policy_epsilon_from_parameters,
    generalized_curvature_lambda_star,
    gradient_pressure_scale,
    per_trial_linear_energy_metric_blocks,
    pseudoinverse_metric_quadratic_form,
    realized_epsilon_energy,
    shared_policy_energy_metric_blocks,
    shared_policy_parameter_matrix,
    summarize_active_broad_epsilon_optimizer,
)


def test_shared_linear_metric_energy_matches_realized_epsilon() -> None:
    features = jnp.asarray(
        [
            [[1.0, 2.0], [0.5, -1.0]],
            [[-1.0, 0.25], [2.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    gains = jnp.asarray(
        [
            [[0.2, -0.3], [0.1, 0.5]],
            [[-0.4, 0.25], [0.0, 0.2]],
        ],
        dtype=jnp.float32,
    )

    epsilon = finite_policy_epsilon_from_parameters(
        features,
        gains,
        policy_class=LINEAR_NO_BIAS_POLICY,
    )
    metric_blocks = shared_policy_energy_metric_blocks(
        features,
        policy_class=LINEAR_NO_BIAS_POLICY,
    )
    metric_energy = energy_from_shared_metric_blocks(gains, metric_blocks)

    np.testing.assert_allclose(
        np.asarray(metric_energy),
        np.asarray(realized_epsilon_energy(epsilon)),
        rtol=1e-6,
    )


def test_affine_metric_includes_gain_bias_cross_terms() -> None:
    features = jnp.asarray(
        [
            [[1.0, 2.0], [0.5, -1.0]],
            [[-1.0, 0.25], [2.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    gains = jnp.asarray(
        [
            [[0.2, -0.3]],
            [[-0.4, 0.25]],
        ],
        dtype=jnp.float32,
    )
    bias = jnp.asarray([[0.7], [-0.2]], dtype=jnp.float32)

    epsilon = finite_policy_epsilon_from_parameters(
        features,
        gains,
        bias=bias,
        policy_class=AFFINE_POLICY,
    )
    metric_blocks = shared_policy_energy_metric_blocks(features, policy_class=AFFINE_POLICY)
    params = shared_policy_parameter_matrix(gains, bias=bias, policy_class=AFFINE_POLICY)
    metric_energy = energy_from_shared_metric_blocks(params, metric_blocks)

    np.testing.assert_allclose(
        np.asarray(metric_energy),
        np.asarray(realized_epsilon_energy(epsilon)),
        rtol=1e-6,
    )


def test_per_trial_metric_energy_matches_realized_epsilon() -> None:
    features = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 1.0], [1.0, -1.0]],
        ],
        dtype=jnp.float32,
    )
    gains = jnp.asarray(
        [
            [[[0.5, 0.0]], [[0.0, -0.25]]],
            [[[0.1, 0.2]], [[-0.3, 0.4]]],
        ],
        dtype=jnp.float32,
    )

    epsilon = finite_policy_epsilon_from_parameters(
        features,
        gains,
        policy_class="per_trial_linear_no_bias",
    )
    metric_blocks = per_trial_linear_energy_metric_blocks(features)
    metric_energy = energy_from_per_trial_metric_blocks(gains, metric_blocks)

    np.testing.assert_allclose(
        np.asarray(metric_energy),
        np.asarray(realized_epsilon_energy(epsilon)),
        rtol=1e-6,
    )


def test_gradient_pressure_uses_pseudoinverse_for_singular_metric() -> None:
    metric_blocks = jnp.asarray([[[4.0, 0.0], [0.0, 0.0]]], dtype=jnp.float32)
    gradient = jnp.asarray([[[2.0, 99.0]]], dtype=jnp.float32)

    q_summary = pseudoinverse_metric_quadratic_form(gradient, metric_blocks)
    pressure = gradient_pressure_scale(gradient, metric_blocks, radius=0.5)

    assert q_summary.rank == 1
    assert q_summary.nullity == 1
    np.testing.assert_allclose(q_summary.value, 1.0)
    np.testing.assert_allclose(pressure.pressure_scale, 1.0)


def test_generalized_curvature_lambda_star_reports_infinite_null_curvature() -> None:
    hessian = jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    metric = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)

    summary = generalized_curvature_lambda_star(hessian, metric)

    assert summary.status == "infinite"
    assert math.isinf(summary.lambda_star)
    assert summary.rank == 1
    assert summary.nullity == 1


def test_generalized_curvature_lambda_star_matches_dense_metric_support() -> None:
    metric_blocks = jnp.asarray([[[2.0, 0.0], [0.0, 8.0]]], dtype=jnp.float32)
    metric = dense_shared_metric_from_blocks(metric_blocks, epsilon_dim=1)
    hessian = jnp.asarray([[4.0, 0.0], [0.0, 8.0]], dtype=jnp.float32)

    summary = generalized_curvature_lambda_star(hessian, metric)

    assert summary.status == "finite"
    np.testing.assert_allclose(summary.max_generalized_eigenvalue, 2.0)
    np.testing.assert_allclose(summary.lambda_star, 1.0)


def test_active_optimizer_summary_prefers_enabled_broad_pgd_over_disabled_adam() -> None:
    run_spec = {
        "hps": {
            "broad_epsilon_pgd_training": {
                "enabled": True,
                "mode": "broad_full_state_epsilon_pgd_l2",
                "adversary_mechanism": "linear_no_bias",
                "inner_maximizer": {
                    "method": "projected_gradient_ascent",
                    "initialization": "zero",
                    "n_steps": 10,
                    "projection": "per_trial_flattened_time_component_l2_ball",
                    "step_size_fraction_of_l2_radius": 0.25,
                },
                "objective": {"kind": "soft_energy", "lambda": 123.0},
                "safety_cap": {"enabled": True, "l2_radius_15cm": 0.01},
                "mechanism": {"no_fake_open_loop_replay": True},
            },
            "policy_adversary_training": {
                "enabled": False,
                "mode": "disabled",
                "inner_optimizer": {
                    "method": "adam",
                    "learning_rate": 1e-5,
                    "n_ascent_steps_per_controller_step": 8,
                },
            },
        }
    }

    summary = summarize_active_broad_epsilon_optimizer(run_spec)

    assert summary["active_lane"] == "broad_epsilon_pgd_training.inner_maximizer"
    assert summary["active_method"] == "projected_gradient_ascent"
    assert summary["active_n_steps"] == 10
    assert summary["inactive_policy_adam_metadata"]["method"] == "adam"
    assert summary["inactive_policy_adam_metadata"]["enabled"] is False
    assert any("inactive metadata" in warning for warning in summary["warnings"])
