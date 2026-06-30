"""Tests for regenerated frozen-policy gate helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rlrmp.analysis.frozen_adversary_audit import (
    dense_shared_metric_from_blocks,
    generalized_curvature_lambda_star,
    support_whitened_generalized_curvature,
)
from rlrmp.analysis.frozen_policy_gate import (
    AFFINE_POLICY,
    DIRECT_EPSILON_MECHANISM,
    metric_geometry_summary,
    selected_epsilon_invariance,
    sha256_json,
    validate_direct_hvp_lambda_source,
)


def test_sha256_json_is_stable_under_key_order() -> None:
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}

    assert sha256_json(left) == sha256_json(right)


def test_selected_epsilon_invariance_records_mean_and_sum_semantics() -> None:
    epsilon = jnp.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 3.0], [0.0, 4.0]],
        ],
        dtype=jnp.float32,
    )

    summary = selected_epsilon_invariance(epsilon)

    assert summary["mean_reduction_original"] == summary["mean_reduction_duplicated"]
    assert summary["mean_reduction_invariant"] is True
    assert summary["sum_reduction_ratio"] == 2.0


def test_direct_epsilon_metric_geometry_uses_batch_mean_metric() -> None:
    gradient = jnp.ones((2, 3, 1), dtype=jnp.float32)

    summary, pressure = metric_geometry_summary(
        DIRECT_EPSILON_MECHANISM,
        features=jnp.zeros((2, 3, 2), dtype=jnp.float32),
        epsilon_dim=1,
        gradient=gradient,
        radius=2.0,
    )

    assert summary["rank"] == 6
    assert summary["nullity"] == 0
    assert summary["condition_number"] == 1.0
    assert pressure > 0.0
    assert summary["used_as_lambda_criterion"] is False
    assert summary["used_as_readiness_criterion"] is False


def test_affine_metric_geometry_reports_singular_shared_policy_metric() -> None:
    features = jnp.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    gradient = jnp.ones((2, 1, 3), dtype=jnp.float32)

    summary, pressure = metric_geometry_summary(
        AFFINE_POLICY,
        features=features,
        epsilon_dim=1,
        gradient=gradient,
        radius=1.0,
    )

    assert summary["metric"] == "shared finite-policy realized-epsilon mean-energy metric"
    assert summary["rank"] < 6
    assert summary["nullity"] > 0
    assert pressure > 0.0
    assert summary["used_as_lambda_criterion"] is False
    assert summary["used_as_readiness_criterion"] is False


def test_direct_hvp_source_validation_rejects_cap_used_criterion() -> None:
    payload = {
        "schema_version": "rlrmp.canonical_soft_lambda_hvp.v1",
        "issue": "06a4dc8",
        "estimator": {
            "method": "per_trial_hvp_lanczos_largest_algebraic",
            "cap_or_interiority_used_as_criterion": True,
        },
        "objective_contract": {
            "safety_cap_role": "provenance only; not used as lambda criterion",
        },
        "pooled_summary": {"lambda_star_p90": 8.0},
        "pooled_beta_mapping": [{"beta": 1.05, "lambda": 10.0}],
    }

    with pytest.raises(ValueError, match="cap/interiority"):
        validate_direct_hvp_lambda_source(payload, beta=1.05)


def test_direct_hvp_source_validation_accepts_provenance_only_cap() -> None:
    payload = {
        "schema_version": "rlrmp.canonical_soft_lambda_hvp.v1",
        "issue": "06a4dc8",
        "estimator": {
            "method": "per_trial_hvp_lanczos_largest_algebraic",
            "cap_or_interiority_used_as_criterion": False,
        },
        "objective_contract": {
            "safety_cap_role": "provenance only; not used as lambda criterion",
        },
        "pooled_summary": {"lambda_star_p90": 8.0},
        "pooled_beta_mapping": [{"beta": 1.05, "lambda": 10.0}],
    }

    source = validate_direct_hvp_lambda_source(payload, beta=1.05)

    assert source["candidate_lambda"] == 10.0
    assert source["lambda_star_p90"] == 8.0
    assert source["launch_basis"] == "fixed_hvp_p90"
    assert source["cap_or_radius_used_as_lambda_criterion"] is False


def test_support_whitened_curvature_matches_dense_generalized_curvature() -> None:
    metric_blocks = jnp.asarray(
        [
            [[2.0, 0.25], [0.25, 1.0]],
            [[1.5, 0.1], [0.1, 0.75]],
        ],
        dtype=jnp.float32,
    )
    epsilon_dim = 2
    dense_metric = dense_shared_metric_from_blocks(metric_blocks, epsilon_dim=epsilon_dim)
    hessian = np.diag(np.linspace(1.0, 3.0, dense_metric.shape[0]))
    hessian = hessian + 0.05 * np.ones_like(hessian)
    dense = generalized_curvature_lambda_star(hessian, dense_metric)
    zero = jnp.zeros((2, epsilon_dim, 2), dtype=jnp.float32)

    def hvp(params):
        flat = params.reshape(-1)
        return (jnp.asarray(hessian) @ flat).reshape(params.shape)

    support = support_whitened_generalized_curvature(
        hvp=hvp,
        zero_params=zero,
        metric_blocks=metric_blocks,
        epsilon_dim=epsilon_dim,
        lanczos_steps=int(dense_metric.shape[0]),
        seed=123,
    )

    assert support.status == "finite"
    assert support.rank == dense.rank
    assert support.nullity == dense.nullity
    assert support.hvp_evaluations == dense.rank
    assert support.lambda_star == pytest.approx(dense.lambda_star, rel=1e-5, abs=1e-5)
