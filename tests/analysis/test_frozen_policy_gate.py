"""Tests for regenerated frozen-policy gate helpers."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.frozen_policy_gate import (
    AFFINE_POLICY,
    DIRECT_EPSILON_MECHANISM,
    metric_geometry_summary,
    selected_epsilon_invariance,
    sha256_json,
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
