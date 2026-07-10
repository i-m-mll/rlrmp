"""Focused tests for the 3b850d6 closed-loop policy audit helpers."""

from __future__ import annotations
from rlrmp.io import load_named_python_module

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "3b850d6"
    / "scripts"
    / "materialize_closed_loop_policy_audit.py"
)


@pytest.fixture(scope="module")
def audit_module():
    return load_named_python_module('closed_loop_policy_audit_3b850d6', SCRIPT)


def test_linear_no_bias_ridge_recovers_known_mapping(audit_module) -> None:
    features = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    true_weights = jnp.asarray(
        [
            [[2.0, -1.0]],
            [[0.5, 3.0]],
        ],
        dtype=jnp.float32,
    )
    target = jnp.einsum("btf,tef->bte", features, true_weights)
    time_mask = jnp.ones_like(target)

    weights, diagnostics = audit_module.fit_linear_no_bias_policy(
        features=features,
        target_delta=target,
        time_mask=time_mask,
        ridge_alpha=1e-9,
    )

    np.testing.assert_allclose(np.asarray(weights), np.asarray(true_weights), rtol=1e-5, atol=1e-5)
    assert diagnostics["relative_residual_norm"] < 1e-5
    assert diagnostics["rank_min"] == 2


def test_safe_mean_ratio_treats_double_zero_as_one(audit_module) -> None:
    raw = jnp.asarray([0.0, 2.0], dtype=jnp.float32)
    selected = jnp.asarray([0.0, 1.0], dtype=jnp.float32)

    assert audit_module.safe_mean_ratio(raw, selected) == pytest.approx(1.5)
