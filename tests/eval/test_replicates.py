"""Semantic tests for replicate-batched evaluation leaves."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.eval.replicates import is_replicate_array


def test_is_replicate_array_requires_matching_leading_axis() -> None:
    assert is_replicate_array(jnp.zeros((3, 2)), 3)
    assert not is_replicate_array(jnp.zeros((2, 3)), 3)


def test_is_replicate_array_rejects_scalars_and_non_arrays() -> None:
    assert not is_replicate_array(jnp.asarray(1.0), 1)
    assert not is_replicate_array([jnp.asarray(1.0)], 1)
