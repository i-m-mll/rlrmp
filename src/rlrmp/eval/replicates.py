"""Shared predicates for replicate-batched evaluation trees."""

from __future__ import annotations

from typing import Any

import equinox as eqx


def is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    """Return whether ``leaf`` is an array batched over ``n_replicates``."""

    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates
