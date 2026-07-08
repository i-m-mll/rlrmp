"""JAX process-global configuration guards."""

from __future__ import annotations

import logging

import jax

logger = logging.getLogger(__name__)


def require_jax_x64(context: str) -> None:
    """Enable JAX x64 for an explicit analysis or test entry point.

    ``jax_enable_x64`` is process-global. Callers should invoke this only from
    entry-point scope, before creating arrays or compiling JAX functions.
    Importing a module must never call this helper.
    """

    if not jax.config.jax_enable_x64:
        logger.warning("Enabling process-global JAX x64 for %s", context)
        jax.config.update("jax_enable_x64", True)


def assert_jax_x64_disabled(context: str, *, allow_x64: bool = False) -> None:
    """Fail fast when a training process starts after x64 was enabled."""

    if allow_x64 or not jax.config.jax_enable_x64:
        return
    raise RuntimeError(
        f"{context} must start with jax_enable_x64 disabled. "
        "An analysis import or setup step enabled JAX x64 before training; "
        "start a fresh process or pass an explicit allow-x64 option if this "
        "training run deliberately requires float64."
    )
