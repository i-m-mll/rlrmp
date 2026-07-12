from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from scripts.prepare_baseline_custody_launch_fork import (
    HISTORY_INDICES,
    _extend_optimizer_histories,
    _validate_launch_fork,
)


def _optimizer(horizon: int) -> tuple[object, ...]:
    values: list[object] = [jnp.asarray(0)] * 33
    for index in HISTORY_INDICES:
        values[index] = jnp.arange(5 * horizon, dtype=jnp.float32).reshape(5, horizon)
    return tuple(values)


def test_launch_fork_extends_histories_but_preserves_source_prefix() -> None:
    source = _optimizer(12_000)
    target = _optimizer(12_200)

    extended = _extend_optimizer_histories(source, target)

    for index in HISTORY_INDICES:
        assert extended[index].shape == (5, 12_200)
        assert bool(jnp.array_equal(extended[index][..., :12_000], source[index]))
        assert bool(jnp.array_equal(extended[index][..., 12_000:], target[index][..., 12_000:]))


def test_launch_fork_rejects_non_source_history_horizon() -> None:
    with pytest.raises(ValueError, match="source horizon is not 12,000"):
        _extend_optimizer_histories(_optimizer(11_999), _optimizer(12_200))


def test_launch_fork_validation_requires_runnable_source_progress() -> None:
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            completed_training_batches=12_200,
            metadata={
                "rlrmp_launch_fork": {
                    "source_completed_batches": 12_000,
                    "target_total_batches": 12_200,
                }
            },
        ),
        slots={"completed_batches": jnp.asarray(12_000), "optimizer": _optimizer(12_200)},
    )

    with pytest.raises(ValueError, match="preserve completed_training_batches=12,000"):
        _validate_launch_fork(loaded)
