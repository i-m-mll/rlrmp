from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from scripts.prepare_stage2_launch_forks import (
    HISTORY_INDICES,
    SOURCE_COMPLETED_BATCHES,
    TARGET_TOTAL_BATCHES,
    extend_optimizer_histories,
    validate_launch_fork,
)


def _optimizer(horizon: int) -> tuple:
    values = [jnp.zeros((1,), dtype=jnp.float32) for _ in range(33)]
    for index in HISTORY_INDICES:
        values[index] = jnp.arange(horizon, dtype=jnp.float32)[None, :]
    return tuple(values)


def test_extend_optimizer_histories_preserves_prefix_and_target_horizon() -> None:
    source = _optimizer(SOURCE_COMPLETED_BATCHES)
    target = _optimizer(TARGET_TOTAL_BATCHES)
    extended = extend_optimizer_histories(source, target)
    for index in HISTORY_INDICES:
        assert extended[index].shape[-1] == TARGET_TOTAL_BATCHES
        assert jnp.array_equal(extended[index][..., :SOURCE_COMPLETED_BATCHES], source[index])


def test_extend_optimizer_histories_rejects_wrong_target_horizon() -> None:
    with pytest.raises(ValueError, match="target template horizon"):
        extend_optimizer_histories(_optimizer(SOURCE_COMPLETED_BATCHES), _optimizer(12_200))


def test_validate_launch_fork_requires_row_bound_provenance() -> None:
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            completed_training_batches=SOURCE_COMPLETED_BATCHES,
            metadata={"rlrmp_stage2_launch_fork": {
                "matrix_row_id": "other", "source_completed_batches": SOURCE_COMPLETED_BATCHES,
                "target_total_batches": TARGET_TOTAL_BATCHES,
            }},
        ),
        slots={"completed_batches": SOURCE_COMPLETED_BATCHES,
               "optimizer": SimpleNamespace(payload=b"serialized")},
    )
    with pytest.raises(ValueError, match="wrong matrix row"):
        validate_launch_fork(loaded, row_id="flat_3e-5")
