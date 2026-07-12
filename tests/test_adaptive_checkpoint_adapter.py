"""Focused contracts for raw nominal to serialized adaptive checkpoint mapping."""

from __future__ import annotations

import jax.numpy as jnp
import jax.tree as jt
import pytest
from feedbax.contracts.checkpoints import BatchHistory
from feedbax.training.checkpoint_custody import CheckpointCompatibilityError

from rlrmp.runtime.adaptive_checkpoint_adapter import NominalToAdaptiveSlotAdapter
from rlrmp.runtime.checkpoint_custody import deserialize_pytree_slot
from rlrmp.train.adaptive_epsilon_native import SerializedPyTreeSlot
from rlrmp.train.executor.slots import (
    ADAPTIVE_EPSILON_STATE,
    COMPLETED_BATCHES,
    DAMAGE_METRIC,
    EPSILON_SCALE,
    MODEL,
    OPTIMIZER,
    PRNG,
    TRAIN_LOSS,
    ZERO_ADVERSARY_GUARD,
)


def _adapter() -> NominalToAdaptiveSlotAdapter:
    model = {"weights": jnp.arange(4, dtype=jnp.float32).reshape(2, 2)}
    optimizer = (
        jnp.asarray(12_000, dtype=jnp.int32),
        jnp.full((5, 4_500), jnp.nan, dtype=jnp.float32),
    )
    initial = {
        ADAPTIVE_EPSILON_STATE: b"adaptive-state",
        ZERO_ADVERSARY_GUARD: b"guard",
        DAMAGE_METRIC: 0.0,
        EPSILON_SCALE: 0.0,
        TRAIN_LOSS: 0.0,
    }
    return NominalToAdaptiveSlotAdapter(
        model,
        optimizer,
        initial,
        source_completed_batches=12_000,
        segment_batch_count=4_500,
    )


def test_adapter_maps_raw_nominal_slots_to_adaptive_serialized_slots() -> None:
    adapter = _adapter()
    source_model = tuple(jt.leaves({"weights": jnp.ones((2, 2), dtype=jnp.float32)}))
    source_optimizer = (
        jnp.asarray(12_000, dtype=jnp.int32),
        jnp.arange(5 * 4_500, dtype=jnp.float32).reshape(5, 4_500),
    )
    transformed = adapter.transform(
        {
            MODEL: source_model,
            OPTIMIZER: source_optimizer,
            PRNG: jnp.asarray([1, 2], dtype=jnp.uint32),
            COMPLETED_BATCHES: jnp.asarray(12_000, dtype=jnp.int32),
        }
    )

    assert isinstance(transformed[MODEL], SerializedPyTreeSlot)
    assert isinstance(transformed[OPTIMIZER], SerializedPyTreeSlot)
    restored_model = deserialize_pytree_slot(
        transformed[MODEL].payload,
        {"weights": jnp.zeros((2, 2), dtype=jnp.float32)},
        slot=MODEL,
    )
    restored_optimizer = deserialize_pytree_slot(
        transformed[OPTIMIZER].payload,
        adapter.optimizer_template,
        slot=OPTIMIZER,
    )
    assert jnp.array_equal(restored_model["weights"], source_model[0])
    assert jnp.array_equal(restored_optimizer[1], source_optimizer[1])
    assert jnp.array_equal(transformed[PRNG], jnp.asarray([1, 2], dtype=jnp.uint32))
    assert int(transformed[COMPLETED_BATCHES]) == 12_000
    assert transformed[TRAIN_LOSS] == 0.0
    assert transformed[ADAPTIVE_EPSILON_STATE] == b"adaptive-state"


def test_adapter_declares_train_loss_as_target_only() -> None:
    adapter = _adapter()

    assert TRAIN_LOSS not in adapter.target_transformed_slots
    assert TRAIN_LOSS in adapter.target_only_slots


def test_adapter_fails_closed_with_source_target_slot_and_path() -> None:
    adapter = _adapter()
    with pytest.raises(
        CheckpointCompatibilityError,
        match="source_slot='optimizer' target_slot='optimizer' path='/1'",
    ):
        adapter.transform(
            {
                MODEL: tuple(jt.leaves({"weights": jnp.ones((2, 2), dtype=jnp.float32)})),
                OPTIMIZER: (
                    jnp.asarray(12_000, dtype=jnp.int32),
                    jnp.zeros((5, 12_000), dtype=jnp.float32),
                ),
                PRNG: jnp.asarray([1, 2], dtype=jnp.uint32),
                COMPLETED_BATCHES: jnp.asarray(12_000, dtype=jnp.int32),
            }
        )


def test_adapter_marks_real_source_and_segment_histories_before_transform() -> None:
    adapter = _adapter()
    source = (
        jnp.asarray(12_000, dtype=jnp.int32),
        jnp.zeros((5, 12_000), dtype=jnp.float32),
    )

    marked_source = adapter.prepare_source_optimizer(source)
    segment_template = adapter.continuation_slot_templates()[OPTIMIZER]

    assert isinstance(marked_source[1], BatchHistory)
    assert marked_source[1].value.shape == (5, 12_000)
    assert isinstance(segment_template[1], BatchHistory)
    assert segment_template[1].value.shape == (5, 4_500)
