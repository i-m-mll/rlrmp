"""Explicit raw-C&S to serialized-adaptive checkpoint topology adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.tree as jt
from feedbax.contracts.checkpoints import BatchHistory
from feedbax.training.checkpoint_custody import CheckpointCompatibilityError

from rlrmp.runtime.checkpoint_custody import serialize_pytree_slot
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


@dataclass(frozen=True)
class NominalToAdaptiveSlotAdapter:
    """Target/post transform after Feedbax extends the raw optimizer tuple."""

    model_template: Any
    optimizer_template: Any
    adaptive_initial_slots: Mapping[str, Any]
    source_completed_batches: int
    segment_batch_count: int

    @property
    def target_transformed_slots(self) -> tuple[str, ...]:
        return (MODEL, OPTIMIZER)

    @property
    def target_only_slots(self) -> dict[str, dict[str, str]]:
        return {
            ADAPTIVE_EPSILON_STATE: {"identity": "rlrmp.adaptive_initial_state.v1"},
            ZERO_ADVERSARY_GUARD: {"identity": "rlrmp.adaptive_initial_guard.v1"},
            DAMAGE_METRIC: {"identity": "rlrmp.adaptive_initial_metric.v1"},
            EPSILON_SCALE: {"identity": "rlrmp.adaptive_initial_metric.v1"},
            TRAIN_LOSS: {"identity": "rlrmp.adaptive_initial_metric.v1"},
        }

    @property
    def transform_metadata(self) -> dict[str, Any]:
        return {
            "identity": "rlrmp.nominal_raw_to_adaptive_serialized.v1",
            "parameters": {
                "model": "raw cs-supervised tuple -> SerializedPyTreeSlot",
                "optimizer": "extended raw optimizer tuple -> SerializedPyTreeSlot",
                "preserved": [PRNG, COMPLETED_BATCHES],
            },
        }

    @property
    def source_slot_transforms(self) -> dict[str, Any]:
        """Mark raw source histories before Feedbax allocates the new segment."""

        return {OPTIMIZER: self.prepare_source_slots}

    @property
    def source_transform_metadata(self) -> dict[str, dict[str, Any]]:
        """Describe the explicit raw-history marking transform."""

        return {
            OPTIMIZER: {
                "identity": "rlrmp.nominal_optimizer_segment_histories.v1",
                "parameters": {
                    "source_completed_batches": self.source_completed_batches,
                    "segment_batch_count": self.segment_batch_count,
                    "batch_axis": -1,
                },
            }
        }

    def continuation_slot_templates(self) -> dict[str, Any]:
        """Raw target topology used before the target/post serialization step."""

        leaves = tuple(jt.leaves(self.optimizer_template))
        indices = self._history_indices(leaves)
        return {
            OPTIMIZER: tuple(
                BatchHistory(leaf, batch_axis=-1) if index in indices else leaf
                for index, leaf in enumerate(leaves)
            )
        }

    def prepare_source_optimizer(self, value: Any) -> tuple[Any, ...]:
        """Mark source histories and fail closed on any source/segment ABI drift."""

        if not isinstance(value, tuple):
            raise CheckpointCompatibilityError(
                "nominal-to-adaptive source history marking requires raw tuple source; "
                f"source_slot={OPTIMIZER!r} target_slot={OPTIMIZER!r} path='/' "
                f"actual_type={type(value).__name__!r}"
            )
        target_leaves = tuple(jt.leaves(self.optimizer_template))
        if len(value) != len(target_leaves):
            raise CheckpointCompatibilityError(
                "nominal-to-adaptive source history marking leaf-count mismatch; "
                f"source_slot={OPTIMIZER!r} target_slot={OPTIMIZER!r} path='/' "
                f"source_leaves={len(value)} target_leaves={len(target_leaves)}"
            )
        indices = self._history_indices(target_leaves)
        marked: list[Any] = []
        for index, (source, target) in enumerate(zip(value, target_leaves, strict=True)):
            if index not in indices:
                marked.append(source)
                continue
            source_shape = getattr(source, "shape", None)
            target_shape = getattr(target, "shape", None)
            expected_source_shape = (*target_shape[:-1], self.source_completed_batches)
            if source_shape != expected_source_shape or getattr(source, "dtype", None) != getattr(
                target, "dtype", None
            ):
                raise CheckpointCompatibilityError(
                    "nominal-to-adaptive source history ABI mismatch; "
                    f"source_slot={OPTIMIZER!r} target_slot={OPTIMIZER!r} path='/{index}' "
                    f"source_shape={source_shape!r} expected_source_shape={expected_source_shape!r} "
                    f"target_segment_shape={target_shape!r}"
                )
            marked.append(BatchHistory(source, batch_axis=-1))
        return tuple(marked)

    def prepare_source_slots(self, slots: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the full source mapping with optimizer histories explicitly marked."""

        if OPTIMIZER not in slots:
            raise CheckpointCompatibilityError(
                "nominal-to-adaptive source history marking missing source slot; "
                f"source_slot={OPTIMIZER!r} target_slot={OPTIMIZER!r} path='/'"
            )
        transformed = dict(slots)
        transformed[OPTIMIZER] = self.prepare_source_optimizer(slots[OPTIMIZER])
        return transformed

    def _history_indices(self, leaves: tuple[Any, ...]) -> frozenset[int]:
        indices = frozenset(
            index
            for index, leaf in enumerate(leaves)
            if eqx.is_array(leaf)
            and leaf.ndim > 0
            and leaf.shape[-1] == self.segment_batch_count
        )
        if not indices:
            raise CheckpointCompatibilityError(
                "nominal-to-adaptive target template has no segment-local histories; "
                f"source_slot={OPTIMIZER!r} target_slot={OPTIMIZER!r} path='/' "
                f"segment_batch_count={self.segment_batch_count}"
            )
        return indices

    def transform(self, slots: Mapping[str, Any]) -> Mapping[str, Any]:
        """Map every documented source/target slot without inference."""

        for slot in (MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES):
            if slot not in slots:
                raise CheckpointCompatibilityError(
                    "nominal-to-adaptive mapping missing source slot; "
                    f"source_slot={slot!r} target_slot={slot!r} path='/'"
                )
        model = eqx.combine(
            _restore_tuple(
                slots[MODEL],
                template=eqx.filter(self.model_template, eqx.is_array),
                source_slot=MODEL,
                target_slot=MODEL,
            ),
            self.model_template,
        )
        optimizer = _restore_tuple(
            slots[OPTIMIZER],
            template=self.optimizer_template,
            source_slot=OPTIMIZER,
            target_slot=OPTIMIZER,
        )
        transformed = dict(slots)
        transformed.update(
            {
                MODEL: SerializedPyTreeSlot(serialize_pytree_slot(model)),
                OPTIMIZER: SerializedPyTreeSlot(serialize_pytree_slot(optimizer)),
            }
        )
        for slot in self.target_only_slots:
            value = self.adaptive_initial_slots.get(slot)
            if value is None:
                raise CheckpointCompatibilityError(
                    "nominal-to-adaptive mapping missing target template slot; "
                    f"source_slot=<none> target_slot={slot!r} path='/'"
                )
            transformed[slot] = value
        return transformed


def _restore_tuple(
    value: Any,
    *,
    template: Any,
    source_slot: str,
    target_slot: str,
) -> Any:
    if not isinstance(value, tuple):
        raise CheckpointCompatibilityError(
            "nominal-to-adaptive mapping requires raw tuple source; "
            f"source_slot={source_slot!r} target_slot={target_slot!r} path='/' "
            f"actual_type={type(value).__name__!r}"
        )
    source_leaves = tuple(
        leaf.value if isinstance(leaf, BatchHistory) else leaf for leaf in value
    )
    target_leaves = jt.leaves(template)
    if len(source_leaves) != len(target_leaves):
        raise CheckpointCompatibilityError(
            "nominal-to-adaptive mapping leaf-count mismatch; "
            f"source_slot={source_slot!r} target_slot={target_slot!r} path='/' "
            f"source_leaves={len(source_leaves)} target_leaves={len(target_leaves)}"
        )
    for index, (source, target) in enumerate(zip(source_leaves, target_leaves, strict=True)):
        if getattr(source, "shape", None) != getattr(target, "shape", None) or getattr(
            source, "dtype", None
        ) != getattr(target, "dtype", None):
            raise CheckpointCompatibilityError(
                "nominal-to-adaptive mapping leaf mismatch; "
                f"source_slot={source_slot!r} target_slot={target_slot!r} path='/{index}' "
                f"source_shape={getattr(source, 'shape', None)!r} "
                f"target_shape={getattr(target, 'shape', None)!r}"
            )
    return jt.unflatten(jt.structure(template), source_leaves)
