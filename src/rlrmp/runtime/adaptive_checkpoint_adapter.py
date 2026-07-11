"""Explicit raw-C&S to serialized-adaptive checkpoint topology adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.tree as jt
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

    @property
    def target_transformed_slots(self) -> tuple[str, ...]:
        return (MODEL, OPTIMIZER, TRAIN_LOSS)

    @property
    def target_only_slots(self) -> dict[str, dict[str, str]]:
        return {
            ADAPTIVE_EPSILON_STATE: {"identity": "rlrmp.adaptive_initial_state.v1"},
            ZERO_ADVERSARY_GUARD: {"identity": "rlrmp.adaptive_initial_guard.v1"},
            DAMAGE_METRIC: {"identity": "rlrmp.adaptive_initial_metric.v1"},
            EPSILON_SCALE: {"identity": "rlrmp.adaptive_initial_metric.v1"},
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

    def continuation_slot_templates(self) -> dict[str, Any]:
        """Raw target topology used before the target/post serialization step."""

        return {OPTIMIZER: tuple(jt.leaves(self.optimizer_template))}

    def transform(self, slots: Mapping[str, Any]) -> Mapping[str, Any]:
        """Map every documented source/target slot without inference."""

        for slot in (MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS):
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
                TRAIN_LOSS: 0.0,
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
    source_leaves = tuple(value)
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
