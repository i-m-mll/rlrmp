"""Guard predicate factories for rlrmp native executor phase programs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY, UpdateKernel
from rlrmp.train.executor.slots import COMPLETED_BATCHES, ZERO_ADVERSARY_GUARD


def make_completion_predicate(
    payload: Any,
    *,
    completed_slot: str = COMPLETED_BATCHES,
    payload_batches_field: str = "n_train_batches",
) -> UpdateKernel:
    """Return true once completed batches reach the payload's planned total."""
    n_train_batches = _required_int(payload, payload_batches_field)

    def predicate(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any] | bool:
        del coordinate, context
        return _slot_int(slots, completed_slot) >= n_train_batches

    return predicate


def make_stop_after_batches_predicate(
    *,
    completed_slot: str = COMPLETED_BATCHES,
    runtime_context_key: str = RLRMP_RUNTIME_CONTEXT_KEY,
) -> UpdateKernel:
    """Return true when runtime context requests an early batch stop."""

    def predicate(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del coordinate
        runtime = context.get(runtime_context_key)
        stop_after_batches = _maybe_get(runtime, "stop_after_batches")
        if stop_after_batches is None:
            return False
        return _slot_int(slots, completed_slot) >= int(stop_after_batches)

    return predicate


def make_zero_adversary_predicate(
    *,
    zero_adversary_slot: str = ZERO_ADVERSARY_GUARD,
) -> UpdateKernel:
    """Return true when zero-adversary bookkeeping says the run should stop."""

    def predicate(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        del coordinate, context
        guard_state = slots.get(zero_adversary_slot)
        return bool(_maybe_get(guard_state, "should_stop", default=False))

    return predicate


def make_stop_predicate(
    payload: Any,
    *,
    completed_slot: str = COMPLETED_BATCHES,
    zero_adversary_slot: str = ZERO_ADVERSARY_GUARD,
    runtime_context_key: str = RLRMP_RUNTIME_CONTEXT_KEY,
    payload_batches_field: str = "n_train_batches",
) -> UpdateKernel:
    """OR completion, runtime stop-after-batches, and zero-adversary exits."""
    completion = make_completion_predicate(
        payload,
        completed_slot=completed_slot,
        payload_batches_field=payload_batches_field,
    )
    stop_after = make_stop_after_batches_predicate(
        completed_slot=completed_slot,
        runtime_context_key=runtime_context_key,
    )
    zero_adversary = make_zero_adversary_predicate(zero_adversary_slot=zero_adversary_slot)

    def predicate(
        slots: Mapping[str, Any],
        coordinate: Any,
        context: Mapping[str, Any],
    ) -> bool:
        return bool(
            completion(slots, coordinate, context)
            or stop_after(slots, coordinate, context)
            or zero_adversary(slots, coordinate, context)
        )

    return predicate


def _required_int(value: Any, name: str) -> int:
    result = _maybe_get(value, name)
    if result is None:
        raise ValueError(f"payload must define {name!r}")
    return int(result)


def _slot_int(slots: Mapping[str, Any], name: str) -> int:
    try:
        return int(slots[name])
    except KeyError as exc:
        raise ValueError(f"missing guard bookkeeping slot {name!r}") from exc


def _maybe_get(value: Any, name: str, *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
