"""Adapters from rlrmp chunk functions to Feedbax executor kernels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import jax
import jax.random as jr

RLRMP_RUNTIME_CONTEXT_KEY = "rlrmp_runtime"

ChunkResult: TypeAlias = Mapping[str, Any]
UpdateKernel: TypeAlias = Callable[[Mapping[str, Any], Any, Mapping[str, Any]], Mapping[str, Any]]


class ChunkKernelAdapterError(ValueError):
    """Raised when an rlrmp chunk adapter violates the executor contract."""


class ChunkFn(Protocol):
    """One chunk of rlrmp training math adapted into an executor update step."""

    def __call__(
        self,
        runtime: Any,
        payload: Any,
        chunk_slots: Mapping[str, Any],
        coordinate: Any,
    ) -> ChunkResult:
        """Run one chunk and return slot updates."""


@dataclass(frozen=True)
class ChunkKernelAdapter:
    """Adapt one rlrmp chunk function to Feedbax's fixed kernel signature.

    The adapter keeps runtime objects out of slots. Method factories bind the
    static payload with :meth:`to_kernel`, while dynamic task/trainer closures
    are read from ``context["rlrmp_runtime"]`` once Feedbax F5 supplies that
    channel to ``execute_training_run_spec``.
    """

    chunk_fn: ChunkFn
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    metric_slots: tuple[str, ...] = ()
    prng_slot: str | None = None
    runtime_context_key: str = RLRMP_RUNTIME_CONTEXT_KEY
    name: str = "rlrmp chunk kernel"
    _write_set: frozenset[str] = field(init=False, repr=False)
    _metric_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_write_set", frozenset(self.writes))
        object.__setattr__(self, "_metric_set", frozenset(self.metric_slots))
        if self.prng_slot is not None and self.prng_slot not in self.reads:
            raise ChunkKernelAdapterError(
                f"{self.name}: prng_slot {self.prng_slot!r} must be listed in reads"
            )
        if self.prng_slot is not None and self.prng_slot not in self.writes:
            raise ChunkKernelAdapterError(
                f"{self.name}: prng_slot {self.prng_slot!r} must be listed in writes"
            )
        unknown_metric_slots = self._metric_set - self._write_set
        if unknown_metric_slots:
            raise ChunkKernelAdapterError(
                f"{self.name}: metric slots must be declared writes; "
                f"unknown={sorted(unknown_metric_slots)!r}"
            )

    def to_kernel(self, payload: Any) -> UpdateKernel:
        """Return a Feedbax update kernel with ``payload`` closed over."""

        def kernel(
            slots: Mapping[str, Any],
            coordinate: Any,
            context: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            runtime = self._runtime(context)
            chunk_slots = self._chunk_slots(slots)
            updates: dict[str, Any] = {}
            if self.prng_slot is not None:
                key_chunk, key_next = jr.split(chunk_slots[self.prng_slot])
                chunk_slots[self.prng_slot] = key_chunk
                updates[self.prng_slot] = key_next
            result = self.chunk_fn(runtime, payload, chunk_slots, coordinate)
            if not isinstance(result, Mapping):
                raise ChunkKernelAdapterError(
                    f"{self.name}: chunk_fn must return a mapping, got {type(result).__name__}"
                )
            updates.update(dict(result))
            self._validate_updates(updates)
            return updates

        return kernel

    def _runtime(self, context: Mapping[str, Any]) -> Any:
        try:
            return context[self.runtime_context_key]
        except KeyError as exc:
            raise ChunkKernelAdapterError(
                f"{self.name}: missing context[{self.runtime_context_key!r}]; "
                "runtime objects must be supplied through Feedbax kernel context, not slots"
            ) from exc

    def _chunk_slots(self, slots: Mapping[str, Any]) -> dict[str, Any]:
        missing = [slot for slot in self.reads if slot not in slots]
        if missing:
            raise ChunkKernelAdapterError(f"{self.name}: missing required input slots {missing!r}")
        return {slot: slots[slot] for slot in self.reads}

    def _validate_updates(self, updates: Mapping[str, Any]) -> None:
        returned = frozenset(updates)
        unexpected = sorted(returned - self._write_set)
        missing = sorted(self._write_set - returned)
        if unexpected:
            raise ChunkKernelAdapterError(
                f"{self.name}: chunk_fn returned undeclared writes {unexpected!r}; "
                f"declared writes={self.writes!r}"
            )
        if missing:
            raise ChunkKernelAdapterError(
                f"{self.name}: chunk_fn omitted declared writes {missing!r}"
            )
        bad_metrics = {
            slot: type(updates[slot]).__name__
            for slot in self.metric_slots
            if not _is_python_float(updates[slot])
        }
        if bad_metrics:
            raise ChunkKernelAdapterError(
                f"{self.name}: metric slots must contain Python float values; "
                f"bad_metrics={bad_metrics!r}"
            )


def _is_python_float(value: Any) -> bool:
    return (
        isinstance(value, float)
        and not isinstance(value, bool)
        and not isinstance(value, jax.Array)
    )
