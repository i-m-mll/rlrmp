"""Initial-slot runtime helpers for native-executor rlrmp methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.random as jr


@dataclass(frozen=True)
class RlrmpRuntime:
    """Runtime-only objects passed through Feedbax kernel context, never slots."""

    components: Mapping[str, Any] = field(default_factory=dict)
    stop_after_batches: int | None = None
    completed_batches_reader: Callable[[], int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def component(self, name: str, default: Any = None) -> Any:
        """Return one runtime component by name."""
        return self.components.get(name, default)

    def read_completed_batches(self) -> int:
        """Return the authoritative cumulative training-batch coordinate."""
        if self.completed_batches_reader is None:
            raise RuntimeError("RLRMP runtime does not expose completed-batch progress")
        value = self.completed_batches_reader()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                "RLRMP runtime completed-batch progress must be a non-negative integer, "
                f"got {value!r}"
            )
        return value


def split_initial_keys(
    key: jr.PRNGKeyArray,
) -> tuple[jr.PRNGKeyArray, jr.PRNGKeyArray, jr.PRNGKeyArray]:
    """Preserve the legacy three-way split shape for future equivalence gates."""
    key_init, key_train, key_adversary = jr.split(key, 3)
    return key_init, key_train, key_adversary
