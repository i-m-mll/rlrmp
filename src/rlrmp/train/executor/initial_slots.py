"""Initial-slot runtime helpers for native-executor rlrmp methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.random as jr


@dataclass(frozen=True)
class RlrmpRuntime:
    """Runtime-only objects passed through Feedbax kernel context, never slots."""

    components: Mapping[str, Any] = field(default_factory=dict)
    stop_after_batches: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def component(self, name: str, default: Any = None) -> Any:
        """Return one runtime component by name."""
        return self.components.get(name, default)


def split_initial_keys(
    key: jr.PRNGKeyArray,
) -> tuple[jr.PRNGKeyArray, jr.PRNGKeyArray, jr.PRNGKeyArray]:
    """Preserve the legacy three-way split shape for future equivalence gates."""
    key_init, key_train, key_adversary = jr.split(key, 3)
    return key_init, key_train, key_adversary
