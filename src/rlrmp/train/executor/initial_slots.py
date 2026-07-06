"""Initial-slot builder protocol for native-executor rlrmp methods."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax.random as jr
from feedbax.config.namespace import TreeNamespace

from rlrmp.train.executor import slots as slot_names


@dataclass(frozen=True)
class RlrmpRuntime:
    """Runtime-only objects passed through Feedbax kernel context, never slots."""

    components: Mapping[str, Any] = field(default_factory=dict)
    stop_after_batches: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def component(self, name: str, default: Any = None) -> Any:
        """Return one runtime component by name."""
        return self.components.get(name, default)


class InitialSlotsBuilder(Protocol):
    """Build executor initial slots plus runtime-only rlrmp objects."""

    def __call__(
        self,
        *,
        run_spec: Mapping[str, Any],
        hps: TreeNamespace,
        args: Namespace,
        key: jr.PRNGKeyArray,
    ) -> tuple[dict[str, Any], RlrmpRuntime]:
        """Return ``(initial_slots, runtime)`` for one method execution."""


@dataclass(frozen=True)
class CsSupervisedInitialSlotsBuilder:
    """Template for the future cs-supervised native-executor slot builder.

    R2 will bind the real task/model/trainer setup here. R0 keeps the shape
    explicit without making the legacy cs-supervised method live.
    """

    family: str = slot_names.CS_SUPERVISED_METHOD_REF

    def __call__(
        self,
        *,
        run_spec: Mapping[str, Any],
        hps: TreeNamespace,
        args: Namespace,
        key: jr.PRNGKeyArray,
    ) -> tuple[dict[str, Any], RlrmpRuntime]:
        del run_spec, hps, args, key
        raise NotImplementedError(
            f"{self.family} initial-slot construction is a scaffold for R2; "
            "R0 intentionally does not bind real training methods live"
        )


def split_initial_keys(
    key: jr.PRNGKeyArray,
) -> tuple[jr.PRNGKeyArray, jr.PRNGKeyArray, jr.PRNGKeyArray]:
    """Preserve the legacy three-way split shape for future equivalence gates."""
    key_init, key_train, key_adversary = jr.split(key, 3)
    return key_init, key_train, key_adversary
