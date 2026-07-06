"""Scaffolding for rlrmp methods that will run through Feedbax's executor."""

from rlrmp.train.executor.adapters import (
    ChunkKernelAdapter,
    ChunkKernelAdapterError,
    ChunkResult,
    RLRMP_RUNTIME_CONTEXT_KEY,
)
from rlrmp.train.executor.guards import (
    make_completion_predicate,
    make_stop_after_batches_predicate,
    make_stop_predicate,
    make_zero_adversary_predicate,
)
from rlrmp.train.executor.initial_slots import (
    CsSupervisedInitialSlotsBuilder,
    InitialSlotsBuilder,
    RlrmpRuntime,
)

__all__ = [
    "ChunkKernelAdapter",
    "ChunkKernelAdapterError",
    "ChunkResult",
    "CsSupervisedInitialSlotsBuilder",
    "InitialSlotsBuilder",
    "RLRMP_RUNTIME_CONTEXT_KEY",
    "RlrmpRuntime",
    "make_completion_predicate",
    "make_stop_after_batches_predicate",
    "make_stop_predicate",
    "make_zero_adversary_predicate",
]
