"""Fixed-seed equivalence harness scaffolding for native executor ports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from feedbax.contracts.worker import (
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    MetricGuardSpec,
    PhaseProgramSpec,
    PhaseSpec,
    PhaseTransitionSpec,
    ProgressCoordinate,
    ResumeCoordinateSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
)
from feedbax.training.phase_executor import InMemoryCheckpointStore, PhaseProgramExecutor

from rlrmp.train.executor.adapters import ChunkKernelAdapter, RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.guards import make_stop_predicate
from rlrmp.train.executor.initial_slots import RlrmpRuntime
from rlrmp.train.executor.slots import (
    COMPLETED_BATCHES,
    MODEL,
    OPTIMIZER,
    PRNG,
    TRAIN_LOSS,
    supervised_state_slots,
)

LEAF_ATOL = 1e-7
LEAF_RTOL = 0.0
TOY_KERNEL_REF = "rlrmp.executor.toy_chunk"
TOY_STOP_PREDICATE_REF = "rlrmp.executor.toy_stop"
TOY_BARRIER = "after_train_chunk"


@dataclass(frozen=True)
class LeafDiff:
    """Maximum absolute difference for one leaf."""

    path: str
    max_abs_diff: float


@dataclass(frozen=True)
class EquivalenceReport:
    """Result of one equivalence check."""

    passed: bool
    leaf_diffs: tuple[LeafDiff, ...]
    completed_batches: tuple[int, int]
    loss_series: tuple[tuple[float, ...], tuple[float, ...]]
    mode: str = "fixed_seed"

    @property
    def max_abs_diff(self) -> float:
        """Largest per-leaf difference in the report."""
        return max((diff.max_abs_diff for diff in self.leaf_diffs), default=0.0)


@dataclass(frozen=True)
class ToyPayload:
    """Small payload used to self-test the harness before real methods go live."""

    n_train_batches: int
    chunk_batches: int = 1


def run_toy_paired_equivalence(
    *,
    n_chunks: int = 3,
    chunk_batches: int = 1,
    seed: int = 0,
) -> EquivalenceReport:
    """Compare the toy legacy loop against the toy executor path."""
    payload = ToyPayload(n_train_batches=n_chunks * chunk_batches, chunk_batches=chunk_batches)
    initial_slots = _toy_initial_slots(seed)
    legacy_slots, legacy_losses = _run_toy_legacy(initial_slots, payload)
    executor_slots, executor_losses = _run_toy_executor(initial_slots, payload)
    return _build_report(
        legacy_slots,
        executor_slots,
        legacy_losses=legacy_losses,
        executor_losses=executor_losses,
    )


def run_toy_resume_equivalence(
    *,
    n_chunks: int = 3,
    kill_after_chunk: int = 1,
    chunk_batches: int = 1,
    seed: int = 0,
) -> EquivalenceReport:
    """Compare uninterrupted toy executor execution against guarded resume."""
    if not 0 < kill_after_chunk < n_chunks:
        raise ValueError("kill_after_chunk must be inside the run")
    payload = ToyPayload(n_train_batches=n_chunks * chunk_batches, chunk_batches=chunk_batches)
    initial_slots = _toy_initial_slots(seed)
    full_slots, full_losses = _run_toy_executor(initial_slots, payload)
    resumed_slots, resumed_losses = _run_toy_executor_resume(
        initial_slots,
        payload,
        stop_after_batches=kill_after_chunk * chunk_batches,
    )
    return _build_report(
        full_slots,
        resumed_slots,
        legacy_losses=full_losses,
        executor_losses=resumed_losses,
    )


def compare_pytrees(left: Any, right: Any) -> tuple[LeafDiff, ...]:
    """Return per-leaf maximum absolute differences for matching pytrees."""
    left_leaves = jt.leaves(left)
    right_leaves = jt.leaves(right)
    if jt.structure(left) != jt.structure(right):
        raise ValueError("cannot compare pytrees with different structures")
    return tuple(
        LeafDiff(path=f"leaf_{index}", max_abs_diff=_max_abs_diff(left_leaf, right_leaf))
        for index, (left_leaf, right_leaf) in enumerate(zip(left_leaves, right_leaves, strict=True))
    )


def _run_toy_legacy(
    initial_slots: Mapping[str, Any],
    payload: ToyPayload,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    slots = dict(initial_slots)
    losses: list[float] = []
    runtime = _toy_runtime(payload)
    coordinate = ProgressCoordinate(run_id="toy-legacy", phase="train_chunk")
    while int(slots[COMPLETED_BATCHES]) < payload.n_train_batches:
        key_chunk, key_next = jr.split(slots[PRNG])
        chunk_slots = {
            MODEL: slots[MODEL],
            OPTIMIZER: slots[OPTIMIZER],
            PRNG: key_chunk,
            COMPLETED_BATCHES: slots[COMPLETED_BATCHES],
        }
        updates = _toy_chunk_fn(runtime, payload, chunk_slots, coordinate)
        updates[PRNG] = key_next
        slots.update(updates)
        losses.append(float(slots[TRAIN_LOSS]))
        coordinate = coordinate.model_copy(update={"global_step": coordinate.global_step + 1})
    return slots, tuple(losses)


def _run_toy_executor(
    initial_slots: Mapping[str, Any],
    payload: ToyPayload,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    executor = _toy_executor(payload)
    result = executor.run(
        initial_slots,
        run_id="toy-executor",
        context={RLRMP_RUNTIME_CONTEXT_KEY: _toy_runtime(payload)},
    )
    return result.slots, _losses_from_progress(result.progress)


def _run_toy_executor_resume(
    initial_slots: Mapping[str, Any],
    payload: ToyPayload,
    *,
    stop_after_batches: int,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    checkpoint_store = InMemoryCheckpointStore()
    executor = _toy_executor(payload, checkpoint_store=checkpoint_store)
    stopped = executor.run(
        initial_slots,
        run_id="toy-resume",
        context={
            RLRMP_RUNTIME_CONTEXT_KEY: _toy_runtime(
                payload,
                stop_after_batches=stop_after_batches,
            )
        },
    )
    resumed = executor.run(
        stopped.slots,
        run_id="toy-resume",
        resume_from_barrier=TOY_BARRIER,
        context={RLRMP_RUNTIME_CONTEXT_KEY: _toy_runtime(payload)},
    )
    return resumed.slots, _losses_from_progress(stopped.progress + resumed.progress)


def _toy_executor(
    payload: ToyPayload,
    *,
    checkpoint_store: InMemoryCheckpointStore | None = None,
) -> PhaseProgramExecutor:
    adapter = ChunkKernelAdapter(
        chunk_fn=_toy_chunk_fn,
        reads=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES),
        writes=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS),
        metric_slots=(TRAIN_LOSS,),
        prng_slot=PRNG,
        name="toy native-executor chunk",
    )
    return PhaseProgramExecutor(
        _toy_program(),
        {TOY_KERNEL_REF: adapter.to_kernel(payload)},
        guard_predicates={TOY_STOP_PREDICATE_REF: make_stop_predicate(payload)},
        checkpoint_store=checkpoint_store,
        state_slots=supervised_state_slots(),
    )


def _toy_program() -> PhaseProgramSpec:
    return PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="train_chunk",
                kind="custom",
                reads=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES],
                writes=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS],
                update_steps=["toy_chunk"],
                legal_next=["done", "train_chunk"],
                checkpoint_barrier=TOY_BARRIER,
                loop_axis="batch",
            ),
            PhaseSpec(
                name="done",
                kind="evaluation",
                reads=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS],
            ),
        ],
        initial_phase="train_chunk",
        update_steps=[
            UpdateStepSpec(
                name="toy_chunk",
                kind="custom",
                kernel=UpdateKernelSpec(kernel_ref=TOY_KERNEL_REF),
                reads=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES],
                writes=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS],
            )
        ],
        transitions=[
            PhaseTransitionSpec(
                source="train_chunk",
                target="done",
                barrier=TOY_BARRIER,
                guard=MetricGuardSpec(
                    predicate_ref=TOY_STOP_PREDICATE_REF,
                    metric_slots=[TRAIN_LOSS],
                    bookkeeping_slots=[COMPLETED_BATCHES],
                ),
            ),
            PhaseTransitionSpec(source="train_chunk", target="train_chunk", barrier=TOY_BARRIER),
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name=TOY_BARRIER,
                phase="train_chunk",
                slots=[
                    CheckpointSlotSpec(slot=MODEL),
                    CheckpointSlotSpec(slot=OPTIMIZER),
                    CheckpointSlotSpec(slot=PRNG),
                    CheckpointSlotSpec(slot=COMPLETED_BATCHES),
                    CheckpointSlotSpec(slot=TRAIN_LOSS, required=False),
                ],
                resume_coordinate=ResumeCoordinateSpec(
                    phase="train_chunk",
                    completed_barrier=TOY_BARRIER,
                    global_step=1,
                ),
            )
        ],
    )


def _toy_chunk_fn(
    runtime: RlrmpRuntime,
    payload: ToyPayload,
    chunk_slots: Mapping[str, Any],
    coordinate: ProgressCoordinate,
) -> Mapping[str, Any]:
    del coordinate
    chunk_batches = int(runtime.component("chunk_batches", getattr(payload, "chunk_batches", 1)))
    step = jnp.asarray(chunk_batches, dtype=jnp.float32)
    jitter = jnp.asarray(0.0, dtype=jnp.float32) * jr.normal(chunk_slots[PRNG], ())
    model = jnp.asarray(chunk_slots[MODEL]) + step + jitter
    optimizer = {
        "step": jnp.asarray(chunk_slots[OPTIMIZER]["step"]) + chunk_batches,
        "diagnostics": jnp.asarray(chunk_slots[OPTIMIZER]["diagnostics"]) + step,
    }
    completed_batches = int(chunk_slots[COMPLETED_BATCHES]) + chunk_batches
    return {
        MODEL: model,
        OPTIMIZER: optimizer,
        COMPLETED_BATCHES: completed_batches,
        TRAIN_LOSS: float(jax.device_get(jnp.mean(model))),
    }


def _toy_initial_slots(seed: int) -> dict[str, Any]:
    return {
        MODEL: jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        OPTIMIZER: {
            "step": jnp.asarray(0, dtype=jnp.int32),
            "diagnostics": jnp.zeros((3,), dtype=jnp.float32),
        },
        PRNG: jr.PRNGKey(seed),
        COMPLETED_BATCHES: 0,
        TRAIN_LOSS: 0.0,
    }


def _toy_runtime(
    payload: ToyPayload,
    *,
    stop_after_batches: int | None = None,
) -> RlrmpRuntime:
    return RlrmpRuntime(
        components={"chunk_batches": payload.chunk_batches},
        stop_after_batches=stop_after_batches,
    )


def _build_report(
    legacy_slots: Mapping[str, Any],
    executor_slots: Mapping[str, Any],
    *,
    legacy_losses: tuple[float, ...],
    executor_losses: tuple[float, ...],
) -> EquivalenceReport:
    leaf_diffs = compare_pytrees(_comparable_slots(legacy_slots), _comparable_slots(executor_slots))
    loss_diffs = [
        abs(left - right) for left, right in zip(legacy_losses, executor_losses, strict=True)
    ]
    passed = (
        all(diff.max_abs_diff <= LEAF_ATOL for diff in leaf_diffs)
        and all(diff <= LEAF_ATOL for diff in loss_diffs)
        and int(legacy_slots[COMPLETED_BATCHES]) == int(executor_slots[COMPLETED_BATCHES])
    )
    return EquivalenceReport(
        passed=passed,
        leaf_diffs=leaf_diffs,
        completed_batches=(
            int(legacy_slots[COMPLETED_BATCHES]),
            int(executor_slots[COMPLETED_BATCHES]),
        ),
        loss_series=(legacy_losses, executor_losses),
    )


def _comparable_slots(slots: Mapping[str, Any]) -> dict[str, Any]:
    return {
        MODEL: slots[MODEL],
        OPTIMIZER: slots[OPTIMIZER],
        PRNG: slots[PRNG],
        COMPLETED_BATCHES: slots[COMPLETED_BATCHES],
        TRAIN_LOSS: slots[TRAIN_LOSS],
    }


def _losses_from_progress(progress: list[ProgressCoordinate]) -> tuple[float, ...]:
    return tuple(
        float(coordinate.metrics[TRAIN_LOSS])
        for coordinate in progress
        if coordinate.phase == "train_chunk" and TRAIN_LOSS in coordinate.metrics
    )


def _max_abs_diff(left: Any, right: Any) -> float:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(abs(left - right))
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.size == 0:
        return 0.0
    return float(jax.device_get(jnp.max(jnp.abs(left_array - right_array))))
