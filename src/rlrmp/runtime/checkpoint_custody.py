"""RLRMP adapters for Feedbax training checkpoint custody."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import (
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    EffectivePhaseSpec,
    ProgressCoordinate,
    StateSlotSpec,
    TrainingBatchProgressSpec,
    derive_consistency_predicate,
)
from feedbax.training.checkpoint_custody import (
    load_latest_checkpoint as feedbax_load_latest_checkpoint,
    write_checkpoint_transaction,
)

from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    feedbax_training_run_spec_from_payload,
)


CS_BARRIER = "after_train_batch"
CS_SUPERVISED_NATIVE_BARRIER = "after_train_chunk"
ADAPTIVE_EPSILON_BARRIER = "after_adaptive_epsilon_train_chunk"
POLICY_ADVERSARY_BARRIER = "after_policy_adversary_train_chunk"
MINIMAX_WARMUP_BARRIER = "after_warmup"
MINIMAX_ADVERSARIAL_BARRIER = "after_adversarial"

_CS_EXTRA_SLOTS = (
    StateSlotSpec(name="completed_batches", role="checkpoint"),
    StateSlotSpec(name="history", role="auxiliary", required=False),
    StateSlotSpec(name="adversary_policy", role="auxiliary", required=False),
    StateSlotSpec(name="adversary_optimizer", role="optimizer", required=False),
    StateSlotSpec(name="adaptive_epsilon_state", role="auxiliary", required=False),
    StateSlotSpec(name="checkpoint_metadata", role="auxiliary", required=False),
)
_CS_BARRIER_SLOTS = (
    CheckpointSlotSpec(slot="model", required=False),
    CheckpointSlotSpec(slot="optimizer", required=False),
    CheckpointSlotSpec(slot="prng"),
    CheckpointSlotSpec(slot="completed_batches"),
    CheckpointSlotSpec(slot="history", required=False),
    CheckpointSlotSpec(slot="adversary_policy", required=False),
    CheckpointSlotSpec(slot="adversary_optimizer", required=False),
    CheckpointSlotSpec(slot="adaptive_epsilon_state", required=False),
    CheckpointSlotSpec(slot="checkpoint_metadata", required=False),
)
_MINIMAX_EXTRA_SLOTS = (
    StateSlotSpec(name="adversary_losses", role="metric", required=False),
    StateSlotSpec(name="controller_losses", role="metric", required=False),
    StateSlotSpec(name="adversary_indices", role="metric", required=False),
    StateSlotSpec(name="active_batch_index", role="checkpoint"),
    StateSlotSpec(name="active_member_index", role="checkpoint"),
    StateSlotSpec(name="spec_digests", role="checkpoint", required=False),
    StateSlotSpec(name="warmup_history", role="auxiliary", required=False),
)
_MINIMAX_BARRIER_EXTRA_SLOTS = (
    CheckpointSlotSpec(slot="controller", axis="replicate", required=False),
    CheckpointSlotSpec(slot="controller_optimizer", axis="replicate", required=False),
    CheckpointSlotSpec(slot="adversary_population", axis="adversary_member", required=False),
    CheckpointSlotSpec(slot="adversary_optimizer", axis="adversary_member", required=False),
    CheckpointSlotSpec(slot="adversary_losses", required=False),
    CheckpointSlotSpec(slot="controller_losses", required=False),
    CheckpointSlotSpec(slot="adversary_indices", required=False),
    CheckpointSlotSpec(slot="active_batch_index"),
    CheckpointSlotSpec(slot="active_member_index"),
    CheckpointSlotSpec(slot="spec_digests", required=False),
    CheckpointSlotSpec(slot="warmup_history", required=False),
)


def has_feedbax_training_spec(run_spec: Mapping[str, Any]) -> bool:
    """Return whether a run recipe carries the composed Feedbax TrainingRunSpec."""

    return FEEDBAX_TRAINING_RUN_SPEC_KEY in run_spec


def has_custody_checkpoint(root: Path) -> bool:
    """Return whether a Feedbax latest pointer is present under ``root``."""

    return (root / "latest.json").is_file()


def cs_custody_training_spec(run_spec: Mapping[str, Any]) -> TrainingRunSpec:
    """Return the C&S TrainingRunSpec augmented with RLRMP checkpoint slots."""

    spec = feedbax_training_run_spec_from_payload(dict(run_spec))
    return _augment_training_spec(
        spec,
        extra_state_slots=_CS_EXTRA_SLOTS,
        barrier_slots={
            CS_BARRIER: _CS_BARRIER_SLOTS,
            CS_SUPERVISED_NATIVE_BARRIER: _CS_BARRIER_SLOTS,
            ADAPTIVE_EPSILON_BARRIER: _CS_BARRIER_SLOTS,
            POLICY_ADVERSARY_BARRIER: _CS_BARRIER_SLOTS,
        },
        consistency_mode="barrier-coordinate",
        batch_progress=TrainingBatchProgressSpec(slot="completed_batches"),
    )


def minimax_custody_training_spec(spec: TrainingRunSpec) -> TrainingRunSpec:
    """Return the minimax TrainingRunSpec augmented with RLRMP checkpoint slots."""

    return _augment_training_spec(
        spec,
        extra_state_slots=_MINIMAX_EXTRA_SLOTS,
        barrier_slots={
            MINIMAX_WARMUP_BARRIER: _MINIMAX_BARRIER_EXTRA_SLOTS,
            MINIMAX_ADVERSARIAL_BARRIER: _MINIMAX_BARRIER_EXTRA_SLOTS,
        },
        consistency_mode="population-barrier",
    )


def write_cs_checkpoint_transaction(
    root: Path,
    *,
    run_spec: Mapping[str, Any],
    completed_batches: int,
    program_step: int,
    slots: Mapping[str, Any],
    status: str = "partial",
) -> Any:
    """Write a Feedbax custody transaction for C&S chunked training."""

    custody_spec = cs_custody_training_spec(run_spec)
    barrier_name = _cs_checkpoint_barrier_name(custody_spec)
    coordinate = ProgressCoordinate(
        run_id=_run_id(run_spec, prefix="rlrmp-cs"),
        phase=_cs_checkpoint_barrier_phase(custody_spec, barrier_name),
        program_step=int(program_step),
        completed_barrier=barrier_name,
    )
    return write_checkpoint_transaction(
        root,
        run_spec=custody_spec,
        phase_program=custody_spec.worker_execution.method_contract.phase_program,
        barrier_name=barrier_name,
        coordinate=coordinate,
        slots=slots,
        status=status,
        history_availability={"history": slots.get("history") is not None},
    )


def write_governed_checkpoint_transaction(**kwargs: Any) -> Any:
    """Route an already-governed checkpoint transaction through the custody adapter."""

    return write_checkpoint_transaction(**kwargs)


def load_cs_checkpoint_transaction(
    root: Path,
    *,
    run_spec: Mapping[str, Any],
    expected_slots: Mapping[str, Any],
) -> Any:
    """Load and validate the latest C&S custody transaction."""

    custody_spec = cs_custody_training_spec(run_spec)
    return feedbax_load_latest_checkpoint(
        root,
        expected_run_spec=custody_spec,
        expected_phase_program=custody_spec.worker_execution.method_contract.phase_program,
        expected_slots=expected_slots,
    )


def _cs_checkpoint_barrier_name(spec: TrainingRunSpec) -> str:
    barriers = spec.worker_execution.method_contract.phase_program.checkpoint_barriers
    names = {barrier.name for barrier in barriers}
    if CS_BARRIER in names:
        return CS_BARRIER
    if CS_SUPERVISED_NATIVE_BARRIER in names:
        return CS_SUPERVISED_NATIVE_BARRIER
    if ADAPTIVE_EPSILON_BARRIER in names:
        return ADAPTIVE_EPSILON_BARRIER
    if POLICY_ADVERSARY_BARRIER in names:
        return POLICY_ADVERSARY_BARRIER
    raise ValueError(f"C&S TrainingRunSpec has no known checkpoint barrier: {sorted(names)!r}")


def _cs_checkpoint_barrier_phase(spec: TrainingRunSpec, barrier_name: str) -> str:
    barriers = spec.worker_execution.method_contract.phase_program.checkpoint_barriers
    for barrier in barriers:
        if barrier.name == barrier_name:
            return barrier.phase
    raise ValueError(f"unknown checkpoint barrier {barrier_name!r}")


def write_minimax_checkpoint_transaction(
    root: Path,
    *,
    training_spec: TrainingRunSpec,
    barrier_name: str,
    batch_idx: int,
    active_member_index: int,
    slots: Mapping[str, Any],
    status: str = "partial",
    population_member_ids: Mapping[str, Sequence[str]] | None = None,
) -> Any:
    """Write a Feedbax custody transaction for minimax training."""

    custody_spec = minimax_custody_training_spec(training_spec)
    phase = "warmup" if barrier_name == MINIMAX_WARMUP_BARRIER else "adversarial"
    coordinate = ProgressCoordinate(
        run_id=_run_id_from_spec(training_spec, prefix="rlrmp-minimax"),
        phase=phase,
        program_step=max(0, int(batch_idx) + 1),
        adversary_member=(
            None if int(active_member_index) < 0 else int(active_member_index)
        ),
        completed_barrier=barrier_name,
    )
    return write_checkpoint_transaction(
        root,
        run_spec=custody_spec,
        phase_program=custody_spec.worker_execution.method_contract.phase_program,
        barrier_name=barrier_name,
        coordinate=coordinate,
        slots=slots,
        status=status,
        population_member_ids=population_member_ids,
        history_availability={
            "adversary_losses": bool(slots.get("adversary_losses")),
            "controller_losses": bool(slots.get("controller_losses")),
            "adversary_indices": bool(slots.get("adversary_indices")),
            "warmup_history": slots.get("warmup_history") is not None,
        },
    )


def load_minimax_checkpoint_transaction(
    root: Path,
    *,
    training_spec: TrainingRunSpec,
    expected_slots: Mapping[str, Any],
    expected_population_member_ids: Mapping[str, Sequence[str]] | None = None,
) -> Any:
    """Load and validate the latest minimax custody transaction."""

    custody_spec = minimax_custody_training_spec(training_spec)
    return feedbax_load_latest_checkpoint(
        root,
        expected_run_spec=custody_spec,
        expected_phase_program=custody_spec.worker_execution.method_contract.phase_program,
        expected_slots=expected_slots,
        expected_population_member_ids=expected_population_member_ids,
    )


def spec_digests(training_spec: TrainingRunSpec) -> dict[str, str]:
    """Return the rlrmp-relevant digests persisted as checkpoint extension data."""

    payload = training_spec.model_dump(mode="json", exclude_none=True)
    effective_phase = training_spec.worker_execution.effective_phase.model_dump(
        mode="json",
        exclude_none=True,
    )
    phase_program = (
        training_spec.worker_execution.effective_phase.phase_program.model_dump(
            mode="json",
            exclude_none=True,
        )
    )
    return {
        "training_run_spec_sha256": _canonical_sha256(payload),
        "effective_phase_sha256": _canonical_sha256(effective_phase),
        "phase_program_sha256": _canonical_sha256(phase_program),
    }


def serialize_pytree_slot(value: Any) -> bytes:
    """Serialize a PyTree through Equinox leaf serialization for custody storage."""

    with tempfile.NamedTemporaryFile() as tmp:
        path = Path(tmp.name)
        eqx.tree_serialise_leaves(path, value)
        return path.with_suffix(".eqx").read_bytes()


def deserialize_pytree_slot(data: bytes, template: Any, *, slot: str) -> Any:
    """Deserialize an Equinox leaf-serialized custody slot through ``template``."""

    with tempfile.NamedTemporaryFile() as tmp:
        path = Path(tmp.name)
        path.with_suffix(".eqx").write_bytes(data)
        return eqx.tree_deserialise_leaves(path, template)


def _augment_training_spec(
    spec: TrainingRunSpec,
    *,
    extra_state_slots: Sequence[StateSlotSpec],
    barrier_slots: Mapping[str, Sequence[CheckpointSlotSpec]],
    consistency_mode: str,
    batch_progress: TrainingBatchProgressSpec | None = None,
) -> TrainingRunSpec:
    base = spec.model_copy(deep=True)
    worker = base.worker_execution
    contract = worker.method_contract.model_copy(deep=True)
    state_slots = _merge_state_slots(contract.state_slots, extra_state_slots)
    program = contract.phase_program.model_copy(deep=True)
    barriers: list[CheckpointBarrierSpec] = []
    for barrier in program.checkpoint_barriers:
        extras = barrier_slots.get(barrier.name, ())
        slots = _merge_checkpoint_slots(barrier.slots, extras)
        metadata = dict(barrier.metadata)
        metadata["consistency_mode"] = consistency_mode
        barriers.append(barrier.model_copy(update={"slots": slots, "metadata": metadata}))
    program = program.model_copy(
        update={
            "checkpoint_barriers": barriers,
            "batch_progress": batch_progress or program.batch_progress,
        }
    )
    contract = contract.model_copy(update={"state_slots": state_slots, "phase_program": program})
    effective = EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=program,
        consistency_predicate=derive_consistency_predicate(program),
        metadata=dict(worker.effective_phase.metadata),
    )
    return base.model_copy(
        update={
            "worker_execution": worker.model_copy(
                update={"method_contract": contract, "effective_phase": effective}
            )
        }
    )


def _merge_state_slots(
    base: Sequence[StateSlotSpec],
    extras: Sequence[StateSlotSpec],
) -> list[StateSlotSpec]:
    by_name = {slot.name: slot for slot in base}
    for slot in extras:
        by_name.setdefault(slot.name, slot)
    return list(by_name.values())


def _merge_checkpoint_slots(
    base: Sequence[CheckpointSlotSpec],
    extras: Sequence[CheckpointSlotSpec],
) -> list[CheckpointSlotSpec]:
    by_name = {slot.slot: slot for slot in base}
    for slot in extras:
        by_name[slot.slot] = slot
    return list(by_name.values())


def _run_id(run_spec: Mapping[str, Any], *, prefix: str) -> str:
    for key in ("run_id", "variant", "run_name", "issue"):
        value = run_spec.get(key)
        if value not in (None, ""):
            return f"{prefix}:{value}"
    return f"{prefix}:{_canonical_sha256(run_spec)[:12]}"


def _run_id_from_spec(spec: TrainingRunSpec, *, prefix: str) -> str:
    metadata = spec.artifacts.metadata if spec.artifacts is not None else {}
    for key in ("run_id", "variant", "run_name", "tracked_spec_dir"):
        value = metadata.get(key)
        if value not in (None, ""):
            return f"{prefix}:{value}"
    return f"{prefix}:{_canonical_sha256(spec.model_dump(mode='json', exclude_none=True))[:12]}"


def _canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(data).hexdigest()
