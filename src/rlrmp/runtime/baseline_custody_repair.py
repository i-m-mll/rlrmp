"""Immutable migration of a verified C&S legacy checkpoint into Feedbax custody."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.tree as jt
import equinox as eqx
from feedbax.contracts.worker import ProgressCoordinate
from rlrmp.runtime.checkpoint_custody import (
    cs_custody_training_spec,
    load_cs_checkpoint_transaction,
    write_governed_checkpoint_transaction,
)
from rlrmp.train.executor.checkpoints import load_latest_checkpoint
from rlrmp.train.executor.slots import MODEL, OPTIMIZER, PRNG


class BaselineCustodyRepairError(ValueError):
    """Raised when a historical checkpoint cannot be repaired truthfully."""


@dataclass(frozen=True)
class BaselineCustodyRepairResult:
    """Validated details of one retry-owned custody source."""

    checkpoint_root: Path
    transaction_id: str
    completed_training_batches: int
    program_step: int
    barrier_visit_ordinal: int


def repair_baseline_custody_source(
    *,
    repair_spec_path: Path,
    source_run_spec: Mapping[str, Any],
    source_templates: Mapping[str, Any],
    legacy_model_template: Any,
    legacy_optimizer_state_template: Any,
    target_root: Path | None = None,
    repo_root: Path,
) -> BaselineCustodyRepairResult:
    """Create and validate a new custody source from verified legacy bytes.

    The original legacy directory is only read.  The repair record must name a
    previously absent target root, pin the legacy source hashes, and establish
    batch progress through the C&S ``completed_batches`` custody slot.
    """

    repair = _read_object(repair_spec_path, label="repair record")
    source = _required_mapping(repair, "source")
    verified = _required_mapping(repair, "verified_progress")
    target = _required_mapping(repair, "target")
    source_root = _resolve_repo_path(repo_root, source.get("legacy_checkpoint_dir"))
    checkpoint_root = source_root.parent
    destination = (
        _resolve_repo_path(repo_root, target.get("checkpoint_root"))
        if target_root is None
        else target_root
    )
    if destination.exists():
        raise BaselineCustodyRepairError(
            f"repair target must be new and empty: {destination}"
        )
    if not bool(target.get("must_be_new")):
        raise BaselineCustodyRepairError("repair record must require a new target custody root")

    completed_batches = _required_nonnegative_int(
        verified, "completed_training_batches", label="verified_progress"
    )
    program_step = _required_nonnegative_int(verified, "phase_step", label="verified_progress")
    barrier_visit_ordinal = _required_nonnegative_int(
        verified, "barrier_visit_ordinal", label="verified_progress"
    )
    _verify_legacy_hashes(source_root, _required_mapping(source, "legacy_files_sha256"))
    _verify_progress_evidence(
        legacy_checkpoint_dir=source_root,
        source=source,
        expected_completed_batches=completed_batches,
        repo_root=repo_root,
    )

    legacy = load_latest_checkpoint(
        checkpoint_root,
        model_template=legacy_model_template,
        optimizer_state_template=legacy_optimizer_state_template,
        history_template=None,
        run_spec=None,
    )
    if int(legacy.completed_batches) != completed_batches:
        raise BaselineCustodyRepairError(
            "legacy checkpoint completed_batches disagrees with repair record: "
            f"checkpoint={legacy.completed_batches} record={completed_batches}"
        )
    custody_spec = cs_custody_training_spec(source_run_spec)
    program = custody_spec.worker_execution.method_contract.phase_program
    barrier = next(
        (
            barrier
            for barrier in program.checkpoint_barriers
            if barrier.name == "after_train_chunk"
        ),
        None,
    )
    if barrier is None:
        raise BaselineCustodyRepairError("C&S custody program lacks after_train_chunk barrier")
    slots: dict[str, Any] = {
        # Feedbax continuation transforms declared PyTree leaves.  Store the
        # native C&S slot trees directly; wrapping them in serialized bytes
        # would make paths such as /1 invisible to the generic fork contract.
        MODEL: tuple(jt.leaves(eqx.filter(legacy.model, eqx.is_array))),
        OPTIMIZER: tuple(jt.leaves(legacy.optimizer_state)),
        PRNG: legacy.key,
        "completed_batches": jnp.asarray(completed_batches, dtype=jnp.int32),
    }
    if legacy.history is not None:
        slots["history"] = legacy.history
    coordinate = ProgressCoordinate(
        run_id=f"{repair['repair_id']}-custody",
        phase=barrier.phase,
        program_step=program_step,
        completed_barrier=barrier.name,
    )
    result = write_governed_checkpoint_transaction(
        destination,
        run_spec=custody_spec,
        phase_program=program,
        barrier_name=barrier.name,
        coordinate=coordinate,
        slots=slots,
        status="final",
        completed_training_batches=completed_batches,
        metadata={
            "barrier_visit_ordinal": barrier_visit_ordinal,
            "custody_repair": {
                "repair_record": str(repair_spec_path.relative_to(repo_root)),
                "legacy_checkpoint_dir": str(source_root.relative_to(repo_root)),
                "legacy_files_sha256": dict(_required_mapping(source, "legacy_files_sha256")),
            },
        },
    )
    loaded = load_cs_checkpoint_transaction(
        destination,
        run_spec=source_run_spec,
        expected_slots=slots,
    )
    if loaded.manifest.completed_training_batches != completed_batches:
        raise BaselineCustodyRepairError(
            "validated repaired source has wrong completed_training_batches: "
            f"{loaded.manifest.completed_training_batches}"
        )
    if loaded.manifest.completed_coordinate.program_step != program_step:
        raise BaselineCustodyRepairError("validated repaired source has wrong program_step")
    if loaded.manifest.metadata.get("barrier_visit_ordinal") != barrier_visit_ordinal:
        raise BaselineCustodyRepairError(
            "validated repaired source has wrong barrier_visit_ordinal"
        )
    return BaselineCustodyRepairResult(
        checkpoint_root=destination,
        transaction_id=result.manifest.transaction_id,
        completed_training_batches=completed_batches,
        program_step=program_step,
        barrier_visit_ordinal=barrier_visit_ordinal,
    )


def _verify_legacy_hashes(source_root: Path, expected_hashes: Mapping[str, Any]) -> None:
    for name, expected in expected_hashes.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise BaselineCustodyRepairError(f"legacy hash for {name!r} must be sha256")
        path = source_root / str(name)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise BaselineCustodyRepairError(
                f"legacy source hash mismatch path={path}: expected={expected} observed={observed}"
            )


def _verify_progress_evidence(
    *,
    legacy_checkpoint_dir: Path,
    source: Mapping[str, Any],
    expected_completed_batches: int,
    repo_root: Path,
) -> None:
    index = _read_object(
        _resolve_repo_path(repo_root, source.get("legacy_checkpoint_index")),
        label="legacy checkpoint index",
    )
    summary = _read_object(
        _resolve_repo_path(repo_root, source.get("training_summary")),
        label="training summary",
    )
    metadata = _read_object(
        legacy_checkpoint_dir / "metadata.json",
        label="legacy checkpoint metadata",
    )
    values = {
        "legacy_checkpoint.metadata.completed_batches": metadata.get("completed_batches"),
        "legacy_checkpoint_index.completed_batches": index.get("completed_batches"),
        "training_summary.completed_batches": summary.get("completed_batches"),
    }
    for path, value in values.items():
        if value != expected_completed_batches:
            raise BaselineCustodyRepairError(
                f"repair evidence mismatch {path}={value!r} expected={expected_completed_batches}"
            )


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselineCustodyRepairError(f"{label} is missing: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineCustodyRepairError(f"{label} must be a JSON object: {path}")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BaselineCustodyRepairError(f"repair record lacks object {key!r}")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], key: str, *, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BaselineCustodyRepairError(f"{label}.{key} must be a non-negative integer")
    return value


def _resolve_repo_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise BaselineCustodyRepairError("repair record path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BaselineCustodyRepairError(f"repair record path escapes repo: {value!r}")
    # ``_artifacts`` is a sanctioned shared worktree link.  Validate the
    # authored path lexically, then retain it without resolving that link.
    return repo_root / relative
