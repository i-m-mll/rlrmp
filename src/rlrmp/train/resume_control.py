"""Fail-closed resume launch controls for training CLIs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from feedbax.contracts.checkpoints import (
    BatchIndexedCheckpointLeafSpec,
    CheckpointContinuationRequest,
)
from feedbax.contracts.training import TrainingRunSpec


LAUNCH_CONTINUATION_PREFIX = "LAUNCH_CONTINUATION"

# These paths are derived from the real C&S supervised optimizer checkpoint
# topology, not inferred from a PyTree at resume time.  They are the three
# gradient diagnostics and three update diagnostics whose final axis is the
# global training-batch horizon for the vmapped C&S GRU executor.
CS_SUPERVISED_BATCH_INDEXED_CHECKPOINT_LEAVES = (
    BatchIndexedCheckpointLeafSpec(slot="optimizer", tree_path="/1"),
    BatchIndexedCheckpointLeafSpec(slot="optimizer", tree_path="/2"),
    BatchIndexedCheckpointLeafSpec(slot="optimizer", tree_path="/3"),
    BatchIndexedCheckpointLeafSpec(slot="optimizer", tree_path="/30"),
    BatchIndexedCheckpointLeafSpec(slot="optimizer", tree_path="/31"),
    BatchIndexedCheckpointLeafSpec(slot="optimizer", tree_path="/32"),
)


@dataclass(frozen=True)
class LaunchContinuation:
    """Resolved launch continuation contract for a training run."""

    resume: bool
    resume_source: str
    completed_batches: int
    stop_target_batches: int
    continuation_batches: int

    def format_line(self) -> str:
        """Return the stable one-line launch summary."""

        return (
            f"{LAUNCH_CONTINUATION_PREFIX} "
            f"resume_source={self.resume_source} "
            f"completed_batches={self.completed_batches} "
            f"stop_target_batches={self.stop_target_batches} "
            f"continuation_batches={self.continuation_batches}"
        )


def completed_batches_from_latest(latest_path: Path) -> int:
    """Read the authoritative completed-training total for a latest pointer.

    Feedbax's progress coordinate can count checkpoint/custody ordering rather
    than individual training batches.  When a pointer names a transaction
    manifest, that manifest's ``completed_training_batches`` is authoritative.
    Pointers predating transaction manifests retain the old coordinate fallback.
    """

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint latest pointer must be an object: {latest_path}")
    manifest_relative_path = payload.get("manifest_relative_path")
    if manifest_relative_path is not None:
        if not isinstance(manifest_relative_path, str) or not manifest_relative_path:
            raise ValueError(
                "checkpoint latest pointer has invalid manifest_relative_path: "
                f"{latest_path}"
            )
        relative_path = Path(manifest_relative_path)
        if relative_path.is_absolute():
            raise ValueError(
                "checkpoint latest pointer manifest_relative_path must be relative: "
                f"{latest_path}"
            )
        checkpoint_root = latest_path.parent.resolve()
        manifest_path = (checkpoint_root / relative_path).resolve()
        try:
            manifest_path.relative_to(checkpoint_root)
        except ValueError as exc:
            raise ValueError(
                "checkpoint latest pointer manifest_relative_path escapes checkpoint root: "
                f"{latest_path}"
            ) from exc
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                "checkpoint latest pointer references missing transaction manifest: "
                f"{manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"checkpoint transaction manifest must be an object: {manifest_path}")
        completed_batches = manifest.get("completed_training_batches")
        if isinstance(completed_batches, bool) or not isinstance(completed_batches, int):
            raise ValueError(
                "checkpoint transaction manifest lacks integer completed_training_batches: "
                f"{manifest_path}"
            )
        if completed_batches < 0:
            raise ValueError(
                "checkpoint transaction manifest has negative completed_training_batches: "
                f"{manifest_path}"
            )
        pointer_completed_batches = payload.get("completed_training_batches")
        if pointer_completed_batches is not None and pointer_completed_batches != completed_batches:
            raise ValueError(
                "checkpoint latest pointer completed_training_batches disagrees with "
                f"transaction manifest: pointer={pointer_completed_batches!r} "
                f"manifest={completed_batches!r}"
            )
        return completed_batches

    # Legacy pointers did not name transaction manifests. Preserve their
    # coordinate semantics only for compatibility; new custody pointers must
    # take their total from the referenced manifest above.
    coordinate = payload.get("completed_coordinate")
    if not isinstance(coordinate, dict) or "global_step" not in coordinate:
        raise ValueError(
            "checkpoint latest pointer lacks completed_coordinate.global_step: "
            f"{latest_path}"
        )
    return int(coordinate["global_step"])


def resolve_launch_continuation(
    *,
    checkpoint_root: Path,
    resume_requested: bool,
    allow_fresh_start: bool,
    stop_target_batches: int,
    completed_batches_from_latest: Callable[[Path], int] = completed_batches_from_latest,
) -> LaunchContinuation:
    """Resolve fail-closed resume/fresh-start semantics before training starts."""

    latest_path = checkpoint_root / "latest.json"
    if resume_requested and not latest_path.is_file():
        if not allow_fresh_start:
            raise FileNotFoundError(
                "--resume requested but no resumable checkpoint state exists at "
                f"{latest_path}; pass --allow-fresh-start to start from batch 0."
            )
        completed_batches = 0
        resume_source = "fresh-start-override"
        resume = False
    elif resume_requested:
        completed_batches = int(completed_batches_from_latest(latest_path))
        resume_source = str(latest_path)
        resume = True
    else:
        completed_batches = 0
        resume_source = "fresh-start"
        resume = False

    continuation_batches = int(stop_target_batches) - int(completed_batches)
    if continuation_batches <= 0:
        raise ValueError(
            "non-positive launch continuation: "
            f"resume_source={resume_source} completed_batches={completed_batches} "
            f"stop_target_batches={int(stop_target_batches)} "
            f"continuation_batches={continuation_batches}"
        )
    return LaunchContinuation(
        resume=resume,
        resume_source=resume_source,
        completed_batches=completed_batches,
        stop_target_batches=int(stop_target_batches),
        continuation_batches=continuation_batches,
    )


def emit_launch_continuation(
    continuation: LaunchContinuation,
    *,
    logger: logging.Logger,
) -> None:
    """Emit the launch continuation summary to stdout and the configured logger."""

    line = continuation.format_line()
    print(line, flush=True)
    logger.info(line)


def attach_cs_supervised_checkpoint_continuation(
    training_spec: TrainingRunSpec,
    continuation: LaunchContinuation,
) -> TrainingRunSpec:
    """Declare a C&S total-length resume on the governed Feedbax spec.

    The target slot template is built at the row's requested total horizon.
    Feedbax then preserves the source prefix and supplies only the target
    template's new tail for the explicitly declared diagnostic leaves.
    """

    if not continuation.resume:
        return training_spec
    return declare_cs_supervised_checkpoint_continuation(
        training_spec,
        source_completed_batches=continuation.completed_batches,
        target_total_batches=continuation.stop_target_batches,
    )


def declare_cs_supervised_checkpoint_continuation(
    training_spec: TrainingRunSpec,
    *,
    source_completed_batches: int,
    target_total_batches: int,
) -> TrainingRunSpec:
    """Attach the durable C&S continuation declaration during row authoring."""

    request = CheckpointContinuationRequest(
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
        batch_indexed_leaves=list(CS_SUPERVISED_BATCH_INDEXED_CHECKPOINT_LEAVES),
    )
    checkpoint_progress = training_spec.checkpoint_progress.model_copy(
        update={"continuation": request}
    )
    return training_spec.model_copy(update={"checkpoint_progress": checkpoint_progress})
