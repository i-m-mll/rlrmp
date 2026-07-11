"""Fail-closed resume launch controls for training CLIs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training import load_checkpoint_custody_documents


LAUNCH_CONTINUATION_PREFIX = "LAUNCH_CONTINUATION"

@dataclass(frozen=True)
class LaunchContinuation:
    """Resolved launch continuation contract for a training run."""

    resume: bool
    resume_source: str
    completed_batches: int
    stop_target_batches: int
    continuation_batches: int
    source_target_batches: int | None = None

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

    Feedbax's phase-progress coordinate can count chunks or other executor
    progress, rather than individual training batches.  A custody source used
    for C&S continuation must therefore name a transaction manifest with an
    explicit ``completed_training_batches`` total.  Coordinates are deliberately
    never a fallback for batch arithmetic.
    """

    documents = load_checkpoint_custody_documents(latest_path.parent)
    latest = documents.latest_pointer.document
    manifest = documents.manifest.document
    manifest_completed_batches = manifest.completed_training_batches
    completed_batches = manifest_completed_batches
    continuation = manifest.metadata.get("checkpoint_continuation")
    if (
        manifest.metadata.get("checkpoint_continuation_applied") is True
        and isinstance(continuation, dict)
    ):
        source_completed = continuation.get("source_completed_batches")
        if isinstance(source_completed, int) and not isinstance(source_completed, bool):
            completed_batches = source_completed
    if completed_batches is None:
        raise ValueError(
            "checkpoint transaction manifest lacks explicit completed_training_batches: "
            f"{documents.manifest_path}"
        )
    if completed_batches < 0:
        raise ValueError(
            "checkpoint transaction manifest has negative completed_training_batches: "
            f"{documents.manifest_path}"
        )
    if (
        latest.completed_training_batches is not None
        and latest.completed_training_batches != manifest_completed_batches
    ):
        raise ValueError(
            "checkpoint latest pointer completed_training_batches disagrees with "
            f"transaction manifest: pointer={latest.completed_training_batches!r} "
            f"manifest={completed_batches!r}"
        )
    return completed_batches


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
        source_target_batches = _source_target_batches_from_latest(latest_path)
        resume_source = str(latest_path)
        resume = True
    else:
        completed_batches = 0
        resume_source = "fresh-start"
        resume = False
        source_target_batches = None

    if not resume_requested or not latest_path.is_file():
        source_target_batches = None

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
        source_target_batches=source_target_batches,
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

    Feedbax records the resumed work as a new segment. Batch histories are
    marked by ``BatchHistory`` in the checkpoint slot tree and remain local to
    that segment; lineage readers concatenate them when a whole-run view is
    needed.
    """

    if not continuation.resume:
        return training_spec
    return declare_cs_supervised_checkpoint_continuation(
        training_spec,
        source_completed_batches=continuation.completed_batches,
        target_total_batches=continuation.stop_target_batches,
    )


def _source_target_batches_from_latest(latest_path: Path) -> int:
    """Read the source run's declared total horizon from typed custody documents."""

    documents = load_checkpoint_custody_documents(latest_path.parent)
    projection = documents.manifest.document.run_contract_binding.canonical_projection
    try:
        value = projection["training_run_spec"]["training_config"]["n_batches"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "checkpoint run-contract binding lacks source training_config.n_batches: "
            f"{documents.manifest_path}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "checkpoint source training_config.n_batches must be a non-negative integer: "
            f"{documents.manifest_path}"
        )
    return value


def declare_cs_supervised_checkpoint_continuation(
    training_spec: TrainingRunSpec,
    *,
    source_completed_batches: int,
    target_total_batches: int,
) -> TrainingRunSpec:
    """Attach the durable C&S continuation declaration during row authoring."""

    request = CheckpointContinuationRequest(
        source_completed_batches=source_completed_batches,
        additional_batches=target_total_batches - source_completed_batches,
    )
    checkpoint_progress = training_spec.checkpoint_progress.model_copy(
        update={"continuation": request}
    )
    return training_spec.model_copy(update={"checkpoint_progress": checkpoint_progress})
