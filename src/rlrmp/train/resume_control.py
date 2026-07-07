"""Fail-closed resume launch controls for training CLIs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


LAUNCH_CONTINUATION_PREFIX = "LAUNCH_CONTINUATION"


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
    """Read the global completed-batch index from a Feedbax latest pointer."""

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
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
