"""Grep-friendly per-batch training-progress logging.

Remote training runs are launched under ``nohup`` and monitored by polling the
log file (see ``feedbax/scripts/deploy/poll_run.sh`` and the RunPod runbook in
the project ``CLAUDE.md``). Between the JIT-compilation message and the
completion sentinel the host emits no structured progress, so a monitor cannot
tell a live run from a hung one while checkpoints silently advance.

This module provides a single stable line format that monitors can grep for the
last batch seen, plus small helpers for the cadence and for wiring the line into
legacy Feedbax trainer-style ``batch_callbacks``
hook.

Line contract (consumed by ``poll_run.sh`` — do not reorder ``phase`` /
``batch`` without updating that consumer)::

    BATCH phase=warmup batch=42/1000 loss=3.21 elapsed=12.3s

- The line always starts with the literal token ``BATCH`` at a word boundary.
- ``phase`` is a short stable identifier (``warmup``, ``adversarial``, ...).
- ``batch`` is ``<index>/<total>`` (zero-based index, ``total`` batch count).
- ``loss`` and ``elapsed`` are optional ``key=value`` fields; omitted when the
  caller cannot supply them without an extra device->host sync.

The helpers here are pure host-side string formatting and a callback factory.
They never read a JAX array, so they add no per-step device->host synchronisation
(the known ``loss_update`` ``float()`` sync cost; see project memory). Callers
that already have a host-side loss scalar may pass it; callers inside feedbax's
warmup loop log batch index + elapsed only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

#: Literal token every progress line starts with. Monitors grep for this.
BATCH_LINE_TOKEN = "BATCH"

#: Default cadence: emit a line every Nth batch for normal-size runs.
DEFAULT_LOG_EVERY = 10

#: At or below this batch count a run is a "smoke" run; log every batch.
SMOKE_TOTAL_THRESHOLD = 50


def batch_log_every(total: int, *, default: int = DEFAULT_LOG_EVERY) -> int:
    """Return the per-batch logging interval for a run of ``total`` batches.

    Smoke-size runs (``total <= SMOKE_TOTAL_THRESHOLD``) log every batch so a
    short remote smoke test produces visible progress; larger runs log every
    ``default`` batches to keep the log readable.

    Args:
        total: Total number of batches in the phase.
        default: Interval to use for non-smoke runs.

    Returns:
        A positive integer interval ``n``; emit a line when
        ``batch % n == 0`` (the last batch should always be logged separately).
    """
    if total <= SMOKE_TOTAL_THRESHOLD:
        return 1
    return max(1, default)


def should_log_batch(batch: int, total: int, *, every: int) -> bool:
    """Return whether batch ``batch`` (zero-based) should be logged.

    Logs on the cadence and always on the final batch so the log ends with a
    line that reports the true last batch.

    Args:
        batch: Zero-based batch index.
        total: Total number of batches in the phase.
        every: Interval from :func:`batch_log_every`.
    """
    return batch % every == 0 or batch == total - 1


def format_batch_line(
    phase: str,
    batch: int,
    total: int,
    *,
    loss: Optional[float] = None,
    elapsed: Optional[float] = None,
    **extra: object,
) -> str:
    """Format one grep-friendly batch-progress line.

    Args:
        phase: Short stable phase identifier (e.g. ``"warmup"``,
            ``"adversarial"``). Must not contain spaces.
        batch: Zero-based batch index.
        total: Total number of batches in the phase.
        loss: Optional host-side loss scalar. Omitted from the line if ``None``.
        elapsed: Optional wall-clock seconds since the phase started. Omitted
            if ``None``.
        **extra: Additional ``key=value`` fields appended verbatim (values are
            ``str()``-formatted). Use for short scalars such as ``adv_loss``.

    Returns:
        A single line beginning with ``BATCH``, with no trailing newline.
    """
    if " " in phase:
        raise ValueError(f"phase must not contain spaces; got {phase!r}")
    parts = [BATCH_LINE_TOKEN, f"phase={phase}", f"batch={batch}/{total}"]
    if loss is not None:
        parts.append(f"loss={loss:.4g}")
    for key, value in extra.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed:.1f}s")
    return " ".join(parts)


def make_batch_log_callbacks(
    phase: str,
    total: int,
    *,
    start_batch: int = 0,
    every: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Mapping[int, Sequence[Callable[[], None]]]:
    """Build a legacy ``batch_callbacks`` mapping.

    feedbax invokes ``batch_callbacks[batch]`` with no arguments on the host
    side of its training loop (outside the JIT-compiled step), so the callbacks
    can log without forcing an extra device->host sync. The loss is not
    available to a no-arg callback, so warmup lines report batch index and
    elapsed wall-clock only — enough for a monitor to report "last batch seen".

    The returned mapping keys are absolute batch indices over
    ``[start_batch, total)`` selected by the cadence, plus the final batch.

    Args:
        phase: Phase identifier passed to :func:`format_batch_line`.
        total: Total number of batches in the phase.
        start_batch: First batch index feedbax will iterate (``idx_start``).
        every: Cadence; defaults to :func:`batch_log_every`.
        logger: Logger to emit lines on; defaults to this module's logger.
        clock: Monotonic clock; injectable for tests.

    Returns:
        Mapping ``{batch_index: [callback]}`` suitable for the
        ``batch_callbacks`` keyword of the retired Feedbax trainer path.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    step = every if every is not None else batch_log_every(total)
    started = clock()

    callbacks: dict[int, list[Callable[[], None]]] = {}
    for batch in range(start_batch, total):
        if not should_log_batch(batch, total, every=step):
            continue

        def _emit(batch: int = batch) -> None:
            log.info(format_batch_line(phase, batch, total, elapsed=clock() - started))

        callbacks.setdefault(batch, []).append(_emit)
    return callbacks


def make_executor_batch_log_callback(
    phase_totals: Mapping[str, int],
    *,
    every: Mapping[str, int] | None = None,
    batch_index: Callable[[Mapping[str, Any], int, int], int] | None = None,
    logger: Optional[logging.Logger] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[Mapping[str, Any]], None]:
    """Build a Feedbax executor ``progress_callback`` that emits BATCH lines.

    Feedbax's native executor reports copied progress-coordinate events on the
    host. This adapter maps those coordinates onto the existing grep-friendly
    line contract without reading metric arrays or otherwise forcing a new
    device-to-host transfer.

    Args:
        phase_totals: Total conceptual batch count per executor phase.
        every: Optional per-phase logging cadence override.
        batch_index: Optional resolver called as ``(coordinate, seen, total)``.
            The default treats each executor progress event as one batch.
        logger: Logger to emit lines on; defaults to this module's logger.
        clock: Monotonic clock; injectable for tests.

    Returns:
        A callback suitable for Feedbax ``execute_training_run_spec``.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    totals = {phase: int(total) for phase, total in phase_totals.items()}
    intervals = dict(every or {})
    started = clock()
    seen_by_phase: dict[str, int] = {}

    def _callback(event: Mapping[str, Any]) -> None:
        raw_coordinate = event.get("coordinate", event)
        if not isinstance(raw_coordinate, Mapping):
            return
        phase = raw_coordinate.get("phase")
        if not isinstance(phase, str):
            return
        total = totals.get(phase, 0)
        if total <= 0:
            return

        seen = seen_by_phase.get(phase, 0)
        seen_by_phase[phase] = seen + 1
        raw_batch = batch_index(raw_coordinate, seen, total) if batch_index is not None else seen
        batch = min(max(int(raw_batch), 0), total - 1)
        step = intervals.get(phase, batch_log_every(total))
        if not should_log_batch(batch, total, every=step):
            return
        log.info(format_batch_line(phase, batch, total, elapsed=clock() - started))

    return _callback
