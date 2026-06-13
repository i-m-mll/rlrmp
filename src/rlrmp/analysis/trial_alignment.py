"""Trial-to-event alignment helpers for go-cue-locked analyses.

The `centerout_delayed_reach` task randomises each trial's go-cue index, so
profiles indexed by absolute trial time (`v_fwd[..., trial, step]`) cannot be
averaged across trials without first re-locking them to each trial's go cue.

The helpers in this module operate on profiles of shape ``(..., n_trials,
n_steps)`` (the leading axes can include replicate, perturbation scale, etc.)
plus a per-trial integer alignment index ``idx`` of shape ``(..., n_trials)``
or ``(n_trials,)`` (broadcast over leading axes). They produce aligned profiles
of shape ``(..., n_trials, n_aligned_steps)`` where column index ``center``
(default: the largest value of ``idx``) corresponds to the alignment event for
every trial.

All three helpers are pure numpy. JAX-array inputs are coerced via
``np.asarray`` — the analysis pipelines that call these helpers already run
``np.array(...)`` on their JAX outputs before reaching here, and the trial
axis is dynamic in size which composes poorly with ``jax.jit``.

Bug: 06f7faf — added after discovering the headline-RMSE numbers in 3702f54 /
f47abb1 / 2bc95fd had been computed in absolute trial time, smearing the
go cue across ~150 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def align_trials(
    profile: np.ndarray,
    idx: np.ndarray,
    pad: float = np.nan,
    center: int | None = None,
    n_aligned_steps: int | None = None,
) -> Tuple[np.ndarray, int]:
    """Re-index a per-trial profile so a per-trial event lands at a fixed column.

    For each trial ``t`` with alignment index ``idx[t]``, sample ``i`` from the
    raw profile is mapped to aligned column ``center + (i - idx[t])`` and
    out-of-range columns are filled with ``pad``.

    Args:
        profile: Shape ``(..., n_trials, n_steps)``. Leading dims are kept;
            common cases are ``(n_rep, n_trials, n_steps)`` and ``(n_trials,
            n_steps)``.
        idx: Shape ``(..., n_trials)`` broadcastable with ``profile``'s leading
            and trial axes (typically ``(n_trials,)``). The integer index of
            the alignment event in each trial.
        pad: Fill value for samples that fall outside ``[0, n_steps)`` after
            shifting. Default ``nan`` so downstream reductions can use
            ``np.nanmean`` / ``np.nanstd`` to ignore padded samples.
        center: Column in the aligned profile where the event lands. If
            ``None`` (default), uses ``int(max(idx))`` so every trial keeps all
            of its pre-event samples without truncation.
        n_aligned_steps: Length of the aligned profile. If ``None`` (default),
            uses ``center + (n_steps - min(idx))`` so every trial keeps all of
            its post-event samples too.

    Returns:
        A tuple ``(aligned, center)``:
            ``aligned``: shape ``(..., n_trials, n_aligned_steps)`` with the
                same leading axes as ``profile``.
            ``center``: the alignment column (so callers can build a time axis
                like ``(np.arange(n_aligned_steps) - center) * dt``).
    """
    profile = np.asarray(profile)
    idx = np.asarray(idx).astype(np.int64)

    if profile.ndim < 2:
        raise ValueError(
            f"profile must have at least 2 dims (trials, steps); got shape {profile.shape}"
        )

    n_steps = profile.shape[-1]
    n_trials = profile.shape[-2]

    if idx.ndim == 1 and idx.shape[0] != n_trials:
        raise ValueError(
            f"idx shape {idx.shape} incompatible with profile trial axis {n_trials}"
        )
    if idx.ndim > 1 and idx.shape[-1] != n_trials:
        raise ValueError(
            f"idx trailing axis {idx.shape[-1]} must equal n_trials {n_trials}"
        )

    if center is None:
        center = int(idx.max())
    if n_aligned_steps is None:
        max_post = n_steps - int(idx.min())  # most samples any trial has post-event
        n_aligned_steps = center + max_post
    assert n_aligned_steps is not None  # always set above; assertion narrows type

    leading_shape = profile.shape[:-2]
    aligned = np.full(
        leading_shape + (n_trials, n_aligned_steps),
        fill_value=pad,
        dtype=profile.dtype if np.issubdtype(profile.dtype, np.floating) else np.float64,
    )

    # Broadcast idx to (n_trials,) per (leading..., trial). If idx already has
    # leading dims, we assume they match.
    if idx.ndim == 1:
        idx_per_trial = np.broadcast_to(idx, leading_shape + (n_trials,))
    else:
        idx_per_trial = np.broadcast_to(idx, leading_shape + (n_trials,))

    # Build the destination column index for each (leading..., trial, step).
    raw_steps = np.arange(n_steps)
    # dest[..., trial, step] = center + (step - idx[..., trial])
    dest = center + (raw_steps[None, :] - idx_per_trial[..., :, None])  # (..., n_trials, n_steps)

    in_range = (dest >= 0) & (dest < n_aligned_steps)

    # Flat-index assignment per trial. The leading dims can be arbitrary, so we
    # iterate over them with np.ndindex (number of leading combos is typically
    # small: n_replicates ≤ ~10 in our use cases).
    flat_profile = profile.reshape(-1, n_trials, n_steps)
    flat_aligned = aligned.reshape(-1, n_trials, n_aligned_steps)
    flat_dest = dest.reshape(-1, n_trials, n_steps)
    flat_in_range = in_range.reshape(-1, n_trials, n_steps)

    for outer in range(flat_profile.shape[0]):
        for t in range(n_trials):
            mask = flat_in_range[outer, t]
            flat_aligned[outer, t, flat_dest[outer, t, mask]] = flat_profile[outer, t, mask]

    return aligned, center


@dataclass(frozen=True)
class TrialTiming:
    """Task timing metadata needed by post-run diagnostics.

    ``go_index`` is always per trial. For non-delayed reaches it is all zeros,
    so callers can use the same movement-window code path for delayed and
    non-delayed tasks.
    """

    is_delayed: bool
    go_index: np.ndarray
    movement_horizon_steps: int
    n_time_steps: int

    @property
    def movement_end_index(self) -> np.ndarray:
        """Return the exclusive movement-window end index for each trial."""

        return self.go_index + int(self.movement_horizon_steps)

    def to_json(self) -> dict[str, object]:
        """Return JSON-compatible timing metadata."""

        return {
            "is_delayed": bool(self.is_delayed),
            "go_index_min": int(np.min(self.go_index)) if self.go_index.size else None,
            "go_index_max": int(np.max(self.go_index)) if self.go_index.size else None,
            "movement_horizon_steps": int(self.movement_horizon_steps),
            "movement_end_index_min": (
                int(np.min(self.movement_end_index)) if self.go_index.size else None
            ),
            "movement_end_index_max": (
                int(np.max(self.movement_end_index)) if self.go_index.size else None
            ),
            "n_time_steps": int(self.n_time_steps),
            "time_basis": (
                "go_cue_aligned_canonical_movement_window"
                if self.is_delayed
                else "absolute_trial_time"
            ),
        }


def trial_timing_from_specs(
    trial_specs: object,
    *,
    n_time_steps: int | None = None,
    movement_horizon_steps: int | None = None,
) -> TrialTiming:
    """Infer diagnostic timing metadata from a Feedbax ``TaskTrialSpec``.

    Delayed reach trial specs store epoch boundaries in
    ``trial_specs.timeline.epoch_bounds``. The movement epoch starts at the
    penultimate boundary, matching the full-Q/R/Q_f loss implementation. For
    non-delayed specs, diagnostics use a zero go index and the whole time axis.
    """

    inferred_n_time = int(n_time_steps) if n_time_steps is not None else infer_trial_n_time(
        trial_specs
    )
    n_trials = infer_trial_count(trial_specs)
    epoch_bounds = getattr(getattr(trial_specs, "timeline", None), "epoch_bounds", None)
    if epoch_bounds is None:
        go_index = np.zeros((n_trials,), dtype=np.int64)
        return TrialTiming(
            is_delayed=False,
            go_index=go_index,
            movement_horizon_steps=int(movement_horizon_steps or inferred_n_time),
            n_time_steps=inferred_n_time,
        )

    bounds = np.asarray(epoch_bounds)
    if bounds.ndim == 1:
        bounds = np.broadcast_to(bounds[None, :], (n_trials, bounds.shape[0]))
    if bounds.shape[-1] < 3:
        go_index = np.zeros((bounds.shape[0],), dtype=np.int64)
        return TrialTiming(
            is_delayed=False,
            go_index=go_index,
            movement_horizon_steps=int(movement_horizon_steps or inferred_n_time),
            n_time_steps=inferred_n_time,
        )

    go_index = np.asarray(bounds[..., -2], dtype=np.int64).reshape(-1)
    if go_index.shape[0] != n_trials:
        go_index = np.broadcast_to(go_index[:1], (n_trials,)).astype(np.int64)
    horizon = int(
        movement_horizon_steps
        if movement_horizon_steps is not None
        else max(0, inferred_n_time - int(np.max(go_index)))
    )
    if horizon < 1:
        raise ValueError(f"movement_horizon_steps must be positive; got {horizon}")
    if np.any(go_index < 0) or np.any(go_index + horizon > inferred_n_time):
        raise ValueError(
            "movement window is outside the rollout time axis: "
            f"go=[{int(go_index.min())}, {int(go_index.max())}], "
            f"horizon={horizon}, n_time_steps={inferred_n_time}"
        )
    return TrialTiming(
        is_delayed=bool(np.any(go_index > 0)),
        go_index=go_index,
        movement_horizon_steps=horizon,
        n_time_steps=inferred_n_time,
    )


def canonical_movement_horizon_from_metadata(
    metadata: dict[str, object],
    *,
    default: int | None = None,
) -> int | None:
    """Return the recorded canonical C&S movement horizon when present."""

    paths = (
        ("task_timing", "movement_window", "cs_horizon_steps"),
        (
            "task_timing",
            "delayed_reach",
            "movement_epoch",
            "cs_schedule_horizon_steps",
        ),
        ("loss_summary", "time_indexing", "canonical_movement_horizon_steps"),
        (
            "loss_summary",
            "delayed_reach",
            "movement_epoch",
            "cs_schedule_horizon_steps",
        ),
        (
            "game_card",
            "delayed_reach_projection",
            "canonical_cs_movement_horizon_steps",
        ),
        ("game_card", "horizon_steps"),
    )
    for path in paths:
        value: object = metadata
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return int(value)
    return default


def infer_trial_count(trial_specs: object) -> int:
    """Infer the leading trial count from common ``TaskTrialSpec`` leaves."""

    for init_state in getattr(trial_specs, "inits", {}).values():
        count = _leading_count(init_state)
        if count is not None:
            return count
    for target_spec in getattr(trial_specs, "targets", {}).values():
        count = _leading_count(getattr(target_spec, "value", target_spec))
        if count is not None:
            return count
    inputs = getattr(trial_specs, "inputs", None)
    values = inputs.values() if isinstance(inputs, dict) else (inputs,)
    for value in values:
        count = _leading_count(value)
        if count is not None:
            return count
    epoch_bounds = getattr(getattr(trial_specs, "timeline", None), "epoch_bounds", None)
    if epoch_bounds is not None:
        bounds = np.asarray(epoch_bounds)
        if bounds.ndim >= 2:
            return int(bounds.shape[0])
    return 1


def infer_trial_n_time(trial_specs: object, *, minimum: int = 1) -> int:
    """Infer the rollout time-axis length from common trial-spec inputs."""

    inputs = getattr(trial_specs, "inputs", None)
    values = inputs.values() if isinstance(inputs, dict) else (inputs,)
    for value in values:
        n_time = _time_count(value)
        if n_time is not None:
            return max(int(n_time), int(minimum))
    for target_spec in getattr(trial_specs, "targets", {}).values():
        n_time = _time_count(getattr(target_spec, "value", target_spec))
        if n_time is not None:
            return max(int(n_time), int(minimum))
    return int(minimum)


def take_per_trial_time_window(
    values: np.ndarray,
    start_index: np.ndarray,
    length: int,
    *,
    trial_axis: int,
    time_axis: int,
) -> np.ndarray:
    """Take a fixed-length time window with a per-trial start index.

    Args:
        values: Array containing a trial axis and a time axis.
        start_index: Integer array of shape ``(n_trials,)``.
        length: Number of time samples to keep.
        trial_axis: Axis corresponding to trials.
        time_axis: Axis corresponding to time.

    Returns:
        An array with the same axis order as ``values`` except that the time
        axis length is ``length``.
    """

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError(f"values must have at least two dimensions; got {array.shape}")
    trial_axis = _normalize_axis(trial_axis, array.ndim)
    time_axis = _normalize_axis(time_axis, array.ndim)
    if trial_axis == time_axis:
        raise ValueError("trial_axis and time_axis must be distinct")

    starts = np.asarray(start_index, dtype=np.int64).reshape(-1)
    n_trials = int(array.shape[trial_axis])
    n_time = int(array.shape[time_axis])
    if starts.shape[0] != n_trials:
        raise ValueError(
            f"start_index has {starts.shape[0]} entries but trial axis has {n_trials}"
        )
    if length < 1:
        raise ValueError(f"length must be positive; got {length}")
    if np.any(starts < 0) or np.any(starts + length > n_time):
        raise ValueError(
            "requested per-trial window is outside the time axis: "
            f"starts=[{int(starts.min())}, {int(starts.max())}], "
            f"length={length}, n_time={n_time}"
        )

    moved = np.moveaxis(array, (trial_axis, time_axis), (0, 1))
    windowed = np.stack(
        [moved[trial, starts[trial] : starts[trial] + length, ...] for trial in range(n_trials)],
        axis=0,
    )
    return np.moveaxis(windowed, (0, 1), (trial_axis, time_axis))


def _leading_count(value: object) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    position = getattr(value, "pos", None)
    if position is not None:
        return int(position.shape[0])
    velocity = getattr(value, "vel", None)
    if velocity is not None:
        return int(velocity.shape[0])
    return None


def _time_count(value: object) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-2])
    position = getattr(value, "pos", None)
    if position is not None and len(position.shape) >= 2:
        return int(position.shape[-2])
    return None


def _normalize_axis(axis: int, ndim: int) -> int:
    return axis + ndim if axis < 0 else axis


def trim_to_full_support(
    aligned: np.ndarray,
    trial_axis: int = -2,
    step_axis: int = -1,
    min_coverage: float = 1.0,
) -> Tuple[np.ndarray, slice]:
    """Trim aligned profiles to columns with (thresholded) trial coverage.

    After ``align_trials`` re-locks each trial to a per-trial event, columns
    near the array edges contain NaN for some trials and not others. Reducing
    over those columns with ``np.nanmean`` / ``np.nanstd`` produces choppy
    edges as different trials drop in/out across columns.

    This helper finds the largest contiguous window of columns whose per-column
    non-NaN coverage (fraction of trials contributing a real sample) is at
    least ``min_coverage``, and returns the trimmed array together with the
    ``slice`` applied to ``step_axis``. Callers should apply the same slice to
    any companion time axis or x-coords.

    Note: NaN-coverage is computed by reducing across *all* axes except
    ``step_axis`` (so trials, replicates, perturbation scales, etc. all count
    toward the per-column coverage). For the typical
    ``(n_rep, n_trials, n_aligned_steps)`` layout that gives exactly the
    "fraction of (replicate, trial) samples contributing" semantics.

    Args:
        aligned: Shape ``(..., n_trials, n_aligned_steps)`` (any number of
            leading axes; ``trial_axis`` and ``step_axis`` need not be the last
            two — defaults match ``align_trials`` output).
        trial_axis: Axis indexing trials. Kept for explicit-naming clarity; the
            implementation treats every non-``step_axis`` axis as a
            sample axis, so this argument is informational only.
        step_axis: Axis indexing aligned time steps. Default ``-1``.
        min_coverage: Minimum fraction (in ``[0.0, 1.0]``) of samples that
            must contribute a non-NaN value per column. ``1.0`` (default)
            requires every sample to contribute — i.e. trim to the strict
            full-support window.

    Returns:
        Tuple ``(trimmed, sl)``:
            ``trimmed``: ``aligned[..., sl]`` (sliced along ``step_axis``).
            ``sl``: the ``slice`` object applied — useful for trimming a
                companion time axis (``t[sl]``) so plotted x-coords match.

    Raises:
        ValueError: if no columns meet ``min_coverage`` (the whole array is
            below threshold).
    """
    del trial_axis  # informational; not used by the implementation
    aligned = np.asarray(aligned)
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError(f"min_coverage must be in [0, 1]; got {min_coverage}")

    n_steps = aligned.shape[step_axis]
    # Count non-NaN samples per column, reducing across every axis except step_axis.
    if np.issubdtype(aligned.dtype, np.floating):
        valid = ~np.isnan(aligned)
    else:
        # Non-float arrays have no NaN concept — every column is fully supported.
        valid = np.ones_like(aligned, dtype=bool)

    # Move step_axis to the back, flatten the rest, count per column.
    moved = np.moveaxis(valid, step_axis, -1)
    flat = moved.reshape(-1, n_steps)  # (n_samples, n_steps)
    n_samples = flat.shape[0]
    per_col_count = flat.sum(axis=0)
    coverage = per_col_count / max(n_samples, 1)

    mask = coverage >= min_coverage
    if not mask.any():
        raise ValueError(
            f"No columns meet min_coverage={min_coverage} (max coverage="
            f"{coverage.max():.3f})"
        )

    # Largest *contiguous* fully-supported window. The aligned-then-padded
    # geometry guarantees that NaN padding only appears at the leading and
    # trailing edges, so the True-region is itself contiguous; if it isn't
    # (e.g. internal NaNs from upstream), we pick the [first_true, last_true]
    # span and warn would-be callers via the explicit slice they get back.
    where = np.nonzero(mask)[0]
    first, last = int(where[0]), int(where[-1])
    sl = slice(first, last + 1)

    # Apply the slice along step_axis.
    idx = [slice(None)] * aligned.ndim
    idx[step_axis] = sl
    trimmed = aligned[tuple(idx)]
    return trimmed, sl


def replicate_mean_curves(
    aligned: np.ndarray,
    trim: bool | float = True,
) -> np.ndarray | Tuple[np.ndarray, slice]:
    """Per-replicate, trial-averaged curve from an already-aligned profile.

    Purpose: produce the curve set used to compute *inter-replicate variance*
    (e.g. ``within_cell_vel_rmse`` and the RMSE ratios). Averaging happens
    across the trial axis using ``np.nanmean``, so the padded columns from
    ``align_trials`` (where some trials have no sample) do not bias the mean.

    If ``trim`` is truthy, the aligned array is first trimmed to columns with
    full (or thresholded) per-column trial coverage via
    ``trim_to_full_support``; the returned slice is then surfaced as part of
    the result so the caller can clip their companion time axis.

    Args:
        aligned: Shape ``(..., n_trials, n_aligned_steps)`` as returned by
            ``align_trials``.
        trim: If ``True`` (default) trim to the strict full-support window
            (``min_coverage=1.0``) before averaging. If a ``float`` in
            ``[0, 1]``, pass it as ``min_coverage`` (e.g. ``0.5`` keeps any
            column with ≥50% trial coverage). If ``False``, no trim — the
            return value reverts to a single array (legacy behaviour).

    Returns:
        If ``trim`` is truthy: ``(curves, sl)`` where ``curves`` has shape
        ``(..., n_kept_steps)`` and ``sl`` is the trim slice. If ``trim`` is
        ``False``, returns the curves array alone (legacy shape).
    """
    aligned = np.asarray(aligned)
    if aligned.ndim < 2:
        raise ValueError(
            f"aligned must have at least 2 dims (trials, steps); got {aligned.shape}"
        )

    if trim is False:
        return np.nanmean(aligned, axis=-2)

    min_coverage = 1.0 if trim is True else float(trim)
    trimmed, sl = trim_to_full_support(aligned, min_coverage=min_coverage)
    return np.nanmean(trimmed, axis=-2), sl


def pooled_trial_mean_with_band(
    aligned: np.ndarray,
    band: str = "sd",
    trim: bool | float = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, slice]:
    """Pool all (replicate, trial) samples and reduce to a single curve + band.

    Purpose: produce the curve used in velocity / hold-drift *plots* where the
    intent is to display *inter-trial variability across the full pooled
    population*, not pre-averaged within-replicate curves. The pooled curve is
    a ``nanmean`` over the flattened ``(replicate, trial)`` axis; the band is
    computed across the same flattened axis.

    If ``trim`` is truthy, the aligned array is first trimmed to columns with
    full (or thresholded) per-column sample coverage via
    ``trim_to_full_support``; the returned slice is surfaced so the caller can
    clip their companion time axis.

    Args:
        aligned: Shape ``(..., n_trials, n_aligned_steps)`` with leading axes
            that all index distinct samples (e.g. ``(n_rep, n_trials, n_steps)``
            flattens to ``n_rep * n_trials`` independent samples).
        band: One of ``"sd"``, ``"sem"``, or ``"none"``. ``"sem"`` divides
            ``nanstd`` by ``sqrt(n_effective)`` per step. ``"none"`` returns a
            zero array of the right shape.
        trim: If ``True`` (default) trim to the strict full-support window
            before reducing. If a ``float`` in ``[0, 1]``, pass it as
            ``min_coverage``. If ``False``, no trim — the return value reverts
            to the legacy ``(mean, lower, upper)`` triple.

    Returns:
        If ``trim`` is truthy: ``(mean, lower, upper, sl)`` where each curve
        has shape ``(n_kept_steps,)`` and ``sl`` is the trim slice. If ``trim``
        is ``False``, returns ``(mean, lower, upper)`` (legacy shape) over the
        full input column range.
    """
    aligned = np.asarray(aligned)
    if aligned.ndim < 2:
        raise ValueError(
            f"aligned must have at least 2 dims; got shape {aligned.shape}"
        )

    sl: slice | None = None
    if trim is not False:
        min_coverage = 1.0 if trim is True else float(trim)
        aligned, sl = trim_to_full_support(aligned, min_coverage=min_coverage)

    # Flatten everything except the time axis.
    n_steps = aligned.shape[-1]
    flat = aligned.reshape(-1, n_steps)  # (n_pooled, n_steps)

    mean = np.nanmean(flat, axis=0)

    if band == "none":
        spread = np.zeros_like(mean)
    elif band == "sd":
        spread = np.nanstd(flat, axis=0, ddof=1)
    elif band == "sem":
        # Effective N per step: number of non-NaN samples
        n_eff = np.sum(~np.isnan(flat), axis=0)
        n_eff = np.maximum(n_eff, 1)
        sd = np.nanstd(flat, axis=0, ddof=1)
        spread = sd / np.sqrt(n_eff)
    else:
        raise ValueError(f"band must be 'sd', 'sem', or 'none'; got {band!r}")

    if sl is None:
        return mean, mean - spread, mean + spread
    return mean, mean - spread, mean + spread, sl
