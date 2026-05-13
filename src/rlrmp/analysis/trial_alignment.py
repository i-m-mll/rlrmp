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
    flat_idx = idx_per_trial.reshape(-1, n_trials)
    flat_aligned = aligned.reshape(-1, n_trials, n_aligned_steps)
    flat_dest = dest.reshape(-1, n_trials, n_steps)
    flat_in_range = in_range.reshape(-1, n_trials, n_steps)

    for outer in range(flat_profile.shape[0]):
        for t in range(n_trials):
            mask = flat_in_range[outer, t]
            flat_aligned[outer, t, flat_dest[outer, t, mask]] = flat_profile[outer, t, mask]

    return aligned, center


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
