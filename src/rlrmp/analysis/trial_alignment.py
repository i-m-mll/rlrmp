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


def replicate_mean_curves(
    aligned: np.ndarray,
) -> np.ndarray:
    """Per-replicate, trial-averaged curve from an already-aligned profile.

    Purpose: produce the curve set used to compute *inter-replicate variance*
    (e.g. ``within_cell_vel_rmse`` and the RMSE ratios). Averaging happens
    across the trial axis using ``np.nanmean``, so the padded columns from
    ``align_trials`` (where some trials have no sample) do not bias the mean.

    Args:
        aligned: Shape ``(..., n_trials, n_aligned_steps)`` as returned by
            ``align_trials``.

    Returns:
        Shape ``(..., n_aligned_steps)`` with the trial axis collapsed via
        ``nanmean``.
    """
    aligned = np.asarray(aligned)
    if aligned.ndim < 2:
        raise ValueError(
            f"aligned must have at least 2 dims (trials, steps); got {aligned.shape}"
        )
    return np.nanmean(aligned, axis=-2)


def pooled_trial_mean_with_band(
    aligned: np.ndarray,
    band: str = "sd",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool all (replicate, trial) samples and reduce to a single curve + band.

    Purpose: produce the curve used in velocity / hold-drift *plots* where the
    intent is to display *inter-trial variability across the full pooled
    population*, not pre-averaged within-replicate curves. The pooled curve is
    a ``nanmean`` over the flattened ``(replicate, trial)`` axis; the band is
    computed across the same flattened axis.

    Args:
        aligned: Shape ``(..., n_trials, n_aligned_steps)`` with leading axes
            that all index distinct samples (e.g. ``(n_rep, n_trials, n_steps)``
            flattens to ``n_rep * n_trials`` independent samples).
        band: One of ``"sd"``, ``"sem"``, or ``"none"``. ``"sem"`` divides
            ``nanstd`` by ``sqrt(n_effective)`` per step. ``"none"`` returns a
            zero array of the right shape.

    Returns:
        Tuple ``(mean, lower, upper)`` each of shape ``(n_aligned_steps,)``.
        ``lower`` and ``upper`` are ``mean - band`` and ``mean + band``
        respectively.
    """
    aligned = np.asarray(aligned)
    if aligned.ndim < 2:
        raise ValueError(
            f"aligned must have at least 2 dims; got shape {aligned.shape}"
        )

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

    return mean, mean - spread, mean + spread
