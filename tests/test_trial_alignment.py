"""Regression tests for `rlrmp.analysis.trial_alignment`.

These guard the go-cue alignment fix (Bug: 06f7faf). The original failure was
that headline RMSE metrics on 3702f54 / f47abb1 / 2bc95fd were computed by
``v_fwd.mean(axis=trial)`` *before* aligning per-trial go indices, smearing the
go cue across ~150 ms and inflating both within-cell and across-cell pairwise
RMSE in correlated ways.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrmp.analysis.trial_alignment import (
    align_trials,
    pooled_trial_mean_with_band,
    replicate_mean_curves,
)


# ---------------------------------------------------------------------------
# align_trials: pulse-at-go test (the regression case)
# ---------------------------------------------------------------------------


def _make_pulse_profile(go_indices, n_steps=40, post_amp=1.0):
    """Build a synthetic profile with a one-step pulse at each trial's go index.

    Returns a (n_trials, n_steps) array where row t has zeros everywhere
    except at column go_indices[t], which equals ``post_amp``.
    """
    n_trials = len(go_indices)
    profile = np.zeros((n_trials, n_steps), dtype=np.float64)
    for t, g in enumerate(go_indices):
        profile[t, g] = post_amp
    return profile


def test_align_trials_locks_post_go_pulse_to_center():
    """A pulse placed at each trial's go index must align to a single column."""
    go_idx = np.array([10, 14, 18, 22, 26])
    profile = _make_pulse_profile(go_idx)
    aligned, center = align_trials(profile, go_idx)

    # Every trial should have its pulse at the center column, zeros elsewhere
    # (in-range), and NaN padding outside.
    assert center == int(go_idx.max()) == 26
    for t in range(len(go_idx)):
        # Pulse landed at center
        assert aligned[t, center] == pytest.approx(1.0), (
            f"trial {t} (go={go_idx[t]}) did not land at center column {center}"
        )
        # No spurious values at the *un*shifted column (unless go == center)
        if go_idx[t] != center:
            assert aligned[t, go_idx[t]] != 1.0 or np.isnan(aligned[t, go_idx[t]])


def test_align_trials_mean_recovers_pulse_at_center():
    """The trial-averaged aligned profile must peak at the center column.

    This is the property the production code was missing: pre-alignment
    averaging spreads the pulse across [min(go_idx), max(go_idx)] (about
    150 ms in the real case); post-alignment averaging concentrates it at
    a single column.
    """
    go_idx = np.array([10, 14, 18, 22, 26])
    profile = _make_pulse_profile(go_idx)
    aligned, center = align_trials(profile, go_idx)

    # nanmean over trials
    mean_curve = np.nanmean(aligned, axis=0)

    peak_col = int(np.nanargmax(mean_curve))
    assert peak_col == center, f"peak at column {peak_col}, expected center {center}"

    # The peak should be the only non-padded column where every trial
    # contributed a non-zero — but in our pulse construction every trial
    # contributes amplitude 1.0 at that column, so the mean is exactly 1.0.
    assert mean_curve[center] == pytest.approx(1.0)


def test_align_trials_replicate_axis():
    """align_trials must handle the (n_rep, n_trials, n_steps) shape used in
    production analyses."""
    go_idx = np.array([5, 8, 11])
    n_rep = 3
    n_steps = 20

    profile = np.zeros((n_rep, len(go_idx), n_steps))
    for r in range(n_rep):
        for t, g in enumerate(go_idx):
            profile[r, t, g] = float(r + 1)  # different amplitude per replicate

    aligned, center = align_trials(profile, go_idx)
    assert aligned.shape[:2] == (n_rep, len(go_idx))
    assert center == 11

    # Each (rep, trial) pulse should land at center
    for r in range(n_rep):
        for t in range(len(go_idx)):
            assert aligned[r, t, center] == pytest.approx(float(r + 1))


def test_align_trials_padding_is_nan_by_default():
    """Samples shifted out of range must be NaN (so nanmean ignores them)."""
    go_idx = np.array([0, 5])  # First trial: no pre-go samples; second: 5 pre-go
    n_steps = 8
    profile = np.ones((2, n_steps))

    aligned, center = align_trials(profile, go_idx)

    # Trial 0 (go=0) has no pre-go data — columns 0..center-1 must be NaN
    assert np.all(np.isnan(aligned[0, :center]))
    # Trial 0 (go=0) has post-go data — columns center..center+(n_steps-1) are 1.0
    assert np.all(aligned[0, center:center + n_steps] == 1.0)
    # Trial 1 (go=5) has 5 pre-go samples
    assert np.all(aligned[1, center - 5:center + (n_steps - 5)] == 1.0)


# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------


def test_replicate_mean_curves_uses_nanmean():
    """replicate_mean_curves must ignore NaN-padded columns."""
    # Aligned shape (n_rep=2, n_trials=3, n_steps=4)
    # Two trials have valid data, one is all NaN in one column.
    aligned = np.array([
        [[1.0, 2.0, np.nan, 4.0],
         [1.0, 2.0, 3.0, 4.0],
         [1.0, np.nan, 3.0, 4.0]],
        [[2.0, np.nan, 4.0, 5.0],
         [2.0, 3.0, 4.0, 5.0],
         [np.nan, 3.0, 4.0, 5.0]],
    ])

    curves = replicate_mean_curves(aligned)
    assert curves.shape == (2, 4)
    # Replicate 0 column 2: nanmean of [nan, 3, 3] = 3.0
    assert curves[0, 2] == pytest.approx(3.0)
    # Replicate 1 column 0: nanmean of [2, 2, nan] = 2.0
    assert curves[1, 0] == pytest.approx(2.0)


def test_pooled_trial_mean_with_band_pools_across_reps_and_trials():
    """The pooled aggregator must flatten (rep, trial) and reduce across that."""
    # Shape (n_rep=2, n_trials=3, n_steps=2): pooled population = 6 samples per step
    aligned = np.array([
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    ])
    mean, lo, hi = pooled_trial_mean_with_band(aligned, band="sd")

    assert mean.shape == (2,)
    # At step 0: pooled values are [1,2,3,1,2,3] → mean=2.0
    assert mean[0] == pytest.approx(2.0)
    # SD with ddof=1 on [1,2,3,1,2,3] = sqrt(0.8) ≈ 0.8944
    assert (hi[0] - mean[0]) == pytest.approx(np.sqrt(0.8))
    # Symmetric band
    assert (mean - lo) == pytest.approx(hi - mean)


def test_pooled_band_modes():
    aligned = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    _, lo_none, hi_none = pooled_trial_mean_with_band(aligned, band="none")
    assert np.allclose(lo_none, hi_none)

    _, lo_sd, hi_sd = pooled_trial_mean_with_band(aligned, band="sd")
    _, lo_sem, hi_sem = pooled_trial_mean_with_band(aligned, band="sem")
    # SEM band is narrower than SD band (n_eff > 1)
    assert (hi_sem[0] - lo_sem[0]) < (hi_sd[0] - lo_sd[0])


# ---------------------------------------------------------------------------
# integration: full pipeline matches expected behaviour
# ---------------------------------------------------------------------------


def test_full_pipeline_recovers_post_go_dynamics():
    """The buggy pipeline (mean-then-vline) versus the fixed pipeline (align,
    then nanmean).

    Construct a profile where every trial has a 10-step ramp starting at its
    own go index. After alignment, the trial-averaged curve must rise sharply
    at the center (go cue); without alignment, the rise is spread out.
    """
    rng = np.random.default_rng(0)
    go_idx = rng.integers(low=5, high=20, size=12)
    n_steps = 40
    n_trials = len(go_idx)

    # Each trial: zero before its go, then a linear ramp 0..1 for 10 steps,
    # then constant 1.
    profile = np.zeros((1, n_trials, n_steps))  # (n_rep=1, n_trials, n_steps)
    for t, g in enumerate(go_idx):
        ramp_len = min(10, n_steps - g)
        profile[0, t, g:g + ramp_len] = np.linspace(0, 1, ramp_len, endpoint=False) + 1.0 / ramp_len
        if g + ramp_len < n_steps:
            profile[0, t, g + ramp_len:] = 1.0

    # Buggy pipeline: average across trials in absolute time
    buggy_curve = profile.mean(axis=1).squeeze()  # (n_steps,)
    # At the *mean* go index, the buggy curve should be substantially below 0.5
    mean_go = int(np.round(go_idx.mean()))
    # At the mean_go time, a mix of trials are "already moving" — non-zero,
    # but the rise is smeared.

    aligned, center = align_trials(profile, go_idx)
    fixed_curve = replicate_mean_curves(aligned).squeeze()  # (n_aligned_steps,)

    # The aligned curve at center should be close to zero (just at go-cue
    # onset every trial is starting); 10 steps later it should be ~= 1.
    assert fixed_curve[center] < 0.3, (
        f"aligned curve at center={center} should be near zero, got {fixed_curve[center]}"
    )
    assert fixed_curve[center + 9] > 0.85, (
        f"aligned curve at center+9 should be near 1.0, got {fixed_curve[center + 9]}"
    )

    # And the buggy curve at the same effective offset (mean_go + 9) is
    # systematically lower than the fixed curve at center+9
    if mean_go + 9 < buggy_curve.size:
        assert buggy_curve[mean_go + 9] < fixed_curve[center + 9], (
            "buggy curve should under-estimate the post-go peak relative to aligned"
        )
