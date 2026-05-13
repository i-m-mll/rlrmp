# Variance Analysis — 6-Cell Anti-Anticipation Matrix

> **Corrected after go-cue alignment fix (Bug: 06f7faf).** The primary
> `vel_rmse_ratio` and the secondary `pos_rmse_ratio` were originally
> computed by averaging trial-mean profiles in *absolute trial time*
> before the within/across-cell pairwise RMSE step. `centerout`'s
> target-on duration is randomized per trial, so this smeared the go
> cue across ~150 ms and produced biased RMSE values. Velocity / hold
> drift figures were similarly drawn in absolute time. The fix re-locks
> each trial to its own go cue before the trial-axis collapse.
>
> **Effect on conclusions:** the winner set is *unchanged* —
> `gru__jerk_smooth_high` (post-fix: 0.040; pre-fix: 0.037) and
> `gru__jerk_motor_smooth_combo` (post-fix: 0.042; pre-fix: 0.036) still
> cleanly beat the < 0.5 threshold. `gru__jerk_motor_pre` shifts from
> 0.461 to 0.552 (was: just beats the threshold; now: just misses it) —
> this is the one cell whose winner-status flips after correction. The
> substantive ordering and the relative gap between the two clear
> winners and the rest is preserved. The auxiliary scalar metrics
> (peak vel, hold drift, TTP, CV peak vel) are computed per-trial before
> the trial-axis collapse and are not affected.

## Setup
- SISU: 0.5
- Perturbation: 0 (clean reach)
- Validation trials: 8 center-out reach directions

## Metrics

**Primary: velocity-RMSE ratio** — within-cell mean pairwise RMSE on the
forward-velocity profile / nearest-neighbor across-cell mean pairwise RMSE.
Matches the prior `baseline_jerk_vrnn_matrix` metric. Prior best (GRU/jerk): 0.758.
Decision threshold: < 0.50.

**Secondary: position-RMSE ratio** — same computation on the forward-position profile.
May be more sensitive to hold-drift differences.

**Auxiliary: CV (SD/mean of peak vel)** — scalar summary of replicate spread on
peak velocity. Was incorrectly used as the primary metric in earlier writeups.
Reported here for completeness; does NOT drive the decision.

## Results Table — post-fix

Post-fix values after go-cue alignment correction (Bug: 06f7faf). The
auxiliary scalar metrics (Mean PV, SD PV, Hold Drift, TTP) use per-trial
go-cue indexing in their computation and are unchanged.

| Cell | Display Name | Vel-RMSE ratio (PRIMARY) | Pos-RMSE ratio | CV (peak vel) | Mean PV (m/s) | SD PV (m/s) | Hold Drift (mm) | TTP (steps) |
|------|------|---------|---------|---------|---------|---------|---------|---------|
| gru__jerk | Control (jerk only) | 1.054 | 1.047 | 0.129 | 0.5769 | 0.0741 | 19.836 ± 4.215 | 61.9 |
| gru__jerk_motor_pre | Pre-go motor mask | 0.552 | 0.581 | 0.073 | 0.8404 | 0.0616 | 5.524 ± 0.525 | 61.6 |
| gru__jerk_smooth_high | Hidden smoothness 1e2 | 0.040 * | 0.038 | 0.006 | 1.0713 | 0.0063 | 35.746 ± 1.473 | 47.4 |
| gru__jerk_motor_smooth_combo | Pre-go + smooth (combo) | 0.042 * | 0.063 | 0.007 | 1.2216 | 0.0081 | 4.624 ± 0.266 | 52.6 |
| gru__jerk_loss_v_terminal | Var A: terminal vel | 0.989 | 1.104 | 0.141 | 0.5979 | 0.0842 | 19.120 ± 4.716 | 62.6 |
| gru__jerk_loss_historical | Var B: historical shape | 1.133 | 1.282 | 0.190 | 0.6186 | 0.1177 | 15.999 ± 2.177 | 66.4 |

\* = beats primary threshold (vel-RMSE ratio < 0.50).

**Conclusion-flip note:** `gru__jerk_motor_pre` moved from 0.461 (just
under the 0.5 threshold) to 0.552 (just over). The motor-pre cell was
originally listed as one of three "winners"; after correction it
narrowly fails the threshold. The two solid winners
(`gru__jerk_smooth_high` and `gru__jerk_motor_smooth_combo`) are
unaffected.

## RMSE Detail (within vs across) — post-fix

Post-fix values after go-cue alignment correction (Bug: 06f7faf):

| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) |
|------|---------|---------|
| gru__jerk | 0.1415 | 0.1343 |
| gru__jerk_motor_pre | 0.0992 | 0.1798 |
| gru__jerk_smooth_high | 0.0069 | 0.1755 |
| gru__jerk_motor_smooth_combo | 0.0074 | 0.1755 |
| gru__jerk_loss_v_terminal | 0.1327 | 0.1343 |
| gru__jerk_loss_historical | 0.1566 | 0.1383 |

## Pre-fix table (deprecated)

Original numbers before the go-cue alignment fix, preserved for traceability:

| Cell | Vel-RMSE ratio (deprecated) | Pos-RMSE ratio (deprecated) |
|------|---------|---------|
| gru__jerk | 1.129 | 1.020 |
| gru__jerk_motor_pre | 0.461 | 0.424 |
| gru__jerk_smooth_high | 0.037 | 0.037 |
| gru__jerk_motor_smooth_combo | 0.036 | 0.055 |
| gru__jerk_loss_v_terminal | 0.882 | 1.162 |
| gru__jerk_loss_historical | 1.131 | 1.249 |

## Decision

**Primary threshold**: vel-RMSE ratio < 0.5
Prior best (GRU/jerk, baseline_jerk_vrnn_matrix): 0.758

**2 cell(s) met the threshold (post-fix):**

- **Hidden smoothness 1e2** (`gru__jerk_smooth_high`): vel-RMSE-ratio = 0.040, CV = 0.006, mean PV = 1.0713 m/s
- **Pre-go + smooth (combo)** (`gru__jerk_motor_smooth_combo`): vel-RMSE-ratio = 0.042, CV = 0.007, mean PV = 1.2216 m/s

**One cell that beat the threshold pre-fix narrowly misses post-fix:**

- **Pre-go motor mask** (`gru__jerk_motor_pre`): vel-RMSE-ratio = 0.552
  (post-fix), 0.461 (pre-fix). The motor-pre cell was the marginal
  third winner pre-fix and is now narrowly above 0.5. Treat as "close
  to threshold; not a clear winner".

## Anticipation (Hold Drift)

Hold drift = max forward displacement (toward target, in mm) before the go cue.
Positive = anticipatory movement.

- Control (jerk only): 19.836 ± 4.215 mm
- Pre-go motor mask: 5.524 ± 0.525 mm
- Hidden smoothness 1e2: 35.746 ± 1.473 mm
- Pre-go + smooth (combo): 4.624 ± 0.266 mm
- Var A: terminal vel: 19.120 ± 4.716 mm
- Var B: historical shape: 15.999 ± 2.177 mm

## Figures

- `figures/rmse_ratio_comparison/` — Bar chart of RMSE ratios (PRIMARY)
- `figures/peak_velocity_distributions/` — Violin/strip plot (CV annotated, auxiliary)
- `figures/forward_velocity_profiles/` — Forward velocity time series per cell
- `figures/hold_drift_profiles/` — Pre-go forward position (anticipation drift)
