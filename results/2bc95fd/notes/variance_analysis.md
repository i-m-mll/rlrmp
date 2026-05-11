# Variance Analysis — 6-Cell Anti-Anticipation Matrix

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

## Results Table

| Cell | Display Name | Vel-RMSE ratio (PRIMARY) | Pos-RMSE ratio | CV (peak vel) | Mean PV (m/s) | SD PV (m/s) | Hold Drift (mm) | TTP (steps) |
|------|------|---------|---------|---------|---------|---------|---------|---------|
| gru__jerk | Control (jerk only) | 1.129 | 1.020 | 0.129 | 0.5769 | 0.0741 | 19.836 ± 4.215 | 61.9 |
| gru__jerk_motor_pre | Pre-go motor mask | 0.461 * | 0.424 | 0.073 | 0.8404 | 0.0616 | 5.524 ± 0.525 | 61.6 |
| gru__jerk_smooth_high | Hidden smoothness 1e2 | 0.037 * | 0.037 | 0.006 | 1.0713 | 0.0063 | 35.746 ± 1.473 | 47.4 |
| gru__jerk_motor_smooth_combo | Pre-go + smooth (combo) | 0.036 * | 0.055 | 0.007 | 1.2216 | 0.0081 | 4.624 ± 0.266 | 52.6 |
| gru__jerk_loss_v_terminal | Var A: terminal vel | 0.882 | 1.162 | 0.141 | 0.5979 | 0.0842 | 19.120 ± 4.716 | 62.6 |
| gru__jerk_loss_historical | Var B: historical shape | 1.131 | 1.249 | 0.190 | 0.6186 | 0.1177 | 15.999 ± 2.177 | 66.4 |

\* = beats primary threshold (vel-RMSE ratio < 0.50).

## RMSE Detail (within vs across)

| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) | Pos within-RMSE (m) | Pos nearest-across-RMSE (m) |
|------|---------|---------|---------|---------|
| gru__jerk | 0.1301 | 0.1152 | 0.0524 | 0.0514 |
| gru__jerk_motor_pre | 0.0734 | 0.1594 | 0.0173 | 0.0409 |
| gru__jerk_smooth_high | 0.0071 | 0.1889 | 0.0017 | 0.0464 |
| gru__jerk_motor_smooth_combo | 0.0068 | 0.1889 | 0.0026 | 0.0464 |
| gru__jerk_loss_v_terminal | 0.1016 | 0.1152 | 0.0475 | 0.0409 |
| gru__jerk_loss_historical | 0.1322 | 0.1168 | 0.0549 | 0.0440 |

## Decision

**Primary threshold**: vel-RMSE ratio < 0.5
Prior best (GRU/jerk, baseline_jerk_vrnn_matrix): 0.758

**3 cell(s) met the threshold:**

- **Pre-go motor mask** (`gru__jerk_motor_pre`): vel-RMSE-ratio = 0.461, CV = 0.073, mean PV = 0.8404 m/s
- **Hidden smoothness 1e2** (`gru__jerk_smooth_high`): vel-RMSE-ratio = 0.037, CV = 0.006, mean PV = 1.0713 m/s
- **Pre-go + smooth (combo)** (`gru__jerk_motor_smooth_combo`): vel-RMSE-ratio = 0.036, CV = 0.007, mean PV = 1.2216 m/s

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
