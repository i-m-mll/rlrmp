# Variance Analysis — 6-Cell Anti-Anticipation Matrix

## Setup
- SISU: 0.5
- Perturbation: 0 (clean reach)
- Validation trials: 8 center-out reach directions
- Per-replicate metric: mean peak forward velocity across 8 directions
- Variance ratio: SD(peak_vel across replicates) / mean(peak_vel across replicates)
- Decision criterion: variance ratio < 0.50 (prior best = 0.76 for gru__jerk baseline)

## Results Table

| Cell | Display Name | Mean PV (m/s) | SD PV (m/s) | Variance Ratio | Hold Drift (mm) | TTP (steps) | Per-rep PV |
|------|------|---------|---------|---------|---------|---------|---------|
| gru__jerk | Control (jerk only) | 0.5769 | 0.0741 | 0.129 | 19.836 ± 4.215 | 61.9 | 0.5074 / 0.5688 / 0.6628 / 0.5031 / 0.6422 |
| gru__jerk_motor_pre | Pre-go motor mask | 0.8404 | 0.0616 | 0.073 | 5.524 ± 0.525 | 61.6 | 0.8176 / 0.7710 / 0.8659 / 0.9328 / 0.8147 |
| gru__jerk_smooth_high | Hidden smoothness 1e2 | 1.0713 | 0.0063 | 0.006 | 35.746 ± 1.473 | 47.4 | 1.0711 / 1.0650 / 1.0654 / 1.0761 / 1.0791 |
| gru__jerk_motor_smooth_combo | Pre-go + smooth (combo) | 1.2216 | 0.0081 | 0.007 | 4.624 ± 0.266 | 52.6 | 1.2085 / 1.2309 / 1.2226 / 1.2225 / 1.2232 |
| gru__jerk_loss_v_terminal | Var A: terminal vel | 0.5979 | 0.0842 | 0.141 | 19.120 ± 4.716 | 62.6 | 0.5440 / 0.6679 / 0.5863 / 0.6968 / 0.4947 |
| gru__jerk_loss_historical | Var B: historical shape | 0.6186 | 0.1177 | 0.190 | 15.999 ± 2.177 | 66.4 | 0.6276 / 0.6505 / 0.4419 / 0.6034 / 0.7697 |

## Metric Note: SD/mean vs RMSE ratio

The task prompt states the prior best (GRU/jerk baseline) was VR = 0.76. However, the
`results/efc4d68/RESULTS.md` records a **pairwise RMSE ratio**
(within-cell RMSE / nearest-neighbor across-cell RMSE = 0.758) for the equivalent GRU/jerk cell,
using a *different variance metric* (pairwise profile RMSE, not SD/mean of peak velocity scalars).

This analysis uses **SD(peak_vel across replicates) / mean(peak_vel across replicates)**, which is a
simpler scalar metric. The two metrics are not directly comparable. The control cell (gru__jerk)
achieves VR = 0.129 on this SD/mean definition — well below 0.5. The threshold < 0.5 for this metric
was set based on the task description but was originally calibrated against the RMSE-ratio definition.
The user should verify whether the 0.5 threshold translates sensibly to this new metric definition.

## Decision

Threshold: variance ratio < 0.5 (SD/mean)
Prior best (gru__jerk RMSE ratio): 0.758 — note metric difference above

**6 cell(s) met the threshold:**

- **Control (jerk only)** (`gru__jerk`): VR = 0.129, mean PV = 0.5769 m/s
- **Pre-go motor mask** (`gru__jerk_motor_pre`): VR = 0.073, mean PV = 0.8404 m/s
- **Hidden smoothness 1e2** (`gru__jerk_smooth_high`): VR = 0.006, mean PV = 1.0713 m/s
- **Pre-go + smooth (combo)** (`gru__jerk_motor_smooth_combo`): VR = 0.007, mean PV = 1.2216 m/s
- **Var A: terminal vel** (`gru__jerk_loss_v_terminal`): VR = 0.141, mean PV = 0.5979 m/s
- **Var B: historical shape** (`gru__jerk_loss_historical`): VR = 0.190, mean PV = 0.6186 m/s

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

- `figures/peak_velocity_distributions/` — Violin/strip plot, one replicate per data point
- `figures/forward_velocity_profiles/` — Forward velocity time series per cell
- `figures/hold_drift_profiles/` — Pre-go forward position (anticipation drift)
