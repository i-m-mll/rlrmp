# Variance Analysis — Lit-Replication 6-Cell Matrix (f47abb1)

## Setup

This matrix tests whether faithful replication of the Chaisanguanthum & Shenoy 2019 (C&S)
loss schedule produces models with systematically better velocity-RMSE ratios and lower
inter-replicate variance than the current production loss.

Two design dimensions are crossed:
1. **Jerk regulariser** (`nn_output_jerk`): on (1e5, cells lit__\*\_jerk) vs off (0.0,
   cells lit__\*\_nojerk). Shahbazi et al. 2025 Eq. 1 used jerk; C&S 2019 did not.
   This axis tests whether the jerk term interacts with the position schedule.
2. **Position schedule**: flat (cells lit__flat\_\*), post-go `(t/T)^6` (cells
   lit__post\_\*), full-trial `(t/T)^6` on both hold and running (cells lit__full\_\*).
   The power-law schedule concentrates ~98% of position-error weight in the last 30% of
   the trial, matching C&S Eq. 15.

### Corrected hold-penalty bug (vs 2bc95fd)

The prior 6-cell matrix (2bc95fd) was run with `--effector-hold-pos 10.0` and
`--effector-hold-vel 10.0`, but the hold-term construction silently failed for
`center_out_delayed_reach` because the task-type check used `==` instead of `in`. Fixed in
commit `22153e4`. This run uses `effector_hold_pos=1.0`, `effector_hold_vel=0.0`, and for
the first time the hold penalty is actually applied. The 2bc95fd hold-drift values therefore
reflect "no kinematic hold penalty" behaviour; they are NOT a fair baseline for hold drift.

### Weight configuration

This matrix does **not** use `nn_hidden_derivative` (it is 0.0 for all cells). The per-cell
distinguishing weights are:

| Cell | `nn_output_jerk` | Running schedule | Hold schedule |
|---|---|---|---|
| `lit__flat_jerk` | 1e5 | flat | flat |
| `lit__post_jerk` | 1e5 | powerlaw | flat |
| `lit__full_jerk` | 1e5 | powerlaw | powerlaw |
| `lit__flat_nojerk` | 0.0 | flat | flat |
| `lit__post_nojerk` | 0.0 | powerlaw | flat |
| `lit__full_nojerk` | 0.0 | powerlaw | powerlaw |

### Run metadata

- Experiment hash: `f47abb1`
- Pod: `jmhwbqd61kw9z3`, RTX 4090, CZ datacenter
- Wall-clock: ~32 min/cell, 6 cells sequential (~3.2 hr total)
- SISU: 0.5, perturbation: 0 (clean reach)
- Validation trials: 8 center-out reach directions
- Git SHA: `15f647bfbcb8df20966e94141667ee41f24af5fe`

## Metrics

**Primary: velocity-RMSE ratio** — within-cell mean pairwise RMSE on the forward-velocity
profile / nearest-neighbor across-cell mean pairwise RMSE. Matches the prior
`baseline_jerk_vrnn_matrix` metric. Prior best (GRU/jerk, 2bc95fd): **0.758**.
Decision threshold: **< 0.50**.

**Secondary: position-RMSE ratio** — same computation on the forward-position profile.

**Auxiliary: CV (SD/mean of peak vel)** — scalar summary of replicate spread on peak
velocity. Reported for completeness; does NOT drive the decision.

### IMPORTANT — cross-schedule absolute loss comparisons

Absolute loss values (and by extension total-loss curves) are **NOT comparable** across
position schedule shapes. The powerlaw `(t/T)^6` concentrates ~98% of position weight in
the last 30% of the trial, making the weighted sum structurally lower for powerlaw cells
than for flat cells. Comparisons should be made **within** schedule shape only:
- Jerk effect: compare `lit__flat_jerk` vs `lit__flat_nojerk` (flat shape)
- Jerk effect: compare `lit__post_jerk` vs `lit__post_nojerk` (post-go PL)
- Jerk effect: compare `lit__full_jerk` vs `lit__full_nojerk` (full-trial PL)
- Schedule effect (jerk on): `lit__flat_jerk` vs `lit__post_jerk` vs `lit__full_jerk`
- Schedule effect (jerk off): `lit__flat_nojerk` vs `lit__post_nojerk` vs `lit__full_nojerk`

## Results Table

*(Populated by `scripts/analyse_lit_replication_6cell.py`)*

| Cell | Display Name | Vel-RMSE ratio (PRIMARY) | Pos-RMSE ratio | CV (peak vel) | Mean PV (m/s) | SD PV (m/s) | Hold Drift (mm) | TTP (steps) |
|------|------|---------|---------|---------|---------|---------|---------|---------|
| lit__flat_jerk | Flat + jerk | — | — | — | — | — | — | — |
| lit__post_jerk | Post-go PL + jerk | — | — | — | — | — | — | — |
| lit__full_jerk | Full-trial PL + jerk | — | — | — | — | — | — | — |
| lit__flat_nojerk | Flat, no jerk | — | — | — | — | — | — | — |
| lit__post_nojerk | Post-go PL, no jerk | — | — | — | — | — | — | — |
| lit__full_nojerk | Full-trial PL, no jerk | — | — | — | — | — | — | — |

\* = beats primary threshold (vel-RMSE ratio < 0.50).

## RMSE Detail (within vs across)

*(Populated by `scripts/analyse_lit_replication_6cell.py`)*

| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) | Pos within-RMSE (m) | Pos nearest-across-RMSE (m) |
|------|---------|---------|---------|---------|

## Decision

**Primary threshold**: vel-RMSE ratio < 0.50
Prior best (GRU/jerk, baseline\_jerk\_vrnn\_matrix, 2bc95fd): 0.758

*(Winner determination populated after analysis script run.)*

## Per-axis findings

### Jerk axis

Compare within same schedule shape (jerk on vs off):
- **Flat**: `lit__flat_jerk` vs `lit__flat_nojerk` — tests jerk effect independently of schedule
- **Post-go PL**: `lit__post_jerk` vs `lit__post_nojerk`
- **Full-trial PL**: `lit__full_jerk` vs `lit__full_nojerk`

Expected hypothesis: jerk regulariser reduces inter-replicate variance (as seen in 2bc95fd
where `gru__jerk_motor_smooth_combo` with jerk achieved vel-RMSE-ratio=0.036). Jerk alone
was insufficient in 2bc95fd (0.461 pre-go mask, 1.129 jerk-only). The question is whether
the corrected hold penalty changes this picture.

### Position schedule axis

Compare within same jerk condition:
- **Jerk on**: flat vs post-go vs full-trial — tests whether C&S schedule helps independently
- **Jerk off**: flat vs post-go vs full-trial — tests schedule effect without jerk confound

Expected: if the position schedule matters, cells with powerlaw schedule should show lower
vel-RMSE ratio. If flat and powerlaw are similar, the schedule shape is not the driver.

## Conditional follow-up triggers (per f47abb1 issue body)

**Pre-go-mask follow-up** (file as new issue on `c99ad9d`): if jerk-disabled cells
(lit__flat_nojerk, lit__post_nojerk, lit__full_nojerk) show significant anticipation
relative to jerk-enabled cells — defined as:
- hold drift > 1 mm averaged across replicates, OR
- visible pre-go velocity ramp in the hold_drift_profiles figure

Then reintroduce `--nn-output-pre-go` as a follow-up matrix lever (suggested starting weight
1e-2, per efc4d68 matrix). The key question is whether anticipation is driven by the absence
of jerk or by the absence of an explicit pre-go motor suppression penalty.

## Anticipation (Hold Drift)

Hold drift = max forward displacement (toward target, in mm) before the go cue.
Positive = anticipatory movement. Threshold for "good hold": < 0.5 mm.
Threshold for "conditional follow-up trigger": > 1 mm.

*(Populated after analysis script run.)*

Note: the 2bc95fd hold-drift values (e.g., gru__jerk: 19.836 mm) reflect the broken hold
penalty (task-type `==` bug) and are NOT comparable to this matrix.

## Figures

- `results/f47abb1/figures/rmse_ratio_comparison/` — Bar chart, vel- and pos-RMSE ratios (PRIMARY)
- `results/f47abb1/figures/peak_velocity_distributions/` — Violin/strip plot (CV annotated, auxiliary)
- `results/f47abb1/figures/forward_velocity_profiles/` — Forward velocity time series per cell
- `results/f47abb1/figures/hold_drift_profiles/` — Pre-go forward position (anticipation drift)

HTML renders at `_artifacts/f47abb1/figures/<name>/figure.html` (gitignored).
Figure specs (JSON) at `results/f47abb1/figures/<name>/spec.json` (tracked).

Training-loss figures: `results/f47abb1/figures/training_loss/` and
`results/f47abb1/figures/training_loss_per_term/` — generated by
`scripts/plot_training_loss_lit_replication.py`.
