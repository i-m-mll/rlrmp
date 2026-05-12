# Variance Analysis — Lit-Replication 6-Cell Matrix (f47abb1)

## Setup

This matrix tests whether faithful replication of the Chaisanguanthum & Shenoy
2019 (C&S) loss schedule produces models with systematically better velocity-RMSE
ratios and lower inter-replicate variance than the current production loss.

Two design dimensions crossed:
1. **Jerk regulariser** (`nn_output_jerk`): on (1e5) vs off (0.0).
   Shahbazi et al. 2025 Eq. 1 used jerk; C&S 2019 did not.
2. **Position schedule**: flat / post-go `(t/T)^6` / full-trial `(t/T)^6`.
   The powerlaw concentrates ~98% of position weight in the last 30% of the trial.

**Corrected hold-penalty bug** (vs 2bc95fd): prior run used `==` check for
`center_out_delayed_reach` task type, which silently failed. Fixed in commit `22153e4`.
This run applies hold penalties correctly for the first time.

**`nn_hidden_derivative` weight**: this matrix used `nn_hidden_derivative=0.001`
(reconstructed from saved `adversarial_model.eqx` hyperparams header — `run.json`
omitted this flag and was patched after the fact; see `cli_flags_reconstruction`
field in each cell's `run.json`). The 2bc95fd matrix used `nn_hidden_derivative=1e2`
in the `gru__jerk_smooth_high` and combo cells; this matrix uses a much smaller
1e-3 (per Shahbazi et al. 2025 Eq. 1).

### Run metadata

- Experiment hash: `f47abb1`
- SISU: 0.5
- Perturbation: 0 (clean reach)
- Validation trials: 8 center-out reach directions
- Pod: jmhwbqd61kw9z3, RTX 4090, CZ datacenter
- Wall-clock: ~32 min/cell, 6 cells sequential
- Git SHA: 15f647bfbcb8df20966e94141667ee41f24af5fe

## Metrics

**Primary: velocity-RMSE ratio** — within-cell mean pairwise RMSE on the
forward-velocity profile / nearest-neighbor across-cell mean pairwise RMSE.
Matches the prior `baseline_jerk_vrnn_matrix` metric.
Prior best (GRU/jerk, 2bc95fd): 0.758.
Decision threshold: < 0.50.

**Secondary: position-RMSE ratio** — same computation on forward-position profile.

**Auxiliary: CV (SD/mean of peak vel)** — scalar summary of replicate spread on
peak velocity. Reported for completeness; does NOT drive the decision.

**IMPORTANT — cross-schedule comparisons**: absolute loss values are NOT
comparable across position schedule shapes. The powerlaw `(t/T)^6` concentrates
~98% of position weight in the last 30% of the trial, making the weighted sum
structurally lower than for flat. Compare WITHIN schedule shape only:
  - Flat: `lit__flat_jerk` vs `lit__flat_nojerk`
  - Post-go: `lit__post_jerk` vs `lit__post_nojerk`
  - Full-trial: `lit__full_jerk` vs `lit__full_nojerk`

## Results Table

| Cell | Display Name | Vel-RMSE ratio (PRIMARY) | Pos-RMSE ratio | CV (peak vel) | Mean PV (m/s) | SD PV (m/s) | Hold Drift (mm) | TTP (steps) |
|------|------|---------|---------|---------|---------|---------|---------|---------|
| lit__flat_jerk | Flat + jerk | 1.157 | 1.293 | 0.092 | 0.5899 | 0.0541 | 22.901 +/- 3.302 | 62.4 |
| lit__post_jerk | Post-go PL + jerk | 1.229 | 1.235 | 0.040 | 0.7909 | 0.0319 | 3.290 +/- 3.042 | 71.8 |
| lit__full_jerk | Full-trial PL + jerk | 1.239 | 1.233 | 0.041 | 0.7905 | 0.0326 | 3.395 +/- 3.100 | 71.8 |
| lit__flat_nojerk | Flat, no jerk | 0.978 | 0.897 | 0.089 | 0.6056 | 0.0541 | 24.318 +/- 3.108 | 58.5 |
| lit__post_nojerk | Post-go PL, no jerk | 1.091 | 1.055 | 0.026 | 0.9686 | 0.0252 | 2.337 +/- 0.561 | 54.6 |
| lit__full_nojerk | Full-trial PL, no jerk | 1.250 | 1.277 | 0.026 | 0.9638 | 0.0247 | 2.742 +/- 0.547 | 54.5 |

\* = beats primary threshold (vel-RMSE ratio < 0.50).

## RMSE Detail (within vs across)

| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) | Pos within-RMSE (m) | Pos nearest-across-RMSE (m) |
|------|---------|---------|---------|---------|
| lit__flat_jerk | 0.1086 | 0.0938 | 0.0497 | 0.0384 |
| lit__post_jerk | 0.0423 | 0.0344 | 0.0144 | 0.0117 |
| lit__full_jerk | 0.0426 | 0.0344 | 0.0144 | 0.0117 |
| lit__flat_nojerk | 0.0918 | 0.0938 | 0.0345 | 0.0384 |
| lit__post_nojerk | 0.0361 | 0.0331 | 0.0082 | 0.0078 |
| lit__full_nojerk | 0.0414 | 0.0331 | 0.0099 | 0.0078 |

## Decision

**Primary threshold**: vel-RMSE ratio < 0.5
Prior best (GRU/jerk, 2bc95fd): 0.758

**No cell met the threshold.**

Best cell: **Flat, no jerk** (`lit__flat_nojerk`) with vel-RMSE-ratio = 0.978

## Headline findings

1. **No cell beats the < 0.5 vel-RMSE-ratio threshold.** Best cell is
   `lit__flat_nojerk` at vel-RMSE-ratio = 0.978. All other cells are above 1.0,
   meaning their within-cell pairwise RMSE is HIGHER than the nearest-across-cell
   RMSE — i.e. replicates within a cell diverge more from each other than from
   replicates of neighbouring cells. This is a strong negative result for the
   lit-replication hypothesis: faithful C&S (no jerk, powerlaw schedule) does
   not produce tightly clustered replicates.
2. **Universal hold-drift failure.** Every cell shows hold drift > 1 mm (the
   conditional follow-up trigger), and the flat-schedule cells show
   catastrophic 23-24 mm pre-go forward drift — clearly visible anticipatory
   reaching. The powerlaw cells reduce this to 2.3-3.4 mm, but none reach the
   < 0.5 mm "good hold" threshold. **The conditional pre-go-mask follow-up
   matrix (`--nn-output-pre-go`) is unambiguously triggered.**
3. **Schedule effect dominates jerk effect.** Within either jerk condition,
   moving from flat to powerlaw schedule causes large changes; turning jerk on
   or off within the same schedule has small effect on vel-RMSE-ratio.
4. **Peak velocities are too high for no-jerk powerlaw cells** (0.96-0.97 m/s),
   above the sanity-check ceiling of 0.8 m/s set in RUN_PLAN. Jerk-on cells
   have more physically plausible peaks (0.59-0.79 m/s).

## Per-axis findings

### Jerk axis (compare within same schedule shape)

| Schedule | Jerk on (vel-RMSE) | Jerk off (vel-RMSE) | Δ (off − on) |
|----------|--------------------|---------------------|--------------|
| Flat      | 1.157  | 0.978  | −0.179 (jerk-off is BETTER, 15% reduction) |
| Post-go PL | 1.229 | 1.091  | −0.138 (jerk-off is BETTER, 11% reduction) |
| Full-trial PL | 1.239 | 1.250 | +0.011 (essentially tied) |

Jerk-OFF gives a SLIGHTLY lower vel-RMSE-ratio than jerk-ON in the flat and
post-go cells, contrary to the Shahbazi prior that the jerk regulariser
funnels replicates. Within-cell vel-RMSE values (m/s) tell a similar story:

| Schedule | Jerk on (within) | Jerk off (within) | Effect |
|----------|------------------|-------------------|--------|
| Flat | 0.109 | 0.092 | jerk-off has tighter within-cell |
| Post-go PL | 0.042 | 0.036 | jerk-off has tighter within-cell |
| Full-trial PL | 0.043 | 0.041 | tied |

CV (auxiliary): jerk-off powerlaw cells (0.026) have lower CV than jerk-on
powerlaw cells (0.040-0.041) — the no-jerk powerlaw replicates cluster more
tightly on peak velocity. But this is offset by peak velocities being too high.

### Position schedule axis (compare within same jerk condition)

| Jerk | Flat (vel-RMSE) | Post-go PL | Full-trial PL |
|------|-----------------|------------|---------------|
| On (1e5)  | 1.157 | 1.229 | 1.239 |
| Off (0.0) | 0.978 | 1.091 | 1.250 |

Within both jerk conditions, FLAT is the BEST vel-RMSE-ratio schedule. The
power-law schedule (with or without jerk) gives WORSE vel-RMSE-ratio than flat
— it makes within-cell variance worse, not better. This contradicts the C&S
prior that the late-trial-concentrated loss reduces replicate variance.

However, **the powerlaw schedule dramatically reduces hold drift** (24 mm →
2-3 mm). So the schedule trade-off is: powerlaw fixes anticipation but
worsens replicate clustering.

Position-RMSE ratio tells the same story as vel-RMSE in every comparison,
so it provides no independent signal.

## Conditional follow-up triggers (per f47abb1 issue body)

**Pre-go-mask follow-up**: if jerk-disabled cells (lit__flat_nojerk,
lit__post_nojerk, lit__full_nojerk) show significant anticipation relative to
jerk-enabled cells (hold drift > 1 mm or visible pre-go velocity ramp), reintroduce
`--nn-output-pre-go` as a follow-up matrix lever (suggested starting weight: 1e-2).

## Anticipation (Hold Drift)

Hold drift = max forward displacement (toward target, in mm) before the go cue.
Positive = anticipatory movement. Threshold for 'good hold': < 0.5 mm.

- Flat + jerk (`lit__flat_jerk`): 22.901 +/- 3.302 mm <-- anticipation trigger (> 1 mm)
- Post-go PL + jerk (`lit__post_jerk`): 3.290 +/- 3.042 mm <-- anticipation trigger (> 1 mm)
- Full-trial PL + jerk (`lit__full_jerk`): 3.395 +/- 3.100 mm <-- anticipation trigger (> 1 mm)
- Flat, no jerk (`lit__flat_nojerk`): 24.318 +/- 3.108 mm <-- anticipation trigger (> 1 mm)
- Post-go PL, no jerk (`lit__post_nojerk`): 2.337 +/- 0.561 mm <-- anticipation trigger (> 1 mm)
- Full-trial PL, no jerk (`lit__full_nojerk`): 2.742 +/- 0.547 mm <-- anticipation trigger (> 1 mm)

## Figures

- `results/f47abb1/figures/rmse_ratio_comparison/` — Bar chart (PRIMARY)
- `results/f47abb1/figures/peak_velocity_distributions/` — Violin (CV annotated, auxiliary)
- `results/f47abb1/figures/forward_velocity_profiles/` — Velocity time series per cell
- `results/f47abb1/figures/hold_drift_profiles/` — Pre-go forward position (anticipation)

HTML renders in `_artifacts/f47abb1/figures/<name>/figure.html`.
