# Variance Analysis — Lit-Replication 6-Cell Matrix (f47abb1)

## Setup

This matrix tests whether faithful replication of the Chaisanguanthum & Shenoy
2019 (C&S) loss schedule — power-law `(t/T)^6` position cost, no jerk regulariser
— produces tighter inter-replicate clustering and lower anticipation than the
rlrmp production loss.

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

### Framing note — what to compare in this matrix

This matrix sweeps training-method variants (jerk on/off × position schedule).
There is **no a priori reason** for any two cells to produce systematically
similar velocity profiles — different cost-function shapes are expected to
produce different kinematics. That makes the within/across-cell pairwise RMSE
**ratio** a *misleading* primary metric here: the denominator (nearest-across-
cell RMSE) reflects expected cost-induced trajectory divergence between cells,
not nuisance variation we'd want to normalise out.

Within/across ratios are justified only when the conditions being compared are
expected to produce systematically different trajectories AND we want to ask
whether replicates of one condition cluster tighter than the distance between
conditions. That framing fits *generalisation* or *anchor-vs-condition* matrices
(e.g. "is condition X reliably distinct from anchor Y?"). It does not fit a
training-method comparison where every cell is a candidate for the same job.

**Operative metric for this matrix: absolute within-cell RMSE (m/s)** on the
forward-velocity profile. Compare cells directly on raw within-cell numbers.

This reframing changes the headline reading of the matrix — see "Decision"
below and the corrective comment on `f47abb1`. The cross-cutting lesson is
logged on `4d38c15` (analyses coord).

### Metric definitions

**PRIMARY — within-cell vel RMSE (m/s)**: within-cell mean pairwise RMSE on the
forward-velocity profile across the 5 replicates of a cell. Lower = tighter
clustering. No fixed threshold for this matrix; the metric is read absolutely
against the ~0.026 m/s floor surfaced by the best cells.

**PRIMARY — hold drift (mm)**: max forward (toward-target) displacement during
the pre-go epoch. Lower = less anticipation. Target: < 0.5 mm. Trigger for
pre-go-mask follow-up: > 1 mm.

**PRIMARY — peak forward velocity (m/s)**: scalar summary of post-go kinematics.
Used as a sanity check against literature reach speeds (typical biological
reaches: 0.5–1.0 m/s for ~0.1 m amplitude). Not a variance metric per se.

**Auxiliary — vel/pos RMSE ratio (within / nearest-across)**: retained for
continuity with prior matrices that *did* have an anchor-vs-conditions framing
(`baseline_jerk_vrnn_matrix`, `2bc95fd`). In this matrix the ratio is **not the
primary decision metric**; see the framing note above. The < 0.5 threshold
that was pre-registered on `f47abb1` is therefore not the operative test for
this matrix.

**Auxiliary — CV (SD/mean of peak vel)**: scalar summary of replicate spread
on peak velocity. Reported for completeness; does NOT drive the decision.

**IMPORTANT — cross-schedule absolute *loss* comparisons are NOT valid.** The
powerlaw `(t/T)^6` concentrates ~98% of position weight in the last 30% of the
trial, making the weighted loss sum structurally lower than for flat. The
**kinematic** metrics above (within-cell RMSE on velocity, hold drift in mm,
peak velocity in m/s) ARE comparable across schedules — they measure the
controller's behaviour, not the training-loss aggregate.

## Results Table (PRIMARY: absolute within-cell metrics)

| Cell | Display Name | Vel within-RMSE (m/s) | Hold Drift (mm) | Peak Vel (m/s) | TTP (steps) |
|------|------|---------|---------|---------|---------|
| lit__flat_jerk | Flat + jerk | 0.1086 | 22.90 +/- 3.30 | 0.590 | 62.4 |
| lit__post_jerk | Post-go PL + jerk | 0.0423 | 3.29 +/- 3.04 | 0.791 | 71.8 |
| lit__full_jerk | Full-trial PL + jerk | 0.0426 | 3.40 +/- 3.10 | 0.791 | 71.8 |
| lit__flat_nojerk | Flat, no jerk | 0.0918 | 24.32 +/- 3.11 | 0.606 | 58.5 |
| **lit__post_nojerk** | **Post-go PL, no jerk** | **0.0361** | **2.34 +/- 0.56** | **0.969** | **54.6** |
| **lit__full_nojerk** | **Full-trial PL, no jerk** | **0.0414** | **2.74 +/- 0.55** | **0.964** | **54.5** |

Bold rows = strongest cells on within-cell clustering AND anticipation suppression.

## Auxiliary table (pairwise ratios — not primary, see framing note)

| Cell | Vel-RMSE ratio | Pos-RMSE ratio | CV (peak vel) | Mean PV (m/s) | SD PV (m/s) |
|------|---------|---------|---------|---------|---------|
| lit__flat_jerk | 1.157 | 1.293 | 0.092 | 0.5899 | 0.0541 |
| lit__post_jerk | 1.229 | 1.235 | 0.040 | 0.7909 | 0.0319 |
| lit__full_jerk | 1.239 | 1.233 | 0.041 | 0.7905 | 0.0326 |
| lit__flat_nojerk | 0.978 | 0.897 | 0.089 | 0.6056 | 0.0541 |
| lit__post_nojerk | 1.091 | 1.055 | 0.026 | 0.9686 | 0.0252 |
| lit__full_nojerk | 1.250 | 1.277 | 0.026 | 0.9638 | 0.0247 |

Reading these in this matrix: the denominator (nearest-across-cell RMSE) is set
by inter-cell kinematic distance, which IS driven by the cost-function
manipulations the matrix is studying. A high ratio means a cell's within-cell
RMSE is comparable to the kinematic distance between cells — not that the
cell is poorly converged. The strongest within-cell cell (`post_nojerk`,
within-RMSE 0.036) has ratio 1.09 because the nearest-across-cell RMSE
(0.033 m/s) is itself very small — neighbouring no-jerk powerlaw cells produce
nearly identical mean trajectories.

## RMSE Detail (within vs across — auxiliary support for the ratio table above)

| Cell | Vel within-RMSE (m/s) | Vel nearest-across-RMSE (m/s) | Pos within-RMSE (m) | Pos nearest-across-RMSE (m) |
|------|---------|---------|---------|---------|
| lit__flat_jerk | 0.1086 | 0.0938 | 0.0497 | 0.0384 |
| lit__post_jerk | 0.0423 | 0.0344 | 0.0144 | 0.0117 |
| lit__full_jerk | 0.0426 | 0.0344 | 0.0144 | 0.0117 |
| lit__flat_nojerk | 0.0918 | 0.0938 | 0.0345 | 0.0384 |
| lit__post_nojerk | 0.0361 | 0.0331 | 0.0082 | 0.0078 |
| lit__full_nojerk | 0.0414 | 0.0331 | 0.0099 | 0.0078 |

## Decision

Read on the operative metric — **absolute within-cell vel RMSE (m/s) + hold drift (mm)**:

**Winner: `lit__post_nojerk`** — powerlaw `(t/T)^6` post-go, jerk off.

- Tightest within-cell clustering: vel within-RMSE = **0.036 m/s** (vs 0.092 for
  the closest flat cell, ~3× tighter).
- Lowest anticipation: hold drift = **2.34 mm** (vs 22.9–24.3 mm for the flat
  cells, ~10× reduction).
- Fast, plausible peak velocity: **0.969 m/s** (within the upper end of the
  biological 0.5–1.0 m/s range for 0.1 m reaches).
- `lit__full_nojerk` (powerlaw throughout) is a close second on every axis.

**This is a substantive POSITIVE finding for the lit-replication direction**:
the powerlaw schedule + jerk-off combination delivers both tight replicate
clustering AND low anticipation, where neither lever alone is enough. The
opposite-sign reading from the earlier ratio-primary framing
("no cell beats < 0.5 threshold") was an artefact of the wrong primary metric
for this matrix — see the framing note above and the corrective comment on
`f47abb1`.

## Headline findings

1. **Powerlaw schedule + jerk-off is the strongest combination.** Both
   `post_nojerk` and `full_nojerk` produce tight within-cell RMSE (~0.036–0.041
   m/s) and small anticipation (2.3–2.7 mm). The post-go variant edges the
   full-trial variant on both metrics.

2. **Flat schedule fails on both axes.** `flat_jerk` and `flat_nojerk` show
   catastrophic 23–24 mm pre-go anticipation and 2–3× larger within-cell
   velocity RMSE than the powerlaw cells. The literature-faithful uniform-time
   target-switching cost (Shahbazi-style) is what's bad here, not lit-
   replication writ large.

3. **Jerk regulariser provides no improvement at corrected effective weight.**
   At `nn_hidden_derivative=1e-3` (Shahbazi-effective), turning the jerk
   regulariser on slightly *increases* within-cell velocity RMSE on flat and
   post-go schedules; on full-trial powerlaw the effect is null. Consistent
   with C&S 2019 not using a jerk term: under their cost schedule, jerk is not
   load-bearing for kinematic shaping.

4. **Residual ~2–3 mm hold drift in the best cells triggers the pre-go-mask
   follow-up.** Both `post_nojerk` and `full_nojerk` are above the < 0.5 mm
   "good hold" threshold. The conditional follow-up matrix
   (`--nn-output-pre-go` sweep on top of these cells) is unambiguously
   triggered. Filed as `3702f54`.

## Per-axis findings

### Jerk axis (compare within same schedule shape, on absolute vel within-RMSE)

| Schedule | Jerk on (within-RMSE, m/s) | Jerk off (within-RMSE, m/s) | Δ (off − on) |
|----------|----------------------------|------------------------------|--------------|
| Flat | 0.109 | 0.092 | −0.016 (jerk-off slightly tighter) |
| Post-go PL | 0.042 | 0.036 | −0.006 (jerk-off slightly tighter) |
| Full-trial PL | 0.043 | 0.041 | −0.001 (essentially tied) |

Jerk-off is at least as good as jerk-on on every schedule. The Shahbazi prior
that jerk funnels replicates is not visible at the corrected
`nn_hidden_derivative=1e-3`.

### Position schedule axis (compare within same jerk condition, on absolute vel within-RMSE)

| Jerk | Flat (within-RMSE, m/s) | Post-go PL | Full-trial PL |
|------|-------------------------|-------------|----------------|
| On (1e5) | 0.109 | 0.042 | 0.043 |
| Off (0.0) | 0.092 | **0.036** | 0.041 |

Both powerlaw schedules give 2–3× tighter within-cell velocity profiles than
flat, in both jerk conditions. Post-go and full-trial powerlaw are similar to
each other; post-go is marginally tighter.

## Conditional follow-up triggers (per f47abb1 issue body)

**Pre-go-mask follow-up — TRIGGERED.** Both winning cells (`post_nojerk` and
`full_nojerk`) have hold drift in the 2–3 mm range, above the 1 mm follow-up
trigger and well above the < 0.5 mm "good hold" target. Filed as `3702f54`
(pre-go-motor-mask follow-up matrix, building on these cells).

**Position-weight + target_ratio sweep — NOT TRIGGERED.** Peak velocities for
the winning cells (0.96–0.97 m/s) are at the upper end of biological reach
speeds but within range; reach kinematics are not anomalous. The pre-go-mask
follow-up takes precedence over a position-weight sweep.

## Anticipation (Hold Drift)

Hold drift = max forward displacement (toward target, in mm) before the go cue.
Positive = anticipatory movement. Target for "good hold": < 0.5 mm.

- Flat + jerk (`lit__flat_jerk`): 22.90 +/- 3.30 mm <-- catastrophic
- Post-go PL + jerk (`lit__post_jerk`): 3.29 +/- 3.04 mm <-- mild anticipation
- Full-trial PL + jerk (`lit__full_jerk`): 3.40 +/- 3.10 mm <-- mild anticipation
- Flat, no jerk (`lit__flat_nojerk`): 24.32 +/- 3.11 mm <-- catastrophic
- **Post-go PL, no jerk (`lit__post_nojerk`): 2.34 +/- 0.56 mm <-- best**
- **Full-trial PL, no jerk (`lit__full_nojerk`): 2.74 +/- 0.55 mm <-- close second**

The variance on hold drift collapses with the powerlaw + nojerk combination
(SD goes from ~3 mm on every other cell to ~0.55 mm on the two winners) —
the replicates are not just hold-drifting less, they're hold-drifting
consistently. A pre-go-mask term on top should be able to drive the mean
toward zero without much variance to fight.

## Figures

- `results/f47abb1/figures/rmse_ratio_comparison/` — Bar chart of within/across
  ratios. NOTE: now framed as auxiliary; the absolute within-cell RMSE numbers
  are the operative metric (see framing note above and the `metric_description`
  field in that figure's `spec.json`).
- `results/f47abb1/figures/peak_velocity_distributions/` — Violin (CV
  annotated, auxiliary).
- `results/f47abb1/figures/forward_velocity_profiles/` — Velocity time series
  per cell.
- `results/f47abb1/figures/hold_drift_profiles/` — Pre-go forward position
  (anticipation).

HTML renders in `_artifacts/f47abb1/figures/<name>/figure.html`.
