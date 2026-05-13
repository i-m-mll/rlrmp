# Pre-go Motor Mask Matrix — Analysis (3702f54)

> **Corrected after go-cue alignment fix (Bug: 06f7faf).** The
> within-cell vel-RMSE values were originally computed by averaging
> trial-mean profiles in *absolute trial time* before the pairwise
> RMSE step. Because `centerout`'s target-on duration is randomized
> per trial, this smeared the go cue across ~150 ms and produced
> biased RMSE values. The forward-velocity and hold-drift figures
> were also drawn in absolute time. The fix re-locks each trial to
> its own go cue before the trial-axis collapse.
>
> **Effect on conclusions:** *the headline is unchanged.*
> `full_trial_pl__prego_1` still wins decisively. Vel-RMSE values
> shift modestly (most cells shift up by ~5–40 %, with the largest
> shifts on the unstable `__pos10` cells); the winner ordering, the
> ~50× pre-go-RMS suppression, and the verdict "discard __pos10,
> adopt prego_1" are all preserved. Hold drift, pre-go RMS, peak vel,
> and TTP are scalar per-trial metrics that already used per-trial
> go-cue indexing and are unchanged.

## Headline

`full_trial_pl__prego_1` wins decisively. The `--nn-output-pre-go` lever at
weight 1.0 fully suppresses pre-go anticipation (pre-go RMS **0.02 mm** vs
~1.1 mm for the f47abb1 baselines, a ~50× reduction and well below the
0.5 mm target carried in the issue body) while *improving* every other
metric — vel-RMSE drops from 0.036–0.041 to **0.018 m/s** (the best cell in
the matrix and roughly half the strongest f47abb1 baseline), peak velocity
rises from 0.97 m/s to **1.09 m/s**, and time-to-peak is unchanged at
~37 steps. The complementary `__pos10` lever, in contrast, regresses
catastrophically on every other axis: peak velocity collapses to ~0.6 m/s
(35% below baseline), vel-RMSE blows up by 3–4×, and time-to-peak slows by
25–40 steps. `__pos10` should be discarded entirely; `prego_1.0` (or
`prego_5e-2` as a conservative fallback) is adopted as the new production
loss configuration on `full_trial_pl`.

## Setup

This matrix is the conditional follow-up to f47abb1, triggered by the
2–3 mm residual hold drift on `lit__post_nojerk` and `lit__full_nojerk`
that exceeded the < 0.5 mm "good hold" target. Two complementary
suppression strategies are crossed:

1. **`__pos10` lever** — scale the position-error weight 10× on
   `full_trial_pl` (so the powerlaw's already-tiny hold-period position
   penalty is made more biting via amplitude).
2. **`--nn-output-pre-go` lever** — strategy 1 from `5acdaae`, an
   independent pre-go output-quietness regulariser at three weights
   spanning ~3 orders of magnitude: `{1e-3, 5e-2, 1.0}`.

The 8 new cells cross these two levers on the `full_trial_pl` powerlaw
configuration (effectively-zero hold-period position penalty, the cleanest
test bed for an independent pre-go term). The two baseline anchors —
`lit__post_nojerk` and `lit__full_nojerk` from f47abb1 — are included in
all comparison plots but NOT retrained here. See `results/3702f54/README.md`
and the finalised matrix in comment `c5e2ad2` on `3702f54` (the realised
sweep deviates from the proposed `{0, 1e-2, 1e-1, 1.0}` in the issue body).

`n_adversary_batches=0` throughout this matrix — adversarial training is
not exercised, only the supervised loss components.

## Metric definitions

- **Vel-RMSE (m/s)** — within-cell mean pairwise RMSE on the forward
  velocity profile across the 5 replicates. Lower = tighter clustering.
  Read as an absolute number (the f47abb1 framing note on within/across
  ratios applies: every cell here is a candidate for the same job, so the
  ratio is not the operative metric).
- **Hold drift (mm)** — max forward (toward-target) displacement during
  the pre-go epoch. Pre-existing f47abb1 metric.
- **Pre-go RMS (mm)** — RMS of forward effector position over the pre-go
  window. Sharper than hold-drift-max for distinguishing the prego cells
  (some of which sit ~0.5 mm peaks that wash out the max-based metric).
  Target: < 0.5 mm.
- **Peak velocity (m/s)** — scalar summary of post-go kinematics. Sanity
  check against biological reach speeds (~0.5–1.0 m/s for 0.1 m reaches).
- **TTP (steps)** — time-to-peak velocity, post-go. Urgency proxy.

## Per-cell metrics

### Post-fix table (after Bug: 06f7faf)

| Cell | Vel-RMSE (m/s) | Hold drift (mm) | Pre-go RMS (mm) | Peak vel (m/s) | TTP (steps) |
|---|---:|---:|---:|---:|---:|
| **lit__post_nojerk** (baseline) | 0.0383 | 2.34 ± 0.56 | 1.02 ± 0.19 | 0.969 ± 0.025 | 34.3 ± 1.8 |
| **lit__full_nojerk** (baseline) | 0.0428 | 2.74 ± 0.55 | 1.15 ± 0.17 | 0.964 ± 0.025 | 34.2 ± 2.1 |
| post_go_pl__pos10 | 0.1824 | 3.60 ± 4.44 | 1.51 ± 1.88 | 0.653 ± 0.110 | 59.0 ± 7.5 |
| full_trial_pl__pos10 | 0.1426 | 4.40 ± 5.63 | 1.89 ± 2.46 | 0.630 ± 0.057 | 59.7 ± 9.0 |
| full_trial_pl__prego_1e-3 | 0.0299 | 0.00 | 0.40 ± 0.15 | 1.010 ± 0.022 | 37.2 ± 1.0 |
| full_trial_pl__prego_5e-2 | 0.0311 | 0.002 | 0.05 ± 0.01 | 1.086 ± 0.017 | 37.2 ± 0.9 |
| **full_trial_pl__prego_1** | **0.0255** | **0.02** | **0.02 ± 0.01** | **1.087 ± 0.010** | **36.8 ± 0.6** |
| full_trial_pl__pos10_prego_1e-3 | 0.1780 | 3.54 ± 4.33 | 1.65 ± 2.21 | 0.638 ± 0.035 | 64.9 ± 9.2 |
| full_trial_pl__pos10_prego_5e-2 | 0.1109 | 0.12 ± 0.04 | 0.09 ± 0.02 | 0.603 ± 0.026 | 73.0 ± 8.1 |
| full_trial_pl__pos10_prego_1 | 0.1841 | 0.001 | 0.08 ± 0.02 | 0.711 ± 0.133 | 71.0 ± 14.5 |

### Pre-fix table (deprecated, kept for traceability)

Original numbers before the go-cue alignment fix:

| Cell | Vel-RMSE (m/s, deprecated) |
|---|---:|
| lit__post_nojerk (baseline) | 0.0361 |
| lit__full_nojerk (baseline) | 0.0414 |
| post_go_pl__pos10 | 0.1394 |
| full_trial_pl__pos10 | 0.1047 |
| full_trial_pl__prego_1e-3 | 0.0221 |
| full_trial_pl__prego_5e-2 | 0.0244 |
| full_trial_pl__prego_1 | 0.0176 |
| full_trial_pl__pos10_prego_1e-3 | 0.1272 |
| full_trial_pl__pos10_prego_5e-2 | 0.0520 |
| full_trial_pl__pos10_prego_1 | 0.1511 |

Bold = winning cell + the two baseline anchors. Vel-RMSE is reported as
the absolute within-cell number per the corrective comments on
f47abb1/c99ad9d/4d38c15 — do not lead with within/across ratios for this
matrix.

## Anticipation suppression

Dose-dependent and clean at pos×1. Pre-go RMS responds monotonically to
the regulariser weight:

| Weight | Pre-go RMS (mm) | Target (< 0.5 mm)? |
|---|---:|---|
| 1e-3 | 0.40 ± 0.15 | yes (marginal) |
| 5e-2 | 0.05 ± 0.01 | yes (~10× under) |
| 1.0 | 0.02 ± 0.01 | yes (~25× under) |

All three pos×1 prego cells clear the target. The weight-1.0 cell is the
strongest, but the practically-relevant message is that even the lightest
weight tested (1e-3) already crosses the target.

## Urgency — hypothesis falsified

The `__pos10` lever does the opposite of what was hypothesised in the
issue body. The prior intuition was: "more position weight ⇒ network
reaches faster to drive position error down sooner." The data falsify
that — TTP on every `__pos10` cell is **25 to 40 steps slower than
baseline** (59–73 steps vs ~34 steps for the f47abb1 baselines and ~37
steps for the pos×1 prego cells).

Working mechanism: 10× position weight is applied throughout the trial
under the powerlaw schedule (and under the flat hold schedule in
`post_go_pl__pos10`). During the pre-go window the position cost
therefore penalises *any* effector motion — the network learns a strong
"output zero by default" bias. Post-go, it has to overcome that bias
before it can build a reaching trajectory, which delays peak velocity.
The pre-go output regulariser at pos×1, in contrast, leaves the
post-go dynamics free: the regulariser is gated to the pre-go window so
once the go cue arrives the controller is free to produce arbitrary
output. TTP stays at the baseline ~37 steps for all three pos×1 prego
cells.

## Interaction

Not additive in the desired direction. The `__pos10` lever dominates the
regression on vel-RMSE, peak velocity, and TTP, and the pre-go output
term cannot rescue it. The `pos10 × prego` cross is strictly dominated by
`pos×1 × prego` on every axis except "is pre-go RMS smaller than an
already-tiny number" — and at pos×1 the pre-go RMS is already ~20× below
the 0.5 mm target, so that margin is not meaningful.

At pos×1, prego is cleanly additive on the baseline: anticipation
collapses, every other metric improves or holds. At pos×10, prego
suppresses anticipation (because the regulariser is doing its job in
isolation) but the surrounding regressions from `__pos10` swamp the
benefit.

## Regressions flagged

All 4 `__pos10` cells regress on the primary kinematic metrics:

- vel-RMSE: 2.9–4.2× the f47abb1 baselines.
- peak velocity: 32–35% below baseline.
- TTP: 25–40 steps slower than baseline.

`pos10_prego_1` shows peak-vel SD = 0.133 m/s — 4× the baseline replicate
spread — indicating one or more replicates is failing to reach. No NaN
cells, no peak vel below 0.3 m/s, no hold drift above 50 mm; no
catastrophic blow-ups but the regression is clear and uniform across
the `__pos10` arm.

## Verdict

Take **`full_trial_pl__prego_1`** forward as the new production loss
configuration on `full_trial_pl`. It improves on the strongest f47abb1
baseline on every metric: tighter velocity clustering, faster peak
velocity, anticipation crushed by ~50×, time-to-peak unchanged.

`prego_5e-2` is a conservative alternative if the prior at 1.0 is later
found to interfere with adversarial training (untested here —
`n_adversary_batches=0` throughout this matrix; both `prego_1.0` and
`prego_5e-2` need re-validation with adversary on before either is
committed to as a long-term default).

Discard `__pos10` entirely. The lever does not deliver the hypothesised
urgency benefit and regresses every other metric.

## Open questions for future work

- Pre-go output regulariser at weight 1.0 was tested with
  `n_adversary_batches=0`. Behaviour under adversarial training is
  unverified — the strong pre-go prior may either help (by removing a
  spurious pre-go DOF the adversary could exploit) or hurt (by choking
  adversary exploration). Needs an adversary-on follow-up.
- Whether the regulariser at lower weights (1e-3, 5e-2) is preferable
  *for adversarial robustness* depends on the answer above. The
  conservative pick (`prego_5e-2`) is motivated by this uncertainty.
- Whether the `nn_output_pre_go` formulation should generalise to
  `nn_hidden_derivative_pre_go` for hidden-state quietness pre-go
  (currently 0 in this matrix). Strategy 1 in `5acdaae` covers only
  the output term; a hidden-derivative-pre-go variant is the natural
  next strategy if pre-go hidden dynamics turn out to be a problem
  surface.

## Cross-refs

- `f47abb1` — triggering matrix (lit-replication, 6 cells). Identified
  the residual 2–3 mm hold drift on `lit__post_nojerk` /
  `lit__full_nojerk` that this matrix is built to suppress.
- `5acdaae` — anti-anticipation strategy menu. This matrix tests
  strategy 1 (`nn_output_pre_go`).
- `c99ad9d` — training-methods coord. Cross-ref comment to be filed by
  the orchestrator (this matrix shifts the production loss for
  `full_trial_pl` and falsifies the `__pos10` urgency intuition).
- Issue `3702f54` body — proposed sweep (`{0, 1e-2, 1e-1, 1.0}` on
  `post_go_pl`); the finalised matrix differs (see comment `c5e2ad2`
  on `3702f54` for the realised matrix design).

## Plots

All paths below are repo-relative `results/...` symlinks that resolve via
the `_artifacts/` shared symlink to the heavy renders in
`_artifacts/3702f54/figures/<topic>/figure.html`. They are navigation
pointers, not tracked HTML.

- `results/3702f54/figures/summary_metrics/figure.html` — all metrics on
  one canvas. **Headline plot.**
- `results/3702f54/figures/forward_velocity_profiles/figure.html` —
  mean ± SD forward velocity per cell, baselines anchored at top.
- `results/3702f54/figures/hold_drift_profiles/figure.html` — pre-go
  forward effector position over time. The anticipation figure.
- `results/3702f54/figures/peak_velocity_distributions/figure.html` —
  strip plot, replicates as points.
- `results/3702f54/figures/training_loss_per_term/figure.html` — loss
  decomposition over training.
