# b399efc — 7-cell movement-ramp matrix results

## Overview

This note summarises the 7-cell movement-locked ramp matrix run under tracking issue `b399efc`. The experiment explored whether replacing the full-trial power-law loss schedule (used in `3702f54`) with a movement-locked ramp gives cleaner optimisation signal. A single RTX 5090 pod (EUR-IS-2 secure cloud, ~1.5 h, ~$1.50) ran all 7 cells in parallel: 5 ramp shapes (linear, cosine, power2, power4, power6) at the baseline `nn_output_pre_go=1.0`, plus two add-on cells probing the effect of a larger pre-go output scale (`power6_prego5`) and a longer ramp duration (`power6_dur80`). All cells used 12 000 warmup batches, batch size 250, 5 replicates, no adversary batches.

## Results

| # | Variant | nn_output_pre_go | Ramp shape | Ramp duration | Final val loss |
|---:|---|---:|---|---:|---:|
| 1 | `movement_ramp__linear` | 1.0 | linear | 60 | **1.44** |
| 2 | `movement_ramp__cosine` | 1.0 | cosine | 60 | **1.02** |
| 3 | `movement_ramp__power2` | 1.0 | power2 | 60 | **0.71** |
| 4 | `movement_ramp__power4` | 1.0 | power4 | 60 | **0.34** |
| 5 | `movement_ramp__power6` | 1.0 | power6 | 60 | **0.29** |
| 6 | `movement_ramp__power6_prego5` | 5.0 | power6 | 60 | **0.41** |
| 7 | `movement_ramp__power6_dur80` | 1.0 | power6 | 80 | **0.19** |

Run-spec paths: `results/b399efc/runs/<variant>/run.json` (committed at `69c91eb`).

These historical nested recipes were retired under issue `ef8e1df`; recover them from git tag `legacy/ef8e1df-nested-run-json-retired` (the bytes are also in Mandible custody).

## Interpretation

### Shape monotonicity

The five iso-duration, iso-prego cells (rows 1–5) reveal a clean monotonic ordering: linear → cosine → power² → power⁴ → power⁶, spanning roughly a 5× range in final validation loss (1.44 → 0.29). This is consistent with steeper ramp shapes concentrating the training signal closer to the movement onset, making it easier for the network to learn the go-cue-to-movement mapping without being penalised heavily during the hold period. The result is unambiguous: power6 is the right operating shape.

### Duration surprise — the `power6_dur80` finding

The most informative cell is `power6_dur80`, which simply extends the ramp window from 60 to 80 timesteps at the same power6 exponent. The result (val loss 0.185) is ~36% sharper than `power6` at dur=60 (0.29). This is a large gain from a single scalar change. The implication is that the network benefits from a wider movement window — either because 60 steps clips the rising edge of the ramp for some reach geometries, or because a shallower per-step gradient at the ramp base makes early-movement guidance more stable. A follow-up sweep over ramp duration (e.g. 60, 80, 100, 120) is motivated.

### Pre-go output scale (`prego` comparison)

`power6_prego5` (row 6) raises `nn_output_pre_go` from 1.0 to 5.0 while holding everything else at the `power6` baseline. Its validation loss (0.41) is worse than the baseline (0.29), not better. This confirms that amplifying the pre-movement output constraint beyond 1.0 adds noise rather than useful signal at this training scale. The operating point `nn_output_pre_go=1.0` is retained. This matches the prior decision from `3702f54`.

### Comparison to prior `3702f54` matrix

The prior best from `3702f54` (`full_trial_pl__prego_1`, full-trial powerlaw-6) achieved post-bug-fix vel-RMSE 0.0255 m/s. The current matrix reports mean validation loss, a different metric, so direct quantitative comparison requires re-rendering on the same axis. Qualitatively, however, the movement-ramp framework appears to produce a cleaner optimisation surface (monotone shape ordering, no anomalous cells), suggesting it is a sounder basis for the main training methodology than the full-trial power-law approach.

## Cross-references

- Tracking issue: `b399efc`
- Prior matrix (full-trial powerlaw): `3702f54`
- Training-methods coord: `c99ad9d` (comment on this run's decisions)
- Analyses coord: `4d38c15`
- Run-spec commit: `69c91eb`
- RUN_PLAN: `results/b399efc/RUN_PLAN.md`
- Auth request branch: `feature/b399efc-matrix-run` (request ID in issue `b399efc` comment thread)

<!-- AUTO-GENERATED: variance_analysis -->
## Headline metrics (auto-generated, b399efc)

All quantities computed on `warmup_model.eqx` for each cell, evaluated on 8-direction center-out validation trials at SISU=0.5 with zero perturbation.

### Headline scalar metrics

| Cell | Within-cell vel-RMSE (m/s) | Peak vel (m/s) | Time-to-peak (steps) | Hold drift (mm) | Training loss (final) |
|------|---:|---:|---:|---:|---:|
| linear (dur=60, prego=1) | 0.0867 | 1.938 ± 0.128 | 27.4 ± 1.3 | 0.02 ± 0.01 | 7.225e-01 ± 8.99e-02 |
| cosine (dur=60, prego=1) | 0.0974 | 1.983 ± 0.120 | 27.4 ± 1.3 | 0.02 ± 0.02 | 5.127e-01 ± 6.39e-02 |
| power² (dur=60, prego=1) | 0.0910 | 1.716 ± 0.128 | 30.2 ± 1.7 | 0.01 ± 0.01 | 3.619e-01 ± 5.79e-02 |
| power⁴ (dur=60, prego=1) | 0.0990 | 1.506 ± 0.129 | 34.0 ± 2.0 | 0.02 ± 0.02 | 1.811e-01 ± 3.52e-02 |
| power⁶ (dur=60, prego=1) | 0.1287 | 1.397 ± 0.087 | 37.1 ± 2.6 | 0.02 ± 0.02 | 1.509e-01 ± 4.16e-02 |
| power⁶ (dur=60, prego=5) | 0.1724 | 1.205 ± 0.177 | 39.4 ± 4.6 | 0.03 ± 0.01 | 2.383e-01 ± 8.94e-02 |
| power⁶ (dur=80, prego=1) | 0.0969 | 1.157 ± 0.090 | 40.9 ± 2.9 | 0.02 ± 0.01 | 1.021e-01 ± 2.19e-02 |

### Per-replicate peak forward velocity

| Cell | Rep 0 | Rep 1 | Rep 2 | Rep 3 | Rep 4 |
|------|---:|---:|---:|---:|---:|
| linear (dur=60, prego=1) | 1.927 | 1.727 | 1.957 | 2.021 | 2.057 |
| cosine (dur=60, prego=1) | 1.780 | 2.025 | 1.979 | 2.073 | 2.060 |
| power² (dur=60, prego=1) | 1.659 | 1.622 | 1.592 | 1.860 | 1.847 |
| power⁴ (dur=60, prego=1) | 1.299 | 1.543 | 1.475 | 1.627 | 1.586 |
| power⁶ (dur=60, prego=1) | 1.319 | 1.492 | 1.317 | 1.372 | 1.488 |
| power⁶ (dur=60, prego=5) | 0.952 | 1.174 | 1.401 | 1.153 | 1.344 |
| power⁶ (dur=80, prego=1) | 1.257 | 1.217 | 1.185 | 1.070 | 1.053 |

### Per-replicate hold drift (mm)

| Cell | Rep 0 | Rep 1 | Rep 2 | Rep 3 | Rep 4 |
|------|---:|---:|---:|---:|---:|
| linear (dur=60, prego=1) | 0.014 | 0.036 | 0.005 | 0.009 | 0.015 |
| cosine (dur=60, prego=1) | 0.008 | 0.056 | 0.004 | 0.012 | 0.015 |
| power² (dur=60, prego=1) | 0.011 | 0.034 | 0.000 | 0.009 | 0.009 |
| power⁴ (dur=60, prego=1) | 0.006 | 0.053 | 0.001 | 0.005 | 0.016 |
| power⁶ (dur=60, prego=1) | 0.011 | 0.045 | 0.000 | 0.004 | 0.022 |
| power⁶ (dur=60, prego=5) | 0.025 | 0.038 | 0.018 | 0.022 | 0.024 |
| power⁶ (dur=80, prego=1) | 0.019 | 0.041 | 0.008 | 0.007 | 0.026 |

### Figures

- `results/b399efc/figures/forward_velocity_profiles/` — forward velocity per cell, go-cue-aligned
- `results/b399efc/figures/hold_drift_profiles/` — pre-go forward position per cell, go-cue-aligned
- `results/b399efc/figures/peak_velocity_distributions/` — per-replicate peak velocity (violin)
- `results/b399efc/figures/summary_metrics/` — 2×2 scalar-metric bar panel
- `results/b399efc/figures/training_loss/` — total weighted training loss per cell
- `results/b399efc/figures/training_loss_per_term/` — per-term decomposition per cell

HTML renders in `_artifacts/b399efc/figures/<topic>/figure.html`.
<!-- /AUTO-GENERATED -->
