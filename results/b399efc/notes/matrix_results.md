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
