# Run Plan

Tracking issue: `b399efc`

## Matrix overview

Seven warmup-only training cells over the movement-locked position ramp
schedule. The matrix sweeps ramp **shape** (linear, cosine, power-2/4/6) at
fixed `nn_output_pre_go=1.0` and ramp duration 60, plus two add-on cells that
probe (a) heavier pre-go penalty under the prior winner's shape and (b) longer
ramp duration under the same shape.

| # | Variant name | `nn_output_pre_go` | Ramp shape | Ramp duration |
|---:|---|---:|---|---:|
| 1 | `movement_ramp__linear` | 1.0 | linear | 60 |
| 2 | `movement_ramp__cosine` | 1.0 | cosine | 60 |
| 3 | `movement_ramp__power2` | 1.0 | power 2 | 60 |
| 4 | `movement_ramp__power4` | 1.0 | power 4 | 60 |
| 5 | `movement_ramp__power6` | 1.0 | power 6 | 60 |
| 6 | `movement_ramp__power6_prego5` | 5.0 | power 6 | 60 |
| 7 | `movement_ramp__power6_dur80` | 1.0 | power 6 | 80 |

## Rationale

### Why `nn_output_pre_go = 1.0` for cells 1–5

The prior smoke matrix (`smoke__full_trial_pl_prego10_1k`,
`smoke__full_trial_pl_prego100_1k`, both at 1000-batch warmup) tested
saturating the pre-go output penalty under the previous best full-trial
powerlaw schedule. Both saturated values were worse than the prior `prego=1`
baseline:

| Cell | `nn_output_pre_go` | Final validation loss (1k batches) |
|---|---:|---:|
| `smoke__full_trial_pl_prego100_1k` | 100.0 | `6.33e+00` |
| `smoke__full_trial_pl_prego10_1k` | 10.0 | `5.19e+00` |
| Prior `full_trial_pl__prego_1` (1k batches) | 1.0 | `5.11e+00` at iter 600, `1.65e+00` at iter 1200 |

See `results/b399efc/notes/smoke_tests.md` for full evidence. The matrix
therefore pins the shape sweep to `prego=1.0` and isolates ramp curvature as
the experimental factor.

### Why `power6` for the `prego=5` add-on (cell 6)

`prego=10` saturated the network in the smoke. `prego=1` may underweight the
pre-go penalty under a stricter movement-locked schedule (the ramp's
near-zero early weight could need slightly heavier pre-go to suppress
anticipatory ramp-up of motor output). Cell 6 tests `prego=5` — between the
finite-but-too-soft `1` and the saturated `10` — under the prior best
power-law shape (`power=6`, matching `results/3702f54/` winner).

### Why `dur=80` for the duration add-on (cell 7)

Cells 1–5 hold ramp duration fixed at 60 (default). Cell 7 probes sensitivity
to ramp duration at the best-guess shape (`power6`): does extending the ramp
to 80 steps further smooth motor onset, or does it just delay useful position
penalisation? Holding `prego=1` so the comparison vs cell 5 isolates duration.

## Shared CLI base flags

All 7 cells use:

- `--adversary-type linear_dynamics`
- `--n-warmup-batches 12000`
- `--n-adversary-batches 0`
- `--batch-size 250`
- `--n-replicates 5`
- `--hidden-type gru`
- `--seed 42`
- `--effector-hold-pos 0.0` `--effector-hold-vel 0.0`
- `--effector-pos-running 1.0`
- `--effector-pos-late-weight 0.0` `--effector-vel-late 0.0` `--effector-final-vel 0.0`
- `--p-catch-trial 0.5`
- `--nn-output 1e-5` `--nn-hidden 1e-5` `--nn-output-jerk 0.0` `--nn-hidden-derivative 1e-3` `--nn-hidden-derivative-pre-go 0.0`
- `--no-loss-update-enabled`
- `--effector-pos-running-schedule movement_ramp`
- `--effector-hold-pos-schedule flat`
- `--checkpoint` `--fused` `--no-streaming-loss` `--checkpoint-every 1000`

Per-cell flags vary `--movement-ramp-shape`, `--movement-ramp-power`,
`--movement-ramp-duration-steps`, and `--nn-output-pre-go` per the table above.

## Expected per-cell wall-clock

Prior 5090 timing: ~30s JIT compilation + ~3-5 minutes for 1000 batches at
`n_replicates=5`, `batch_size=250`, GRU. Extrapolated to 12,000 batches per
cell: roughly **30–45 minutes per cell**. Seven cells sequential ⇒ **~4–5
hours wall-clock**. Two cells parallel (if VRAM allows) ⇒ **~2–3 hours**.

## Outputs

Each cell writes to `_artifacts/b399efc/runs/<variant>/` on the pod. Final
`run.json` specs are committed to `results/b399efc/runs/<variant>.json` after
all cells complete (per CLAUDE.md §9 post-training-run protocol).

Historical nested run recipes were retired under issue `ef8e1df`; recover them from git tag `legacy/ef8e1df-nested-run-json-retired` (the bytes are also in Mandible custody).

Co-authored-by: Claude Opus 4.7 <noreply@anthropic.com>, Codex (GPT-5) <codex@openai.com>
