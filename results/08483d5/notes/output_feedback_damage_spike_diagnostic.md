# 6D output-feedback H-infinity damage spike diagnostic

## Method

- No-launch diagnostic for issue `08483d5`.
- Reused the deterministic no-noise output-feedback H-infinity conventions from `results/08483d5/scripts/compute_output_feedback_damage_beta_curve_dense.py`.
- No cap, radius, trust-region, PGD, or training defaults enter this calculation.
- For every beta, the Riccati solution, robust estimator covariance, feedback gains, and fixed optimal disturbance policy are recomputed.
- Direct deterministic rollouts were computed for every row and compared with the affine value recursion.

## Headline

- Max damage beta: `1.342` with `D=166706128571`.
- At beta 1.350: `D=134393.885795`, `D_x=118858.458335`, `D_u=-97.6347851105`, `D_T=15633.0622454`, `E_w=0.000868684094201`, `D/E_w=154709734.749`.
- Max rollout-vs-recursion damage mismatch across all rows: `0.000327060377458`.
- All finite checks passed: `True`.

## Spike-neighborhood rows

| beta | D | D_x | D_u | D_T | E_w | D/E_w | max ||x|| adv | max ||u|| adv | max ||w|| adv | recursion mismatch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.345 | 923665 | 814712 | 280.327 | 108673 | 0.00605388 | 1.52574e+08 | 18.3217 | 20.7839 | 0.0158372 | 3.13e-08 |
| 1.346 | 522794 | 461367 | 54.0319 | 61372.8 | 0.00341708 | 1.52994e+08 | 18.3195 | 20.782 | 0.0118878 | 2.85e-08 |
| 1.347 | 336833 | 297413 | -33.6392 | 39453.3 | 0.00219552 | 1.53418e+08 | 18.3175 | 20.7802 | 0.00952027 | 8.76e-08 |
| 1.348 | 235539 | 208085 | -72.0787 | 27526 | 0.00153101 | 1.53845e+08 | 18.3157 | 20.7783 | 0.00794282 | 1.19e-08 |
| 1.349 | 174279 | 154049 | -89.7426 | 20320 | 0.00112966 | 1.54276e+08 | 18.314 | 20.7765 | 0.00681649 | 2.44e-09 |
| 1.350 | 134394 | 118858 | -97.6348 | 15633.1 | 0.000868684 | 1.5471e+08 | 18.3123 | 20.7747 | 0.00597196 | 3.2e-09 |
| 1.351 | 106960 | 94648.2 | -100.597 | 12412.6 | 0.000689413 | 1.55147e+08 | 18.3107 | 20.7728 | 0.00531523 | 1.05e-08 |
| 1.352 | 87272.4 | 77269.7 | -100.964 | 10103.7 | 0.000560923 | 1.55587e+08 | 18.3091 | 20.771 | 0.00478991 | 8.47e-09 |
| 1.353 | 72657.2 | 64365.7 | -99.9374 | 8391.4 | 0.00046566 | 1.56031e+08 | 18.3075 | 20.7692 | 0.00436015 | 6.66e-09 |
| 1.354 | 61503.8 | 54515.9 | -98.1675 | 7085.99 | 0.000393053 | 1.56477e+08 | 18.3059 | 20.7674 | 0.00400204 | 9.8e-09 |
| 1.355 | 52794.5 | 46822.9 | -96.0187 | 6067.65 | 0.000336429 | 1.56926e+08 | 18.3043 | 20.7655 | 0.00369904 | 6.05e-09 |

## Landmark rows

| beta | D | D_x | D_u | D_T | E_w | D/E_w | max ||x|| adv | max ||u|| adv | max ||w|| adv | recursion mismatch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.300 | 3505.26 | 3064.25 | 10.0744 | 430.935 | 2.52669e-05 | 1.38729e+08 | 18.3929 | 20.8725 | 0.00105251 | 7.25e-08 |
| 1.325 | 24825.7 | 21714.3 | 84.0464 | 3027.39 | 0.000171173 | 1.45033e+08 | 18.3505 | 20.822 | 0.00270612 | 3.88e-05 |
| 1.350 | 134394 | 118858 | -97.6348 | 15633.1 | 0.000868684 | 1.5471e+08 | 18.3123 | 20.7747 | 0.00597196 | 3.2e-09 |
| 1.375 | 9516.41 | 8540.64 | -61.9692 | 1037.73 | 5.71747e-05 | 1.66444e+08 | 18.2744 | 20.7301 | 0.00149489 | 8.09e-10 |
| 1.400 | 3704.96 | 3373.81 | -45.8305 | 376.987 | 2.06571e-05 | 1.79355e+08 | 18.2392 | 20.6882 | 0.000875707 | 1.66e-09 |

## Conditioning at max damage

- Riccati spectral-radius max: `0.0724054935715`.
- Riccati spectral-radius margin min: `0.927594506429`.
- Riccati bracket condition max: `27559.2201865`.
- Riccati max P condition: `1.03729717024e+12`.
- Estimator precision min eigenvalue: `0.0627923178388`.
- Estimator precision max condition: `1283.78466997`.
- Gain operator norm max: `883.950014275`.
- Policy operator norm max: `3.33219578026`.
- Adversarial joint cumulative operator norm max: `138396.646507`.

## Interpretation

The spike is real under the implemented deterministic closed-loop equations, because direct rollout and value recursion agree, but it is a stress-test condition rather than a useful curriculum target. At the maximum row the damage is state-running-dominated, and the conditioning/transient diagnostics flag: direct rollout state norm jumps 259.3x above clean; direct rollout control norm jumps 155.8x above clean.

## Outputs

- CSV: `results/08483d5/notes/output_feedback_damage_spike_diagnostic.csv`
- JSON: `results/08483d5/notes/output_feedback_damage_spike_diagnostic.json`
- Markdown: `results/08483d5/notes/output_feedback_damage_spike_diagnostic.md`
- Damage plot: `_artifacts/08483d5/figures/output_feedback_damage_spike_diagnostic/spike_damage_components.png`
- Conditioning plot: `_artifacts/08483d5/figures/output_feedback_damage_spike_diagnostic/spike_conditioning.png`

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --no-sync python results/08483d5/scripts/compute_output_feedback_damage_spike_diagnostic.py
```
