<!-- AUTO-GENERATED: sisu_spectrum_special -->
# SISU Spectrum Special Analysis

This is a SISU-conditioned post-run analysis for the two e4800d6 H0 spectrum rows. It is discovery-trained robustness evidence, not teacher/distillation behavior and not formal H-infinity equivalence.

**Low-SISU check:** Low-SISU reaching check passed: SISU 0.0 and 0.5 have endpoint errors <= 0.050 m and peak speeds >= 0.200 m/s in both targetfix rows.

## Velocity Profiles

- Figure spec: `results/e4800d6/figures/sisu_spectrum_velocity_profiles_targetfix/spec.json`
- Figure render: `_artifacts/e4800d6/figures/sisu_spectrum_velocity_profiles_targetfix/figure.html`
- Compact arrays: `_artifacts/e4800d6/sisu_spectrum_special_targetfix/sisu_velocity_profile_curves.npz`

## Within-Network SISU=1 vs SISU=0 Comparison

| row | SISU=0 endpoint (m) | SISU=1 endpoint (m) | endpoint ratio 1/0 | SISU=0 peak (m/s) | SISU=1 peak (m/s) | peak ratio 1/0 |
|---|---:|---:|---:|---:|---:|---:|
| raw strong gamma-1.05 targetfix | 0.003008 | 0.001176 | 0.39095 | 0.734671 | 0.772513 | 1.05 |
| effective 020a65b PGD targetfix | 0.003104 | 0.001408 | 0.45366 | 0.736338 | 0.784026 | 1.06 |

## Input Contract

Both rows use the SISU scalar on `trial_specs.inputs['input']`; the materialized validation bank has `input = 1.0` by default and no separate `sisu` key. The special profile materializer therefore changes `input` to 0.0, 0.5, and 1.0 and zeroes `epsilon` for the nominal profile comparison.

## Per-SISU Metrics

| row | SISU | endpoint error mean (m) | peak velocity mean (m/s) | final position mean (m) |
|---|---:|---:|---:|---|
| raw strong gamma-1.05 targetfix | 0 | 0.003008 | 0.734671 | [0.147111, 0.000187] |
| raw strong gamma-1.05 targetfix | 0.5 | 0.001085 | 0.761831 | [0.149666, 0.000116] |
| raw strong gamma-1.05 targetfix | 1 | 0.001176 | 0.772513 | [0.149632, 0.000028] |
| effective 020a65b PGD targetfix | 0 | 0.003104 | 0.736338 | [0.147064, 0.000438] |
| effective 020a65b PGD targetfix | 0.5 | 0.001373 | 0.768374 | [0.149926, 0.000171] |
| effective 020a65b PGD targetfix | 1 | 0.001408 | 0.784026 | [0.149990, 0.000121] |
<!-- /AUTO-GENERATED -->
