<!-- AUTO-GENERATED: soft_lambda_sweep -->
# Soft lambda estimator and direct-epsilon sweep

Issue: `093d949`. Source no-PGD runs: `c92ebd8`.

No controller weights were updated. This is a frozen-batch audit of existing c92 open-loop no-PGD substrates.

## Lambda estimates

| row | old beta lambda | B-corrected beta lambda | per-trial p90 beta lambda | old grad floor | B-corrected grad floor | per-trial p90 grad floor | per-trial p90 curvature |
|---|---:|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 1.19627e+07 | 9.57013e+07 | 1.11397e+08 | 6.1034e+06 | 4.88272e+07 | 5.6835e+07 | 5.25579e+06 |
| `open_loop_moderate` | 1.11209e+07 | 8.89672e+07 | 1.02932e+08 | 5.67393e+06 | 4.53914e+07 | 5.25161e+07 | 4.79675e+06 |
| `open_loop_stress` | 1.2052e+07 | 9.64158e+07 | 1.35806e+08 | 6.14897e+06 | 4.91918e+07 | 6.92886e+07 | 4.55017e+06 |

## Direct-epsilon sweep

The sweep is centered on `per_trial_p90.lambda_beta`; the batch-corrected value is reported only as a comparison.

| row | multiplier | lambda | norm/cap max | cap-bound | raw loss gain | energy penalty | penalized gain | finite |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `open_loop_small` | 0.25 | 2.78491e+07 | 1 | 100.000% | 7202.89 | 575.407 | 6627.48 | finite |
| `open_loop_small` | 0.5 | 5.56983e+07 | 1 | 100.000% | 7205.63 | 1150.81 | 6054.81 | finite |
| `open_loop_small` | 1 | 1.11397e+08 | 1 | 100.000% | 7209.16 | 2301.63 | 4907.53 | finite |
| `open_loop_small` | 2 | 2.22793e+08 | 1 | 100.000% | 7210.61 | 4603.26 | 2607.36 | finite |
| `open_loop_small` | 4 | 4.45586e+08 | 0.25 | 0.000% | 806.72 | 575.407 | 231.314 | finite |
| `open_loop_moderate` | 0.25 | 2.57329e+07 | 1 | 100.000% | 6473.6 | 531.682 | 5941.92 | finite |
| `open_loop_moderate` | 0.5 | 5.14658e+07 | 1 | 100.000% | 6481.42 | 1063.36 | 5418.06 | finite |
| `open_loop_moderate` | 1 | 1.02932e+08 | 1 | 100.000% | 6490.77 | 2126.73 | 4364.04 | finite |
| `open_loop_moderate` | 2 | 2.05863e+08 | 1 | 100.000% | 6493.86 | 4253.45 | 2240.4 | finite |
| `open_loop_moderate` | 4 | 4.11726e+08 | 0.25 | 0.000% | 709.501 | 531.682 | 177.819 | finite |
| `open_loop_stress` | 0.25 | 3.39514e+07 | 1 | 100.000% | 6179.08 | 701.489 | 5477.59 | finite |
| `open_loop_stress` | 0.5 | 6.79028e+07 | 1 | 100.000% | 6188.99 | 1402.98 | 4786.01 | finite |
| `open_loop_stress` | 1 | 1.35806e+08 | 1 | 100.000% | 6197.1 | 2805.96 | 3391.14 | finite |
| `open_loop_stress` | 2 | 2.71611e+08 | 0.995226 | 0.000% | 3725.7 | 2992.75 | 732.953 | finite |
| `open_loop_stress` | 4 | 5.43222e+08 | 0.25 | 0.000% | 723.765 | 701.489 | 22.2769 | finite |

## Transition read

- `open_loop_small`: p90 estimate lands cap-dominated, but the narrow grid brackets the transition above it; center cap-bound = `100.000%`, last cap-dominated multiplier = `2.0`, first interior multiplier = `4.0`.
- `open_loop_moderate`: p90 estimate lands cap-dominated, but the narrow grid brackets the transition above it; center cap-bound = `100.000%`, last cap-dominated multiplier = `2.0`, first interior multiplier = `4.0`.
- `open_loop_stress`: p90 estimate lands cap-dominated, but the narrow grid brackets the transition above it; center cap-bound = `100.000%`, last cap-dominated multiplier = `1.0`, first interior multiplier = `2.0`.

## Runtime caveats

- Curvature uses bounded finite directions, not HVP/power iteration. HVP remains a later estimator-strengthening option rather than a separate adversary mechanism.
- The direct soft-energy objective reduction remains the existing `mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]` code path.
<!-- /AUTO-GENERATED -->
