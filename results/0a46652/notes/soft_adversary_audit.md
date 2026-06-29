# Soft adversary audit

Issue: `0a46652`. Source no-PGD runs: `c92ebd8`.

## Lambda estimates

| row | lambda_star | beta=1.4 lambda | grad floor | max curvature |
|---|---:|---:|---:|---:|
| `open_loop_small` | 6.1034e+06 | 1.19627e+07 | 6.1034e+06 | 5.03136e+06 |
| `open_loop_moderate` | 5.67393e+06 | 1.11209e+07 | 5.67393e+06 | 4.49428e+06 |
| `open_loop_stress` | 6.14897e+06 | 1.2052e+07 | 6.14897e+06 | 4.01264e+06 |

## Frozen-batch audits

| row | mechanism | status | gain over zero | energy mean | norm max | cap-bound |
|---|---|---|---:|---:|---:|---:|
| `open_loop_small` | `open_loop_direct_epsilon` | no_lock | 6953.89 | 2.06616e-05 | 0.0045455 | 100.000% |
| `open_loop_small` | `closed_loop_linear_no_bias` | no_lock | 0 | 0 | 0 | 0.000% |
| `open_loop_small` | `closed_loop_affine` | no_lock | 0 | 0 | 0 | 0.000% |
| `open_loop_moderate` | `open_loop_direct_epsilon` | no_lock | 6238.49 | 2.06616e-05 | 0.0045455 | 100.000% |
| `open_loop_moderate` | `closed_loop_linear_no_bias` | no_lock | 0 | 0 | 0 | 0.000% |
| `open_loop_moderate` | `closed_loop_affine` | no_lock | 0 | 0 | 0 | 0.000% |
| `open_loop_stress` | `open_loop_direct_epsilon` | no_lock | 5921.62 | 2.06616e-05 | 0.0045455 | 100.000% |
| `open_loop_stress` | `closed_loop_linear_no_bias` | no_lock | 0 | 0 | 0 | 0.000% |
| `open_loop_stress` | `closed_loop_affine` | no_lock | 0 | 0 | 0 | 0.000% |

No controller weights were updated. These are local frozen-batch audits only.
