<!-- AUTO-GENERATED: beta1p4_stabilization_diagnostics -->
# Beta 1.4 Stabilization Diagnostics

Endpoint stabilization probes reuse the c92 probe contract. AUC values are mean signed-direction-aligned absolute hand-position displacement after probe onset in `mm*s`.

| Row | Source | Training condition | Feedback AUC | Mechanical AUC | Command AUC | Process-force AUC | Feedback peak | Mechanical peak |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `baseline_no_pgd_h0_const_band16` | `33b0dcb` | no-PGD H0 6D open-loop moderate const_band16 baseline | 8.999 | 2.709 | 0.8936 | 4.524 | 31.34 | 8.31 |
| `direct_epsilon` | `b413bb0` | beta 1.4 direct-epsilon PGD | 7.66 | 0.7374 | 0.9446 | 0.5302 | 24.37 | 2.237 |
| `linear_no_bias` | `b413bb0` | beta 1.4 finite linear no-bias adversary | 7.216 | 1.803 | 1.803 | not available | 29.05 | 6.927 |
| `affine` | `b413bb0` | beta 1.4 finite affine adversary | 4.442 | 1.979 | 1.979 | not available | 15.36 | 7.727 |

## Baseline Comparisons

Negative AUC deltas mean lower endpoint displacement than the no-PGD H0 const_band16 baseline.

| Row | Feedback AUC delta | Mechanical AUC delta | Command AUC delta | Process-force AUC delta |
|---|---:|---:|---:|---:|
| `direct_epsilon` | -1.339 | -1.972 | 0.05103 | -3.994 |
| `linear_no_bias` | -1.784 | -0.9058 | 0.9096 | not available |
| `affine` | -4.558 | -0.7304 | 1.085 | not available |
<!-- /AUTO-GENERATED -->
