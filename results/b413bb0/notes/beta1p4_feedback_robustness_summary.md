<!-- AUTO-GENERATED: beta1p4_feedback_robustness_summary -->
# Beta 1.4 Feedback and Robustness Summary

This summary compares the three completed b413 beta 1.4 rows with the existing 33b0dcb H0 no-PGD `const_band16` baseline. All values use validation-selected checkpoints and the calibrated moderate perturbation bank.

## Sensory and Non-Sensory Deviations

| Row | Sensory max (mm) | Sensory AUC (mm*s) | Non-sensory max (mm) | Non-sensory AUC (mm*s) | Peak dx/OL | AUC dx/OL |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_no_pgd_h0_const_band16` | 2.685 | 0.6212 | 7.28 | 2.04 | 0.477 | not available |
| `direct_epsilon` | 2.552 | 0.6356 | 6.992 | 1.979 | 0.4537 | not available |
| `linear_no_bias` | 3.466 | 0.7215 | 10.42 | 3.339 | 0.6271 | not available |
| `affine` | 2.388 | 0.5213 | 9.668 | 3.533 | 0.5725 | not available |

## Feedback and Stabilization

| Row | Feedback delta action | Ablation dependence | Feedback audit | Stab feedback AUC | Stab mechanical AUC | Stab command AUC | Stab process-force AUC |
|---|---:|---:|---|---:|---:|---:|---:|
| `baseline_no_pgd_h0_const_band16` | 0.5446 | 2.938 | `pass` | 8.999 | 2.709 | 0.8936 | 4.524 |
| `direct_epsilon` | 0.5983 | 3.36 | `pass` | 7.66 | 0.7374 | 0.9446 | 0.5302 |
| `linear_no_bias` | 0.65 | 2.854 | `pass` | 7.216 | 1.803 | 1.803 | not available |
| `affine` | 0.2277 | 2.414 | `warn` | 4.442 | 1.979 | 1.979 | not available |

## Deltas Against Baseline

Negative deviation/AUC deltas mean lower response than the baseline.

| Row | Sensory AUC delta | Non-sensory AUC delta | Feedback AUC delta | Mechanical AUC delta | Feedback delta-action delta |
|---|---:|---:|---:|---:|---:|
| `direct_epsilon` | 0.01434 | -0.06129 | -1.339 | -1.972 | 0.0537 |
| `linear_no_bias` | 0.1003 | 1.299 | -1.784 | -0.9058 | 0.1054 |
| `affine` | -0.09996 | 1.493 | -4.558 | -0.7304 | -0.3169 |

## Output Manifests

- Summary JSON: `results/b413bb0/notes/beta1p4_feedback_robustness_summary.json`
- Stabilization detail: `_artifacts/b413bb0/stabilization_diagnostics/beta1p4_stabilization_diagnostics/per_probe_detail.json`
<!-- /AUTO-GENERATED -->
