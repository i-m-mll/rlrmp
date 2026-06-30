# ae9f30f Post-run Materialization Summary
This summary covers the three completed 12k rows as trained-model evidence and keeps `linear_no_bias_b1p4` as stopped/suppressed context only. Old hard-cap ratios are not used as pass/fail criteria.
## Materialized artifacts
- Run specs: `results/ae9f30f/runs/<row>.json` for all four rows, including the stopped row.
- Bulk runs: `_artifacts/ae9f30f/runs/<row>/`; the stopped row has checkpoint/log/sentinel context, not a completed `training_summary.json`.
- Velocity figure: `results/ae9f30f/figures/nominal_velocity_overlay/figure.html` with spec at `results/ae9f30f/figures/nominal_velocity_overlay/spec.json`.
- Standard bundle: `results/ae9f30f/notes/gru_postrun_materialization_soft_lambda_wave1_trained_final.json`.

## Training and adversary diagnostics
| row | status | mechanism | beta | lambda | batches | final train loss | final validation loss | adv energy mean | adv energy max | adv gain | radius ratio mean/max | nonfinite |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_epsilon_b1p05` | completed | `direct_epsilon` | 1.05 | 2.546e+08 | 1.2e+04 | 4928 | 4481 | 1.087e-06 | 1.519e-06 | 161.7 | 0.8211/1 | 0 |
| `direct_epsilon_b1p4` | completed | `direct_epsilon` | 1.4 | 4.526e+08 | 1.2e+04 | 4694 | 4430 | 3.452e-07 | 1.519e-06 | 94.64 | 0.4359/1 | 0 |
| `linear_no_bias_b1p05` | completed | `linear_no_bias` | 1.05 | 2.546e+08 | 1.2e+04 | 4439 | 4392 | 0 | 0 | 0 | 0/0 | 0 |
| `linear_no_bias_b1p4` | spec_only | `linear_no_bias` | 1.4 | 4.526e+08 | missing | missing | missing | missing | missing | missing | missing/missing | missing |

## Nominal behavior and velocity profile
| row/reference | peak forward velocity (m/s) | time to peak (s) | endpoint error (m) | terminal speed (m/s) | command norm mean |
|---|---:|---:|---:|---:|---:|
| `direct_epsilon_b1p05` | 0.7546 | 0.16 | 0.001272 | 0.002526 | 3.413 |
| `direct_epsilon_b1p4` | 0.745 | 0.16 | 0.001772 | 0.002456 | 3.33 |
| `linear_no_bias_b1p05` | 0.7319 | 0.16 | 0.003254 | 0.002128 | 3.262 |
| C&S H-infinity nominal 6D | 0.7606 | 0.16 | n/a | n/a | n/a |
| C&S extLQG/output-feedback 8D | 0.7289 | 0.16 | n/a | n/a | n/a |

`linear_no_bias_b1p4` is omitted from the trained-model profile because it stopped at the 1000-batch gate after accepted adversary suppression.

## Standard certificate and map sidecars
| row | certificate status | classification | state-weighted action mismatch | delta/ref RMS | map status | transition status |
|---|---|---|---:|---:|---|---|
| `direct_epsilon_b1p05` | `partial_standard_certificate_blocked` | `external_rollout_mismatch` | 2.071 | 7.618/5.294 | `missing` | `not_applicable` |
| `direct_epsilon_b1p4` | `partial_standard_certificate_blocked` | `external_rollout_mismatch` | 2.039 | 7.559/5.294 | `missing` | `not_applicable` |
| `linear_no_bias_b1p05` | `partial_standard_certificate_blocked` | `external_rollout_mismatch` | 1.998 | 7.483/5.294 | `missing` | `not_applicable` |

The response-map components remain blocked by a 6D GRU feedback basis versus 8D analytical output-feedback reference contract mismatch; transition/value/Bellman rows are not applicable to nonlinear GRU rows without an approved local-linear certificate.

| row | map aggregate delta ratio | candidate/reference cosine | norm ratio | best scalar gain | scalar residual ratio |
|---|---:|---:|---:|---:|---:|
| `direct_epsilon_b1p05` | 1.197 | 0.1597 | 0.6314 | 0.1008 | 0.3885 |
| `direct_epsilon_b1p4` | 1.142 | 0.1521 | 0.5579 | 0.08483 | 0.304 |
| `linear_no_bias_b1p05` | 1.545 | 0.1146 | 0.8619 | 0.09875 | 0.7332 |

## Perturbation and feedback diagnostics
- The generic final-checkpoint postrun bundle materialized standard, evaluation, figure, and map-decomposition diagnostics.
- Calibrated perturbation-response and feedback-ablation sidecars did not complete in this worker pass. The first postrun attempt skipped because `canonical_15cm` is not a recognized calibration reach; the numeric `0.15` retry entered JAX compilation/evaluation for the perturbation bank and was interrupted after several silent minutes.
- Therefore sensory/non-sensory perturbation loss gains, AUC diagnostics, stabilization, and early/mid/late reach perturbation tables are missing from this pass; the missing input is a completed perturbation-response/feedback-ablation manifest for these `ae9f30f` final checkpoints. Existing baseline artifacts for `020a65b` are present, but no same-bank `ae9f30f` diagnostic table was completed to compare against them.

## Interpretation
- Direct epsilon rows trained with nonzero adversaries through 12k. The beta 1.05 row reaches a nominal peak velocity close to the deterministic 6D H-infinity trace; beta 1.4 is lower but still above the extLQG output-feedback reference.
- `linear_no_bias_b1p05` completed but its late-run adversary diagnostics are zero, so its nominal profile is near extLQG and should not be treated as evidence that the linear mechanism solved the robust training objective.
- `linear_no_bias_b1p4` remains stopped/suppressed context, not a trained-model conclusion.
- Euclid audit result included here: no evidence was found for beta/lambda propagation, batch mean/sum, sign, or `adv_penalty=lambda*energy` regression. The plausible current explanation is that the training-time finite-policy fixed-step optimizer missed or suppressed a narrow positive basin, but that remains an open question.
