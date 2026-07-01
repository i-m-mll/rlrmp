# Adaptive-lambda frozen replay for 08483d5

## Headline

fail: finite/nonzero may hold, but damage tracking or guard independence failed.

This run did not update controller weights and wrote no repo files or ledger state.

## Resolved model/baseline identity

- Requested exact baseline: `6D no-PGD H0 const_band16 baseline/model`.
- Exact local status: `not_materialized_as_a single local artifact`.
- Checked candidates:
  - `results/33b0dcb/runs/h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64.json`: materialized_but_8d_48d_not_6d_no_integrator.
  - `results/ffff699/runs/delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed42/run.json`: 6d_no_integrator_but_artifact_directory_empty_and_not_const_band16.
- Empirical replay source used:
  - `ae9f30f/direct_epsilon_b1p05` at checkpoint 12000.
  - Run spec: `results/ae9f30f/runs/direct_epsilon_b1p05.json`.
  - Model: `_artifacts/ae9f30f/runs/direct_epsilon_b1p05/checkpoints/checkpoint_0012000/model.eqx`.
  - Contract: no_integrator=True, physical_state_dim=6, state_dim=36, epsilon_dim=6, initial_hidden_encoder=True.

## Lambda source

- Source: `results/06a4dc8/canonical_soft_lambda_hvp.json`.
- `lambda_curv_p90`: `254905215`.
- Initial `lambda0 = beta^2 * lambda_curv_p90` for beta=1.05: `281032999`.
- The source validator accepted this as cap-independent; cap/trust radius was not used as the lambda criterion.

## Lambda-zero bracket

Coarse bracket: active at `281032999`, inactive at `843098998`.

| lambda | damage | energy_mean | objective_gain | nonzero | cap_boundary | finite |
|---:|---:|---:|---:|:---:|---:|:---:|
| 2.81033e+10 | 0 | 0 | -1.28585e-07 | False | 0 | True |
| 8.43099e+09 | 0 | 0 | -1.28585e-07 | False | 0 | True |
| 2.81033e+09 | 0 | 0 | -1.28585e-07 | False | 0 | True |
| 8.43099e+08 | 0 | 0 | -1.28585e-07 | False | 0 | True |
| 2.81033e+08 | 274.483 | 6.73176e-07 | 85.298 | True | 0 | True |
| 8.43099e+07 | 501.603 | 1.51889e-06 | 373.546 | True | 1 | True |
| 2.81033e+07 | 499.564 | 1.51889e-06 | 456.879 | True | 1 | True |
| 8.43099e+06 | 498.555 | 1.51888e-06 | 485.749 | True | 1 | True |
| 2.81033e+06 | 498.245 | 1.51889e-06 | 493.977 | True | 1 | True |

## Update rule

For iteration `k`, rerun the frozen inner maximizer from zero epsilon at the current lambda, compute paired damage `D_k = J_selected - J_zero`, then update:

`log(lambda_{k+1}) = log(lambda_k) + clip(eta * log(max(D_k, eps) / D_ref), +/- max_log_step)`

Here `D_ref = 6131.690677`, `eta = 0.5`, `max_log_step = 0.75`. If damage is below target, lambda decreases; if damage is above target, lambda increases.

## Adaptive replay diagnostics

| iter | lambda | damage | damage/ref | energy_mean | objective | objective_gain | nonzero | cap_boundary | finite | next_lambda |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|---:|
| 0 | 2.81033e+08 | 274.483 | 0.04476 | 6.73176e-07 | 4281.43 | 85.298 | True | 0 | True | 1.32751e+08 |
| 1 | 1.32751e+08 | 502.28 | 0.08192 | 1.51889e-06 | 4496.78 | 300.647 | True | 1 | True | 6.27069e+07 |
| 2 | 6.27069e+07 | 500.982 | 0.0817 | 1.51889e-06 | 4601.87 | 405.737 | True | 1 | True | 2.96207e+07 |
| 3 | 2.96207e+07 | 499.639 | 0.08148 | 1.51889e-06 | 4650.78 | 454.649 | True | 1 | True | 1.39918e+07 |
| 4 | 1.39918e+07 | 498.851 | 0.08136 | 1.51889e-06 | 4673.73 | 477.599 | True | 1 | True | 6.60926e+06 |
| 5 | 6.60926e+06 | 498.454 | 0.08129 | 1.51888e-06 | 4684.55 | 488.416 | True | 1 | True | 3.12199e+06 |
| 6 | 3.12199e+06 | 498.26 | 0.08126 | 1.51889e-06 | 4689.65 | 493.518 | True | 1 | True | 1.47473e+06 |
| 7 | 1.47473e+06 | 498.167 | 0.08124 | 1.51889e-06 | 4692.06 | 495.927 | True | 1 | True | 696611 |

## Pass/fail assessment

- Pass: `False`.
- Finite all iterations: `True`.
- Nonzero all iterations: `True`.
- Any guard/cap binding: `True`.
- First damage: `274.483`; last damage: `498.167`; target: `6131.69`.
- Moved toward target: `True`.

Blockers and uncertainties:
- selected adversary was guard/cap-bound in at least one adaptive row.
- Exact requested 6D no-PGD H0 const_band16 baseline checkpoint was not found as a runnable local artifact; the replay source is a materialized 6D direct-epsilon checkpoint.
- The replay uses the existing projected direct-epsilon inner optimizer with its stabilization cap, so any cap-bound row fails the pure-soft interpretation.
- This is a single small frozen batch and should be treated as a mechanism diagnostic, not a launch result.
