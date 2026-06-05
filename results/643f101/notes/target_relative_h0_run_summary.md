# Target-Relative Multi-Target H0 Encoder Run Summary

Issue: `643f101`.

This note summarizes the two local C&S GRU rows that add an initial hidden-state
encoder to the prior target-relative multi-target perturbation-training setup.
All post-run metrics below use validation-selected best checkpoints per
replicate. Analytical action and I/O metrics are audit-only and were not used for
checkpoint selection.

## H0 Encoder Decision

The implemented H0 encoder is the standard minimal choice for this screen: a
single affine map from the first controller-visible target-relative delayed
feedback vector to the GRU hidden state. The context vector is
`[target_x - delayed_x, target_y - delayed_y, -delayed_vx, -delayed_vy]`, with
shape `[4]`; the output width is exactly the GRU hidden size (`180`). Weights and
bias are initialized to exactly zero, so the no-H0 behavior is preserved at
initialization. There is no MLP, no separately tuned hidden width, no pre-roll,
and no delayed-reach task change.

Uncertainty: this is the lowest-risk parameterization, but it may be too weak to
solve recovery from perturbed initial states. A richer encoder or delayed-reach
conditioning remains a separate hypothesis.

## Training Rows

Both rows used:

- batch size `64`, five replicates, `12000` batches;
- full analytical `Q/R/Q_f` objective;
- target-relative multi-target static training with seen and held-out validation
  targets declared in the run spec;
- randomized perturbation training mixture from the previous multi-target rows;
- peak LR `1e-3` or `3e-3`, global-norm clip `5`, linear warmup from `0.1 * LR`
  over `500` batches, cosine decay to alpha `0.01`;
- local CPU execution, not Modal.

Training diagnostics were emitted for every step. The smoke test verified that
the H0 weights moved off zero after one batch. Full-run diagnostics include
gradient norm, clipping fraction, update/parameter ratio, LR, train loss, and
validation loss arrays. Clipping fraction was `1.0` throughout both full runs.

| run | duration | batches/sec | mean grad norm | mean update/param | final validation |
|---|---:|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 1830.5 s | 6.555 | 25418.7 | 0.0006308 | 4464.95 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 1828.5 s | 6.563 | 15162.8 | 0.001283 | 4350.26 |

## Validation Selection And Nominal Kinematics

The materializer used `validation_selected_per_replicate`. For `1e-3`, selected
durable checkpoints were around batches `5000-6000`; for `3e-3`, they were
around batches `2000-4500`. Final checkpoints were worse than selected
checkpoints by roughly `175-410` validation units for `1e-3` and `262-634` for
`3e-3`.

The selected H0 rows preserved the extLQG-like nominal velocity timing:

| run | mean peak forward velocity | peak time | endpoint error |
|---|---:|---:|---:|
| H0 `1e-3` | 0.73096 m/s | 0.16 s | 0.00383 m |
| H0 `3e-3` | 0.72391 m/s | 0.16 s | 0.00333 m |
| extLQG 4D reference | 0.73274 m/s | 0.16 s | 0.00336 m |

Figures were written under
`_artifacts/643f101/figures/gru_postrun_target_relative_multitarget_h0_validation_selected/`.

## Feedback-Control Quality Lens Bundle

The table compares the new H0 rows with the previous non-H0 multi-target rows
(`ba82f3d`) and the previous good fixed-target perturbation rows (`aacb9ed`).
Lower mismatch/ratio values are better when the denominator is extLQG.

| row | selected validation | selected/extLQG deterministic | action mismatch | obs-action | covariance-weighted obs-action | peak velocity | endpoint | split-bank x0+epsilon ratio | perturb-bank init-pos ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed non-multi `1e-3` | 4366 | 0.9995 | 0.0021 | 0.8887 | 0.8908 | 0.7323 | 0.00341 | 15.95 | 14.56 |
| fixed non-multi `3e-3` | 4365 | 0.9993 | 0.0002 | 0.9306 | 0.9453 | 0.7309 | 0.00319 | 15.74 | 12.70 |
| multi-target no H0 `1e-3` | 4052 | 0.9276 | 1.829 | 1.876 | 1.848 | 0.7390 | 0.00355 | 11.68 | 3.861 |
| multi-target no H0 `3e-3` | 3891 | 0.8906 | 1.802 | 2.346 | 2.282 | 0.7317 | 0.00338 | 39.45 | 3.165 |
| multi-target H0 `1e-3` | 4159 | 0.9521 | 1.775 | 1.299 | 1.466 | 0.7310 | 0.00383 | 7.615 | 8.047 |
| multi-target H0 `3e-3` | 3974 | 0.9098 | 1.792 | 1.828 | 1.826 | 0.7239 | 0.00333 | 6.859 | 6.265 |

Interpretation: H0 improves the nominal observation-action map mismatch relative
to the previous multi-target rows, especially at `1e-3`, and the split-bank
initial-state/process-epsilon stress ratio improves. It does not improve the
perturbation-response bank's initial-position delta-cost ratio relative to the
non-H0 multi-target rows. The result therefore does not support "minimal H0
alone fixes feedback recovery".

## Recurrent And Action Diagnostics

| row | command norm | first-5 command norm | command jerk | hidden norm | local spectral radius | update gate high/low saturation | terminal speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed non-multi `1e-3` | 3.193 | 14.15 | 0.2589 | 5.236 | 1.211 | 0.0237 / 0.312 | 0.00520 |
| fixed non-multi `3e-3` | 3.207 | 14.21 | 0.3459 | 5.678 | 1.456 | 0.0394 / 0.521 | 0.00504 |
| multi-target no H0 `1e-3` | 3.327 | 13.28 | 0.7295 | 3.067 | 1.184 | 0.0081 / 0.156 | 0.00594 |
| multi-target no H0 `3e-3` | 3.398 | 14.06 | 1.141 | 1.900 | 1.237 | 0.0140 / 0.0745 | 0.00561 |
| multi-target H0 `1e-3` | 3.210 | 14.04 | 0.1340 | 2.204 | 1.058 | 0.0050 / 0.0089 | 0.00605 |
| multi-target H0 `3e-3` | 3.206 | 13.93 | 0.1817 | 1.446 | 0.979 | 0.0076 / 0.0108 | 0.00632 |

The H0 rows have lower hidden-state norms, lower gate saturation, and lower
local recurrent spectral radius than the prior rows. Those are cleaner recurrent
dynamics, but they did not translate into better perturbation-response recovery.

## Perturbation-Response Class Ratios

These are GRU/extLQG delta-cost ratios by perturbation class where an analytical
denominator is meaningful.

| row | initial position | initial velocity | process position | process velocity | force/filter | integrator |
|---|---:|---:|---:|---:|---:|---:|
| fixed non-multi `1e-3` | 14.56 | 4.654 | 0.991 | 0.612 | 2.900 | 10.86 |
| fixed non-multi `3e-3` | 12.70 | 4.013 | 0.975 | 0.564 | 2.552 | 9.255 |
| multi-target no H0 `1e-3` | 3.861 | 3.172 | 0.992 | 0.603 | 2.901 | 12.67 |
| multi-target no H0 `3e-3` | 3.165 | 2.770 | 0.955 | 0.538 | 2.512 | 10.48 |
| multi-target H0 `1e-3` | 8.047 | 4.643 | 1.045 | 0.671 | 3.204 | 12.56 |
| multi-target H0 `3e-3` | 6.265 | 3.754 | 1.010 | 0.605 | 2.868 | 11.37 |

The strongest remaining weakness is recovery from initial-position offsets. H0
did not beat the earlier non-H0 multi-target rows on this perturbation-response
metric.

## Materialized Artifacts

- Standard certificate:
  `results/643f101/notes/gru_standard_certificates_target_relative_multitarget_h0_validation_selected.md`
- Objective comparator:
  `results/643f101/notes/objective_comparator_target_relative_multitarget_h0_validation_selected.md`
- Evaluation diagnostics:
  `results/643f101/notes/gru_evaluation_diagnostics_target_relative_multitarget_h0_validation_selected.json`
- Map-error decomposition:
  `results/643f101/notes/gru_map_error_decomposition_target_relative_multitarget_h0_validation_selected.md`
- Perturbation response:
  `results/643f101/notes/gru_perturbation_response_target_relative_multitarget_h0_validation_selected.md`
- H-infinity phenotype sidecar:
  `results/643f101/notes/hinf_phenotype_target_relative_multitarget_h0_validation_selected.md`
- Post-run manifest:
  `results/643f101/notes/gru_postrun_materialization_target_relative_multitarget_h0_validation_selected.json`

Residual: the run specs declare seen and held-out target sets, but this post-run
materializer does not yet emit a separate per-target-bin H0 table. The values
above are pooled validation-selected diagnostics.
