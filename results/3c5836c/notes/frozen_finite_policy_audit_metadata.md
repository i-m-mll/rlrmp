# Frozen finite-policy audit metadata

This is a local, saved-artifact audit for ae9f30f. It confirms optimizer provenance and selected PGD scalar behavior, and it records why exact live-graph frozen replay is not yet available from the saved cache.

## Confirmed facts

- The ae9f30f finite linear_no_bias rows used the broad-epsilon PGD lane, not the disabled policy_adversary_training Adam metadata.
- linear_no_bias_b1p4 has checkpoint context at 500 and 1000 batches but no completed training summary or training_diagnostics.npz in the local artifact cache.
- The direct_epsilon_b1p05 direct-epsilon control retained nonzero selected epsilon energy and positive selected objective gain at the final logged batch.
- The direct_epsilon_b1p4 direct-epsilon control retained nonzero selected epsilon energy and positive selected objective gain at the final logged batch.
- The completed linear_no_bias_b1p05 row was not zero at every sampled early checkpoint: selected energy was nonzero at 500 batches, but the saved samples show zero selected energy by 3000 batches and at the final batch.
- The completed linear_no_bias_b1p05 diagnostics selected zero epsilon at the final logged batch: selected gain over zero and selected epsilon energy are both zero in the saved scalar diagnostics.
- The linear_no_bias_b1p05 final PGD endpoint was not necessarily zero; the saved diagnostics show it was rejected in favor of the zero selected candidate under the active lambda/objective.

## Optimizer provenance

| Row | Status | Mechanism | Active lane | Method | Steps | Step frac | Lambda | Policy Adam active? |
|---|---|---|---|---|---:|---:|---:|---|
| direct_epsilon_b1p05 | completed_with_training_diagnostics | direct_epsilon | broad_epsilon_pgd_training.inner_maximizer | projected_gradient_ascent | 10 | 0.25 | 2.546e+08 | no |
| direct_epsilon_b1p4 | completed_with_training_diagnostics | direct_epsilon | broad_epsilon_pgd_training.inner_maximizer | projected_gradient_ascent | 10 | 0.25 | 4.526e+08 | no |
| linear_no_bias_b1p05 | completed_with_training_diagnostics | linear_no_bias | broad_epsilon_pgd_training.inner_maximizer | projected_gradient_ascent | 10 | 0.25 | 2.546e+08 | no |
| linear_no_bias_b1p4 | stopped_context_only | linear_no_bias | broad_epsilon_pgd_training.inner_maximizer | projected_gradient_ascent | 10 | 0.25 | 4.526e+08 | no |

## Checkpoint and diagnostic coverage

| Row | Checkpoints | Latest | Training summary | Diagnostics NPZ | Sentinels |
|---|---:|---|---|---|---|
| direct_epsilon_b1p05 | 24 | checkpoint_0012000 | yes | yes | direct_epsilon_b1p05.done, direct_epsilon_b1p05.pid, direct_epsilon_b1p05.started |
| direct_epsilon_b1p4 | 24 | checkpoint_0012000 | yes | yes | direct_epsilon_b1p4.done, direct_epsilon_b1p4.pid, direct_epsilon_b1p4.started |
| linear_no_bias_b1p05 | 24 | checkpoint_0012000 | yes | yes | linear_no_bias_b1p05.done, linear_no_bias_b1p05.pid, linear_no_bias_b1p05.started |
| linear_no_bias_b1p4 | 2 | checkpoint_0001000 | no | no | linear_no_bias_b1p4.failed, linear_no_bias_b1p4.pid, linear_no_bias_b1p4.started, linear_no_bias_b1p4.stopped |

## Selected PGD scalar diagnostics

| Row | Completed batches | Selected gain mean | Selected energy mean | Radius ratio mean | Cap boundary mean | Final endpoint gap mean |
|---|---:|---:|---:|---:|---:|---:|
| direct_epsilon_b1p05 | 500 | 1507.64 | 1.519e-06 | 1 | 1 | 1.25532 |
| direct_epsilon_b1p05 | 1000 | 1001.59 | 1.519e-06 | 1 | 1 | 0.518353 |
| direct_epsilon_b1p05 | 3000 | 379.875 | 1.513e-06 | 0.997522 | 0.990625 | 0.611984 |
| direct_epsilon_b1p05 | 6000 | 197.515 | 1.332e-06 | 0.922803 | 0.7375 | 0.187175 |
| direct_epsilon_b1p05 | 12000 | 153.455 | 1.181e-06 | 0.863587 | 0.496875 | 1.39488 |
| direct_epsilon_b1p4 | 500 | 1313.83 | 1.490e-06 | 0.986973 | 0.959375 | 0.727874 |
| direct_epsilon_b1p4 | 1000 | 644.756 | 1.385e-06 | 0.943448 | 0.675 | 1.08623 |
| direct_epsilon_b1p4 | 3000 | 160.838 | 6.987e-07 | 0.624799 | 0.040625 | 2.31884 |
| direct_epsilon_b1p4 | 6000 | 103.754 | 4.345e-07 | 0.474797 | 0.01875 | 1.43864 |
| direct_epsilon_b1p4 | 12000 | 87.3243 | 3.379e-07 | 0.438451 | 0.021875 | 0.9328 |
| linear_no_bias_b1p05 | 500 | 2360.64 | 2.000e-05 | 3.45894 | 1 | 0 |
| linear_no_bias_b1p05 | 1000 | 69.5277 | 3.578e-06 | 0.661747 | 0.2 | 560.5 |
| linear_no_bias_b1p05 | 3000 | 0 | 0 | 0 | 0 | 448.32 |
| linear_no_bias_b1p05 | 6000 | 0 | 0 | 0 | 0 | 505.861 |
| linear_no_bias_b1p05 | 12000 | 0 | 0 | 0 | 0 | 511.297 |
| linear_no_bias_b1p4 | n/a | n/a | n/a | n/a | n/a | n/a |

## Unsupported or blocked in this cache

- Exact selected direct-epsilon tensors were not persisted, so exact same-batch direct-to-linear projection cannot be reconstructed from this cache.
- The saved run cache lacks a compact replay descriptor for each selected batch: raw trial batch, per-update PRNG subkey, pre-update checkpoint/model id, and trial target metadata are not all present together.
- The issue-linked branch used for this audit has reusable finite-policy primitives, but exact ae9f30f live finite-graph replay should wait until the run-producing finite graph integration is committed or otherwise made available.
- No ae9f30f affine finite-policy row artifacts were present locally, so affine lambda/gain-bias conclusions are not inferred from these rows.

## Stabilization-table caveats

- Small stabilization-table values are not by themselves evidence of a unit bug: endpoint/reach is dimensionless endpoint delta divided by 0.15 m, and AUC dx is raw m*s even though older notes often displayed mm*s.
- The current ae9f30f stabilization note labels the baseline as no_pgd_h0_6d_const_band16, but the table source was the 020a65b calibrated H0/no-PGD artifact. The explicit prior const-band16 artifact appears to be 3244f1a/33b0dcb and is same-family but not numerically identical.
- Future stabilization tables should show mm and mm*s companion columns and blocked-row counts beside evaluated-row counts. The linear_no_bias_b1p05 plant/command comparison is fragile because only 54 rows were evaluated while 144 process-epsilon rows were blocked.

## Next replay instrumentation

- selected epsilon arrays or compact replay representation
- checkpoint/pre-update model id for the frozen objective
- deterministic batch descriptor and per-update PRNG key/subkey
- trial targets and active optimizer config
- finite-policy parameters at selected, final-endpoint, and zero proposals
