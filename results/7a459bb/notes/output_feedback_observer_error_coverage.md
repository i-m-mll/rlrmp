# Observer-Error Coverage for Output-Feedback Rollout Recovery

Issue: `3becdec`. Parent: `7a459bb`. Umbrella: `43e8728`.

This materialization tests observer-error coverage as the remaining small
coverage-style diagnostic before moving away from coverage/noise changes. The
task and cost are unchanged; coverage only changes the training distribution.
All rows use `strong_optimizer_whitened` from scratch.

Method: Observer-error coverage uses leading singular disturbance directions of the analytical LQR disturbance-to-observer-error map. Trajectory rows train on signed full-trial disturbance samples; state rows train on the time-indexed (x, xhat) states induced by those samples.

Runtime: `213.18` seconds.

Artifacts:

- Manifest: `results/7a459bb/notes/output_feedback_observer_error_coverage_manifest.json`
- Arrays: `_artifacts/7a459bb/output_feedback_observer_error_coverage/output_feedback_observer_error_coverage.npz`

## Observer-Error Grid

| objective | modes | scale | iters | objective ratio | gain rel err | exact L2 ratio | lambda/gamma^2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trajectory | 1 | 0.3 | 2000 | 1.30188 | 0.99092 | 1.87392 | 4.31658 |
| trajectory | 1 | 1 | 2000 | 0.866875 | 0.98714 | 12.8083 | 40.666 |
| state | 1 | 0.3 | 2000 | 1.00921 | 0.991487 | 1.14629 | 2.14408 |
| state | 1 | 1 | 2000 | 1.00181 | 0.993387 | 1.18294 | 2.13009 |

## Comparison

| source | objective | modes | scale | objective ratio | gain rel err | exact L2 ratio | lambda/gamma^2 |
|---|---|---:|---:|---:|---:|---:|---:|
| no coverage | none | n/a | n/a | 1.01317 | 0.979472 | 1.15587 | 2.09679 |
| eigenspectrum best exact-L2 | trajectory | 4 | 0.3 | 1.24307 | 0.989895 | 1.26408 | 2.29358 |
| observer-error best exact-L2 | trajectory | 1 | 0.3 | 1.30188 | 0.99092 | 1.87392 | 4.31658 |
| eigenspectrum best exact-L2 | state | 1 | 0.3 | 1.00811 | 0.990895 | 1.101 | 2.13193 |
| observer-error best exact-L2 | state | 1 | 0.3 | 1.00921 | 0.991487 | 1.14629 | 2.14408 |

## Standard Certificate Coverage

All `8` observer-error evaluation rows have
`full_standard_certificate` status, covering nominal-clean and Riccati-epsilon
evaluation lenses for each trained controller.

## Verdict

Best observer-error gain error is `0.98714` (strong_optimizer_whitened_observer_error_trajectory_m1_s1__scratch).
Best observer-error exact-L2 ratio is `1.14629` (strong_optimizer_whitened_observer_error_state_m1_s0.3__scratch).
Observer-error coverage does not rescue the free time-varying output-feedback rollout bridge in this small grid.
