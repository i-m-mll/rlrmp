# Output-Feedback Failure Decomposition

Issue: `c45adde`. Source bridge issue: `7a459bb`. Standard certificate
cross-reference: `d01c35a`.

This materialization is the standard failure-decomposition companion to the
bridge standard certificate. It answers why a row failed after the certificate
has answered whether the learned controller is equivalent to the analytical
reference. It does not change the bridge gate.

Scope: Saved no-coverage output-feedback key rows. The materializer reuses stored gains and rollout arrays, evaluates the same clean training objective at learned/reference gains, and projects gain error through the standard-certificate evaluation state distributions.

## Source Inputs

- Rollout-recovery manifest: `results/7a459bb/notes/output_feedback_rollout_recovery_manifest.json`
- Standard-certificate manifest: `results/7a459bb/notes/output_feedback_sweep_standard_certificates_manifest.json`
- Saved no-coverage arrays: `_artifacts/7a459bb/output_feedback_rollout_recovery/output_feedback_rollout_recovery.npz`

## Key Rows

| run | class | objective ratio | learned proj-grad<sup>1</sup> | reference proj-grad<sup>1</sup> | action mismatch | Bellman residual | strong visited error | weak/unvisited error<sup>2</sup> | best interp alpha | best interp objective ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_coverage__strong_optimizer_whitened__scratch__nominal_clean | optimizer_basin | 1.01317 | 160.953 | 6.71818e-12 | 0.481571 | 0.481571 | 2.78387e-05 | 0.999972 | 1 | 1 |
| no_coverage__strong_optimizer_whitened__scratch__riccati_epsilon_response | optimizer_basin | 1.01317 | 160.953 | 6.71818e-12 | 1.21589 | 1.21589 | 3.0602e-05 | 0.999969 | 1 | 1 |
| no_coverage__strong_optimizer_whitened__bellman_init__nominal_clean | not_failure | 1 | 0.00404095 | 6.71818e-12 | 1.42474e-05 | 1.42474e-05 | 4.72217e-07 | 1 | 1 | 1 |
| no_coverage__strong_optimizer_whitened__bellman_init__riccati_epsilon_response | not_failure | 1 | 0.00404095 | 6.71818e-12 | 0.000631263 | 0.000631263 | 2.97951e-05 | 0.99997 | 1 | 1 |

<sup>1</sup> Objective and projected-gradient diagnostics are evaluated in the
same whitened L-BFGS-B theta parameterization used by the saved
`strong_optimizer_whitened` no-coverage fits. There are no active box
constraints, so projected-gradient norm equals gradient norm for these rows.

<sup>2</sup> The visited/weakly visited decomposition projects gain error
through the evaluation-lens estimated-state covariance used by the standard
state-weighted action mismatch. Weak/unvisited gain error is explanatory only;
it is not the certificate gate.

## Definitions

- `under_identification`: the training objective is already near the reference,
  but certificate mismatch remains and gain error lies mostly in weakly visited
  or unvisited state directions.
- `optimizer_basin`: the learned point is still objectively worse than the
  reference and/or has a large projected gradient in the optimizer
  parameterization.
- `objective_mismatch`: the learned point is stationary under the training
  objective while the analytical reference is not.
- `mixed`: more than one of the preceding signals is active.

## Interpretation

The no-coverage from-scratch key row is an optimizer-basin failure under this
diagnostic: moving along the straight line toward the analytical reference
reduces the clean training objective, and the learned projected gradient remains
large. The Bellman-initialized key row is not a substantive failure: objective,
gradient, and certificate residuals are all near the reference.
