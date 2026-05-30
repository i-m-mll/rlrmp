# Output-Feedback Failure Decomposition

Issue: `c45adde`. Source bridge issue: `7a459bb`. Standard certificate
cross-reference: `d01c35a`.

This materialization is the standard failure-decomposition companion to the
bridge standard certificate. It answers why a row failed after the certificate
has answered whether the learned controller is equivalent to the analytical
reference. It does not change the bridge gate.

Scope: Current output-feedback no-coverage, initial-state coverage, eigenspectrum coverage, and observer-error coverage rows. Saved arrays are reused where available; deterministic coverage reruns are cached under ignored artifacts so future applications do not lose the controller and rollout arrays required for the decomposition.

## Source Inputs

- Rollout-recovery manifest: `results/7a459bb/notes/output_feedback_rollout_recovery_manifest.json`
- Standard-certificate manifest: `results/7a459bb/notes/output_feedback_sweep_standard_certificates_manifest.json`
- Observer-error manifest: `results/7a459bb/notes/output_feedback_observer_error_coverage_manifest.json`
- Saved no-coverage arrays: `_artifacts/7a459bb/output_feedback_rollout_recovery/output_feedback_rollout_recovery.npz`
- Observer-error arrays: `_artifacts/7a459bb/output_feedback_observer_error_coverage/output_feedback_observer_error_coverage.npz`
- Deterministic coverage array cache: `_artifacts/7a459bb/output_feedback_failure_decomposition/deterministic_coverage_arrays.npz`

## Key Rows

| run | class | objective ratio | learned proj-grad<sup>1</sup> | reference proj-grad<sup>1</sup> | action mismatch | Bellman residual | strong visited error | weak/unvisited error<sup>2</sup> | best interp alpha | best interp objective ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_coverage__strong_optimizer_whitened__scratch__nominal_clean | optimizer_basin | 1.01317 | 160.953 | 6.71818e-12 | 0.481571 | 0.481571 | 2.78387e-05 | 0.999972 | 1 | 1 |
| no_coverage__strong_optimizer_whitened__scratch__riccati_epsilon_response | optimizer_basin | 1.01317 | 160.953 | 6.71818e-12 | 1.21589 | 1.21589 | 3.0602e-05 | 0.999969 | 1 | 1 |
| no_coverage__strong_optimizer_whitened__bellman_init__nominal_clean | not_failure | 1 | 0.00404095 | 6.71818e-12 | 1.42474e-05 | 1.42474e-05 | 4.72217e-07 | 1 | 1 | 1 |
| no_coverage__strong_optimizer_whitened__bellman_init__riccati_epsilon_response | not_failure | 1 | 0.00404095 | 6.71818e-12 | 0.000631263 | 0.000631263 | 2.97951e-05 | 0.99997 | 1 | 1 |
| initial_state_coverage__initial_state_scale_0x__strong_optimizer_whitened__scratch__nominal_clean | optimizer_basin | 1.34926 | 4.84392e+07 | n/a | 60.905 | 60.905 | 0.0009209 | 0.999079 | n/a | n/a |
| initial_state_coverage__initial_state_scale_0x__strong_optimizer_whitened__scratch__riccati_epsilon_response | optimizer_basin | 1.34926 | 4.84392e+07 | n/a | 46.1896 | 46.1896 | 0.000941322 | 0.999059 | n/a | n/a |
| initial_state_coverage__initial_state_scale_0.3x__strong_optimizer_whitened__scratch__nominal_clean | optimizer_basin | 1.02388 | 1012.15 | n/a | 0.599149 | 0.599149 | 0.000546599 | 0.999453 | n/a | n/a |
| initial_state_coverage__initial_state_scale_0.3x__strong_optimizer_whitened__scratch__riccati_epsilon_response | optimizer_basin | 1.02388 | 1012.15 | n/a | 3.71524 | 3.71524 | 0.000506845 | 0.999493 | n/a | n/a |
| initial_state_coverage__initial_state_scale_1x__strong_optimizer_whitened__scratch__nominal_clean | optimizer_basin | 1.01317 | 160.953 | n/a | 0.481571 | 0.481571 | 2.78387e-05 | 0.999972 | n/a | n/a |
| initial_state_coverage__initial_state_scale_1x__strong_optimizer_whitened__scratch__riccati_epsilon_response | optimizer_basin | 1.01317 | 160.953 | n/a | 1.21589 | 1.21589 | 3.0602e-05 | 0.999969 | n/a | n/a |
| initial_state_coverage__initial_state_scale_3x__strong_optimizer_whitened__scratch__nominal_clean | optimizer_basin | 1.00609 | 159.115 | n/a | 4.11683 | 4.11683 | 6.04022e-05 | 0.99994 | n/a | n/a |
| initial_state_coverage__initial_state_scale_3x__strong_optimizer_whitened__scratch__riccati_epsilon_response | optimizer_basin | 1.00609 | 159.115 | n/a | 0.926309 | 0.926309 | 5.51707e-05 | 0.999945 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s0.3__scratch__nominal_clean | optimizer_basin | 1.21274 | 96609.6 | n/a | 2.7781 | 2.7781 | 0.000727681 | 0.999272 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s0.3__scratch__riccati_epsilon_response | optimizer_basin | 1.21274 | 96609.6 | n/a | 1.73799 | 1.73799 | 0.000859418 | 0.999141 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s1__scratch__nominal_clean | mixed | 0.920518 | 236998 | n/a | 4.2595 | 4.2595 | 0.000183514 | 0.999816 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s1__scratch__riccati_epsilon_response | mixed | 0.920518 | 236998 | n/a | 1.31473 | 1.31473 | 0.000189372 | 0.999811 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s3__scratch__nominal_clean | mixed | 0.529351 | 1.76641e+06 | n/a | 1.11915 | 1.11915 | 0.000883035 | 0.999117 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s3__scratch__riccati_epsilon_response | mixed | 0.529351 | 1.76641e+06 | n/a | 0.979531 | 0.979531 | 0.000806791 | 0.999193 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m4_s0.3__scratch__nominal_clean | optimizer_basin | 1.24307 | 66773.3 | n/a | 5.61198 | 5.61198 | 0.00121338 | 0.998787 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m4_s0.3__scratch__riccati_epsilon_response | optimizer_basin | 1.24307 | 66773.3 | n/a | 1.89092 | 1.89092 | 0.00109895 | 0.998901 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m4_s1__scratch__nominal_clean | optimizer_basin | 1.26974 | 113249 | n/a | 3.10671 | 3.10671 | 0.000313136 | 0.999687 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m4_s1__scratch__riccati_epsilon_response | optimizer_basin | 1.26974 | 113249 | n/a | 3.03993 | 3.03993 | 0.000316653 | 0.999683 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m4_s3__scratch__nominal_clean | optimizer_basin | 1.08044 | 1.31498e+06 | n/a | 3.07819 | 3.07819 | 0.00092237 | 0.999078 | n/a | n/a |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m4_s3__scratch__riccati_epsilon_response | optimizer_basin | 1.08044 | 1.31498e+06 | n/a | 4.13128 | 4.13128 | 0.00116197 | 0.998838 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m1_s0.3__scratch__nominal_clean | optimizer_basin | 1.00811 | 853.223 | n/a | 0.404735 | 0.404735 | 0.000148965 | 0.999851 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m1_s0.3__scratch__riccati_epsilon_response | optimizer_basin | 1.00811 | 853.223 | n/a | 0.760023 | 0.760023 | 0.00014036 | 0.99986 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m1_s1__scratch__nominal_clean | mixed | 0.993385 | 732.264 | n/a | 0.601611 | 0.601611 | 5.78341e-05 | 0.999942 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m1_s1__scratch__riccati_epsilon_response | mixed | 0.993385 | 732.264 | n/a | 0.758951 | 0.758951 | 5.78082e-05 | 0.999942 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m1_s3__scratch__nominal_clean | mixed | 0.966389 | 6299.37 | n/a | 0.448204 | 0.448204 | 0.000113359 | 0.999887 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m1_s3__scratch__riccati_epsilon_response | mixed | 0.966389 | 6299.37 | n/a | 1.31062 | 1.31062 | 0.00012696 | 0.999873 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s0.3__scratch__nominal_clean | optimizer_basin | 1.01407 | 1432.34 | n/a | 1.18294 | 1.18294 | 0.000109171 | 0.999891 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s0.3__scratch__riccati_epsilon_response | optimizer_basin | 1.01407 | 1432.34 | n/a | 2.48574 | 2.48574 | 0.000108138 | 0.999892 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s1__scratch__nominal_clean | optimizer_basin | 1.01666 | 6677.97 | n/a | 0.651671 | 0.651671 | 0.000371052 | 0.999629 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s1__scratch__riccati_epsilon_response | optimizer_basin | 1.01666 | 6677.97 | n/a | 1.07052 | 1.07052 | 0.000364368 | 0.999636 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s3__scratch__nominal_clean | mixed | 0.992097 | 4992.5 | n/a | 0.317931 | 0.317931 | 0.000257057 | 0.999743 | n/a | n/a |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s3__scratch__riccati_epsilon_response | mixed | 0.992097 | 4992.5 | n/a | 1.04034 | 1.04034 | 0.000257922 | 0.999742 | n/a | n/a |
| observer_error__trajectory__strong_optimizer_whitened_observer_error_trajectory_m1_s0.3__scratch__nominal_clean | optimizer_basin | 1.30188 | 329198 | n/a | 2.60021 | 2.60021 | 0.000943269 | 0.999057 | n/a | n/a |
| observer_error__trajectory__strong_optimizer_whitened_observer_error_trajectory_m1_s0.3__scratch__riccati_epsilon_response | optimizer_basin | 1.30188 | 329198 | n/a | 3.91874 | 3.91874 | 0.00101921 | 0.998981 | n/a | n/a |
| observer_error__trajectory__strong_optimizer_whitened_observer_error_trajectory_m1_s1__scratch__nominal_clean | mixed | 0.866875 | 816568 | n/a | 2.1879 | 2.1879 | 2.0068e-05 | 0.99998 | n/a | n/a |
| observer_error__trajectory__strong_optimizer_whitened_observer_error_trajectory_m1_s1__scratch__riccati_epsilon_response | mixed | 0.866875 | 816568 | n/a | 1.30412 | 1.30412 | 4.45721e-05 | 0.999955 | n/a | n/a |
| observer_error__state__strong_optimizer_whitened_observer_error_state_m1_s0.3__scratch__nominal_clean | optimizer_basin | 1.00921 | 517.856 | n/a | 0.467273 | 0.467273 | 0.000152862 | 0.999847 | n/a | n/a |
| observer_error__state__strong_optimizer_whitened_observer_error_state_m1_s0.3__scratch__riccati_epsilon_response | optimizer_basin | 1.00921 | 517.856 | n/a | 0.629222 | 0.629222 | 0.000141209 | 0.999859 | n/a | n/a |
| observer_error__state__strong_optimizer_whitened_observer_error_state_m1_s1__scratch__nominal_clean | optimizer_basin | 1.00181 | 3889.78 | n/a | 0.584815 | 0.584815 | 0.00020857 | 0.999791 | n/a | n/a |
| observer_error__state__strong_optimizer_whitened_observer_error_state_m1_s1__scratch__riccati_epsilon_response | optimizer_basin | 1.00181 | 3889.78 | n/a | 0.819506 | 0.819506 | 0.000204117 | 0.999796 | n/a | n/a |

<sup>1</sup> Objective and projected-gradient diagnostics are evaluated in each
row's training objective and optimizer parameterization. For no-coverage rows,
the materializer recomputes learned/reference gradients directly from the
saved objective. For coverage rows, the final optimizer gradient is taken from
the saved or cached fit summary because the optimizer closure itself is not
stored in tracked results.

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

The from-scratch key rows remain optimizer-basin failures under this diagnostic:
their learned controllers are not stationary under the objective that trained
them, and the standard-certificate mismatches remain large. Coverage changes
the training distribution, but it does not rescue this free time-varying
architecture. The Bellman-initialized no-coverage key row remains the sanity
check: objective, gradient, and certificate residuals are all near the
reference.
