# Output-Feedback Sweep Standard Certificates

Issue: `7a459bb`. Standard certificate cross-reference: `d01c35a`.

This materialization applies the bridge standard-certificate row contract to the
recent output-feedback coverage/noise sweep outputs. It reruns deterministic
sweep cells and released-stochastic process-noise evaluations as needed so the
standard certificate is computed from fitted gains, trajectories, and
covariances rather than inferred from scalar summaries.

Result: Full standard-certificate components are now available for the no-coverage/reference rows and for rerun deterministic initial-state and eigenspectrum coverage rows on nominal-clean and Riccati-epsilon evaluation lenses. Released-stochastic process-noise rows are rerun with common random numbers so state/action, transition, value-gap, Bellman-Hessian, visited-subspace, behavioral, exact-L2/gamma, and gain-diagnostic fields are all explicit when defined.

Raw gain recovery is reported only in `gain_diagnostic_sidecar` rows and is not
used as the certificate gate.

## Source Inputs

- No-coverage/reference manifest: `results/7a459bb/notes/output_feedback_rollout_recovery_manifest.json`
- Saved no-coverage arrays: `_artifacts/7a459bb/output_feedback_rollout_recovery/output_feedback_rollout_recovery.npz`
- Initial-state coverage manifest: `results/7a459bb/notes/output_feedback_initial_state_variability_sweep_manifest.json`
- Process-noise stochastic manifest: `results/7a459bb/notes/output_feedback_process_noise_sweep_manifest.json`
- Eigenspectrum coverage manifest: `results/7a459bb/notes/output_feedback_eigenspectrum_coverage_sweep_manifest.json`

## Availability by Distribution

| distribution family | rows | standard state/action | transition/value/Bellman | available sidecars |
|---|---:|---|---|---|
| eigenspectrum state coverage | 12 | available | available/available/available | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| eigenspectrum trajectory coverage | 12 | available | available/available/available | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| initial-state coverage | 8 | available | available/available/available | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| no-coverage/reference | 6 | available | available/available/available | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| process-noise stochastic | 20 | available | available/available/available | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |

## Key Rows

| run | status | distribution | objective ratio | gain sidecar<sup>2</sup> | action mismatch<sup>1</sup> | transition mismatch | value gap | Bellman residual<sup>1</sup> | exact L2 sidecar | lambda/gamma^2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_coverage__analytical_lqr_reference__nominal_clean | full_standard_certificate | no-coverage/reference | n/a | 0 | 0 | 0 | 0 | 0 | 1 | 1.55512 |
| no_coverage__analytical_lqr_reference__riccati_epsilon_response | full_standard_certificate | no-coverage/reference | n/a | 0 | 0 | 0 | 0 | 0 | 1 | 1.55512 |
| no_coverage__strong_optimizer_whitened__scratch__nominal_clean | full_standard_certificate | no-coverage/reference | 1.01317 | 0.979472 | 0.481571 | 9.46452e-05 | 0.000111684 | 0.481571 | 1.15587 | 2.09679 |
| no_coverage__strong_optimizer_whitened__scratch__riccati_epsilon_response | full_standard_certificate | no-coverage/reference | 1.01317 | 0.979472 | 1.21589 | 0.0156233 | 0.0251373 | 1.21589 | 1.15587 | 2.09679 |
| no_coverage__strong_optimizer_whitened__bellman_init__nominal_clean | full_standard_certificate | no-coverage/reference | 1 | 0.000130853 | 1.42474e-05 | 1.98244e-11 | 7.48717e-12 | 1.42474e-05 | 0.999992 | 1.55511 |
| no_coverage__strong_optimizer_whitened__bellman_init__riccati_epsilon_response | full_standard_certificate | no-coverage/reference | 1 | 0.000130853 | 0.000631263 | 1.4222e-09 | -1.48666e-06 | 0.000631263 | 0.999992 | 1.55511 |
| initial_state_coverage__initial_state_scale_0x__strong_optimizer_whitened__scratch__nominal_clean | full_standard_certificate | initial-state coverage | 1.34926 | 0.995752 | 60.905 | 0.0201456 | 0.130902 | 60.905 | 725.745 | 2396.6 |
| initial_state_coverage__initial_state_scale_0x__strong_optimizer_whitened__scratch__riccati_epsilon_response | full_standard_certificate | initial-state coverage | 1.34926 | 0.995752 | 46.1896 | 0.154119 | 15.7088 | 46.1896 | 725.745 | 2396.6 |
| initial_state_coverage__initial_state_scale_0.3x__strong_optimizer_whitened__scratch__nominal_clean | full_standard_certificate | initial-state coverage | 1.02388 | 0.984132 | 0.599149 | 0.00394317 | 0.00146184 | 0.599149 | 1.17134 | 2.07104 |
| initial_state_coverage__initial_state_scale_0.3x__strong_optimizer_whitened__scratch__riccati_epsilon_response | full_standard_certificate | initial-state coverage | 1.02388 | 0.984132 | 3.71524 | 0.0176558 | 0.0256383 | 3.71524 | 1.17134 | 2.07104 |
| initial_state_coverage__initial_state_scale_1x__strong_optimizer_whitened__scratch__nominal_clean | full_standard_certificate | initial-state coverage | 1.01317 | 0.979472 | 0.481571 | 9.46452e-05 | 0.000111684 | 0.481571 | 1.15587 | 2.09679 |
| initial_state_coverage__initial_state_scale_1x__strong_optimizer_whitened__scratch__riccati_epsilon_response | full_standard_certificate | initial-state coverage | 1.01317 | 0.979472 | 1.21589 | 0.0156233 | 0.0251373 | 1.21589 | 1.15587 | 2.09679 |
| initial_state_coverage__initial_state_scale_3x__strong_optimizer_whitened__scratch__nominal_clean | full_standard_certificate | initial-state coverage | 1.00609 | 0.989249 | 4.11683 | 6.43176e-05 | 5.72413e-05 | 4.11683 | 1.16597 | 2.11421 |
| initial_state_coverage__initial_state_scale_3x__strong_optimizer_whitened__scratch__riccati_epsilon_response | full_standard_certificate | initial-state coverage | 1.00609 | 0.989249 | 0.926309 | 0.0155555 | 0.0266966 | 0.926309 | 1.16597 | 2.11421 |
| process_noise_stochastic__0.0__analytical_lqr_reference | full_standard_certificate | process-noise stochastic | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__0.0__strong_optimizer_whitened__scratch | full_standard_certificate | process-noise stochastic | 1.00049 | 0.979472 | 0.483841 | 0.0157251 | 0.00961143 | 0.483841 | 1.15587 | 2.09679 |
| process_noise_stochastic__0.0__strong_optimizer_whitened__bellman_init | full_standard_certificate | process-noise stochastic | 1 | 0.000130853 | 1.36949e-05 | 1.82408e-08 | 1.3272e-08 | 1.36949e-05 | 0.999992 | 1.55511 |
| process_noise_stochastic__0.3__analytical_lqr_reference | full_standard_certificate | process-noise stochastic | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__0.3__strong_optimizer_whitened__scratch | full_standard_certificate | process-noise stochastic | 1.00107 | 0.979472 | 0.461485 | 0.017948 | 0.0157194 | 0.461485 | 1.15587 | 2.09679 |
| process_noise_stochastic__0.3__strong_optimizer_whitened__bellman_init | full_standard_certificate | process-noise stochastic | 1 | 0.000130853 | 1.80998e-05 | 1.45057e-08 | 1.8104e-08 | 1.80998e-05 | 0.999992 | 1.55511 |
| process_noise_stochastic__1.0__analytical_lqr_reference | full_standard_certificate | process-noise stochastic | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__1.0__strong_optimizer_whitened__scratch | full_standard_certificate | process-noise stochastic | 1.00288 | 0.979472 | 0.448525 | 0.0237821 | 0.0311977 | 0.448525 | 1.15587 | 2.09679 |
| process_noise_stochastic__1.0__strong_optimizer_whitened__bellman_init | full_standard_certificate | process-noise stochastic | 1 | 0.000130853 | 1.99352e-05 | 1.20486e-08 | 1.30414e-07 | 1.99352e-05 | 0.999992 | 1.55511 |
| process_noise_stochastic__3.0__analytical_lqr_reference | full_standard_certificate | process-noise stochastic | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__3.0__strong_optimizer_whitened__scratch | full_standard_certificate | process-noise stochastic | 1.00836 | 0.979472 | 0.434274 | 0.030923 | 0.071191 | 0.434274 | 1.15587 | 2.09679 |
| process_noise_stochastic__3.0__strong_optimizer_whitened__bellman_init | full_standard_certificate | process-noise stochastic | 1 | 0.000130853 | 2.04779e-05 | 1.08058e-08 | 5.06952e-07 | 2.04779e-05 | 0.999992 | 1.55511 |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s0.3__scratch__nominal_clean | full_standard_certificate | eigenspectrum trajectory coverage | 1.21274 | 0.98896 | 2.7781 | 0.0221631 | 0.124301 | 2.7781 | 8.5038 | 26.3121 |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s0.3__scratch__riccati_epsilon_response | full_standard_certificate | eigenspectrum trajectory coverage | 1.21274 | 0.98896 | 1.73799 | 0.0614736 | 0.315152 | 1.73799 | 8.5038 | 26.3121 |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s3__scratch__nominal_clean | full_standard_certificate | eigenspectrum state coverage | 0.992097 | 0.99309 | 0.317931 | 0.00418572 | 0.0024483 | 0.317931 | 1.12496 | 1.93248 |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s3__scratch__riccati_epsilon_response | full_standard_certificate | eigenspectrum state coverage | 0.992097 | 0.99309 | 1.04034 | 0.00640358 | 0.0150816 | 1.04034 | 1.12496 | 1.93248 |

<sup>1</sup> State-weighted action mismatch and Bellman-Hessian residual can
match exactly when the Bellman action Hessian is a scalar multiple of the action
cost geometry on that row. In that case they are the same evidence expressed
through two certificate views; they diverge when downstream value geometry
weights action directions differently.

<sup>2</sup> Gain mismatch is a diagnostic sidecar, not the bridge gate. The
gate is disturbance-relevant same-game behavior under the standard certificate
components.

## Computation Notes

- The initial-state and eigenspectrum rows are not inferred from the compact
  tracked manifests. They are rerun from the tracked sweep specifications so
  fitted gains, nominal-clean rollouts, and Riccati-epsilon rollouts are
  available for the full certificate.
- Process-noise stochastic rows are rerun with the tracked common-random-number
  settings so sampled state/estimate/action trajectories are available for the
  same component bundle.
- Evaluation lenses are not training axes. Nominal-clean, Riccati-epsilon, and
  process-noise stochastic rows describe where the finished controller is
  evaluated; they do not by themselves mean the controller was trained with a
  robust objective or coverage distribution.

## Verdict

The standard certificate application does not rescue the bridge. All rows in
this materialization now have full component availability where the linear
output-feedback quantities are defined. The rerun initial-state, process-noise,
and eigenspectrum rows continue to show behaviorally close but certificate-poor
from-scratch recovery.
