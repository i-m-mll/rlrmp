# Output-Feedback Sweep Standard Certificates

Issue: `7a459bb`. Standard certificate cross-reference: `d01c35a`.

This materialization applies the bridge standard-certificate row contract to the
recent output-feedback coverage/noise sweep outputs. It is intentionally a
partial certificate report: the saved no-coverage/reference artifact contains
gains plus nominal-clean/Riccati-epsilon rollout arrays, while the newer sweep
manifests mostly preserve small scalar summaries.

Result: Full standard-certificate components are available only for the saved no-coverage/reference rows backed by the rollout-recovery NPZ; those rows are now evaluated on both nominal-clean and Riccati-epsilon state lenses. The recent compact sweep manifests provide deterministic behavioral, optimizer, gain-diagnostic, exact-L2/gamma, and coverage-rank sidecars where those summaries were saved, but they do not include enough raw gains, rollout state/action arrays, covariances, or value matrices to recompute every standard component.

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
| eigenspectrum state coverage | 6 | missing | missing/missing/missing | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| eigenspectrum trajectory coverage | 6 | missing | missing/missing/missing | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| initial-state coverage | 4 | missing | missing/missing/missing | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| no-coverage/reference | 6 | available | available/available/available | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |
| process-noise stochastic | 20 | missing | missing/missing/missing | behavioral_action_sidecar, deterministic_exact_l2_and_gamma_sidecar, gain_diagnostic_sidecar, rollout_behavior_sidecar |

## Key Rows

| run | status | distribution | objective ratio | gain sidecar | action mismatch | exact L2 sidecar | lambda/gamma^2 |
|---|---|---|---:|---:|---:|---:|---:|
| no_coverage__analytical_lqr_reference__nominal_clean | full_standard_certificate | no-coverage/reference | n/a | 0 | 0 | 1 | 1.55512 |
| no_coverage__analytical_lqr_reference__riccati_epsilon_response | full_standard_certificate | no-coverage/reference | n/a | 0 | 0 | 1 | 1.55512 |
| no_coverage__strong_optimizer_whitened__scratch__nominal_clean | full_standard_certificate | no-coverage/reference | 1.01317 | 0.979472 | 0.481571 | 1.15587 | 2.09679 |
| no_coverage__strong_optimizer_whitened__scratch__riccati_epsilon_response | full_standard_certificate | no-coverage/reference | 1.01317 | 0.979472 | 1.21589 | 1.15587 | 2.09679 |
| no_coverage__strong_optimizer_whitened__bellman_init__nominal_clean | full_standard_certificate | no-coverage/reference | 1 | 0.000130853 | 1.42474e-05 | 0.999992 | 1.55511 |
| no_coverage__strong_optimizer_whitened__bellman_init__riccati_epsilon_response | full_standard_certificate | no-coverage/reference | 1 | 0.000130853 | 0.000631263 | 0.999992 | 1.55511 |
| initial_state_coverage__initial_state_scale_0x__strong_optimizer_whitened__scratch | partial_summary_certificate | initial-state coverage | 1.34926 | 0.995752 | 0.833172 | 725.745 | 2396.6 |
| initial_state_coverage__initial_state_scale_0.3x__strong_optimizer_whitened__scratch | partial_summary_certificate | initial-state coverage | 1.02388 | 0.984132 | 0.209065 | 1.17134 | 2.07104 |
| initial_state_coverage__initial_state_scale_1x__strong_optimizer_whitened__scratch | partial_summary_certificate | initial-state coverage | 1.01317 | 0.979472 | 0.0119778 | 1.15587 | 2.09679 |
| initial_state_coverage__initial_state_scale_3x__strong_optimizer_whitened__scratch | partial_summary_certificate | initial-state coverage | 1.00609 | 0.989249 | 0.0176281 | 1.16597 | 2.11421 |
| process_noise_stochastic__0.0__analytical_lqr_reference | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__0.0__strong_optimizer_whitened__scratch | sidecar_only_missing_inputs | process-noise stochastic | 1.00049 | 0.979472 | 0.0473433 | 1.15587 | 2.09679 |
| process_noise_stochastic__0.0__strong_optimizer_whitened__bellman_init | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0.000130853 | 8.09201e-05 | 0.999992 | 1.55511 |
| process_noise_stochastic__0.3__analytical_lqr_reference | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__0.3__strong_optimizer_whitened__scratch | sidecar_only_missing_inputs | process-noise stochastic | 1.00107 | 0.979472 | 0.04822 | 1.15587 | 2.09679 |
| process_noise_stochastic__0.3__strong_optimizer_whitened__bellman_init | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0.000130853 | 8.21771e-05 | 0.999992 | 1.55511 |
| process_noise_stochastic__1.0__analytical_lqr_reference | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__1.0__strong_optimizer_whitened__scratch | sidecar_only_missing_inputs | process-noise stochastic | 1.00288 | 0.979472 | 0.0499561 | 1.15587 | 2.09679 |
| process_noise_stochastic__1.0__strong_optimizer_whitened__bellman_init | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0.000130853 | 8.48496e-05 | 0.999992 | 1.55511 |
| process_noise_stochastic__3.0__analytical_lqr_reference | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0 | 0 | 1 | 1.55512 |
| process_noise_stochastic__3.0__strong_optimizer_whitened__scratch | sidecar_only_missing_inputs | process-noise stochastic | 1.00836 | 0.979472 | 0.0544303 | 1.15587 | 2.09679 |
| process_noise_stochastic__3.0__strong_optimizer_whitened__bellman_init | sidecar_only_missing_inputs | process-noise stochastic | 1 | 0.000130853 | 9.18628e-05 | 0.999992 | 1.55511 |
| eigenspectrum__trajectory__strong_optimizer_whitened_eigen_trajectory_m1_s0.3__scratch | partial_summary_certificate | eigenspectrum trajectory coverage | 1.21274 | 0.98896 | 0.689252 | 8.5038 | 26.3121 |
| eigenspectrum__state__strong_optimizer_whitened_eigen_state_m4_s3__scratch | partial_summary_certificate | eigenspectrum state coverage | 0.992097 | 0.99309 | 0.256241 | 1.12496 | 1.93248 |

## Missing Components

- Initial-state coverage rows lack saved fitted gains and rollout state/action
  arrays for each scale, so formal state-weighted action mismatch, closed-loop
  transition mismatch, value-policy gap, and Bellman-Hessian residual are
  marked `missing`. The tracked manifest still records the saved training
  ensemble effective-rank diagnostics and deterministic behavioral/exact-audit
  sidecars.
- Process-noise stochastic rows are evaluation summaries. They expose stochastic
  cost/action sidecars and deterministic exact-L2/gamma sidecars, but not the
  sampled state/action trajectories or covariances required for the formal
  standard components.
- Eigenspectrum trajectory/state coverage rows expose coverage-induced xhat
  rank diagnostics by objective/mode/scale, plus deterministic behavioral and
  exact-audit sidecars. They do not include per-row fitted gains or rollout
  arrays, so the formal linear components are marked `missing`.

## Verdict

The standard certificate application does not rescue the bridge. The only rows
with full component availability are the saved no-coverage/reference rows on
nominal-clean and Riccati-epsilon evaluation lenses. The recent initial-state,
process-noise, and eigenspectrum rows remain partial/sidecar-only from current
tracked artifacts, and their saved sidecars continue to show behaviorally close
but certificate-poor from-scratch recovery.
