# Phase-Modulated Linear Recurrent Output-Feedback Bridge

Issue: `d6d25d6`. Umbrella: `1fabee8`.

Scope: Oracle Kalman recurrent reference plus clamped-spline phase-modulated linear recurrence. The spline basis modulates A/B/C/D matrices over tau=t/(T-1), not additive phase offsets.

Non-goals: No GRU training, no supervised imitation optimization rows, no broad robust-epsilon arm, and no claim that projected-oracle diagnostic rows are bridge passes.

Runtime: `89.38` seconds.

Verdict: The exact oracle and clamped-spline projected-oracle rows were materialized. Exact-oracle sanity rows have max aggregate response-map mismatch 0. The r=12 projected-oracle nominal replay row has action mismatch 24.54 and combined matrix residual 0.01274. 8 r=12 reward rows were optimized with bounded Adam over phase-modulated recurrent coefficients.

Audit note: The prior exact_process_eigen rows were state-trajectory covariance coverage directions, not process-eigen disturbance sequences. They are retained under state_coverage_eigen labels.

## Rows

| row | family | lens | verdict | objective ratio | action mismatch | matrix residual | obs I/O | proc I/O | io-map |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| phase_modulated_recurrent__pm_linrec_exact_oracle_nominal_replay | exact_oracle_sanity | nominal_clean | exact_oracle_sanity_pass | 1 | 0 | 0 | 0 | 0 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_exact_oracle_process_io | exact_oracle_sanity | process_io | exact_oracle_sanity_pass | 656.22396 | 0 | 0 | 0 | 0 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_exact_oracle_process_measurement_io | exact_oracle_sanity | process_measurement_io | exact_oracle_sanity_pass | 664.39558 | 0 | 0 | 0 | 0 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.1648125 | 12406.437 | 0.15105496 | 0.50047359 | 0.1038565 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.1346193 | 191.5243 | 0.055506791 | 0.43012287 | 0.58184335 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.0008825 | 64.732114 | 0.013642344 | 0.20552348 | 0.029962722 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1.0000609 | 24.543846 | 0.012744718 | 0.25422973 | 0.10900394 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1.0000002 | 0.12881805 | 0.0048766187 | 0.16277743 | 0.015810533 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1 | 0.040147103 | 0.0023077801 | 0.11044878 | 0.0030258819 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1 | 1.6218418e-23 | 1.5500104e-15 | 1.5641413e-26 | 1.2974114e-26 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.63447 | 4.5471523 | 0.012744718 | 0.25422973 | 0.10900394 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.24546 | 3.2312072 | 0.0048766187 | 0.16277743 | 0.015810533 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.23243 | 0.085271906 | 0.0023077801 | 0.11044878 | 0.0030258819 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.22396 | 2.8123112e-24 | 1.5500104e-15 | 1.5641413e-26 | 1.2974114e-26 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 665.47391 | 2.2937547 | 0.012744718 | 0.25422973 | 0.10900394 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 665.09706 | 0.91649284 | 0.0048766187 | 0.16277743 | 0.015810533 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 664.94429 | 0.037814896 | 0.0023077801 | 0.11044878 | 0.0030258819 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 664.39558 | 7.7824656e-25 | 1.5500104e-15 | 1.5641413e-26 | 1.2974114e-26 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.5383958 | 12432.099 | 0.15105496 | 0.50047359 | 0.1038565 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.4958733 | 53.167955 | 0.055506791 | 0.43012287 | 0.58184335 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.3202128 | 64.158735 | 0.013642344 | 0.20552348 | 0.029962722 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.3191389 | 24.378338 | 0.012744718 | 0.25422973 | 0.10900394 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_clean_scratch_reward | reward_lens | nominal_clean | reward_trained_non_equivalent | 3.7286225 | 687138.46 | 0.012744718 | 1.0000407 | 1.0024616 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_state_coverage_eigen_m1_s0p3_reward | reward_lens | state_coverage_eigen_m1_s0.3 | reward_trained_non_equivalent | 6.2331343 | 49349533 | 0.012744718 | 1.0011303 | 1.0001422 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_state_coverage_eigen_m4_s0p3_reward | reward_lens | state_coverage_eigen_m4_s0.3 | reward_trained_non_equivalent | 8.5118189 | 282685.51 | 0.012744718 | 0.99866363 | 0.99396742 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_state_coverage_eigen_m4_s1_reward | reward_lens | state_coverage_eigen_m4_s1 | reward_trained_non_equivalent | 131.23026 | 109415.74 | 0.012744718 | 0.99967113 | 0.99639833 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_observer_error_svd_m1_s0p3_reward | reward_lens | observer_error_svd_m1_s0.3 | reward_trained_non_equivalent | 4.6618076 | 184379.64 | 0.012744718 | 1.0000277 | 1.0031633 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_mixed_process_observer_reward | reward_lens | mixed_process_observer | reward_trained_non_equivalent | 89.270334 | 4058.2856 | 0.012744718 | 1.0020515 | 1.0129543 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_nominal_then_reward | projection_warm_start_then_reward_lens | nominal_clean | reward_trained_non_equivalent | 1.0000609 | 24.543846 | 0.012744718 | 0.25422973 | 0.10900394 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_state_coverage_eigen_m4_then_reward | projection_warm_start_then_reward_lens | state_coverage_eigen_m4_s0.3 | reward_trained_non_equivalent | 1.3191389 | 24.378338 | 0.012744718 | 0.25422973 | 0.10900394 | standard_components_available |

## Certificate Boundary

These rows are diagnostics, not bridge passes. The recurrent controller is an
augmented-linear system over `z_t = [x_t; h_t]`, and action/visited-state
components are reported through the existing augmented recurrent adapter. The
formal I/O-map certificate is consumed through the standard certificate
component builder; projected-oracle rows remain diagnostic even when those
components are available. No supervised imitation rows are materialized here.

- `bellman_hessian_residual:missing`: 30
- `closed_loop_transition_mismatch:missing`: 30
- `disturbance_history_to_action_map_mismatch:available`: 30
- `disturbance_history_to_state_map_mismatch:available`: 30
- `observation_history_to_action_map_mismatch:available`: 30
- `optimizer_metadata:available`: 30
- `recurrence_gru_diagnostics:available`: 30
- `state_weighted_action_mismatch:available`: 30
- `value_policy_gap:missing`: 30
- `visited_subspace_diagnostics:available`: 30

## Interpretation

The exact-oracle rows replay the finite-horizon Kalman recurrent reference under
nominal, process I/O, and process+measurement I/O probes. Ranked rows then
project each time-varying recurrent/readout matrix onto a clamped B-spline
partition of unity. State-coverage eigen rows preserve the old coverage
semantics explicitly: they perturb initial state/estimator coverage directions
from a state-trajectory covariance, not process-eigen disturbances. Reward rows
optimize trainable phase-modulated recurrent coefficients against the true
quadratic rollout objective on their retained training distributions; projection
warm-start rows use the projected oracle controller before reward fine-tuning.
