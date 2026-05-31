# Phase-Modulated Linear Recurrent Output-Feedback Bridge

Issue: `d6d25d6`. Umbrella: `1fabee8`.

Scope: Oracle Kalman recurrent reference plus clamped-spline phase-modulated linear recurrence. The spline basis modulates A/B/C/D matrices over tau=t/(T-1), not additive phase offsets.

Non-goals: No GRU training, no broad robust-epsilon arm, and no claim that projection/imitation diagnostic rows are bridge passes.

Runtime: `70.60` seconds.

Verdict: The oracle recurrent reference and clamped-spline projections were materialized. The r=12 nominal imitation row has action mismatch 24.54 and combined matrix residual 0.01274. 8 r=12 reward rows were optimized with bounded Adam over phase-modulated recurrent coefficients.

## Rows

| row | family | lens | verdict | objective ratio | action mismatch | matrix residual | io-map |
|---|---|---|---|---:|---:|---:|---|
| phase_modulated_recurrent__pm_linrec_r3_oracle_matrix_projection | oracle_matrix_projection | oracle_matrix_projection | representation_diagnostic | 1.1648125 | 12406.437 | 0.15105496 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_oracle_matrix_projection | oracle_matrix_projection | oracle_matrix_projection | representation_diagnostic | 1.1346193 | 191.5243 | 0.055506791 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_oracle_matrix_projection | oracle_matrix_projection | oracle_matrix_projection | representation_diagnostic | 1.0008825 | 64.732114 | 0.013642344 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_oracle_matrix_projection | oracle_matrix_projection | oracle_matrix_projection | representation_diagnostic | 1.0000609 | 24.543846 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_action_imitation_nominal | action_imitation | nominal_clean | imitation_diagnostic | 1.1648125 | 12406.437 | 0.15105496 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_action_imitation_nominal | action_imitation | nominal_clean | imitation_diagnostic | 1.1346193 | 191.5243 | 0.055506791 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_action_imitation_nominal | action_imitation | nominal_clean | imitation_diagnostic | 1.0008825 | 64.732114 | 0.013642344 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_action_imitation_nominal | action_imitation | nominal_clean | imitation_diagnostic | 1.0000609 | 24.543846 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_io_map_imitation_exact_process_eigen_m4 | io_map_imitation | exact_process_eigen_m4 | imitation_diagnostic | 1.5383958 | 12432.099 | 0.15105496 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_io_map_imitation_exact_process_eigen_m4 | io_map_imitation | exact_process_eigen_m4 | imitation_diagnostic | 1.4958733 | 53.167955 | 0.055506791 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_io_map_imitation_exact_process_eigen_m4 | io_map_imitation | exact_process_eigen_m4 | imitation_diagnostic | 1.3202128 | 64.158735 | 0.013642344 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_io_map_imitation_exact_process_eigen_m4 | io_map_imitation | exact_process_eigen_m4 | imitation_diagnostic | 1.3191389 | 24.378338 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_clean_scratch_reward | reward_lens | nominal_clean | reward_trained_non_equivalent | 3.7286225 | 687138.46 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_exact_process_eigen_m1_s0p3_reward | reward_lens | exact_process_eigen_m1_s0.3 | reward_trained_non_equivalent | 6.2331343 | 49349533 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_exact_process_eigen_m4_s0p3_reward | reward_lens | exact_process_eigen_m4_s0.3 | reward_trained_non_equivalent | 8.5118189 | 282685.51 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_exact_process_eigen_m4_s1_reward | reward_lens | exact_process_eigen_m4_s1 | reward_trained_non_equivalent | 131.23026 | 109415.74 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_observer_error_svd_m1_s0p3_reward | reward_lens | observer_error_svd_m1_s0.3 | reward_trained_non_equivalent | 4.6618076 | 184379.64 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_mixed_process_observer_reward | reward_lens | mixed_process_observer | reward_trained_non_equivalent | 89.270334 | 4058.2856 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_imitation_nominal_then_reward | imitation_then_reward_lens | nominal_clean | reward_trained_non_equivalent | 1.0000609 | 24.543846 | 0.012744718 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_imitation_exact_eigen_m4_then_reward | imitation_then_reward_lens | exact_process_eigen_m4_s0.3 | reward_trained_non_equivalent | 1.3191389 | 24.378338 | 0.012744718 | standard_components_available |

## Certificate Boundary

These rows are diagnostics, not bridge passes. The recurrent controller is an
augmented-linear system over `z_t = [x_t; h_t]`, and action/visited-state
components are reported through the existing augmented recurrent adapter. The
formal I/O-map certificate is consumed through the standard certificate
component builder; projection/imitation rows remain diagnostic even when those
components are available.

- `bellman_hessian_residual:missing`: 20
- `closed_loop_transition_mismatch:missing`: 20
- `disturbance_history_to_action_map_mismatch:available`: 20
- `disturbance_history_to_state_map_mismatch:available`: 20
- `observation_history_to_action_map_mismatch:available`: 20
- `optimizer_metadata:available`: 20
- `recurrence_gru_diagnostics:available`: 20
- `state_weighted_action_mismatch:available`: 20
- `value_policy_gap:missing`: 20
- `visited_subspace_diagnostics:available`: 20

## Interpretation

The oracle row constructs the exact finite-horizon Kalman recurrent reference.
Ranked rows then project each time-varying recurrent/readout matrix onto a
clamped B-spline partition of unity. Reward rows optimize trainable
phase-modulated recurrent coefficients against the true quadratic rollout
objective on their retained training distributions; imitation-then-reward rows
use the projected oracle controller as the warm start before reward fine-tuning.
