# Phase-Modulated Linear Recurrent Output-Feedback Bridge

Issue: `d6d25d6`. Follow-up: `a06307d`. Umbrella: `1fabee8`.

Scope: Oracle Kalman recurrent reference plus clamped-spline phase-modulated linear recurrence. The spline basis modulates A/B/C/D matrices over tau=t/(T-1), not additive phase offsets.

Non-goals: No GRU training, no broad robust-epsilon arm, and no claim that projected-oracle diagnostic rows are bridge passes. Supervised rows fit action and response maps; they are not reward-trained rows.

Runtime: `9.38` seconds.

Verdict: The exact oracle and clamped-spline projected-oracle rows were materialized. Exact-oracle sanity rows have max aggregate response-map mismatch 0. The r=12 projected-oracle nominal replay row has action mismatch 24.54 and combined matrix residual 0.01274. 10 supervised action/response-map rows were optimized, with 5 representation pass rows. 0 r=12 reward rows were optimized after the supervised gate.

Audit note: The prior exact_process_eigen rows were state-trajectory covariance coverage directions, not process-eigen disturbance sequences. They are retained under state_coverage_eigen labels.

Reward gating: `not_requested`.

## Rows

| row | family | lens | verdict | objective ratio | action mismatch | R_u | matrix residual | obs action | meas action | meas output | proc action | proc output | cost sidecar | io-map |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| phase_modulated_recurrent__pm_linrec_exact_oracle_nominal_replay | exact_oracle_sanity | nominal_clean | exact_oracle_sanity_pass | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_exact_oracle_process_io | exact_oracle_sanity | process_io | exact_oracle_sanity_pass | 656.22396 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_exact_oracle_process_measurement_io | exact_oracle_sanity | process_measurement_io | exact_oracle_sanity_pass | 664.39558 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.1648125 | 12406.437 | 0.22134806 | 0.15105496 | 0.50047359 | 0.42434372 | 0.38757497 | 0.1038565 | 0.045304559 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.1346193 | 191.5243 | 0.15989193 | 0.055506791 | 0.43012287 | 0.46473334 | 0.41462247 | 0.58184335 | 0.42071436 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.0008825 | 64.732114 | 0.0017231064 | 0.013642344 | 0.20552348 | 0.27601756 | 0.24806742 | 0.029962722 | 0.017142024 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1.0000609 | 24.543846 | 0.00013843374 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1.0000002 | 0.12881805 | 4.1563062e-07 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1 | 0.040147103 | 1.7769645e-09 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1 | 1.6218418e-23 | 7.1289985e-29 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.63447 | 4.5471523 | 0.032497133 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.24546 | 3.2312072 | 0.0010099182 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.23243 | 0.085271906 | 2.1719823e-05 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.22396 | 2.8123112e-24 | 1.1347363e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 665.47391 | 2.2937547 | 0.032964512 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 665.09706 | 0.91649284 | 0.0038490703 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 664.94429 | 0.037814896 | 0.0019700806 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 664.39558 | 7.7824656e-25 | 1.2276543e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.5383958 | 12432.099 | 0.22286013 | 0.15105496 | 0.50047359 | 0.42434372 | 0.38757497 | 0.1038565 | 0.045304559 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.4958733 | 53.167955 | 0.1589176 | 0.055506791 | 0.43012287 | 0.46473334 | 0.41462247 | 0.58184335 | 0.42071436 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.3202128 | 64.158735 | 0.0016984704 | 0.013642344 | 0.20552348 | 0.27601756 | 0.24806742 | 0.029962722 | 0.017142024 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.3191389 | 24.378338 | 0.00013673777 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_nominal_action_fit | supervised_action_fit | nominal_clean | supervised_representation_pass | 1.0000147 | 27.985417 | 1.5626548e-05 | 0.012744718 | 1.0256714 | 1.0397089 | 0.98847649 | 1.0033363 | 0.98825244 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_mixed_history_action_fit | supervised_action_fit | mixed_process_observer | supervised_representation_pass | 656.96973 | 3.2848773 | 0.05568759 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_process_io_map_fit | supervised_io_map_fit | process_io | supervised_representation_non_equivalent | 656.63447 | 4.5471523 | 0.032497133 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_process_measurement_io_map_fit | supervised_io_map_fit | process_measurement_io | supervised_representation_non_equivalent | 665.47391 | 2.2937547 | 0.032964512 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_action_io_combined_fit | supervised_action_io_map_fit | mixed_process_measurement_io | supervised_representation_non_equivalent | 665.27938 | 3.8556437 | 0.060398973 | 0.012744718 | 0.24241362 | 0.29947423 | 0.2693138 | 0.10688281 | 0.044690504 | 0.0014325246 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_nominal_action_fit | supervised_action_fit | nominal_clean | supervised_representation_pass | 1 | 0.11561008 | 1.0351405e-08 | 0.0048766187 | 1.0189334 | 1.0289749 | 1.0179519 | 1.0525974 | 1.1369961 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_mixed_history_action_fit | supervised_action_fit | mixed_process_observer | supervised_representation_pass | 656.57658 | 3.1701086 | 0.025452985 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_process_io_map_fit | supervised_io_map_fit | process_io | supervised_representation_pass | 882.21692 | 126.24189 | 68.487841 | 0.0048766187 | 0.37085168 | 1.4134995 | 1.9362883 | 0.01492392 | 0.0058755214 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_process_measurement_io_map_fit | supervised_io_map_fit | process_measurement_io | supervised_representation_non_equivalent | 659.30715 | 23.850982 | 2.1270864 | 0.0048766187 | 0.14914217 | 0.17547 | 0.1459923 | 0.014035163 | 0.0055926243 | missing | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_action_io_combined_fit | supervised_action_io_map_fit | mixed_process_measurement_io | supervised_representation_non_equivalent | 665.70146 | 0.72429249 | 0.022875686 | 0.0048766187 | 0.14218922 | 0.1640421 | 0.12894744 | 0.013786942 | 0.0050817798 | 0.00018812412 | standard_components_available |

## Certificate Boundary

These rows are diagnostics, not bridge passes. The recurrent controller is an
augmented-linear system over `z_t = [x_t; h_t]`, and action/visited-state
components are reported through the existing augmented recurrent adapter. The
formal I/O-map certificate is consumed through the standard certificate
component builder; projected-oracle rows remain diagnostic even when those
components are available. Supervised rows are optimized against external action
and/or response-map losses and are not reward-trained.

- `bellman_hessian_residual:missing`: 32
- `closed_loop_transition_mismatch:missing`: 32
- `disturbance_history_to_action_map_mismatch:available`: 32
- `disturbance_history_to_cost_quadratic:available`: 2
- `disturbance_history_to_cost_quadratic:missing`: 30
- `disturbance_history_to_output_map_mismatch:available`: 32
- `disturbance_history_to_state_map_mismatch:available`: 32
- `measurement_history_to_action_map_mismatch:available`: 32
- `measurement_history_to_output_map_mismatch:available`: 32
- `observation_history_to_action_map_mismatch:available`: 32
- `optimizer_metadata:available`: 32
- `recurrence_gru_diagnostics:available`: 32
- `state_weighted_action_mismatch:available`: 32
- `value_policy_gap:missing`: 32
- `visited_subspace_diagnostics:available`: 32

## Interpretation

The exact-oracle rows replay the finite-horizon Kalman recurrent reference under
nominal, process I/O, and process+measurement I/O probes. Ranked rows then
project each time-varying recurrent/readout matrix onto a clamped B-spline
partition of unity. State-coverage eigen rows preserve the old coverage
semantics explicitly: they perturb initial state/estimator coverage directions
from a state-trajectory covariance, not process-eigen disturbances. Supervised
rows optimize trainable phase-modulated recurrent coefficients against exact
oracle action histories and/or finite-horizon response maps. Reward rows are
gated on supervised representation success and optimize the true quadratic
rollout objective on their retained training distributions only after that gate.
