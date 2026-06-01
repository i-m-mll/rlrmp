# Phase-Modulated Linear Recurrent Output-Feedback Bridge

Issue: `d6d25d6`. Follow-ups: `a06307d`, `ad309f5`. Umbrella: `1fabee8`.

Scope: Oracle Kalman recurrent reference plus clamped-spline phase-modulated linear recurrence. The spline basis modulates A/B/C/D matrices over tau=t/(T-1), not additive phase offsets.

Non-goals: No GRU training, no broad robust-epsilon arm, and no claim that projected-oracle diagnostic rows are bridge passes. Supervised rows fit action and response maps; they are not reward-trained rows. r=60 reward-control rows are capacity sanity checks, not compact bridge claims.

Runtime: `86.22` seconds.

Verdict: The exact oracle and clamped-spline projected-oracle rows were materialized. Exact-oracle sanity rows have max aggregate response-map mismatch 0. The r=12 projected-oracle nominal replay row has action mismatch 24.54 and combined matrix residual 0.01274. 20 supervised action/response-map rows were optimized, with 13 representation pass rows. 14 reward rows were optimized after the supervised gate.

Audit note: The prior exact_process_eigen rows were state-trajectory covariance coverage directions, not process-eigen disturbance sequences. They are retained under state_coverage_eigen labels.

Reward gating: `released_after_supervised_action_io_representation_pass`.

## Rows

| row | family | lens | verdict | clean objective ratio | lens objective ratio | action mismatch | R_u | matrix residual | obs action | meas action | meas output | proc action | proc output | cost sidecar | io-map |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| phase_modulated_recurrent__pm_linrec_exact_oracle_nominal_replay | exact_oracle_sanity | nominal_clean | exact_oracle_sanity_pass | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_exact_oracle_process_io | exact_oracle_sanity | process_io | exact_oracle_sanity_pass | 656.22396 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_exact_oracle_process_measurement_io | exact_oracle_sanity | process_measurement_io | exact_oracle_sanity_pass | 664.39558 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.1648125 | 1.1648125 | 12406.437 | 0.22134806 | 0.15105496 | 0.50047359 | 0.42434372 | 0.38757497 | 0.1038565 | 0.045304559 | 0.0042755588 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.1346193 | 1.1346193 | 191.5243 | 0.15989193 | 0.055506791 | 0.43012287 | 0.46473334 | 0.41462247 | 0.58184335 | 0.42071436 | 0.019030453 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_legacy_projected_oracle_nominal_replay | legacy_projected_oracle_replay | nominal_clean | legacy_projected_oracle_replay_diagnostic | 1.0008825 | 1.0008825 | 64.732114 | 0.0017231064 | 0.013642344 | 0.20552348 | 0.27601756 | 0.24806742 | 0.029962722 | 0.017142024 | 0.00061247882 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1.0000609 | 1.0000609 | 24.543846 | 0.00013843374 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1.0000002 | 1.0000002 | 0.12881805 | 4.1563062e-07 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | 0.00021210491 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1 | 1 | 0.040147103 | 1.7769645e-09 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | 3.5319737e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_nominal_replay | projected_oracle_replay | nominal_clean | projected_oracle_replay_diagnostic | 1 | 1 | 1.6218418e-23 | 7.1289985e-29 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.63447 | 1.0006256 | 4.5471523 | 0.032497133 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.24546 | 1.0000328 | 3.2312072 | 0.0010099182 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | 0.00021210491 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.23243 | 1.0000129 | 0.085271906 | 2.1719823e-05 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | 3.5319737e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_io_eval | projected_oracle_io_eval | process_io | projected_oracle_io_diagnostic | 656.22396 | 1 | 2.8123112e-24 | 1.1347363e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 665.47391 | 1.001623 | 2.2937547 | 0.032964512 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 665.09706 | 1.0010558 | 0.91649284 | 0.0038490703 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | 0.00021210491 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 664.94429 | 1.0008259 | 0.037814896 | 0.0019700806 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | 3.5319737e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_measurement_io_eval | projected_oracle_io_eval | process_measurement_io | projected_oracle_io_diagnostic | 664.39558 | 1 | 7.7824656e-25 | 1.2276543e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r3_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.5383958 | 1.1662826 | 12432.099 | 0.22286013 | 0.15105496 | 0.50047359 | 0.42434372 | 0.38757497 | 0.1038565 | 0.045304559 | 0.0042755588 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r5_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.4958733 | 1.1340457 | 53.167955 | 0.1589176 | 0.055506791 | 0.43012287 | 0.46473334 | 0.41462247 | 0.58184335 | 0.42071436 | 0.019030453 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r8_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.3202128 | 1.0008746 | 64.158735 | 0.0016984704 | 0.013642344 | 0.20552348 | 0.27601756 | 0.24806742 | 0.029962722 | 0.017142024 | 0.00061247882 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_state_coverage_eigen_m4_s0p3_eval | projected_oracle_state_coverage_eval | state_coverage_eigen_m4_s0.3 | state_coverage_projection_diagnostic | 1.3191389 | 1.0000605 | 24.378338 | 0.00013673777 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_nominal_action_fit | supervised_action_fit | nominal_clean | supervised_representation_pass | 1.0000147 | 1.0000147 | 27.985417 | 1.5626548e-05 | 0.012744718 | 1.0256714 | 1.0397089 | 0.98847649 | 1.0033363 | 0.98825244 | 0.63562098 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_mixed_history_action_fit | supervised_action_fit | mixed_process_observer | supervised_representation_pass | 656.96973 | 1.0006602 | 3.2848773 | 0.05568759 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_process_io_map_fit | supervised_io_map_fit | process_io | supervised_representation_non_equivalent | 656.63447 | 1.0006256 | 4.5471523 | 0.032497133 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_process_measurement_io_map_fit | supervised_io_map_fit | process_measurement_io | supervised_representation_non_equivalent | 665.47391 | 1.001623 | 2.2937547 | 0.032964512 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_supervised_action_io_combined_fit | supervised_action_io_map_fit | mixed_process_measurement_io | supervised_representation_non_equivalent | 665.27938 | 1.0008598 | 3.8556437 | 0.060398973 | 0.012744718 | 0.24241362 | 0.29947423 | 0.2693138 | 0.10688281 | 0.044690504 | 0.0014325246 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_nominal_action_fit | supervised_action_fit | nominal_clean | supervised_representation_pass | 1 | 1 | 0.11561008 | 1.0351405e-08 | 0.0048766187 | 1.0189334 | 1.0289749 | 1.0179519 | 1.0525974 | 1.1369961 | 1.3194351 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_mixed_history_action_fit | supervised_action_fit | mixed_process_observer | supervised_representation_pass | 656.57658 | 1.0000614 | 3.1701086 | 0.025452985 | 0.0048766187 | 0.16277743 | 0.21845499 | 0.20591281 | 0.015810533 | 0.0059635834 | 0.00021210491 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_process_io_map_fit | supervised_io_map_fit | process_io | supervised_representation_pass | 882.21692 | 1.3443839 | 126.24189 | 68.487841 | 0.0048766187 | 0.37085168 | 1.4134995 | 1.9362883 | 0.01492392 | 0.0058755214 | 0.0001926193 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_process_measurement_io_map_fit | supervised_io_map_fit | process_measurement_io | supervised_representation_non_equivalent | 659.30715 | 0.99234127 | 23.850982 | 2.1270864 | 0.0048766187 | 0.14914217 | 0.17547 | 0.1459923 | 0.014035163 | 0.0055926243 | 0.00019873616 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r20_supervised_action_io_combined_fit | supervised_action_io_map_fit | mixed_process_measurement_io | supervised_representation_non_equivalent | 665.70146 | 1.0014947 | 0.72429249 | 0.022875686 | 0.0048766187 | 0.14218922 | 0.1640421 | 0.12894744 | 0.013786942 | 0.0050817798 | 0.00018812412 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_supervised_nominal_action_fit | supervised_action_fit | nominal_clean | supervised_representation_pass | 1 | 1 | 0.040147103 | 1.7769645e-09 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | 3.5319737e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_supervised_mixed_history_action_fit | supervised_action_fit | mixed_process_observer | supervised_representation_pass | 656.55338 | 1.000026 | 0.096109332 | 0.015919673 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | 3.5319737e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_supervised_process_io_map_fit | supervised_io_map_fit | process_io | supervised_representation_pass | 656.23243 | 1.0000129 | 0.085271906 | 2.1719823e-05 | 0.0023077801 | 0.11044878 | 0.14442961 | 0.12819103 | 0.0030258819 | 0.0011885646 | 3.5319737e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_supervised_process_measurement_io_map_fit | supervised_io_map_fit | process_measurement_io | supervised_representation_non_equivalent | 648.43533 | 0.97597778 | 25.349238 | 3.5710715 | 0.0023077801 | 0.098781992 | 0.097970454 | 0.070783073 | 0.0044710522 | 0.0016322143 | 4.0606444e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r30_supervised_action_io_combined_fit | supervised_action_io_map_fit | mixed_process_measurement_io | supervised_representation_non_equivalent | 665.47278 | 1.0011507 | 0.032858696 | 0.013284804 | 0.0023077801 | 0.085122893 | 0.087322038 | 0.061820837 | 0.0043578959 | 0.0013778241 | 3.4225134e-05 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_nominal_action_fit | supervised_action_fit | nominal_clean | supervised_representation_pass | 1 | 1 | 1.6218418e-23 | 7.1289985e-29 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_mixed_history_action_fit | supervised_action_fit | mixed_process_observer | supervised_representation_pass | 656.53628 | 1 | 2.4829779e-24 | 8.9673832e-27 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_process_io_map_fit | supervised_io_map_fit | process_io | supervised_representation_pass | 656.22396 | 1 | 2.8123112e-24 | 1.1347363e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_process_measurement_io_map_fit | supervised_io_map_fit | process_measurement_io | supervised_representation_pass | 664.39558 | 1 | 7.7824656e-25 | 1.2276543e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_action_io_combined_fit | supervised_action_io_map_fit | mixed_process_measurement_io | supervised_representation_pass | 664.7079 | 1 | 7.6625162e-25 | 9.3830279e-27 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_clean_scratch_reward | reward_lens | nominal_clean | reward_trained_non_equivalent | 31.934966 | 31.934966 | 5796.5988 | 1.0218078 | 0.012744718 | 0.99988702 | 1.0000612 | 0.9884762 | 1.0020975 | 0.99515748 | 0.71422981 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_state_coverage_eigen_m1_s0p3_reward | reward_lens | state_coverage_eigen_m1_s0.3 | reward_trained_non_equivalent | 48.389838 | 33.500657 | 3836.7721 | 1.0156628 | 0.012744718 | 0.99988958 | 1.0000559 | 0.98846467 | 1.0020559 | 0.99505977 | 0.7148736 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_state_coverage_eigen_m4_s0p3_reward | reward_lens | state_coverage_eigen_m4_s0.3 | reward_trained_non_equivalent | 51.222823 | 38.832848 | 5233.9186 | 1.014667 | 0.012744718 | 0.99979579 | 0.99992273 | 0.98829438 | 1.0017605 | 0.99444759 | 0.70500419 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_state_coverage_eigen_m4_s1_reward | reward_lens | state_coverage_eigen_m4_s1 | reward_trained_non_equivalent | 237.69149 | 52.29619 | 23723.924 | 1.0029649 | 0.012744718 | 0.99977113 | 0.99992026 | 0.98830972 | 1.0022063 | 0.99509219 | 0.69948122 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_observer_error_svd_m1_s0p3_reward | reward_lens | observer_error_svd_m1_s0.3 | reward_trained_non_equivalent | 31.946049 | 22.655274 | 1140.3807 | 1.0128503 | 0.012744718 | 0.99988699 | 1.0000612 | 0.98847543 | 1.0020962 | 0.99515574 | 0.7142125 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_mixed_process_observer_reward | reward_lens | mixed_process_observer | reward_trained_non_equivalent | 507.60692 | 0.77315899 | 4.1912493 | 0.98886474 | 0.012744718 | 1.0000585 | 0.99984791 | 0.98756518 | 1.0006948 | 0.99295004 | 0.725592 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_nominal_then_reward | projection_warm_start_then_reward_lens | nominal_clean | reward_trained_non_equivalent | 1.0000609 | 1.0000609 | 24.543846 | 0.00013843374 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r12_projected_oracle_state_coverage_eigen_m4_then_reward | projection_warm_start_then_reward_lens | state_coverage_eigen_m4_s0.3 | reward_trained_non_equivalent | 1.3191389 | 1.0000605 | 24.378338 | 0.00013673777 | 0.012744718 | 0.25422973 | 0.3049617 | 0.29497161 | 0.10900394 | 0.047580488 | 0.001471869 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_nominal_then_reward | projection_warm_start_then_reward_lens | nominal_clean | reward_trained_reference_equivalent | 1 | 1 | 1.6218418e-23 | 7.1289985e-29 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_projected_oracle_process_measurement_then_reward | projection_warm_start_then_reward_lens | process_measurement_io | reward_trained_reference_equivalent | 664.39558 | 1 | 7.7824656e-25 | 1.2276543e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_action_io_nominal_then_reward | supervised_action_io_warm_start_then_reward_lens | nominal_clean | reward_trained_reference_equivalent | 1 | 1 | 1.6218418e-23 | 7.1289985e-29 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_supervised_action_io_process_measurement_then_reward | supervised_action_io_warm_start_then_reward_lens | process_measurement_io | reward_trained_reference_equivalent | 664.39558 | 1 | 7.7824656e-25 | 1.2276543e-26 | 1.5500104e-15 | 1.5641413e-26 | 1.1732871e-26 | 1.2577263e-26 | 1.2974114e-26 | 4.4689595e-27 | 1.9656016e-28 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_clean_scratch_reward | reward_lens | nominal_clean | reward_trained_non_equivalent | 35.886158 | 35.886158 | 873.91247 | 1.004666 | 1.5500104e-15 | 0.99993807 | 0.99995911 | 0.98814582 | 1.0007677 | 0.99277049 | 0.72153222 | standard_components_available |
| phase_modulated_recurrent__pm_linrec_r60_process_measurement_scratch_reward | reward_lens | process_measurement_io | reward_trained_non_equivalent | 513.82981 | 0.77337934 | 1.1034511 | 0.99366306 | 1.5500104e-15 | 1.0000544 | 0.9999954 | 0.98793557 | 1.0005022 | 0.99235074 | 0.72758166 | standard_components_available |

## Certificate Boundary

These rows are diagnostics, not bridge passes. The recurrent controller has an
augmented state `z_t = [x_t; h_t]`, but these rows do not yet supply the full
augmented-linear certificate contract (augmented action sensitivities,
transitions, value matrices, and Bellman Hessians). The standard certificate
therefore uses the recurrent I/O-map path: external response-map components and
visited-state/action diagnostics are available, while static transition/value
rows are explicitly not applicable. Projected-oracle rows remain diagnostic
even when response-map components are available. Supervised rows are optimized
against external action and/or response-map losses and are not reward-trained.

- `bellman_hessian_residual:not_applicable`: 56
- `closed_loop_transition_mismatch:not_applicable`: 56
- `disturbance_history_to_action_map_mismatch:available`: 56
- `disturbance_history_to_cost_quadratic:available`: 56
- `disturbance_history_to_output_map_mismatch:available`: 56
- `disturbance_history_to_state_map_mismatch:available`: 56
- `measurement_history_to_action_map_mismatch:available`: 56
- `measurement_history_to_output_map_mismatch:available`: 56
- `observation_history_to_action_map_mismatch:available`: 56
- `optimizer_metadata:available`: 56
- `recurrence_gru_diagnostics:available`: 56
- `state_weighted_action_mismatch:available`: 56
- `value_policy_gap:not_applicable`: 56
- `visited_subspace_diagnostics:available`: 56

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
The r=60 reward-control rows are capacity sanity checks: projected-oracle and
supervised action+I/O warm starts carry low-learning-rate Adam, gradient
clipping, and proximal preservation metadata to separate preservation behavior
from scratch discovery rows.
