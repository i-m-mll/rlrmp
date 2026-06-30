# ae9f30f Perturbation and Feedback Diagnostic Summary

Generated from local materialized diagnostics for the three completed `ae9f30f` rows. `linear_no_bias_b1p4` remains stopped context and is intentionally excluded.

## Inputs

- perturbation_detail: `_artifacts/ae9f30f/perturbation_response/gru_soft_lambda_wave1_trained_final/gru_perturbation_response_soft_lambda_wave1_trained_final_manifest_detail.json`
- feedback_ablation_detail: `_artifacts/ae9f30f/feedback_ablation/gru_soft_lambda_wave1_trained_final/gru_feedback_ablation_soft_lambda_wave1_trained_final_detail.json`
- baseline_perturbation_detail: `_artifacts/020a65b/perturbation_response/gru_h0_pgd_bank_two_rows_validation_selected_calibrated/gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected_calibrated_manifest_detail.json`
- baseline_feedback_note: `results/020a65b/notes/gru_feedback_ablation_h0_pgd_bank_two_rows_validation_selected.md`

Checkpoint policy: ae9f30f perturbation and feedback-ablation materializers both report `validation_selected_per_replicate`; the existing baseline comparison uses the same policy.

## Perturbation Reach/Stabilization by Level and Timing

| row | level | timing | n | endpoint/reach | peak dx/open-loop | AUC dx | recovery s | dFull-QRF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_epsilon_b1p05 | small | early | 18 | 6.433e-04 | 0.359 | 7.332e-04 | 0.5056 | 20.4 |
| direct_epsilon_b1p05 | small | mid | 18 | 0.003427 | 0.3797 | 8.010e-04 | n/a | 53.69 |
| direct_epsilon_b1p05 | small | late | 18 | 0.01847 | 0.5351 | 7.371e-04 | n/a | 239.8 |
| direct_epsilon_b1p05 | moderate | early | 18 | 0.002294 | 0.359 | 0.001467 | 0.5068 | 81.85 |
| direct_epsilon_b1p05 | moderate | mid | 18 | 0.009635 | 0.3797 | 0.001602 | n/a | 216.3 |
| direct_epsilon_b1p05 | moderate | late | 18 | 0.04331 | 0.5311 | 0.001468 | n/a | 957.1 |
| direct_epsilon_b1p05 | stress | early | 18 | 0.008914 | 0.3591 | 0.003669 | 0.5119 | 521.1 |
| direct_epsilon_b1p05 | stress | mid | 18 | 0.0309 | 0.38 | 0.004005 | n/a | 1412 |
| direct_epsilon_b1p05 | stress | late | 18 | 0.1134 | 0.5246 | 0.003612 | n/a | 6148 |
| direct_epsilon_b1p4 | small | early | 18 | 0.001348 | 0.3628 | 8.187e-04 | n/a | 22.73 |
| direct_epsilon_b1p4 | small | mid | 18 | 0.004535 | 0.3818 | 8.717e-04 | n/a | 56.56 |
| direct_epsilon_b1p4 | small | late | 18 | 0.01763 | 0.5396 | 7.403e-04 | n/a | 244 |
| direct_epsilon_b1p4 | moderate | early | 18 | 0.004335 | 0.3628 | 0.001637 | n/a | 91.08 |
| direct_epsilon_b1p4 | moderate | mid | 18 | 0.01241 | 0.3819 | 0.001742 | n/a | 227.2 |
| direct_epsilon_b1p4 | moderate | late | 18 | 0.04252 | 0.5363 | 0.001476 | n/a | 974.6 |
| direct_epsilon_b1p4 | stress | early | 18 | 0.01541 | 0.363 | 0.004085 | n/a | 575.6 |
| direct_epsilon_b1p4 | stress | mid | 18 | 0.03905 | 0.3821 | 0.004339 | n/a | 1457 |
| direct_epsilon_b1p4 | stress | late | 18 | 0.1154 | 0.5319 | 0.00365 | n/a | 6257 |
| linear_no_bias_b1p05 | small | early | 6 | 2.152e-04 | 0.1625 | 3.820e-04 | n/a | 5.388 |
| linear_no_bias_b1p05 | small | mid | 6 | 6.722e-04 | 0.1872 | 4.170e-04 | n/a | 13.37 |
| linear_no_bias_b1p05 | small | late | 6 | 0.00191 | 0.2532 | 3.056e-04 | n/a | 107.6 |
| linear_no_bias_b1p05 | moderate | early | 6 | 8.522e-04 | 0.1625 | 7.637e-04 | n/a | 21.56 |
| linear_no_bias_b1p05 | moderate | mid | 6 | 0.002603 | 0.1873 | 8.326e-04 | n/a | 53.41 |
| linear_no_bias_b1p05 | moderate | late | 6 | 0.009118 | 0.2595 | 6.289e-04 | n/a | 432.1 |
| linear_no_bias_b1p05 | stress | early | 6 | 0.005111 | 0.1628 | 0.001904 | n/a | 135.4 |
| linear_no_bias_b1p05 | stress | mid | 6 | 0.01501 | 0.1874 | 0.002059 | n/a | 331.1 |
| linear_no_bias_b1p05 | stress | late | 6 | 0.046 | 0.2777 | 0.001588 | n/a | 3195 |

## No-PGD H0 6D const_band16 Baseline: Same-Bank Perturbation Comparison

| row | level | timing | n | endpoint/reach | peak dx/open-loop | AUC dx | recovery s | dFull-QRF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_pgd_h0_6d_const_band16 | small | early | 22 | 0.002844 | 0.3649 | 0.001027 | n/a | 51.34 |
| no_pgd_h0_6d_const_band16 | small | mid | 22 | 0.004316 | 0.4011 | 9.576e-04 | n/a | 80.79 |
| no_pgd_h0_6d_const_band16 | small | late | 22 | 0.01061 | 0.5708 | 6.913e-04 | n/a | 353 |
| no_pgd_h0_6d_const_band16 | moderate | early | 22 | 0.01047 | 0.3649 | 0.002054 | n/a | 205.4 |
| no_pgd_h0_6d_const_band16 | moderate | mid | 22 | 0.01481 | 0.4012 | 0.001915 | n/a | 323.3 |
| no_pgd_h0_6d_const_band16 | moderate | late | 22 | 0.03381 | 0.572 | 0.001381 | n/a | 1428 |
| no_pgd_h0_6d_const_band16 | stress | early | 22 | 0.03971 | 0.3651 | 0.005132 | n/a | 1284 |
| no_pgd_h0_6d_const_band16 | stress | mid | 22 | 0.05546 | 0.4017 | 0.004789 | n/a | 2028 |
| no_pgd_h0_6d_const_band16 | stress | late | 22 | 0.1071 | 0.5792 | 0.00345 | n/a | 9509 |

## Sensory vs Non-Sensory Perturbation Robustness

| row | group | n | endpoint/reach | peak dx/open-loop | AUC dx | recovery s | dFull-QRF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_epsilon_b1p05 | sensory | 108 | 0.01085 | 0.1685 | 8.787e-04 | n/a | 426.7 |
| direct_epsilon_b1p05 | non_sensory | 186 | 0.02247 | 0.4467 | 0.002144 | 0.3872 | 952.7 |
| direct_epsilon_b1p4 | sensory | 108 | 0.01123 | 0.1475 | 9.025e-04 | n/a | 395.7 |
| direct_epsilon_b1p4 | non_sensory | 186 | 0.02477 | 0.45 | 0.002289 | 0.3793 | 978.4 |
| linear_no_bias_b1p05 | sensory | 108 | 0.00775 | 0.1223 | 9.308e-04 | 0.5608 | 386.4 |
| linear_no_bias_b1p05 | non_sensory | 78 | 0.006598 | 0.3297 | 0.001739 | 0.4773 | 380.8 |
| no_pgd_h0_6d_const_band16 | sensory | 108 | 0.006577 | 0.06814 | 8.090e-04 | n/a | 422.3 |
| no_pgd_h0_6d_const_band16 | non_sensory | 222 | 0.03033 | 0.465 | 0.002697 | n/a | 1595 |

## Feedback-Ablation Supported Deltas

AUC is not emitted by the current feedback-ablation pipeline; AUC-bearing values above come from perturbation-response diagnostics.

| row | group | mode | n | dAction | dFull-QRF | dEndpoint | dTerminal speed | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_epsilon_b1p05 | nominal | frozen_nominal_observation_tape | 1 | 0 | 283.1 | 0.003279 | 0.006445 | not supported |
| direct_epsilon_b1p05 | nominal | lagged_observation_history | 1 | 0.6181 | 2.383e+04 | 0.05023 | 0.09082 | not supported |
| direct_epsilon_b1p05 | nominal | zeroed_perturbation_observation_deviation | 1 | 0 | 283.1 | 0.003279 | 0.006445 | not supported |
| direct_epsilon_b1p05 | non_sensory | frozen_nominal_observation_tape | 2 | 0.3775 | 913.8 | 0.006423 | 0.01196 | not supported |
| direct_epsilon_b1p05 | non_sensory | lagged_observation_history | 2 | 0.3413 | 1.160e+04 | 0.02504 | 0.04358 | not supported |
| direct_epsilon_b1p05 | non_sensory | zeroed_perturbation_observation_deviation | 2 | 0.3775 | 913.8 | 0.006423 | 0.01196 | not supported |
| direct_epsilon_b1p05 | sensory | frozen_nominal_observation_tape | 2 | 0.07984 | 249 | 0.003344 | 0.006408 | not supported |
| direct_epsilon_b1p05 | sensory | lagged_observation_history | 2 | 0.6337 | 2.077e+04 | 0.04708 | 0.078 | not supported |
| direct_epsilon_b1p05 | sensory | zeroed_perturbation_observation_deviation | 2 | 0.1098 | 107.8 | 0.001699 | 0.003189 | not supported |
| direct_epsilon_b1p4 | nominal | frozen_nominal_observation_tape | 1 | 0 | 268.4 | 0.002944 | 0.006331 | not supported |
| direct_epsilon_b1p4 | nominal | lagged_observation_history | 1 | 0.5871 | 2.296e+04 | 0.04864 | 0.0902 | not supported |
| direct_epsilon_b1p4 | nominal | zeroed_perturbation_observation_deviation | 1 | 0 | 268.4 | 0.002944 | 0.006331 | not supported |
| direct_epsilon_b1p4 | non_sensory | frozen_nominal_observation_tape | 2 | 0.3319 | 873.5 | 0.005976 | 0.01178 | not supported |
| direct_epsilon_b1p4 | non_sensory | lagged_observation_history | 2 | 0.3231 | 1.124e+04 | 0.0244 | 0.04297 | not supported |
| direct_epsilon_b1p4 | non_sensory | zeroed_perturbation_observation_deviation | 2 | 0.3319 | 873.5 | 0.005976 | 0.01178 | not supported |
| direct_epsilon_b1p4 | sensory | frozen_nominal_observation_tape | 2 | 0.06516 | 243.6 | 0.003174 | 0.006348 | not supported |
| direct_epsilon_b1p4 | sensory | lagged_observation_history | 2 | 0.6009 | 2.033e+04 | 0.04612 | 0.07884 | not supported |
| direct_epsilon_b1p4 | sensory | zeroed_perturbation_observation_deviation | 2 | 0.09336 | 109.6 | 0.001692 | 0.003211 | not supported |
| linear_no_bias_b1p05 | nominal | frozen_nominal_observation_tape | 1 | 0 | 203.5 | 0.001224 | 0.006679 | not supported |
| linear_no_bias_b1p05 | nominal | lagged_observation_history | 1 | 0.578 | 1.693e+04 | 0.03904 | 0.07791 | not supported |
| linear_no_bias_b1p05 | nominal | zeroed_perturbation_observation_deviation | 1 | 0 | 203.5 | 0.001224 | 0.006679 | not supported |
| linear_no_bias_b1p05 | non_sensory | frozen_nominal_observation_tape | 1 | 0.1643 | 657.2 | 0.003355 | 0.006694 | not supported |
| linear_no_bias_b1p05 | non_sensory | lagged_observation_history | 1 | 0.05224 | -2.97 | -1.073e-05 | 2.532e-05 | not supported |
| linear_no_bias_b1p05 | non_sensory | zeroed_perturbation_observation_deviation | 1 | 0.1643 | 657.2 | 0.003355 | 0.006694 | not supported |
| linear_no_bias_b1p05 | sensory | frozen_nominal_observation_tape | 2 | 0.04701 | 200.1 | 0.001616 | 0.006674 | not supported |
| linear_no_bias_b1p05 | sensory | lagged_observation_history | 2 | 0.5881 | 1.543e+04 | 0.03763 | 0.07027 | not supported |
| linear_no_bias_b1p05 | sensory | zeroed_perturbation_observation_deviation | 2 | 0.07349 | 96.11 | 9.953e-04 | 0.003349 | not supported |

## Normalized Feedback-Use Indices

| row | status | score | ablation dependence | perturbation rescue | correction vs open-loop | warnings |
| --- | --- | --- | --- | --- | --- | --- |
| direct_epsilon_b1p05 | available | 2.053 | 3.912 | 0.1933 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| direct_epsilon_b1p4 | available | 1.923 | 3.666 | 0.1812 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| linear_no_bias_b1p05 | available | 1.1 | 2.059 | 0.1415 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| no_pgd_h0_6d_const_band16 | available | 0.03079 | 0.01546 | 0.04611 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Missing Metrics

- `feedback_ablation_auc`: not_supported - src/rlrmp/analysis/pipelines/gru_feedback_ablation.py emits paired dAction, dFull-QRF, dEndpoint, dTerminal-speed deltas and normalized feedback-use indices, but no AUC field in ablation rows or summary tables.
- `feedback_ablation_cli_feedback_scale_manifest_flag`: missing_cli_flag - scripts/materialize_gru_feedback_ablation.py does not expose feedback_scale_manifest_path, so calibrated force/filter rows require invoking materialize_gru_feedback_ablation through the Python API.

## Notes

- Perturbation levels are parsed from calibrated perturbation IDs (`small`, `moderate`, `stress`); timing uses materialized `timing_bin` (`early`, `mid`, `late`).
- Sensory perturbations are `sensory_feedback`; non-sensory aggregates `initial_state`, `process_epsilon`, and `command_input`.
- Lower `endpoint/reach`, `peak dx/open-loop`, `AUC dx`, `recovery s`, and `dFull-QRF` indicate smaller perturbation effects under these diagnostic summaries.
