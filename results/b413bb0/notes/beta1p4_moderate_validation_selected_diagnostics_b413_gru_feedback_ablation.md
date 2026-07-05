# GRU Feedback Ablation Diagnostic

- Issue: `b413bb0`
- Source experiment: `b413bb0`
- Scope: `beta1p4_moderate_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `direct_epsilon` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `linear_no_bias` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `affine` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `direct_epsilon` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 240.303 | 0.00191778 | 0.00726483 |
| `direct_epsilon` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 240.303 | 0.00191778 | 0.00726483 |
| `direct_epsilon` | `nominal` | `shuffled_observation_history` | evaluated | 0.0499394 | -1.49955 | 1.13993e-05 | -1.64171e-05 |
| `direct_epsilon` | `nominal` | `lagged_observation_history` | evaluated | 0.0499394 | -1.49955 | 1.13993e-05 | -1.64171e-05 |
| `direct_epsilon` | `nominal` | `position_only_observation` | evaluated | 10.8807 | 3.57598e+06 | 0.612829 | 0.345459 |
| `direct_epsilon` | `nominal` | `velocity_only_observation` | evaluated | 3.23916 | 163131 | 0.14334 | 0.0489935 |
| `direct_epsilon` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.238183 | 1848.46 | 0.0112313 | 0.00680421 |
| `direct_epsilon` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.238183 | 1848.46 | 0.0112313 | 0.00680421 |
| `direct_epsilon` | `initial_state` | `shuffled_observation_history` | evaluated | 0.412191 | 240.612 | 0.00326227 | 0.00650875 |
| `direct_epsilon` | `initial_state` | `lagged_observation_history` | evaluated | 0.0492018 | -0.541561 | -2.25606e-05 | -7.30232e-06 |
| `direct_epsilon` | `initial_state` | `position_only_observation` | evaluated | 10.5676 | 3.10325e+06 | 0.573961 | 0.473535 |
| `direct_epsilon` | `initial_state` | `velocity_only_observation` | evaluated | 3.20226 | 131748 | 0.130301 | 0.0485329 |
| `direct_epsilon` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.467591 | 2586.42 | 0.0106915 | 0.039414 |
| `direct_epsilon` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.467591 | 2586.42 | 0.0106915 | 0.039414 |
| `direct_epsilon` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.413536 | 217.324 | 0.00272331 | 0.00625013 |
| `direct_epsilon` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0475636 | 1.69534 | -1.05735e-05 | -1.21757e-05 |
| `direct_epsilon` | `process_epsilon` | `position_only_observation` | evaluated | 11.1056 | 3.69143e+06 | 0.624481 | 0.349509 |
| `direct_epsilon` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.35445 | 158191 | 0.144289 | 0.0452839 |
| `direct_epsilon` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.186346 | 185.661 | 0.00349868 | 0.00708506 |
| `direct_epsilon` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0496057 | -2.15262 | 6.0164e-07 | -1.27657e-05 |
| `direct_epsilon` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.440932 | 1325.87 | 0.0100844 | 0.0320186 |
| `direct_epsilon` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.598306 | 12621.7 | 0.0359661 | 0.0485105 |
| `direct_epsilon` | `sensory_feedback` | `position_only_observation` | evaluated | 10.8426 | 3.57592e+06 | 0.61441 | 0.345279 |
| `direct_epsilon` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.20545 | 163076 | 0.144921 | 0.0488137 |
| `direct_epsilon` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 240.303 | 0.00191778 | 0.00726483 |
| `direct_epsilon` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 240.303 | 0.00191778 | 0.00726483 |
| `direct_epsilon` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0499394 | -1.49955 | 1.13993e-05 | -1.64171e-05 |
| `direct_epsilon` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0499394 | -1.49955 | 1.13993e-05 | -1.64171e-05 |
| `direct_epsilon` | `delayed_observation` | `position_only_observation` | evaluated | 10.8807 | 3.57598e+06 | 0.612829 | 0.345459 |
| `direct_epsilon` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0499394 | -1.49955 | 1.13993e-05 | -1.64171e-05 |
| `linear_no_bias` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 134.995 | 0.000662155 | 0.00250802 |
| `linear_no_bias` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 134.995 | 0.000662155 | 0.00250802 |
| `linear_no_bias` | `nominal` | `shuffled_observation_history` | evaluated | 0.171717 | -37.4693 | 0.000156975 | -0.00182058 |
| `linear_no_bias` | `nominal` | `lagged_observation_history` | evaluated | 0.171717 | -37.4693 | 0.000156975 | -0.00182058 |
| `linear_no_bias` | `nominal` | `position_only_observation` | evaluated | 5.44102 | 1.99927e+06 | 0.307627 | 0.548335 |
| `linear_no_bias` | `nominal` | `velocity_only_observation` | evaluated | 2.43574 | 138703 | 0.0734567 | 0.137266 |
| `linear_no_bias` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.319541 | 796.439 | 0.00415734 | 0.00376135 |
| `linear_no_bias` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.319541 | 796.439 | 0.00415734 | 0.00376135 |
| `linear_no_bias` | `initial_state` | `shuffled_observation_history` | evaluated | 0.499893 | 129.867 | 0.00117951 | 0.00254586 |
| `linear_no_bias` | `initial_state` | `lagged_observation_history` | evaluated | 0.157924 | 104.309 | 0.000116488 | 0.000542495 |
| `linear_no_bias` | `initial_state` | `position_only_observation` | evaluated | 5.30968 | 1.83487e+06 | 0.297803 | 0.560568 |
| `linear_no_bias` | `initial_state` | `velocity_only_observation` | evaluated | 2.40702 | 128071 | 0.0706284 | 0.138519 |
| `linear_no_bias` | `process_epsilon` | `normal` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `process_epsilon` | `frozen_nominal_observation_tape` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `process_epsilon` | `shuffled_observation_history` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `process_epsilon` | `lagged_observation_history` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `process_epsilon` | `position_only_observation` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `process_epsilon` | `velocity_only_observation` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.135563 | 93.0053 | 0.00129801 | 0.00175914 |
| `linear_no_bias` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.173445 | -62.6112 | 0.000135555 | -0.00202138 |
| `linear_no_bias` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.515012 | 644.685 | 0.00405928 | 0.0132392 |
| `linear_no_bias` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.649997 | 11715.1 | 0.0165689 | 0.0424565 |
| `linear_no_bias` | `sensory_feedback` | `position_only_observation` | evaluated | 5.41906 | 1.99923e+06 | 0.308263 | 0.547586 |
| `linear_no_bias` | `sensory_feedback` | `velocity_only_observation` | evaluated | 2.42591 | 138661 | 0.0740926 | 0.136517 |
| `linear_no_bias` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 134.995 | 0.000662155 | 0.00250802 |
| `linear_no_bias` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 134.995 | 0.000662155 | 0.00250802 |
| `linear_no_bias` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.171717 | -37.4693 | 0.000156975 | -0.00182058 |
| `linear_no_bias` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.171717 | -37.4693 | 0.000156975 | -0.00182058 |
| `linear_no_bias` | `delayed_observation` | `position_only_observation` | evaluated | 5.44102 | 1.99927e+06 | 0.307627 | 0.548335 |
| `linear_no_bias` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.171717 | -37.4693 | 0.000156975 | -0.00182058 |
| `affine` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `affine` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 106.949 | 0.000957527 | 0.00281075 |
| `affine` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 106.949 | 0.000957527 | 0.00281075 |
| `affine` | `nominal` | `shuffled_observation_history` | evaluated | 0.0247754 | -0.735542 | -7.86751e-06 | 1.36777e-05 |
| `affine` | `nominal` | `lagged_observation_history` | evaluated | 0.0247754 | -0.735542 | -7.86751e-06 | 1.36777e-05 |
| `affine` | `nominal` | `position_only_observation` | evaluated | 3.12321 | 4.17661e+06 | 0.380393 | 1.01115 |
| `affine` | `nominal` | `velocity_only_observation` | evaluated | 2.33703 | 140683 | 0.0741896 | 0.13109 |
| `affine` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `affine` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0424895 | 441.988 | 0.0026959 | 0.000493333 |
| `affine` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0424895 | 441.988 | 0.0026959 | 0.000493333 |
| `affine` | `initial_state` | `shuffled_observation_history` | evaluated | 0.128393 | 82.9186 | 0.000352841 | 0.00155943 |
| `affine` | `initial_state` | `lagged_observation_history` | evaluated | 0.0235543 | -0.223843 | 1.88237e-06 | 2.65975e-05 |
| `affine` | `initial_state` | `position_only_observation` | evaluated | 3.03945 | 3.96208e+06 | 0.370603 | 0.960339 |
| `affine` | `initial_state` | `velocity_only_observation` | evaluated | 2.3394 | 129385 | 0.0692719 | 0.128773 |
| `affine` | `process_epsilon` | `normal` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `process_epsilon` | `frozen_nominal_observation_tape` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `process_epsilon` | `shuffled_observation_history` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `process_epsilon` | `lagged_observation_history` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `process_epsilon` | `position_only_observation` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `process_epsilon` | `velocity_only_observation` | not_available | n/a | n/a | n/a | n/a |
| `affine` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `affine` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0223346 | 104.555 | 0.00139897 | 0.00261609 |
| `affine` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0248063 | -0.647685 | -1.67107e-05 | 1.47578e-05 |
| `affine` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.132901 | 220.969 | 0.00228859 | 0.00734805 |
| `affine` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.227668 | 6410.64 | 0.0153672 | 0.0348646 |
| `affine` | `sensory_feedback` | `position_only_observation` | evaluated | 3.11734 | 4.17661e+06 | 0.380834 | 1.01095 |
| `affine` | `sensory_feedback` | `velocity_only_observation` | evaluated | 2.32873 | 140681 | 0.074631 | 0.130895 |
| `affine` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `affine` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 106.949 | 0.000957527 | 0.00281075 |
| `affine` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 106.949 | 0.000957527 | 0.00281075 |
| `affine` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0247754 | -0.735542 | -7.86751e-06 | 1.36777e-05 |
| `affine` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0247754 | -0.735542 | -7.86751e-06 | 1.36777e-05 |
| `affine` | `delayed_observation` | `position_only_observation` | evaluated | 3.12321 | 4.17661e+06 | 0.380393 | 1.01115 |
| `affine` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0247754 | -0.735542 | -7.86751e-06 | 1.36777e-05 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `direct_epsilon` | available | 1.86236 | 3.35973 | 0.364987 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `linear_no_bias` | available | 1.43235 | 2.8536 | 0.0110959 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `affine` | available | 1.20982 | 2.41441 | 0.00523077 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `direct_epsilon` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `linear_no_bias` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `affine` | `warn` | `warn` | `pass` | `pass` | `warn` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `direct_epsilon` | 0 | 10000 | 4500 | -5500 | 304.033 | 6 |
| `direct_epsilon` | 1 | 11500 | 2500 | -9000 | 288.325 | 6 |
| `direct_epsilon` | 2 | 12000 | 3000 | -9000 | 272.657 | 6 |
| `direct_epsilon` | 3 | 12000 | 2000 | -10000 | 265.987 | 6 |
| `direct_epsilon` | 4 | 12000 | 2000 | -10000 | 273.497 | 6 |
| `linear_no_bias` | 0 | 12000 | 9500 | -2500 | 15776.9 | 4 |
| `linear_no_bias` | 1 | 11500 | 4000 | -7500 | 373.961 | 4 |
| `linear_no_bias` | 2 | 12000 | 12000 | 0 | 16039.3 | 4 |
| `linear_no_bias` | 3 | 12000 | 2000 | -10000 | 351.464 | 4 |
| `linear_no_bias` | 4 | 12000 | 12000 | 0 | 11496.8 | 4 |
| `affine` | 0 | 12000 | 10000 | -2000 | 408.123 | 4 |
| `affine` | 1 | 5000 | 7000 | 2000 | 16930.4 | 4 |
| `affine` | 2 | 4000 | 4000 | 0 | 17023.5 | 4 |
| `affine` | 3 | 11500 | 8000 | -3500 | 16996.4 | 4 |
| `affine` | 4 | 11500 | 11500 | 0 | 425.523 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
