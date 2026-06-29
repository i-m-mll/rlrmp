# GRU Feedback Ablation Diagnostic

- Issue: `d55c5f0`
- Source experiment: `d55c5f0`
- Scope: `soft_pgd_feedback_robustness_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `soft_pgd_ofb1p05` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `soft_pgd_ofb1p4` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `soft_pgd_ofb1p8` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `soft_pgd_ofb1p05` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p05` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `nominal` | `shuffled_observation_history` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `nominal` | `lagged_observation_history` | evaluated | 0.537431 | 20601.2 | 0.0436301 | 0.0872293 |
| `soft_pgd_ofb1p05` | `nominal` | `position_only_observation` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `nominal` | `velocity_only_observation` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p05` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0468594 | -0.764127 | -1.82716e-05 | -1.20739e-05 |
| `soft_pgd_ofb1p05` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0468594 | -0.764127 | -1.82716e-05 | -1.20739e-05 |
| `soft_pgd_ofb1p05` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0468594 | -0.764127 | -1.82716e-05 | -1.20739e-05 |
| `soft_pgd_ofb1p05` | `initial_state` | `lagged_observation_history` | evaluated | 0.535124 | 21618.9 | 0.0487292 | 0.0835724 |
| `soft_pgd_ofb1p05` | `initial_state` | `position_only_observation` | evaluated | 0.0468594 | -0.764127 | -1.82716e-05 | -1.20739e-05 |
| `soft_pgd_ofb1p05` | `initial_state` | `velocity_only_observation` | evaluated | 0.0468594 | -0.764127 | -1.82716e-05 | -1.20739e-05 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0456372 | 1.49363 | -1.44013e-05 | -2.16833e-05 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0456372 | 1.49363 | -1.44013e-05 | -2.16833e-05 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0456372 | 1.49363 | -1.44013e-05 | -2.16833e-05 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.541806 | 21507.5 | 0.0468773 | 0.0852618 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `position_only_observation` | evaluated | 0.0456372 | 1.49363 | -1.44013e-05 | -2.16833e-05 |
| `soft_pgd_ofb1p05` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0456372 | 1.49363 | -1.44013e-05 | -2.16833e-05 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0475391 | -2.35571 | 1.16737e-05 | -1.89574e-05 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0475391 | -2.35571 | 1.16737e-05 | -1.89574e-05 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0475391 | -2.35571 | 1.16737e-05 | -1.89574e-05 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0475391 | -2.35571 | 1.16737e-05 | -1.89574e-05 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0475391 | -2.35571 | 1.16737e-05 | -1.89574e-05 |
| `soft_pgd_ofb1p05` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.18039 | 162538 | 0.142431 | 0.0962778 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 242.049 | 0.00179429 | 0.00697553 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.374529 | 239.648 | 0.00166974 | 0.00673984 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.537431 | 20601.2 | 0.0436301 | 0.0872293 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `position_only_observation` | evaluated | 0.047972 | -1.81894 | 1.27238e-05 | -9.59641e-06 |
| `soft_pgd_ofb1p05` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.22846 | 162582 | 0.140852 | 0.0965584 |
| `soft_pgd_ofb1p4` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p4` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `nominal` | `shuffled_observation_history` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `nominal` | `lagged_observation_history` | evaluated | 0.535628 | 20199.5 | 0.0431573 | 0.0863466 |
| `soft_pgd_ofb1p4` | `nominal` | `position_only_observation` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `nominal` | `velocity_only_observation` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p4` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0466673 | -0.920926 | -1.93359e-05 | -1.47923e-05 |
| `soft_pgd_ofb1p4` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0466673 | -0.920926 | -1.93359e-05 | -1.47923e-05 |
| `soft_pgd_ofb1p4` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0466673 | -0.920926 | -1.93359e-05 | -1.47923e-05 |
| `soft_pgd_ofb1p4` | `initial_state` | `lagged_observation_history` | evaluated | 0.533421 | 21265 | 0.0483701 | 0.082447 |
| `soft_pgd_ofb1p4` | `initial_state` | `position_only_observation` | evaluated | 0.0466673 | -0.920926 | -1.93359e-05 | -1.47923e-05 |
| `soft_pgd_ofb1p4` | `initial_state` | `velocity_only_observation` | evaluated | 0.0466673 | -0.920926 | -1.93359e-05 | -1.47923e-05 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0454287 | 1.38937 | -1.45939e-05 | -2.01527e-05 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0454287 | 1.38937 | -1.45939e-05 | -2.01527e-05 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0454287 | 1.38937 | -1.45939e-05 | -2.01527e-05 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.540666 | 21052 | 0.046289 | 0.0839894 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `position_only_observation` | evaluated | 0.0454287 | 1.38937 | -1.45939e-05 | -2.01527e-05 |
| `soft_pgd_ofb1p4` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0454287 | 1.38937 | -1.45939e-05 | -2.01527e-05 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.04723 | -2.26391 | 1.2282e-05 | -1.28543e-05 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.04723 | -2.26391 | 1.2282e-05 | -1.28543e-05 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.04723 | -2.26391 | 1.2282e-05 | -1.28543e-05 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.04723 | -2.26391 | 1.2282e-05 | -1.28543e-05 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `position_only_observation` | evaluated | 0.04723 | -2.26391 | 1.2282e-05 | -1.28543e-05 |
| `soft_pgd_ofb1p4` | `sensory_feedback` | `velocity_only_observation` | evaluated | 4.0091 | 224990 | 0.136674 | 0.159618 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 242.325 | 0.00179212 | 0.00687702 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.372312 | 239.904 | 0.00167383 | 0.00669692 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.535628 | 20199.5 | 0.0431573 | 0.0863466 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `position_only_observation` | evaluated | 0.047679 | -1.81545 | 1.27622e-05 | -3.71001e-06 |
| `soft_pgd_ofb1p4` | `delayed_observation` | `velocity_only_observation` | evaluated | 4.06092 | 225034 | 0.135089 | 0.159794 |
| `soft_pgd_ofb1p8` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p8` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `nominal` | `shuffled_observation_history` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `nominal` | `lagged_observation_history` | evaluated | 0.538807 | 20234.6 | 0.0434113 | 0.0862638 |
| `soft_pgd_ofb1p8` | `nominal` | `position_only_observation` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `nominal` | `velocity_only_observation` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p8` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0471876 | -0.932828 | -1.27662e-05 | -9.1241e-06 |
| `soft_pgd_ofb1p8` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0471876 | -0.932828 | -1.27662e-05 | -9.1241e-06 |
| `soft_pgd_ofb1p8` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0471876 | -0.932828 | -1.27662e-05 | -9.1241e-06 |
| `soft_pgd_ofb1p8` | `initial_state` | `lagged_observation_history` | evaluated | 0.536854 | 21203.1 | 0.0482782 | 0.0824017 |
| `soft_pgd_ofb1p8` | `initial_state` | `position_only_observation` | evaluated | 0.0471876 | -0.932828 | -1.27662e-05 | -9.1241e-06 |
| `soft_pgd_ofb1p8` | `initial_state` | `velocity_only_observation` | evaluated | 0.0471876 | -0.932828 | -1.27662e-05 | -9.1241e-06 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0458624 | 1.62845 | -1.20144e-05 | -7.82235e-06 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0458624 | 1.62845 | -1.20144e-05 | -7.82235e-06 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0458624 | 1.62845 | -1.20144e-05 | -7.82235e-06 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.543631 | 21117.4 | 0.0465607 | 0.0841231 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `position_only_observation` | evaluated | 0.0458624 | 1.62845 | -1.20144e-05 | -7.82235e-06 |
| `soft_pgd_ofb1p8` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0458624 | 1.62845 | -1.20144e-05 | -7.82235e-06 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0478186 | -2.31234 | 8.46708e-06 | -1.26009e-05 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0478186 | -2.31234 | 8.46708e-06 | -1.26009e-05 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0478186 | -2.31234 | 8.46708e-06 | -1.26009e-05 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0478186 | -2.31234 | 8.46708e-06 | -1.26009e-05 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0478186 | -2.31234 | 8.46708e-06 | -1.26009e-05 |
| `soft_pgd_ofb1p8` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.95534 | 211879 | 0.138381 | 0.137133 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 242.223 | 0.00188852 | 0.00704733 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.376335 | 238.923 | 0.00174822 | 0.00683125 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.538807 | 20234.6 | 0.0434113 | 0.0862638 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `position_only_observation` | evaluated | 0.0482128 | -1.7979 | 1.35046e-05 | -6.78563e-06 |
| `soft_pgd_ofb1p8` | `delayed_observation` | `velocity_only_observation` | evaluated | 4.00322 | 211927 | 0.136865 | 0.137412 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `soft_pgd_ofb1p05` | available | 0.52971 | 1.00721 | 0.0522146 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `soft_pgd_ofb1p4` | available | 0.659556 | 1.26684 | 0.0522712 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `soft_pgd_ofb1p8` | available | 0.649384 | 1.24652 | 0.0522469 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `soft_pgd_ofb1p05` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `soft_pgd_ofb1p4` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `soft_pgd_ofb1p8` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `soft_pgd_ofb1p05` | 0 | 9500 | 2000 | -7500 | 280.605 | 6 |
| `soft_pgd_ofb1p05` | 1 | 10500 | 4000 | -6500 | 298.294 | 6 |
| `soft_pgd_ofb1p05` | 2 | 11000 | 2500 | -8500 | 276.47 | 6 |
| `soft_pgd_ofb1p05` | 3 | 10500 | 2000 | -8500 | 275.013 | 6 |
| `soft_pgd_ofb1p05` | 4 | 11000 | 4000 | -7000 | 309.574 | 6 |
| `soft_pgd_ofb1p4` | 0 | 9500 | 2000 | -7500 | 280.874 | 6 |
| `soft_pgd_ofb1p4` | 1 | 10500 | 4000 | -6500 | 296.711 | 6 |
| `soft_pgd_ofb1p4` | 2 | 11000 | 2500 | -8500 | 309.622 | 6 |
| `soft_pgd_ofb1p4` | 3 | 10500 | 2000 | -8500 | 282.754 | 6 |
| `soft_pgd_ofb1p4` | 4 | 11000 | 4500 | -6500 | 306.97 | 6 |
| `soft_pgd_ofb1p8` | 0 | 9500 | 3000 | -6500 | 287.338 | 6 |
| `soft_pgd_ofb1p8` | 1 | 10500 | 4000 | -6500 | 291.705 | 6 |
| `soft_pgd_ofb1p8` | 2 | 11000 | 2500 | -8500 | 316.309 | 6 |
| `soft_pgd_ofb1p8` | 3 | 11000 | 2000 | -9000 | 271.267 | 6 |
| `soft_pgd_ofb1p8` | 4 | 11000 | 2500 | -8500 | 307.204 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
