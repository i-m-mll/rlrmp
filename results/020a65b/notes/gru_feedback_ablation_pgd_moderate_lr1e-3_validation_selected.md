# GRU Feedback Ablation Diagnostic

- Issue: `020a65b`
- Source experiment: `020a65b`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 242.797 | 0.00279619 | 0.0048918 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 242.797 | 0.00279619 | 0.0048918 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.323632 | 264.659 | 0.00297831 | 0.00525001 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.47772 | 14752.9 | 0.03819 | 0.0546905 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 9.61005 | 3.68245e+06 | 0.615955 | 0.226362 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 6.73555 | 410445 | 0.172873 | 0.280327 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0904009 | 970.204 | 0.00707773 | -0.00100832 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0904009 | 970.204 | 0.00707773 | -0.00100832 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.32376 | 261.34 | 0.00305054 | 0.0028844 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.4807 | 15475.7 | 0.041211 | 0.0421145 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 9.48748 | 3.37361e+06 | 0.592717 | 0.193845 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 6.72573 | 396284 | 0.167055 | 0.274427 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00976543 | 239.661 | 0.0026523 | 0.00359733 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00976543 | 239.661 | 0.0026523 | 0.00359733 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.323457 | 264.717 | 0.00304131 | 0.00512822 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.477668 | 14814.2 | 0.0384293 | 0.0540652 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 9.62501 | 3.69209e+06 | 0.616902 | 0.226152 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.74053 | 409827 | 0.173075 | 0.278865 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0403041 | -4.9255 | 4.13917e-05 | -0.00011876 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0403041 | -4.9255 | 4.13917e-05 | -0.00011876 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.334871 | 626.541 | 0.00561493 | 0.0166342 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0403041 | -4.9255 | 4.13917e-05 | -0.00011876 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0403041 | -4.9255 | 4.13917e-05 | -0.00011876 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0403041 | -4.9255 | 4.13917e-05 | -0.00011876 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.342355 | 201.248 | 0.0036024 | 0.00464232 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.342355 | 201.248 | 0.0036024 | 0.00464232 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.314425 | 262.096 | 0.00327661 | 0.00496808 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.470968 | 14806.5 | 0.0389119 | 0.0504917 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 9.64344 | 3.75387e+06 | 0.622094 | 0.231548 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.71067 | 411898 | 0.173747 | 0.281095 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | available | 1.50095 | 2.83108 | 0.170827 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | 0 | 8500 | 5500 | -3000 | 125.186 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | 1 | 9500 | 5500 | -4000 | 114.342 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | 2 | 5500 | 3000 | -2500 | 115.556 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | 3 | 5500 | 7000 | 1500 | 131.516 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64` | 4 | 11000 | 4500 | -6500 | 124.282 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
