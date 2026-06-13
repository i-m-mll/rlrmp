# GRU Feedback Ablation Diagnostic

- Issue: `ffff699`
- Source experiment: `ffff699`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 1153.93 | 0.00922145 | 0.00954814 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 1153.93 | 0.00922145 | 0.00954814 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `shuffled_observation_history` | evaluated | 0.667472 | 220300 | 0.153607 | 0.0103517 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `lagged_observation_history` | evaluated | 0.451051 | 90768.9 | 0.0782585 | 0.129943 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `position_only_observation` | evaluated | 14.148 | 1.21652e+07 | 0.1836 | 5.03524 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `nominal` | `velocity_only_observation` | evaluated | 0.639983 | 143730 | 0.131995 | 0.0106799 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0362872 | 1642.87 | 0.0114691 | 0.00865392 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0362872 | 1642.87 | 0.0114691 | 0.00865392 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `shuffled_observation_history` | evaluated | 0.669208 | 220274 | 0.153763 | 0.00993831 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `lagged_observation_history` | evaluated | 0.447809 | 86169.3 | 0.0763252 | 0.124738 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `position_only_observation` | evaluated | 14.0793 | 1.20389e+07 | 0.183701 | 5.01203 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `initial_state` | `velocity_only_observation` | evaluated | 0.64091 | 144056 | 0.132021 | 0.00978572 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00868194 | 1125.88 | 0.00912665 | 0.00904711 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00868194 | 1125.88 | 0.00912665 | 0.00904711 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.667289 | 220300 | 0.153651 | 0.0102949 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.450914 | 90760.4 | 0.0782831 | 0.129857 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `position_only_observation` | evaluated | 14.1481 | 1.21653e+07 | 0.183605 | 5.03512 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.639959 | 143718 | 0.132031 | 0.0110563 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.116168 | 1130.05 | 0.00919597 | 0.00796421 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.116168 | 1130.05 | 0.00919597 | 0.00796421 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.695138 | 221341 | 0.153678 | 0.0204382 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.510096 | 91065.4 | 0.0799005 | 0.133445 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `position_only_observation` | evaluated | 14.1453 | 1.21482e+07 | 0.183696 | 5.03046 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.674444 | 143705 | 0.132148 | 0.0093437 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.416716 | 1126.93 | 0.00914375 | 0.00758728 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.416716 | 1126.93 | 0.00914375 | 0.00758728 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.668151 | 220620 | 0.153907 | 0.00881311 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.468475 | 85805.6 | 0.0783463 | 0.127005 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `position_only_observation` | evaluated | 14.1412 | 1.21343e+07 | 0.184276 | 5.02932 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.676893 | 143811 | 0.132127 | 0.00961962 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | available | 13.8735 | 26.8654 | 0.881614 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | 0 | 3000 | 4000 | 1000 | 14.446 | 8 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | 1 | 3000 | 5000 | 2000 | 12.4734 | 8 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | 2 | 2000 | 4000 | 2000 | 18.6729 | 8 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | 3 | 3000 | 3000 | 0 | 18.1742 | 8 |
| `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | 4 | 3000 | 3000 | 0 | 14.4126 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
