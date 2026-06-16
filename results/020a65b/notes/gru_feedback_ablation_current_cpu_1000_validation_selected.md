# GRU Feedback Ablation Diagnostic

- Issue: `020a65b`
- Source experiment: `020a65b`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 104.211 | 6.15632e-05 | 0.00215677 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 104.211 | 6.15632e-05 | 0.00215677 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `shuffled_observation_history` | evaluated | 0.112442 | 127.595 | 8.03272e-05 | 0.00231964 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `lagged_observation_history` | evaluated | 0.369306 | 6751.53 | 0.0138169 | 0.036296 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `position_only_observation` | evaluated | 5.07109 | 3.4455e+06 | 0.483333 | 1.45464 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `nominal` | `velocity_only_observation` | evaluated | 5.57441 | 539220 | 0.166362 | 0.70972 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0309697 | 280.487 | 0.000836878 | -0.00192369 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0309697 | 280.487 | 0.000836878 | -0.00192369 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `shuffled_observation_history` | evaluated | 0.11242 | 142.606 | 0.000968219 | 0.00175167 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `lagged_observation_history` | evaluated | 0.369252 | 8868.05 | 0.0245508 | 0.0237795 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `position_only_observation` | evaluated | 0.0234076 | 13.3186 | 3.66927e-05 | 7.40476e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `initial_state` | `velocity_only_observation` | evaluated | 0.0234076 | 13.3186 | 3.66927e-05 | 7.40476e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.11696 | 113.723 | -0.000412629 | 0.00219814 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0245731 | 1.98124 | -1.53291e-05 | -3.14375e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.115772 | 141.95 | 0.000303775 | 0.00299677 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0245731 | 1.98124 | -1.53291e-05 | -3.14375e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `position_only_observation` | evaluated | 5.07141 | 3.44166e+06 | 0.483175 | 1.45092 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `process_epsilon` | `velocity_only_observation` | evaluated | 5.56884 | 537923 | 0.166589 | 0.707719 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.00711705 | 123.45 | 0.000422618 | 0.00226274 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0234534 | 1.10056 | -7.74466e-05 | 3.83639e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0234534 | 1.10056 | -7.74466e-05 | 3.83639e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.371642 | 6590.41 | 0.0138805 | 0.0340788 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `position_only_observation` | evaluated | 5.06713 | 3.44552e+06 | 0.483695 | 1.45474 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `sensory_feedback` | `velocity_only_observation` | evaluated | 5.57201 | 539239 | 0.166723 | 0.709826 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.00711705 | 123.45 | 0.000422618 | 0.00226274 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00711705 | 123.45 | 0.000422618 | 0.00226274 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.11246 | 128.859 | 0.00011041 | 0.00235129 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0234534 | 1.10056 | -7.74466e-05 | 3.83639e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `position_only_observation` | evaluated | 5.0906 | 3.51554e+06 | 0.487998 | 1.47334 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.57201 | 539239 | 0.166723 | 0.709826 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | available | 0.908509 | 1.76974 | 0.047278 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | 0 | 1000 | 1000 | 0 | 283.188 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | 1 | 1000 | 1000 | 0 | 291.711 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | 2 | 1000 | 500 | -500 | 278.58 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | 3 | 1000 | 1000 | 0 | 370.776 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000` | 4 | 1000 | 1000 | 0 | 302.391 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
