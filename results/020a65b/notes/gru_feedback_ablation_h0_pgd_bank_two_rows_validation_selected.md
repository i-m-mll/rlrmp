# GRU Feedback Ablation Diagnostic

- Issue: `020a65b`
- Source experiment: `020a65b`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0481681 | -0.312513 | -4.00636e-06 | 1.49573e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 220.866 | 0.00162229 | 0.00505491 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0481681 | -0.312513 | -4.00636e-06 | 1.49573e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0481681 | -0.312513 | -4.00636e-06 | 1.49573e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0481681 | -0.312513 | -4.00636e-06 | 1.49573e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0481681 | -0.312513 | -4.00636e-06 | 1.49573e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0479517 | 0.0811629 | 2.59848e-06 | 5.8567e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0479517 | 0.0811629 | 2.59848e-06 | 5.8567e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0479517 | 0.0811629 | 2.59848e-06 | 5.8567e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0479517 | 0.0811629 | 2.59848e-06 | 5.8567e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0479517 | 0.0811629 | 2.59848e-06 | 5.8567e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0479517 | 0.0811629 | 2.59848e-06 | 5.8567e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0481771 | -0.0487832 | -3.16961e-05 | 4.80085e-07 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0481771 | -0.0487832 | -3.16961e-05 | 4.80085e-07 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0481771 | -0.0487832 | -3.16961e-05 | 4.80085e-07 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0481771 | -0.0487832 | -3.16961e-05 | 4.80085e-07 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0481771 | -0.0487832 | -3.16961e-05 | 4.80085e-07 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0481771 | -0.0487832 | -3.16961e-05 | 4.80085e-07 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0496016 | 212.784 | 0.00211324 | 0.00482626 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0479275 | -0.297212 | -3.52079e-06 | 9.30137e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0817284 | -1.37244 | 1.92227e-06 | 8.13551e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 273.246 | 0.00320867 | 0.00542835 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0817284 | -1.37244 | 1.92227e-06 | 8.13551e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0817284 | -1.37244 | 1.92227e-06 | 8.13551e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0817284 | -1.37244 | 1.92227e-06 | 8.13551e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0817284 | -1.37244 | 1.92227e-06 | 8.13551e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0805407 | -1.24223 | 1.31623e-06 | 8.20133e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0805407 | -1.24223 | 1.31623e-06 | 8.20133e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0805407 | -1.24223 | 1.31623e-06 | 8.20133e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0805407 | -1.24223 | 1.31623e-06 | 8.20133e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0805407 | -1.24223 | 1.31623e-06 | 8.20133e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0805407 | -1.24223 | 1.31623e-06 | 8.20133e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0820183 | 2.49868 | -2.10106e-07 | -2.15845e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0820183 | 2.49868 | -2.10106e-07 | -2.15845e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0820183 | 2.49868 | -2.10106e-07 | -2.15845e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0820183 | 2.49868 | -2.10106e-07 | -2.15845e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0820183 | 2.49868 | -2.10106e-07 | -2.15845e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0820183 | 2.49868 | -2.10106e-07 | -2.15845e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.54897 | 19.1415 | 0.00318425 | 0.00530233 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0806116 | -0.645895 | 4.38877e-06 | 8.79161e-06 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | available | 0.0307886 | 0.0154649 | 0.0461123 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | available | 0.0706241 | 0.137441 | 0.00380728 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 0 | 11000 | 1500 | -9500 | 119.613 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 1 | 10500 | 2500 | -8000 | 115.015 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 2 | 11000 | 3000 | -8000 | 104.6 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 3 | 7500 | 2500 | -5000 | 118.215 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 4 | 7000 | 3000 | -4000 | 112.498 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 0 | 11000 | 1000 | -10000 | 126.608 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 1 | 6500 | 1500 | -5000 | 117.999 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 2 | 6500 | 1000 | -5500 | 136.556 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 3 | 10500 | 1500 | -9000 | 144.386 | 8 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 4 | 5500 | 1500 | -4000 | 117.516 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
