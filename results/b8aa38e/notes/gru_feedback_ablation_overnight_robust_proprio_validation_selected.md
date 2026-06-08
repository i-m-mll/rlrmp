# GRU Feedback Ablation Diagnostic

- Issue: `b8aa38e`
- Source experiment: `b8aa38e`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 221.773 | 0.00181562 | 0.00556398 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 221.773 | 0.00181562 | 0.00556398 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.482511 | 252.101 | 0.0018409 | 0.00592099 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.491635 | 4634.57 | 0.0171868 | 0.0241774 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.90028 | 918079 | 0.30827 | 0.306352 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.30037 | 166995 | 0.146473 | -0.00257657 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.141046 | 891.009 | 0.00464547 | 0.00280349 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.141046 | 891.009 | 0.00464547 | 0.00280349 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.482244 | 246.541 | 0.00252075 | 0.00447498 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.48609 | 5034.05 | 0.0205318 | 0.0192334 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.75455 | 824330 | 0.295635 | 0.267346 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.22304 | 145345 | 0.137748 | -0.00533706 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00686998 | 214.991 | 0.00153299 | 0.00434294 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00686998 | 214.991 | 0.00153299 | 0.00434294 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.482511 | 250.758 | 0.00198698 | 0.00569 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.491193 | 4748.46 | 0.0178355 | 0.0230408 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.90539 | 923129 | 0.309448 | 0.306227 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.30061 | 166000 | 0.146321 | -0.00304902 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0330715 | -2.19112 | 6.20049e-05 | -0.000117424 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.11021 | 201.726 | 0.00310128 | 0.00400281 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.508139 | 430.56 | 0.00488599 | 0.0125234 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0330715 | -2.19112 | 6.20049e-05 | -0.000117424 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0330715 | -2.19112 | 6.20049e-05 | -0.000117424 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0330715 | -2.19112 | 6.20049e-05 | -0.000117424 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.503141 | 200.975 | 0.00309277 | 0.00415389 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.503141 | 200.975 | 0.00309277 | 0.00415389 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.470966 | 234.679 | 0.00246915 | 0.00489938 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.485534 | 4985.82 | 0.0193875 | 0.01901 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.89265 | 933696 | 0.312115 | 0.305158 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.28324 | 167074 | 0.147643 | -0.00453373 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 206.698 | 0.00135731 | 0.00469281 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 206.698 | 0.00135731 | 0.00469281 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.51276 | 242.814 | 0.00168157 | 0.00584392 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.511855 | 5327.59 | 0.0189537 | 0.0288982 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.21358 | 965727 | 0.317791 | 0.258925 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.32497 | 167776 | 0.146796 | -0.00227275 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.144804 | 937.824 | 0.00507331 | 0.00236194 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.144804 | 937.824 | 0.00507331 | 0.00236194 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.51276 | 238.123 | 0.00240171 | 0.00449196 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.505555 | 5701.59 | 0.022099 | 0.0244053 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.04358 | 865373 | 0.304155 | 0.223099 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.24101 | 146086 | 0.138021 | -0.00460362 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00683448 | 211.031 | 0.00115027 | 0.00394432 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00683448 | 211.031 | 0.00115027 | 0.00394432 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.512755 | 243.183 | 0.00184472 | 0.00563758 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.511419 | 5441.48 | 0.0195725 | 0.0278391 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.21977 | 970787 | 0.318937 | 0.258623 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.3252 | 166811 | 0.146646 | -0.00267242 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0352464 | 5.43879 | -6.65833e-05 | 0.000125954 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.118405 | 208.043 | 0.00264878 | 0.00366489 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.540986 | 420.924 | 0.00461861 | 0.0116297 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0352464 | 5.43879 | -6.65833e-05 | 0.000125954 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0352464 | 5.43879 | -6.65833e-05 | 0.000125954 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0352464 | 5.43879 | -6.65833e-05 | 0.000125954 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.514337 | 250.795 | 0.00161401 | 0.00559167 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.514337 | 250.795 | 0.00161401 | 0.00559167 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.521267 | 278.007 | 0.0017246 | 0.00581318 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.513293 | 5538.38 | 0.0194014 | 0.0293786 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.21402 | 966525 | 0.317956 | 0.259519 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.32443 | 167613 | 0.146791 | -0.00221606 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 208.616 | 0.00138993 | 0.00464318 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 208.616 | 0.00138993 | 0.00464318 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.595866 | 245.366 | 0.00171212 | 0.00574625 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.559597 | 6975.31 | 0.0226951 | 0.0370784 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.96396 | 1.10272e+06 | 0.342065 | 0.159251 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.35917 | 167141 | 0.146494 | -0.00250786 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.153163 | 953.076 | 0.00495721 | 0.00301409 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.153163 | 953.076 | 0.00495721 | 0.00301409 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.596072 | 239.403 | 0.00234048 | 0.00476605 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.551741 | 7355.04 | 0.0255555 | 0.0338781 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.73439 | 983122 | 0.325908 | 0.131255 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.2594 | 145504 | 0.137591 | -0.00413695 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00722228 | 212.686 | 0.00115482 | 0.00407488 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00722228 | 212.686 | 0.00115482 | 0.00407488 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.595835 | 245.71 | 0.00185557 | 0.00559481 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.559187 | 7088.52 | 0.0232525 | 0.0361685 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.97225 | 1.1079e+06 | 0.343141 | 0.158595 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.35936 | 166239 | 0.146343 | -0.002811 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0367853 | 4.61088 | -6.27978e-05 | 8.20946e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.133124 | 214.632 | 0.0025797 | 0.00410194 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.630457 | 439.57 | 0.00468486 | 0.0117177 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0367853 | 4.61088 | -6.27978e-05 | 8.20946e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0367853 | 4.61088 | -6.27978e-05 | 8.20946e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0367853 | 4.61088 | -6.27978e-05 | 8.20946e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.597496 | 254.348 | 0.00165941 | 0.00557599 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.597496 | 254.348 | 0.00165941 | 0.00557599 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.604952 | 279.47 | 0.00176378 | 0.00574219 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.561088 | 7234.56 | 0.0231856 | 0.0374065 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.96437 | 1.10367e+06 | 0.342266 | 0.15961 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.3594 | 166981 | 0.146498 | -0.00245762 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 203.945 | 0.00127329 | 0.00412412 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 203.945 | 0.00127329 | 0.00412412 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.89747 | 241.984 | 0.00155576 | 0.00518403 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.80215 | 8531.71 | 0.0254928 | 0.0431462 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 6.46692 | 1.51649e+06 | 0.400189 | 0.0681352 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.45474 | 168456 | 0.146873 | -0.00221184 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.206581 | 935.76 | 0.00402588 | 0.00395894 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.206581 | 935.76 | 0.00402588 | 0.00395894 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.902606 | 233.141 | 0.00190169 | 0.00489286 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.794666 | 9731.98 | 0.0291122 | 0.0471883 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 6.1259 | 1.33288e+06 | 0.377188 | 0.0638619 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.3134 | 146768 | 0.137599 | -0.00237702 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0108555 | 203.667 | 0.000927376 | 0.00426316 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0108555 | 203.667 | 0.000927376 | 0.00426316 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.8973 | 242.294 | 0.00165337 | 0.00517577 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.80163 | 8623.4 | 0.0259082 | 0.0427861 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 6.47879 | 1.52247e+06 | 0.40118 | 0.06813 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.45458 | 167826 | 0.146786 | -0.00223254 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.048195 | 3.13732 | -5.4155e-05 | -1.00806e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.123554 | 226.045 | 0.00219971 | 0.00485951 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.911463 | 467.234 | 0.00444418 | 0.0105035 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.048195 | 3.13732 | -5.4155e-05 | -1.00806e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.048195 | 3.13732 | -5.4155e-05 | -1.00806e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.048195 | 3.13732 | -5.4155e-05 | -1.00806e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.895925 | 251.465 | 0.00154611 | 0.0051079 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.895925 | 251.465 | 0.00154611 | 0.0051079 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.898601 | 274.687 | 0.00161186 | 0.00544492 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.802852 | 8993.77 | 0.0263099 | 0.0437811 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 6.46838 | 1.51909e+06 | 0.400596 | 0.0666511 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.455 | 168287 | 0.146894 | -0.00238752 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 221.219 | 0.00186817 | 0.00545405 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 221.219 | 0.00186817 | 0.00545405 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.480187 | 251.758 | 0.00190674 | 0.00580314 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.484364 | 4586.62 | 0.0172481 | 0.0231767 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.88205 | 913424 | 0.307504 | 0.307502 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.29847 | 166726 | 0.14652 | -0.00262013 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.139404 | 909.67 | 0.00494495 | 0.00262113 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.139404 | 909.67 | 0.00494495 | 0.00262113 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.479884 | 246.12 | 0.00263463 | 0.00434584 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.478981 | 4998.77 | 0.0206672 | 0.0181724 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.74034 | 821562 | 0.295226 | 0.268655 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.22354 | 145088 | 0.13782 | -0.00545305 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00683968 | 215.181 | 0.00160449 | 0.00416871 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00683968 | 215.181 | 0.00160449 | 0.00416871 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.480188 | 250.412 | 0.00205918 | 0.00557317 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.483922 | 4700.54 | 0.0179016 | 0.0220549 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.88725 | 918457 | 0.308685 | 0.307357 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.29871 | 165728 | 0.146369 | -0.00310027 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0330882 | -2.39558 | 6.31418e-05 | -0.000113378 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.111031 | 196.115 | 0.00315673 | 0.00389044 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.50644 | 423.339 | 0.00491319 | 0.0125106 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0330882 | -2.39558 | 6.31418e-05 | -0.000113378 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0330882 | -2.39558 | 6.31418e-05 | -0.000113378 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0330882 | -2.39558 | 6.31418e-05 | -0.000113378 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.501081 | 195.647 | 0.00314905 | 0.00405966 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.501081 | 195.647 | 0.00314905 | 0.00405966 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.467597 | 234.508 | 0.00254578 | 0.00482142 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.478072 | 4921.15 | 0.019421 | 0.0180233 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.87352 | 928683 | 0.311294 | 0.306188 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.28167 | 166811 | 0.147695 | -0.00456731 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 206.4 | 0.00135315 | 0.00465918 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 206.4 | 0.00135315 | 0.00465918 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.513203 | 242.503 | 0.00167384 | 0.00581578 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.510975 | 5301.21 | 0.0188862 | 0.0286718 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.20041 | 961715 | 0.317079 | 0.259453 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.32768 | 167732 | 0.146775 | -0.00218133 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.145174 | 936.236 | 0.00504932 | 0.00233489 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.145174 | 936.236 | 0.00504932 | 0.00233489 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.513203 | 237.953 | 0.00239907 | 0.00446462 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.504751 | 5699.01 | 0.0221007 | 0.0243381 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.03084 | 861927 | 0.303519 | 0.223735 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.24339 | 146044 | 0.138017 | -0.00450562 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00681588 | 209.59 | 0.00114645 | 0.00386407 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00681588 | 209.59 | 0.00114645 | 0.00386407 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.513189 | 242.884 | 0.00183562 | 0.00560709 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.510534 | 5415.94 | 0.0195082 | 0.0276236 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.20662 | 966768 | 0.318227 | 0.259139 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.32791 | 166764 | 0.146624 | -0.00259112 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0352973 | 5.48286 | -6.50177e-05 | 0.000124276 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.118285 | 207.316 | 0.00263394 | 0.00364637 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.541228 | 421.74 | 0.0045935 | 0.0115841 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0352973 | 5.48286 | -6.50177e-05 | 0.000124276 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0352973 | 5.48286 | -6.50177e-05 | 0.000124276 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0352973 | 5.48286 | -6.50177e-05 | 0.000124276 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.514493 | 249.489 | 0.00160554 | 0.00551937 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.514493 | 249.489 | 0.00160554 | 0.00551937 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.521472 | 277.347 | 0.00172524 | 0.00573696 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.512321 | 5507.3 | 0.0193297 | 0.0290953 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.20094 | 962538 | 0.317251 | 0.259981 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.3275 | 167567 | 0.146771 | -0.00217454 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 207.838 | 0.00138708 | 0.00467217 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 207.838 | 0.00138708 | 0.00467217 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.595494 | 244.638 | 0.00170638 | 0.00578824 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.559296 | 7109.74 | 0.0229688 | 0.0379798 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.96682 | 1.11668e+06 | 0.344237 | 0.167147 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.36315 | 167135 | 0.146498 | -0.00238363 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.153121 | 951.608 | 0.0049571 | 0.00311247 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.153121 | 951.608 | 0.0049571 | 0.00311247 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.595707 | 238.837 | 0.00234472 | 0.00484928 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.551419 | 7483.06 | 0.0258318 | 0.0348624 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.73654 | 995414 | 0.327973 | 0.138762 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.26404 | 145498 | 0.137612 | -0.00394333 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00726899 | 211.379 | 0.00114685 | 0.00413451 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00726899 | 211.379 | 0.00114685 | 0.00413451 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.595464 | 244.976 | 0.00185117 | 0.00564163 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.558891 | 7223.78 | 0.0235276 | 0.037085 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.97512 | 1.1219e+06 | 0.345317 | 0.166521 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.36336 | 166235 | 0.146349 | -0.00267383 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0368533 | 4.62444 | -6.0529e-05 | 7.79891e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.134076 | 213.794 | 0.00256132 | 0.00418983 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.630369 | 435.923 | 0.00465727 | 0.0115566 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0368533 | 4.62444 | -6.0529e-05 | 7.79891e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0368533 | 4.62444 | -6.0529e-05 | 7.79891e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0368533 | 4.62444 | -6.0529e-05 | 7.79891e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.597137 | 253.046 | 0.00165466 | 0.00557692 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.597137 | 253.046 | 0.00165466 | 0.00557692 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.60487 | 278.858 | 0.00176251 | 0.00574827 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.560797 | 7370.01 | 0.0234676 | 0.0382651 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.96737 | 1.11779e+06 | 0.344457 | 0.167456 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.36313 | 166971 | 0.146501 | -0.0023725 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 207.001 | 0.00130071 | 0.00397069 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 207.001 | 0.00130071 | 0.00397069 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.908419 | 245.328 | 0.00157382 | 0.00503915 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.803279 | 8010.98 | 0.0242397 | 0.0411323 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 6.51537 | 1.41659e+06 | 0.386114 | 0.0780573 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.45401 | 167679 | 0.146356 | -0.00339432 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.202672 | 930.573 | 0.00412117 | 0.00391687 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.202672 | 930.573 | 0.00412117 | 0.00391687 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.913587 | 236.316 | 0.00190675 | 0.00479648 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.794979 | 9267.04 | 0.028093 | 0.0456214 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 6.16527 | 1.24144e+06 | 0.363201 | 0.0924615 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.31015 | 146045 | 0.137083 | -0.00344813 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0104977 | 215.151 | 0.000973124 | 0.00445039 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0104977 | 215.151 | 0.000973124 | 0.00445039 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.908208 | 245.643 | 0.00166452 | 0.00505222 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.802798 | 8097.89 | 0.0246388 | 0.0407976 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 6.52718 | 1.4223e+06 | 0.387083 | 0.0786337 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.45404 | 167027 | 0.146245 | -0.00335604 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0472297 | 2.68338 | -4.31638e-05 | -4.6172e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.123133 | 237.209 | 0.00216374 | 0.00495054 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.922331 | 442.312 | 0.00432162 | 0.00946101 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0472297 | 2.68338 | -4.31638e-05 | -4.6172e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0472297 | 2.68338 | -4.31638e-05 | -4.6172e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0472297 | 2.68338 | -4.31638e-05 | -4.6172e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.909733 | 255.623 | 0.00159071 | 0.00510401 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.909733 | 255.623 | 0.00159071 | 0.00510401 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.914246 | 280.575 | 0.00169074 | 0.00545755 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.804344 | 8462.41 | 0.0250693 | 0.0419526 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 6.51569 | 1.41876e+06 | 0.386484 | 0.076663 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.45519 | 167487 | 0.14637 | -0.00327867 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 233.349 | 0.00188627 | 0.0056821 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 233.349 | 0.00188627 | 0.0056821 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.362716 | 260.43 | 0.00192555 | 0.00612104 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.537735 | 14913.1 | 0.0363423 | 0.0625972 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 10.0188 | 3.72503e+06 | 0.619634 | 0.299488 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 5.55887 | 220512 | 0.15033 | 0.171326 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.100021 | 828.286 | 0.00580657 | 0.00141207 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.100021 | 828.286 | 0.00580657 | 0.00141207 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.363478 | 255.393 | 0.00330787 | 0.00387838 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.53967 | 15990.2 | 0.041549 | 0.0533576 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 9.83698 | 3.40805e+06 | 0.597159 | 0.298349 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 5.55 | 198191 | 0.142554 | 0.167056 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00930346 | 233.015 | 0.00146291 | 0.00506356 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00930346 | 233.015 | 0.00146291 | 0.00506356 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.362599 | 260.479 | 0.00198714 | 0.00605032 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.53772 | 14979.1 | 0.0366078 | 0.0621421 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 10.0355 | 3.73471e+06 | 0.620589 | 0.299588 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 5.56306 | 219900 | 0.15035 | 0.171262 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0415389 | -1.85692 | 3.84665e-05 | -9.2058e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.100095 | 227.839 | 0.00276874 | 0.00534735 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.373934 | 655.874 | 0.00577168 | 0.0170448 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0415389 | -1.85692 | 3.84665e-05 | -9.2058e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0415389 | -1.85692 | 3.84665e-05 | -9.2058e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0415389 | -1.85692 | 3.84665e-05 | -9.2058e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.378331 | 227.435 | 0.00277537 | 0.00550232 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.378331 | 227.435 | 0.00277537 | 0.00550232 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.352515 | 245.931 | 0.00234155 | 0.00575094 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.529643 | 15041 | 0.0374587 | 0.0586735 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 10.0603 | 3.79405e+06 | 0.625855 | 0.299732 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.57174 | 222341 | 0.151412 | 0.172574 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 213.619 | 0.00142453 | 0.00491714 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 213.619 | 0.00142453 | 0.00491714 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.435106 | 249.604 | 0.00176318 | 0.00598359 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.59851 | 19577.4 | 0.0424258 | 0.0776998 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 11.396 | 4.51273e+06 | 0.688994 | 0.183417 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 6.58395 | 249259 | 0.157562 | 0.192784 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.123428 | 884.957 | 0.00580297 | 0.00184219 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.123428 | 884.957 | 0.00580297 | 0.00184219 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.437244 | 248.125 | 0.00307448 | 0.0042233 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.600053 | 20721.2 | 0.0472055 | 0.0714994 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 11.1533 | 4.11559e+06 | 0.662226 | 0.219881 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 6.54949 | 225736 | 0.149584 | 0.18971 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00981433 | 218.977 | 0.0010468 | 0.00462894 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00981433 | 218.977 | 0.0010468 | 0.00462894 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.43484 | 250.035 | 0.00182914 | 0.00593079 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.598582 | 19653.9 | 0.0426883 | 0.0773542 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 11.4134 | 4.52304e+06 | 0.689915 | 0.183591 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.58629 | 248741 | 0.157668 | 0.193025 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0457146 | 1.53035 | -3.07069e-05 | 2.93574e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.114367 | 231.814 | 0.00222667 | 0.00500317 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.448498 | 653.293 | 0.00566617 | 0.0165371 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0457146 | 1.53035 | -3.07069e-05 | 2.93574e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0457146 | 1.53035 | -3.07069e-05 | 2.93574e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0457146 | 1.53035 | -3.07069e-05 | 2.93574e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.440822 | 256.041 | 0.00169875 | 0.00587045 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.440822 | 256.041 | 0.00169875 | 0.00587045 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.447022 | 280.09 | 0.00176506 | 0.00604065 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.600919 | 20188.8 | 0.0431935 | 0.0786354 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 11.3948 | 4.5111e+06 | 0.688981 | 0.182424 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.56036 | 248448 | 0.157446 | 0.191688 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 213.576 | 0.00138918 | 0.00493729 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 213.576 | 0.00138918 | 0.00493729 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.524588 | 249.829 | 0.00171132 | 0.00606559 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.682547 | 30210.9 | 0.0535432 | 0.103325 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 13.9233 | 4.98826e+06 | 0.715387 | 0.714016 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 6.12751 | 215023 | 0.146335 | 0.348342 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.165738 | 921.105 | 0.00507382 | 0.00329603 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.165738 | 921.105 | 0.00507382 | 0.00329603 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.527733 | 246.421 | 0.00260914 | 0.00502602 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.681097 | 31840.6 | 0.0577056 | 0.101822 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 13.5915 | 4.55254e+06 | 0.68401 | 0.788055 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 6.1295 | 193786 | 0.137824 | 0.3467 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0109702 | 217.63 | 0.000975694 | 0.00478103 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0109702 | 217.63 | 0.000975694 | 0.00478103 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.524312 | 250.305 | 0.00176544 | 0.00603058 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.682824 | 30288.2 | 0.0537675 | 0.103035 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 13.9418 | 4.99852e+06 | 0.716236 | 0.714749 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.13804 | 215044 | 0.1465 | 0.348227 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0522742 | 0.940413 | -2.39846e-05 | 9.0787e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.143981 | 233.363 | 0.00223504 | 0.00528452 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.539419 | 826.846 | 0.00665868 | 0.0199219 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0522742 | 0.940413 | -2.39846e-05 | 9.0787e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0522742 | 0.940413 | -2.39846e-05 | 9.0787e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0522742 | 0.940413 | -2.39846e-05 | 9.0787e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.527872 | 254.637 | 0.00164129 | 0.00596214 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.527872 | 254.637 | 0.00164129 | 0.00596214 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.535818 | 278.794 | 0.00172011 | 0.00613017 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.685306 | 31058.9 | 0.0543614 | 0.104362 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 13.9152 | 4.98696e+06 | 0.715534 | 0.710762 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.1121 | 214786 | 0.146285 | 0.348022 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 213.104 | 0.00129843 | 0.00382323 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 213.104 | 0.00129843 | 0.00382323 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.691843 | 251.093 | 0.00159656 | 0.00483228 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.916432 | 94053.1 | 0.0965696 | 0.211072 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 17.5611 | 5.31625e+06 | 0.687219 | 1.89926 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 5.24428 | 190389 | 0.144676 | 0.207127 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.248782 | 889.278 | 0.00366397 | 0.00396438 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.248782 | 889.278 | 0.00366397 | 0.00396438 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.696974 | 242.159 | 0.00189132 | 0.0047197 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.913726 | 103055 | 0.1024 | 0.223285 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 17.0296 | 4.83838e+06 | 0.647975 | 1.94246 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 5.13124 | 168661 | 0.135346 | 0.207269 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0123094 | 222.202 | 0.000816151 | 0.00429437 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0123094 | 222.202 | 0.000816151 | 0.00429437 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.691991 | 251.593 | 0.00163012 | 0.00485032 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.917268 | 94149.6 | 0.0967359 | 0.211119 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 17.5806 | 5.3264e+06 | 0.68797 | 1.90067 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 5.24945 | 190620 | 0.14516 | 0.206745 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0719601 | -0.195699 | -1.68833e-05 | -1.65701e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.153004 | 253.893 | 0.00204787 | 0.00539617 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.71351 | 1132.13 | 0.00788155 | 0.0241449 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0719601 | -0.195699 | -1.68833e-05 | -1.65701e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0719601 | -0.195699 | -1.68833e-05 | -1.65701e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0719601 | -0.195699 | -1.68833e-05 | -1.65701e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.689583 | 257.401 | 0.00151749 | 0.00556606 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.689583 | 257.401 | 0.00151749 | 0.00556606 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.706758 | 282.346 | 0.0015953 | 0.00553606 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.920665 | 95877.4 | 0.0974778 | 0.21276 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 17.5559 | 5.3194e+06 | 0.687664 | 1.89647 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.23926 | 190418 | 0.14455 | 0.208189 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | available | 0.679875 | 1.18213 | 0.177625 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | available | 0.726387 | 1.26702 | 0.185759 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | available | 0.829942 | 1.47143 | 0.188453 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | available | 1.02072 | 1.8558 | 0.185643 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | available | 0.678019 | 1.17584 | 0.1802 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | available | 0.723336 | 1.2613 | 0.185371 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | available | 0.830475 | 1.47268 | 0.188266 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | available | 1.02414 | 1.86403 | 0.184249 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | available | 1.61213 | 3.06145 | 0.162807 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | available | 1.80457 | 3.43366 | 0.175479 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | available | 2.16767 | 4.15157 | 0.183774 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | available | 2.67995 | 5.18082 | 0.179084 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 0 | 11500 | 500 | -11000 | -884.642 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -1432.13 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 2 | 10500 | 500 | -10000 | -1129.64 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 3 | 10500 | 500 | -10000 | -451.177 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 4 | 11500 | 500 | -11000 | -595.222 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 0 | 10500 | 500 | -10000 | -848.571 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -1068.18 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 2 | 10000 | 500 | -9500 | -1119.06 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 3 | 12000 | 500 | -11500 | -522.048 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 500 | -11500 | -557.943 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 0 | 11000 | 500 | -10500 | -853.364 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -817.365 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 2 | 10000 | 500 | -9500 | -977.108 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 3 | 11500 | 500 | -11000 | -530.07 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 500 | -11500 | -685.145 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 0 | 10500 | 500 | -10000 | -923.564 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 1 | 11500 | 500 | -11000 | -696.716 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 2 | 11000 | 500 | -10500 | -1001.6 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 3 | 12000 | 500 | -11500 | -816.832 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 4 | 11000 | 500 | -10500 | -876.933 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 0 | 11500 | 500 | -11000 | -824.609 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -1082.87 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 2 | 11000 | 500 | -10500 | -1026.9 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 3 | 10500 | 500 | -10000 | -1338.43 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 4 | 10000 | 500 | -9500 | -651.82 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 0 | 10500 | 500 | -10000 | -709.647 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -1092.3 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 2 | 10000 | 500 | -9500 | -1321.57 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 3 | 12000 | 500 | -11500 | -647.168 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 500 | -11500 | -694.95 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 0 | 11000 | 500 | -10500 | -758.435 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 1 | 11500 | 500 | -11000 | -812.83 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 2 | 10000 | 500 | -9500 | -946.582 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 3 | 11500 | 500 | -11000 | -603.328 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 1000 | -11000 | -447.657 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 0 | 8000 | 500 | -7500 | -879.26 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -1153.71 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 2 | 9000 | 500 | -8500 | -966.645 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 3 | 10500 | 500 | -10000 | -919.537 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 4 | 11000 | 500 | -10500 | -584.989 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 0 | 11000 | 500 | -10500 | -674.427 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 1 | 11500 | 500 | -11000 | -335.914 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 2 | 10500 | 500 | -10000 | -445.25 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 3 | 5000 | 500 | -4500 | -538.275 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 4 | 10000 | 500 | -9500 | -820.542 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 0 | 10500 | 500 | -10000 | -435.766 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 1 | 11500 | 500 | -11000 | -561.224 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 2 | 10000 | 500 | -9500 | -767.212 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 3 | 11500 | 500 | -11000 | -630.478 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 500 | -11500 | -856.377 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 0 | 11500 | 500 | -11000 | -515.803 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 1 | 12000 | 500 | -11500 | -563.444 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 2 | 11500 | 1000 | -10500 | -283.027 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 3 | 12000 | 500 | -11500 | -989.396 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 500 | -11500 | -1099.1 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 0 | 11000 | 500 | -10500 | -568.836 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -1217.62 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 2 | 9000 | 3500 | -5500 | -167.368 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 3 | 10500 | 500 | -10000 | -918.826 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 4 | 10000 | 500 | -9500 | -736.773 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
