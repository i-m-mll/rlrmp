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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.11021 | 201.726 | 0.00310128 | 0.00400281 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.11021 | 201.726 | 0.00310128 | 0.00400281 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.508139 | 430.56 | 0.00488599 | 0.0125234 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.507038 | 3597.09 | 0.0155909 | 0.00957114 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.86169 | 917646 | 0.30931 | 0.304486 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.2828 | 166955 | 0.147606 | -0.00462719 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0345843 | -1.22066 | -1.71181e-06 | 3.30177e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0345843 | -1.22066 | -1.71181e-06 | 3.30177e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0346097 | -1.20361 | -1.66509e-06 | 3.20214e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.21977 | 970787 | 0.318937 | 0.258623 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.3252 | 166811 | 0.146646 | -0.00267242 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.102342 | 162.764 | 0.00245676 | 0.00278341 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.102342 | 162.764 | 0.00245676 | 0.00278341 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.54139 | 411.797 | 0.00463957 | 0.0118335 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0341629 | -1.12874 | -4.53048e-06 | 3.24513e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.18245 | 965683 | 0.318891 | 0.257015 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.31247 | 167732 | 0.147895 | -0.00418215 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.102342 | 162.764 | 0.00245676 | 0.00278341 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0341629 | -1.12874 | -4.53048e-06 | 3.24513e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.512827 | 242.164 | 0.00233777 | 0.00479474 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.509836 | 5720.56 | 0.021232 | 0.0241063 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.21574 | 982739 | 0.321778 | 0.257185 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.31247 | 167732 | 0.147895 | -0.00418215 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 208.616 | 0.00138993 | 0.00464318 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 208.616 | 0.00138993 | 0.00464318 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.595866 | 245.366 | 0.00171212 | 0.00574625 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.559597 | 6975.31 | 0.0226951 | 0.0370784 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0359893 | -1.37669 | -1.35456e-06 | 4.11665e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0359893 | -1.37669 | -1.35456e-06 | 4.11665e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0360201 | -1.36692 | -1.21793e-06 | 3.53228e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.97225 | 1.1079e+06 | 0.343141 | 0.158595 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.35936 | 166239 | 0.146343 | -0.002811 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.115819 | 167.435 | 0.00238221 | 0.0031371 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.115819 | 167.435 | 0.00238221 | 0.0031371 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.630817 | 433.621 | 0.00472362 | 0.0118548 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0356885 | -1.30771 | -2.63042e-06 | 2.75667e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.94244 | 1.10268e+06 | 0.343057 | 0.157745 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.35398 | 167099 | 0.147487 | -0.00401394 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.115819 | 167.435 | 0.00238221 | 0.0031371 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0356885 | -1.30771 | -2.63042e-06 | 2.75667e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.596057 | 244.74 | 0.00230044 | 0.00494391 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.562002 | 7354.62 | 0.0247491 | 0.0328676 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.98073 | 1.12301e+06 | 0.346247 | 0.156547 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.35398 | 167099 | 0.147487 | -0.00401394 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 203.945 | 0.00127329 | 0.00412412 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 203.945 | 0.00127329 | 0.00412412 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.89747 | 241.984 | 0.00155576 | 0.00518403 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.80215 | 8531.71 | 0.0254928 | 0.0431462 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0477813 | -0.216067 | -3.79952e-07 | 6.1431e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0477813 | -0.216067 | -3.79952e-07 | 6.1431e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0478357 | -0.201392 | 5.01369e-08 | 6.30218e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 6.47879 | 1.52247e+06 | 0.40118 | 0.06813 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.45458 | 167826 | 0.146786 | -0.00223254 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.103041 | 179.937 | 0.00201984 | 0.00390889 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.103041 | 179.937 | 0.00201984 | 0.00390889 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.911413 | 472.222 | 0.00457928 | 0.0106327 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0473043 | -0.165149 | 2.10858e-06 | 7.24343e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 6.44408 | 1.51647e+06 | 0.400936 | 0.0679199 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.45222 | 168432 | 0.147619 | -0.00242707 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.103041 | 179.937 | 0.00201984 | 0.00390889 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0473043 | -0.165149 | 2.10858e-06 | 7.24343e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.89676 | 241.615 | 0.00196754 | 0.00508076 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.800726 | 8869.22 | 0.0270648 | 0.0416724 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 6.49529 | 1.5469e+06 | 0.404895 | 0.0688924 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.45222 | 168432 | 0.147619 | -0.00242707 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.111031 | 196.115 | 0.00315673 | 0.00389044 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.111031 | 196.115 | 0.00315673 | 0.00389044 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.50644 | 423.339 | 0.00491319 | 0.0125106 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.499857 | 3554.83 | 0.0156519 | 0.00879633 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.84387 | 913018 | 0.308548 | 0.305626 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.28146 | 166682 | 0.147653 | -0.00468613 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0346087 | -1.13626 | -1.29622e-06 | 3.96734e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0346087 | -1.13626 | -1.29622e-06 | 3.96734e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0346346 | -1.11734 | -1.16671e-06 | 3.79295e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.20662 | 966768 | 0.318227 | 0.259139 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.32791 | 166764 | 0.146624 | -0.00259112 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.102312 | 162.692 | 0.00244794 | 0.00276007 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.102312 | 162.692 | 0.00244794 | 0.00276007 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.541671 | 413.012 | 0.00461586 | 0.0117942 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0341775 | -1.04044 | -4.05893e-06 | 3.44452e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.16892 | 961671 | 0.318174 | 0.257554 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.31547 | 167688 | 0.14787 | -0.00408045 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.102312 | 162.692 | 0.00244794 | 0.00276007 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0341775 | -1.04044 | -4.05893e-06 | 3.44452e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.513246 | 241.824 | 0.00232268 | 0.00477596 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.508749 | 5693.62 | 0.021158 | 0.0239616 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.2009 | 978397 | 0.321014 | 0.257731 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.31547 | 167688 | 0.14787 | -0.00408045 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 207.838 | 0.00138708 | 0.00467217 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 207.838 | 0.00138708 | 0.00467217 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.595494 | 244.638 | 0.00170638 | 0.00578824 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.559296 | 7109.74 | 0.0229688 | 0.0379798 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0360759 | -1.36625 | -1.47496e-06 | 4.0596e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0360759 | -1.36625 | -1.47496e-06 | 4.0596e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0361076 | -1.356 | -1.15801e-06 | 3.72281e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.97512 | 1.1219e+06 | 0.345317 | 0.166521 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.36336 | 166235 | 0.146349 | -0.00267383 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.116821 | 166.996 | 0.00236892 | 0.00323183 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.116821 | 166.996 | 0.00236892 | 0.00323183 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.630675 | 430.762 | 0.00470484 | 0.0117226 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0357871 | -1.29582 | -4.77877e-07 | 3.30484e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.94594 | 1.11664e+06 | 0.345219 | 0.165706 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.3589 | 167094 | 0.14748 | -0.00382397 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.116821 | 166.996 | 0.00236892 | 0.00323183 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0357871 | -1.29582 | -4.77877e-07 | 3.30484e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.595658 | 243.934 | 0.00228393 | 0.00502731 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.562069 | 7476.27 | 0.0249819 | 0.0338578 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.98345 | 1.13717e+06 | 0.348419 | 0.164621 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.3589 | 167094 | 0.14748 | -0.00382397 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 207.001 | 0.00130071 | 0.00397069 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 207.001 | 0.00130071 | 0.00397069 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.908419 | 245.328 | 0.00157382 | 0.00503915 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.803279 | 8010.98 | 0.0242397 | 0.0411323 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0466497 | -0.451032 | -3.65435e-06 | 5.92783e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0466497 | -0.451032 | -3.65435e-06 | 5.92783e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0467006 | -0.432438 | -3.57023e-06 | 6.45387e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 6.52718 | 1.4223e+06 | 0.387083 | 0.0786337 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.45404 | 167027 | 0.146245 | -0.00335604 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.103181 | 187.994 | 0.00196242 | 0.00399757 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.103181 | 187.994 | 0.00196242 | 0.00399757 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.922192 | 440.756 | 0.00439547 | 0.00927907 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0462192 | -0.446256 | -4.09493e-06 | 7.48941e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 6.49289 | 1.41657e+06 | 0.386776 | 0.0780842 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.4522 | 167660 | 0.147018 | -0.00336743 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.103181 | 187.994 | 0.00196242 | 0.00399757 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0462192 | -0.446256 | -4.09493e-06 | 7.48941e-05 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.90777 | 245.174 | 0.00193619 | 0.00502391 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.801691 | 8326.09 | 0.0257036 | 0.0398391 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 6.54707 | 1.44682e+06 | 0.390838 | 0.0788972 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.4522 | 167660 | 0.147018 | -0.00336743 |
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
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.100095 | 227.839 | 0.00276874 | 0.00534735 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.100095 | 227.839 | 0.00276874 | 0.00534735 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.373934 | 655.874 | 0.00577168 | 0.0170448 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.559599 | 11665 | 0.0326057 | 0.0420236 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 10.0006 | 3.72579e+06 | 0.620315 | 0.298644 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 5.54007 | 220425 | 0.150994 | 0.169906 |
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
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0451809 | -0.647002 | -2.82852e-06 | 2.24552e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0451809 | -0.647002 | -2.82852e-06 | 2.24552e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0452663 | -0.636206 | -2.75018e-06 | 2.12411e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 11.4134 | 4.52304e+06 | 0.689915 | 0.183591 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.58629 | 248741 | 0.157668 | 0.193025 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0922429 | 188.153 | 0.00205502 | 0.0040255 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0922429 | 188.153 | 0.00205502 | 0.0040255 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.447925 | 667.653 | 0.00578572 | 0.0168966 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0449385 | -0.576609 | -1.67184e-06 | 1.71625e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 11.3921 | 4.5127e+06 | 0.689625 | 0.182526 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 6.58251 | 249234 | 0.158193 | 0.191893 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0922429 | 188.153 | 0.00205502 | 0.0040255 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0449385 | -0.576609 | -1.67184e-06 | 1.71625e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.435669 | 248.823 | 0.00212625 | 0.00545415 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.594021 | 19885.6 | 0.0437103 | 0.0750975 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 11.4636 | 4.59479e+06 | 0.695632 | 0.182114 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.58251 | 249234 | 0.158193 | 0.191893 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 213.576 | 0.00138918 | 0.00493729 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 213.576 | 0.00138918 | 0.00493729 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.524588 | 249.829 | 0.00171132 | 0.00606559 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.682547 | 30210.9 | 0.0535432 | 0.103325 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0516194 | -0.437496 | -4.60338e-06 | 2.3066e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0516194 | -0.437496 | -4.60338e-06 | 2.3066e-05 |
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
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0517296 | -0.421859 | -4.57624e-06 | 2.29634e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 13.9418 | 4.99852e+06 | 0.716236 | 0.714749 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.13804 | 215044 | 0.1465 | 0.348227 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.118738 | 191.204 | 0.0020755 | 0.00426898 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.118738 | 191.204 | 0.0020755 | 0.00426898 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.538889 | 846.465 | 0.0068044 | 0.0202494 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0513982 | -0.329303 | -3.72883e-06 | 2.549e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 13.9208 | 4.98824e+06 | 0.716073 | 0.713348 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 6.09278 | 215001 | 0.147021 | 0.347673 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.118738 | 191.204 | 0.0020755 | 0.00426898 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0513982 | -0.329303 | -3.72883e-06 | 2.549e-05 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.524599 | 249.194 | 0.00209397 | 0.0056539 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.678482 | 30619.5 | 0.0549221 | 0.101128 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 14.0128 | 5.07938e+06 | 0.722661 | 0.705741 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.09278 | 215001 | 0.147021 | 0.347673 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 213.104 | 0.00129843 | 0.00382323 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 213.104 | 0.00129843 | 0.00382323 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.691843 | 251.093 | 0.00159656 | 0.00483228 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.916432 | 94053.1 | 0.0965696 | 0.211072 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0716568 | 0.40383 | -4.37341e-06 | 9.73002e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0716568 | 0.40383 | -4.37341e-06 | 9.73002e-06 |
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
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0718599 | 0.407309 | -4.15989e-06 | 8.70029e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 17.5806 | 5.3264e+06 | 0.68797 | 1.90067 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 5.24945 | 190620 | 0.14516 | 0.206745 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.118453 | 209.711 | 0.00191463 | 0.00406777 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.118453 | 209.711 | 0.00191463 | 0.00406777 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.711268 | 1171.5 | 0.00811044 | 0.0245922 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0714965 | 0.561448 | -5.19764e-06 | 7.35148e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 17.558 | 5.31625e+06 | 0.687835 | 1.8995 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 5.23377 | 190385 | 0.145292 | 0.207372 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.118453 | 209.711 | 0.00191463 | 0.00406777 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0714965 | 0.561448 | -5.19764e-06 | 7.35148e-06 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.689883 | 250.964 | 0.00190759 | 0.0049435 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.915054 | 93843.7 | 0.0974178 | 0.208616 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 17.6893 | 5.40017e+06 | 0.693494 | 1.90446 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.23377 | 190385 | 0.145292 | 0.207372 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | available | 0.679875 | 1.18213 | 0.177625 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | available | 0.728667 | 1.27157 | 0.185759 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | available | 0.83313 | 1.47781 | 0.188453 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | available | 1.02555 | 1.86545 | 0.185643 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | available | 0.678019 | 1.17584 | 0.1802 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | available | 0.725303 | 1.26524 | 0.185371 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | available | 0.833442 | 1.47862 | 0.188266 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | available | 1.02932 | 1.87438 | 0.184249 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | available | 1.61213 | 3.06145 | 0.162807 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | available | 1.81562 | 3.45577 | 0.175479 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | available | 2.17256 | 4.16135 | 0.183774 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | available | 2.67995 | 5.18082 | 0.179084 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 0 | 11500 | 3000 | -8500 | 161.958 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 1 | 10500 | 2500 | -8000 | 149.215 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 2 | 10500 | 9000 | -1500 | 172.94 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 3 | 10500 | 4500 | -6000 | 164.493 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 4 | 11500 | 11000 | -500 | 168.895 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 0 | 10500 | 9500 | -1000 | 171.691 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 1 | 10500 | 2000 | -8500 | 150.114 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 2 | 10000 | 3500 | -6500 | 164.9 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 3 | 12000 | 2000 | -10000 | 153.754 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 1500 | -10500 | 161.099 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 0 | 11000 | 4000 | -7000 | 161.518 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 1 | 10500 | 4000 | -6500 | 164.856 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 2 | 10000 | 11000 | 1000 | 170.654 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 3 | 11500 | 1500 | -10000 | 143.823 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 5500 | -6500 | 165.75 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 0 | 10500 | 9500 | -1000 | 163.001 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 1 | 11500 | 10000 | -1500 | 160.5 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 2 | 11000 | 10500 | -500 | 164.007 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 3 | 12000 | 6500 | -5500 | 161.546 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 4 | 11000 | 8500 | -2500 | 159.962 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 0 | 11500 | 4500 | -7000 | 170.256 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 1 | 10500 | 2500 | -8000 | 135.905 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 2 | 11000 | 1500 | -9500 | 171.513 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 3 | 10500 | 2000 | -8500 | 120.156 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 4 | 10000 | 7500 | -2500 | 163.928 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 0 | 10500 | 8500 | -2000 | 165.986 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 1 | 10500 | 2000 | -8500 | 151.454 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 2 | 10000 | 8000 | -2000 | 165.706 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 3 | 12000 | 7000 | -5000 | 162.311 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 2000 | -10000 | 164.887 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 0 | 11000 | 4500 | -6500 | 166.798 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 1 | 11500 | 2000 | -9500 | 95.6742 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 2 | 10000 | 11000 | 1000 | 170.808 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 3 | 11500 | 1500 | -10000 | 153.751 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 1500 | -10500 | 156.041 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 0 | 8000 | 9500 | 1500 | 162.138 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 1 | 10500 | 10000 | -500 | 160.671 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 2 | 9000 | 9000 | 0 | 161.178 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 3 | 10500 | 1000 | -9500 | 118.103 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 4 | 11000 | 9000 | -2000 | 162.447 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 0 | 11000 | 3000 | -8000 | 129.358 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 1 | 11500 | 3000 | -8500 | 127.204 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 2 | 10500 | 3000 | -7500 | 123.336 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 3 | 5000 | 9000 | 4000 | 158.531 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 4 | 10000 | 7000 | -3000 | 154.069 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 0 | 10500 | 3000 | -7500 | 150.473 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 1 | 11500 | 4000 | -7500 | 137.599 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 2 | 10000 | 2500 | -7500 | 133.218 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 3 | 11500 | 1500 | -10000 | 136.275 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 1500 | -10500 | 153.631 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 0 | 11500 | 2500 | -9000 | 134.227 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 1 | 12000 | 2500 | -9500 | 142.508 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 2 | 11500 | 4500 | -7000 | 160.55 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 3 | 12000 | 2500 | -9500 | 155.08 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 4500 | -7500 | 160.575 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 0 | 11000 | 8500 | -2500 | 160.007 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 1 | 10500 | 10500 | 0 | 157.711 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 2 | 9000 | 3000 | -6000 | 142.424 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 3 | 10500 | 1500 | -9000 | 139.97 | 6 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 4 | 10000 | 1500 | -8500 | 140.581 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
