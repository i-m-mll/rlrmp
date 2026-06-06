# GRU Feedback Ablation Diagnostic

- Issue: `b35595c`
- Source experiment: `b35595c`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `fixed_bank_rescored_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 196.344 | 0.00165403 | 0.00532234 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 196.344 | 0.00165403 | 0.00532234 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.466474 | 296.708 | 0.00217757 | 0.00634513 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.470363 | 3743.8 | 0.0143553 | 0.0199702 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.92471 | 900433 | 0.304115 | 0.294387 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.30905 | 168456 | 0.14674 | -0.00183167 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.104783 | 424.908 | 0.00114434 | 0.00311684 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.104783 | 424.908 | 0.00114434 | 0.00311684 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.466118 | 293.188 | 0.00255393 | 0.0054207 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.466823 | 3980.37 | 0.0167433 | 0.0160583 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.82448 | 830929 | 0.294828 | 0.263736 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.25735 | 152008 | 0.140178 | -0.00403717 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0460861 | 393.057 | 0.00206828 | 0.00487473 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0460861 | 393.057 | 0.00206828 | 0.00487473 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.466548 | 268.961 | 0.00299812 | 0.00496353 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.467344 | 4411.9 | 0.0184274 | 0.0113943 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.95934 | 933252 | 0.311725 | 0.292497 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.31028 | 161999 | 0.145709 | -0.00597701 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0363389 | -2.85102 | 0.000152508 | -6.10229e-05 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0933566 | 316.026 | 0.00365294 | 0.00712766 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.482493 | 410.433 | 0.00421088 | 0.0110169 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.47917 | 3383.59 | 0.0140795 | 0.00981402 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.8857 | 900563 | 0.30507 | 0.291866 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0363389 | -2.85102 | 0.000152508 | -6.10229e-05 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.461692 | 322.69 | 0.00355811 | 0.00712574 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.461692 | 322.69 | 0.00355811 | 0.00712574 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.428858 | 251.939 | 0.00252771 | 0.00623045 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.453776 | 4669.83 | 0.0176074 | 0.0193481 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.91707 | 911075 | 0.307061 | 0.290416 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.29043 | 168604 | 0.147579 | -0.00368862 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 185.481 | 0.00157069 | 0.00590913 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 185.481 | 0.00157069 | 0.00590913 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.540661 | 233.485 | 0.00190316 | 0.00590187 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.622596 | 6317.69 | 0.0211996 | 0.0370912 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.40778 | 1.52759e+06 | 0.39141 | 0.527159 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.31713 | 168694 | 0.146749 | -4.87488e-05 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.107062 | 444.206 | 0.00178399 | 0.00450417 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.107062 | 444.206 | 0.00178399 | 0.00450417 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.540097 | 219.079 | 0.00235074 | 0.0052866 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.619767 | 6887.42 | 0.0244264 | 0.0351442 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.2951 | 1.41782e+06 | 0.380336 | 0.493915 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.26706 | 152232 | 0.140486 | -0.0014537 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0545925 | 465.962 | 0.00210213 | 0.00978804 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0545925 | 465.962 | 0.00210213 | 0.00978804 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.53992 | 214.198 | 0.00247745 | 0.00495074 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.618739 | 7014.25 | 0.0246671 | 0.0313598 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.45323 | 1.57346e+06 | 0.399018 | 0.532096 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.32472 | 163159 | 0.14593 | -0.00198917 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.044224 | -3.34979 | 0.000112014 | -0.000177129 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0965408 | 259.283 | 0.00283636 | 0.00653453 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.550422 | 383.068 | 0.0037544 | 0.00985902 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.630677 | 5670.02 | 0.0203405 | 0.0272717 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.38488 | 1.52947e+06 | 0.392271 | 0.526103 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.044224 | -3.34979 | 0.000112014 | -0.000177129 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.537063 | 262.163 | 0.00277015 | 0.0064947 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.537063 | 262.163 | 0.00277015 | 0.0064947 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.496881 | 244.244 | 0.00217701 | 0.00619474 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.601508 | 7079.7 | 0.0232065 | 0.0361483 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.39658 | 1.54628e+06 | 0.39435 | 0.531072 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.29907 | 168739 | 0.147281 | -0.00123509 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0377097 | -6.74788 | -8.98458e-05 | 0.00025295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 244.342 | 0.00148434 | 0.00512764 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.492656 | 323.476 | 0.00204894 | 0.00739631 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0377097 | -6.74788 | -8.98458e-05 | 0.00025295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0377097 | -6.74788 | -8.98458e-05 | 0.00025295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0377097 | -6.74788 | -8.98458e-05 | 0.00025295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.107873 | 707.006 | 0.00336764 | 0.00340112 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.107873 | 707.006 | 0.00336764 | 0.00340112 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.492625 | 320.221 | 0.00277435 | 0.00585081 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0372432 | -4.84919 | -8.84733e-05 | 0.000217131 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.1007 | 871953 | 0.304355 | 0.216237 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0372432 | -4.84919 | -8.84733e-05 | 0.000217131 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0379256 | -3.62864 | -0.000103494 | 0.000216947 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0379256 | -3.62864 | -0.000103494 | 0.000216947 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.492544 | 334.331 | 0.00354081 | 0.00471294 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.490902 | 6218.37 | 0.0230766 | 0.0215445 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0379256 | -3.62864 | -0.000103494 | 0.000216947 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0379256 | -3.62864 | -0.000103494 | 0.000216947 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0373483 | 4.60446 | 4.88181e-05 | -0.000136367 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0377097 | 376.572 | 0.0022217 | 0.00690541 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.488599 | 296.682 | 0.00214584 | 0.00663184 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.494174 | 5275.58 | 0.0186755 | 0.0271282 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.22578 | 945811 | 0.314479 | 0.243415 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0373483 | 4.60446 | 4.88181e-05 | -0.000136367 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0369691 | 0.765339 | 3.66723e-05 | 0.000161804 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0369691 | 0.765339 | 3.66723e-05 | 0.000161804 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0369691 | 0.765339 | 3.66723e-05 | 0.000161804 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0369691 | 0.765339 | 3.66723e-05 | 0.000161804 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0369691 | 0.765339 | 3.66723e-05 | 0.000161804 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.32922 | 166574 | 0.146222 | -0.00298672 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0517942 | -5.54522 | -6.04124e-05 | 3.84587e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 232.455 | 0.00146188 | 0.00645423 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.520062 | 291.325 | 0.00210329 | 0.00780284 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0517942 | -5.54522 | -6.04124e-05 | 3.84587e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0517942 | -5.54522 | -6.04124e-05 | 3.84587e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0517942 | -5.54522 | -6.04124e-05 | 3.84587e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.121458 | 685.686 | 0.00359497 | 0.00592925 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.121458 | 685.686 | 0.00359497 | 0.00592925 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.520424 | 293.734 | 0.00298972 | 0.00735775 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0509237 | -4.09273 | -5.62905e-05 | 5.78533e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.61394 | 1.71604e+06 | 0.422365 | 0.561987 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0509237 | -4.09273 | -5.62905e-05 | 5.78533e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0520623 | -4.24316 | -8.42603e-05 | 0.000105592 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0520623 | -4.24316 | -8.42603e-05 | 0.000105592 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.518789 | 300.22 | 0.00332869 | 0.00710329 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.61223 | 10307.7 | 0.0310524 | 0.0488095 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0520623 | -4.24316 | -8.42603e-05 | 0.000105592 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0520623 | -4.24316 | -8.42603e-05 | 0.000105592 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0471288 | 4.69258 | 2.95307e-05 | 3.02095e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0517942 | 331.528 | 0.00218871 | 0.0075482 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.515324 | 259.097 | 0.0017337 | 0.00678826 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.616818 | 9130.71 | 0.0270515 | 0.04898 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.74547 | 1.85522e+06 | 0.435712 | 0.604564 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0471288 | 4.69258 | 2.95307e-05 | 3.02095e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0470883 | 2.89377 | 2.06342e-05 | 7.4588e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0470883 | 2.89377 | 2.06342e-05 | 7.4588e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0470883 | 2.89377 | 2.06342e-05 | 7.4588e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0470883 | 2.89377 | 2.06342e-05 | 7.4588e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0470883 | 2.89377 | 2.06342e-05 | 7.4588e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.27217 | 166591 | 0.146238 | -0.000785952 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0394365 | -8.85537 | -7.61147e-05 | 0.00015041 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 241.857 | 0.0011128 | 0.00526018 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.568682 | 327.943 | 0.00161844 | 0.00735054 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0394365 | -8.85537 | -7.61147e-05 | 0.00015041 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0394365 | -8.85537 | -7.61147e-05 | 0.00015041 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0394365 | -8.85537 | -7.61147e-05 | 0.00015041 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.112856 | 639.478 | 0.00230648 | 0.00398878 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.112856 | 639.478 | 0.00230648 | 0.00398878 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.568791 | 324.015 | 0.00228565 | 0.0062623 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0389234 | -7.16642 | -6.8379e-05 | 0.00023644 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.79483 | 971104 | 0.322049 | 0.112961 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0389234 | -7.16642 | -6.8379e-05 | 0.00023644 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0396607 | -6.46119 | -8.99464e-05 | 0.000258629 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0396607 | -6.46119 | -8.99464e-05 | 0.000258629 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.56838 | 336.841 | 0.00302586 | 0.00513476 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.532404 | 7362.27 | 0.0247199 | 0.0290168 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0396607 | -6.46119 | -8.99464e-05 | 0.000258629 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0396607 | -6.46119 | -8.99464e-05 | 0.000258629 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0390175 | 7.25726 | 3.92217e-05 | -9.82747e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0394365 | 380.062 | 0.00179511 | 0.00697694 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.564713 | 310.584 | 0.00162029 | 0.00665228 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.535483 | 6410.56 | 0.0205249 | 0.033974 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.95657 | 1.05688e+06 | 0.33375 | 0.13537 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0390175 | 7.25726 | 3.92217e-05 | -9.82747e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0386707 | 2.44944 | 2.40206e-05 | 8.68316e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0386707 | 2.44944 | 2.40206e-05 | 8.68316e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0386707 | 2.44944 | 2.40206e-05 | 8.68316e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0386707 | 2.44944 | 2.40206e-05 | 8.68316e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0386707 | 2.44944 | 2.40206e-05 | 8.68316e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.37233 | 167835 | 0.146243 | -0.00104991 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0527161 | -4.74147 | -4.86684e-05 | -5.81212e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 237.097 | 0.00158182 | 0.00641909 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.656937 | 299.006 | 0.00224295 | 0.00788684 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0527161 | -4.74147 | -4.86684e-05 | -5.81212e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0527161 | -4.74147 | -4.86684e-05 | -5.81212e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0527161 | -4.74147 | -4.86684e-05 | -5.81212e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.142372 | 709.522 | 0.00358214 | 0.00628518 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.142372 | 709.522 | 0.00358214 | 0.00628518 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.656841 | 298.109 | 0.00295475 | 0.00768271 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0515727 | -3.50201 | -3.85627e-05 | -4.18148e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.91 | 1.67447e+06 | 0.419617 | 0.4714 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0515727 | -3.50201 | -3.85627e-05 | -4.18148e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0531434 | -3.54012 | -4.23428e-05 | -5.7208e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0531434 | -3.54012 | -4.23428e-05 | -5.7208e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.65762 | 307.654 | 0.00338992 | 0.00746964 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.747567 | 10856.8 | 0.0318783 | 0.0503587 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0531434 | -3.54012 | -4.23428e-05 | -5.7208e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0531434 | -3.54012 | -4.23428e-05 | -5.7208e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0486702 | 3.62063 | 3.65722e-05 | 0.000144909 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0527161 | 333.768 | 0.00230732 | 0.00736853 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.652196 | 255.611 | 0.00171668 | 0.00645364 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.749769 | 9664.86 | 0.0281467 | 0.0495986 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 5.09106 | 1.82276e+06 | 0.435053 | 0.507586 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0486702 | 3.62063 | 3.65722e-05 | 0.000144909 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0486927 | 2.01347 | 2.55727e-05 | 4.69899e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0486927 | 2.01347 | 2.55727e-05 | 4.69899e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0486927 | 2.01347 | 2.55727e-05 | 4.69899e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0486927 | 2.01347 | 2.55727e-05 | 4.69899e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0486927 | 2.01347 | 2.55727e-05 | 4.69899e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.35017 | 167740 | 0.146845 | -0.00106691 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | available | 0.639622 | 1.189 | 0.0902468 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | available | 0.71911 | 1.33952 | 0.0986965 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | available | 0.705134 | 1.26789 | 0.14238 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | available | 0.79254 | 1.43939 | 0.145692 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | available | 0.79795 | 1.4657 | 0.130199 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | available | 0.832353 | 1.51466 | 0.15005 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Effective checkpoint policy: `fixed_bank_rescored_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 0 | 6000 | 6000 | 0 | 137.877 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 1 | 7000 | 7000 | 0 | 136.482 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 2 | 9000 | 9000 | 0 | 136.384 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 3 | 10000 | 10000 | 0 | 134.581 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 4 | 12000 | 12000 | 0 | 149.378 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 0 | 9000 | 9000 | 0 | 124.025 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 1 | 12000 | 12000 | 0 | 121.893 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 2 | 5000 | 5000 | 0 | 120.423 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 3 | 9000 | 9000 | 0 | 118.878 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 4 | 9000 | 9000 | 0 | 131.681 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 0 | 9000 | 9000 | 0 | 134.172 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 1 | 9000 | 9000 | 0 | 134.902 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 2 | 12000 | 12000 | 0 | 129.149 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 3 | 9000 | 9000 | 0 | 134.435 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 12000 | 0 | 138.529 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 0 | 8000 | 8000 | 0 | 126.26 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 1 | 10000 | 10000 | 0 | 122.376 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 2 | 12000 | 12000 | 0 | 119.759 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 3 | 10000 | 10000 | 0 | 123.903 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 4 | 12000 | 12000 | 0 | 127.145 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 0 | 6000 | 9000 | 3000 | 135.223 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 1 | 10000 | 10000 | 0 | 133.88 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 2 | 12000 | 12000 | 0 | 127.809 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 3 | 9000 | 9000 | 0 | 132.573 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 12000 | 0 | 136.716 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 0 | 8000 | 8000 | 0 | 130.029 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 1 | 8000 | 8000 | 0 | 121.901 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 2 | 12000 | 12000 | 0 | 118.643 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 3 | 12000 | 12000 | 0 | 120.946 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 4 | 8000 | 11000 | 3000 | 126.595 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
