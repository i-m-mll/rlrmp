# GRU Feedback Ablation Diagnostic

- Issue: `b35595c`
- Source experiment: `b35595c`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

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
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 178.197 | 0.00171212 | 0.00526298 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 178.197 | 0.00171212 | 0.00526298 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.494885 | 273.441 | 0.00226231 | 0.00632591 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.495928 | 4171.48 | 0.0161372 | 0.0221351 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.88876 | 931351 | 0.310157 | 0.319196 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.30156 | 167256 | 0.146607 | -0.00254919 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.106151 | 463 | 0.00189811 | 0.00320242 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.106151 | 463 | 0.00189811 | 0.00320242 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.49464 | 266.262 | 0.00271276 | 0.00542372 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.491989 | 4459.04 | 0.0186901 | 0.018473 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.78164 | 859486 | 0.300661 | 0.289243 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.2441 | 150878 | 0.140098 | -0.00460975 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0450481 | 454.821 | 0.00274295 | 0.00550691 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0450481 | 454.821 | 0.00274295 | 0.00550691 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.494842 | 242.956 | 0.00317808 | 0.00493721 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.492975 | 4882.18 | 0.0202443 | 0.0139078 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.92218 | 964929 | 0.317773 | 0.318115 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.30287 | 160806 | 0.145533 | -0.00633633 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.035127 | -4.36242 | 0.000148021 | -1.89167e-05 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0913572 | 288.97 | 0.00375634 | 0.00712283 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.509687 | 361.234 | 0.00428784 | 0.0106449 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.504325 | 3813.77 | 0.0158712 | 0.0123143 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.85371 | 931654 | 0.311097 | 0.316952 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.28192 | 167224 | 0.147453 | -0.00384782 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.489203 | 295.468 | 0.00364082 | 0.00710727 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.489203 | 295.468 | 0.00364082 | 0.00710727 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.454612 | 249.305 | 0.00268811 | 0.0063164 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.477795 | 5106.69 | 0.0191148 | 0.0213941 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.87988 | 941992 | 0.312936 | 0.31609 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.28007 | 167366 | 0.147369 | -0.00416014 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 198.198 | 0.00153949 | 0.00610387 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 198.198 | 0.00153949 | 0.00610387 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.643797 | 244.652 | 0.00190985 | 0.00602317 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.687107 | 6114.6 | 0.0207601 | 0.0344094 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.75587 | 1.20266e+06 | 0.354391 | 0.256341 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.32826 | 169068 | 0.147028 | -0.000385143 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.106947 | 474.921 | 0.00186371 | 0.00468405 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.106947 | 474.921 | 0.00186371 | 0.00468405 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.644129 | 236.589 | 0.00227259 | 0.00549119 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.684274 | 6466.92 | 0.0232906 | 0.0316869 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.62577 | 1.10886e+06 | 0.342944 | 0.225114 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.26994 | 152595 | 0.14053 | -0.00180496 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0512542 | 503.015 | 0.00234025 | 0.00887348 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0512542 | 503.015 | 0.00234025 | 0.00887348 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.642996 | 227.548 | 0.00247237 | 0.0051002 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.683308 | 6775.2 | 0.0241002 | 0.027445 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.80337 | 1.2401e+06 | 0.361518 | 0.254494 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.33314 | 163309 | 0.14599 | -0.00286751 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0397817 | -2.4631 | 0.000101331 | -0.000109351 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.105228 | 277.35 | 0.00300571 | 0.00689953 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.656408 | 399.15 | 0.00402965 | 0.0101248 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.695131 | 5353.94 | 0.0196772 | 0.0237273 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.72777 | 1.20355e+06 | 0.355173 | 0.255067 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.31469 | 169044 | 0.147654 | -0.00123657 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.642075 | 289.179 | 0.0029143 | 0.00719741 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.642075 | 289.179 | 0.0029143 | 0.00719741 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.597934 | 246.041 | 0.00232353 | 0.00597938 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.664329 | 6755.83 | 0.0226093 | 0.0331688 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.74309 | 1.21672e+06 | 0.357069 | 0.255791 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.30824 | 169173 | 0.147581 | -0.00108428 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 240.021 | 0.00141703 | 0.00514591 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 240.021 | 0.00141703 | 0.00514591 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.502192 | 317.796 | 0.00193017 | 0.0073655 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0374106 | -6.45119 | -8.30497e-05 | 0.000196035 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0374106 | -6.45119 | -8.30497e-05 | 0.000196035 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0374106 | -6.45119 | -8.30497e-05 | 0.000196035 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.108616 | 690.035 | 0.00320656 | 0.00341861 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.108616 | 690.035 | 0.00320656 | 0.00341861 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0369224 | -4.58421 | -7.58292e-05 | 0.000189845 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0369224 | -4.58421 | -7.58292e-05 | 0.000189845 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0369224 | -4.58421 | -7.58292e-05 | 0.000189845 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.25024 | 150590 | 0.139683 | -0.00444229 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0376281 | -3.43361 | -9.05952e-05 | 0.000186951 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0447725 | 785.405 | 0.00405331 | 0.00978601 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0376281 | -3.43361 | -9.05952e-05 | 0.000186951 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0376281 | -3.43361 | -9.05952e-05 | 0.000186951 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0376281 | -3.43361 | -9.05952e-05 | 0.000186951 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.31399 | 160647 | 0.145175 | -0.00613007 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0367516 | 4.08302 | 4.01058e-05 | -0.000103318 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0374106 | 365.647 | 0.00205586 | 0.00686103 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.498372 | 285.303 | 0.00196831 | 0.00660158 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.505904 | 5340.05 | 0.0187243 | 0.0278111 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.20808 | 957348 | 0.316302 | 0.2544 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0367516 | 4.08302 | 4.01058e-05 | -0.000103318 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0364063 | 0.644691 | 3.08279e-05 | 0.000212324 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.496761 | 359.893 | 0.00207108 | 0.00694284 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.546971 | 264.89 | 0.0017453 | 0.00623787 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0364063 | 0.644691 | 3.08279e-05 | 0.000212324 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0364063 | 0.644691 | 3.08279e-05 | 0.000212324 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.31971 | 166949 | 0.146375 | -0.00255191 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 217.474 | 0.000994729 | 0.0064721 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 217.474 | 0.000994729 | 0.0064721 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.600802 | 275.252 | 0.00142607 | 0.00792103 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0494512 | -5.50557 | -5.58363e-05 | -7.671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0494512 | -5.50557 | -5.58363e-05 | -7.671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0494512 | -5.50557 | -5.58363e-05 | -7.671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.117744 | 579.64 | 0.00230414 | 0.00619505 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.117744 | 579.64 | 0.00230414 | 0.00619505 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0487127 | -3.93093 | -5.29746e-05 | 1.76168e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0487127 | -3.93093 | -5.29746e-05 | 1.76168e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0487127 | -3.93093 | -5.29746e-05 | 1.76168e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.22105 | 151473 | 0.139423 | -0.000844016 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0497619 | -3.69114 | -6.71786e-05 | 0.000140253 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.060871 | 731.885 | 0.00272027 | 0.0160916 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0497619 | -3.69114 | -6.71786e-05 | 0.000140253 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0497619 | -3.69114 | -6.71786e-05 | 0.000140253 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0497619 | -3.69114 | -6.71786e-05 | 0.000140253 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.28681 | 162636 | 0.145059 | -0.00119228 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0449283 | 3.48195 | 2.86179e-05 | 1.56955e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0494512 | 326.46 | 0.00160703 | 0.00749891 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.595739 | 260.476 | 0.00151939 | 0.00648785 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.678941 | 8147.44 | 0.0240379 | 0.0465602 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.87357 | 1.64116e+06 | 0.405291 | 0.452696 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0449283 | 3.48195 | 2.86179e-05 | 1.56955e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0445143 | 1.40052 | 8.54138e-06 | 1.03556e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.590181 | 327.906 | 0.00165537 | 0.0079515 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.642101 | 246.865 | 0.00133364 | 0.00681681 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0445143 | 1.40052 | 8.54138e-06 | 1.03556e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0445143 | 1.40052 | 8.54138e-06 | 1.03556e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.28433 | 167862 | 0.146003 | 2.17542e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 243.883 | 0.00144663 | 0.00548201 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 243.883 | 0.00144663 | 0.00548201 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.584934 | 322.411 | 0.00192595 | 0.00762848 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0391202 | -6.32405 | -7.67483e-05 | 0.000106723 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0391202 | -6.32405 | -7.67483e-05 | 0.000106723 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0391202 | -6.32405 | -7.67483e-05 | 0.000106723 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.114553 | 697.169 | 0.00311715 | 0.00433576 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.114553 | 697.169 | 0.00311715 | 0.00433576 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0385645 | -4.74601 | -6.99915e-05 | 0.000185759 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0385645 | -4.74601 | -6.99915e-05 | 0.000185759 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0385645 | -4.74601 | -6.99915e-05 | 0.000185759 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.2779 | 150835 | 0.139694 | -0.00349212 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0393478 | -3.883 | -8.29864e-05 | 0.000205173 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0474857 | 799.353 | 0.00386899 | 0.0114432 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0393478 | -3.883 | -8.29864e-05 | 0.000205173 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0393478 | -3.883 | -8.29864e-05 | 0.000205173 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0393478 | -3.883 | -8.29864e-05 | 0.000205173 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.3527 | 161305 | 0.145289 | -0.00502041 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0381857 | 3.69888 | 3.7124e-05 | -5.91822e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0391202 | 369.733 | 0.00207722 | 0.00719884 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.580867 | 284.888 | 0.0019433 | 0.00692263 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.555796 | 7011.1 | 0.0224854 | 0.037023 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.9774 | 1.09721e+06 | 0.341065 | 0.154021 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0381857 | 3.69888 | 3.7124e-05 | -5.91822e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0379084 | 0.29295 | 2.44258e-05 | 0.00016695 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.578338 | 364.826 | 0.00210263 | 0.00716683 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.636546 | 264.657 | 0.00174132 | 0.00631664 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0379084 | 0.29295 | 2.44258e-05 | 0.00016695 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0379084 | 0.29295 | 2.44258e-05 | 0.00016695 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.36259 | 167200 | 0.146481 | -0.00229716 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 224.931 | 0.00153742 | 0.00644711 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 224.931 | 0.00153742 | 0.00644711 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.616454 | 285.127 | 0.00218944 | 0.00795442 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0551219 | -4.88784 | -5.43796e-05 | -2.31307e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0551219 | -4.88784 | -5.43796e-05 | -2.31307e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0551219 | -4.88784 | -5.43796e-05 | -2.31307e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.147379 | 695.953 | 0.00354794 | 0.00643397 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.147379 | 695.953 | 0.00354794 | 0.00643397 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0539721 | -3.62076 | -5.26359e-05 | -4.50579e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0539721 | -3.62076 | -5.26359e-05 | -4.50579e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0539721 | -3.62076 | -5.26359e-05 | -4.50579e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.23113 | 151395 | 0.140056 | -0.000578598 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0557473 | -3.67606 | -6.89078e-05 | 6.2824e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0812884 | 822.859 | 0.00402096 | 0.0167603 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0557473 | -3.67606 | -6.89078e-05 | 6.2824e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0557473 | -3.67606 | -6.89078e-05 | 6.2824e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0557473 | -3.67606 | -6.89078e-05 | 6.2824e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.32799 | 162902 | 0.145911 | -0.000924161 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0508225 | 3.61549 | 4.02779e-05 | 8.14502e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0551219 | 324.459 | 0.00223364 | 0.00736973 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.61134 | 255.573 | 0.00180551 | 0.00677317 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.70624 | 9314.51 | 0.027565 | 0.0494257 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.64374 | 1.9397e+06 | 0.445645 | 0.671743 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0508225 | 3.61549 | 4.02779e-05 | 8.14502e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0508155 | 1.53135 | 2.63656e-05 | 2.46298e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.610209 | 319.264 | 0.00223698 | 0.00754288 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.654749 | 247 | 0.00174749 | 0.00707363 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0508155 | 1.53135 | 2.63656e-05 | 2.46298e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0508155 | 1.53135 | 2.63656e-05 | 2.46298e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.31227 | 167736 | 0.146781 | -0.000565398 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | available | 0.63947 | 1.18043 | 0.0985128 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | available | 0.770243 | 1.4355 | 0.104986 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | available | 0.703955 | 1.26599 | 0.141919 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | available | 0.808183 | 1.47616 | 0.140209 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | available | 0.810844 | 1.47733 | 0.144353 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | available | 0.774702 | 1.39441 | 0.15499 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

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
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 0 | 10000 | 6000 | -4000 | 137.877 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 1 | 11000 | 7000 | -4000 | 136.482 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 2 | 10000 | 9000 | -1000 | 136.384 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 3 | 10000 | 10000 | 0 | 134.581 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 4 | 10000 | 12000 | 2000 | 149.378 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 0 | 6000 | 9000 | 3000 | 124.025 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 1 | 8000 | 12000 | 4000 | 121.893 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 2 | 6000 | 5000 | -1000 | 120.423 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 3 | 5000 | 9000 | 4000 | 118.878 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 4 | 6000 | 9000 | 3000 | 131.681 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 0 | 12000 | 9000 | -3000 | 133.785 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 1 | 12000 | 9000 | -3000 | 133.946 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 2 | 11000 | 12000 | 1000 | 128.969 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 3 | 11000 | 9000 | -2000 | 132.631 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 12000 | 0 | 138.587 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 0 | 11000 | 8000 | -3000 | 124.826 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 1 | 11000 | 10000 | -1000 | 122.547 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 2 | 7000 | 12000 | 5000 | 120.113 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 3 | 5000 | 10000 | 5000 | 123.252 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 4 | 7000 | 12000 | 5000 | 126.968 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 0 | 12000 | 6000 | -6000 | 130.719 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 1 | 12000 | 10000 | -2000 | 132.482 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 2 | 12000 | 12000 | 0 | 127.971 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 3 | 11000 | 9000 | -2000 | 132.311 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 12000 | 0 | 136.827 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 0 | 12000 | 8000 | -4000 | 128.914 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 1 | 12000 | 8000 | -4000 | 121.395 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 2 | 12000 | 12000 | 0 | 118.624 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 3 | 10000 | 12000 | 2000 | 120.895 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 4 | 10000 | 8000 | -2000 | 126.064 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
