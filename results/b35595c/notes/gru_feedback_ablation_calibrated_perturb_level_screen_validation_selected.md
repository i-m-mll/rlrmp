# GRU Feedback Ablation Diagnostic

- Issue: `b35595c`
- Source experiment: `b35595c`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 223.33 | 0.00177611 | 0.0056189 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 223.33 | 0.00177611 | 0.0056189 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.479999 | 253.601 | 0.00179956 | 0.00599407 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.487902 | 4717.29 | 0.0172791 | 0.0252796 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.89581 | 932308 | 0.310412 | 0.316865 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.30215 | 167395 | 0.146547 | -0.00241975 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.141137 | 879.249 | 0.00450847 | 0.00294014 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.141137 | 879.249 | 0.00450847 | 0.00294014 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.479751 | 248.071 | 0.00249396 | 0.00457945 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.482445 | 5123.68 | 0.0206535 | 0.0204209 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.75413 | 837499 | 0.297771 | 0.277025 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.22525 | 145720 | 0.137837 | -0.00509851 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00689767 | 215.529 | 0.00147273 | 0.00445285 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00689767 | 215.529 | 0.00147273 | 0.00445285 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.479999 | 252.273 | 0.00194604 | 0.00577491 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.48746 | 4830.98 | 0.0179253 | 0.0241431 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.9007 | 937419 | 0.311595 | 0.316836 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.30234 | 166406 | 0.146399 | -0.00286726 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0333163 | -2.73259 | 5.61216e-05 | -0.000112392 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0333163 | -2.73259 | 5.61216e-05 | -0.000112392 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0333163 | -2.73259 | 5.61216e-05 | -0.000112392 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0333163 | -2.73259 | 5.61216e-05 | -0.000112392 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0333163 | -2.73259 | 5.61216e-05 | -0.000112392 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.28493 | 167359 | 0.147669 | -0.00443409 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.501434 | 202.188 | 0.00302342 | 0.00424172 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.501434 | 202.188 | 0.00302342 | 0.00424172 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.468472 | 233.435 | 0.00240311 | 0.00501555 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.482143 | 5049.66 | 0.0194157 | 0.0200151 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.88557 | 948282 | 0.314281 | 0.316105 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.28536 | 167486 | 0.147711 | -0.00439205 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0347329 | -1.32218 | -1.64818e-06 | 2.95113e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 207.411 | 0.0013287 | 0.00458266 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0347329 | -1.32218 | -1.64818e-06 | 2.95113e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0347329 | -1.32218 | -1.64818e-06 | 2.95113e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0347329 | -1.32218 | -1.64818e-06 | 2.95113e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0347329 | -1.32218 | -1.64818e-06 | 2.95113e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.144688 | 927.771 | 0.00490093 | 0.00214656 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0341165 | -0.995563 | 1.45371e-06 | 3.49581e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0341165 | -0.995563 | 1.45371e-06 | 3.49581e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0341165 | -0.995563 | 1.45371e-06 | 3.49581e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.0424 | 859199 | 0.302952 | 0.219067 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0341165 | -0.995563 | 1.45371e-06 | 3.49581e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00685083 | 207.87 | 0.00110725 | 0.00372014 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.034758 | -1.29982 | -1.18071e-06 | 2.90418e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.034758 | -1.29982 | -1.18071e-06 | 2.90418e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.509201 | 5324.6 | 0.0192394 | 0.0269135 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.034758 | -1.29982 | -1.18071e-06 | 2.90418e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.034758 | -1.29982 | -1.18071e-06 | 2.90418e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0358401 | 3.09687 | -7.72537e-05 | 7.27033e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0358401 | 3.09687 | -7.72537e-05 | 7.27033e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0358401 | 3.09687 | -7.72537e-05 | 7.27033e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0358401 | 3.09687 | -7.72537e-05 | 7.27033e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0358401 | 3.09687 | -7.72537e-05 | 7.27033e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.32166 | 167081 | 0.146418 | -0.00237933 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0353664 | 3.50379 | -6.5537e-05 | 0.00016295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0353664 | 3.50379 | -6.5537e-05 | 0.00016295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0353664 | 3.50379 | -6.5537e-05 | 0.00016295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.508683 | 5813.85 | 0.0213785 | 0.0232735 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0353664 | 3.50379 | -6.5537e-05 | 0.00016295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0353664 | 3.50379 | -6.5537e-05 | 0.00016295 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0361544 | -1.23964 | -2.25653e-06 | 5.24867e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 208.862 | 0.00132896 | 0.00467192 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0361544 | -1.23964 | -2.25653e-06 | 5.24867e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0361544 | -1.23964 | -2.25653e-06 | 5.24867e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0361544 | -1.23964 | -2.25653e-06 | 5.24867e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0361544 | -1.23964 | -2.25653e-06 | 5.24867e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.152475 | 936.816 | 0.00476099 | 0.0030315 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0354353 | -0.94437 | -2.70877e-07 | 3.18311e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0354353 | -0.94437 | -2.70877e-07 | 3.18311e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0354353 | -0.94437 | -2.70877e-07 | 3.18311e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.75434 | 980090 | 0.325278 | 0.126015 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0354353 | -0.94437 | -2.70877e-07 | 3.18311e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00723474 | 209.724 | 0.00108007 | 0.00410926 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0361846 | -1.22234 | -1.94761e-06 | 4.8837e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0361846 | -1.22234 | -1.94761e-06 | 4.8837e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.559953 | 7005.09 | 0.0229841 | 0.036333 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0361846 | -1.22234 | -1.94761e-06 | 4.8837e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0361846 | -1.22234 | -1.94761e-06 | 4.8837e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0372068 | 2.70066 | -6.97106e-05 | 1.79654e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0372068 | 2.70066 | -6.97106e-05 | 1.79654e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0372068 | 2.70066 | -6.97106e-05 | 1.79654e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0372068 | 2.70066 | -6.97106e-05 | 1.79654e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0372068 | 2.70066 | -6.97106e-05 | 1.79654e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.36333 | 167332 | 0.14652 | -0.00208084 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0368726 | 3.04642 | -6.00446e-05 | 0.000137352 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0368726 | 3.04642 | -6.00446e-05 | 0.000137352 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0368726 | 3.04642 | -6.00446e-05 | 0.000137352 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.563874 | 7506.02 | 0.0249538 | 0.0330306 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0368726 | 3.04642 | -6.00446e-05 | 0.000137352 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0368726 | 3.04642 | -6.00446e-05 | 0.000137352 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 237.339 | 0.00171613 | 0.00618512 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 237.339 | 0.00171613 | 0.00618512 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.624214 | 255.252 | 0.00172551 | 0.00632092 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.675007 | 6540.34 | 0.0214908 | 0.0365138 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.76195 | 1.20316e+06 | 0.354528 | 0.254489 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.32847 | 169196 | 0.147008 | -0.000162905 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.142337 | 835.549 | 0.00405199 | 0.00433546 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.142337 | 835.549 | 0.00405199 | 0.00433546 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.624638 | 248.724 | 0.00234153 | 0.00525702 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.670677 | 7021.44 | 0.0248373 | 0.0328543 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.58982 | 1.07944e+06 | 0.339301 | 0.213034 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.25058 | 147399 | 0.138329 | -0.00201257 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00778858 | 223.803 | 0.00135911 | 0.00542075 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00778858 | 223.803 | 0.00135911 | 0.00542075 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.624077 | 254.269 | 0.00182655 | 0.00616649 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.674431 | 6644.18 | 0.0220146 | 0.0355885 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.76894 | 1.20886e+06 | 0.35563 | 0.254352 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.32926 | 168313 | 0.146861 | -0.000401124 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0384503 | -1.92542 | 4.35387e-05 | -0.000104923 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0384503 | -1.92542 | 4.35387e-05 | -0.000104923 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0384503 | -1.92542 | 4.35387e-05 | -0.000104923 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0384503 | -1.92542 | 4.35387e-05 | -0.000104923 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0384503 | -1.92542 | 4.35387e-05 | -0.000104923 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.31853 | 169171 | 0.147844 | -0.00151337 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.645336 | 231.12 | 0.00271717 | 0.00495119 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.645336 | 231.12 | 0.00271717 | 0.00495119 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.611868 | 237.556 | 0.00206905 | 0.00556575 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.668148 | 6702.12 | 0.0229527 | 0.0321643 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.75979 | 1.22639e+06 | 0.358776 | 0.255496 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.31795 | 169281 | 0.147914 | -0.00136534 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0442694 | -0.491261 | -7.2243e-07 | 4.71479e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 208.598 | 0.00123777 | 0.00534442 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0442694 | -0.491261 | -7.2243e-07 | 4.71479e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0442694 | -0.491261 | -7.2243e-07 | 4.71479e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0442694 | -0.491261 | -7.2243e-07 | 4.71479e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0442694 | -0.491261 | -7.2243e-07 | 4.71479e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.156866 | 818.522 | 0.00376811 | 0.00466415 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0433293 | -0.480746 | 2.03315e-06 | 1.93472e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0433293 | -0.480746 | 2.03315e-06 | 1.93472e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0433293 | -0.480746 | 2.03315e-06 | 1.93472e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.68301 | 1.47775e+06 | 0.388242 | 0.408493 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0433293 | -0.480746 | 2.03315e-06 | 1.93472e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00926731 | 202.681 | 0.000861485 | 0.00532095 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0443143 | -0.489641 | -5.41462e-07 | 4.11539e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0443143 | -0.489641 | -5.41462e-07 | 4.11539e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.679766 | 8096.56 | 0.0244868 | 0.0458154 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0443143 | -0.489641 | -5.41462e-07 | 4.11539e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0443143 | -0.489641 | -5.41462e-07 | 4.11539e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0442352 | 1.02456 | -6.68217e-05 | -1.76984e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0442352 | 1.02456 | -6.68217e-05 | -1.76984e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0442352 | 1.02456 | -6.68217e-05 | -1.76984e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0442352 | 1.02456 | -6.68217e-05 | -1.76984e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0442352 | 1.02456 | -6.68217e-05 | -1.76984e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.28897 | 168003 | 0.146058 | -0.00029869 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0436958 | 1.11706 | -5.83832e-05 | 6.25442e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0436958 | 1.11706 | -5.83832e-05 | 6.25442e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0436958 | 1.11706 | -5.83832e-05 | 6.25442e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.682222 | 8575.86 | 0.0261675 | 0.0444369 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0436958 | 1.11706 | -5.83832e-05 | 6.25442e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0436958 | 1.11706 | -5.83832e-05 | 6.25442e-06 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0508878 | -0.729724 | 2.90394e-06 | 2.75751e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 207.976 | 0.00163577 | 0.00527093 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0508878 | -0.729724 | 2.90394e-06 | 2.75751e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0508878 | -0.729724 | 2.90394e-06 | 2.75751e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0508878 | -0.729724 | 2.90394e-06 | 2.75751e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0508878 | -0.729724 | 2.90394e-06 | 2.75751e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.196639 | 967.315 | 0.005282 | 0.00507735 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0493315 | -0.518303 | 1.06488e-06 | 3.09688e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0493315 | -0.518303 | 1.06488e-06 | 3.09688e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0493315 | -0.518303 | 1.06488e-06 | 3.09688e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 4.43091 | 1.73959e+06 | 0.425744 | 0.619139 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0493315 | -0.518303 | 1.06488e-06 | 3.09688e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0124084 | 213.891 | 0.0013977 | 0.00527711 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0509854 | -0.737482 | 2.90135e-06 | 2.76979e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0509854 | -0.737482 | 2.90135e-06 | 2.76979e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.707515 | 9391.68 | 0.0280997 | 0.0492207 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0509854 | -0.737482 | 2.90135e-06 | 2.76979e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0509854 | -0.737482 | 2.90135e-06 | 2.76979e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0512266 | 1.80007 | -6.14635e-05 | 1.06671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0512266 | 1.80007 | -6.14635e-05 | 1.06671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0512266 | 1.80007 | -6.14635e-05 | 1.06671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0512266 | 1.80007 | -6.14635e-05 | 1.06671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0512266 | 1.80007 | -6.14635e-05 | 1.06671e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.31508 | 167922 | 0.146871 | -0.000370194 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0509064 | 1.81849 | -4.64366e-05 | 1.90355e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0509064 | 1.81849 | -4.64366e-05 | 1.90355e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0509064 | 1.81849 | -4.64366e-05 | 1.90355e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.712095 | 9972.31 | 0.0299929 | 0.0480564 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0509064 | 1.81849 | -4.64366e-05 | 1.90355e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0509064 | 1.81849 | -4.64366e-05 | 1.90355e-05 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | available | 0.67786 | 1.18 | 0.175716 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | available | 0.71422 | 1.24451 | 0.183933 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | available | 0.817361 | 1.44891 | 0.185816 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | available | 0.805632 | 1.43611 | 0.175158 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | available | 0.8113 | 1.44934 | 0.173264 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | available | 0.784074 | 1.36911 | 0.19904 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 0 | 10000 | 2000 | -8000 | -129.831 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 1 | 11000 | 1000 | -10000 | -133.963 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 2 | 10000 | 1000 | -9000 | -244.045 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 3 | 10000 | 1000 | -9000 | -237.741 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 4 | 10000 | 1000 | -9000 | -307.347 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 0 | 12000 | 2000 | -10000 | -179.19 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 1 | 12000 | 1000 | -11000 | -184.138 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 2 | 11000 | 1000 | -10000 | -205.405 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 3 | 11000 | 2000 | -9000 | -173.236 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 4 | 12000 | 1000 | -11000 | -267.146 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 0 | 12000 | 3000 | -9000 | -157.689 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 1 | 12000 | 1000 | -11000 | -325.861 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 2 | 12000 | 1000 | -11000 | -249.343 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 3 | 11000 | 2000 | -9000 | -195.719 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 4 | 12000 | 1000 | -11000 | -321.212 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 0 | 6000 | 9000 | 3000 | -118.536 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 1 | 8000 | 3000 | -5000 | -131.441 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 2 | 6000 | 2000 | -4000 | -231.681 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 3 | 5000 | 3000 | -2000 | -176.534 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 4 | 6000 | 7000 | 1000 | -139.979 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 0 | 11000 | 1000 | -10000 | -220.314 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 1 | 11000 | 2000 | -9000 | -207.631 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 2 | 7000 | 2000 | -5000 | -142.824 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 3 | 5000 | 3000 | -2000 | -135.988 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 4 | 7000 | 6000 | -1000 | -147.409 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 0 | 12000 | 3000 | -9000 | -174.75 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 1 | 12000 | 1000 | -11000 | -163.445 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 2 | 12000 | 2000 | -10000 | -178.369 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 3 | 10000 | 3000 | -7000 | -165.862 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 4 | 10000 | 1000 | -9000 | -283.713 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
