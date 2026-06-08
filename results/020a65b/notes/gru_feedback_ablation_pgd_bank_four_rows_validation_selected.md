# GRU Feedback Ablation Diagnostic

- Issue: `020a65b`
- Source experiment: `020a65b`
- Scope: `pgd_bank_four_rows_validation_selected`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 212.472 | 0.00126115 | 0.00488604 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 212.472 | 0.00126115 | 0.00488604 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.44637 | 247 | 0.00154613 | 0.00596896 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.606485 | 19410 | 0.0417527 | 0.0778771 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 11.5871 | 4.41242e+06 | 0.678825 | 0.277372 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 6.35794 | 228103 | 0.150018 | 0.257803 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.119415 | 835.427 | 0.00530719 | 0.00175249 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.119415 | 835.427 | 0.00530719 | 0.00175249 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.447738 | 245.677 | 0.00290622 | 0.0042052 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.607356 | 20715.5 | 0.0469358 | 0.0717078 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 11.3484 | 4.03247e+06 | 0.653055 | 0.331635 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 6.33798 | 205875 | 0.142237 | 0.254669 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00939739 | 209.557 | 0.00083818 | 0.00459868 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00939739 | 209.557 | 0.00083818 | 0.00459868 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.44611 | 247.424 | 0.00161144 | 0.00590954 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.606564 | 19489.4 | 0.0420254 | 0.0775025 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 11.6047 | 4.42248e+06 | 0.679737 | 0.277842 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.36046 | 227702 | 0.150119 | 0.257747 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.091596 | 188.129 | 0.00190714 | 0.00394952 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.091596 | 188.129 | 0.00190714 | 0.00394952 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.459548 | 703.731 | 0.00577389 | 0.0170019 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.63019 | 15265.6 | 0.0376332 | 0.057315 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 11.5845 | 4.41239e+06 | 0.679471 | 0.276435 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 6.34754 | 228079 | 0.150664 | 0.256866 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.091596 | 188.129 | 0.00190714 | 0.00394952 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.091596 | 188.129 | 0.00190714 | 0.00394952 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.447214 | 246.029 | 0.0018956 | 0.00540547 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.602544 | 19732.8 | 0.0430701 | 0.075129 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 11.6561 | 4.49e+06 | 0.685249 | 0.271819 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.34754 | 228079 | 0.150664 | 0.256866 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 245.43 | 0.00294252 | 0.00305901 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 245.43 | 0.00294252 | 0.00305901 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.493813 | 269.772 | 0.00318971 | 0.00362897 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.538028 | 18440.3 | 0.0428578 | 0.05959 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 10.0452 | 4.56048e+06 | 0.642317 | 0.761486 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 6.02174 | 305219 | 0.144298 | 0.381694 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.254108 | 1318.55 | 0.00864219 | -0.0014293 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.254108 | 1318.55 | 0.00864219 | -0.0014293 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.487617 | 275.688 | 0.00321275 | 0.00277163 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.560574 | 17828.1 | 0.0431791 | 0.0503156 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 9.53576 | 4.20844e+06 | 0.615435 | 0.790788 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 6.01816 | 283231 | 0.134414 | 0.377206 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0205303 | 251.825 | 0.00310569 | 0.0016057 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0205303 | 251.825 | 0.00310569 | 0.0016057 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.493928 | 270.414 | 0.00319163 | 0.00356657 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.53991 | 18483.2 | 0.042964 | 0.0591083 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 10.0653 | 4.56907e+06 | 0.643054 | 0.760457 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 6.02163 | 304163 | 0.144319 | 0.38289 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.332126 | 30.7184 | 0.00282608 | 0.00411387 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.332126 | 30.7184 | 0.00282608 | 0.00411387 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.564147 | 1380.2 | 0.00757805 | 0.0293265 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.608775 | 12334 | 0.0350649 | 0.034577 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 10.02 | 4.56026e+06 | 0.642201 | 0.762541 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 6.02546 | 305005 | 0.144182 | 0.382749 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.332126 | 30.7184 | 0.00282608 | 0.00411387 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.332126 | 30.7184 | 0.00282608 | 0.00411387 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.523808 | 267.767 | 0.00311486 | 0.00378234 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.545776 | 18741.8 | 0.0435762 | 0.0624214 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 10.2636 | 4.54521e+06 | 0.640621 | 0.778445 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.02546 | 305005 | 0.144182 | 0.382749 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 225.523 | 0.00213647 | 0.00514044 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 225.523 | 0.00213647 | 0.00514044 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.516015 | 255.78 | 0.00237213 | 0.00622919 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.739332 | 43664.5 | 0.0670285 | 0.129422 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 14.1134 | 8.46051e+06 | 0.951066 | 0.55279 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 7.11066 | 245670 | 0.150015 | 0.483208 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.165298 | 1038.22 | 0.00677192 | 0.00431349 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.165298 | 1038.22 | 0.00677192 | 0.00431349 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.517082 | 250.974 | 0.00330097 | 0.00561338 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.732804 | 47599.8 | 0.0725387 | 0.132655 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 13.7456 | 7.76371e+06 | 0.914332 | 0.527629 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 7.05043 | 223858 | 0.141473 | 0.482381 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00995544 | 243.661 | 0.00192292 | 0.00510655 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00995544 | 243.661 | 0.00192292 | 0.00510655 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.516043 | 256.192 | 0.00243907 | 0.00621812 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.739869 | 43774.1 | 0.0672754 | 0.129224 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 14.129 | 8.47512e+06 | 0.95201 | 0.553161 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 7.11178 | 245508 | 0.15022 | 0.482901 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.124337 | 186.558 | 0.00293191 | 0.0047493 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.124337 | 186.558 | 0.00293191 | 0.0047493 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.53142 | 1104.01 | 0.00843139 | 0.0274587 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.771624 | 33876.9 | 0.0605 | 0.0984216 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 14.1098 | 8.46047e+06 | 0.951862 | 0.552399 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 7.09647 | 245631 | 0.150811 | 0.482816 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.124337 | 186.558 | 0.00293191 | 0.0047493 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.124337 | 186.558 | 0.00293191 | 0.0047493 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.515976 | 255.011 | 0.00290302 | 0.00595271 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.736298 | 44402.2 | 0.0687809 | 0.127854 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 14.1975 | 8.60878e+06 | 0.959913 | 0.558284 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 7.09647 | 245631 | 0.150811 | 0.482816 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 265.741 | 0.00336712 | 0.00595721 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 265.741 | 0.00336712 | 0.00595721 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.34118 | 303.51 | 0.00365558 | 0.00718097 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 1.14046 | 33087.5 | 0.0606147 | 0.107187 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 15.6442 | 6.42384e+06 | 0.557699 | 2.62254 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 4.0674 | 166558 | 0.147286 | 0.0355531 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.286479 | 1520.07 | 0.00871006 | 0.00593091 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.286479 | 1520.07 | 0.00871006 | 0.00593091 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.31615 | 293.182 | 0.00358302 | 0.00693631 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 1.13382 | 36037.4 | 0.0636022 | 0.113995 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 15.1492 | 5.92951e+06 | 0.525688 | 2.54607 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.92427 | 145050 | 0.137295 | 0.0355268 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0188075 | 282.08 | 0.00343533 | 0.00602081 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0188075 | 282.08 | 0.00343533 | 0.00602081 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.34207 | 304.042 | 0.00366053 | 0.00719731 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 1.14121 | 33107.3 | 0.0606639 | 0.107185 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 15.665 | 6.43625e+06 | 0.558326 | 2.62486 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 4.07283 | 166489 | 0.147342 | 0.0353483 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.525174 | 15.9501 | 0.00338069 | 0.00597179 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.525174 | 15.9501 | 0.00338069 | 0.00597179 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.39582 | 4957.2 | 0.0180408 | 0.0596678 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 1.27284 | 15884 | 0.0421225 | 0.045286 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 15.5678 | 6.42359e+06 | 0.557713 | 2.62256 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 4.17075 | 166308 | 0.1473 | 0.0355676 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.525174 | 15.9501 | 0.00338069 | 0.00597179 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.525174 | 15.9501 | 0.00338069 | 0.00597179 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.32072 | 304.061 | 0.00364826 | 0.00718185 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 1.16215 | 33167.9 | 0.061176 | 0.107959 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 15.6923 | 6.53562e+06 | 0.565412 | 2.63939 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 4.17075 | 166308 | 0.1473 | 0.0355676 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | available | 1.83998 | 3.51242 | 0.167534 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | available | 1.40466 | 2.60856 | 0.200761 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | available | 2.27428 | 4.3395 | 0.209066 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | available | 2.11279 | 3.96041 | 0.26517 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | 0 | 11000 | 2000 | -9000 | 134.436 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | 1 | 9500 | 5000 | -4500 | 146.683 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | 2 | 9000 | 10000 | 1000 | 158.759 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | 3 | 9500 | 8500 | -1000 | 147.411 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | 4 | 11000 | 3000 | -8000 | 130.525 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | 0 | 11000 | 3500 | -7500 | 101.803 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | 1 | 5000 | 2500 | -2500 | 135.881 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | 2 | 6000 | 2000 | -4000 | 137.83 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | 3 | 5500 | 3000 | -2500 | 110.651 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | 4 | 10000 | 1500 | -8500 | 140.345 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 0 | 6000 | 9000 | 3000 | 146.552 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 1 | 5500 | 2000 | -3500 | 108.215 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 2 | 6000 | 9500 | 3500 | 146.626 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 3 | 6000 | 8500 | 2500 | 144.057 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 4 | 6500 | 1500 | -5000 | 131.447 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 0 | 12000 | 2000 | -10000 | 143.683 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 1 | 10500 | 2000 | -8500 | 132.42 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 2 | 11000 | 2000 | -9000 | 112.024 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 3 | 10500 | 3000 | -7500 | 152.649 | 8 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 4 | 12000 | 1500 | -10500 | 129.28 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
