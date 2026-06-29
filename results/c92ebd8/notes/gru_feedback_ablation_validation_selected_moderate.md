# GRU Feedback Ablation Diagnostic

- Issue: `c92ebd8`
- Source experiment: `c92ebd8`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `open_loop_small` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `open_loop_moderate` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `open_loop_stress` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `closed_loop_small` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `closed_loop_moderate` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `closed_loop_stress` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `closed_loop_cmd_lateral_small` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `closed_loop_cmd_lateral_moderate` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `closed_loop_cmd_lateral_stress` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `open_loop_small` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `nominal` | `shuffled_observation_history` | evaluated | 0.286758 | 240.592 | 0.00169862 | 0.00656866 |
| `open_loop_small` | `nominal` | `lagged_observation_history` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `nominal` | `position_only_observation` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `nominal` | `velocity_only_observation` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `initial_state` | `lagged_observation_history` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `initial_state` | `position_only_observation` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `initial_state` | `velocity_only_observation` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.477498 | 17318.7 | 0.0422417 | 0.0682448 |
| `open_loop_small` | `process_epsilon` | `position_only_observation` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0412212 | -2.42805 | 8.86367e-06 | -3.49844e-05 |
| `open_loop_small` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0412212 | -2.42805 | 8.86367e-06 | -3.49844e-05 |
| `open_loop_small` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.303039 | 751.768 | 0.00697523 | 0.022024 |
| `open_loop_small` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.50825 | 12126.5 | 0.0345256 | 0.0474966 |
| `open_loop_small` | `sensory_feedback` | `position_only_observation` | evaluated | 8.51928 | 4.75509e+06 | 0.700822 | 0.713228 |
| `open_loop_small` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.4405 | 176541 | 0.140987 | 0.14219 |
| `open_loop_small` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 243.447 | 0.00184131 | 0.00659934 |
| `open_loop_small` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `delayed_observation` | `position_only_observation` | evaluated | 8.54165 | 4.75512e+06 | 0.699551 | 0.71403 |
| `open_loop_small` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.4932 | 176572 | 0.139715 | 0.142992 |
| `open_loop_moderate` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `shuffled_observation_history` | evaluated | 0.373564 | 239.267 | 0.00170832 | 0.00686772 |
| `open_loop_moderate` | `nominal` | `lagged_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `position_only_observation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `velocity_only_observation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `lagged_observation_history` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `position_only_observation` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `velocity_only_observation` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.54353 | 21223.4 | 0.0466434 | 0.0843124 |
| `open_loop_moderate` | `process_epsilon` | `position_only_observation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0474215 | -2.436 | 7.72761e-06 | -1.48497e-05 |
| `open_loop_moderate` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0474215 | -2.436 | 7.72761e-06 | -1.48497e-05 |
| `open_loop_moderate` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.399097 | 1248.85 | 0.00965932 | 0.0307219 |
| `open_loop_moderate` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.585002 | 13582.6 | 0.0372142 | 0.051753 |
| `open_loop_moderate` | `sensory_feedback` | `position_only_observation` | evaluated | 10.5244 | 3.86295e+06 | 0.640646 | 0.151627 |
| `open_loop_moderate` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.2738 | 165378 | 0.143397 | 0.104539 |
| `open_loop_moderate` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 241.781 | 0.00183557 | 0.00700159 |
| `open_loop_moderate` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `position_only_observation` | evaluated | 10.5669 | 3.863e+06 | 0.639084 | 0.151965 |
| `open_loop_moderate` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.3271 | 165424 | 0.141836 | 0.104877 |
| `open_loop_stress` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `nominal` | `shuffled_observation_history` | evaluated | 0.678391 | 247.227 | 0.00180722 | 0.00716426 |
| `open_loop_stress` | `nominal` | `lagged_observation_history` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `nominal` | `position_only_observation` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `nominal` | `velocity_only_observation` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `initial_state` | `lagged_observation_history` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `initial_state` | `position_only_observation` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `initial_state` | `velocity_only_observation` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `process_epsilon` | `lagged_observation_history` | evaluated | 1.03757 | 46282.4 | 0.0702071 | 0.135682 |
| `open_loop_stress` | `process_epsilon` | `position_only_observation` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0745257 | -2.14124 | 2.38873e-06 | -8.23866e-06 |
| `open_loop_stress` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0745257 | -2.14124 | 2.38873e-06 | -8.23866e-06 |
| `open_loop_stress` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.86709 | 3479.46 | 0.0151203 | 0.0699062 |
| `open_loop_stress` | `sensory_feedback` | `lagged_observation_history` | evaluated | 1.97226 | 29187.3 | 0.0561339 | 0.0650161 |
| `open_loop_stress` | `sensory_feedback` | `position_only_observation` | evaluated | 15.9643 | 3.56497e+06 | 0.535624 | 1.97518 |
| `open_loop_stress` | `sensory_feedback` | `velocity_only_observation` | evaluated | 8.3859 | 206897 | 0.158021 | 0.127052 |
| `open_loop_stress` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 244.386 | 0.0019416 | 0.0072125 |
| `open_loop_stress` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `delayed_observation` | `position_only_observation` | evaluated | 16.0157 | 3.56503e+06 | 0.534044 | 1.97509 |
| `open_loop_stress` | `delayed_observation` | `velocity_only_observation` | evaluated | 8.43484 | 206959 | 0.156441 | 0.126959 |
| `closed_loop_small` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_small` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `nominal` | `shuffled_observation_history` | evaluated | 0.199013 | 230.641 | 0.0016555 | 0.00689641 |
| `closed_loop_small` | `nominal` | `lagged_observation_history` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `nominal` | `position_only_observation` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `nominal` | `velocity_only_observation` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_small` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0400805 | -0.67325 | -1.87741e-05 | 8.65041e-05 |
| `closed_loop_small` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0400805 | -0.67325 | -1.87741e-05 | 8.65041e-05 |
| `closed_loop_small` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0400805 | -0.67325 | -1.87741e-05 | 8.65041e-05 |
| `closed_loop_small` | `initial_state` | `lagged_observation_history` | evaluated | 0.0400805 | -0.67325 | -1.87741e-05 | 8.65041e-05 |
| `closed_loop_small` | `initial_state` | `position_only_observation` | evaluated | 0.0400805 | -0.67325 | -1.87741e-05 | 8.65041e-05 |
| `closed_loop_small` | `initial_state` | `velocity_only_observation` | evaluated | 0.0400805 | -0.67325 | -1.87741e-05 | 8.65041e-05 |
| `closed_loop_small` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_small` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.03833 | 2.58279 | -2.78547e-06 | -5.69881e-05 |
| `closed_loop_small` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.03833 | 2.58279 | -2.78547e-06 | -5.69881e-05 |
| `closed_loop_small` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.03833 | 2.58279 | -2.78547e-06 | -5.69881e-05 |
| `closed_loop_small` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.401942 | 18410.5 | 0.0439916 | 0.0712103 |
| `closed_loop_small` | `process_epsilon` | `position_only_observation` | evaluated | 0.03833 | 2.58279 | -2.78547e-06 | -5.69881e-05 |
| `closed_loop_small` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.03833 | 2.58279 | -2.78547e-06 | -5.69881e-05 |
| `closed_loop_small` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_small` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0403746 | -2.5419 | 1.5898e-05 | 3.9757e-06 |
| `closed_loop_small` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0403746 | -2.5419 | 1.5898e-05 | 3.9757e-06 |
| `closed_loop_small` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.202048 | 265.847 | 0.00330104 | 0.00902249 |
| `closed_loop_small` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.415943 | 15594.1 | 0.038354 | 0.0699909 |
| `closed_loop_small` | `sensory_feedback` | `position_only_observation` | evaluated | 5.63761 | 8.76134e+06 | 0.878707 | 2.58544 |
| `closed_loop_small` | `sensory_feedback` | `velocity_only_observation` | evaluated | 4.47241 | 420720 | 0.18906 | 0.352074 |
| `closed_loop_small` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_small` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 227.019 | 0.00174551 | 0.00696515 |
| `closed_loop_small` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0405209 | -2.13887 | 1.87553e-05 | 2.29061e-05 |
| `closed_loop_small` | `delayed_observation` | `position_only_observation` | evaluated | 5.63971 | 8.76135e+06 | 0.877935 | 2.58571 |
| `closed_loop_small` | `delayed_observation` | `velocity_only_observation` | evaluated | 4.48639 | 420728 | 0.188288 | 0.352338 |
| `closed_loop_moderate` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_moderate` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `nominal` | `shuffled_observation_history` | evaluated | 0.295763 | 234.75 | 0.00157368 | 0.00725151 |
| `closed_loop_moderate` | `nominal` | `lagged_observation_history` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `nominal` | `position_only_observation` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `nominal` | `velocity_only_observation` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_moderate` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0486697 | -0.500136 | -1.96698e-05 | -1.4063e-05 |
| `closed_loop_moderate` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0486697 | -0.500136 | -1.96698e-05 | -1.4063e-05 |
| `closed_loop_moderate` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0486697 | -0.500136 | -1.96698e-05 | -1.4063e-05 |
| `closed_loop_moderate` | `initial_state` | `lagged_observation_history` | evaluated | 0.0486697 | -0.500136 | -1.96698e-05 | -1.4063e-05 |
| `closed_loop_moderate` | `initial_state` | `position_only_observation` | evaluated | 0.0486697 | -0.500136 | -1.96698e-05 | -1.4063e-05 |
| `closed_loop_moderate` | `initial_state` | `velocity_only_observation` | evaluated | 0.0486697 | -0.500136 | -1.96698e-05 | -1.4063e-05 |
| `closed_loop_moderate` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_moderate` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0455806 | 2.088 | -4.29947e-06 | 9.79645e-06 |
| `closed_loop_moderate` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0455806 | 2.088 | -4.29947e-06 | 9.79645e-06 |
| `closed_loop_moderate` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0455806 | 2.088 | -4.29947e-06 | 9.79645e-06 |
| `closed_loop_moderate` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.472954 | 18924.9 | 0.0442855 | 0.0785842 |
| `closed_loop_moderate` | `process_epsilon` | `position_only_observation` | evaluated | 0.0455806 | 2.088 | -4.29947e-06 | 9.79645e-06 |
| `closed_loop_moderate` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0455806 | 2.088 | -4.29947e-06 | 9.79645e-06 |
| `closed_loop_moderate` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_moderate` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0486735 | -2.1423 | 8.18009e-06 | -2.17364e-05 |
| `closed_loop_moderate` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0486735 | -2.1423 | 8.18009e-06 | -2.17364e-05 |
| `closed_loop_moderate` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.297197 | 263.661 | 0.00250041 | 0.00859385 |
| `closed_loop_moderate` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.481996 | 16685.8 | 0.0392274 | 0.0743963 |
| `closed_loop_moderate` | `sensory_feedback` | `position_only_observation` | evaluated | 7.77444 | 6.1696e+06 | 0.77584 | 1.29182 |
| `closed_loop_moderate` | `sensory_feedback` | `velocity_only_observation` | evaluated | 5.28584 | 323237 | 0.155132 | 0.253315 |
| `closed_loop_moderate` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_moderate` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 233.272 | 0.00168701 | 0.00741344 |
| `closed_loop_moderate` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0486569 | -1.87763 | 9.64148e-06 | -1.67399e-05 |
| `closed_loop_moderate` | `delayed_observation` | `position_only_observation` | evaluated | 7.77463 | 6.1696e+06 | 0.775505 | 1.29189 |
| `closed_loop_moderate` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.28966 | 323238 | 0.154797 | 0.253383 |
| `closed_loop_stress` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_stress` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `nominal` | `shuffled_observation_history` | evaluated | 0.561002 | 239.237 | 0.00170308 | 0.00739918 |
| `closed_loop_stress` | `nominal` | `lagged_observation_history` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `nominal` | `position_only_observation` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `nominal` | `velocity_only_observation` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_stress` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0655833 | -0.727296 | 2.42147e-06 | 1.37923e-06 |
| `closed_loop_stress` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0655833 | -0.727296 | 2.42147e-06 | 1.37923e-06 |
| `closed_loop_stress` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0655833 | -0.727296 | 2.42147e-06 | 1.37923e-06 |
| `closed_loop_stress` | `initial_state` | `lagged_observation_history` | evaluated | 0.0655833 | -0.727296 | 2.42147e-06 | 1.37923e-06 |
| `closed_loop_stress` | `initial_state` | `position_only_observation` | evaluated | 0.0655833 | -0.727296 | 2.42147e-06 | 1.37923e-06 |
| `closed_loop_stress` | `initial_state` | `velocity_only_observation` | evaluated | 0.0655833 | -0.727296 | 2.42147e-06 | 1.37923e-06 |
| `closed_loop_stress` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_stress` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0630318 | 2.24557 | 8.47945e-07 | 3.39742e-06 |
| `closed_loop_stress` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0630318 | 2.24557 | 8.47945e-07 | 3.39742e-06 |
| `closed_loop_stress` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0630318 | 2.24557 | 8.47945e-07 | 3.39742e-06 |
| `closed_loop_stress` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.632728 | 26548.3 | 0.0522455 | 0.0986106 |
| `closed_loop_stress` | `process_epsilon` | `position_only_observation` | evaluated | 0.0630318 | 2.24557 | 8.47945e-07 | 3.39742e-06 |
| `closed_loop_stress` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0630318 | 2.24557 | 8.47945e-07 | 3.39742e-06 |
| `closed_loop_stress` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_stress` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0649276 | -1.23766 | 8.56579e-06 | -4.76987e-07 |
| `closed_loop_stress` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0649276 | -1.23766 | 8.56579e-06 | -4.76987e-07 |
| `closed_loop_stress` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.56563 | 283.139 | 0.00282713 | 0.00910215 |
| `closed_loop_stress` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.635679 | 24105.1 | 0.0482155 | 0.093056 |
| `closed_loop_stress` | `sensory_feedback` | `position_only_observation` | evaluated | 12.1733 | 4.1979e+06 | 0.652704 | 0.691151 |
| `closed_loop_stress` | `sensory_feedback` | `velocity_only_observation` | evaluated | 4.03592 | 175244 | 0.146778 | 0.0669979 |
| `closed_loop_stress` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_stress` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 238.821 | 0.00181593 | 0.00761863 |
| `closed_loop_stress` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0650573 | -1.0607 | 9.12498e-06 | 6.19043e-07 |
| `closed_loop_stress` | `delayed_observation` | `position_only_observation` | evaluated | 12.1744 | 4.1979e+06 | 0.652439 | 0.691172 |
| `closed_loop_stress` | `delayed_observation` | `velocity_only_observation` | evaluated | 4.04443 | 175245 | 0.146513 | 0.0670187 |
| `closed_loop_cmd_lateral_small` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_small` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `nominal` | `shuffled_observation_history` | evaluated | 0.253837 | 232.254 | 0.00166451 | 0.00713403 |
| `closed_loop_cmd_lateral_small` | `nominal` | `lagged_observation_history` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `nominal` | `position_only_observation` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `nominal` | `velocity_only_observation` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0472798 | -0.747371 | -5.5183e-06 | -1.64883e-05 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0472798 | -0.747371 | -5.5183e-06 | -1.64883e-05 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0472798 | -0.747371 | -5.5183e-06 | -1.64883e-05 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `lagged_observation_history` | evaluated | 0.0472798 | -0.747371 | -5.5183e-06 | -1.64883e-05 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `position_only_observation` | evaluated | 0.0472798 | -0.747371 | -5.5183e-06 | -1.64883e-05 |
| `closed_loop_cmd_lateral_small` | `initial_state` | `velocity_only_observation` | evaluated | 0.0472798 | -0.747371 | -5.5183e-06 | -1.64883e-05 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0459884 | 2.1687 | -4.42757e-06 | -1.58517e-05 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0459884 | 2.1687 | -4.42757e-06 | -1.58517e-05 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0459884 | 2.1687 | -4.42757e-06 | -1.58517e-05 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.44248 | 20081.3 | 0.0456607 | 0.0808544 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `position_only_observation` | evaluated | 0.0459884 | 2.1687 | -4.42757e-06 | -1.58517e-05 |
| `closed_loop_cmd_lateral_small` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0459884 | 2.1687 | -4.42757e-06 | -1.58517e-05 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0481583 | -1.98172 | 5.27205e-06 | -1.7763e-05 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0481583 | -1.98172 | 5.27205e-06 | -1.7763e-05 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.257235 | 301.979 | 0.00356719 | 0.0114244 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.455148 | 17204 | 0.0406335 | 0.0715205 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `position_only_observation` | evaluated | 7.56899 | 1.07366e+07 | 0.980991 | 2.53274 |
| `closed_loop_cmd_lateral_small` | `sensory_feedback` | `velocity_only_observation` | evaluated | 5.55576 | 369407 | 0.149722 | 0.540944 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 233.173 | 0.00179306 | 0.00737152 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0481891 | -1.77892 | 6.87792e-06 | -2.27585e-05 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `position_only_observation` | evaluated | 7.57363 | 1.07366e+07 | 0.980308 | 2.53308 |
| `closed_loop_cmd_lateral_small` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.57396 | 369414 | 0.149039 | 0.541287 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `shuffled_observation_history` | evaluated | 0.256789 | 231.499 | 0.00164862 | 0.00687286 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `lagged_observation_history` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `position_only_observation` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `nominal` | `velocity_only_observation` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.052802 | -0.876134 | -2.17973e-05 | -8.45942e-06 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.052802 | -0.876134 | -2.17973e-05 | -8.45942e-06 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `shuffled_observation_history` | evaluated | 0.052802 | -0.876134 | -2.17973e-05 | -8.45942e-06 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `lagged_observation_history` | evaluated | 0.052802 | -0.876134 | -2.17973e-05 | -8.45942e-06 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `position_only_observation` | evaluated | 0.052802 | -0.876134 | -2.17973e-05 | -8.45942e-06 |
| `closed_loop_cmd_lateral_moderate` | `initial_state` | `velocity_only_observation` | evaluated | 0.052802 | -0.876134 | -2.17973e-05 | -8.45942e-06 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0505925 | 2.06531 | -3.26757e-06 | 2.18018e-05 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0505925 | 2.06531 | -3.26757e-06 | 2.18018e-05 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0505925 | 2.06531 | -3.26757e-06 | 2.18018e-05 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.480558 | 21787.7 | 0.0470968 | 0.0857864 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `position_only_observation` | evaluated | 0.0505925 | 2.06531 | -3.26757e-06 | 2.18018e-05 |
| `closed_loop_cmd_lateral_moderate` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0505925 | 2.06531 | -3.26757e-06 | 2.18018e-05 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0536594 | -2.29296 | -4.5834e-06 | -5.1993e-06 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0536594 | -2.29296 | -4.5834e-06 | -5.1993e-06 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.260939 | 315.012 | 0.00338366 | 0.0108533 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.484569 | 19126.1 | 0.0428719 | 0.0768668 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `position_only_observation` | evaluated | 8.871 | 1.16861e+07 | 1.02446 | 2.45757 |
| `closed_loop_cmd_lateral_moderate` | `sensory_feedback` | `velocity_only_observation` | evaluated | 7.98889 | 1.06081e+06 | 0.167877 | 1.12836 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 231.688 | 0.00177277 | 0.00706946 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.053242 | -2.08572 | 2.58252e-07 | -1.38966e-05 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `position_only_observation` | evaluated | 8.87235 | 1.16861e+07 | 1.02396 | 2.45798 |
| `closed_loop_cmd_lateral_moderate` | `delayed_observation` | `velocity_only_observation` | evaluated | 7.99893 | 1.06082e+06 | 0.16738 | 1.12877 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `shuffled_observation_history` | evaluated | 0.61028 | 243.327 | 0.00165028 | 0.00703947 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `lagged_observation_history` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `position_only_observation` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `nominal` | `velocity_only_observation` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.10518 | -0.833768 | 2.2424e-06 | 6.30745e-06 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.10518 | -0.833768 | 2.2424e-06 | 6.30745e-06 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `shuffled_observation_history` | evaluated | 0.10518 | -0.833768 | 2.2424e-06 | 6.30745e-06 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `lagged_observation_history` | evaluated | 0.10518 | -0.833768 | 2.2424e-06 | 6.30745e-06 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `position_only_observation` | evaluated | 0.10518 | -0.833768 | 2.2424e-06 | 6.30745e-06 |
| `closed_loop_cmd_lateral_stress` | `initial_state` | `velocity_only_observation` | evaluated | 0.10518 | -0.833768 | 2.2424e-06 | 6.30745e-06 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.103074 | 2.27608 | -9.08667e-06 | -7.08733e-06 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.103074 | 2.27608 | -9.08667e-06 | -7.08733e-06 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.103074 | 2.27608 | -9.08667e-06 | -7.08733e-06 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.806296 | 42631.2 | 0.0654922 | 0.131286 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `position_only_observation` | evaluated | 0.103074 | 2.27608 | -9.08667e-06 | -7.08733e-06 |
| `closed_loop_cmd_lateral_stress` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.103074 | 2.27608 | -9.08667e-06 | -7.08733e-06 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.101764 | -1.32858 | 1.02283e-05 | -1.76927e-06 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.101764 | -1.32858 | 1.02283e-05 | -1.76927e-06 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.636184 | 929.691 | 0.00732036 | 0.0246528 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.826526 | 33692.1 | 0.058498 | 0.10479 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `position_only_observation` | evaluated | 12.7082 | 6.21698e+06 | 0.796845 | 0.994071 |
| `closed_loop_cmd_lateral_stress` | `sensory_feedback` | `velocity_only_observation` | evaluated | 6.98083 | 313112 | 0.155447 | 0.165586 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 241.624 | 0.00180225 | 0.00726472 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.101309 | -1.22605 | 1.25664e-05 | 9.13314e-06 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `position_only_observation` | evaluated | 12.7607 | 6.217e+06 | 0.79577 | 0.994257 |
| `closed_loop_cmd_lateral_stress` | `delayed_observation` | `velocity_only_observation` | evaluated | 6.9898 | 313129 | 0.154373 | 0.165772 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `open_loop_small` | available | 1.38104 | 2.70957 | 0.0525146 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `open_loop_moderate` | available | 1.6708 | 3.28942 | 0.0521679 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `open_loop_stress` | available | 2.42333 | 4.79415 | 0.0525002 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `closed_loop_small` | available | 0.923137 | 1.79714 | 0.0491324 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `closed_loop_moderate` | available | 1.24003 | 2.42966 | 0.050398 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `closed_loop_stress` | available | 1.88623 | 3.72113 | 0.0513214 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `closed_loop_cmd_lateral_small` | available | 1.22258 | 2.39475 | 0.050408 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `closed_loop_cmd_lateral_moderate` | available | 1.41941 | 2.78878 | 0.0500435 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `closed_loop_cmd_lateral_stress` | available | 1.95205 | 3.85217 | 0.0519364 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `open_loop_small` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `open_loop_moderate` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `open_loop_stress` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `closed_loop_small` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `closed_loop_moderate` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `closed_loop_stress` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `closed_loop_cmd_lateral_small` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `closed_loop_cmd_lateral_moderate` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `closed_loop_cmd_lateral_stress` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 0 | 10000 | 3000 | -7000 | 291.249 | 6 |
| `open_loop_small` | 1 | 12000 | 3000 | -9000 | 293.367 | 6 |
| `open_loop_small` | 2 | 9500 | 2500 | -7000 | 304.661 | 6 |
| `open_loop_small` | 3 | 11500 | 4000 | -7500 | 303.524 | 6 |
| `open_loop_small` | 4 | 12000 | 2500 | -9500 | 286.306 | 6 |
| `open_loop_moderate` | 0 | 10500 | 2000 | -8500 | 278.379 | 6 |
| `open_loop_moderate` | 1 | 11500 | 4000 | -7500 | 297.102 | 6 |
| `open_loop_moderate` | 2 | 12000 | 2500 | -9500 | 279.278 | 6 |
| `open_loop_moderate` | 3 | 12000 | 2500 | -9500 | 301.125 | 6 |
| `open_loop_moderate` | 4 | 10000 | 2500 | -7500 | 307.909 | 6 |
| `open_loop_stress` | 0 | 10000 | 3500 | -6500 | 320.85 | 6 |
| `open_loop_stress` | 1 | 11500 | 1500 | -10000 | 296.53 | 6 |
| `open_loop_stress` | 2 | 12000 | 2000 | -10000 | 318.707 | 6 |
| `open_loop_stress` | 3 | 12000 | 10000 | -2000 | 321.358 | 6 |
| `open_loop_stress` | 4 | 10000 | 1500 | -8500 | 278.789 | 6 |
| `closed_loop_small` | 0 | 10500 | 10000 | -500 | 310.389 | 6 |
| `closed_loop_small` | 1 | 11500 | 7500 | -4000 | 301.863 | 6 |
| `closed_loop_small` | 2 | 12000 | 8000 | -4000 | 317.242 | 6 |
| `closed_loop_small` | 3 | 11500 | 9500 | -2000 | 326.121 | 6 |
| `closed_loop_small` | 4 | 12000 | 6500 | -5500 | 316.318 | 6 |
| `closed_loop_moderate` | 0 | 12000 | 5000 | -7000 | 268.536 | 6 |
| `closed_loop_moderate` | 1 | 11500 | 6500 | -5000 | 301.73 | 6 |
| `closed_loop_moderate` | 2 | 12000 | 7000 | -5000 | 297.135 | 6 |
| `closed_loop_moderate` | 3 | 11500 | 11000 | -500 | 305.953 | 6 |
| `closed_loop_moderate` | 4 | 10000 | 8000 | -2000 | 309.353 | 6 |
| `closed_loop_stress` | 0 | 10000 | 3000 | -7000 | 303.941 | 6 |
| `closed_loop_stress` | 1 | 11500 | 8000 | -3500 | 307.142 | 6 |
| `closed_loop_stress` | 2 | 12000 | 2500 | -9500 | 283.745 | 6 |
| `closed_loop_stress` | 3 | 12000 | 3000 | -9000 | 299.627 | 6 |
| `closed_loop_stress` | 4 | 10500 | 2500 | -8000 | 290.418 | 6 |
| `closed_loop_cmd_lateral_small` | 0 | 10000 | 11500 | 1500 | 304.709 | 6 |
| `closed_loop_cmd_lateral_small` | 1 | 11500 | 7500 | -4000 | 296.816 | 6 |
| `closed_loop_cmd_lateral_small` | 2 | 12000 | 9000 | -3000 | 297.496 | 6 |
| `closed_loop_cmd_lateral_small` | 3 | 11500 | 5500 | -6000 | 296.68 | 6 |
| `closed_loop_cmd_lateral_small` | 4 | 12000 | 8000 | -4000 | 284.889 | 6 |
| `closed_loop_cmd_lateral_moderate` | 0 | 12000 | 5000 | -7000 | 281.503 | 6 |
| `closed_loop_cmd_lateral_moderate` | 1 | 11500 | 5500 | -6000 | 286.018 | 6 |
| `closed_loop_cmd_lateral_moderate` | 2 | 12000 | 9000 | -3000 | 293.352 | 6 |
| `closed_loop_cmd_lateral_moderate` | 3 | 11500 | 7000 | -4500 | 300.493 | 6 |
| `closed_loop_cmd_lateral_moderate` | 4 | 12000 | 6500 | -5500 | 298.688 | 6 |
| `closed_loop_cmd_lateral_stress` | 0 | 12000 | 6000 | -6000 | 313.714 | 6 |
| `closed_loop_cmd_lateral_stress` | 1 | 11500 | 2500 | -9000 | 270.42 | 6 |
| `closed_loop_cmd_lateral_stress` | 2 | 11500 | 5000 | -6500 | 303.663 | 6 |
| `closed_loop_cmd_lateral_stress` | 3 | 11500 | 3000 | -8500 | 306.199 | 6 |
| `closed_loop_cmd_lateral_stress` | 4 | 12000 | 4500 | -7500 | 304.68 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
