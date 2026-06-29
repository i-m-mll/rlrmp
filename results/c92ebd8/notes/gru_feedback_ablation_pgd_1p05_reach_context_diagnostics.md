# GRU Feedback Ablation Diagnostic

- Issue: `c92ebd8`
- Source experiment: `c92ebd8`
- Scope: `pgd_1p05_reach_context_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `open_loop_small` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `open_loop_moderate` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `open_loop_stress` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `small` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `moderate` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `stress` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `open_loop_small` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 243.447 | 0.00184131 | 0.00659934 |
| `open_loop_small` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 243.447 | 0.00184131 | 0.00659934 |
| `open_loop_small` | `nominal` | `shuffled_observation_history` | evaluated | 0.286758 | 240.592 | 0.00169862 | 0.00656866 |
| `open_loop_small` | `nominal` | `lagged_observation_history` | evaluated | 0.476493 | 16437.8 | 0.0385971 | 0.0739537 |
| `open_loop_small` | `nominal` | `position_only_observation` | evaluated | 8.54165 | 4.75512e+06 | 0.699551 | 0.71403 |
| `open_loop_small` | `nominal` | `velocity_only_observation` | evaluated | 3.4932 | 176572 | 0.139715 | 0.142992 |
| `open_loop_small` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.154761 | 1657.8 | 0.0106374 | 0.000878397 |
| `open_loop_small` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.154761 | 1657.8 | 0.0106374 | 0.000878397 |
| `open_loop_small` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0410962 | -1.34365 | -1.25854e-05 | -9.50103e-06 |
| `open_loop_small` | `initial_state` | `lagged_observation_history` | evaluated | 0.479282 | 17984.5 | 0.0447502 | 0.0601867 |
| `open_loop_small` | `initial_state` | `position_only_observation` | evaluated | 8.38327 | 4.25274e+06 | 0.670837 | 0.565403 |
| `open_loop_small` | `initial_state` | `velocity_only_observation` | evaluated | 3.48415 | 145132 | 0.126358 | 0.137271 |
| `open_loop_small` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.34369 | 2560.25 | 0.0105394 | 0.0358261 |
| `open_loop_small` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.34369 | 2560.25 | 0.0105394 | 0.0358261 |
| `open_loop_small` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.286037 | 216.384 | 0.00267924 | 0.00483417 |
| `open_loop_small` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.477498 | 17318.7 | 0.0422417 | 0.0682448 |
| `open_loop_small` | `process_epsilon` | `position_only_observation` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0395754 | 1.85464 | -8.70998e-06 | 3.28595e-05 |
| `open_loop_small` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.089784 | 212.751 | 0.00311301 | 0.00579739 |
| `open_loop_small` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.089784 | 212.751 | 0.00311301 | 0.00579739 |
| `open_loop_small` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.303038 | 751.73 | 0.00697501 | 0.0220231 |
| `open_loop_small` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.508249 | 12126.7 | 0.0345258 | 0.0474976 |
| `open_loop_small` | `sensory_feedback` | `position_only_observation` | evaluated | 8.51928 | 4.75509e+06 | 0.700822 | 0.713228 |
| `open_loop_small` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.4405 | 176541 | 0.140987 | 0.14219 |
| `open_loop_small` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_small` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 243.447 | 0.00184131 | 0.00659934 |
| `open_loop_small` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0415796 | -1.93066 | 1.0703e-05 | -1.88684e-05 |
| `open_loop_small` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.476493 | 16437.8 | 0.0385971 | 0.0739537 |
| `open_loop_small` | `delayed_observation` | `position_only_observation` | evaluated | 8.54165 | 4.75512e+06 | 0.699551 | 0.71403 |
| `open_loop_small` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.4932 | 176572 | 0.139715 | 0.142992 |
| `open_loop_moderate` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 241.781 | 0.00183557 | 0.00700159 |
| `open_loop_moderate` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 241.781 | 0.00183557 | 0.00700159 |
| `open_loop_moderate` | `nominal` | `shuffled_observation_history` | evaluated | 0.373564 | 239.267 | 0.00170832 | 0.00686772 |
| `open_loop_moderate` | `nominal` | `lagged_observation_history` | evaluated | 0.539068 | 20337.1 | 0.0434558 | 0.0867 |
| `open_loop_moderate` | `nominal` | `position_only_observation` | evaluated | 10.5669 | 3.863e+06 | 0.639084 | 0.151965 |
| `open_loop_moderate` | `nominal` | `velocity_only_observation` | evaluated | 3.3271 | 165424 | 0.141836 | 0.104877 |
| `open_loop_moderate` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.251976 | 1832.88 | 0.0111012 | 0.0060893 |
| `open_loop_moderate` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.251976 | 1832.88 | 0.0111012 | 0.0060893 |
| `open_loop_moderate` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `lagged_observation_history` | evaluated | 0.536749 | 21279.9 | 0.0483413 | 0.0825457 |
| `open_loop_moderate` | `initial_state` | `position_only_observation` | evaluated | 10.2257 | 3.3644e+06 | 0.601586 | 0.258482 |
| `open_loop_moderate` | `initial_state` | `velocity_only_observation` | evaluated | 3.3064 | 134186 | 0.128925 | 0.103964 |
| `open_loop_moderate` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.430408 | 2585.08 | 0.0103209 | 0.0391716 |
| `open_loop_moderate` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.430408 | 2585.08 | 0.0103209 | 0.0391716 |
| `open_loop_moderate` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.373244 | 216.361 | 0.00253684 | 0.00592751 |
| `open_loop_moderate` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.54353 | 21223.4 | 0.0466434 | 0.0843124 |
| `open_loop_moderate` | `process_epsilon` | `position_only_observation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.159452 | 195.634 | 0.00339698 | 0.0066637 |
| `open_loop_moderate` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.159452 | 195.634 | 0.00339698 | 0.0066637 |
| `open_loop_moderate` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.399095 | 1248.78 | 0.00965899 | 0.0307207 |
| `open_loop_moderate` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.585 | 13582.9 | 0.0372144 | 0.0517543 |
| `open_loop_moderate` | `sensory_feedback` | `position_only_observation` | evaluated | 10.5244 | 3.86295e+06 | 0.640646 | 0.151627 |
| `open_loop_moderate` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.27381 | 165378 | 0.143397 | 0.104539 |
| `open_loop_moderate` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 241.781 | 0.00183557 | 0.00700159 |
| `open_loop_moderate` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.539068 | 20337.1 | 0.0434558 | 0.0867 |
| `open_loop_moderate` | `delayed_observation` | `position_only_observation` | evaluated | 10.5669 | 3.863e+06 | 0.639084 | 0.151965 |
| `open_loop_moderate` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.3271 | 165424 | 0.141836 | 0.104877 |
| `open_loop_stress` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 244.386 | 0.0019416 | 0.0072125 |
| `open_loop_stress` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 244.386 | 0.0019416 | 0.0072125 |
| `open_loop_stress` | `nominal` | `shuffled_observation_history` | evaluated | 0.678391 | 247.227 | 0.00180722 | 0.00716426 |
| `open_loop_stress` | `nominal` | `lagged_observation_history` | evaluated | 1.02015 | 45096.4 | 0.0675362 | 0.135385 |
| `open_loop_stress` | `nominal` | `position_only_observation` | evaluated | 16.0157 | 3.56503e+06 | 0.534044 | 1.97509 |
| `open_loop_stress` | `nominal` | `velocity_only_observation` | evaluated | 8.43484 | 206959 | 0.156441 | 0.126959 |
| `open_loop_stress` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.389573 | 1919.31 | 0.00969053 | 0.00710992 |
| `open_loop_stress` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.389573 | 1919.31 | 0.00969053 | 0.00710992 |
| `open_loop_stress` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0722219 | -0.967472 | 5.65134e-06 | -1.53976e-05 |
| `open_loop_stress` | `initial_state` | `lagged_observation_history` | evaluated | 1.26047 | 51435 | 0.0737715 | 0.145274 |
| `open_loop_stress` | `initial_state` | `position_only_observation` | evaluated | 15.3199 | 3.07448e+06 | 0.479137 | 2.00352 |
| `open_loop_stress` | `initial_state` | `velocity_only_observation` | evaluated | 8.28372 | 173725 | 0.142219 | 0.126856 |
| `open_loop_stress` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.720576 | 2592.52 | 0.0100479 | 0.0393741 |
| `open_loop_stress` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.720576 | 2592.52 | 0.0100479 | 0.0393741 |
| `open_loop_stress` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.679656 | 217.177 | 0.00242899 | 0.00640066 |
| `open_loop_stress` | `process_epsilon` | `lagged_observation_history` | evaluated | 1.03757 | 46282.4 | 0.0702071 | 0.135682 |
| `open_loop_stress` | `process_epsilon` | `position_only_observation` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0754327 | 1.81954 | -1.42371e-05 | -2.41031e-06 |
| `open_loop_stress` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.30359 | 182.57 | 0.00352154 | 0.00730603 |
| `open_loop_stress` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.30359 | 182.57 | 0.00352154 | 0.00730603 |
| `open_loop_stress` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.86705 | 3479.25 | 0.0151198 | 0.069904 |
| `open_loop_stress` | `sensory_feedback` | `lagged_observation_history` | evaluated | 1.97222 | 29187.8 | 0.0561343 | 0.065018 |
| `open_loop_stress` | `sensory_feedback` | `position_only_observation` | evaluated | 15.9643 | 3.56497e+06 | 0.535624 | 1.97518 |
| `open_loop_stress` | `sensory_feedback` | `velocity_only_observation` | evaluated | 8.3859 | 206897 | 0.158021 | 0.127052 |
| `open_loop_stress` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 244.386 | 0.0019416 | 0.0072125 |
| `open_loop_stress` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0738091 | -1.47999 | 6.22292e-06 | -4.71446e-06 |
| `open_loop_stress` | `delayed_observation` | `lagged_observation_history` | evaluated | 1.02015 | 45096.4 | 0.0675362 | 0.135385 |
| `open_loop_stress` | `delayed_observation` | `position_only_observation` | evaluated | 16.0157 | 3.56503e+06 | 0.534044 | 1.97509 |
| `open_loop_stress` | `delayed_observation` | `velocity_only_observation` | evaluated | 8.43484 | 206959 | 0.156441 | 0.126959 |
| `small` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `small` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 272.522 | 0.0031039 | 0.00486504 |
| `small` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 272.522 | 0.0031039 | 0.00486504 |
| `small` | `nominal` | `shuffled_observation_history` | evaluated | 0.375528 | 266.304 | 0.00292824 | 0.00452502 |
| `small` | `nominal` | `lagged_observation_history` | evaluated | 0.516022 | 19141.5 | 0.044667 | 0.0692025 |
| `small` | `nominal` | `position_only_observation` | evaluated | 10.4124 | 3.55823e+06 | 0.611667 | 0.449846 |
| `small` | `nominal` | `velocity_only_observation` | evaluated | 3.43741 | 167092 | 0.147763 | 0.00878115 |
| `small` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `small` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.209335 | 2167.27 | 0.0122642 | -0.000626512 |
| `small` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.209335 | 2167.27 | 0.0122642 | -0.000626512 |
| `small` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0501398 | 0.195751 | -1.08204e-05 | 3.37196e-05 |
| `small` | `initial_state` | `lagged_observation_history` | evaluated | 0.518942 | 19629.5 | 0.0467782 | 0.0579638 |
| `small` | `initial_state` | `position_only_observation` | evaluated | 10.1009 | 3.07139e+06 | 0.568706 | 0.496012 |
| `small` | `initial_state` | `velocity_only_observation` | evaluated | 3.40384 | 134879 | 0.132311 | 0.0032896 |
| `small` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `small` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.447731 | 2625.15 | 0.0129446 | 0.0288876 |
| `small` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.447731 | 2625.15 | 0.0129446 | 0.0288876 |
| `small` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.371408 | 242.276 | 0.0030899 | 0.00385063 |
| `small` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.518304 | 19458.2 | 0.0461876 | 0.0655654 |
| `small` | `process_epsilon` | `position_only_observation` | evaluated | 0.0483315 | 1.29367 | -2.88804e-06 | 1.75555e-06 |
| `small` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0483315 | 1.29367 | -2.88804e-06 | 1.75555e-06 |
| `small` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `small` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.191715 | 97.3862 | 0.00310902 | 0.00311073 |
| `small` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.191715 | 97.3862 | 0.00310902 | 0.00311073 |
| `small` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.404902 | 1826.01 | 0.00907928 | 0.0359267 |
| `small` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.568691 | 11886.8 | 0.0353277 | 0.0265445 |
| `small` | `sensory_feedback` | `position_only_observation` | evaluated | 10.3247 | 3.55806e+06 | 0.611672 | 0.448092 |
| `small` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.37345 | 166917 | 0.147768 | 0.00702684 |
| `small` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `small` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0507442 | -1.85682 | 1.13477e-05 | 2.31073e-05 |
| `small` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 272.522 | 0.0031039 | 0.00486504 |
| `small` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0507442 | -1.85682 | 1.13477e-05 | 2.31073e-05 |
| `small` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.516022 | 19141.5 | 0.044667 | 0.0692025 |
| `small` | `delayed_observation` | `position_only_observation` | evaluated | 10.4124 | 3.55823e+06 | 0.611667 | 0.449846 |
| `small` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.43741 | 167092 | 0.147763 | 0.00878115 |
| `moderate` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 282.087 | 0.00333262 | 0.00412715 |
| `moderate` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 282.087 | 0.00333262 | 0.00412715 |
| `moderate` | `nominal` | `shuffled_observation_history` | evaluated | 0.454076 | 279.028 | 0.00325786 | 0.00406245 |
| `moderate` | `nominal` | `lagged_observation_history` | evaluated | 0.559578 | 22175.1 | 0.0485308 | 0.0707839 |
| `moderate` | `nominal` | `position_only_observation` | evaluated | 11.774 | 3.66908e+06 | 0.605508 | 0.661372 |
| `moderate` | `nominal` | `velocity_only_observation` | evaluated | 3.46652 | 166451 | 0.147595 | 0.0114356 |
| `moderate` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.271411 | 2346.84 | 0.0125932 | -0.00131416 |
| `moderate` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.271411 | 2346.84 | 0.0125932 | -0.00131416 |
| `moderate` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0559146 | 1.09879 | 2.00668e-06 | 1.20921e-05 |
| `moderate` | `initial_state` | `lagged_observation_history` | evaluated | 0.556963 | 22397.2 | 0.0498609 | 0.0588239 |
| `moderate` | `initial_state` | `position_only_observation` | evaluated | 11.368 | 3.16475e+06 | 0.557305 | 0.771465 |
| `moderate` | `initial_state` | `velocity_only_observation` | evaluated | 3.38339 | 134290 | 0.131904 | 0.00599432 |
| `moderate` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.529709 | 2596.55 | 0.0133375 | 0.0224461 |
| `moderate` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.529709 | 2596.55 | 0.0133375 | 0.0224461 |
| `moderate` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.452778 | 240.561 | 0.00306273 | 0.00299961 |
| `moderate` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.566664 | 22261.5 | 0.049489 | 0.0650882 |
| `moderate` | `process_epsilon` | `position_only_observation` | evaluated | 0.0543958 | 2.13547 | 1.08684e-05 | 3.57451e-06 |
| `moderate` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0543958 | 2.13547 | 1.08684e-05 | 3.57451e-06 |
| `moderate` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.266616 | 54.6145 | 0.00301002 | 0.00146717 |
| `moderate` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.266616 | 54.6145 | 0.00301002 | 0.00146717 |
| `moderate` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.491813 | 2772.22 | 0.0109677 | 0.046055 |
| `moderate` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.629825 | 12637.4 | 0.0364888 | 0.0194995 |
| `moderate` | `sensory_feedback` | `position_only_observation` | evaluated | 11.6588 | 3.66885e+06 | 0.605186 | 0.658712 |
| `moderate` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.43888 | 166223 | 0.147272 | 0.00877565 |
| `moderate` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.057065 | -1.40284 | 8.47445e-06 | -7.00359e-06 |
| `moderate` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 282.087 | 0.00333262 | 0.00412715 |
| `moderate` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.057065 | -1.40284 | 8.47445e-06 | -7.00359e-06 |
| `moderate` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.559578 | 22175.1 | 0.0485308 | 0.0707839 |
| `moderate` | `delayed_observation` | `position_only_observation` | evaluated | 11.774 | 3.66908e+06 | 0.605508 | 0.661372 |
| `moderate` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.46652 | 166451 | 0.147595 | 0.0114356 |
| `stress` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `stress` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 284.253 | 0.00347528 | 0.00674597 |
| `stress` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 284.253 | 0.00347528 | 0.00674597 |
| `stress` | `nominal` | `shuffled_observation_history` | evaluated | 0.927108 | 284.693 | 0.00347428 | 0.00657617 |
| `stress` | `nominal` | `lagged_observation_history` | evaluated | 1.15303 | 55858.3 | 0.0780183 | 0.145841 |
| `stress` | `nominal` | `position_only_observation` | evaluated | 18.2101 | 4.29496e+06 | 0.431067 | 3.06042 |
| `stress` | `nominal` | `velocity_only_observation` | evaluated | 7.50583 | 176654 | 0.150659 | 0.0641793 |
| `stress` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `stress` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.420822 | 2694.06 | 0.0139897 | 0.0068538 |
| `stress` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.420822 | 2694.06 | 0.0139897 | 0.0068538 |
| `stress` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0779891 | -0.542183 | -6.58333e-06 | 2.45188e-06 |
| `stress` | `initial_state` | `lagged_observation_history` | evaluated | 1.2095 | 60090.5 | 0.0813096 | 0.153296 |
| `stress` | `initial_state` | `position_only_observation` | evaluated | 17.124 | 3.81284e+06 | 0.370138 | 3.02791 |
| `stress` | `initial_state` | `velocity_only_observation` | evaluated | 7.32294 | 144708 | 0.135752 | 0.0642871 |
| `stress` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `stress` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.979932 | 2999.99 | 0.0138647 | 0.0384606 |
| `stress` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.979932 | 2999.99 | 0.0138647 | 0.0384606 |
| `stress` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.943938 | 250.411 | 0.00315346 | 0.00599255 |
| `stress` | `process_epsilon` | `lagged_observation_history` | evaluated | 1.16215 | 56159.8 | 0.078544 | 0.145772 |
| `stress` | `process_epsilon` | `position_only_observation` | evaluated | 0.0821949 | 2.26791 | 8.40957e-06 | -1.06646e-05 |
| `stress` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0821949 | 2.26791 | 8.40957e-06 | -1.06646e-05 |
| `stress` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `stress` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.515551 | 26.6665 | 0.00289032 | 0.00534269 |
| `stress` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.515551 | 26.6665 | 0.00289032 | 0.00534269 |
| `stress` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.44842 | 6092.83 | 0.0190131 | 0.0735928 |
| `stress` | `sensory_feedback` | `lagged_observation_history` | evaluated | 1.6532 | 29734.3 | 0.0573125 | 0.0747859 |
| `stress` | `sensory_feedback` | `position_only_observation` | evaluated | 18.1208 | 4.29471e+06 | 0.430482 | 3.05902 |
| `stress` | `sensory_feedback` | `velocity_only_observation` | evaluated | 7.48678 | 176396 | 0.150075 | 0.062776 |
| `stress` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `stress` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0811133 | -1.37375 | -7.56514e-06 | 7.72186e-06 |
| `stress` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 284.253 | 0.00347528 | 0.00674597 |
| `stress` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0811133 | -1.37375 | -7.56514e-06 | 7.72186e-06 |
| `stress` | `delayed_observation` | `lagged_observation_history` | evaluated | 1.15303 | 55858.3 | 0.0780183 | 0.145841 |
| `stress` | `delayed_observation` | `position_only_observation` | evaluated | 18.2101 | 4.29496e+06 | 0.431067 | 3.06042 |
| `stress` | `delayed_observation` | `velocity_only_observation` | evaluated | 7.50583 | 176654 | 0.150659 | 0.0641793 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `open_loop_small` | available | 1.53574 | 2.70957 | 0.361914 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `open_loop_moderate` | available | 1.82735 | 3.28942 | 0.365267 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `open_loop_stress` | available | 2.62846 | 4.89101 | 0.365913 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `small` | available | 1.7035 | 3.0522 | 0.3548 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `moderate` | available | 1.87724 | 3.3932 | 0.361289 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `stress` | available | 2.74641 | 5.07791 | 0.41491 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `open_loop_small` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `open_loop_moderate` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `open_loop_stress` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `small` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `moderate` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `stress` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 0 | 10000 | 3000 | -7000 | 291.249 | 6 |
| `open_loop_small` | 1 | 12000 | 3000 | -9000 | 293.366 | 6 |
| `open_loop_small` | 2 | 9500 | 2500 | -7000 | 304.661 | 6 |
| `open_loop_small` | 3 | 11500 | 4000 | -7500 | 303.523 | 6 |
| `open_loop_small` | 4 | 12000 | 2500 | -9500 | 286.305 | 6 |
| `open_loop_moderate` | 0 | 10500 | 2000 | -8500 | 278.379 | 6 |
| `open_loop_moderate` | 1 | 11500 | 4000 | -7500 | 297.101 | 6 |
| `open_loop_moderate` | 2 | 12000 | 2500 | -9500 | 279.278 | 6 |
| `open_loop_moderate` | 3 | 12000 | 2500 | -9500 | 301.125 | 6 |
| `open_loop_moderate` | 4 | 10000 | 2500 | -7500 | 307.908 | 6 |
| `open_loop_stress` | 0 | 10000 | 3500 | -6500 | 320.849 | 6 |
| `open_loop_stress` | 1 | 11500 | 1500 | -10000 | 296.528 | 6 |
| `open_loop_stress` | 2 | 12000 | 2000 | -10000 | 318.706 | 6 |
| `open_loop_stress` | 3 | 12000 | 10000 | -2000 | 321.357 | 6 |
| `open_loop_stress` | 4 | 10000 | 1500 | -8500 | 278.788 | 6 |
| `small` | 0 | 5000 | 3000 | -2000 | 283.06 | 6 |
| `small` | 1 | 2500 | 2000 | -500 | 266.854 | 6 |
| `small` | 2 | 3500 | 2500 | -1000 | 277.604 | 6 |
| `small` | 3 | 3500 | 2500 | -1000 | 285.57 | 6 |
| `small` | 4 | 4500 | 2500 | -2000 | 285.775 | 6 |
| `moderate` | 0 | 2500 | 2500 | 0 | 291.174 | 6 |
| `moderate` | 1 | 3500 | 2500 | -1000 | 305.434 | 6 |
| `moderate` | 2 | 3500 | 2000 | -1500 | 286.896 | 6 |
| `moderate` | 3 | 2500 | 2000 | -500 | 286.491 | 6 |
| `moderate` | 4 | 10000 | 2000 | -8000 | 270.858 | 6 |
| `stress` | 0 | 10500 | 1000 | -9500 | 269.518 | 6 |
| `stress` | 1 | 7500 | 1000 | -6500 | 305.635 | 6 |
| `stress` | 2 | 3500 | 1000 | -2500 | 332.012 | 6 |
| `stress` | 3 | 12000 | 1000 | -11000 | 276.705 | 6 |
| `stress` | 4 | 8500 | 2000 | -6500 | 326.557 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
