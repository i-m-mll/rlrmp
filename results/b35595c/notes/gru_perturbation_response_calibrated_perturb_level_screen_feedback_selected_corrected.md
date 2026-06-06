# GRU perturbation-response bank

Issue: `b35595c`. Source experiment: `b35595c`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 36 |
| `delayed_observation` | 72 |
| `initial_state` | 24 |
| `process_epsilon` | 144 |
| `sensory_feedback` | 72 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 36 |
| `delayed_observation_offset` | 72 |
| `initial_position_offset` | 12 |
| `initial_velocity_offset` | 12 |
| `process_epsilon_force_state_xy` | 36 |
| `process_epsilon_integrator_xy` | 36 |
| `process_epsilon_position_xy` | 36 |
| `process_epsilon_velocity_xy` | 36 |
| `sensory_feedback_offset` | 72 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.168495 | 0.00890585 | 0.00176913 | 0.059038 | 0.0109428 | 0.862001 | 0.101097 | 0.443993 | NA | 0.00482297 | 0.0110053 | 2525.16 | 4.88142 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.27311 | 0.0040274 | 0.000909298 | 0.0382969 | 0.00565052 | 1.64084 | 0.163866 | 0.440927 | 0.550947 | 0.000951014 | 0.00287044 | 394.432 | 1.39102 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.268316 | 0.02 | 0.00655591 | 0.0703657 | 0.0199416 | 1.17472 | 0.160989 | 0.01325 | NA | 0.000797773 | 0.00398936 | 510.546 | 2.43414 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.157393 | 0.00448583 | 0.00122098 | 0.0342941 | 0.00851563 | 0.734205 | 0.0944358 | 0.163187 | 0.514146 | 1.65391e-05 | 0.000138528 | 40.5941 | 1.67236 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.168495 | 0.00890585 | 0.00176913 | 0.059038 | 0.0109428 | 0.862001 | 0.101097 | 0.443993 | NA | 0.00482297 | 0.0110053 | 2525.16 | 4.88142 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.181117 | 0.0109047 | 0.00167364 | 0.0547255 | 0.0115902 | 0.592996 | 0.10867 | 0.59 | NA | 0.0076998 | 0.0403982 | 4177.34 | 16.2211 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.171246 | 0.02 | 0.00590798 | 0.0538423 | 0.0137069 | 1.01622 | 0.102748 | 0.227944 | NA | 0.00630797 | 0.0226858 | 2901.39 | 1.09823 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.170648 | 0.00776295 | 0.00176075 | 0.0598677 | 0.010716 | 1.1738 | 0.102389 | 0.38516 | 0.574757 | 0.00322687 | 0.00586141 | 1361.01 | 0.442122 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.272932 | 0.00402676 | 0.000909282 | 0.0382848 | 0.00564815 | 1.6393 | 0.163759 | 0.440851 | 0.55725 | 0.000975994 | 0.00284877 | 393.831 | 1.38891 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.131132 | 0.00516458 | 0.00170395 | 246.943 | 9.08379 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.248574 | 0.0148164 | 0.00163778 | 6729.01 | 4.64915 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.125779 | 0.00673653 | 0.00196567 | 599.541 | 7.75042 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.294248 | 0.00428676 | 0.00122429 | 212.963 | 0.440392 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.257562 | 0.00337481 | 0.000333598 | 623.289 | 8.38581 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.267521 | 0.00442063 | 0.00117 | 347.043 | 1.18541 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.268316 | 0.02 | 0.00655591 | 510.546 | 2.43414 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.157393 | 0.00448583 | 0.00122098 | 40.5941 | 1.67236 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.131132 | 0.00516458 | 0.00170395 | 246.943 | 9.08379 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.248574 | 0.0148164 | 0.00163778 | 6729.01 | 4.64915 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.125779 | 0.00673653 | 0.00196567 | 599.541 | 7.75042 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.13691 | 0.00679095 | 0.00178264 | 516.95 | 54.3566 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.240467 | 0.0165528 | 0.00140988 | 10796.5 | 14.652 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.165974 | 0.00937047 | 0.00182842 | 1218.58 | 46.5012 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.216192 | 0.02 | 0.00688975 | 1330.59 | 1.18844 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.110502 | 0.02 | 0.00435642 | 4586.26 | 1.14096 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.187045 | 0.02 | 0.00647777 | 2787.33 | 1.00034 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.150842 | 0.00510784 | 0.00155945 | 147.098 | 0.507266 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.210687 | 0.011946 | 0.00186796 | 3496.75 | 0.451825 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.150415 | 0.00623505 | 0.00185484 | 439.192 | 0.36419 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.293949 | 0.00428669 | 0.00122519 | 212.389 | 0.439205 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.257485 | 0.00337781 | 0.000333808 | 624.977 | 8.40852 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.267361 | 0.00441579 | 0.00116885 | 344.127 | 1.17545 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.191958 | 0.008139 | 0.00164712 | 0.0579447 | 0.00994805 | 1.02476 | 0.115175 | 0.429535 | NA | 0.00428484 | 0.00436425 | 2075.96 | 4.01306 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.306735 | 0.00421485 | 0.00101855 | 0.0412134 | 0.00572139 | 1.78798 | 0.184041 | 0.435604 | NA | 0.00122355 | 0.00234582 | 451.601 | 1.59264 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.281671 | 0.02 | 0.00665056 | 0.0710528 | 0.0189378 | 1.1712 | 0.169003 | 0.0156042 | NA | 0.00162492 | 0.00325105 | 464.553 | 2.21485 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.208192 | 0.00408649 | 0.000892579 | 0.0342946 | 0.00844376 | 0.903163 | 0.124915 | 0.14425 | 0.327458 | 1.38869e-05 | 6.17146e-05 | 32.1449 | 1.32427 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.191958 | 0.008139 | 0.00164712 | 0.0579447 | 0.00994805 | 1.02476 | 0.115175 | 0.429535 | NA | 0.00428484 | 0.00436425 | 2075.96 | 4.01306 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.196103 | 0.0100262 | 0.001566 | 0.050155 | 0.0106366 | 0.672225 | 0.117662 | 0.59 | NA | 0.00693092 | 0.0349732 | 3560.66 | 13.8265 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.144431 | 0.02 | 0.00627792 | 0.0444257 | 0.0115674 | 0.849061 | 0.0866588 | 0.228069 | NA | 0.00856873 | 0.0173891 | 2740.67 | 1.03739 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.21268 | 0.00720229 | 0.00162969 | 0.0598677 | 0.0100783 | 1.39959 | 0.127608 | 0.367875 | 0.486667 | 0.00275127 | 0.00832703 | 1165.45 | 0.378593 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.306528 | 0.0042229 | 0.0010198 | 0.0412467 | 0.00572987 | 1.78753 | 0.183917 | 0.435878 | NA | 0.00123206 | 0.0023091 | 452.474 | 1.59572 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.153953 | 0.00485568 | 0.00159777 | 169.44 | 6.23285 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.276323 | 0.0135054 | 0.00157304 | 5610.1 | 3.87608 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.145598 | 0.00605595 | 0.00177056 | 448.347 | 5.79589 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.33449 | 0.00517229 | 0.00162924 | 283.561 | 0.586383 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.284765 | 0.00340843 | 0.000353893 | 758.496 | 10.2049 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.300951 | 0.00406385 | 0.00107252 | 312.748 | 1.06827 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.281671 | 0.02 | 0.00665056 | 464.553 | 2.21485 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.208192 | 0.00408649 | 0.000892579 | 32.1449 | 1.32427 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.153953 | 0.00485568 | 0.00159777 | 169.44 | 6.23285 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.276323 | 0.0135054 | 0.00157304 | 5610.1 | 3.87608 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.145598 | 0.00605595 | 0.00177056 | 448.347 | 5.79589 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.138198 | 0.00618276 | 0.00165165 | 427.935 | 44.9968 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.280726 | 0.0156275 | 0.00136853 | 9342.76 | 12.6792 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.169384 | 0.00826837 | 0.00167782 | 911.292 | 34.7752 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.208058 | 0.02 | 0.00741857 | 1349.2 | 1.20506 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0741616 | 0.02 | 0.00440405 | 4222.29 | 1.05041 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.151074 | 0.02 | 0.00701114 | 2650.51 | 0.951238 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.188052 | 0.00486422 | 0.00142925 | 94.9686 | 0.327498 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.259559 | 0.0108574 | 0.00175299 | 3092.59 | 0.399602 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.190427 | 0.00588529 | 0.00170683 | 308.786 | 0.256053 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.334263 | 0.00517927 | 0.00163043 | 283.162 | 0.585557 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.284685 | 0.00342253 | 0.000354895 | 762.739 | 10.262 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.300637 | 0.00406691 | 0.00107408 | 311.523 | 1.06408 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.166831 | 0.008679 | 0.00172993 | 0.058432 | 0.0106395 | 0.876959 | 0.100099 | 0.440694 | NA | 0.00484342 | 0.00996209 | 2391.66 | 4.62334 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.287121 | 0.00404049 | 0.000932491 | 0.0391693 | 0.00566643 | 1.83627 | 0.172273 | 0.437153 | 0.59 | 0.00105897 | 0.00258761 | 383.077 | 1.35098 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.281816 | 0.02 | 0.00638697 | 0.0732003 | 0.0199564 | 1.27969 | 0.16909 | 0.0145417 | 0.5875 | 0.000825644 | 0.00333374 | 447.387 | 2.13301 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.16503 | 0.00434416 | 0.00110298 | 0.0342941 | 0.00833919 | 0.790873 | 0.0990182 | 0.157292 | 0.462233 | 6.22951e-06 | 5.33782e-05 | 33.2195 | 1.36854 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.166831 | 0.008679 | 0.00172994 | 0.058432 | 0.0106395 | 0.876959 | 0.100099 | 0.440694 | NA | 0.00484342 | 0.00996209 | 2391.66 | 4.62334 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.181938 | 0.0107113 | 0.00164409 | 0.0532773 | 0.0113812 | 0.586126 | 0.109163 | 0.59 | NA | 0.00777076 | 0.0397161 | 4020.91 | 15.6137 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.171905 | 0.02 | 0.00588535 | 0.053436 | 0.0135947 | 1.07532 | 0.103143 | 0.228194 | NA | 0.00656501 | 0.0217421 | 2830.61 | 1.07144 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.169204 | 0.00755867 | 0.00171615 | 0.0598677 | 0.010433 | 1.24046 | 0.101522 | 0.381125 | NA | 0.00323438 | 0.005481 | 1284.65 | 0.417315 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.287121 | 0.00404049 | 0.000932491 | 0.0391693 | 0.00566643 | 1.83627 | 0.172273 | 0.437153 | 0.59 | 0.00105897 | 0.00258761 | 383.077 | 1.35098 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.130164 | 0.00501627 | 0.00165725 | 222.855 | 8.19773 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.247527 | 0.014505 | 0.00161914 | 6403.86 | 4.4245 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.122803 | 0.00651571 | 0.00191341 | 548.263 | 7.08754 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.308339 | 0.00443946 | 0.00129117 | 222.027 | 0.459136 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.27201 | 0.00331854 | 0.000340406 | 598.293 | 8.04951 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.281014 | 0.00436346 | 0.0011659 | 328.909 | 1.12347 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.281816 | 0.02 | 0.00638697 | 447.387 | 2.13301 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.16503 | 0.00434416 | 0.00110298 | 33.2195 | 1.36854 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.130164 | 0.00501627 | 0.00165725 | 222.855 | 8.19773 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.247527 | 0.014505 | 0.00161914 | 6403.86 | 4.4245 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.122803 | 0.00651571 | 0.00191341 | 548.263 | 7.08754 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.135933 | 0.00664855 | 0.00174229 | 496.366 | 52.1922 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.244921 | 0.0163091 | 0.00139741 | 10399.3 | 14.113 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.16496 | 0.00917639 | 0.00179256 | 1167.04 | 44.5348 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.215303 | 0.02 | 0.00686314 | 1266.94 | 1.13159 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.112209 | 0.02 | 0.00434509 | 4560.09 | 1.13445 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.188203 | 0.02 | 0.00644782 | 2664.8 | 0.956366 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.148742 | 0.00496173 | 0.00151339 | 130.217 | 0.449051 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.206482 | 0.0116542 | 0.00184022 | 3327.63 | 0.429973 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.152388 | 0.00606011 | 0.00179485 | 396.097 | 0.328454 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.308339 | 0.00443946 | 0.00129117 | 222.027 | 0.459136 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.27201 | 0.00331854 | 0.000340406 | 598.293 | 8.04951 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.281014 | 0.00436346 | 0.0011659 | 328.909 | 1.12347 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.212746 | 0.0074445 | 0.00154845 | 0.0571595 | 0.00918575 | 1.24649 | 0.127647 | 0.418187 | NA | 0.00389124 | 0.00305177 | 1694.68 | 3.27601 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.386977 | 0.00421242 | 0.00104265 | 0.0446511 | 0.00574228 | 2.25619 | 0.232186 | 0.425076 | NA | 0.00134316 | 0.00296519 | 467.131 | 1.64741 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.317506 | 0.02 | 0.00634266 | 0.0773403 | 0.019248 | 1.22637 | 0.190503 | 0.0146875 | NA | 0.00141075 | 0.00113942 | 372.78 | 1.77731 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.237077 | 0.00394804 | 0.000884525 | 0.0345427 | 0.00870878 | 1.03361 | 0.142246 | 0.137479 | 0.291687 | 2.42681e-05 | 8.10384e-05 | 37.2707 | 1.53544 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.212746 | 0.0074445 | 0.00154845 | 0.0571595 | 0.00918575 | 1.24649 | 0.127647 | 0.418187 | NA | 0.00389124 | 0.00305177 | 1694.68 | 3.27601 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.212315 | 0.00928691 | 0.00148603 | 0.0465186 | 0.00981596 | 0.779874 | 0.127389 | 0.59 | NA | 0.00650577 | 0.0277927 | 2942.65 | 11.4266 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.159119 | 0.02 | 0.00629605 | 0.0428443 | 0.0109616 | 0.811021 | 0.0954712 | 0.227889 | NA | 0.00947145 | 0.0100649 | 2569.71 | 0.972681 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.25632 | 0.00681667 | 0.00151924 | 0.0598677 | 0.0097883 | 1.69458 | 0.153792 | 0.355778 | 0.535 | 0.00243692 | 0.00745083 | 1056.63 | 0.343243 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.386977 | 0.00421242 | 0.00104265 | 0.0446511 | 0.00574228 | 2.25619 | 0.232186 | 0.425076 | NA | 0.00134316 | 0.00296519 | 467.131 | 1.64741 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.169534 | 0.00464207 | 0.00149015 | 132.177 | 4.86212 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.297171 | 0.0120077 | 0.00149605 | 4599.46 | 3.17783 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.171532 | 0.0056837 | 0.00165914 | 352.413 | 4.55573 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.373724 | 0.00539398 | 0.00168016 | 272.738 | 0.564002 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.437795 | 0.00309758 | 0.000345191 | 830.358 | 11.1717 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.34941 | 0.00414572 | 0.0011026 | 298.298 | 1.01891 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.317506 | 0.02 | 0.00634266 | 372.78 | 1.77731 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.237077 | 0.00394804 | 0.000884525 | 37.2707 | 1.53544 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.169534 | 0.00464207 | 0.00149015 | 132.177 | 4.86212 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.297171 | 0.0120077 | 0.00149605 | 4599.46 | 3.17783 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.171532 | 0.0056837 | 0.00165914 | 352.413 | 4.55573 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.139384 | 0.00576215 | 0.00155502 | 371.622 | 39.0756 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.325734 | 0.0144393 | 0.00131693 | 7689.23 | 10.4351 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.171827 | 0.00765922 | 0.00158613 | 767.084 | 29.2721 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.235491 | 0.02 | 0.00734822 | 1212.83 | 1.08326 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0738728 | 0.02 | 0.00444283 | 3997.13 | 0.994398 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.167992 | 0.02 | 0.00709711 | 2499.16 | 0.89692 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.208506 | 0.00472624 | 0.00131619 | 77.6144 | 0.267652 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.335214 | 0.0100536 | 0.00164807 | 2840.78 | 0.367064 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.225239 | 0.00567022 | 0.00159347 | 251.489 | 0.208542 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.373724 | 0.00539398 | 0.00168016 | 272.738 | 0.564002 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.437795 | 0.00309758 | 0.000345191 | 830.358 | 11.1717 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.34941 | 0.00414572 | 0.0011026 | 298.298 | 1.01891 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.172634 | 0.00819686 | 0.00163454 | 0.0577763 | 0.0101497 | 0.984954 | 0.103581 | 0.433118 | NA | 0.00412549 | 0.00685975 | 2141.08 | 4.13895 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.342388 | 0.00413515 | 0.000961856 | 0.0439719 | 0.0059042 | 2.28056 | 0.205433 | 0.426455 | 0.568677 | 0.000915014 | 0.00250119 | 423.57 | 1.49378 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.30025 | 0.02 | 0.00617188 | 0.0764111 | 0.0201061 | 1.47585 | 0.18015 | 0.0149792 | 0.573299 | 0.000541119 | 0.00257566 | 391.535 | 1.86673 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.180812 | 0.00412123 | 0.000931455 | 0.0342941 | 0.00817089 | 0.954248 | 0.108487 | 0.14475 | 0.361396 | -1.39645e-06 | 4.2803e-05 | 29.4125 | 1.21171 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.172634 | 0.00819686 | 0.00163454 | 0.0577763 | 0.0101497 | 0.984954 | 0.103581 | 0.433118 | NA | 0.00412549 | 0.00685975 | 2141.08 | 4.13895 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.189455 | 0.0101477 | 0.00157195 | 0.0502434 | 0.0107756 | 0.603358 | 0.113673 | 0.59 | NA | 0.00681723 | 0.0358787 | 3659.32 | 14.2096 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.188168 | 0.02 | 0.00581786 | 0.0552294 | 0.0137433 | 1.2178 | 0.112901 | 0.228444 | NA | 0.00599623 | 0.0214445 | 2816.59 | 1.06613 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.178934 | 0.00716052 | 0.00161132 | 0.0598677 | 0.0100892 | 1.44125 | 0.10736 | 0.36875 | 0.563185 | 0.00265065 | 0.00604756 | 1167.17 | 0.379153 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.342388 | 0.00413515 | 0.000961856 | 0.0439719 | 0.0059042 | 2.28056 | 0.205433 | 0.426455 | 0.568677 | 0.000915014 | 0.00250119 | 423.57 | 1.49378 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.134221 | 0.00474603 | 0.00153847 | 188.528 | 6.935 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.256444 | 0.0137797 | 0.00158209 | 5759.63 | 3.9794 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.127238 | 0.00606485 | 0.00178307 | 475.096 | 6.14169 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.351259 | 0.00466739 | 0.00136303 | 245.433 | 0.507537 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.345265 | 0.00342172 | 0.000371058 | 685.387 | 9.22127 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.330638 | 0.00431633 | 0.00115148 | 339.89 | 1.16098 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.30025 | 0.02 | 0.00617188 | 391.535 | 1.86673 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.180812 | 0.00412123 | 0.000931455 | 29.4125 | 1.21171 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.134221 | 0.00474603 | 0.00153847 | 188.528 | 6.935 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.256444 | 0.0137797 | 0.00158209 | 5759.63 | 3.9794 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.127238 | 0.00606485 | 0.00178307 | 475.096 | 6.14169 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.137361 | 0.00613397 | 0.00164081 | 418.704 | 44.0262 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.26299 | 0.0157864 | 0.00137359 | 9574.75 | 12.994 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.168014 | 0.00852274 | 0.00170146 | 984.501 | 37.5688 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.224506 | 0.02 | 0.00677106 | 1200.61 | 1.07235 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.135298 | 0.02 | 0.0043179 | 4688.54 | 1.16641 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.2047 | 0.02 | 0.00636463 | 2560.61 | 0.918974 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.15367 | 0.00473641 | 0.00139952 | 107.853 | 0.371931 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.218749 | 0.0109841 | 0.00177661 | 3052.38 | 0.394406 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.164382 | 0.00576102 | 0.00165783 | 341.282 | 0.283 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.351259 | 0.00466739 | 0.00136303 | 245.433 | 0.507537 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.345265 | 0.00342172 | 0.000371058 | 685.387 | 9.22127 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.330638 | 0.00431633 | 0.00115148 | 339.89 | 1.16098 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.23339 | 0.00682018 | 0.00144214 | 0.0558637 | 0.00853039 | 1.532 | 0.140034 | 0.40691 | NA | 0.00344379 | 0.0030509 | 1388.62 | 2.68436 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.481538 | 0.00381433 | 0.000936629 | 0.0436638 | 0.00544261 | 3.15541 | 0.288923 | 0.403722 | 0.501667 | 0.00111318 | 0.00427758 | 503.172 | 1.77451 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.371114 | 0.02 | 0.0059299 | 0.0841138 | 0.0197289 | 1.45804 | 0.222668 | 0.0143333 | 0.546667 | 0.000976725 | 0.000474383 | 299.303 | 1.42699 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.250131 | 0.00376325 | 0.00082558 | 0.0345262 | 0.00839 | 1.21418 | 0.150078 | 0.129521 | 0.275375 | 1.74684e-05 | 1.67023e-05 | 35.3623 | 1.45682 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.23339 | 0.00682018 | 0.00144214 | 0.0558637 | 0.00853039 | 1.532 | 0.140034 | 0.40691 | NA | 0.00344379 | 0.0030509 | 1388.62 | 2.68436 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.211038 | 0.0085548 | 0.00140286 | 0.0423905 | 0.00905077 | 0.967034 | 0.126623 | 0.59 | NA | 0.0058999 | 0.0246731 | 2404.26 | 9.33601 | none |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.210428 | 0.02 | 0.00611297 | 0.0471062 | 0.01136 | 0.896194 | 0.126257 | 0.227917 | NA | 0.00910829 | 0.00896505 | 2429.31 | 0.919538 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.281638 | 0.00651914 | 0.00143374 | 0.0598677 | 0.00942276 | 1.95291 | 0.168983 | 0.35066 | 0.493333 | 0.0023485 | 0.00477176 | 941.835 | 0.305953 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.481538 | 0.00381433 | 0.000936629 | 0.0436638 | 0.00544261 | 3.15541 | 0.288923 | 0.403722 | 0.501667 | 0.00111318 | 0.00427758 | 503.172 | 1.77451 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.183583 | 0.00446254 | 0.00140074 | 114.672 | 4.21819 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.314827 | 0.0108132 | 0.00140353 | 3775.24 | 2.60836 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.201761 | 0.00518474 | 0.00152214 | 275.949 | 3.56726 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.403409 | 0.00507951 | 0.00152133 | 223.417 | 0.46201 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.630844 | 0.00256604 | 0.000274026 | 1023.39 | 13.7688 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.41036 | 0.00379745 | 0.00101453 | 262.713 | 0.897362 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.371114 | 0.02 | 0.0059299 | 299.303 | 1.42699 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.250131 | 0.00376325 | 0.00082558 | 35.3623 | 1.45682 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.183583 | 0.00446254 | 0.00140074 | 114.672 | 4.21819 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.314827 | 0.0108132 | 0.00140353 | 3775.24 | 2.60836 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.201761 | 0.00518474 | 0.00152214 | 275.949 | 3.56726 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.139877 | 0.00534138 | 0.00146624 | 319.019 | 33.5444 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.318728 | 0.0132046 | 0.00124389 | 6234.73 | 8.46122 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.174509 | 0.00711844 | 0.00149845 | 659.023 | 25.1485 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.302844 | 0.02 | 0.00704066 | 1055.79 | 0.943001 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0667904 | 0.02 | 0.00444981 | 3983.28 | 0.990953 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.261649 | 0.02 | 0.00684844 | 2248.85 | 0.807085 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.211048 | 0.00461975 | 0.00126228 | 70.669 | 0.243701 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.369817 | 0.00956987 | 0.00159095 | 2538.22 | 0.32797 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.264048 | 0.00536779 | 0.001448 | 216.62 | 0.179627 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.403409 | 0.00507951 | 0.00152133 | 223.417 | 0.46201 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.630844 | 0.00256604 | 0.000274026 | 1023.39 | 13.7688 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.41036 | 0.00379745 | 0.00101453 | 262.713 | 0.897362 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
