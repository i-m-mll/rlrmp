# GRU perturbation-response bank

Issue: `b8aa38e`. Source experiment: `b8aa38e`.

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

### `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.198858 | 0.00534013 | 0.00114713 | 0.0487144 | 0.00712532 | 1.33229 | 0.119315 | 0.391277 | NA | 0.00209056 | 0.00546167 | 882.37 | 1.70572 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.225105 | 0.00347036 | 0.000750113 | 0.0327428 | 0.00466726 | 1.19401 | 0.135063 | 0.435906 | NA | 0.000896112 | 0.00219679 | 364.582 | 1.28576 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.188727 | 0.02 | 0.00784108 | 0.0529679 | 0.017955 | 0.640342 | 0.113236 | 0.0147188 | NA | 0.0029977 | 0.00841885 | 981.939 | 4.6816 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.121955 | 0.00500078 | 0.00187488 | 0.0342941 | 0.00847257 | 0.467573 | 0.0731733 | 0.193057 | NA | 0.000371468 | 0.0015225 | 109.652 | 4.51735 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.198858 | 0.00534013 | 0.00114713 | 0.0487144 | 0.00712532 | 1.33229 | 0.119315 | 0.391277 | NA | 0.00209056 | 0.00546167 | 882.37 | 1.70572 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.185978 | 0.0107797 | 0.00168422 | 0.0542049 | 0.0114583 | 0.639541 | 0.111587 | 0.59 | NA | 0.00799504 | 0.0390035 | 4034.59 | 15.6668 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.139851 | 0.02 | 0.00633685 | 0.0458466 | 0.0121924 | 0.711965 | 0.0839108 | 0.228135 | NA | 0.00833238 | 0.0213756 | 2948.31 | 1.11599 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.177509 | 0.00788493 | 0.00190183 | 0.0598677 | 0.0106249 | 1.09572 | 0.106505 | 0.388596 | NA | 0.00353397 | 0.00819569 | 1364.8 | 0.443354 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.225181 | 0.00346979 | 0.000750038 | 0.0327433 | 0.00466667 | 1.19465 | 0.135108 | 0.435887 | NA | 0.000897851 | 0.00215645 | 364.461 | 1.28533 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.134864 | 0.00355297 | 0.00114397 | 92.93 | 3.41843 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.305077 | 0.00858421 | 0.0012027 | 2385.45 | 1.64814 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.156632 | 0.0038832 | 0.00109473 | 168.728 | 2.18119 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.193984 | 0.00317145 | 0.000970574 | 106.945 | 0.221155 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.23717 | 0.00350615 | 0.000331008 | 734.278 | 9.87906 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.244161 | 0.0037335 | 0.000948758 | 252.523 | 0.862555 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.188727 | 0.02 | 0.00784108 | 981.939 | 4.6816 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.121955 | 0.00500078 | 0.00187488 | 109.652 | 4.51735 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.134864 | 0.00355297 | 0.00114397 | 92.93 | 3.41843 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.305077 | 0.00858421 | 0.0012027 | 2385.45 | 1.64814 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.156632 | 0.0038832 | 0.00109473 | 168.728 | 2.18119 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.137082 | 0.00692628 | 0.00184391 | 532.828 | 56.0262 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.254893 | 0.0163286 | 0.00140073 | 10449.5 | 14.1811 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.16596 | 0.0090843 | 0.00180802 | 1121.45 | 42.7949 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.173287 | 0.02 | 0.00769867 | 1676.44 | 1.49734 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.099557 | 0.02 | 0.00440612 | 4353 | 1.08293 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.14671 | 0.02 | 0.00690577 | 2815.49 | 1.01044 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.137231 | 0.00556123 | 0.00192787 | 201.145 | 0.693647 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.243129 | 0.0117269 | 0.00184545 | 3464.5 | 0.447657 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.152166 | 0.00636665 | 0.00193216 | 428.772 | 0.355549 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.194006 | 0.00317122 | 0.00097055 | 106.87 | 0.220999 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.23728 | 0.00350534 | 0.00033101 | 734.055 | 9.87607 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.244255 | 0.00373281 | 0.000948554 | 252.456 | 0.862327 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.213249 | 0.00478547 | 0.00102786 | 0.0473904 | 0.00644344 | 1.55065 | 0.12795 | 0.37949 | NA | 0.00163574 | 0.00417116 | 711.268 | 1.37496 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.268851 | 0.00357921 | 0.000802243 | 0.0364572 | 0.00488004 | 1.42724 | 0.161311 | 0.426491 | NA | 0.000833929 | 0.0018194 | 365.482 | 1.28893 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.236691 | 0.02 | 0.00735016 | 0.0616098 | 0.0183192 | 0.806823 | 0.142015 | 0.0142813 | NA | 0.0023663 | 0.00585502 | 723.802 | 3.45088 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.142436 | 0.00469494 | 0.0015928 | 0.0342941 | 0.00817385 | 0.566217 | 0.0854616 | 0.174531 | NA | 0.000184252 | 0.000678205 | 63.9845 | 2.63597 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.213249 | 0.00478547 | 0.00102786 | 0.0473904 | 0.00644344 | 1.55065 | 0.12795 | 0.37949 | NA | 0.00163574 | 0.00417116 | 711.268 | 1.37496 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.18528 | 0.0103792 | 0.0016221 | 0.0513491 | 0.0110287 | 0.623568 | 0.111168 | 0.59 | NA | 0.00749735 | 0.0381257 | 3719.06 | 14.4415 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.157417 | 0.02 | 0.00622761 | 0.0467279 | 0.0121232 | 0.836591 | 0.0944504 | 0.228148 | NA | 0.00814076 | 0.0182685 | 2772.47 | 1.04943 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.189346 | 0.00747628 | 0.00179453 | 0.0598677 | 0.0101445 | 1.25652 | 0.113608 | 0.377663 | NA | 0.00311941 | 0.00673147 | 1210.56 | 0.393249 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.268851 | 0.00357921 | 0.000802243 | 0.0364572 | 0.00488004 | 1.42724 | 0.161311 | 0.426491 | NA | 0.000833929 | 0.0018194 | 365.482 | 1.28893 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.146319 | 0.00315453 | 0.000985954 | 59.1352 | 2.17529 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.322327 | 0.00767728 | 0.00111292 | 1948.03 | 1.34592 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.171102 | 0.00352458 | 0.000984713 | 126.635 | 1.63704 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.235289 | 0.00364797 | 0.00112387 | 132.08 | 0.273133 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.282621 | 0.00341021 | 0.000352941 | 734.795 | 9.88602 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.288642 | 0.00367945 | 0.000929917 | 229.57 | 0.784152 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.236691 | 0.02 | 0.00735016 | 723.802 | 3.45088 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.142436 | 0.00469494 | 0.0015928 | 63.9845 | 2.63597 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.146319 | 0.00315453 | 0.000985954 | 59.1352 | 2.17529 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.322327 | 0.00767728 | 0.00111292 | 1948.03 | 1.34592 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.171102 | 0.00352458 | 0.000984713 | 126.635 | 1.63704 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.1344 | 0.0066126 | 0.001757 | 484.283 | 50.9217 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.258676 | 0.0158423 | 0.00137632 | 9657.24 | 13.1059 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.162765 | 0.00868269 | 0.00173297 | 1015.66 | 38.7578 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.204937 | 0.02 | 0.00747584 | 1440.63 | 1.28672 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.100682 | 0.02 | 0.00438864 | 4287.56 | 1.06665 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.166633 | 0.02 | 0.00681833 | 2589.21 | 0.929237 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.151002 | 0.00527277 | 0.0017754 | 150.856 | 0.520226 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.250358 | 0.0110787 | 0.00178411 | 3128.39 | 0.404228 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.166679 | 0.00607736 | 0.00182409 | 352.443 | 0.292255 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.235289 | 0.00364797 | 0.00112387 | 132.08 | 0.273133 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.282621 | 0.00341021 | 0.000352941 | 734.795 | 9.88602 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.288642 | 0.00367945 | 0.000929917 | 229.57 | 0.784152 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.23995 | 0.00422254 | 0.000907143 | 0.0457859 | 0.00579687 | 1.83398 | 0.14397 | 0.365564 | NA | 0.00122904 | 0.00311015 | 565.993 | 1.09413 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.335218 | 0.00370371 | 0.000878225 | 0.0410465 | 0.00508983 | 1.82975 | 0.201131 | 0.417151 | NA | 0.000829599 | 0.0020334 | 372.087 | 1.31222 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.332441 | 0.02 | 0.00653974 | 0.0781718 | 0.0191956 | 1.15093 | 0.199464 | 0.0139974 | NA | 0.00135482 | 0.00325219 | 456.498 | 2.17645 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.190165 | 0.00432953 | 0.00113625 | 0.0342941 | 0.00803678 | 0.771012 | 0.114099 | 0.152013 | 0.433667 | 1.68037e-05 | 0.0001313 | 32.9367 | 1.35689 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.23995 | 0.00422254 | 0.000907143 | 0.0457859 | 0.00579687 | 1.83398 | 0.14397 | 0.365564 | NA | 0.00122904 | 0.00311015 | 565.993 | 1.09413 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.18977 | 0.0101927 | 0.00158979 | 0.0496144 | 0.0108174 | 0.648821 | 0.113862 | 0.59 | NA | 0.00723626 | 0.0369444 | 3542.28 | 13.7551 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.17755 | 0.02 | 0.006132 | 0.0484005 | 0.0120954 | 0.980223 | 0.10653 | 0.228192 | NA | 0.00799791 | 0.016771 | 2646 | 1.00156 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.203489 | 0.00722072 | 0.00172339 | 0.0598677 | 0.00981007 | 1.43566 | 0.122093 | 0.370627 | NA | 0.00291637 | 0.00590184 | 1124.88 | 0.365414 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.335218 | 0.00370371 | 0.000878225 | 0.0410465 | 0.00508983 | 1.82975 | 0.201131 | 0.417151 | NA | 0.000829599 | 0.0020334 | 372.087 | 1.31222 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.163758 | 0.00284547 | 0.00084389 | 46.4199 | 1.70756 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.366075 | 0.00667954 | 0.00100409 | 1553.46 | 1.0733 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.190016 | 0.00314262 | 0.000873453 | 98.1004 | 1.26817 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.296606 | 0.00425232 | 0.00134576 | 182.103 | 0.376576 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.35459 | 0.00316684 | 0.000351617 | 711.476 | 9.57227 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.354458 | 0.00369197 | 0.000937298 | 222.683 | 0.760627 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.332441 | 0.02 | 0.00653974 | 456.498 | 2.17645 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.190165 | 0.00432953 | 0.00113625 | 32.9367 | 1.35689 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.163758 | 0.00284547 | 0.00084389 | 46.4199 | 1.70756 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.366075 | 0.00667953 | 0.00100409 | 1553.46 | 1.0733 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.190016 | 0.00314263 | 0.000873453 | 98.1004 | 1.26817 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.135413 | 0.00650849 | 0.00171098 | 475.099 | 49.956 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.269385 | 0.0155112 | 0.00135943 | 9155.32 | 12.4248 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.16451 | 0.0085584 | 0.00169895 | 996.432 | 38.0241 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.242314 | 0.02 | 0.00725367 | 1277.56 | 1.14108 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.103841 | 0.02 | 0.00438173 | 4218.3 | 1.04942 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.186494 | 0.02 | 0.00676058 | 2442.15 | 0.876458 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.168008 | 0.0050869 | 0.00166036 | 126.037 | 0.434637 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.259882 | 0.0107024 | 0.00174652 | 2940.63 | 0.379966 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.182576 | 0.00587285 | 0.0017633 | 307.971 | 0.255378 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.296606 | 0.00425232 | 0.00134576 | 182.103 | 0.376576 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.35459 | 0.00316684 | 0.000351617 | 711.476 | 9.57227 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.354458 | 0.00369197 | 0.000937298 | 222.683 | 0.760627 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64`

- Evaluated: 348
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.2737 | 0.00360143 | 0.000775731 | 0.0436502 | 0.00516987 | 2.29267 | 0.16422 | 0.35349 | 0.531667 | 0.000858176 | 0.000703374 | 395.404 | 0.764359 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.328562 | 0.00276597 | 0.000673792 | 0.0336758 | 0.00397609 | 2.15627 | 0.197137 | 0.400759 | NA | 0.000347966 | 0.00102152 | 190.448 | 0.671644 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.503792 | 0.02 | 0.00525783 | 0.102487 | 0.020665 | 1.91544 | 0.302275 | 0.0144271 | 0.467437 | 9.55592e-05 | 0.000207102 | 234.927 | 1.12007 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.262163 | 0.0038536 | 0.000811726 | 0.0343475 | 0.00830486 | 1.17471 | 0.157298 | 0.130852 | 0.298439 | -2.86627e-05 | 3.87752e-05 | 33.7791 | 1.3916 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.2737 | 0.00360143 | 0.000775731 | 0.0436502 | 0.00516987 | 2.29267 | 0.16422 | 0.35349 | 0.531667 | 0.000858176 | 0.000703374 | 395.404 | 0.764359 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.218822 | 0.0095404 | 0.00155925 | 0.0467203 | 0.0100783 | 0.882678 | 0.131293 | 0.59 | NA | 0.00639378 | 0.0233413 | 2979.19 | 11.5685 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.211443 | 0.02 | 0.00605014 | 0.0504521 | 0.012197 | 1.04523 | 0.126866 | 0.228359 | NA | 0.00763753 | 0.0127141 | 2551.34 | 0.965727 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.227203 | 0.00714393 | 0.00170134 | 0.0598677 | 0.00991862 | 1.47992 | 0.136322 | 0.366941 | NA | 0.00259335 | 0.00858482 | 1132.7 | 0.367954 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.328562 | 0.00276597 | 0.000673792 | 0.0336758 | 0.00397609 | 2.15627 | 0.197137 | 0.400759 | NA | 0.000347966 | 0.00102152 | 190.448 | 0.671644 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.156069 | 0.00273537 | 0.000810187 | 38.4014 | 1.41259 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.458888 | 0.00532054 | 0.000839795 | 1074.48 | 0.742372 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.206142 | 0.00274838 | 0.000677211 | 73.3284 | 0.947935 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.25364 | 0.00342763 | 0.00105182 | 106.527 | 0.22029 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.371213 | 0.00202719 | 0.000246729 | 321.506 | 4.32557 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.360832 | 0.00284309 | 0.000722822 | 143.311 | 0.489513 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.503792 | 0.02 | 0.00525783 | 234.927 | 1.12007 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.262163 | 0.0038536 | 0.000811726 | 33.7791 | 1.3916 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.156069 | 0.00273537 | 0.000810187 | 38.4014 | 1.41259 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.458888 | 0.00532054 | 0.000839795 | 1074.48 | 0.742372 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.206142 | 0.00274838 | 0.000677211 | 73.3284 | 0.947935 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.147066 | 0.00619654 | 0.00168848 | 432.011 | 45.4253 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.327872 | 0.0143453 | 0.00131369 | 7626.43 | 10.3499 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.181528 | 0.00807936 | 0.00167559 | 879.129 | 33.5478 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.269056 | 0.02 | 0.0072144 | 1207.89 | 1.07885 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.108343 | 0.02 | 0.004374 | 4251.78 | 1.05775 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.256929 | 0.02 | 0.006562 | 2194.34 | 0.787522 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.166127 | 0.00523007 | 0.00173324 | 133.53 | 0.460477 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.305134 | 0.0105346 | 0.00170944 | 3002.72 | 0.387989 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.210347 | 0.00566714 | 0.00166134 | 261.84 | 0.217125 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.25364 | 0.00342763 | 0.00105182 | 106.527 | 0.22029 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.371213 | 0.00202719 | 0.000246729 | 321.506 | 4.32557 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.360832 | 0.00284309 | 0.000722822 | 143.311 | 0.489513 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
