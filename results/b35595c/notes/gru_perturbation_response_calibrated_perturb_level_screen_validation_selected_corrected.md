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
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.164652 | 0.00887693 | 0.00177752 | 0.0587183 | 0.0108008 | 0.839856 | 0.0987915 | 0.445583 | NA | 0.00512248 | 0.0110385 | 2494.98 | 4.82307 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.268307 | 0.00397552 | 0.000922354 | 0.0374841 | 0.0055072 | 1.6733 | 0.160984 | 0.440601 | NA | 0.00104916 | 0.00275868 | 368.597 | 1.29991 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.271598 | 0.02 | 0.00652471 | 0.0709592 | 0.0197477 | 1.22341 | 0.162959 | 0.01325 | NA | 0.00103003 | 0.00375715 | 483.291 | 2.30419 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.159218 | 0.00441019 | 0.00117823 | 0.0342941 | 0.00836862 | 0.752047 | 0.0955308 | 0.16 | 0.500331 | 1.63964e-05 | 9.63323e-05 | 36.439 | 1.50118 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.164652 | 0.00887693 | 0.00177752 | 0.0587183 | 0.0108008 | 0.839856 | 0.0987915 | 0.445583 | NA | 0.00512248 | 0.0110385 | 2494.98 | 4.82307 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.179602 | 0.010953 | 0.0016764 | 0.0545135 | 0.0116388 | 0.582043 | 0.107761 | 0.59 | NA | 0.00809908 | 0.0410166 | 4168.42 | 16.1865 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.162417 | 0.02 | 0.00595927 | 0.0516753 | 0.0133247 | 1.01088 | 0.09745 | 0.228299 | NA | 0.006943 | 0.0216517 | 2831.92 | 1.07193 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.166021 | 0.00772283 | 0.00177066 | 0.0598677 | 0.0105211 | 1.16736 | 0.0996126 | 0.385938 | NA | 0.0034686 | 0.00528233 | 1336.6 | 0.43419 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.268226 | 0.00397585 | 0.000922474 | 0.0374811 | 0.00550739 | 1.67187 | 0.160936 | 0.440493 | NA | 0.00107473 | 0.00269136 | 368.158 | 1.29837 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.126466 | 0.0051342 | 0.00172598 | 237.446 | 8.73445 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.244882 | 0.0147715 | 0.00163305 | 6664.45 | 4.60455 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.122609 | 0.00672511 | 0.00197352 | 583.034 | 7.53703 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.291985 | 0.00435362 | 0.00128849 | 213.377 | 0.441248 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.250643 | 0.00327721 | 0.000328562 | 577.013 | 7.76319 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.262293 | 0.00429571 | 0.00115001 | 315.4 | 1.07733 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.271598 | 0.02 | 0.00652471 | 483.291 | 2.30419 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.159218 | 0.00441019 | 0.00117823 | 36.439 | 1.50118 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.126466 | 0.0051342 | 0.00172598 | 237.446 | 8.73445 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.244882 | 0.0147715 | 0.00163305 | 6664.45 | 4.60455 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.122609 | 0.00672511 | 0.00197352 | 583.034 | 7.53703 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.135766 | 0.00690812 | 0.00179218 | 538.983 | 56.6733 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.238867 | 0.0165032 | 0.0014064 | 10717.9 | 14.5454 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.164175 | 0.00944771 | 0.00183062 | 1248.33 | 47.6367 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.207793 | 0.02 | 0.00697484 | 1310.12 | 1.17016 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.103329 | 0.02 | 0.00435864 | 4497.3 | 1.11883 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.176128 | 0.02 | 0.00654432 | 2688.34 | 0.964812 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.146141 | 0.00505425 | 0.00157861 | 139.512 | 0.481105 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.206212 | 0.0119182 | 0.00186341 | 3453.62 | 0.446251 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.14571 | 0.00619598 | 0.00186995 | 416.665 | 0.34551 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.291891 | 0.00435472 | 0.00128914 | 213.019 | 0.440507 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.250654 | 0.00327889 | 0.000328721 | 578.271 | 7.78012 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.262134 | 0.00429394 | 0.00114956 | 313.185 | 1.06976 | none |
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
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.182341 | 0.00815316 | 0.00163726 | 0.0575603 | 0.0100438 | 0.964835 | 0.109405 | 0.430868 | NA | 0.0042899 | 0.00594848 | 2097.39 | 4.05448 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.308368 | 0.00414559 | 0.00098531 | 0.0412126 | 0.00577055 | 1.9191 | 0.185021 | 0.429503 | 0.57 | 0.00107159 | 0.00267882 | 413.343 | 1.45772 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.278798 | 0.02 | 0.00641848 | 0.072551 | 0.019517 | 1.31941 | 0.167279 | 0.0134167 | NA | 0.00105064 | 0.00327837 | 426.976 | 2.0357 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.185633 | 0.00408944 | 0.000904285 | 0.0342941 | 0.0080797 | 0.89624 | 0.11138 | 0.146 | 0.356521 | -3.02264e-06 | 5.93578e-05 | 28.125 | 1.15866 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.182341 | 0.00815316 | 0.00163726 | 0.0575603 | 0.0100438 | 0.964835 | 0.109405 | 0.430868 | NA | 0.0042899 | 0.00594848 | 2097.39 | 4.05448 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.194828 | 0.010059 | 0.00156618 | 0.0501016 | 0.0106758 | 0.642518 | 0.116897 | 0.59 | NA | 0.00700938 | 0.035106 | 3595.34 | 13.9611 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.160307 | 0.02 | 0.00601582 | 0.050298 | 0.012729 | 1.03705 | 0.0961844 | 0.228389 | NA | 0.0072369 | 0.0207931 | 2734.82 | 1.03518 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.190162 | 0.00716691 | 0.00161841 | 0.0598677 | 0.0100559 | 1.36728 | 0.114097 | 0.369347 | 0.556429 | 0.00276418 | 0.00780765 | 1152.23 | 0.374299 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.308314 | 0.00415309 | 0.000986633 | 0.0412444 | 0.00577999 | 1.91638 | 0.184989 | 0.429642 | 0.556667 | 0.00108652 | 0.00262348 | 413.771 | 1.45923 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.142618 | 0.00480057 | 0.00156838 | 176.068 | 6.47667 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.270215 | 0.0136261 | 0.00157373 | 5651.18 | 3.90447 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.134191 | 0.00603283 | 0.00176966 | 464.905 | 6.00994 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.337143 | 0.00489876 | 0.00150571 | 255.852 | 0.529083 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.282074 | 0.00343479 | 0.000363671 | 669.918 | 9.01316 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.305885 | 0.00410323 | 0.00108654 | 314.259 | 1.07343 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.278798 | 0.02 | 0.00641848 | 426.976 | 2.0357 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.185633 | 0.00408944 | 0.000904285 | 28.125 | 1.15866 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.142618 | 0.00480057 | 0.00156838 | 176.068 | 6.47667 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.270215 | 0.0136261 | 0.00157373 | 5651.18 | 3.90447 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.134191 | 0.00603283 | 0.00176966 | 464.905 | 6.00994 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.13952 | 0.00613786 | 0.00164484 | 421.229 | 44.2916 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.273766 | 0.0156794 | 0.00136798 | 9424.68 | 12.7903 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.171197 | 0.00835975 | 0.00168571 | 940.127 | 35.8755 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.213559 | 0.02 | 0.00702731 | 1217.85 | 1.08774 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.095388 | 0.02 | 0.00434891 | 4440.88 | 1.10479 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.171975 | 0.02 | 0.00667123 | 2545.73 | 0.913633 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.166761 | 0.00478257 | 0.00141012 | 95.4607 | 0.329195 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.229457 | 0.0108972 | 0.00176113 | 3036.62 | 0.39237 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.174267 | 0.00582095 | 0.00168399 | 324.612 | 0.269177 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.337189 | 0.00490553 | 0.00150695 | 255.516 | 0.528389 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.282165 | 0.00344583 | 0.000364544 | 673.546 | 9.06197 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.30559 | 0.00410791 | 0.0010884 | 312.25 | 1.06657 | none |
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
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.165421 | 0.00866732 | 0.00173273 | 0.0583562 | 0.0105851 | 0.872723 | 0.0992528 | 0.441062 | NA | 0.004838 | 0.00998744 | 2381.95 | 4.60457 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.285941 | 0.0040254 | 0.000939005 | 0.0390243 | 0.00561871 | 1.84361 | 0.171565 | 0.436573 | NA | 0.0010467 | 0.00253718 | 376.004 | 1.32604 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.283962 | 0.02 | 0.00636998 | 0.0735469 | 0.0199012 | 1.29518 | 0.170377 | 0.0149375 | 0.5875 | 0.000828857 | 0.00323318 | 437.626 | 2.08648 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.166325 | 0.00431958 | 0.00108267 | 0.0342941 | 0.00830549 | 0.795942 | 0.0997947 | 0.155688 | 0.452604 | 5.58778e-06 | 3.66215e-05 | 32.1563 | 1.32474 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.165421 | 0.00866732 | 0.00173273 | 0.0583562 | 0.0105851 | 0.872723 | 0.0992528 | 0.441062 | NA | 0.004838 | 0.00998744 | 2381.95 | 4.60457 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.18125 | 0.0107288 | 0.00164496 | 0.053199 | 0.0113989 | 0.580992 | 0.10875 | 0.59 | NA | 0.00774887 | 0.0400206 | 4018.64 | 15.6048 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.170106 | 0.02 | 0.00590437 | 0.0526731 | 0.0134462 | 1.07665 | 0.102063 | 0.228479 | NA | 0.00666192 | 0.0213034 | 2805.13 | 1.06179 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.167852 | 0.00754468 | 0.00171881 | 0.0598677 | 0.0103644 | 1.23678 | 0.100711 | 0.38134 | NA | 0.00323454 | 0.0052699 | 1276.99 | 0.414828 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.285941 | 0.0040254 | 0.000939005 | 0.0390243 | 0.00561871 | 1.84361 | 0.171565 | 0.436573 | NA | 0.0010467 | 0.00253718 | 376.004 | 1.32604 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.128624 | 0.00500291 | 0.00166447 | 219.462 | 8.07292 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.24589 | 0.0144896 | 0.00161766 | 6384.16 | 4.41089 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.121749 | 0.00650943 | 0.00191606 | 542.222 | 7.00944 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.308241 | 0.00448299 | 0.00132138 | 223.638 | 0.462467 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.270135 | 0.00328179 | 0.000338752 | 586.731 | 7.89394 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.279447 | 0.00431143 | 0.00115688 | 317.644 | 1.08499 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.283962 | 0.02 | 0.00636998 | 437.626 | 2.08648 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.166325 | 0.00431958 | 0.00108267 | 32.1563 | 1.32474 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.128624 | 0.00500291 | 0.00166447 | 219.462 | 8.07292 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.24589 | 0.0144896 | 0.00161766 | 6384.16 | 4.41089 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.121749 | 0.00650943 | 0.00191606 | 542.222 | 7.00944 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.135403 | 0.00669091 | 0.00174552 | 504.375 | 53.0344 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.244189 | 0.0162929 | 0.00139631 | 10374.5 | 14.0793 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.164158 | 0.00920257 | 0.00179306 | 1177.03 | 44.9158 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.213876 | 0.02 | 0.00689031 | 1257.62 | 1.12326 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.111385 | 0.02 | 0.0043463 | 4527.52 | 1.12635 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.185057 | 0.02 | 0.00647651 | 2630.26 | 0.94397 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.147775 | 0.00494246 | 0.00151627 | 127.412 | 0.43938 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.204537 | 0.0116418 | 0.00183865 | 3314.65 | 0.428294 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.151244 | 0.0060498 | 0.00180152 | 388.918 | 0.322502 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.308241 | 0.00448299 | 0.00132138 | 223.638 | 0.462467 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.270135 | 0.00328179 | 0.000338752 | 586.731 | 7.89394 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.279447 | 0.00431143 | 0.00115688 | 317.644 | 1.08499 | none |
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
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.202119 | 0.0075898 | 0.00156171 | 0.0570426 | 0.00939 | 1.1634 | 0.121272 | 0.421208 | NA | 0.00363849 | 0.00383737 | 1787.42 | 3.45529 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.36398 | 0.00419157 | 0.00102363 | 0.0442451 | 0.00579787 | 2.20233 | 0.218388 | 0.423382 | 0.57 | 0.0011133 | 0.00245194 | 446.798 | 1.5757 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.311061 | 0.02 | 0.00628719 | 0.0770925 | 0.0193954 | 1.33205 | 0.186637 | 0.0126042 | NA | 0.00103181 | 0.00182715 | 371.181 | 1.76969 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.218498 | 0.00395737 | 0.00087229 | 0.0345247 | 0.00840623 | 1.02307 | 0.131099 | 0.138979 | 0.310167 | 5.15088e-06 | 5.96754e-05 | 32.9555 | 1.35767 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.202119 | 0.0075898 | 0.00156171 | 0.0570426 | 0.00939 | 1.1634 | 0.121272 | 0.421208 | NA | 0.00363849 | 0.00383737 | 1787.42 | 3.45529 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.207345 | 0.00944195 | 0.00149948 | 0.047151 | 0.00999399 | 0.732248 | 0.124407 | 0.59 | NA | 0.00610725 | 0.0298592 | 3094.62 | 12.0168 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.166009 | 0.02 | 0.00614338 | 0.0468025 | 0.0117834 | 0.940717 | 0.0996054 | 0.228312 | NA | 0.00796169 | 0.0145387 | 2631.45 | 0.996052 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.232204 | 0.00685944 | 0.00153796 | 0.0598677 | 0.00981861 | 1.62015 | 0.139322 | 0.358285 | 0.534 | 0.00226172 | 0.00809958 | 1075.53 | 0.349385 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.36398 | 0.00419157 | 0.00102363 | 0.0442451 | 0.00579787 | 2.20233 | 0.218388 | 0.423382 | 0.57 | 0.0011133 | 0.00245194 | 446.798 | 1.5757 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.15794 | 0.00464136 | 0.00150245 | 147.749 | 5.43494 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.288851 | 0.0124152 | 0.00151379 | 4832.74 | 3.339 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.159567 | 0.00571281 | 0.0016689 | 381.778 | 4.93534 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.370862 | 0.00524667 | 0.00162953 | 275.935 | 0.570613 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.375092 | 0.00322514 | 0.000360356 | 756.7 | 10.1807 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.345986 | 0.00410289 | 0.00108099 | 307.76 | 1.05123 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.311061 | 0.02 | 0.00628719 | 371.181 | 1.76969 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.218498 | 0.00395737 | 0.00087229 | 32.9555 | 1.35767 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.15794 | 0.00464136 | 0.00150245 | 147.749 | 5.43494 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.288851 | 0.0124152 | 0.00151379 | 4832.74 | 3.339 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.159567 | 0.00571281 | 0.0016689 | 381.778 | 4.93534 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.13961 | 0.00580007 | 0.00156891 | 375.666 | 39.5008 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.310623 | 0.0147471 | 0.00132844 | 8110.91 | 11.0074 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.171802 | 0.00777872 | 0.0016011 | 797.28 | 30.4244 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.235679 | 0.02 | 0.00714821 | 1179.19 | 1.05321 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0860046 | 0.02 | 0.00439603 | 4215.44 | 1.04871 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.176343 | 0.02 | 0.00688591 | 2499.72 | 0.897119 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.19308 | 0.00469312 | 0.00133563 | 83.5175 | 0.288009 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.29706 | 0.0102253 | 0.00167292 | 2866.3 | 0.370362 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.206472 | 0.00565985 | 0.00160534 | 276.788 | 0.22952 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.370862 | 0.00524667 | 0.00162953 | 275.935 | 0.570613 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.375092 | 0.00322514 | 0.000360356 | 756.7 | 10.1807 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.345986 | 0.00410289 | 0.00108099 | 307.76 | 1.05123 | none |
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
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.170675 | 0.00814441 | 0.00163472 | 0.0575825 | 0.0100092 | 0.982072 | 0.102405 | 0.432694 | NA | 0.00437558 | 0.00685808 | 2106.67 | 4.07243 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.340589 | 0.00409926 | 0.000975446 | 0.0438154 | 0.0058058 | 2.3014 | 0.204353 | 0.424003 | NA | 0.00101057 | 0.00245417 | 407.164 | 1.43593 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.305654 | 0.02 | 0.00613807 | 0.0772735 | 0.0199954 | 1.49471 | 0.183392 | 0.015875 | 0.567222 | 0.000697559 | 0.00250185 | 373.118 | 1.77892 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.185503 | 0.00407257 | 0.000891976 | 0.0342941 | 0.00813333 | 0.962974 | 0.111302 | 0.143125 | 0.349771 | -2.51352e-06 | 3.2158e-05 | 28.0628 | 1.15611 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.170675 | 0.00814441 | 0.00163472 | 0.0575825 | 0.0100092 | 0.982072 | 0.102405 | 0.432694 | NA | 0.00437558 | 0.00685808 | 2106.67 | 4.07243 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.188418 | 0.0101511 | 0.00156929 | 0.0499064 | 0.010777 | 0.597023 | 0.113051 | 0.59 | NA | 0.00719666 | 0.0364563 | 3634.03 | 14.1113 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.185215 | 0.02 | 0.00585801 | 0.0537644 | 0.0134327 | 1.2132 | 0.111129 | 0.228264 | NA | 0.0065824 | 0.020645 | 2758.14 | 1.044 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.178164 | 0.00711039 | 0.00161149 | 0.0598677 | 0.00993206 | 1.44424 | 0.106899 | 0.367951 | NA | 0.00283371 | 0.00568466 | 1143.3 | 0.371397 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.340589 | 0.00409926 | 0.000975446 | 0.0438154 | 0.0058058 | 2.3014 | 0.204353 | 0.424003 | NA | 0.00101057 | 0.00245417 | 407.164 | 1.43593 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.131758 | 0.00471112 | 0.00154923 | 179.154 | 6.59016 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.253421 | 0.0137084 | 0.00157688 | 5685.98 | 3.92851 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.126846 | 0.00601372 | 0.00177806 | 454.885 | 5.88041 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.349087 | 0.00476322 | 0.00143303 | 248.711 | 0.514315 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.342639 | 0.00334051 | 0.000368286 | 663.044 | 8.92067 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.33004 | 0.00419406 | 0.00112502 | 309.736 | 1.05798 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.305654 | 0.02 | 0.00613807 | 373.118 | 1.77892 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.185503 | 0.00407257 | 0.000891976 | 28.0628 | 1.15611 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.131758 | 0.00471112 | 0.00154923 | 179.154 | 6.59016 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.253421 | 0.0137084 | 0.00157688 | 5685.97 | 3.92851 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.126846 | 0.00601372 | 0.00177806 | 454.885 | 5.88041 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.136371 | 0.00619174 | 0.00164171 | 428.918 | 45.1002 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.262358 | 0.0157239 | 0.00136994 | 9480.3 | 12.8658 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.166524 | 0.00853769 | 0.00169622 | 992.863 | 37.8879 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.221155 | 0.02 | 0.00683444 | 1181.02 | 1.05485 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.134276 | 0.02 | 0.00432005 | 4614.19 | 1.14791 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.200215 | 0.02 | 0.00641953 | 2479.19 | 0.889753 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.151965 | 0.00469331 | 0.00140448 | 100.795 | 0.347589 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.217657 | 0.010916 | 0.00176992 | 3007.9 | 0.388658 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.164871 | 0.00572186 | 0.00166007 | 321.199 | 0.266347 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.349087 | 0.00476322 | 0.00143303 | 248.711 | 0.514315 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.342639 | 0.00334051 | 0.000368286 | 663.044 | 8.92067 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.33004 | 0.00419406 | 0.00112502 | 309.736 | 1.05798 | none |
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
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.243952 | 0.00678616 | 0.00144263 | 0.0558016 | 0.00844079 | 1.64577 | 0.146371 | 0.407028 | NA | 0.00347892 | 0.00193885 | 1358.08 | 2.62531 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.468893 | 0.0036337 | 0.000916429 | 0.0411752 | 0.00519118 | 3.08319 | 0.281336 | 0.403635 | NA | 0.00105457 | 0.00345542 | 432.974 | 1.52695 | none |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.380915 | 0.02 | 0.00593845 | 0.0842882 | 0.0196499 | 1.45578 | 0.228549 | 0.0150833 | 0.59 | 0.00106981 | 0.000487308 | 298.909 | 1.42511 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.255151 | 0.00375908 | 0.00083606 | 0.0344089 | 0.00839246 | 1.2134 | 0.15309 | 0.128542 | 0.273833 | 2.1002e-05 | 2.11568e-05 | 36.1125 | 1.48773 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.243952 | 0.00678616 | 0.00144263 | 0.0558016 | 0.00844079 | 1.64577 | 0.146371 | 0.407028 | NA | 0.00347892 | 0.00193885 | 1358.08 | 2.62531 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.206287 | 0.00848166 | 0.0013975 | 0.0417913 | 0.00899423 | 1.03516 | 0.123772 | 0.59 | NA | 0.00583563 | 0.0264719 | 2362.52 | 9.17395 | none |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.221692 | 0.02 | 0.0061787 | 0.0463605 | 0.0111183 | 0.849445 | 0.133015 | 0.228229 | NA | 0.0094961 | 0.00918921 | 2451.61 | 0.927978 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.302452 | 0.00651632 | 0.0014379 | 0.0598677 | 0.00949674 | 1.97462 | 0.181471 | 0.350104 | 0.46 | 0.00228872 | 0.00520841 | 964.34 | 0.313263 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.468893 | 0.0036337 | 0.000916429 | 0.0411752 | 0.00519118 | 3.08319 | 0.281336 | 0.403635 | NA | 0.00105457 | 0.00345542 | 432.974 | 1.52695 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.194961 | 0.00447719 | 0.00141052 | 114.899 | 4.22656 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.324681 | 0.0107133 | 0.00139043 | 3684.63 | 2.54576 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.212215 | 0.00516797 | 0.00152693 | 274.7 | 3.55111 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.40198 | 0.00493776 | 0.00148306 | 210.933 | 0.436193 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.599673 | 0.0021717 | 0.000239491 | 825.808 | 11.1105 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.405026 | 0.00379164 | 0.00102673 | 262.181 | 0.895542 | none |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.380915 | 0.02 | 0.00593845 | 298.909 | 1.42511 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.255151 | 0.00375908 | 0.00083606 | 36.1125 | 1.48773 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.194961 | 0.00447719 | 0.00141052 | 114.899 | 4.22656 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.324681 | 0.0107133 | 0.00139043 | 3684.63 | 2.54576 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.212215 | 0.00516797 | 0.00152693 | 274.7 | 3.55111 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.13758 | 0.00531987 | 0.00146688 | 314.374 | 33.056 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.310023 | 0.0131117 | 0.00123285 | 6142.57 | 8.33614 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.171259 | 0.00701339 | 0.00149278 | 630.624 | 24.0648 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.323229 | 0.02 | 0.00713787 | 1088.17 | 0.971915 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0579244 | 0.02 | 0.00447076 | 3984.27 | 0.991198 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.283922 | 0.02 | 0.00692746 | 2282.39 | 0.819122 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.217915 | 0.0046585 | 0.00127759 | 72.1007 | 0.248638 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.409908 | 0.00952636 | 0.00156915 | 2604.46 | 0.336529 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.279534 | 0.00536409 | 0.00146697 | 216.46 | 0.179495 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.40198 | 0.00493776 | 0.00148306 | 210.933 | 0.436193 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.599673 | 0.0021717 | 0.000239491 | 825.808 | 11.1105 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.405026 | 0.00379164 | 0.00102673 | 262.181 | 0.895542 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
