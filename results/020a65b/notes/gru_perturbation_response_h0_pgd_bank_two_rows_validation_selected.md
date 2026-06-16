# GRU perturbation-response bank

Issue: `020a65b`. Source experiment: `020a65b`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 18 |
| `delayed_observation` | 36 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 36 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 36 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 36 |
| `target_aligned_lateral_command_load_pulse` | 6 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

- Evaluated: 146
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.159556 | 0.00329529 | 0.000888211 | 0.0315621 | 0.00458954 | 0.98516 | 0.0957338 | 0.380435 | NA | 0.000332931 | 0.00125128 | 108.434 | 1.48512 | 1.43337 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.165944 | 0.00321859 | 0.000840829 | 0.031407 | 0.00457711 | 1.01075 | 0.0995665 | 0.377411 | NA | 0.000608773 | 0.0011298 | 103.5 | 1.41754 | 1.36814 | none |
| `delayed_observation/delayed_observation_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.100937 | 0.00124601 | 0.000300795 | 0.0126009 | 0.00164227 | 0.553655 | 0.0605623 | 0.425223 | NA | 0.000103159 | 0.000183525 | 44.5482 | 0.0599577 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0695342 | 0.01 | 0.0045527 | 0.0196134 | 0.00726088 | 0.20637 | 0.0417205 | 0.0151953 | NA | 0.00219834 | 0.00100817 | 261.559 | 7.01459 | 5.88763 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.224757 | 0.00674542 | 0.00228462 | 0.0499525 | 0.0113945 | 0.818823 | 0.134854 | 0.173 | NA | 0.000464837 | 0.000171848 | 83.1308 | 2.26993 | 1.79318 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0101007 | 0.000217364 | 5.85808e-05 | 0.00208278 | 0.000304468 | 0.0655502 | 0.00606041 | 0.37962 | NA | 1.42171e-06 | 9.86578e-06 | 0.474173 | 1.49089 | 1.43893 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0339627 | 0.00156524 | 0.000340815 | 0.00616058 | 0.00166208 | 0.0902602 | 0.0203776 | 0.59 | NA | 0.000205305 | 0.000741276 | 27.2634 | 24.7715 | 24.8563 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.257698 | 0.05 | 0.0172408 | 0.0812052 | 0.0237932 | 1.15793 | 0.154619 | 0.229013 | NA | 0.0332768 | 0.0269225 | 13254.5 | 1.12884 | 1.49562 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.188557 | 0.00626 | 0.00174055 | 0.0499101 | 0.00886875 | 1.07953 | 0.113134 | 0.373263 | NA | 0.00157616 | 0.00429068 | 351.31 | 0.367369 | 1.55619 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.100937 | 0.00124601 | 0.000300795 | 0.0126009 | 0.00164227 | 0.553655 | 0.0605623 | 0.425223 | NA | 0.000103159 | 0.000183525 | 44.5482 | 0.0599577 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.163792 | 0.00374684 | 0.00125827 | 59.4912 | 2.49891 | 2.20971 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.158111 | 0.00274549 | 0.000408806 | 186.604 | 1.21461 | 1.21682 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.156766 | 0.00339355 | 0.000997557 | 79.2077 | 1.90393 | 1.69708 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.172254 | 0.00359165 | 0.001163 | 50.4248 | 2.11808 | 1.87295 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.161593 | 0.0027334 | 0.000407622 | 184.894 | 1.20348 | 1.20566 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.163986 | 0.00333072 | 0.000951865 | 75.1814 | 1.80715 | 1.61082 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.0830185 | 0.00143135 | 0.000468091 | 21.8055 | 0.0142987 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.128014 | 0.00104449 | 0.000114488 | 84.4799 | 2.39383 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.0917792 | 0.0012622 | 0.000319808 | 27.3593 | 0.0409148 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0695342 | 0.01 | 0.0045527 | 261.559 | 7.01459 | 5.88763 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.224757 | 0.00674542 | 0.00228462 | 83.1308 | 2.26993 | 1.79318 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0107853 | 0.000247223 | 8.31122e-05 | 0.259508 | 2.50243 | 2.21281 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00919751 | 0.000181006 | 2.68214e-05 | 0.818406 | 1.22292 | 1.22514 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0103193 | 0.000223862 | 6.58088e-05 | 0.344605 | 1.9016 | 1.695 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0461287 | 0.00212471 | 0.000581134 | 35.4089 | 49.1811 | 48.9028 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0181392 | 0.000843453 | 7.64356e-05 | 18.7312 | 10.7702 | 10.8381 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0376202 | 0.00172755 | 0.000364875 | 27.6502 | 32.8134 | 32.9884 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.326261 | 0.05 | 0.0216496 | 9116.66 | 1.83211 | 3.81634 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.159336 | 0.05 | 0.0111981 | 17872.8 | 1.00043 | 1.05319 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.287498 | 0.05 | 0.0188746 | 12774.1 | 1.03151 | 1.76737 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.20372 | 0.00666534 | 0.00231002 | 159.765 | 0.482352 | 2.50778 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.178155 | 0.00565688 | 0.000920933 | 632.4 | 0.387955 | 1.31647 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.183798 | 0.00645778 | 0.00199071 | 261.765 | 0.28843 | 1.9657 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.0830185 | 0.00143135 | 0.000468091 | 21.8055 | 0.0142987 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.128014 | 0.00104449 | 0.000114488 | 84.4799 | 2.39383 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.0917792 | 0.0012622 | 0.000319808 | 27.3593 | 0.0409148 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64`

- Evaluated: 146
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.235263 | 0.00270887 | 0.000438318 | 0.0301633 | 0.00459987 | 1.33976 | 0.141158 | 0.355177 | 0.42846 | 0.0004297 | 1.71318e-05 | 83.235 | 1.13999 | 1.10027 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.233149 | 0.00266892 | 0.000422867 | 0.0300171 | 0.0045449 | 1.3564 | 0.139889 | 0.35426 | 0.411985 | 0.000441344 | 4.00813e-05 | 83.1784 | 1.13922 | 1.09952 | none |
| `delayed_observation/delayed_observation_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.286759 | 0.0018161 | 0.000263499 | 0.0239353 | 0.00309267 | 1.44055 | 0.172056 | 0.384542 | 0.463023 | 0.000223901 | 0.000200719 | 92.651 | 0.1247 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.26357 | 0.01 | 0.00242237 | 0.0538514 | 0.0111631 | 0.811913 | 0.158142 | 0.0153828 | 0.335484 | 5.6472e-06 | 7.14783e-06 | 46.5069 | 1.24724 | 1.04686 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.336623 | 0.00623187 | 0.00130208 | 0.0499525 | 0.0124027 | 1.19403 | 0.201974 | 0.149727 | 0.33682 | 7.35726e-06 | 9.0383e-06 | 46.0557 | 1.25758 | 0.993451 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0155183 | 0.000178764 | 2.88862e-05 | 0.00199043 | 0.000303412 | 0.0885724 | 0.00931096 | 0.355466 | 0.426323 | 2.11625e-06 | 7.66429e-08 | 0.36204 | 1.13832 | 1.09865 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0340932 | 0.00131606 | 0.000243437 | 0.00621268 | 0.00145584 | 0.133463 | 0.0204559 | 0.59 | NA | 0.000388312 | 0.00265498 | 26.0556 | 23.674 | 23.755 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 1.49405 | 0.05 | 0.0114319 | 0.251316 | 0.0394318 | 6.28962 | 0.896427 | 0.228781 | 0.434981 | 0.0158462 | 0.00592667 | 10092.8 | 0.85957 | 1.13885 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.347313 | 0.00560642 | 0.000990753 | 0.0507392 | 0.00942112 | 1.72717 | 0.208388 | 0.355784 | 0.427573 | 0.00138734 | 0.000100529 | 287.306 | 0.300439 | 1.27268 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.286759 | 0.0018161 | 0.000263499 | 0.0239353 | 0.00309267 | 1.44055 | 0.172056 | 0.384542 | 0.463023 | 0.000223901 | 0.000200719 | 92.651 | 0.1247 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.281369 | 0.00301352 | 0.000513387 | 31.0974 | 1.30624 | 1.15506 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.135364 | 0.00263223 | 0.000396106 | 164.26 | 1.06918 | 1.07111 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.289057 | 0.00248087 | 0.000405459 | 54.3477 | 1.30637 | 1.16444 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.280212 | 0.00291707 | 0.000479582 | 31.2852 | 1.31413 | 1.16204 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.133251 | 0.00264234 | 0.000397123 | 165.033 | 1.07421 | 1.07615 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.285984 | 0.00244735 | 0.000391895 | 53.2174 | 1.2792 | 1.14022 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.255154 | 0.00182315 | 0.000308436 | 28.3114 | 0.0185648 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.20214 | 0.001534 | 0.000174482 | 148.656 | 4.21232 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.402985 | 0.00209115 | 0.000307579 | 100.986 | 0.151021 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.26357 | 0.01 | 0.00242237 | 46.5069 | 1.24724 | 1.04686 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.336623 | 0.00623187 | 0.00130208 | 46.0557 | 1.25758 | 0.993451 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0186217 | 0.000198856 | 3.38268e-05 | 0.136352 | 1.31484 | 1.16267 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00886995 | 0.000173805 | 2.61591e-05 | 0.714523 | 1.06769 | 1.06963 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0190631 | 0.00016363 | 2.66727e-05 | 0.235245 | 1.29813 | 1.15709 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.050775 | 0.00154001 | 0.00037549 | 27.8744 | 38.7161 | 38.4969 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0136855 | 0.000907416 | 7.86721e-05 | 22.5277 | 12.9531 | 13.0347 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0378189 | 0.00150074 | 0.000276151 | 27.7646 | 32.9491 | 33.1249 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 1.78454 | 0.05 | 0.011882 | 3261.53 | 0.655447 | 1.36532 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.508992 | 0.05 | 0.0109071 | 17899 | 1.0019 | 1.05473 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 2.18861 | 0.05 | 0.0115065 | 9117.88 | 0.736267 | 1.26151 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.428226 | 0.005897 | 0.00111467 | 94.9839 | 0.286769 | 1.49094 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.15241 | 0.00568355 | 0.000936988 | 578.492 | 0.354883 | 1.20425 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.461304 | 0.0052387 | 0.000920599 | 188.443 | 0.207639 | 1.41509 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.255154 | 0.00182315 | 0.000308436 | 28.3114 | 0.0185648 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.20214 | 0.001534 | 0.000174482 | 148.656 | 4.21232 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.402985 | 0.00209115 | 0.000307579 | 100.986 | 0.151021 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
