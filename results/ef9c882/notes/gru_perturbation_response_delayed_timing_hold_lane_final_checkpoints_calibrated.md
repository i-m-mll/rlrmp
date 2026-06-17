# GRU perturbation-response bank

Issue: `ef9c882`. Source experiment: `ef9c882`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 54 |
| `delayed_observation` | 108 |
| `initial_state` | 24 |
| `process_epsilon` | 144 |
| `sensory_feedback` | 108 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 36 |
| `delayed_observation_offset` | 108 |
| `initial_position_offset` | 12 |
| `initial_velocity_offset` | 12 |
| `process_epsilon_force_state_xy` | 36 |
| `process_epsilon_integrator_xy` | 36 |
| `process_epsilon_position_xy` | 36 |
| `process_epsilon_velocity_xy` | 36 |
| `sensory_feedback_offset` | 108 |
| `target_aligned_lateral_command_load_pulse` | 18 |
| `target_stream_jump` | 1 |

## Evaluation

### `hold__force_filter`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.279232 | 0.0180342 | 0.00740724 | 0.0884771 | 0.0254653 | 1.74462 | 0.251309 | 0.727296 | 0.581448 | 0.0142727 | 0.00180387 | 6973.02 | 13.4796 | 13.3901 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.257136 | 0.0165113 | 0.00679128 | 0.0790486 | 0.0232326 | 1.74217 | 0.231422 | 0.76338 | 0.512 | 0.0131716 | 0.00335856 | 5178.07 | 10.0098 | 9.94334 | inflated_ratio |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.11678 | 0.00577648 | 0.00220368 | 0.0280189 | 0.00836891 | 0.742491 | 0.105102 | 0.794707 | 0.67747 | 0.00387317 | 0.000527485 | 1229.4 | 0.0151974 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.1287 | 0.0200221 | 0.0149336 | 0.0445031 | 0.0203781 | 1.02146 | 0.11583 | 0.258859 | 0.671508 | 0.00858814 | 0.000180078 | 1830.64 | 8.72795 | 7.32572 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.229511 | 0.0163526 | 0.00927025 | 0.0651219 | 0.0268586 | 1.63736 | 0.20656 | 0.604786 | 0.617891 | 0.0115233 | -2.1713e-05 | 4346.45 | 179.061 | 141.453 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.279486 | 0.0180205 | 0.00740422 | 0.0884664 | 0.0254505 | 1.74981 | 0.251537 | 0.726793 | 0.58277 | 0.0143299 | 0.00166768 | 6914.81 | 13.3671 | 13.2784 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.198572 | 0.0461805 | 0.0119309 | 0.120837 | 0.0494158 | 0.757548 | 0.178715 | 0.89 | NA | 0.0417828 | 0.089938 | 35157.1 | 136.519 | 137.354 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.0725068 | 0.0200089 | 0.0122039 | 0.0322727 | 0.0149447 | 0.576025 | 0.0652561 | 0.342674 | 0.670714 | 0.0114609 | 0.000195775 | 2801.11 | 1.06027 | 1.40476 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.196468 | 0.0204565 | 0.008889 | 0.0784789 | 0.0278772 | 1.35464 | 0.176821 | 0.739153 | 0.595558 | 0.0161837 | 0.0046751 | 6435.17 | 2.09045 | 7.68104 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.11678 | 0.00577648 | 0.00220368 | 0.0280189 | 0.00836891 | 0.742491 | 0.105102 | 0.794707 | 0.67747 | 0.00387317 | 0.000527485 | 1229.4 | 0.0151974 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.167186 | 0.0101601 | 0.00523298 | 2156.91 | 79.3418 | 70.1593 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.163883 | 0.0203998 | 0.00732785 | 4641.64 | 3.20696 | 3.21277 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.506628 | 0.0235426 | 0.00966089 | 14120.5 | 182.539 | 162.708 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.178365 | 0.0115125 | 0.00574645 | 3690.97 | 135.772 | 120.059 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.170698 | 0.0178749 | 0.00656775 | 3489.44 | 2.4109 | 2.41526 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.422345 | 0.0201466 | 0.00805962 | 8353.8 | 107.992 | 96.259 | inflated_ratio; inflated_ratio |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.150326 | 0.00483504 | 0.00221398 | 1017.25 | 0.00607755 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.0429958 | 0.00351196 | 0.000938889 | 370.956 | 0.123055 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.157017 | 0.00898243 | 0.00345818 | 2300 | 0.0318143 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.1287 | 0.0200221 | 0.0149336 | 1830.64 | 8.72795 | 7.32572 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.229511 | 0.0163526 | 0.00927025 | 4346.45 | 179.061 | 141.453 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.169194 | 0.0101875 | 0.00524849 | 2055.3 | 75.6042 | 66.8542 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.163843 | 0.0204008 | 0.00732812 | 4645.57 | 3.20968 | 3.21549 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.50542 | 0.0234732 | 0.00963606 | 14043.6 | 181.545 | 161.821 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.133491 | 0.0266448 | 0.00927795 | 7844.34 | 824.822 | 820.153 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.317622 | 0.0768145 | 0.0159397 | 83640.7 | 113.51 | 114.225 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.144603 | 0.0350821 | 0.010575 | 13986.1 | 533.715 | 536.562 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.121407 | 0.0200127 | 0.0136941 | 1916.27 | 1.71155 | 3.56521 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.00539056 | 0.0200122 | 0.0105232 | 4186.78 | 1.04158 | 1.09651 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0907229 | 0.0200018 | 0.0123944 | 2300.29 | 0.825546 | 1.41448 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.25396 | 0.018093 | 0.00935453 | 6419.03 | 22.1359 | 115.086 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.103715 | 0.0257438 | 0.00907741 | 7087.89 | 0.915845 | 3.1078 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.231728 | 0.0175327 | 0.00823507 | 5798.6 | 4.80835 | 32.7697 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.150326 | 0.00483504 | 0.00221398 | 1017.25 | 0.00607755 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.0429958 | 0.00351196 | 0.000938889 | 370.956 | 0.123055 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.157017 | 0.00898243 | 0.00345818 | 2300 | 0.0318143 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `hold__start_pos_zero_vel`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.28244 | 0.00529389 | 0.00191804 | 0.0508328 | 0.0109047 | 1.98748 | 0.254196 | 0.438551 | 0.627126 | 0.00177132 | 0.000387947 | 355.324 | 0.68688 | 0.682321 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.285386 | 0.00469702 | 0.0017556 | 0.0471363 | 0.00995025 | 2.12761 | 0.256848 | 0.423352 | 0.628164 | 0.00152228 | 0.000220928 | 261.884 | 0.506251 | 0.502891 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.299385 | 0.00512312 | 0.00199717 | 0.0416172 | 0.00989694 | 1.82116 | 0.269447 | 0.50254 | 0.624819 | 0.0024256 | 0.00079633 | 590.22 | 0.00729608 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.175204 | 0.02 | 0.0137865 | 0.0454976 | 0.0205581 | 1.77358 | 0.157684 | 0.0135885 | 0.541379 | 0.00652738 | 0.00173682 | 1293.61 | 6.16758 | 5.17671 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.296588 | 0.00790019 | 0.00412887 | 0.0549329 | 0.0170403 | 1.89105 | 0.266929 | 0.328141 | 0.507787 | 0.00390169 | 0.00100492 | 818.156 | 33.7056 | 26.6265 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.281934 | 0.00526488 | 0.00190718 | 0.0506449 | 0.010857 | 1.98652 | 0.25374 | 0.438088 | 0.616222 | 0.0017539 | 0.000421846 | 348.742 | 0.674156 | 0.669681 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.344608 | 0.0144332 | 0.00491239 | 0.0524635 | 0.017893 | 1.49521 | 0.310148 | 0.77399 | 0.639306 | 0.0111802 | 0.013817 | 3850.48 | 14.9518 | 15.0433 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.130708 | 0.02 | 0.0108054 | 0.0385879 | 0.01594 | 1.21648 | 0.117637 | 0.228672 | 0.586158 | 0.00739624 | 0.00273714 | 1643.65 | 0.622153 | 0.824296 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.441573 | 0.0113455 | 0.00435972 | 0.090852 | 0.0239704 | 3.04337 | 0.397416 | 0.451043 | 0.584276 | 0.00562386 | 0.00244655 | 1756.31 | 0.570534 | 2.09634 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.299385 | 0.00512312 | 0.00199717 | 0.0416172 | 0.00989694 | 1.82116 | 0.269447 | 0.50254 | 0.624819 | 0.0024256 | 0.00079633 | 590.22 | 0.00729608 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.244058 | 0.00374906 | 0.00162021 | 233.994 | 8.60745 | 7.61128 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.366513 | 0.00783035 | 0.00216981 | 476.975 | 0.329548 | 0.330145 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.23675 | 0.00430225 | 0.0019641 | 355.003 | 4.58922 | 4.09062 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.24714 | 0.00325402 | 0.00145188 | 142.298 | 5.23442 | 4.62863 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.378156 | 0.00721256 | 0.0021311 | 464.864 | 0.32118 | 0.321762 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.230864 | 0.00362448 | 0.00168382 | 178.491 | 2.3074 | 2.05672 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.4004 | 0.00627792 | 0.00271625 | 1071.59 | 0.00640221 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.209165 | 0.00358257 | 0.0011154 | 217.994 | 0.0723137 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.288591 | 0.00550887 | 0.00215987 | 481.079 | 0.00665445 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.175204 | 0.02 | 0.0137865 | 1293.61 | 6.16758 | 5.17671 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.296588 | 0.00790019 | 0.00412887 | 818.156 | 33.7056 | 26.6265 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.242794 | 0.00368686 | 0.00159592 | 222.16 | 8.17215 | 7.22636 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.366925 | 0.00783083 | 0.00216972 | 475.458 | 0.328499 | 0.329094 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.236083 | 0.00427695 | 0.00195591 | 348.607 | 4.50653 | 4.01692 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.255795 | 0.00614293 | 0.0029862 | 588.526 | 61.8828 | 61.5325 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.536879 | 0.0295487 | 0.00822791 | 10361.6 | 14.0618 | 14.1504 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.241152 | 0.00760785 | 0.00352306 | 601.332 | 22.947 | 23.0694 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.164276 | 0.02 | 0.0124656 | 1298.85 | 1.16009 | 2.4165 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.104373 | 0.02 | 0.00869249 | 2139.15 | 0.532173 | 0.560238 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.123475 | 0.02 | 0.011258 | 1492.97 | 0.535808 | 0.918046 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.403619 | 0.0105726 | 0.00488387 | 1816.18 | 6.26308 | 32.5622 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.43135 | 0.010583 | 0.00252924 | 673.846 | 0.0870695 | 0.295459 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.48975 | 0.012881 | 0.00566605 | 2778.91 | 2.30435 | 15.7045 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.4004 | 0.00627792 | 0.00271625 | 1071.59 | 0.00640221 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.209165 | 0.00358257 | 0.0011154 | 217.994 | 0.0723137 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00778064, 0.0155613, 0.0354896, 0.0389032, 0.0709792, 0.177448, 0.337641, 0.675282, 1.6882 | 0.288591 | 0.00550887 | 0.00215987 | 481.079 | 0.00665445 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
