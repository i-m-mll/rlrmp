# GRU perturbation-response bank

Issue: `bf71d86`. Source experiment: `bf71d86`.

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

### `timing__fixed_go10`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.152414 | 0.0117007 | 0.00327206 | 0.056414 | 0.0134657 | 0.743377 | 0.10669 | 0.646176 | 0.548 | 0.00709637 | 0.00703733 | 2233.45 | 4.3175 | 4.28885 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.138693 | 0.0117795 | 0.00329062 | 0.0554595 | 0.0133299 | 0.698845 | 0.0970849 | 0.666892 | NA | 0.0075589 | 0.00788103 | 2269.29 | 4.38679 | 4.35767 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.173319 | 0.0079507 | 0.00265437 | 0.0403157 | 0.00968281 | 0.892945 | 0.121323 | 0.646962 | 0.557778 | 0.00578312 | 0.00322043 | 4205.7 | 0.04681 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.0696891 | 0.020032 | 0.0129107 | 0.0253868 | 0.0140887 | 0.452481 | 0.0487824 | 0.123924 | 0.484722 | 0.0117718 | 0.000289973 | 3141.63 | 14.9784 | 12.572 | inflated_ratio; inflated_ratio |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.221908 | 0.0117976 | 0.00531121 | 0.0498426 | 0.0170337 | 1.36589 | 0.155336 | 0.608034 | 0.38803 | 0.00754472 | 0.000426279 | 1469.08 | 60.5219 | 47.8106 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.152547 | 0.0116953 | 0.00327092 | 0.056427 | 0.0134638 | 0.744057 | 0.106783 | 0.646322 | 0.5425 | 0.0070076 | 0.00705285 | 2234.32 | 4.31918 | 4.29051 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.133204 | 0.0263222 | 0.00462271 | 0.101963 | 0.0278145 | 0.481329 | 0.0932431 | 0.69 | NA | 0.020895 | 0.0834937 | 15582 | 60.5068 | 60.877 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.00918162 | 0.0200003 | 0.00981855 | 0.02003 | 0.00986861 | 0.0358053 | 0.00642713 | 0.244897 | NA | 0.0141025 | 0.000391486 | 4119.82 | 1.55943 | 2.0661 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.0856165 | 0.0168781 | 0.00504631 | 0.0598677 | 0.0186645 | 0.322303 | 0.0599315 | 0.675145 | NA | 0.0117692 | 0.00654049 | 3643.11 | 1.18345 | 4.34842 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.173319 | 0.0079507 | 0.00265437 | 0.0403157 | 0.00968281 | 0.892945 | 0.121323 | 0.646962 | 0.557778 | 0.00578312 | 0.00322043 | 4205.7 | 0.04681 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.126198 | 0.00775083 | 0.00309493 | 613.5 | 22.5676 | 19.9558 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.23855 | 0.0166205 | 0.00315808 | 4804.18 | 3.31926 | 3.32528 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.0924948 | 0.0107309 | 0.00356317 | 1282.67 | 16.5814 | 14.78 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.0922172 | 0.00881523 | 0.00347169 | 810.227 | 29.8042 | 26.3548 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.231664 | 0.0169276 | 0.00317753 | 4982.41 | 3.44241 | 3.44865 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.0921971 | 0.00959573 | 0.00322263 | 1015.23 | 13.1241 | 11.6983 | inflated_ratio; inflated_ratio |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.383224 | 0.016683 | 0.00627238 | 11797.2 | 0.0634594 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.0517133 | 0.00225544 | 0.000300252 | 200.894 | 0.0600565 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.0850185 | 0.00491362 | 0.00139048 | 618.973 | 0.00770908 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.0696891 | 0.020032 | 0.0129107 | 3141.63 | 14.9784 | 12.572 | inflated_ratio; inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.221908 | 0.0117976 | 0.00531121 | 1469.08 | 60.5219 | 47.8106 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.126578 | 0.00773629 | 0.00309203 | 615.016 | 22.6234 | 20.0051 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.23856 | 0.0166207 | 0.00315813 | 4806.27 | 3.32071 | 3.32673 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.0925034 | 0.010729 | 0.00356261 | 1281.67 | 16.5684 | 14.7683 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.0932698 | 0.0198037 | 0.00488974 | 4895.12 | 514.716 | 511.802 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.194346 | 0.0366177 | 0.00436517 | 34579.4 | 46.9281 | 47.2237 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.111998 | 0.022545 | 0.00461323 | 7271.53 | 277.483 | 278.964 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0128732 | 0.020001 | 0.0123751 | 3986.48 | 3.5606 | 7.41683 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.00665925 | 0.02 | 0.00657879 | 4234.27 | 1.05339 | 1.10895 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0080124 | 0.02 | 0.0105018 | 4138.69 | 1.48533 | 2.54494 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.0604062 | 0.0142043 | 0.00573378 | 2118.13 | 7.30435 | 37.9759 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.129591 | 0.0208154 | 0.00416005 | 6207.77 | 0.802122 | 2.7219 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.0668526 | 0.0156146 | 0.00524511 | 2603.42 | 2.15883 | 14.7128 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.383224 | 0.016683 | 0.00627238 | 11797.2 | 0.0634594 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.0517133 | 0.00225544 | 0.000300252 | 200.894 | 0.0600565 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.0850185 | 0.00491362 | 0.00139048 | 618.973 | 0.00770908 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `timing__fixed_go20`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.251067 | 0.0141375 | 0.00514836 | 0.0738499 | 0.0191292 | 1.62628 | 0.200854 | 0.614169 | 0.585385 | 0.00868326 | 0.00248351 | 3282.78 | 6.34598 | 6.30386 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.250206 | 0.0135553 | 0.00496025 | 0.0712064 | 0.01869 | 1.6434 | 0.200165 | 0.614521 | NA | 0.00812316 | 0.00304567 | 2999.04 | 5.79747 | 5.75899 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.14403 | 0.00524652 | 0.00188558 | 0.0293639 | 0.007111 | 0.940759 | 0.115224 | 0.620724 | 0.722014 | 0.00307427 | 0.00097045 | 1294.29 | 0.0144057 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.139412 | 0.0200019 | 0.0135402 | 0.0400583 | 0.017835 | 1.07866 | 0.11153 | 0.11787 | 0.476 | 0.00929423 | 0.00118946 | 2295.71 | 10.9453 | 9.18684 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.292097 | 0.0172711 | 0.00858573 | 0.0729621 | 0.0264146 | 2.09256 | 0.233678 | 0.585594 | 0.439615 | 0.0122299 | 0.00320211 | 3684.94 | 151.809 | 119.924 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.251139 | 0.0141456 | 0.00515232 | 0.0738959 | 0.0191382 | 1.62779 | 0.200911 | 0.613996 | 0.528636 | 0.00870602 | 0.00248408 | 3281.45 | 6.3434 | 6.3013 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.227162 | 0.0265855 | 0.00632609 | 0.0865099 | 0.0286767 | 0.803497 | 0.18173 | 0.767534 | NA | 0.0207414 | 0.0506784 | 15064.2 | 58.4959 | 58.8538 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.0407085 | 0.0200213 | 0.0111842 | 0.02343 | 0.0120308 | 0.293835 | 0.0325668 | 0.295061 | 0.549091 | 0.012185 | 0.00170215 | 3495.55 | 1.32313 | 1.75303 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.176202 | 0.015222 | 0.00583197 | 0.0665155 | 0.0203216 | 1.12167 | 0.140961 | 0.600088 | 0.502153 | 0.00865621 | 0.0030427 | 2698.8 | 0.876698 | 3.22129 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.14403 | 0.00524652 | 0.00188558 | 0.0293639 | 0.007111 | 0.940759 | 0.115224 | 0.620724 | 0.722014 | 0.00307427 | 0.00097045 | 1294.29 | 0.0144057 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.383875 | 0.0183951 | 0.0076505 | 6219.37 | 228.779 | 202.302 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.238377 | 0.0145606 | 0.00412444 | 2687.61 | 1.8569 | 1.86026 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.130949 | 0.00945678 | 0.00367015 | 941.369 | 12.1693 | 10.8472 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.356462 | 0.0165369 | 0.00702277 | 5368.58 | 197.483 | 174.628 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.254665 | 0.0142524 | 0.0039186 | 2656.42 | 1.83536 | 1.83868 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.139492 | 0.00987664 | 0.00393938 | 972.106 | 12.5667 | 11.2014 | inflated_ratio; inflated_ratio |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.217054 | 0.00908358 | 0.00352165 | 2935.83 | 0.0157924 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.100895 | 0.00249 | 0.000575554 | 154.959 | 0.0463246 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.114141 | 0.00416599 | 0.00155954 | 792.085 | 0.00986512 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.139412 | 0.0200019 | 0.0135402 | 2295.71 | 10.9453 | 9.18684 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.292097 | 0.0172711 | 0.00858573 | 3684.94 | 151.809 | 119.924 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.384577 | 0.0184241 | 0.0076618 | 6219.26 | 228.775 | 202.298 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.238121 | 0.0145561 | 0.00412413 | 2684.55 | 1.85479 | 1.85815 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.13072 | 0.0094566 | 0.00367103 | 940.542 | 12.1586 | 10.8377 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.138998 | 0.0164636 | 0.00560096 | 3790.93 | 398.611 | 396.355 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.37233 | 0.0435708 | 0.00772727 | 35656 | 48.3891 | 48.6939 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.170159 | 0.019722 | 0.00565005 | 5745.61 | 219.254 | 220.424 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0853962 | 0.0200003 | 0.0129338 | 2703.82 | 2.41496 | 5.03044 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0158624 | 0.0200304 | 0.00843319 | 4020.67 | 1.00026 | 1.053 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0208668 | 0.0200333 | 0.0121856 | 3762.16 | 1.35019 | 2.3134 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.210845 | 0.0147842 | 0.00695775 | 2598.38 | 8.96047 | 46.5861 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.200696 | 0.0177367 | 0.0051108 | 3729.86 | 0.481945 | 1.63542 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.117064 | 0.0131453 | 0.00542736 | 1768.16 | 1.46621 | 9.99245 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.217054 | 0.00908358 | 0.00352165 | 2935.83 | 0.0157924 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.100895 | 0.00249 | 0.000575554 | 154.959 | 0.0463246 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.114141 | 0.00416599 | 0.00155954 | 792.085 | 0.00986512 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `timing__go10_15`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.194323 | 0.00854671 | 0.00302351 | 0.0561957 | 0.0118861 | 0.975199 | 0.145742 | 0.524137 | 0.726667 | 0.00351731 | 0.0077203 | 1370.97 | 2.65024 | 2.63265 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.19388 | 0.00881095 | 0.00310956 | 0.0564143 | 0.0122271 | 0.92183 | 0.14541 | 0.553116 | NA | 0.00424605 | 0.00782699 | 1475.09 | 2.85151 | 2.83258 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.212513 | 0.0079007 | 0.0031096 | 0.0461174 | 0.0106097 | 1.22852 | 0.159385 | 0.570063 | 0.628647 | 0.00555075 | 0.00297547 | 3745.77 | 0.041691 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.0951604 | 0.0200132 | 0.0131319 | 0.0298182 | 0.0153902 | 0.782213 | 0.0713703 | 0.0587813 | 0.43988 | 0.00992955 | 0.000869018 | 2815.18 | 13.422 | 11.2656 | inflated_ratio; inflated_ratio |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.312594 | 0.0114711 | 0.00576931 | 0.06416 | 0.0199791 | 2.33245 | 0.234445 | 0.450206 | 0.371418 | 0.00913336 | 0.000350197 | 2197.75 | 90.5409 | 71.5248 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.194341 | 0.00854334 | 0.00302416 | 0.0561922 | 0.0118838 | 0.974111 | 0.145756 | 0.524286 | 0.7275 | 0.00351877 | 0.007704 | 1367.71 | 2.64394 | 2.62639 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.221139 | 0.0226353 | 0.0049622 | 0.075117 | 0.0243057 | 0.806441 | 0.165854 | 0.739153 | NA | 0.0168732 | 0.0492839 | 9418.97 | 36.5749 | 36.7987 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.01894 | 0.02 | 0.0105621 | 0.0205127 | 0.0108713 | 0.0749901 | 0.014205 | 0.228806 | NA | 0.0124362 | 0.00173093 | 3860.95 | 1.46144 | 1.93627 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.143333 | 0.0119186 | 0.00459814 | 0.0605582 | 0.0155792 | 0.618821 | 0.107499 | 0.5595 | 0.67 | 0.00611184 | 0.00429177 | 2098.11 | 0.681564 | 2.5043 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.212513 | 0.0079007 | 0.0031096 | 0.0461174 | 0.0106097 | 1.22852 | 0.159385 | 0.570063 | 0.628647 | 0.00555075 | 0.00297547 | 3745.77 | 0.041691 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.108124 | 0.00595678 | 0.00286578 | 383.96 | 14.1239 | 12.4893 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.359058 | 0.0119523 | 0.0030136 | 3088.34 | 2.13377 | 2.13763 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.115787 | 0.0077311 | 0.00319116 | 640.625 | 8.28152 | 7.38178 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.101392 | 0.00699193 | 0.00339286 | 527.948 | 19.4206 | 17.173 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.348698 | 0.0124839 | 0.00312193 | 3405.23 | 2.35271 | 2.35698 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.131548 | 0.00695701 | 0.0028139 | 492.088 | 6.36134 | 5.67022 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.413788 | 0.0172509 | 0.00750798 | 10361.9 | 0.0557387 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.136899 | 0.00293717 | 0.000550813 | 642.624 | 0.19211 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.0868519 | 0.003514 | 0.00127002 | 232.764 | 0.00289899 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.0951604 | 0.0200132 | 0.0131319 | 2815.18 | 13.422 | 11.2656 | inflated_ratio; inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.312594 | 0.0114711 | 0.00576931 | 2197.75 | 90.5409 | 71.5248 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.107927 | 0.00595901 | 0.00286862 | 383.701 | 14.1144 | 12.4809 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.359246 | 0.011943 | 0.00301366 | 3079.03 | 2.12734 | 2.13119 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.115851 | 0.00772804 | 0.0031902 | 640.408 | 8.27872 | 7.37928 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.127 | 0.0144794 | 0.00465575 | 2212.99 | 232.693 | 231.375 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.374511 | 0.0360245 | 0.00560502 | 22619.6 | 30.6973 | 30.8906 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.161905 | 0.017402 | 0.00462583 | 3424.33 | 130.673 | 131.37 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.019357 | 0.02 | 0.0130145 | 3645.59 | 3.25612 | 6.7826 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0207053 | 0.02 | 0.00745091 | 4105.3 | 1.02131 | 1.07517 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0167578 | 0.02 | 0.0112208 | 3831.95 | 1.37524 | 2.35632 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.0811175 | 0.00982217 | 0.00504939 | 1007.49 | 3.47433 | 18.0633 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.251337 | 0.0154592 | 0.00418129 | 4117.79 | 0.53207 | 1.80551 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.0975436 | 0.0104743 | 0.00456373 | 1169.04 | 0.969398 | 6.60661 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.413788 | 0.0172509 | 0.00750798 | 10361.9 | 0.0557387 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.136899 | 0.00293717 | 0.000550813 | 642.624 | 0.19211 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.0077406, 0.0154812, 0.0361112, 0.038703, 0.0722224, 0.180556, 0.355861, 0.711722, 1.77931 | 0.0868519 | 0.003514 | 0.00127002 | 232.764 | 0.00289899 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
