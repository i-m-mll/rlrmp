# GRU perturbation-response bank

Issue: `b413bb0`. Source experiment: `33b0dcb`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 18 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 36 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
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

### `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64`

- Evaluated: 110
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.173104 | 0.00386516 | 0.00087576 | 0.0366568 | 0.00516328 | 1.10003 | 0.103862 | 0.385224 | NA | 0.00166528 | 0.00244892 | 314.204 | 1.51727 | 1.5084 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.17057 | 0.00384839 | 0.000862929 | 0.0366141 | 0.00518987 | 1.0999 | 0.102342 | 0.384526 | NA | 0.00162301 | 0.00299606 | 315.743 | 1.5247 | 1.51579 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.202714 | 0.015 | 0.00538852 | 0.050632 | 0.0137512 | 0.610824 | 0.121628 | 0.0297891 | NA | 0.00223116 | 0.000749062 | 249.738 | 2.9767 | 2.49847 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.167931 | 0.00329283 | 0.000926134 | 0.0280711 | 0.00708819 | 0.569596 | 0.100758 | 0.155797 | 0.281234 | 0.000149238 | 0.000134258 | 24.133 | 2.48552 | 1.96349 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.173245 | 0.00386472 | 0.000875741 | 0.0366575 | 0.00516302 | 1.10046 | 0.103947 | 0.385154 | NA | 0.00165842 | 0.00238205 | 314.195 | 1.51723 | 1.50836 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.0229864, 0.0352696, 0.130182 | 0.150799 | 0.00775049 | 0.0012334 | 0.0382005 | 0.00821 | 0.547609 | 0.0904796 | 0.59 | NA | 0.00626471 | 0.021649 | 1417.5 | NA | 13.845 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (6); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (6) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.111452 | 0.015 | 0.00482526 | 0.0322441 | 0.0083847 | 0.477434 | 0.0668714 | 0.232182 | NA | 0.00772554 | 0.00674321 | 1107.83 | 1.04835 | 1.38895 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.166391 | 0.00561665 | 0.00133345 | 0.0449007 | 0.00769874 | 0.951626 | 0.0998347 | 0.374857 | NA | 0.00265768 | 0.00421483 | 495.709 | 0.396545 | 1.4792 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152662, 0.0717786, 0.719702 | 0.195114 | 0.00268455 | 0.00062124 | 0.0258842 | 0.003639 | 0.975572 | 0.117069 | 0.427324 | NA | 0.000884318 | 0.000726037 | 134.772 | 0.540622 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.100701 | 0.00278021 | 0.000935797 | 34.246 | 3.14933 | 2.78486 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.289964 | 0.00581992 | 0.000838365 | 845.724 | 1.45956 | 1.46345 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.128646 | 0.00299536 | 0.00085312 | 62.642 | 2.02444 | 1.80452 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.0997686 | 0.00271423 | 0.000902047 | 32.3062 | 2.97095 | 2.62712 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.285956 | 0.00582996 | 0.000839326 | 850.418 | 1.46766 | 1.47157 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.125986 | 0.00300099 | 0.000847413 | 64.5043 | 2.08463 | 1.85817 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.202714 | 0.015 | 0.00538852 | 249.738 | 2.9767 | 2.49847 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.167931 | 0.00329283 | 0.000926134 | 24.133 | 2.48552 | 1.96349 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.100739 | 0.00278017 | 0.000935746 | 34.241 | 3.14887 | 2.78446 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.29028 | 0.00581843 | 0.000838156 | 845.684 | 1.45949 | 1.46338 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.128716 | 0.00299558 | 0.000853321 | 62.6612 | 2.02507 | 1.80508 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.0229864 | 0.103309 | 0.00507258 | 0.00134321 | 204.538 | NA | 53.4629 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.130182 | 0.221651 | 0.0115764 | 0.00102239 | 3633.2 | NA | 12.4043 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.0352696 | 0.127438 | 0.0066025 | 0.00133461 | 414.764 | NA | 39.7799 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.143564 | 0.015 | 0.00582571 | 604.236 | 1.34932 | 2.81044 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0696138 | 0.015 | 0.00333172 | 1665.82 | 1.03604 | 1.09069 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.121179 | 0.015 | 0.00531834 | 1053.43 | 0.94519 | 1.61943 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.125763 | 0.00391147 | 0.00125154 | 49.298 | 0.41635 | 2.20965 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.242823 | 0.00822267 | 0.00131784 | 1293.7 | 0.412424 | 1.41811 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.130587 | 0.00471581 | 0.00143098 | 144.128 | 0.291176 | 2.03628 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152662, 0.0717786, 0.719702 | 0.180457 | 0.00305475 | 0.000936273 | 70.7241 | 0.372035 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152662, 0.0717786, 0.719702 | 0.223434 | 0.00226481 | 0.000233415 | 247.252 | 0.787457 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152662, 0.0717786, 0.719702 | 0.181452 | 0.00273409 | 0.000694033 | 86.339 | 0.354165 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
