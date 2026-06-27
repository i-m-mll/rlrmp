# GRU perturbation-response bank

Issue: `3244f1a`. Source experiment: `33b0dcb`.

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
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.173374 | 0.00386483 | 0.000875587 | 0.0366555 | 0.00516415 | 1.09926 | 0.104024 | 0.385208 | NA | 0.00160262 | 0.00238435 | 313.871 | NA | 1.5068 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.170605 | 0.0038499 | 0.000863066 | 0.0366095 | 0.00519159 | 1.09849 | 0.102363 | 0.384625 | NA | 0.00154804 | 0.00305058 | 315.77 | NA | 1.51591 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.202537 | 0.015 | 0.00539049 | 0.0505942 | 0.0137463 | 0.610861 | 0.121522 | 0.027875 | NA | 0.00214029 | 0.000649612 | 250.026 | NA | 2.50135 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.167932 | 0.00329305 | 0.000925972 | 0.0280466 | 0.00708345 | 0.569594 | 0.100759 | 0.155688 | 0.281 | 0.000130075 | 0.000131465 | 23.9932 | NA | 1.95212 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.173154 | 0.00386241 | 0.000875643 | 0.036657 | 0.00515922 | 1.10341 | 0.103893 | 0.385146 | NA | 0.00158382 | 0.00248665 | 313.55 | NA | 1.50526 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.0229864, 0.0352696, 0.130182 | 0.150706 | 0.00775236 | 0.00123351 | 0.038201 | 0.00821224 | 0.544372 | 0.0904238 | 0.59 | NA | 0.00615546 | 0.0220407 | 1417.74 | NA | 13.8473 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (6); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (6) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.11113 | 0.015 | 0.00482567 | 0.0322364 | 0.00838203 | 0.477251 | 0.066678 | 0.231937 | NA | 0.00761484 | 0.00702508 | 1107.72 | NA | 1.38881 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.166248 | 0.00561491 | 0.00133349 | 0.0449007 | 0.00769587 | 0.954631 | 0.0997488 | 0.374917 | NA | 0.00257375 | 0.00437268 | 495.198 | NA | 1.47767 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0153291, 0.0717096, 0.721063 | 0.195581 | 0.00268612 | 0.000621387 | 0.025903 | 0.00364181 | 0.975796 | 0.117349 | 0.427264 | NA | 0.000840653 | 0.000682207 | 134.738 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.100812 | 0.00278058 | 0.000935631 | 34.2186 | NA | 2.78263 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.290472 | 0.00581895 | 0.000838147 | 844.713 | NA | 1.4617 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.128836 | 0.00299497 | 0.000852984 | 62.6823 | NA | 1.80569 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.0997451 | 0.00271471 | 0.000902395 | 32.289 | NA | 2.62572 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.28605 | 0.00583383 | 0.000839579 | 850.498 | NA | 1.47171 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.12602 | 0.00300117 | 0.000847225 | 64.5216 | NA | 1.85867 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.202537 | 0.015 | 0.00539049 | 250.026 | NA | 2.50135 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.167932 | 0.00329305 | 0.000925972 | 23.9932 | NA | 1.95212 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.100844 | 0.0027812 | 0.000935813 | 34.2292 | NA | 2.7835 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.290019 | 0.00581083 | 0.000837365 | 843.846 | NA | 1.4602 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.1286 | 0.00299521 | 0.00085375 | 62.5735 | NA | 1.80255 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.0229864 | 0.103362 | 0.00507501 | 0.00134338 | 204.768 | NA | 53.523 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.130182 | 0.22129 | 0.0115772 | 0.00102231 | 3633.33 | NA | 12.4047 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.0352696 | 0.127467 | 0.00660492 | 0.00133482 | 415.108 | NA | 39.8129 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.143579 | 0.015 | 0.00582596 | 604.117 | NA | 2.80989 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0689228 | 0.015 | 0.00333182 | 1665.75 | NA | 1.09064 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.120888 | 0.015 | 0.00531924 | 1053.29 | NA | 1.6192 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.125912 | 0.00391245 | 0.00125162 | 49.3081 | NA | 2.21011 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.242408 | 0.00821659 | 0.00131719 | 1292.25 | NA | 1.41652 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.130423 | 0.00471569 | 0.00143165 | 144.035 | NA | 2.03497 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0153291, 0.0717096, 0.721063 | 0.180729 | 0.00305228 | 0.000935085 | 70.4088 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0153291, 0.0717096, 0.721063 | 0.224446 | 0.00226699 | 0.000233596 | 247.157 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0153291, 0.0717096, 0.721063 | 0.181568 | 0.0027391 | 0.00069548 | 86.6489 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
