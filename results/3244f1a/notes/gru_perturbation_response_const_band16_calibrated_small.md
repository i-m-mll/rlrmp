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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.33792, 0.43121, 0.970616 | 0.084393 | 0.00193377 | 0.000438572 | 0.0183266 | 0.00256592 | 0.553673 | 0.0506358 | 0.385938 | NA | 0.000536746 | 0.00062063 | 77.4847 | NA | 1.48792 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.33792, 0.43121, 0.970616 | 0.0834452 | 0.00192429 | 0.000431978 | 0.0183035 | 0.00258159 | 0.553643 | 0.0500671 | 0.384958 | NA | 0.00050595 | 0.000962739 | 78.1096 | NA | 1.49992 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.0075 | 0.101321 | 0.0075 | 0.00269484 | 0.0253059 | 0.00687301 | 0.30559 | 0.0607923 | 0.033125 | NA | 0.000655047 | 0.000151641 | 62.4617 | NA | 2.49955 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.0128725 | 0.0840161 | 0.00164646 | 0.000463056 | 0.0140305 | 0.0035428 | 0.284979 | 0.0504097 | 0.155688 | 0.28075 | 3.18953e-05 | 3.31046e-05 | 6.00463 | NA | 1.95418 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.0512, 0.0653349, 0.147063 | 0.0841405 | 0.00193274 | 0.000438661 | 0.0183273 | 0.00256313 | 0.55584 | 0.0504843 | 0.386188 | NA | 0.000525295 | 0.000659992 | 77.384 | NA | 1.48599 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.0114932, 0.0176348, 0.065091 | 0.075009 | 0.00387825 | 0.000616818 | 0.0190929 | 0.00410898 | 0.269327 | 0.0450054 | 0.59 | NA | 0.00241868 | 0.00934746 | 354.936 | NA | 13.8669 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (6); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (6) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.0015 | 0.0554632 | 0.0075 | 0.00241304 | 0.0161013 | 0.00418926 | 0.238511 | 0.0332779 | 0.2315 | NA | 0.00310634 | 0.00213898 | 276.847 | NA | 1.3884 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00295888, 0.00364525, 0.00689036 | 0.0811537 | 0.00280847 | 0.000667582 | 0.0224504 | 0.00383247 | 0.478523 | 0.0486922 | 0.375229 | NA | 0.000927864 | 0.00125199 | 122.686 | NA | 1.46438 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.00766454, 0.0358548, 0.360532 | 0.0967339 | 0.00133448 | 0.000310231 | 0.0128976 | 0.00180615 | 0.493722 | 0.0580403 | 0.427514 | NA | 0.000233977 | 0.000139077 | 32.4318 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.33792 | 0.0504011 | 0.00139028 | 0.000467845 | 8.55471 | NA | 2.78265 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 0.970616 | 0.138368 | 0.00291379 | 0.000421378 | 208.237 | NA | 1.44134 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.43121 | 0.0644101 | 0.00149725 | 0.000426494 | 15.6621 | NA | 1.80471 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.33792 | 0.0498619 | 0.00135734 | 0.000451253 | 8.07322 | NA | 2.62603 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 0.970616 | 0.137485 | 0.00291512 | 0.000421052 | 210.13 | NA | 1.45444 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.43121 | 0.0629888 | 0.00150041 | 0.000423627 | 16.1254 | NA | 1.85809 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.0075 | 0.101321 | 0.0075 | 0.00269484 | 62.4617 | NA | 2.49955 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.0128725 | 0.0840161 | 0.00164646 | 0.000463056 | 6.00463 | NA | 1.95418 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.0512 | 0.0504176 | 0.00139058 | 0.000467937 | 8.55736 | NA | 2.78352 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.147063 | 0.137711 | 0.00291027 | 0.00042117 | 207.96 | NA | 1.43942 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.0653349 | 0.0642934 | 0.00149737 | 0.000426877 | 15.6347 | NA | 1.80156 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.0114932 | 0.0516654 | 0.00253795 | 0.000671722 | 51.214 | NA | 53.546 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.065091 | 0.109666 | 0.00579317 | 0.000511243 | 909.724 | NA | 12.4237 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.0176348 | 0.0636955 | 0.00330363 | 0.00066749 | 103.871 | NA | 39.8491 | inflated_ratio; no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; process_epsilon row addresses epsilon_index 6, but selected extLQG comparator exposes 6 process disturbance dimensions (2); process_epsilon row addresses epsilon_index 7, but selected extLQG comparator exposes 6 process disturbance dimensions (2) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.0015 | 0.0717874 | 0.0075 | 0.00291328 | 151.037 | NA | 2.81004 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.0015 | 0.0341578 | 0.00749999 | 0.00166608 | 416.238 | NA | 1.09012 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.0015 | 0.0604445 | 0.00750001 | 0.00265975 | 263.267 | NA | 1.61886 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00295888 | 0.0629427 | 0.00195624 | 0.000625975 | 12.3281 | NA | 2.21029 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.00689036 | 0.115332 | 0.00411127 | 0.00066076 | 319.728 | NA | 1.4019 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00364525 | 0.0651866 | 0.00235789 | 0.000716012 | 36.0018 | NA | 2.03458 | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.00766454, 0.0358548, 0.360532 | 0.0904135 | 0.00152656 | 0.0004677 | 17.6091 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.00766454, 0.0358548, 0.360532 | 0.10907 | 0.00110709 | 0.00011509 | 58.0239 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.00766454, 0.0358548, 0.360532 | 0.0907177 | 0.00136979 | 0.000347903 | 21.6624 | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
