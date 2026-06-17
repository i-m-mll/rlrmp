# GRU perturbation-response bank

Issue: `4d79e07`. Source experiment: `4d79e07`.

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

### `delayed_movement_bank_pgd_clip5`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.4343 | 0.00872253 | 0.00221363 | 0.0733388 | 0.0194057 | 2.47565 | 0.39087 | 0.451778 | 0.631979 | 0.000257575 | 0.000552274 | 251.346 | 0.48588 | 0.482655 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.418094 | 0.00855437 | 0.00215504 | 0.0710765 | 0.018873 | 2.49279 | 0.376285 | 0.449686 | 0.625328 | 0.000245687 | 0.000519776 | 222.355 | 0.429837 | 0.426984 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.226403 | 0.00249171 | 0.000495302 | 0.0263444 | 0.00521805 | 1.27172 | 0.203763 | 0.512956 | 0.691294 | 0.000146371 | 0.000260275 | 91.2581 | 0.000939806 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.333319 | 0.0200054 | 0.010544 | 0.0963414 | 0.0276001 | 1.94328 | 0.299987 | 0.182367 | 0.633422 | 0.000159865 | 0.000331293 | 183.06 | 0.872779 | 0.732559 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.431017 | 0.0131802 | 0.00484806 | 0.0891822 | 0.0296852 | 2.46503 | 0.387916 | 0.396539 | 0.632312 | 0.000334923 | 0.000446956 | 479.473 | 19.7528 | 15.6042 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.43501 | 0.00872409 | 0.00221557 | 0.0733459 | 0.0194196 | 2.47067 | 0.391509 | 0.451918 | 0.631999 | 0.000257503 | 0.000519011 | 255.404 | 0.493724 | 0.490446 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.410327 | 0.0135382 | 0.00404713 | 0.0638499 | 0.021309 | 1.948 | 0.369294 | 0.616091 | 0.64 | 0.00983815 | 0.0364784 | 3168.48 | 12.3036 | 12.3789 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.405779 | 0.02 | 0.00709176 | 0.109971 | 0.0242385 | 2.16009 | 0.365201 | 0.236151 | 0.648926 | 0.000190302 | 0.000445675 | 462.088 | 0.174909 | 0.231738 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.448971 | 0.0104887 | 0.00282148 | 0.0839902 | 0.0232358 | 2.37657 | 0.404074 | 0.444063 | 0.629932 | 0.000252425 | 0.000459717 | 316.04 | 0.102665 | 0.377226 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.226403 | 0.00249171 | 0.000495302 | 0.0263444 | 0.00521805 | 1.27172 | 0.203763 | 0.512956 | 0.691294 | 0.000146371 | 0.000260275 | 91.2581 | 0.000939806 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.344187 | 0.0100923 | 0.00311388 | 104.562 | 3.84632 | 3.40117 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.631906 | 0.00815507 | 0.00142958 | 581.812 | 0.401981 | 0.402709 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.326807 | 0.00792018 | 0.00209744 | 67.6648 | 0.87472 | 0.779687 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.345468 | 0.010134 | 0.00312845 | 145.547 | 5.35393 | 4.7343 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.598377 | 0.0078811 | 0.0013377 | 441.531 | 0.305059 | 0.305612 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.310438 | 0.00764806 | 0.00199896 | 79.9877 | 1.03402 | 0.921681 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.121832 | 0.0018875 | 0.000454396 | 56.0199 | 0.000278825 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.453638 | 0.00437498 | 0.000769266 | 238.237 | 0.0658668 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.103739 | 0.00121265 | 0.000262244 | -20.4831 | -0.000236041 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.333319 | 0.0200054 | 0.010544 | 183.06 | 0.872779 | 0.732559 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.431017 | 0.0131802 | 0.00484806 | 479.473 | 19.7528 | 15.6042 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.344005 | 0.0100904 | 0.00311568 | 103.499 | 3.80721 | 3.36659 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.634264 | 0.00815288 | 0.00143 | 594.814 | 0.410965 | 0.411709 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.326761 | 0.00792899 | 0.00210104 | 67.8976 | 0.87773 | 0.782369 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.292816 | 0.00826014 | 0.00296483 | 323.885 | 34.0561 | 33.8633 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.617616 | 0.0243503 | 0.00617782 | 8497.36 | 11.5318 | 11.6045 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.320548 | 0.00800415 | 0.00299874 | 684.212 | 26.1097 | 26.249 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.330411 | 0.0200001 | 0.00919435 | 197.061 | 0.176008 | 0.36663 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.558353 | 0.02 | 0.00471699 | 944.933 | 0.235079 | 0.247476 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.328573 | 0.02 | 0.00736395 | 244.269 | 0.0876651 | 0.150204 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.358488 | 0.0111473 | 0.00373146 | 124.1 | 0.427958 | 2.22498 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.657047 | 0.0105889 | 0.00198575 | 712.188 | 0.0920237 | 0.312271 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.331378 | 0.00972993 | 0.00274724 | 111.832 | 0.0927339 | 0.631997 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.121832 | 0.0018875 | 0.000454396 | 56.0199 | 0.000278825 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.453638 | 0.00437498 | 0.000769266 | 238.237 | 0.0658668 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00777878, 0.0155576, 0.0382411, 0.0388939, 0.0764822, 0.191206, 0.369948, 0.739897, 1.84974 | 0.103739 | 0.00121265 | 0.000262244 | -20.4831 | -0.000236041 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
