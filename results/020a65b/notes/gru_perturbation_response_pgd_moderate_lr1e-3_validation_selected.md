# GRU perturbation-response bank

Issue: `020a65b`. Source experiment: `020a65b`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 12 |
| `delayed_observation` | 3 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 3 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 3 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 3 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.140535 | 0.00369476 | 0.00094045 | 0.0323846 | 0.00543306 | 0.82764 | 0.0843213 | 0.393932 | NA | 0.00101649 | 0.0016118 | 155.662 | 2.13196 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.0706566 | 0.0010426 | 0.000225198 | 0.00991142 | 0.00141551 | 0.35585 | 0.0423939 | 0.439208 | NA | -0.000447695 | 0.000893976 | 43.5408 | 2.04959 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0871761 | 0.01 | 0.00400924 | 0.0253678 | 0.00914788 | 0.266441 | 0.0523057 | 0.0147109 | NA | 0.00129168 | 0.00227288 | 199.921 | 5.36157 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.178116 | 0.00787631 | 0.00297446 | 0.0499525 | 0.0134508 | 0.600994 | 0.10687 | 0.210117 | NA | 0.00126456 | 0.00245679 | 219.516 | 5.99402 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00927492 | 0.000243828 | 6.20443e-05 | 0.0021373 | 0.00035836 | 0.0547011 | 0.00556495 | 0.394227 | NA | 5.09014e-06 | 6.97015e-06 | 0.677058 | 2.1288 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0316768 | 0.00174615 | 0.000361761 | 0.00677754 | 0.00185138 | 0.0825083 | 0.0190061 | 0.59 | NA | 0.0006776 | 0.000809989 | 35.2941 | 32.0681 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.372091 | 0.05 | 0.0158058 | 0.121224 | 0.0316034 | 1.79071 | 0.223255 | 0.228688 | NA | 0.0252349 | 0.0646795 | 13573 | 1.15597 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.162878 | 0.00687988 | 0.00185319 | 0.0499101 | 0.010002 | 0.884718 | 0.0977269 | 0.391727 | NA | 0.00283224 | 0.00441988 | 458.457 | 0.479414 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0707019 | 0.00104243 | 0.000225128 | 0.00991421 | 0.00141529 | 0.356134 | 0.0424211 | 0.439083 | NA | -0.000436381 | 0.000898017 | 43.6816 | 2.05621 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.141218 | 0.00428529 | 0.00141118 | 109.245 | 4.58881 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.126923 | 0.00335111 | 0.000469497 | 262.313 | 1.70741 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.153465 | 0.00344788 | 0.000940674 | 95.428 | 2.29382 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0538488 | 0.00100104 | 0.000306489 | 59.3985 | 1.35004 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0790611 | 0.00101888 | 9.51142e-05 | -3.28206 | -4.20063 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0790598 | 0.00110787 | 0.000273992 | 74.506 | 3.93123 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0871761 | 0.01 | 0.00400924 | 199.921 | 5.36157 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.178116 | 0.00787631 | 0.00297446 | 219.516 | 5.99402 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00931752 | 0.000282805 | 9.31177e-05 | 0.475511 | 4.58534 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00835978 | 0.000221298 | 3.0998e-05 | 1.14146 | 1.70565 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0101475 | 0.00022738 | 6.20174e-05 | 0.414206 | 2.28567 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0450691 | 0.00230049 | 0.000612011 | 41.8875 | 58.1797 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0145486 | 0.000955281 | 8.15136e-05 | 25.6265 | 14.7349 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0354127 | 0.00198268 | 0.000391758 | 38.3683 | 45.5329 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.409906 | 0.05 | 0.0193676 | 8057.45 | 1.61925 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.290397 | 0.05 | 0.0110187 | 19887.6 | 1.11321 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.415969 | 0.05 | 0.0170311 | 12774 | 1.0315 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.174448 | 0.00741552 | 0.0025551 | 272.34 | 0.822231 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.14045 | 0.00655723 | 0.001022 | 773.047 | 0.474236 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.173737 | 0.00666689 | 0.00198247 | 329.985 | 0.363599 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0538447 | 0.000999429 | 0.000305958 | 59.0947 | 1.34314 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0791037 | 0.00101854 | 9.51107e-05 | -2.03621 | -2.6061 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0791572 | 0.00110932 | 0.000274316 | 73.9862 | 3.9038 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
