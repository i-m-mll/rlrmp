# GRU perturbation-response bank

Issue: `aacb9ed`. Source experiment: `aacb9ed`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int].

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 12 |
| `delayed_observation` | 1 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 1 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 1 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 1 |
| `target_stream_jump` | 1 |

## Evaluation

### `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Mean delta pos traj | Mean delta vel traj | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0332141 | 0.000359564 | 0.0035455 | 0.000700329 | 0.01052 | 245.355 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.235156 | 0.000853673 | 0.00627387 | -8.67225e-06 | -0.0006455 | -6.26974 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; delayed_observation rows require a clean pre-noise observation-history adapter in both GRU and analytical paths (1) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.064582 | 0.00777422 | 0.00810687 | 0.00231116 | 0.00134242 | 282.418 | 14.5626 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.265651 | 0.00314457 | 0.0193366 | 0.000387819 | 0.000197373 | 66.5063 | 4.65412 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00463759 | 6.527e-05 | 0.000484404 | 4.76284e-06 | 0.000170674 | 1.78919 | 2.89953 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0130961 | 0.000180394 | 0.00141103 | 7.89269e-05 | 0.00148246 | 16.2574 | 10.8582 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.135328 | 0.0167318 | 0.00885869 | 0.041626 | 0.0240561 | 13316.1 | 0.991412 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.136014 | 0.00130671 | 0.0107067 | 0.00164545 | 0.015169 | 738.631 | 0.611831 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0674678 | 0.000529809 | 0.00369342 | -3.62465e-05 | 0.000109934 | -2.94955 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; sensory_feedback rows require a matching extLQG measurement-channel adapter rather than controller-internal mutation (1) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Mean delta pos traj | Mean delta vel traj | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0383743 | 0.0003317 | 0.00322617 | 0.00067685 | 0.00850132 | 215.747 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.313674 | 0.000919334 | 0.00670415 | -2.66613e-05 | -0.000129441 | -2.03058 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; delayed_observation rows require a clean pre-noise observation-history adapter in both GRU and analytical paths (1) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0795116 | 0.00747355 | 0.00877176 | 0.00214865 | 0.00137515 | 246.304 | 12.7004 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.279125 | 0.00288852 | 0.0193052 | 0.000228035 | 0.000674898 | 57.3513 | 4.01346 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00550341 | 6.08069e-05 | 0.000451658 | 4.46741e-06 | 0.000141259 | 1.57494 | 2.55233 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0139507 | 0.000169216 | 0.00131064 | 7.23885e-05 | 0.00129557 | 13.8568 | 9.25489 | none |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.170602 | 0.0165757 | 0.00960339 | 0.0414063 | 0.0238346 | 13097.9 | 0.975171 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.145811 | 0.00129172 | 0.0100521 | 0.00180894 | 0.0161118 | 681.161 | 0.564228 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0651021 | 0.000536383 | 0.00366794 | -3.50875e-05 | 8.24473e-05 | -3.04366 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; sensory_feedback rows require a matching extLQG measurement-channel adapter rather than controller-internal mutation (1) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_and_process_epsilon - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state and process_epsilon. Command-input, sensory-feedback, and delayed-observation GRU rows are evaluated through temporary external graph adapters, but do not yet have matching analytical extLQG channel adapters; target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
