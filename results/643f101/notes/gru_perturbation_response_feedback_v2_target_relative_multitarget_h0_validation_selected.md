# GRU perturbation-response bank

Issue: `643f101`. Source experiment: `643f101`.

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

### `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.324617 | 0.00344158 | 0.00119976 | 0.0255528 | 0.00742676 | 1.21416 | 0.19477 | 0.372 | NA | 0.000614872 | 0.000166828 | 29.6119 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.106005 | 0.01 | 0.00378736 | 0.0284987 | 0.00941657 | 0.440334 | 0.0636033 | 0.015 | NA | 0.000455653 | 0.00241444 | 155.881 | 8.03786 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.247564 | 0.00677582 | 0.00193775 | 0.0499525 | 0.0125177 | 1.18351 | 0.148538 | 0.1645 | 0.525 | 3.11835e-05 | 0.000423453 | 65.7681 | 4.60247 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00424464 | 0.000270436 | 4.24976e-05 | 0.00247724 | 0.000315021 | 0.0300196 | 0.00254678 | 0.552333 | NA | 6.53183e-06 | 0.000213175 | 1.97792 | 3.20539 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0128953 | 0.000855153 | 0.000114666 | 0.00569765 | 0.000915852 | 0.0545317 | 0.0077372 | 0.59 | NA | 0.00010548 | 0.00156278 | 18.8569 | 12.5945 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.210896 | 0.05 | 0.00972097 | 0.0869605 | 0.0152116 | 1.55316 | 0.126537 | 0.4095 | NA | 0.0383464 | 0.0517437 | 14032.8 | 1.04478 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.111974 | 0.00568168 | 0.000911174 | 0.0499101 | 0.00681525 | 0.857584 | 0.0671845 | 0.526667 | NA | 0.00231183 | 0.017334 | 811.116 | 0.671873 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.183327 | 0.00391759 | 0.00130501 | 0.0283972 | 0.00700005 | 1.00103 | 0.109996 | 0.248 | 0.51 | -0.00094894 | 0.00114443 | 33.688 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.431795 | 0.00291347 | 0.00101054 | 0.0227414 | 0.00634965 | 1.39552 | 0.259077 | 0.356 | NA | 0.00049975 | -0.000808137 | -0.0506612 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.117753 | 0.01 | 0.00363279 | 0.0311041 | 0.00928418 | 0.605854 | 0.070652 | 0.017 | NA | 0.000580653 | 0.00150526 | 121.145 | 6.24674 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.257846 | 0.00640775 | 0.00160485 | 0.0499525 | 0.0122117 | 1.49841 | 0.154707 | 0.1535 | 0.409792 | 3.66488e-05 | 9.41267e-05 | 53.3964 | 3.73669 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00422791 | 0.000254705 | 4.02017e-05 | 0.00242349 | 0.000294389 | 0.0327716 | 0.00253675 | 0.553333 | NA | 7.68863e-06 | 0.000136877 | 1.77201 | 2.87169 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0132159 | 0.000816674 | 0.000109561 | 0.00531248 | 0.000874817 | 0.0521448 | 0.00792953 | 0.59 | NA | 0.000130681 | 0.00135062 | 17.0823 | 11.4092 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.240287 | 0.05 | 0.00962148 | 0.0890156 | 0.0151536 | 2.18823 | 0.144172 | 0.412167 | NA | 0.0384661 | 0.0497434 | 13561.7 | 1.0097 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.111279 | 0.00529924 | 0.000851128 | 0.0499101 | 0.00634727 | 1.11992 | 0.0667673 | 0.521333 | NA | 0.00230135 | 0.0162501 | 731.645 | 0.606045 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.169818 | 0.00324227 | 0.00107834 | 0.0245982 | 0.00569002 | 1.07788 | 0.101891 | 0.238 | 0.54 | -0.000809011 | 0.00113128 | 9.03323 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
