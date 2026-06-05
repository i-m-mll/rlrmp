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
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0711053 | 0.00383397 | 0.000594361 | 0.0363397 | 0.00455462 | 0.455521 | 0.0426632 | 0.549667 | NA | 0.00110911 | 0.0186346 | 413.002 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.21757 | 0.00151116 | 0.000495093 | 0.0123025 | 0.00344142 | 0.697291 | 0.130542 | 0.426 | NA | -0.000540811 | -0.00114648 | -19.049 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0645419 | 0.01 | 0.00466284 | 0.0190047 | 0.00723417 | 0.184852 | 0.0387251 | 0.006 | NA | 0.00210658 | 0.00115878 | 282.227 | 14.5527 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.266473 | 0.00645462 | 0.00187816 | 0.0499525 | 0.0119341 | 1.0366 | 0.159884 | 0.162 | 0.485 | 0.00037172 | 0.000335187 | 66.3265 | 4.64154 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00461616 | 0.000253264 | 3.93532e-05 | 0.0023984 | 0.000299466 | 0.0299825 | 0.0027697 | 0.549667 | NA | 4.46412e-06 | 0.000168297 | 1.79201 | 2.9041 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0131431 | 0.000802337 | 0.000108486 | 0.00535992 | 0.000860139 | 0.0545678 | 0.00788585 | 0.59 | NA | 7.32565e-05 | 0.00147324 | 16.3302 | 10.9069 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.134639 | 0.05 | 0.0100432 | 0.0647087 | 0.012321 | 0.977778 | 0.0807833 | 0.413167 | NA | 0.0413552 | 0.0242578 | 13320.8 | 0.991764 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.135696 | 0.00525283 | 0.000787583 | 0.0499101 | 0.00655603 | 1.02478 | 0.0814177 | 0.5185 | NA | 0.00155283 | 0.0154914 | 737.891 | 0.611219 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0498921 | 0.00082411 | 0.000264207 | 0.00785893 | 0.00180918 | 0.199976 | 0.0299353 | 0.29 | 0.425 | 5.72258e-05 | -0.000317568 | -17.025 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0843125 | 0.00359769 | 0.000557095 | 0.0358166 | 0.00424094 | 0.562724 | 0.0505875 | 0.543167 | NA | 0.00100994 | 0.0157228 | 363.944 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.30534 | 0.00147717 | 0.000504438 | 0.0128225 | 0.0037429 | 0.846595 | 0.183204 | 0.422 | NA | -0.000628186 | 0.00127211 | -11.4068 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0803549 | 0.01 | 0.00447677 | 0.0209587 | 0.00747652 | 0.234687 | 0.0482129 | 0.0195 | NA | 0.00183953 | 0.00205547 | 244.423 | 12.6034 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.27812 | 0.00627965 | 0.00172242 | 0.0499525 | 0.011877 | 1.11338 | 0.166872 | 0.158 | 0.412222 | 0.000223618 | 0.000884811 | 58.7112 | 4.10862 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00567084 | 0.000236415 | 3.63946e-05 | 0.0023638 | 0.00028048 | 0.0378153 | 0.00340251 | 0.542333 | NA | 3.69672e-06 | 0.000287558 | 1.5837 | 2.56652 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.014101 | 0.000739885 | 0.000101285 | 0.00500704 | 0.000793863 | 0.0654988 | 0.00846063 | 0.59 | NA | 5.96794e-05 | 0.00215115 | 13.7152 | 9.16033 | none |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.170774 | 0.05 | 0.0099511 | 0.0692542 | 0.012532 | 1.25557 | 0.102464 | 0.411167 | NA | 0.0410601 | 0.0248399 | 13105.1 | 0.975709 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.147192 | 0.00501253 | 0.000773225 | 0.0499101 | 0.00621027 | 1.21571 | 0.0883152 | 0.5145 | NA | 0.0015241 | 0.0172317 | 683.968 | 0.566553 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0535206 | 0.000822698 | 0.000264622 | 0.00789663 | 0.00184768 | 0.193023 | 0.0321124 | 0.296 | 0.31 | -3.77084e-05 | 2.48405e-05 | -16.2641 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
