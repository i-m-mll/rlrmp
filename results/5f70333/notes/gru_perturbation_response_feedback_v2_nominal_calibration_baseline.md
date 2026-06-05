# GRU perturbation-response bank

Issue: `5f70333`. Source experiment: `5f70333`.

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

### `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0793133 | 0.00390845 | 0.000584146 | 0.0367136 | 0.00499573 | 0.432561 | 0.047588 | 0.547167 | NA | 0.00118665 | 0.0229414 | 480.998 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.147727 | 0.00133642 | 0.00044959 | 0.0110913 | 0.00340087 | 0.356613 | 0.088636 | 0.388 | NA | 0.000281126 | 0.00141894 | 22.2315 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0157729 | 0.01 | 0.00560738 | 0.0110997 | 0.0060328 | 0.0587071 | 0.00946376 | 0.02 | NA | 0.00568175 | 0.000686437 | 599.33 | 30.9039 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.304069 | 0.00684724 | 0.0022074 | 0.0499525 | 0.0144291 | 0.832315 | 0.182441 | 0.175 | NA | 0.00150394 | 0.00830817 | 363.702 | 25.452 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00523883 | 0.000257881 | 3.85407e-05 | 0.00242308 | 0.000329535 | 0.0285413 | 0.0031433 | 0.547 | NA | 3.94535e-06 | 0.000252965 | 2.09196 | 3.39019 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0137127 | 0.000782414 | 0.000108471 | 0.00553093 | 0.000839803 | 0.057311 | 0.00822763 | 0.59 | NA | 7.46012e-05 | 0.00158524 | 15.49 | 10.3457 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.0405007 | 0.05 | 0.0104538 | 0.0525207 | 0.0109422 | 0.296735 | 0.0243004 | 0.414333 | NA | 0.0452203 | 0.00738654 | 14081.4 | 1.04839 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.139275 | 0.00544381 | 0.000762826 | 0.0499101 | 0.00736497 | 0.861386 | 0.0835651 | 0.521333 | 0.586667 | 0.00169744 | 0.0181693 | 890.371 | 0.737522 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0558352 | 0.00121404 | 0.000338601 | 0.00856043 | 0.00233378 | 0.186055 | 0.0335011 | 0.3 | 0.405 | 8.83432e-05 | 0.000112726 | 15.0329 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
