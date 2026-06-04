# GRU perturbation-response bank

Issue: `ba82f3d`. Source experiment: `ba82f3d`.

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

### `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Mean delta pos traj | Mean delta vel traj | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0525397 | 0.00100209 | 0.00647369 | 0.00135315 | 0.0151119 | 353.388 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.485526 | 0.00142177 | 0.00904391 | -6.80835e-05 | 0.000330152 | 0.747181 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.146815 | 0.00538966 | 0.013914 | 0.000241112 | 0.000525355 | 74.8746 | 3.86083 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.255969 | 0.00245003 | 0.0197177 | 7.27538e-06 | 1.60362e-05 | 45.3276 | 3.17203 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00387498 | 6.94579e-05 | 0.000472598 | 6.08484e-06 | 0.000163883 | 1.79039 | 2.90147 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0123801 | 0.000187156 | 0.00150129 | 9.73337e-05 | 0.00154234 | 18.9726 | 12.6718 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.26351 | 0.01583 | 0.0162854 | 0.0372957 | 0.0469151 | 13329.1 | 0.992383 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.0882751 | 0.0014667 | 0.0100922 | 0.00207289 | 0.0158772 | 727.751 | 0.602819 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0378429 | 0.00063005 | 0.00326839 | -7.64885e-05 | 9.0034e-05 | -6.85216 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 8
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Mean delta pos traj | Mean delta vel traj | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0588599 | 0.000909762 | 0.00593613 | 0.00113017 | 0.0140045 | 303.566 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.772026 | 0.00128424 | 0.0089302 | -4.84057e-05 | -0.000145566 | -2.32606 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.156501 | 0.00514292 | 0.014074 | 0.000259632 | 0.000208203 | 61.384 | 3.16521 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.290671 | 0.0020371 | 0.0188613 | -7.67797e-06 | 5.4584e-05 | 39.5895 | 2.77048 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00433611 | 6.33073e-05 | 0.000434515 | 5.39394e-06 | 0.000139585 | 1.54994 | 2.51182 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0133285 | 0.000174323 | 0.00138199 | 8.87988e-05 | 0.00139065 | 15.6888 | 10.4785 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.302851 | 0.0156436 | 0.0161949 | 0.0374558 | 0.0396347 | 12821.9 | 0.954622 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.11637 | 0.00130073 | 0.00964378 | 0.00179531 | 0.0141685 | 650.011 | 0.538425 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.04829 | 0.000537032 | 0.00315304 | -5.42256e-05 | 0.000145561 | -4.82485 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
