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
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0588371 | 0.00396718 | 0.000632107 | 0.03638 | 0.00448891 | 0.421011 | 0.0353023 | 0.561667 | NA | 0.00179473 | 0.0163861 | 411.512 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.502635 | 0.00234845 | 0.000839707 | 0.0186729 | 0.00560111 | 1.53628 | 0.301581 | 0.352 | NA | 0.00101311 | -0.00209211 | -8.61846 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.147464 | 0.01 | 0.00322738 | 0.0371948 | 0.00963142 | 0.696554 | 0.0884785 | 0.0135 | NA | 0.000270465 | 0.000416774 | 74.2919 | 3.83078 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.256825 | 0.00620601 | 0.00146012 | 0.0499525 | 0.0120465 | 1.26477 | 0.154095 | 0.1495 | 0.4055 | 5.35644e-06 | 1.454e-05 | 45.0222 | 3.15066 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00388277 | 0.00026188 | 4.17274e-05 | 0.00240108 | 0.000296277 | 0.0277845 | 0.00232966 | 0.561667 | NA | 7.51772e-06 | 0.000117882 | 1.79278 | 2.90536 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0123946 | 0.000851654 | 0.000112383 | 0.00540842 | 0.000912481 | 0.0453328 | 0.00743678 | 0.59 | NA | 0.000122201 | 0.00138205 | 19.0135 | 12.6991 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.263758 | 0.05 | 0.00949819 | 0.0898758 | 0.0154123 | 2.57573 | 0.158255 | 0.412667 | NA | 0.0378164 | 0.046658 | 13320.5 | 0.991746 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.088213 | 0.00535553 | 0.000880797 | 0.0499101 | 0.00627084 | 0.991186 | 0.0529278 | 0.527 | NA | 0.00240337 | 0.0157293 | 728.02 | 0.603042 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.109411 | 0.00174383 | 0.000563223 | 0.0144964 | 0.00343176 | 0.852396 | 0.0656467 | 0.206 | 0.425 | -0.000220108 | 0.000131407 | -29.4884 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 70
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0657681 | 0.00361654 | 0.000576431 | 0.0354782 | 0.00413174 | 0.509518 | 0.0394609 | 0.5525 | NA | 0.0015374 | 0.0144536 | 356.222 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 1 | evaluated=1 | 0.01 | 0.791254 | 0.0020399 | 0.000705479 | 0.0176518 | 0.00505314 | 1.95572 | 0.474752 | 0.348 | NA | 0.000928036 | -0.00353343 | -35.9519 | NA | denominator_guarded |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.15777 | 0.01 | 0.00307383 | 0.038885 | 0.00967281 | 0.8026 | 0.0946617 | 0.018 | NA | 0.000270857 | 0.000143889 | 60.7013 | 3.13 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.293164 | 0.00572863 | 0.00121964 | 0.0499525 | 0.0115245 | 1.47277 | 0.175899 | 0.1395 | 0.328 | -9.55254e-06 | 2.39759e-05 | 39.8349 | 2.78766 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00434357 | 0.000238746 | 3.80631e-05 | 0.00234156 | 0.000272657 | 0.0336276 | 0.00260614 | 0.553167 | NA | 5.94359e-06 | 0.000103629 | 1.55195 | 2.51507 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0133414 | 0.000784529 | 0.000104717 | 0.00500185 | 0.000840859 | 0.0527193 | 0.00800485 | 0.59 | NA | 0.000100353 | 0.00113767 | 15.7432 | 10.5149 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.303699 | 0.05 | 0.00938633 | 0.0917412 | 0.0152036 | 2.82287 | 0.182219 | 0.412667 | NA | 0.0379006 | 0.038983 | 12813.2 | 0.953974 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.116849 | 0.00496078 | 0.000782075 | 0.0499101 | 0.00595686 | 1.23361 | 0.0701095 | 0.517333 | NA | 0.00204829 | 0.0137482 | 649.547 | 0.538041 | none |
| `sensory_feedback/sensory_feedback_offset` | 1 | evaluated=1 | 0.01 | 0.0992006 | 0.00110784 | 0.000354595 | 0.010471 | 0.00235775 | 1.08299 | 0.0595204 | 0.24 | 0.305 | -2.18838e-05 | -0.000436659 | -26.7393 | NA | denominator_guarded |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
