# GRU perturbation-response bank

Issue: `3992394`. Source experiment: `5f70333`.

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

- Evaluated: 56
- Blocked: 12
- Not implemented: 3
- Rollout trials per replicate: 1

### `lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 56
- Blocked: 12
- Not implemented: 3
- Rollout trials per replicate: 1

## Residuals

- ExtLQG comparator: placeholder - The current materializer defines and evaluates the GRU-side bank. ExtLQG perturbation rollout plumbing is not yet wired to the same declarative bank, so comparator rows are explicit placeholders.
- Full-Q/R/Q_f perturbation cost: not_available - The full analytical Q/R/Q_f loss is available for training and checkpoint selection, but this perturbation materializer does not yet bind that loss object to perturbed post-hoc trial specs.
