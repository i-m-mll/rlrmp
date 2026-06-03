# GRU perturbation-response bank

Issue: `3992394`. Source experiment: `5f70333`.

The bank is controller-independent: it perturbs external task, plant, sensory, observation, or target interfaces and does not mutate GRU internals.

## Bank

| Channel | Count |
|---|---:|
| `delayed_observation` | 1 |
| `initial_state` | 8 |
| `plant_force` | 12 |
| `sensory_feedback` | 1 |
| `target_stream` | 1 |

## Evaluation

### `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64`

- Evaluated: 8
- Blocked: 12
- Not implemented: 3
- Rollout trials per replicate: 1

### `lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64`

- Evaluated: 8
- Blocked: 12
- Not implemented: 3
- Rollout trials per replicate: 1

## Residuals

- ExtLQG comparator: placeholder - The current materializer defines and evaluates the GRU-side bank. ExtLQG perturbation rollout plumbing is not yet wired to the same declarative bank, so comparator rows are explicit placeholders.
- Full-Q/R/Q_f perturbation cost: not_available - The full analytical Q/R/Q_f loss is available for training and checkpoint selection, but this perturbation materializer does not yet bind that loss object to perturbed post-hoc trial specs.
