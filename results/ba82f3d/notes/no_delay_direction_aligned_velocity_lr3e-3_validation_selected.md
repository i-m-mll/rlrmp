# No-delay direction-aligned velocity metadata

Run: `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64`

Checkpoint policy: `validation_selected_per_replicate`

per-replicate validation-selected numbered checkpoints from sparse training-history validation records

## Outputs

- Pooled figure: `_artifacts/ba82f3d/figures/no_delay_direction_aligned_velocity_lr3e-3_validation_selected/pooled_direction_aligned_velocity_with_matched_extlqg.html`
- Per-direction figure: `_artifacts/ba82f3d/figures/no_delay_direction_aligned_velocity_lr3e-3_validation_selected/per_direction_aligned_velocity_with_matched_extlqg.html`

## Bank

- Bank kind: `uniform_static_targets`
- Directions: 20 uniformly spaced angles
- Reach length: 0.15 m
- Movement horizon: 60 steps
- The bank is constructed in the no-delay materializer; the delayed eval-bank helper is not imported or edited.

## Projection

target-aligned velocity: dot(effector velocity, unit(target - initial_position))

## Summary

- GRU pooled samples: 100 (5 replicates x 20 directions)
- GRU pooled peak mean velocity: 0.720578 m/s at 0.16 s
- extLQG reach length: 0.15 m
- extLQG samples: 2100
- extLQG peak mean velocity: 0.730927 m/s at 0.16 s
- extLQG parity: fixed_point: local port of extLQG/computeOFC/computeExtKalman

## Direction Table

| Direction | Angle (deg) | Unit vector | Peak mean velocity (m/s) | Peak time (s) |
|---:|---:|---|---:|---:|
| 0 | 0 | [1, 0] | 0.731739 | 0.16 |
| 1 | 18 | [0.951057, 0.309017] | 0.715684 | 0.16 |
| 2 | 36 | [0.809017, 0.587785] | 0.7174 | 0.16 |
| 3 | 54 | [0.587785, 0.809017] | 0.73215 | 0.16 |
| 4 | 72 | [0.309017, 0.951057] | 0.72645 | 0.16 |
| 5 | 90 | [6.12323e-17, 1] | 0.714291 | 0.16 |
| 6 | 108 | [-0.309017, 0.951057] | 0.731981 | 0.16 |
| 7 | 126 | [-0.587785, 0.809017] | 0.733942 | 0.16 |
| 8 | 144 | [-0.809017, 0.587785] | 0.706005 | 0.16 |
| 9 | 162 | [-0.951057, 0.309017] | 0.712799 | 0.16 |
| 10 | 180 | [-1, 1.22465e-16] | 0.732614 | 0.16 |
| 11 | 198 | [-0.951057, -0.309017] | 0.712988 | 0.16 |
| 12 | 216 | [-0.809017, -0.587785] | 0.701203 | 0.16 |
| 13 | 234 | [-0.587785, -0.809017] | 0.721901 | 0.16 |
| 14 | 252 | [-0.309017, -0.951057] | 0.717329 | 0.16 |
| 15 | 270 | [-1.83697e-16, -1] | 0.706612 | 0.16 |
| 16 | 288 | [0.309017, -0.951057] | 0.726791 | 0.16 |
| 17 | 306 | [0.587785, -0.809017] | 0.737312 | 0.16 |
| 18 | 324 | [0.809017, -0.587785] | 0.712433 | 0.16 |
| 19 | 342 | [0.951057, -0.309017] | 0.719937 | 0.16 |
