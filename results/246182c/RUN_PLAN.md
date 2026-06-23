# Delayed Post-Movement Cost-Tail Ablation

## Planned Row

| Run | Comparator | Tail mode | Batches | Batch | Hidden | Reps | LR | Schedule | Perturbation | PGD |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `hold__start_pos_zero_vel_lr1e-2_flat_tail` | `ef9c882/hold__start_pos_zero_vel_lr1e-2` | `flat_after_canonical_horizon` | 12000 | 64 | 180 | 5 | 1e-2 | warmup-cosine | calibrated movement-age, small | off |

## Contract

- Target visible from trial start.
- Go cue sampled uniformly from steps 10..30.
- Catch fraction 0.5.
- Force-filter feedback enabled.
- Perturbation training enabled, calibrated, movement-age timed, physical level `small`.
- No PGD or adversarial phase.
- Full analytical Q/R/Qf objective.
- `nn_output_pre_go=0`.
- Pre-go start-position L2 hold `1e6`.
- Pre-go zero-velocity hold `1e5`.
- Gradient clip 5.

## Tail Semantics

Default delayed full-Q/R/Qf rows score the canonical 60 movement-age stages and
place terminal Qf at `go + 59`. The flat-tail diagnostic mode keeps stages
0..59 unchanged, then reuses stage 59 Q/R weights for every remaining trial
transition after the canonical horizon and places terminal Qf at the final
rollout state.

## Launch Gate

This file and `runs/hold__start_pos_zero_vel_lr1e-2_flat_tail.json` are setup
only. A billable remote launch still requires a separate explicit confirmation.
