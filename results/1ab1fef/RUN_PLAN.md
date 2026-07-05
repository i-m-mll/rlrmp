# Epsilon-scaled Adaptive Curriculum Short Trial

No training launch is authorized by this lock. The row is ready for user review
against issue 1ab1fef before any smoke or full run.

## Locked Row

| Field | Lock |
| --- | --- |
| Tracking issue | `1ab1fef` |
| Parent method issue | `08483d5` |
| Comparison issue | `91a090c` |
| Row label | `epsilon_scaled_short_3500to1000` |
| Tracked recipe | `results/1ab1fef/runs/epsilon_scaled_short_3500to1000.json` |
| Bulk output dir | `_artifacts/1ab1fef/runs/epsilon_scaled_short_3500to1000/` |
| Baseline spec | `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json` |
| Baseline checkpoint to stage | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest` |
| Task/model | Same clean 6D no-PGD H0 `const_band16` baseline contract as `91a090c`: C&S no-integrator LSS, target-relative static multitarget support, force-filter feedback, hidden size 180, 5 replicates, batch size 64 |
| Controller optimizer | AdamW, lr `3e-3`, 500-batch warmup from `0.1 * lr`, cosine alpha `0.01`, global gradient clip 5 |
| Randomized perturbation training | Preserve the `91a090c` calibrated moderate fixed-target perturbation-bank training; it remains orthogonal to the adaptive epsilon outer schedule |
| Adaptive adversary | `direct_epsilon` soft-energy inner objective, cap-free: no projection, no safety cap, no inherited radius, no trust region, no hard budget |
| Inner optimizer | Adam ascent, 12 steps, lr `2e-5`, zero initialization, full-trial epsilon mask |
| Soft-energy lambda | Explicit `281032999.21861446` |
| Controller-training mode | `epsilon_scaled_outer_training` |
| Mode semantics | Optimize full-strength `delta*` with `task_loss - lambda * epsilon_energy`, stop-gradient it, apply `outer_weight * delta*` in the epsilon channel, and optimize the controller on that scaled-adversary rollout |
| Lambda tracking | Keep full-strength held-out threat damage as the lambda-update signal for comparability; diagnostics also record applied scaled damage/loss |
| Damage target | 0 -> 3500 during ramp, 3500 -> 1000 during anneal, then hold 1000 |
| Schedule | 1,000 adaptive batches ramp, 2,500 adaptive batches anneal, 1,000 adaptive batches hold |
| Global stop | `--n-train-batches 16500`, assuming resume from the 12,000-batch baseline checkpoint |
| Checkpoint cadence | Every 500 batches for the full row |
| Stop criteria | Stop on nonfinite/exploding loss, failed checkpoint/resume, missing smoke diagnostics, or optimized adversary gain over zero staying zero for two consecutive active checkpoints |
| Cloud budget if later approved | Secure-cloud GPU by default; no RunPod/Modal launch is authorized by this artifact |

## Command Shape

Before a full launch, stage or copy the baseline checkpoint tree into the new
bulk output directory so `--resume` starts from completed batch 12,000. Then use
this command shape only after explicit user launch approval:

```bash
PYTHONPATH=src uv run --no-sync python scripts/train_cs_nominal_gru.py \
  --run-spec results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json \
  --output-dir _artifacts/1ab1fef/runs/epsilon_scaled_short_3500to1000 \
  --spec-dir results/1ab1fef/runs/epsilon_scaled_short_3500to1000 \
  --issue 1ab1fef \
  --n-train-batches 16500 \
  --broad-epsilon-pgd-training \
  --broad-epsilon-pgd-mechanism direct_epsilon \
  --broad-epsilon-pgd-objective soft_energy \
  --broad-epsilon-pgd-energy-lambda 281032999.21861446 \
  --broad-epsilon-pgd-inner-optimizer-method adam \
  --broad-epsilon-pgd-adam-lr 2e-5 \
  --broad-epsilon-pgd-steps 12 \
  --adaptive-epsilon-curriculum \
  --adaptive-epsilon-controller-training-mode epsilon_scaled_outer_training \
  --adaptive-epsilon-damage-start 0 \
  --adaptive-epsilon-damage-peak 3500 \
  --adaptive-epsilon-damage-final 1000 \
  --adaptive-epsilon-damage-ramp-batches 1000 \
  --adaptive-epsilon-damage-anneal-batches 2500 \
  --adaptive-epsilon-update-interval-batches 50 \
  --adaptive-epsilon-ema-alpha 0.1 \
  --adaptive-epsilon-eta 0.1 \
  --adaptive-epsilon-deadband-frac 0.1 \
  --adaptive-epsilon-max-log-step 0.1 \
  --adaptive-epsilon-outer-weight-start 0 \
  --adaptive-epsilon-outer-weight-final 1 \
  --adaptive-epsilon-outer-weight-ramp-batches 1000 \
  --checkpoint-interval-batches 500 \
  --log-step 100 \
  --full-train \
  --resume
```

## Smoke Gate

Before any full launch, run a very short resumed smoke row that emits at least
one checkpoint and writes `training_diagnostics.json` plus
`training_diagnostics.npz`. After staging the 12,000-batch baseline checkpoint,
use the full command shape with `--stop-after-batches 12001` so the smoke gates
one adaptive batch beyond the baseline. The smoke passes only if diagnostics
include:

- `adaptive_epsilon_target_damage`
- `adaptive_epsilon_lambda_value`
- `adaptive_epsilon_outer_weight`
- `adaptive_epsilon_epsilon_scale_used`
- `adaptive_epsilon_training_batch_full_strength_damage_raw`
- `adaptive_epsilon_training_batch_applied_scaled_damage_raw`
- `adaptive_epsilon_adaptive_update_full_strength_damage_raw`
- `adaptive_epsilon_adaptive_update_applied_scaled_damage_raw`

## Post-run Plan

Compare `epsilon_scaled_short_3500to1000` to `91a090c/short_3500to1000` and
`91a090c/medium_3500to1000`, plus the 6D analytical extLQG and output-feedback
H-infinity comparators where applicable. Required outputs:

- Nominal velocity profile and peak velocity comparison.
- Damage, lambda, target, and epsilon-scale evolution.
- Full-strength threat damage versus applied scaled training damage.
- Quality stats: terminal position error, time-to-peak, nominal validation loss
  if available, final adversary gain or energy, and zero-adversary/nonfinite
  counts.
- Reuse analytical-only comparator figures when unchanged; they do not need to
  be regenerated solely for this row.
