# Adaptive curriculum ramp-duration comparison

This issue owns the next adaptive soft-adversary experiment under
[references:08483d5]. It is a two-wave experiment issue. Wave 1 is locked here;
wave 2 will stay on the same issue but will not launch until the wave-1 result
is reviewed.

## Plain-language lock

Wave 1 asks whether the length of the ramp-up and anneal changes the quality of
the adaptive soft-adversary result. Both rows continue from the same clean 6D
no-PGD H0 const_band16 baseline checkpoint. The controller, task, nominal
stochasticity, target support, randomized perturbation bank, optimizer, and
adaptive lambda rule stay fixed. The only planned difference is schedule
duration.

Before launching the two rows, the run manager must acquire a secure RTX 5090
pod, stage the baseline checkpoint, then run a very short resumed smoke row that
emits at least one checkpoint and writes `training_diagnostics.*`. The smoke
passes only if the diagnostics sidecar contains adaptive damage and lambda
series, including `adaptive_epsilon_adaptive_update_damage_raw`,
`adaptive_epsilon_target_damage`, and `adaptive_epsilon_lambda_value`.

## Locked rows

| Field | Lock |
| --- | --- |
| Tracking issue | `91a090c` |
| Parent method issue | `08483d5` |
| Code branch | `feature/91a090c-ramp-duration-wave`, based on `feature/948e66f-diagnostics-stitch` so checkpoint-cadence diagnostics are present |
| Feedbax deploy branch | Use `feature/8268865-runpod-deploy-safety` for RunPod deploy helpers until that fix is integrated |
| Baseline spec | `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json` |
| Baseline checkpoint | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest` |
| Shared task/model | 6D C&S no-integrator state, target-relative `const_band16`, nominal GRU hidden size 180, 5 replicates, H0/initial-hidden encoder, full analytical Q/R/Qf |
| Randomized perturbation baseline | Calibrated moderate fixed-target perturbation-bank training remains enabled and independent of the adaptive outer adversarial weight |
| Batch/seed | Batch size 64, seed 42 |
| Controller optimizer | AdamW, lr `3e-3`, 500-batch warmup from `0.1 * lr`, cosine alpha `0.01`, global gradient clip 5 |
| Adaptive adversary | `direct_epsilon` soft-energy objective, cap-free: no projection, no safety cap, no inherited radius, no trust region, no hard budget |
| Lambda update | Held-out otherwise-nominal damage batch; update every 50 controller batches; EMA alpha 0.1; eta 0.1; 10% deadband; max log step 0.1 |
| Smoke row | `smoke_diagnostics_gate`, 2 adaptive batches, checkpoint interval 1, diagnostics sidecar required before launch |
| Short row | `short_3500to1000`, 3500 adaptive batches, total `n_train_batches=15500`, damage target 0 -> 3500 over 1000 batches then 3500 -> 1000 over 2500 batches |
| Medium row | `medium_3500to1000`, 5250 adaptive batches, total `n_train_batches=17250`, damage target 0 -> 3500 over 1500 batches then 3500 -> 1000 over 3750 batches |
| Parallelism | Run the short and medium rows in parallel after the smoke gate passes |
| Stop criteria | Stop on nonfinite/exploding loss, failed checkpoint/resume, missing smoke diagnostics, or optimized adversary gain over zero staying zero for two consecutive active checkpoints |
| Budget/cost exposure | One secure-cloud RTX 5090 pod retried until acquired; billing starts at pod creation; stop the pod after artifacts are synced |

## Run specs

- `results/91a090c/runs/smoke_diagnostics_gate.json`
- `results/91a090c/runs/short_3500to1000.json`
- `results/91a090c/runs/medium_3500to1000.json`

## Post-run plan

| Output | Reason |
| --- | --- |
| Nominal velocity profile figure with both rows, analytical 6D extLQG, and output-feedback H-infinity on one subplot | Directly compares whether ramp duration moves the nominal reach phenotype toward or away from the analytical references |
| Peak velocity table | Gives the most compact first-pass quality comparison against the prior baseline and analytical rows |
| Damage/lambda figure with damage on one subplot and lambda on a separate subplot, one curve per row | Shows whether the two schedules actually differ in adaptive-damage tracking and lambda response |
| Additional quality stats | Include at least terminal position error, time-to-peak, nominal validation loss if available, final adversary gain/energy, and any zero-adversary/nonfinite counts |
| Post-run issue summary | Record whether wave 1 answers, brackets, or motivates wave 2 damage-bound rows |

