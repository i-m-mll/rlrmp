# Adaptive curriculum 3500-to-1000 run plan

This issue owns the initial launch of the adaptive soft-adversary curriculum row.
It is linked to the parent method issue [references:08483d5], but its run specs,
bulk outputs, and post-run verdict belong under issue e148f33.

## Plain-language lock

The run continues from the existing clean 6D C&S no-PGD H0 `const_band16`
baseline, not from scratch.  The controller, task, nominal stochasticity,
target-relative support, randomized perturbation bank, and optimizer baseline
stay fixed.  The only new training pressure is a cap-free direct-epsilon
soft-energy adversary whose lambda is adapted from held-out nominal damage
batches.

The adaptive continuation is 7500 controller batches.  Its damage target starts
at 0, ramps linearly to 3500 over the first 2500 continuation batches, then
cosine-anneals to 1000 over the next 5000 batches.  The outer adversarial weight
ramps from 0 to 1 over the same first 2500 continuation batches and then stays
at 1.  This outer weight applies only to the optimized epsilon adversary; the
randomized perturbation-bank training remains active and orthogonal.

## Locked row

| Field | Lock |
| --- | --- |
| Question | Can cap-free direct-epsilon soft-energy training continue from the clean 6D H0 baseline, follow a 3500-to-1000 damage curriculum, and avoid zero-adversary collapse while preserving nominal reach quality? |
| Row label | `adaptive_curriculum_3500to1000` |
| Tracking issue | `e148f33` |
| Parent method issue | `08483d5` |
| Code branch / base | `feature/e148f33-adaptive-curriculum-run` from `main` at `0c3741dd3d63b06f0e72c23228584c5960e02034`, with issue-local launch guards for continuation-relative scheduling and consecutive-zero adversary stopping |
| Baseline spec | `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json` |
| Baseline checkpoint | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest` -> `checkpoint_0012000` |
| Output dir | `_artifacts/e148f33/runs/adaptive_curriculum_3500to1000` |
| Tracked run spec | `results/e148f33/runs/adaptive_curriculum_3500to1000.json` |
| Training length | Resume at 12000 completed baseline batches; stop at 19500 total completed batches, i.e. 7500 adaptive continuation batches |
| Task/model preserved | 6D C&S no-integrator state, target-relative `const_band16`, nominal GRU hidden size 180, 5 replicates, H0/initial-hidden encoder, full analytical Q/R/Qf |
| Randomized perturbation baseline | Calibrated moderate fixed-target perturbation-bank training remains enabled and independent of the adaptive outer adversarial weight |
| Batch/seed | Batch size 64, seed 42 |
| Optimizer | AdamW controller optimizer, lr `3e-3`, 500-batch warmup from `0.1 * lr`, cosine alpha `0.01`, global gradient clip 5 |
| Checkpoints/logging | Checkpoint every 500 completed batches, progress log step 100, training diagnostics sidecar enabled |
| Adaptive adversary | `direct_epsilon` soft-energy objective, cap-free: no projection, no safety cap, no inherited radius, no trust region, no hard budget |
| Damage schedule | Continuation-relative target: 0 -> 3500 over 2500 batches, then cosine 3500 -> 1000 over 5000 batches |
| Outer adversarial weight | Continuation-relative: 0 -> 1 over 2500 batches, then hold at 1 |
| Lambda update | Held-out otherwise-nominal damage batch; update every 50 controller batches; EMA alpha 0.1; eta 0.1; 10% deadband; max log step 0.1 |
| Stop criteria | Stop cleanly on nonfinite/exploding loss, failed checkpoint/resume, or optimized adversary gain over zero staying zero for two consecutive active checkpoints |
| Expected artifacts | `training_summary.json`, `training_diagnostics.{json,npz}`, checkpoint tree, tracked run spec, post-run synced `_artifacts/e148f33/runs/adaptive_curriculum_3500to1000/` |
| Budget/cost exposure | One secure-cloud RTX 5090 pod, one row, expected 7500 resumed controller batches plus bootstrap/smoke overhead; pod billing starts at creation and must be stopped after post-run sync |

## Launch commands

The first remote command is a checkpoint gate that preserves the 500-batch
cadence:

```bash
cd /workspace/rlrmp && uv run --no-sync python scripts/train_cs_nominal_gru.py --run-spec results/e148f33/runs/adaptive_curriculum_3500to1000.json --full-train --resume --stop-after-batches 12500
```

If the checkpoint gate is finite and diagnostics are sane, resume immediately to
the full 19500-batch target:

```bash
cd /workspace/rlrmp && uv run --no-sync python scripts/train_cs_nominal_gru.py --run-spec results/e148f33/runs/adaptive_curriculum_3500to1000.json --full-train --resume
```

## Post-run plan

| Diagnostic / comparison | Why it matters |
| --- | --- |
| Adaptive damage trace: held-out `adaptive_update_damage_raw` vs target | Confirms whether lambda actually tracks the intended curriculum rather than training to the beta spike or collapsing to the final target. |
| Lambda trace, EMA, log-ratio error, update count, log step | Shows whether adaptation is doing work, saturating at the max step, sitting in the deadband, or drifting monotonically. |
| Inner adversary gain over zero and energy diagnostics | Detects zero-adversary collapse and distinguishes weak adversaries from high-lambda energy suppression. |
| Outer adversarial weight and target schedule sidecars | Confirms the schedule is continuation-relative after baseline resume and that the first 2500 adaptive batches are a real ramp. |
| Nominal validation loss and reach-quality summaries | Guards against preserving damage at the cost of losing the original clean reach phenotype. |
| Perturbation-bank validation bins | Ensures randomized perturbation training remains orthogonal and has not been accidentally disabled or reweighted by the adaptive adversarial weight. |
| Comparison to baseline H0 `const_band16` row | Quantifies the incremental effect of adaptive training relative to the exact checkpoint used as the starting point. |
| Comparison to previous `adaptive_pn_b1p05` lock | Separates the new curriculum intuition from the older beta-1.05 paired-noise target. |
| OFB beta anchors | Interpret damage scale using beta 1.05 and beta 1.4 references while avoiding the narrow beta-spike target region. |

## Residual uncertainties before launch

- The live merged code contains the log-ratio lambda update and held-out nominal
  adaptive damage batch.
- The launch branch adds the two remaining guards found during lock review:
  continuation-relative adaptive schedule indexing after baseline resume, and a
  clean stop after two consecutive active checkpoints with zero optimized
  adversary gain.
- The deploy helper does not copy shared `_artifacts` checkpoint bytes, so the
  12000-batch baseline checkpoint must be explicitly staged onto the pod before
  the resumed training command starts.
