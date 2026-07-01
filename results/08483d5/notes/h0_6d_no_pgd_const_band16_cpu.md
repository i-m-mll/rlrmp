# 6D No-PGD H0 const_band16 CPU Baseline

This local run materializes a 6D no-integrator no-PGD H0 `const_band16`
baseline for issue `08483d5`, using the live 33b0dcb `const_band16` row as the
source contract and changing only the issue/output path plus the 6D
no-integrator state and the issue-08483d5 moderate perturbation-training level.

## Source

- Source run spec: `results/33b0dcb/runs/h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64.json`
- Source row: no-PGD H0 target-relative `const_band16`, full analytical Q/R/Qf,
  C&S nominal GRU, hidden size 180, 5 replicates, seed 42, batch size 64,
  controller lr 3e-3, 500-batch warmup from 0.1 fraction, cosine alpha 0.01,
  gradient clip 5, checkpoints every 500 batches.
- Source run difference preserved as provenance only: the live 33b0dcb row used
  the 8D physical / 48D delayed C&S state and `perturbation_physical_level=small`.

## Changed Knobs

- `--issue 08483d5`
- `--output-dir _artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu`
- `--no-integrator-state`, producing a 6D physical state and 36D delayed state.
- `--perturbation-physical-level moderate`, matching the persistent 08483d5
  baseline contract instead of copying the older 33b0dcb small-level setting.

No broad-epsilon, PGD, policy-adversary, cap, radius, or trust-region training
flags were used.

## Command

```bash
env JAX_PLATFORM_NAME=cpu PYTHONPATH=src uv run --no-sync python scripts/train_cs_nominal_gru.py \
  --issue 08483d5 \
  --n-train-batches 12000 \
  --batch-size 64 \
  --controller-lr 0.003 \
  --gradient-clip-norm 5 \
  --lr-warmup-batches 500 \
  --lr-warmup-init-fraction 0.1 \
  --lr-cosine-alpha 0.01 \
  --n-replicates 5 \
  --hidden-size 180 \
  --loss-objective full_analytical_qrf \
  --target-relative-multitarget \
  --target-support-profile const_band16 \
  --initial-hidden-encoder \
  --force-filter-feedback \
  --perturbation-training \
  --perturbation-calibrated-timing \
  --perturbation-physical-level moderate \
  --no-integrator-state \
  --output-dir _artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu \
  --full-train \
  --resume \
  --checkpoint-interval-batches 500 \
  --log-step 100
```

## Outputs

- Tracked run spec: `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json`
- Tracked graph manifest:
  `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/model.graph.manifest.json`
- Bulk output directory: `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/`
- Final model: `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/trained_model.eqx`
- Latest checkpoint:
  `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest`
  -> `checkpoint_0012000`
- Training log: `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/train.log`
- Training diagnostics:
  `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/training_diagnostics.json`
  and `.npz`

## Result

- RLRMP commit: `899278bb30006153c28a0c92aee9dffedc6c4633`
- Completed batches: 12000 / 12000
- Training duration from summary: 753.8859 s
- Command wall time from `/usr/bin/time`: 761.60 s
- Final mean train loss: 4518.3449
- Final mean validation loss: 4394.9014
- Best nonzero mean validation loss: 4394.9014 at batch 12000
- Final checkpoint metadata confirms: `adversarial_phase=none`,
  `no_integrator_state=true`, `physical_state_dim=6`, `state_dim=36`,
  `target_support_profile=const_band16`, `perturbation_physical_level=moderate`.

The run completed successfully. A runnable checkpoint and final model now exist
under the bulk output directory.
