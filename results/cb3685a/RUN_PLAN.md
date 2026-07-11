# Harmonized Nominal Baseline Run Plan

Issue: `cb3685a`
Umbrella: `8a79381`
Status: spec locked; **not authorized to launch**.

## Purpose

This is the Stage-1 fork source for the Stage-2 adaptive-epsilon learning-rate
comparison. It trains on the same no-integrator game, target support, and
moderate open-loop-calibrated perturbation bank that the later continuation
rows will use, so a checkpoint fork does not introduce a task seam.

## Locked baseline row

| Row | Batches | Batch / seed | Optimizer and LR | Checkpoints | Artifact route |
|---|---:|---|---|---|---|
| `harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64` | 12,000 | batch 64; seed 42; one `vmap` ensemble with 5 independently initialized replicas | AdamW, weight decay 0, global-norm clip 5; linear warmup `3e-4 -> 3e-3` for batches 1-500, then cosine decay to `3e-5` at batch 12,000 | every 500 batches | `_artifacts/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64/` |

The five replicas are vectorized members of the one baseline row. They are not
five separately launched or compared rows.

## Task and model contract

- C&S no-integrator game: 6 physical states, 36 delay-augmented states, 61
  sampled time points / 60 control-cost stages, 10 ms step, and a 15 cm reach.
  The loss is the full analytical `Q/R/Q_f` objective: position weight `1e6`,
  velocity weight `1e5`, `R = I_2`, and no hidden-state regularizer.
- One 180-unit GRU ensemble, target-relative delayed feedback plus delayed
  force/filter input (six controller inputs), five-step feedback delay, and
  an affine zero-initialized trainable H0 encoder. There is no go cue and no
  delayed-reach phase.
- `const_band16` static target support: 56 seen directions and 16 held-out
  directions, all at 15 cm. It is not a fixed `[0.15, 0]`-only training task.
- The ordinary C&S rollout-noise preset remains active. This baseline has no
  broad-epsilon, PGD, policy, or adaptive-epsilon adversary.

## Exact outer task identity for the Stage-2 gate

The current writer emits the baseline task identity in the outer RLRMP recipe,
without a hand-authored duplicate: `game_card` is the complete no-integrator
game payload and `training_distribution.perturbation_training` is the complete
calibrated perturbation-bank payload. The lower-level
`hps.perturbation_training` is also serialized for execution, but the
prelaunch matrix must snapshot the two outer payloads into
`matrix.metadata.rlrmp_task_identity = {game_card, perturbation_training}` and
compare them byte-for-byte after canonical JSON normalization. The current
writer does not emit a top-level `rlrmp_task_identity` field on an individual
run recipe, so this matrix metadata is the governed gate-owned snapshot rather
than a second baseline-spec source of truth.

## Perturbation-training bank

The locked bank is the moderate, open-loop-calibrated bank governed by
`results/ea6ccb4/data_products/perturbation_open_loop_calibration.json`,
identity `03edd3141b62d1b1cf045097114caac7bc96f1236a433875976aec974d9bb97a`.
It samples 45% nominal trials, 45% single-family trials, and 10% mild-combined
trials. The active families are initial-position, initial-velocity,
process-epsilon, command-input, and sensory-feedback. The mild-combined trial
uses initial-position plus command-input at the configured half-amplitude
scale. `delayed_observation` is explicitly inactive.

The physical effect level is moderate (10% of reach scale), with calibrated
timing and open-loop amplitudes. Training uses the bank as a randomized task
distribution; it is distinct from a training-time worst-case adversary.

## Non-billable seam acceptance check

After the batch-12,000 custody checkpoint exists, make an isolated 200-batch,
adversary-free seam probe. It must keep the task and perturbation-bank payloads
byte-identical to the baseline, continue at the terminal LR `3e-5`, and write
only to an isolated probe directory. The probe is an acceptance check, not a
Stage-2 row and never a Stage-2 fork source. Compare the first probe batches
with the end of the baseline for clean loss and pre-clip gradient norm; an
order-of-magnitude discontinuity fails the seam check.

## Reserved Stage-2 value and configuration

No numeric Stage-2 setpoint is locked until this baseline completes. The
shared setpoint will be derived as:

```text
R* = round_to_2_significant_figures(
       1024 / mean_clean_loss_over_baseline_final_quarter
     )
```

- Numerator convention: `excess`.
- Denominator window: `baseline_final_quarter`, batches 9001-12000 aggregated
  across the single vectorized ensemble row.
- The prelaunch gate must print the raw numerator, denominator-window label,
  raw quotient, and rounded `R*`; the Stage-2 lock table must display those
  components rather than only the rounded value.

The deferred Stage-2 clamp settings are `eta=0.2`, `max_log_step=0.15`, and
`deadband_frac=0.03`; the EMA alpha and update interval remain unchanged, and
`lambda_min` is approximately `1e-3` times the analytical seed. The
application-ramp freeze switch is explicitly **off** for these rows: the
full-strength damage measurement is valid from batch 0, so the clamp and
damage EMA both operate from batch 0; the ramp is diagnostic-only.

## Operational boundary

The selected operational environment is a secure RTX 5090. Before pod
creation, verify an available CUDA-12.8.1-or-newer image tag; this document
and `runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json` authorize
specification only. The billable launch remains blocked on a separate explicit
user confirmation. The nominal-GRU `--run-spec` path has passed its
non-billable dry run against this compact recipe, including RLRMP-method
registration and sibling graph-sidecar resolution.
