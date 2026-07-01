# Adaptive Soft-Adversary No-Launch Run Plan

## Status

This is a run lock, not launch approval. No local training, remote training,
pod acquisition, push, protected auth request, merge, issue closure, or comment
on `2e60620` is authorized by this artifact.

The planned row is not currently runnable with existing training flags. Current
training code supports broad-epsilon direct-epsilon PGD and a static
soft-energy lambda, but the live training path still requires a safety-cap
radius for `soft_energy`, projects every inner proposal onto a per-trial L2
ball, and has no stateful adaptive-lambda update at a 50-batch cadence.

## Objective

Test whether a cap-free direct-epsilon soft adversary can stay active during
controller training while matching the conservative beta 1.05 output-feedback
damage scale from the latest frozen replay. The scientific question is whether
adaptive lambda can keep adversarial pressure finite and behaviorally meaningful
without inherited cap, radius, projection, or trust-region values defining the
effect.

## Row

| Row | Role | Status | Controller-driving target | Diagnostic target |
| --- | --- | --- | --- | --- |
| `adaptive_pn_b1p05` | primary | `implementation_required_no_launch` | beta 1.05 paired-noise output-feedback damage, `1911.8930971469426` | beta 1.05 deterministic output-feedback damage, `447.8904023668665` |

No comparator row is locked here. The deterministic beta 1.05 value is retained
as a secondary diagnostic target because the conservative replay showed the
paired-noise target was already inside the 10% deadband at the initial lambda.

## Baseline Task And Controller Contract

- Baseline source: `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json`.
- Initial checkpoint: `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest/model.eqx`.
- Task support: target-relative `const_band16`, fixed 15 cm reach, 72 normal
  validation targets, directions may vary.
- Controller: GRU, 5 replicates, hidden size 180, 6D no-integrator physical
  state, 36D delay-augmented C&S LSS state, force-filter feedback included.
- Loss: full analytical Q/R/Qf (`full_analytical_qrf`).
- Training scale for the planned row: batch size 64, seed 42, controller
  gradient clip 5, 12000 controller batches unless a later launch spec changes
  this explicitly.
- Perturbation baseline: keep the existing 08483d5 moderate randomized
  perturbation-training contract from the baseline row.

## Adversary Mechanism

- Mechanism: direct-epsilon sequence over shape `batch x 60 x 6`.
- Inner objective: `mean(task_cost) - lambda * mean(sum_t,d epsilon[t,d]^2)`.
- Inner optimizer controls from the conservative replay: cap-free Adam ascent,
  12 steps, learning rate `2e-5`, Adam `b1=0.9`, `b2=0.999`, `eps=1e-8`,
  gradient L2 clipping at `1e6`, zero epsilon initialization, select the best
  finite soft objective encountered.
- Guardrail: no projection, safety cap, inherited radius, trust region, or hard
  budget is part of the planned scientific run. Epsilon norm and energy may be
  logged as diagnostics only.

## Lambda Initialization And Target

- Lambda source: `results/06a4dc8/canonical_soft_lambda_hvp.json`.
- Source value: `lambda_star_p90 = 254905214.71076143`.
- Initial lambda: beta 1.05 candidate `lambda0 = 281032999.21861446`.
- Primary damage target source:
  `results/08483d5/notes/output_feedback_damage_estimate_beta1p05.json`.
- Primary damage target: beta 1.05 paired nominal-noise output-feedback damage
  `1911.8930971469426`.
- Secondary diagnostic target: beta 1.05 deterministic output-feedback damage
  `447.8904023668665`.

## Adaptive Rule

- Update cadence: once every 50 controller batches.
- Evaluation batch: batch size 64 drawn from the normal fixed-15-cm validation
  trials. Do not use repeated singleton +x trials; directions may vary unless
  later implementation deliberately selects a named target subset.
- Replicate handling: aggregate paired damage over all 5 checkpoint replicates
  and all evaluation trials.
- Damage estimate: mean adversarial total cost minus mean clean total cost,
  using paired clean/adversarial rollout keys.
- EMA: initialize from the first aggregate damage; alpha `0.1`.
- Lambda update outside deadband:
  `log(lambda_next) = log(lambda) + clip(0.1 * log(EMA_damage / target), -0.1, 0.1)`.
- Deadband: if EMA damage is within +/-10% of target, lambda is unchanged.
- Lambda state must persist across checkpoints and resumes.

## Stopping, Checkpoints, And Logging

- This artifact stops at no-launch planning.
- A future approved run should stop normally at 12000 controller batches.
- Abort conditions for the future implementation: nonfinite controller loss,
  nonfinite adversary objective, nonfinite lambda, failed checkpoint write, or
  repeated zero/nonzero adversary status inconsistent with the locked rule.
- Checkpoints: keep the repo baseline cadence of every 500 batches unless a
  later spec changes it. Include adaptive-lambda state in checkpoints.
- Logging: keep batch progress lines, add one adaptive-lambda record every 50
  batches, and include the evaluation batch provenance and target source in the
  tracked run spec.

## Post-Run Diagnostics

Post-run analysis should report:

- Lambda trajectory, EMA damage trajectory, raw damage trajectory, and deadband
  decisions.
- Clean and adversarial task cost decompositions on the same evaluation batch.
- Epsilon energy, epsilon norm, max absolute epsilon, and per-replicate damage
  as diagnostic sidecars only.
- Nominal movement quality: validation loss, endpoint quality, peak velocity,
  time-to-peak, forward-velocity RMSE, and pre-go drift where applicable.
- Replay against beta 1.05 paired-noise and deterministic targets, plus beta
  1.4 paired-noise/deterministic targets as context only.
- Confirmation that no cap, radius, projection, or trust-region guard was active.

## Expected Artifacts

- Lock artifact: `results/08483d5/runs/adaptive_pn_b1p05_lock.json`.
- Future tracked run recipe after implementation and explicit launch approval:
  `results/08483d5/runs/adaptive_pn_b1p05.json`.
- Future bulk outputs:
  `_artifacts/08483d5/runs/adaptive_pn_b1p05/`.
- Future notes:
  `results/08483d5/notes/adaptive_pn_b1p05.md`.

## Implementation Prerequisites

1. Add a cap-free direct-epsilon soft-energy inner maximizer for live training.
   The current `PgdFullStateEpsilonTrainingConfig` rejects `soft_energy` without
   a safety-cap radius and the runtime path projects onto an L2 radius.
2. Add stateful adaptive-lambda training support with lambda, EMA damage, update
   cadence, target source, and checkpoint/resume persistence.
3. Add fixed validation-batch damage evaluation every 50 batches, using all 5
   replicates and paired clean/adversarial rollouts.
4. Extend run-spec metadata and validation so the planned adaptive-lambda
   contract is recorded without pretending old cap/radius fields are active.
5. Add lightweight tests or dry-run checks proving that the no-cap adaptive
   config is accepted and that attempts to set a cap/radius/projection for this
   row are absent or diagnostic-only.

## Launch Gate

After implementation, a fresh launch-facing spec must still be presented to the
user and explicitly approved before any training launch or billable resource is
created.
