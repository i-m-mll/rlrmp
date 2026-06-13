# Delayed Nominal Peak Recovery Plan

Issue: 6c36536

Date: 2026-06-09

## Current State

The delayed C&S GRU task now has the intended high-level contract: target visible
from trial start, sampled go cue, restored catch trials, prep-only
`nn_output_pre_go`, and canonical 60-step movement-window full Q/R/Q_f scoring
from the go cue. The corrected fixed evaluation bank now uses an explicit
uniform 20-direction, 0.15 m no-catch/catch lens for extLQG comparisons.

The previous apparent half-plane/direction split was an evaluation-bank artifact:
the old helper inferred directions from the mixed validation-target list and
median-rescaled reach length. With the corrected uniform bank, the old direction
split nearly disappears. The remaining nominal issue is milder: the delayed
no-PGD `p_catch_trial=0.5`, `nn_output_pre_go=1e5`, lr=3e-3 row launches with
an extLQG-like initial velocity/acceleration profile but peaks slightly lower
and decays slightly earlier than the matched extLQG trace.

## Working Hypotheses

1. Premature command decay: the controller may produce a near-analytic initial
   movement command after the go cue, then reduce command earlier than extLQG.
2. Premature force/filter support decay: command may not be the first quantity
   to fall; the force/filter state may stop supporting acceleration earlier than
   the analytical rollout.
3. Checkpoint selection: an earlier checkpoint may match extLQG kinematics
   better than the current final/selected checkpoint while preserving low
   pre-go/catch leakage.
4. Prep penalty too strong: `nn_output_pre_go=1e5` may over-bias the shared
   output pathway toward suppression, even if it no longer prevents the initial
   movement burst.
5. Catch fraction too high: `p_catch_trial=0.5` may dilute movement-gradient
   signal enough to bias the movement period toward conservative under-release.

## Immediate Local Diagnostics

Run these on the existing `p_catch_trial=0.5`, `nn_output_pre_go=1e5`,
lr=3e-3 delayed no-PGD row before interpreting new training runs:

- Numeric command-decay diagnostic: compare GRU vs extLQG movement command
  magnitude/direction over movement steps 0-30 and report when GRU support first
  drops below a fixed fraction of the extLQG profile after matching early launch.
- Numeric force/filter/acceleration diagnostic: compare GRU vs extLQG
  force/filter support and acceleration, with the same decay-onset reporting.
- Checkpoint sweep every 1000 batches: evaluate corrected uniform 20-direction
  no-catch/catch banks, tracking peak velocity, time-to-peak, velocity-profile
  shape error, endpoint error, and pre-go/catch leakage.

The diagnostic should answer whether the current profile is caused by premature
controller output decay, downstream force/filter decay, or checkpoint choice.

## Immediate RunPod Rows

Launch two 8D delayed no-PGD rows on one RTX 4090, in parallel if packing allows:

1. Prep-penalty ablation:
   - `p_catch_trial=0.5`
   - `nn_output_pre_go=1e4`
   - lr=3e-3, batch 64, hidden 180, 5 replicates, 12000 batches
   - full analytical Q/R/Q_f, target-relative multi-target, force/filter feedback

2. Catch-fraction ablation:
   - `p_catch_trial=0.40`
   - `nn_output_pre_go=1e5`
   - lr=3e-3, batch 64, hidden 180, 5 replicates, 12000 batches
   - same objective/model/training settings as above

Evaluate both with the corrected fixed banks:

- no-catch velocity profiles with matched 0.15 m extLQG trace;
- catch/pre-go leakage profiles;
- checkpoint/diagnostic summaries using the same movement-window convention.

Do not vary Q/R/Q_f in this pass. The purpose is to isolate delayed-specific
anti-anticipation and movement-support pressure while keeping the canonical C&S
movement objective fixed.

## Deferred Follow-Ups

- [issue:bf71d86] covers go-cue timing distribution and go-cue stratification
  ablations.
- [issue:bcd69a9] covers explicit go-cue output gating. This may be useful, but
  it changes interpretation because pre-go output suppression becomes built into
  the architecture rather than learned.

## Coordination

Cross-reference this plan on:

- [issue:6c36536] delayed-reach task contract and active delayed-training
  tracking surface;
- [issue:198be43] catch-trial restoration and anti-anticipation gate;
- [issue:ffff699] current concrete delayed 8D/6D run surface;
- [issue:c99ad9d] training-methods coordination.
