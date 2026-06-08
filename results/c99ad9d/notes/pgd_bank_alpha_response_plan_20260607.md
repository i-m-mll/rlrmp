# PGD Broad-Epsilon + Calibrated-Bank Training Response Plan

Date: 2026-06-07

## Context

The latest external critique distinguishes three training knobs:

- `rho`: the broad/full-state epsilon budget.
- PGD strength: how well the inner maximizer finds a bad epsilon inside that budget.
- `alpha`: how strongly the outer optimizer is pushed by the adversarial broad-epsilon rollout loss.

It also distinguishes broad/full-state epsilon from the calibrated perturbation bank. The bank is a
restricted family of physical/sensory/task perturbations and is useful for ordinary feedback
competence and transfer, but it is not the formal C&S H-infinity channel. The formal C&S channel is
the broad/full-state additive epsilon channel.

## Current Implementation Reading

The calibrated perturbation bank is not a separate output penalty. When enabled, it changes the
training trial distribution, and the same full Q/R/Q_f rollout loss is computed on the sampled
perturbed trial. Thus `beta` is implicit in the nominal/single-family/combined mixture fractions.

The current PGD lane is also not an explicit separately weighted loss term. It is a Feedbax
`pre_step_fn`:

- `cs_nominal_gru.py` builds `PgdFullStateEpsilonTrainingConfig`.
- `cs_nominal_gru.py` passes `make_broad_epsilon_pgd_pre_step(...)` into the trainer.
- `cs_perturbation_training.py::make_broad_epsilon_pgd_pre_step` creates an adversarial epsilon
  delta for the sampled batch and returns the trial spec with `epsilon = base_epsilon + delta`.

Therefore, when PGD is enabled today, it is applied to every sampled training trial in that batch.
In the recent `pgd_moderate` row, calibrated perturbation training was disabled, so every training
trial was target-relative/multitarget/stochastic plus PGD epsilon, not calibrated-bank plus PGD.

If calibrated perturbation training and PGD are both enabled with the current code, the expected
semantics are: sample nominal/bank trials according to the perturbation mixture, then add PGD
broad/full-state epsilon on top of every sampled trial. This means `beta` remains implicit in the
bank sampling fractions, while `alpha` is effectively implicit and high: all outer updates are made
on adversarialized versions of the sampled trials.

## Important Implementation Prerequisite

The persisted PGD HPS payload records inner-maximizer settings under
`broad_epsilon_pgd_training.inner_maximizer`. The pre-step config reader currently reads top-level
`n_steps` and `step_size_fraction` with defaults. The previous run used the defaults, so this did not
change that run, but future attempts to set `--broad-epsilon-pgd-steps 10` or similar would likely
be ignored unless the reader is fixed to accept the nested fields.

This must be fixed before any stronger-PGD run.

## Interpretation of the Previous PGD Run

The previous PGD row should not be read as "calibrated feedback baseline plus robust pressure."
It was "force/filter-feedback target-relative baseline plus PGD broad epsilon." That explains why
the perturbation bank transfer worsened: the run did not preserve the calibrated-bank training
pressure that produced ordinary feedback competence in earlier rows.

The run also used the deterministic-game-card moderate radius (`L2 = 0.00123243`) while the
included output-feedback robust exact audit used `L2 = 0.00454550`. That is a budget-source mismatch.

## Diagnostic and Perturbation-Bank Additions from the Review

These are part of the next work, not optional extras.

1. Add the output-feedback robust analytical controller to the same perturbation-response diagnostic
   bank as extLQG and the GRUs. The current robust analytical artifact is an exact output-feedback
   audit in analytical coordinates; it is not yet emitted through the same calibrated-bank response
   schema. The next diagnostic packet should include, where possible:

   - extLQG / LQR-Kalman analytical controller;
   - output-feedback robust analytical controller;
   - matched non-robust GRU;
   - PGD-trained GRU.

   This is needed before using the bank to say whether the GRU is or is not matching the robust
   analytical transfer phenotype.

2. Add a stronger same-channel broad-epsilon frozen-policy audit for trained GRUs. This is separate
   from the calibrated bank. It should compare the non-robust and PGD GRUs at matched broad-epsilon
   budgets with a much stronger post hoc inner maximizer than training uses. It should report
   worst-case full-Q/R/Q_f cost, worst-case/nominal ratio, achieved epsilon norm, and epsilon energy
   by time/component group.

3. Expand the calibrated perturbation bank with human-protocol-like lateral mechanical perturbations.
   The current bank has useful command/process/sensory rows, but the motor-control robustness claim
   is easier to interpret with target-aligned lateral/tangential plant-load probes. Add early/mid/late
   signed lateral force/load pulses or steps, with target-relative alignment.

4. Expand false-feedback probes. Add radial/tangential position- and velocity-feedback offsets,
   signed and early/mid/late, so the diagnostic can test whether a robust-like high-gain controller
   becomes more sensitive to false sensory feedback rather than only asking whether mechanical
   perturbations are attenuated.

5. Make target-relative radial/tangential decomposition a default reporting view for the bank. Norms
   remain useful, but lateral/radial decomposition is needed for comparing across reach directions
   and for interpreting the expected robust-control phenotype.

6. Keep response-size metrics prominent: max and AUC of `delta x`, max and AUC of `delta u`,
   endpoint error, terminal speed, delta cost, attenuation-style metrics, and GRU/extLQG plus
   GRU/robust-analytical ratios where denominators are well-defined. These should be reported per
   perturbation family and timing bin, not only as a pooled aggregate.

7. Predeclare phenotype criteria for the next packet. A useful robust-like GRU should show at least
   structured evidence beyond higher peak velocity, especially reduced plant/mechanical deviation
   and/or stronger useful corrective command responses. Sensory false-feedback rows may move in the
   opposite direction and should be interpreted separately.

## Options

### Option A: Minimal continuation with current semantics

Enable calibrated perturbation training again and keep PGD on every sampled trial.

This corresponds to:

```text
sample nominal/bank trial from the calibrated mixture
add PGD broad/full-state epsilon to that same trial
optimize ordinary full-Q/R/Q_f rollout loss
```

Advantages:

- Smallest implementation change.
- Keeps the user's intended interpretation: beta is the sampling mixture, not a new penalty.
- Ensures every update receives broad-epsilon pressure.

Risks:

- No explicit alpha. PGD pressure may dominate every trial, especially if the corrected budget is
  large.
- It cannot answer whether partial PGD pressure would preserve feedback competence better.

### Option B: Explicit PGD mixture or alpha

Add explicit control over PGD weighting, either by:

- applying PGD only to a configurable fraction of trials/batches; or
- computing clean/bank and PGD losses separately and combining them with an explicit coefficient.

Advantages:

- Directly tests 5.5 Pro's alpha recommendation.
- Lets us prevent PGD from overwriting the calibrated-bank feedback policy.

Risks:

- More implementation and smoke-testing before the next run.
- Requires deciding whether alpha is a sample fraction, a loss coefficient, or gradient-ratio tuned.

## Recommended Next Path

Proceed in two layers.

First, do the implementation hardening that is required either way:

1. Fix PGD config parsing so nested `inner_maximizer.n_steps` and
   `inner_maximizer.step_size_fraction_of_l2_radius` are actually consumed.
2. Add PGD inner-loop diagnostics: declared radius, achieved epsilon norm/radius, inner loss before
   and after PGD, per-step improvement when feasible, projection saturation, and epsilon energy by
   component/time group.
3. Add a run-spec field that states whether PGD is applied to every sampled trial, only a subset, or
   a separately weighted branch.
4. Add or regenerate the same-bank robust analytical perturbation emitter if possible, so the bank
   can include extLQG and output-feedback robust analytical references.
5. Add the perturbation-bank expansions and target-relative reporting updates listed above, or at
   minimum mark any not-yet-implemented expansion as a named residual before interpreting the next
   training screen.

Second, run a controlled screen. My preferred first screen is Option A, because it matches the user's
current preference and minimizes new machinery:

- Force/filter feedback on.
- Target-relative multitarget on.
- Calibrated perturbation training on, probably using the better of the prior `cal_small` and
  `cal_moderate` settings.
- PGD broad/full-state epsilon on every sampled trial.
- Budget source explicitly set to the output-feedback robust analytical audit radius, or run a
  matched-budget pair if we are still uncertain.
- PGD inner-max strength increased after the config-reader fix, with achieved norms logged.

This is the cleanest immediate test of:

```text
ordinary feedback competence from calibrated-bank sampling
+ formal broad-epsilon robust pressure from PGD
```

If that row still shifts only kinematics and does not improve same-channel audit or bank transfer,
then implement Option B and sweep alpha/fraction explicitly.

## Decisions Still Needed

1. Budget source:
   - Use output-feedback robust exact-audit radius (`0.00454550`) for the next PGD run, or
   - run a short ladder around the deterministic game-card and output-feedback radii.

2. PGD application:
   - Keep PGD on every sampled trial for the first corrected run, or
   - first implement explicit alpha/sample-fraction control.

3. Training start:
   - Train from scratch with the calibrated-bank+PGD regime for a clean comparison, or
   - fine-tune from the best existing calibrated-bank force/filter-feedback checkpoint. Fine-tuning
     may preserve feedback competence better, but it needs a separately verified arbitrary-checkpoint
     initialization path and a clear optimizer/schedule policy.

My recommendation is: fix the PGD instrumentation/config bug, then run one from-scratch
calibrated-bank+PGD row with PGD on every sampled trial and the corrected output-feedback budget.
Use the diagnostics to decide whether explicit alpha/fraction is necessary.
