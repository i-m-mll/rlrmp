# Candidate procedure specs for adaptive soft-adversary bounds

Date: 2026-06-30

Review inputs:

- `/Users/mll/Downloads/outputs/gpt8.md`
- `/Users/mll/Downloads/outputs/gpt9.md`

Relevant live ledger context:

- [b413bb0](http://localhost:19420/#/issues?repo=rlrmp&issue=b413bb0&sidebar=hidden) is closed and superseded. Its useful result is diagnostic: fixed beta/lambda can become inactive late, and the finite rows do not establish valid bounded finite-policy comparisons.
- [0a46652](http://localhost:19420/#/issues?repo=rlrmp&issue=0a46652&sidebar=hidden) remains the broader calibrated soft-constraint adversary umbrella.
- [54389a4](http://localhost:19420/#/issues?repo=rlrmp&issue=54389a4&sidebar=hidden) remains the lambda-calibration and frozen-audit umbrella.
- [2e60620](http://localhost:19420/#/issues?repo=rlrmp&issue=2e60620&sidebar=hidden) is an open deferred optimizer-benchmark reminder. It is not launch authorization.

This file is planning-only. It does not authorize training, pod acquisition,
protected auth, push, merge, issue closure, or Mandible comments.

## 1. Review stance and confidence

### What I treat as hard requirements

- Do not let inherited energy caps, trust radii, or projection radii define new
  launch-facing lambda values. They may be cited only as historical diagnostics
  or provenance.
- Keep the corrected per-trial soft objective contract: optimize
  `mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]`, with the batch reduction
  explicit.
- Keep three quantities separate:
  - `lambda_curv`: local curvature/well-posedness scale from the corrected HVP
    object.
  - `lambda_zero`: activity threshold above which zero epsilon can be the true
    soft optimum for the current controller and mechanism.
  - `lambda_guard` / `lambda_cap`: cap-avoidance, trust-region, or guard
    pressure, which depends on an external bound and is therefore not a
    soft-game scale unless the bound is independently justified.
- Define analytical damage as a paired within-controller counterfactual. The
  preferred target uses the same analytical controller, same trial, same target,
  same initial condition, same nominal noise seed, and same Q/R/Qf cost, with
  versus without the matched analytical adversarial disturbance.
- Keep analytical cost differences labeled by what they measure. Clean
  H-infinity-beta cost minus clean extLQG/nominal cost is nominal conservatism,
  not adversarial damage. H-infinity beta under matched disturbance minus a
  nominal clean baseline is total robust-condition burden, not pure damage.
- Use total Q/R/Qf cost for the damage target when that is the training
  objective. Kinematic summaries remain phenotype sidecars or a separately
  labeled phenotype target.
- Treat cost/energy ratio as a diagnostic sidecar, not the sole target, because
  it becomes unstable when selected energy is tiny.
- If a later launch-facing spec needs task/controller settings, carry forward
  only the baseline contract: no-PGD H0 6D open-loop moderate `const_band16`
  baseline row, target-relative `const_band16`, 6D C&S no-integrator state,
  C&S nominal GRU hidden size 180 with 5 replicates, h0/initial-hidden encoder
  baseline, calibrated moderate perturbation-training baseline, full analytical
  Q/R/Qf, batch size 64, seed 42, controller lr `3e-3`, 500-batch warmup from
  0.1 fraction, cosine alpha `0.01`, gradient clip 5, checkpoints every 500
  batches, and log step 100. Re-verify these before launch-facing use.
- The next work should be about closed-loop adversaries or adaptive soft
  adversary procedures, not baseline task changes, unless a task change is
  explicitly justified as part of the experiment question.
- Treat `direct_epsilon` as the first proof-of-concept mechanism. Finite
  closed-loop mechanisms should follow only after the direct path shows the
  adaptive controller, diagnostics, and pass/fail gates are meaningful.

### What looks plausible but not proven

- The separation of `lambda_curv`, `lambda_zero`, and `lambda_guard` / cap
  pressure is conceptually consistent with the repo's corrected HVP/p90 contract
  and cap invalidation work.
- Adaptive lambda is a plausible next mechanism because a fixed beta-scaled
  lambda can become inactive late in training. That does not prove adaptive
  lambda will work; it only identifies a failure mode it is designed to test.
- A paired damage-matching target is a cleaner first target than direct
  phenotype matching because it stays closer to the analytical cost/game object.
- Mechanism-specific lambdas are likely needed. Direct epsilon, linear no-bias,
  and affine closed-loop policies have different parameterizations and different
  opportunities to become zero, overly strong, or optimizer-limited.

### Speculative ideas to keep separate from requirements

- A high nonbinding guard cap could be chosen from a physical/process-validity
  envelope, or from a deliberately labeled pilot distribution of selected
  energies. This is not yet a default bound.
- Phenotype matching, such as peak velocity or trajectory-shape matching to an
  analytical reference, may be useful after damage matching. It should be
  labeled as phenotype-calibrated training, not as proof of analytical
  H-infinity equivalence.
- Dual-style lambda updates from the locomotion paper are inspiration only here.
  I did not verify the paper in this pass, and its physical force cap is not
  automatically transferable to RLRMP process epsilon.

## 2. Candidate procedure A: direct-epsilon adaptive damage matching

### Objective

Test whether a direct-epsilon soft adversary can remain nonzero, finite,
recoverable, and mostly independent of any guard cap when lambda is adapted to
match a paired analytical-reference damage target.

This is the recommended first proof-of-concept because it changes the fewest
moving parts: no finite-policy parameterization, no affine bias, and no need to
interpret closed-loop policy gains before the adaptive rule is debugged.

### Mechanism

- Mechanism: `direct_epsilon` only.
- Inner game: maximize `Delta J - lambda * E` under the corrected per-trial
  soft objective.
- Controller training: unchanged baseline task/controller settings if and only
  if this advances to a launch-facing spec.
- Start with a no-weight-update frozen check on representative held-out batches.
  Advance to a short training smoke only after the frozen check shows nonzero
  selected epsilon across the intended lambda update range.

### Lambda and bound rule

- Initialize `lambda_0` from the corrected cap-independent HVP/p90 source,
  such as `lambda(beta) = beta^2 * lambda_curv_p90`, but treat this only as an
  initialization and lower-sanity reference, not as a final fixed training
  truth.
- Define a damage target before training:
  - Preferred: `D_beta_ref = mean_i [J_i(K_beta, F_beta) - J_i(K_beta, 0)]`,
    where `K_beta` is the analytical H-infinity controller and `F_beta` is its
    matched analytical adversarial disturbance. Compute this as paired rollouts
    with the same trial, target, initial condition, nominal noise seed, horizon,
    disturbance-channel convention, Q/R/Qf, masks, state basis, and loss
    reduction.
  - The learned target compared to `D_beta_ref` is paired incremental learned
    damage: `mean_i [J_i(psi, epsilon_star) - J_i(psi, 0)]`, using the same
    task/loss convention.
  - Do not use `mean_i[J_i(K_beta, 0) - J_i(K_ref, 0)]` as damage; that is
    nominal conservatism. Do not call
    `mean_i[J_i(K_beta, F_beta) - J_i(K_ref, 0)]` pure damage; that is total
    robust-condition burden relative to the nominal clean baseline.
  - Optional separate candidate: a fixed-victim training-pressure target, such
    as damage induced by a beta-level adversary solved against a fixed reference
    controller. Keep this separate from matched-pair damage and label it as a
    fixed-victim or training-pressure target.
  - Fallback only if the analytical target is unavailable: a predeclared pilot
    target from held-out no-PGD baseline vulnerability, explicitly labeled as
    empirical and not analytical.
- Update in log space on a slow cadence:
  - If selected paired `Delta J` is consistently below target or zero, decrease
    lambda.
  - If selected paired `Delta J` is above target or destabilizing, increase
    lambda.
  - Clip per-update changes and smooth over held-out batches so one noisy
    adversary update does not dominate.
- Use any hard cap only as a validity guard. The guard must be chosen and
  justified before a run spec. It must not enter lambda choice, readiness, or
  scientific interpretation.

### Diagnostics

Record at least:

- `lambda`, `log_lambda_delta`, and update reason.
- `J_clean`, `J_adv`, paired `Delta J`, `E`, `Delta J / E` as a diagnostic,
  `lambda * E`, and `Delta J - lambda * E`.
- Zero-selected fraction and zero-incumbent acceptance rate.
- Guard-bound fraction and selected-energy distribution.
- Nonfinite/overflow status.
- Clean task learning curve and adversarial task learning curve.
- Kinematic sidecars: velocity RMSE, peak velocity, time-to-peak, endpoint
  error, terminal speed, and trajectory-shape distance.

### Pass/fail gate

Pass as a proof-of-concept only if:

- The selected adversary remains nonzero after the early training phase and at
  late checkpoints.
- The guard is not materially determining selected epsilon. Any sustained guard
  binding, or any evidence that the guard rather than lambda sets the adversary
  scale, fails the pure-soft interpretation.
- `Delta J` tracks the target within a predeclared tolerance without lambda
  runaway or oscillation.
- The controller still learns the nominal task and does not collapse under the
  adversary.
- The row is interpretable as adaptive-soft direct epsilon, not as hard-capped
  PGD.

Fail or bracket if:

- The adaptive rule drives lambda below a curvature/well-posedness sanity floor
  to keep activity.
- The exact or best-found soft optimum is still zero for most late checkpoints.
- Recoverable damage cannot be achieved without guard binding or unstable
  states.

### Main risks

- The analytical damage target may be unavailable, mismatched to the learned GRU
  loss, or too high/low for a training controller.
- Matched-pair damage may not be monotone in beta because beta changes both the
  analytical controller and the analytical adversary. If monotone training
  pressure is the goal, use a separate fixed-victim target and label it that way.
- Adaptive lambda can chase controller learning dynamics and become a second
  uncontrolled curriculum.
- A direct-epsilon proof may not transfer to finite closed-loop mechanisms.
- A high guard can still become a hidden hard game unless the diagnostics fail
  closed.

## 3. Candidate procedure B: activity-window and frozen lambda-controller audit

### Objective

Before any new billable training, estimate whether each mechanism has a usable
soft activity window at frozen checkpoints:

```text
lambda_curv < lambda < lambda_zero
```

This is not a replacement for training. It is a guard against launching another
row where fixed or adaptive lambda must either go zero or become guard-bound.

### Mechanism

- First run on `direct_epsilon`.
- Repeat only after the direct path is understood for `linear_no_bias` and
  `affine` using the live graph-component closed-loop finite-policy path.
- Use frozen controllers and fixed held-out batches. Do not update controller
  weights.
- Include Adam as the training-relevant inner optimizer. Use L-BFGS-B or an
  equivalent stronger solver only as a frozen-audit reference, consistent with
  [2e60620](http://localhost:19420/#/issues?repo=rlrmp&issue=2e60620&sidebar=hidden).

### Lambda and bound rule

- Estimate or reuse `lambda_curv` from corrected HVP/Lanczos sources where the
  object matches the mechanism.
- Estimate `lambda_zero` empirically by bracketing lambda values and checking
  where the selected soft optimum becomes zero under the best available inner
  optimizer.
- Keep cap/trust-radius quantities as sidecars only.
- Exercise the proposed adaptive lambda update on the frozen batch sequence:
  the audit should show whether the controller would move lambda into a stable,
  active, non-guard-bound range.

### Diagnostics

Record:

- `lambda_curv` source and convention.
- Empirical `lambda_zero` bracket with optimizer and stopping details.
- Best penalized objective gain over zero.
- Best raw task gain and selected energy.
- Whether Adam agrees with the reference solver's zero/nonzero classification.
- Guard-bound fraction as a diagnostic sidecar, not a success criterion.
- Per-mechanism differences in scale and optimizer reliability.

### Pass/fail gate

Pass for a later training spec only if:

- A nonempty active window is observed or strongly suggested for the mechanism.
- Adam or the intended training inner optimizer matches the reference solver's
  zero/nonzero classification on the relevant frozen cases.
- The adaptive lambda rule moves toward the active window without using a cap as
  the scale-setter.

Fail or defer if:

- The active window is empty or too narrow to target robustly.
- Adam fails the reference classification for the intended mechanism.
- The only nonzero solutions are guard-bound.

### Main risks

- Frozen windows may not predict co-training dynamics.
- `lambda_zero` is global over the searched disturbance family and can be
  optimizer-limited if the reference solver is still weak.
- The audit can overfit to the selected held-out batches unless the batch
  provenance and sampling rule are explicit.

## 4. Candidate procedure C: finite closed-loop adaptive lambda after direct proof

### Objective

Test whether closed-loop finite adversaries can hit the same damage or behavior
target as direct epsilon while staying live, bounded by a nonbinding guard, and
recoverable for the controller.

This should follow candidate A, not replace it, unless the explicit goal is
only to debug finite-adversary infrastructure.

### Mechanism

- Mechanisms: `linear_no_bias` first, then `affine`.
- Use the live graph-component rollout path. Do not use legacy/static clean
  rollout materialization for launch-facing closed-loop finite rows.
- Use one shared finite adversary policy per batch, not independent per-trial
  policies.
- Keep the no-bias policy centered so zero feature input produces zero
  disturbance. Treat affine bias as a separate stronger mechanism.
- Consider a clean/adversarial outer-loss mixture or adversarial ramp only as a
  training-stability curriculum. It must be reported as a curriculum, not as a
  baseline task change.

### Lambda and bound rule

- Use the same paired target definition as candidate A, preferably
  `D_beta_ref`.
- Maintain separate adaptive lambda states for each mechanism and possibly for
  each substrate/regime. Do not force a single global lambda across direct,
  linear, and affine mechanisms.
- Initialize from the relevant cap-independent HVP/generalized-eigen source
  when available.
- Use high validity guards only after defining their physical or numerical
  justification. Guard values from [b413bb0](http://localhost:19420/#/issues?repo=rlrmp&issue=b413bb0&sidebar=hidden)
  are historical diagnostics only.

### Diagnostics

All candidate A diagnostics, plus:

- Finite-policy parameter norms and feature sensitivity.
- Realized process-epsilon energy over the live perturbed rollout.
- Bias contribution for affine rows.
- Per-target and per-time disturbance summaries.
- Adam-vs-reference solver agreement from the frozen audit.
- Whether closed-loop feedback amplifies perturbations beyond the intended
  recoverable damage target.

### Pass/fail gate

Pass only if:

- The direct-epsilon adaptive procedure has already passed or the finite run is
  explicitly framed as infrastructure/debugging rather than scientific
  comparison.
- Frozen audits show optimizer reliability for the mechanism.
- Training selects nonzero adversaries without relying on the guard.
- Damage stays recoverable and close to the common target.
- Kinematic sidecars remain interpretable relative to the no-PGD H0 baseline
  and analytical beta reference.

Fail or bracket if:

- Linear or affine policies produce large realized perturbations that cannot be
  controlled by lambda without hard guard binding.
- Affine bias behaves like an open-loop forcing channel that overwhelms the
  closed-loop interpretation.
- Direct epsilon and finite mechanisms require incompatible targets, implying
  the common target is poorly specified.

### Main risks

- Closed-loop finite policies can amplify small parameter changes into large
  realized perturbations.
- Mechanism-specific adaptation may make cross-mechanism comparisons harder.
- Affine rows can look effective by injecting bias rather than by discovering a
  meaningful closed-loop disturbance policy.
- A clean/adversarial mixture can stabilize training but complicate the claim
  that the row is a pure soft-adversary procedure.

## 5. Recommended next step

The safest next step is candidate A plus a small piece of candidate B:

1. Draft a no-launch implementation plan for `direct_epsilon` adaptive damage
   matching.
2. Materialize the paired analytical damage target `D_beta_ref` or state plainly
   that it does not yet exist.
3. Run a frozen, no-weight-update adaptive-lambda replay on held-out batches to
   confirm the update rule can keep `direct_epsilon` nonzero without cap
   binding.
4. Only after that, prepare a user-approved smoke run spec. The smoke should be
   direct-epsilon first, with finite mechanisms deferred until the direct path
   and optimizer evidence are clean.

Finite closed-loop mechanisms should follow when:

- The direct adaptive procedure has a stable diagnostic surface.
- The live finite mechanism's frozen audit has a nonempty active window.
- Adam or the intended training inner optimizer agrees with the stronger frozen
  reference on zero/nonzero classification.
- The guard cap is justified as a nonbinding validity guard.

## 6. Questions to resolve before any run spec

- What exact paired analytical reference defines `D_beta_ref` for the beta of
  interest, and is it computed under the same Q/R/Qf, masks, state basis, target
  support, nominal noise seeds, and task convention as training?
- Should beta index target damage, target phenotype, or both? If both, which one
  is the primary gate and which one is a sidecar?
- Is any fixed-victim training-pressure target needed, or is the matched-pair
  damage target sufficient for the first proof-of-concept?
- What lambda update cadence, smoothing, clipping, and freeze policy should be
  used? For interpretability, should lambda adapt only during an early
  calibration window and then freeze?
- Is there a physical or simulator-validity basis for a process-epsilon guard?
  If not, what pilot procedure is acceptable for choosing a high nonbinding
  guard, and how will it be labeled?
- What exact tolerance defines "near target" damage and "rare enough" guard
  binding? These must be locked before launch rather than tuned after results.
- Should lambda be shared across replicates, separate per replicate, separate
  per mechanism, or separate per substrate/regime?
- What is the minimum frozen-audit evidence needed before including
  `linear_no_bias` or `affine` in a training smoke?
- Does [2e60620](http://localhost:19420/#/issues?repo=rlrmp&issue=2e60620&sidebar=hidden)
  need to be resolved before any finite closed-loop training row, or only before
  choosing between Adam and L-BFGS for a mechanism that fails the frozen audit?
- Which tracking issue should own the adaptive-lambda proof-of-concept, and
  should it be a new child under [54389a4](http://localhost:19420/#/issues?repo=rlrmp&issue=54389a4&sidebar=hidden)
  or under the broader [0a46652](http://localhost:19420/#/issues?repo=rlrmp&issue=0a46652&sidebar=hidden)
  umbrella?
- Which existing no-PGD H0 `const_band16` baseline artifact is the comparison
  source of record for the eventual smoke, and does it still match the baseline
  contract?
