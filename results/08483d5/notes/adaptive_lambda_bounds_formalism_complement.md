# Adaptive Lambda and Bound-Choice Complement

## 1. Purpose and relationship to the existing formalism record

This note is a proposed supplement to the existing soft-constraint adversary
lambda formalism. It does not replace the existing record. The existing record
defines the shared soft game, corrected realized-epsilon energy convention,
family-specific lambda estimation, fixed-lambda row semantics, adaptive-row
labeling, and stale cap/trust-radius constraints. This complement adds a
methods-facing contract for adaptive lambda rows and for choosing energy,
force, and guard bounds without reintroducing inherited cap values as
scientific defaults.

The existing formalism record of reference is:

- Materialized review copy:
  `/tmp/rlrmp_adaptive_bounds_20260630_205501/soft_constraint_adversary_lambda_formalism.md`
- Tracked record of reference:
  `results/54389a4/notes/soft_constraint_adversary_lambda_formalism.md`
- Pinned artifact:
  `artifact:sha256:23c78bbacff6c7f43d83749ce743adf6551c46fd0b00f47593c51a4649daca82`
- Commit:
  `af1e26a` on `integration/54389a4-soft-lambda-calibration`

This complement is planning material only. It is not launch authorization, not
a run spec, and not a protected-branch merge request. Any future run spec must
lock concrete row parameters, budget exposure, diagnostics, pass/fail gates,
and user approval separately.

The central distinction is:

- A fixed-lambda row trains against one fixed soft game. This is the cleanest
  H-infinity-like training object when the lambda source is family-specific,
  cap-independent, and validated by frozen audits.
- An adaptive-lambda row is a curriculum. Its lambda changes according to a
  predeclared rule. It can be scientifically useful, but it must be labeled as
  adaptive and compared against fixed-lambda rows rather than described as a
  fixed-beta H-infinity-equivalent row.

For adaptive rows, beta should index a reference target, such as paired
analytical damage or phenotype from an output-feedback H-infinity reference. It
should not be treated as a guarantee that `lambda = beta^2 * lambda_star` will
keep a learned soft adversary active throughout training.

## 2. Normative additions

### 2.1 Adaptive lambda semantics

An adaptive-lambda row must define the following before any run spec:

- `method_class`: `adaptive_lambda_curriculum`.
- `mechanism`: for example `direct_epsilon`, `linear_no_bias`, `affine`, `MLP`,
  or `GRUadv`.
- `target_signal`: the scalar or vector that drives lambda adaptation.
- `target_source`: paired analytical reference, empirical pilot, phenotype
  reference, fixed-victim training-pressure reference, or another named source.
- `lambda_initialization`: cap-independent source for `lambda_0`.
- `lambda_update_rule`: update equation, cadence, smoothing, clipping, and
  freeze policy.
- `lambda_bounds`: allowed numerical range and what happens if a bound is hit.
- `guard_policy`: whether a hard guard exists, what it means, and what binding
  rate fails the soft-row interpretation.
- `fixed_lambda_comparator`: at least one fixed-lambda row or frozen replay
  comparator that keeps adaptive-curriculum conclusions anchored to the fixed
  soft-game formalism.

Lambda updates are controller logic, not a proof that the trained controller has
an analytical H-infinity level. Unless a future method deliberately
differentiates through lambda updates, the default interpretation is
stop-gradient adaptation: lambda is updated from diagnostics outside the
controller-gradient objective.

### 2.2 Three lambda quantities

Future methods documents and run specs should keep these quantities separate:

#### `lambda_curv`

`lambda_curv` is the local soft-game curvature or well-posedness scale. It is
estimated from the corrected HVP/Lanczos or generalized-eigen object in the
realized-epsilon energy metric for the active adversary family.

It is cap-independent. It may initialize an adaptive row or define a sanity
floor. It must not include a hard-cap, trust-radius, or optimizer-interiority
term unless that term is separately labeled as diagnostic-only.

#### `lambda_zero`

`lambda_zero` is the activity threshold above which zero epsilon can be the
true soft optimum for the current controller, batch distribution, and adversary
mechanism:

```text
lambda_zero = sup_{epsilon != 0} Delta J(epsilon) / E(epsilon)
```

In practice this is an empirical bracket, not a single exact constant. It
should be estimated with frozen controllers, deterministic held-out batches,
and both the intended inner optimizer and a stronger reference procedure where
feasible.

The useful pure-soft activity window is:

```text
lambda_curv < lambda < lambda_zero
```

If this window is empty, unstable, or optimizer-dependent for a mechanism, that
mechanism should not advance to launch-facing training without reframing the
row as infrastructure debugging, a curriculum experiment, or a hard-bounded
game.

#### `lambda_guard`

`lambda_guard` is any bound-dependent pressure scale, including cap-avoidance,
trust-region, optimizer-interiority, or numerical-stability thresholds.

It answers a different question from `lambda_curv`:

```text
What lambda keeps the optimizer interior to this chosen guard?
```

It does not answer:

```text
What lambda calibrates the soft objective for this adversary family?
```

`lambda_guard` is therefore diagnostic-only unless a future run deliberately
defines a scientifically meaningful bounded-disturbance game. Historical
energy caps, radii, and trust-region values may be reported as provenance or
sidecars, but they must not become defaults or launch-facing lambda floors.

### 2.3 Bound taxonomy

Each bound or norm-like quantity in a soft-adversary row must be assigned to one
of these categories.

#### A. Scientific energy definition

This is the energy term in the soft game:

```text
E(epsilon) = mean_i sum_t,d mask[i,t] * epsilon[i,t,d]^2
```

The reduction must match the task-loss reduction. If the task loss is a
per-trial batch mean, energy is also a per-trial batch mean. This is the only
energy definition that enters the fixed soft objective unless a run spec
explicitly defines a different scientific objective.

The scientific energy definition should not divide by time, dimension, target
count, or reach length unless the objective is deliberately changed and named.

#### B. Physical-validity guard

A physical-validity guard limits disturbances to a domain that is independently
justified by the plant, task, simulator, biological regime, or experimental
apparatus. Examples could include measured disturbance amplitudes, known force
or acceleration envelopes, or a simulator-valid process-noise range.

If a physical-validity guard is scientifically meaningful and expected to bind,
the row is a bounded-disturbance experiment, not a pure soft row. If the row is
intended to remain a pure soft row, the physical guard should be high and
binding should be rare enough to pass a predeclared threshold.

No force limit from another paper or system transfers automatically to RLRMP
process epsilon. A force-like guard requires an explicit mapping from process
epsilon to the physical quantity being bounded.

#### C. Numerical optimizer guard

A numerical optimizer guard prevents nonfinite values, invalid simulator
states, unstable line-search proposals, or optimizer excursions that are not
part of the scientific object. It may include trust radii, proposal rejection,
temporary clipping, or emergency finite checks.

This guard is not a success criterion. Frequent binding is a failure of the
pure-soft interpretation or of the optimizer setup. The run must report the
binding rate and should fail closed if the selected adversary scale is being
set by this guard.

#### D. Diagnostic sidecars

Diagnostic sidecars include historical caps, cap ratios, selected-energy
percentiles, realized force summaries, raw policy-output norms, trust-radius
ratios, and other quantities useful for interpreting optimizer behavior.

They may explain why a row behaved as it did. They must not determine
`lambda_star`, `lambda_0`, pass/fail success, or launch-facing defaults unless
they are promoted through an explicit scientific or numerical-guard rationale.

### 2.4 Analytical damage target taxonomy

For adaptive damage matching, the preferred analytical target is paired
within-controller damage. Let `K_beta` be the analytical H-infinity controller at
the beta of interest, `F_beta` its matched analytical adversarial disturbance,
and `K_ref` the extLQG/nominal clean reference. For trial `i`, `J_i(K, F)` uses
the same task, target, initial condition, nominal noise seed, horizon,
disturbance-channel convention, Q/R/Qf, masks, state basis, and loss reduction
that training will use.

The preferred pure damage target is:

```text
D_beta_ref = mean_i [J_i(K_beta, F_beta) - J_i(K_beta, 0)]
```

This is the extra cost caused by the matched analytical disturbance against its
own matched analytical controller. The learned adaptive row should compare this
to paired incremental learned damage:

```text
D_hat_psi(lambda) =
    mean_i [J_i(psi, epsilon_star(lambda)) - J_i(psi, 0)]
```

Two nearby quantities are useful but should not be renamed as pure damage:

```text
N_beta = mean_i [J_i(K_beta, 0) - J_i(K_ref, 0)]
```

`N_beta` is nominal conservatism: the clean-task cost paid by using the robust
analytical controller instead of the nominal/extLQG controller.

```text
T_beta = mean_i [J_i(K_beta, F_beta) - J_i(K_ref, 0)]
```

`T_beta` is total robust-condition burden relative to the nominal clean
baseline. Under matched conventions it decomposes into nominal conservatism plus
matched adversarial damage, but it is not pure damage.

Matched-pair damage may not be monotone in beta because beta changes both the
analytical controller and the matched analytical adversary. If the scientific
question needs a monotone training-pressure target, define a separate
fixed-victim target, for example:

```text
D_beta_fixed_victim =
    mean_i [J_i(K_ref, F_beta_given_K_ref) - J_i(K_ref, 0)]
```

where the beta-level adversary is solved against the fixed reference controller.
This is a different target family and should be labeled as fixed-victim or
training-pressure calibration, not as matched-pair analytical damage.

When the training objective is total Q/R/Qf cost, `D_beta_ref` and
`D_hat_psi(lambda)` should use that same total cost. Kinematic summaries such as
velocity RMSE, peak velocity, time-to-peak, endpoint error, terminal speed, and
trajectory shape remain phenotype sidecars or separate phenotype targets. The
cost/energy ratio `Delta_J / E_selected` is also a diagnostic sidecar rather
than the sole target, because it becomes unstable when selected energy is tiny.

## 3. Direct-epsilon-first adaptive procedure

The first adaptive proof-of-concept should use `direct_epsilon`. It has the
fewest moving parts: no policy feature basis, no shared finite-policy
parameterization, no affine bias decomposition, and no closed-loop gain
interpretation before the adaptive rule is debugged.

### 3.1 Hard requirements before a run spec

The direct-epsilon adaptive procedure must satisfy these requirements before
any launch-facing run spec:

1. Use the corrected per-trial soft objective:

```text
mean_i [Delta J_i(epsilon_i) - lambda * E_i(epsilon_i)]
```

2. Initialize lambda from a cap-independent source, preferably:

```text
lambda_0 = beta^2 * p90(lambda_curv_direct)
```

where `lambda_curv_direct` comes from the corrected direct-epsilon HVP/Lanczos
or equivalent generalized-eigen artifact. This initializes the adaptive row; it
does not settle the final training lambda.

3. Estimate an empirical `lambda_zero_direct` bracket on frozen controllers and
deterministic held-out batches. The bracket must state optimizer, stopping
criteria, batch provenance, target support, masks, and whether the zero
incumbent was exactly available.

4. Define the target signal before training. The preferred first target is
paired analytical damage:

```text
D_beta_ref = mean_i [J_i(K_beta, F_beta) - J_i(K_beta, 0)]
```

computed under the same task, target, initial condition, nominal noise seed,
horizon, disturbance-channel convention, Q/R/Qf, masks, target centering, state
basis, and loss reduction as the training objective. The learned quantity
compared against this target is:

```text
D_hat_psi(lambda) =
    mean_i [J_i(psi, epsilon_star(lambda)) - J_i(psi, 0)]
```

Clean robust-controller cost minus clean nominal cost is nominal conservatism,
not adversarial damage. Matched robust-controller cost under disturbance minus a
nominal clean baseline is total burden, not pure damage.

5. If the paired analytical damage target is not available, any empirical
fallback target must be explicitly labeled as empirical and not analytical. It
should not be presented as beta-equivalent H-infinity calibration. A
fixed-victim training-pressure target may be useful, but it is a separate target
family and should be named that way.

6. Define a guard policy before training. A guard may be physical-validity,
numerical, or diagnostic-only. It must have a source, a binding-rate failure
threshold, and a rule for what happens when it binds. It must not define
lambda.

7. Run a frozen, no-controller-update adaptive replay on held-out batches before
any training smoke. This replay should show that the update rule can move
lambda into an active, non-guard-bound regime for `direct_epsilon`.

### 3.2 Candidate adaptive update

A simple candidate update is a slow log-space controller:

```text
error_k = (smooth(Delta J_k) - D_beta_ref) / (D_beta_ref + eps)

log_lambda_{k+1} =
    clip(
        log_lambda_k + eta_lambda * clip(error_k, -c, c),
        log_lambda_min,
        log_lambda_max
    )
```

Interpretation:

- If selected damage is too high, increase lambda.
- If selected damage is too low or the selected adversary is zero, decrease
  lambda.
- Smooth over held-out batches or evaluation windows so one noisy adversary
  selection does not dominate the curriculum.
- Use clipped updates and record every clip event.
- Treat movement below the `lambda_curv` sanity floor as a failure or bracketed
  result unless a new formal argument justifies the row.

Candidate variants:

- Freeze lambda after an early calibration window to recover a clearer
  fixed-game interpretation for the remainder of training.
- Log the cost/energy ratio as an attenuation-like diagnostic; do not use it as
  the sole default target because it is unstable when selected energy is tiny.
- Use a fixed-victim training-pressure target only as a separately labeled
  candidate family, not as a replacement for matched-pair damage.
- Use phenotype error as a second-stage correction only after damage matching
  is coherent.

### 3.3 Pass gate

A direct-epsilon adaptive row passes the proof-of-concept gate only if all of
the following are true under predeclared tolerances:

- The selected adversary remains nonzero after early training and at late
  checkpoints.
- The guard is not materially setting the selected disturbance scale.
- Selected `Delta J` tracks the target signal within the locked tolerance
  without lambda runaway, persistent clipping, or oscillation.
- The row remains finite: no persistent nonfinite objectives, gradients, HVPs,
  selected epsilon, or simulator states.
- The controller still learns the nominal task and does not collapse under the
  adaptive adversary.
- The row is interpretable as adaptive-soft direct epsilon, not hard-capped PGD
  with a soft side penalty.
- Frozen replay and training diagnostics agree on the qualitative active-window
  story.

### 3.4 Fail or bracket conditions

The row fails or brackets rather than passes if any of these occur:

- The update rule must drive lambda below the cap-independent curvature sanity
  floor to maintain activity.
- The best-found or exact soft optimum remains zero for most late checkpoints.
- Recoverable target damage can be achieved only when the guard binds
  materially.
- The intended optimizer disagrees with a stronger frozen-audit reference on
  zero/nonzero classification in the launch-relevant region.
- Damage matching succeeds only by making nominal task learning fail.
- The target signal is later found to be computed under a different objective,
  mask, state basis, or reduction convention from training.

## 4. Extension to finite closed-loop mechanisms

Finite closed-loop mechanisms should follow the direct-epsilon proof of concept
rather than replace it, unless the row is explicitly scoped as infrastructure
debugging.

### 4.1 Hard requirements

Before launch-facing finite adaptive rows:

- Use the live graph-component closed-loop path. A clean-rollout or static
  feature replay is a diagnostic, not a launch-facing live closed-loop row.
- Estimate or cite mechanism-specific `lambda_curv_A`; do not reuse the
  direct-epsilon estimate as the finite-policy lambda.
- Estimate `lambda_zero_A` on frozen held-out batches for each mechanism.
- Keep separate lambda states for `direct_epsilon`, `linear_no_bias`, `affine`,
  MLP, or GRU adversaries unless a future spec gives a specific reason to
  couple them.
- Use the same target signal only if the target is mechanism-independent, such
  as analytical damage or an analytical phenotype. The lambda needed to reach
  that target remains mechanism-specific.
- Demonstrate optimizer reliability for the intended finite-policy inner
  optimizer against a stronger frozen-audit reference where feasible.
- Persist feature provenance, feature scaling, target centering, time or phase
  inputs, delayed-feedback conventions, and live perturbed-rollout status.

### 4.2 Mechanism order

Recommended order:

1. `linear_no_bias`: tests the restricted feedback-like policy without a shared
   open-loop component.
2. `affine`: adds bias/open-loop capacity and must decompose bias energy from
   feedback gain energy.
3. MLP or recurrent adversary: only after the finite-policy diagnostics show
   which failure mode the nonlinear policy is meant to address.

Affine success does not rescue a failed no-bias linear row by itself. It may
show that a shared open-loop component is needed, or that the row has moved
away from a pure closed-loop feedback analogue.

### 4.3 Additional diagnostics for finite mechanisms

Finite rows need all direct-epsilon diagnostics plus:

- Feature mean/RMS and conditioning summaries.
- Live feature source and timing audit.
- Realized process-epsilon energy over the perturbed rollout.
- Policy parameter norms and energy-metric gradient norms.
- Bias energy, feedback energy, and cross-term reconciliation for affine rows.
- Raw policy-output norm versus selected/clipped epsilon norm.
- Per-target and per-time disturbance summaries.
- Closed-loop amplification indicators: whether small parameter or feature
  changes create much larger realized disturbances.
- Adam or intended-optimizer agreement with a frozen reference on selected
  energy, objective gain, guard binding, and zero/nonzero classification.

### 4.4 Finite pass/fail gates

A finite adaptive row should pass only if:

- The direct-epsilon adaptive procedure has already produced a stable
  diagnostic surface, or the finite row is explicitly infrastructure-only.
- Frozen audits show a nonempty active window for the mechanism.
- The intended optimizer is reliable enough in the launch-relevant active
  window.
- Training selects nonzero adversaries without relying on the guard.
- Damage stays recoverable and close to the shared target.
- Kinematic sidecars remain interpretable relative to the no-PGD H0 baseline
  and analytical beta reference.

The row should fail or bracket if:

- The finite policy is either inactive or guard-bound across the relevant
  training window.
- Affine bias dominates the disturbance energy and the row was intended as a
  closed-loop feedback analogue.
- Direct and finite mechanisms require incompatible damage or phenotype
  targets, implying the common target is poorly specified.
- The row cannot separate optimizer failure from true zero-adversary optimality.

## 5. Required diagnostics and run-spec fields

### 5.1 Required scalar diagnostics

Record these at every locked diagnostic cadence for adaptive rows:

- `lambda`
- `log_lambda_delta`
- `lambda_update_reason`
- `lambda_curv_source`
- `lambda_curv_value_or_summary`
- `lambda_zero_bracket`
- `J_clean`
- `J_adv`
- `Delta_J`
- `E_selected`
- `Delta_J_over_E` diagnostic, reported with small-energy caveats
- `lambda_times_E`
- `Delta_J_minus_lambda_E`
- zero-selected fraction
- zero-incumbent acceptance rate
- guard-bound fraction
- selected norm and selected norm/guard ratio when a guard exists
- raw policy-output norm for policy adversaries
- KKT or radial-margin diagnostic
- gradient norm at zero in the realized-energy metric
- optimizer step count and termination reason
- finite/nonfinite status for objective, gradients, HVPs, selected epsilon,
  policy raw outputs, and rollout state
- clean task metrics and adversarial task metrics
- kinematic sidecars: velocity RMSE, peak velocity, time-to-peak, endpoint
  error, terminal speed, and trajectory-shape distance

### 5.2 Required replay and provenance fields

Persist enough information for a future frozen replay or post-run audit:

- controller checkpoint or model identity before adversary selection
- deterministic batch descriptor
- target and trial metadata
- PRNG keys or subkeys used by adversary selection
- mechanism and feature configuration
- lambda, beta, target, guard, optimizer, and update-rule configuration
- zero proposal, final proposal, selected proposal, and objective components
- masks, reductions, Q/R/Qf, state basis, disturbance channel, and output
  transform
- fixed-lambda comparator identity
- nonfinite flags and rejected-proposal counts

Bulk arrays can live outside tracked result notes, but scalar summaries and
replay pointers should be durable.

### 5.3 Required run-spec fields

Before any launch-facing run spec, lock these fields:

- issue/tracking context and explicit no-launch/launch status
- method class: fixed-lambda, preplanned schedule, or adaptive curriculum
- adversary mechanism and family-specific lambda source
- beta meaning: fixed lambda multiplier, paired analytical damage target,
  fixed-victim training-pressure target, phenotype target, or another explicit
  role
- baseline comparator source of record
- controller/task contract, including state basis, hidden size, target support,
  Q/R/Qf, loss masks, and reductions
- inner optimizer and reference optimizer, if any
- `lambda_0`, update cadence, smoothing, clipping, freeze policy, and allowed
  lambda range
- target signal and target source
- guard category, guard source, guard failure threshold, and whether guard
  binding invalidates the pure-soft interpretation
- fixed-lambda comparator row or frozen replay comparator
- pass/fail tolerances for target tracking, guard binding, zero fraction,
  nonfinite events, and nominal task learning
- expected artifacts and post-run analysis plan

## 6. Explicit non-goals and unresolved questions

### Non-goals

- Do not set inherited cap, radius, or trust-region values as defaults.
- Do not treat historical finite rows as valid bounded finite-policy evidence
  when their live finite-policy epsilon was not constrained by the inherited
  cap/projection being discussed.
- Do not treat a frozen finite-policy lambda gate and a live training launch
  path as the same object.
- Do not use `2e60620` as launch authorization; it remains an optimizer
  benchmark reminder unless separately resolved.
- Do not change the baseline task/controller contract inside this complement.
- Do not use a clean/adversarial outer-loss mixture as a hidden baseline task
  change. If used, it is a stability curriculum and must be labeled as such.
- Do not claim analytical H-infinity equivalence from adaptive lambda alone.
- Do not define adversarial damage as clean H-infinity-beta cost minus clean
  extLQG/nominal cost; that is nominal conservatism.
- Do not call H-infinity beta under matched disturbance minus a nominal clean
  baseline pure damage; that is total robust-condition burden.
- Do not use cost/energy ratio as the sole default target for adaptive lambda.
  It is a diagnostic unless a separate, explicitly labeled method chooses it.

### Unresolved questions before any run spec

- What exact paired analytical run defines `D_beta_ref`, and does it match the
  training Q/R/Qf, masks, state basis, target support, nominal noise seeds, and
  reductions?
- Is a fixed-victim training-pressure target needed for monotone beta pressure,
  or is matched-pair analytical damage the right first target?
- What tolerance defines successful target tracking?
- Should adaptive lambda freeze after an early calibration window, continue
  throughout training, or be evaluated in both modes?
- What physical or simulator-validity basis, if any, can justify a process
  epsilon guard?
- If no physical guard exists, what numerical guard policy is acceptable, and
  what binding threshold fails the pure-soft interpretation?
- Should lambda states be separate per replicate, per mechanism, per substrate,
  or per checkpoint regime?
- Which fixed-lambda comparator is required for the first adaptive proof of
  concept?
- What frozen-audit optimizer evidence is enough before including
  `linear_no_bias` or `affine` in a training smoke?
- Should phenotype matching be a second-stage correction after damage matching,
  or a separate adaptive-curriculum family?
- Which issue should own the eventual adaptive-lambda proof-of-concept if it
  advances beyond review-only planning?
