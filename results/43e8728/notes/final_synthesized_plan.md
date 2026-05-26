# Structured plan: cs2019-to-RNN game-equivalence programme

## Abstract

This is the central plan for the cs2019-to-RNN robustness programme. It is
controlled by the GPT-5.5 Pro review and the immediately preceding synthesis on
`0b1f109`. Older plans such as `35f64be`, `6f783fa`, and `f695729` are
historical inputs only where they support this framing.

The first scientific task is not to reproduce human behavior directly. The
first task is to prove that the rlrmp/feedbax training and evaluation stack can
instantiate the same closed-loop input-output game solved by the finite-horizon
H-infinity Riccati target. That game must be specified by plant, cost, horizon,
state and observation structure, delay treatment, disturbance channel, norm,
and gamma convention before GRU results are interpreted.

The project then asks whether training a flexible recurrent controller against
that certified game induces a cs2019-like composite robustness signature. The
signature is not Delta-v alone. It includes nominal speed/trajectory,
feedback-response gain, induced gain or an equivalent robustness metric,
perturbation-induced displacement, endpoint quality, cost decomposition, and
control-energy sanity.

## Plan at a glance

| Phase | Main question | Main output | Blocking issues |
|---|---|---|---|
| 0. Analytical game card | What exact game is the Riccati target solving? | One auditable game card and analytical reference bundle | `6ec6b19`, `020a65b`, `1ad3c16` |
| 1. Adversary equivalence | Is the training adversary the same object as the H-infinity adversary? | Decision on open-loop epsilon, state-dependent adversary, or both | `020a65b`, `1ad3c16` |
| 2. Feedbax implementation parity | Does feedbax implement the game card exactly? | State-map and `B_w` regression tests | `020a65b`, `b41c940` |
| 3. Linear same-game round trip | Can learned linear controllers reproduce LQR and H-infinity references? | Linear pass/fail certificate | `6ec6b19`, `daa48c8`, `63cec06` |
| 4. GRU same-game test | What changes when only controller architecture becomes recurrent/flexible? | GRU outcome classification under the certified game | `daa48c8`, `63cec06`, `8fcb6c7`, `cf56e1e` |
| 5. Perturbation-family transfer matrix | Does broad adversarial training transfer to restricted physical perturbations? | Matched broad-vs-restricted evaluation matrix | `1ad3c16`, `020a65b`, `63cec06` |
| 6. Bridges to production task | Which added task features preserve or change the signature? | Bridge reports for plant, targets, delayed reach, and interpretation | `b6084c7`, `b41c940`, `f7b1b17`, `8fcb6c7` |

The intended order is sequential through Phase 3. Later phases can be planned in
parallel, but their scientific interpretation depends on earlier certificates.

## Stop-and-ask gates

Future agents must stop, summarize the choice, ask the user for an explicit
decision, and suggest fresh strong-model review when a choice changes the
formal game or downstream interpretation.

1. **Open-loop epsilon vs state-dependent H-infinity adversary.** Decide whether
   the first gate trains/evaluates open-loop rollout epsilon only, the
   Riccati-implied state-dependent adversary / closed-loop induced-gain object,
   or both as separate arms. The default recommendation is to run both and
   treat disagreement as a formal result.
2. **Exact game-card fields.** Stop if the state vector, observation structure,
   delay augmentation, `B_w`, coordinate scaling, epsilon norm,
   time-integration convention, gamma mapping, or gamma-to-training-budget
   mapping is implicit.
3. **State-map feasibility.** Stop if feedbax cannot implement the chosen `B_w`
   exactly. Decide whether to expose/materialize the needed state, rebuild the
   Riccati target on a smaller state, or relabel the experiment as a different
   game.
4. **Restricted-field family.** Stop before defining restricted fields as one
   undifferentiated arm. Ask whether the reviewer/user wants an E1-like
   curl-field exposure, E2-like mixed perturbations, worst-case restricted
   fields, or multiple labeled arms.
5. **Pass/fail tolerances.** Set tolerances after the analytical target is
   materialized but before inspecting GRU outcomes.
6. **Bridge transitions.** Stop before moving from cs_faithful to production,
   single reach to center-out, or hold-free movement to delayed reach.

## Phase 0: Analytical game card

**Purpose.** Create one compact reference object that every later training and
evaluation comparison points back to.

**Experiment/task.** Materialize the finite-horizon LQR and H-infinity reference
for a cs2019-faithful point-mass game, preferably hold-free, movement-only, and
single-reach unless the user explicitly approves a broader first gate.

**Required decisions and contents.**

- Discrete-time state vector, including eight-state/six-state variants if both
  are considered, delay augmentation, observation/information structure, and
  exact state ordering.
- Plant and task regime.
- `dt`, horizon, update convention, and whether epsilon enters as
  `x[t+1] = A x[t] + B u[t] + B_w epsilon[t]` or with timestep scaling.
- Running and terminal schedules: `Q_t`, `R_t`, control scaling, terminal
  weights, delay-distributed cost convention, and Delta-v baseline.
- Exact disturbance channel `B_w`, affected coordinates, coordinate scaling,
  and whether lag buffers/integrator states are mutable in the simulator.
- Epsilon metric, timestep integration, coordinate weighting, and relation to
  the H-infinity gamma penalty.
- `gamma_star`, chosen gamma values, and mapping from gamma to any PGD epsilon
  budget.
- Analytical LQR and H-infinity gains, nominal trajectories, cost time courses,
  induced gain, feedback-response metric, and Delta-v.
- Both the Riccati-implied state-feedback worst-case disturbance or
  closed-loop induced-gain characterization and any open-loop nominal-rollout
  epsilon characterization.

**Associated issues.**

- `6ec6b19`: cost schedule. In this phase, the schedule is part of the game
  definition, not an optional sweep.
- `020a65b`: full-state epsilon adversary implementation. The game card defines
  the intended `B_w` and adversary object.
- `1ad3c16`: perturbation amplitude. The gamma/epsilon mapping must be fixed
  before training/evaluation claims are made.

**Exit criterion.** A future reader can reconstruct the exact mathematical game
and analytical reference numbers without reading current code defaults.

## Phase 1: Formal adversary equivalence

**Purpose.** Resolve whether the trainable adversary is actually the same
formal object as the H-infinity adversary.

**Experiment/task.**

- Compute or characterize the closed-loop object: Riccati-implied
  state-dependent worst-case disturbance policy, closed-loop induced gain, or a
  dynamic-programming equivalent.
- Optimize the open-loop object: rollout-level epsilon trajectories under the
  exact training norm and budget, with independent restarts and step sweeps.
- Compare worst-case cost/loss, induced gain, nominal trajectory, Delta-v,
  feedback response, and cost components.

**Decision point.** If open-loop and state-dependent objects disagree
materially, stop. The options are to make the training adversary
state-dependent, relabel the first experiment as an open-loop-surrogate game, or
ask for a narrow external formal-equivalence review.

**Associated issues.**

- `020a65b`: likely receives the implementation revision.
- `1ad3c16`: sets the amplitude/budget interpretation used in the comparison.

**Exit criterion.** The plan has a named adversary object for the first gate,
and any surrogate relationship is stated rather than assumed.

## Phase 2: Feedbax implementation parity

**Purpose.** Prove that feedbax/rlrmp implements the Phase 0 game card rather
than a nearby but different game.

**Experiment/task.**

- Build a row-by-row map from epsilon coordinates to feedbax state leaves or
  buffers.
- State whether delay buffers are explicit mutable state or implicit simulator
  history.
- Check coordinate scaling between Riccati coordinates, feedbax leaves, and PGD
  projection coordinates.
- Add basis-vector epsilon regression tests with zero control showing that the
  empirical update recovers the intended `B_w` row by row.
- Confirm information-structure parity: trained linear controllers and GRUs must
  see the same state/observation structure as the Riccati target, unless a
  deliberately different observation experiment is declared.

**Associated issues.**

- `020a65b`: adversary wiring and regression tests.
- `b41c940`: GraphSpec/current feedbax migration if it is needed for clean,
  explicit state and perturbation specification.

**Exit criterion.** The simulator-level regression test suite demonstrates that
the implemented perturbation channel and state representation match the game
card. Failure blocks training.

## Phase 3: Linear same-game round trip

**Purpose.** Before interpreting GRUs, show that the learning/evaluation stack
can recover the analytical linear solutions in the exact same game.

**Experiment/task.**

- Train a time-varying linear controller under the Phase 0-2 certified game.
- Verify LQR warmup against analytical LQR under the same plant, cost, state,
  horizon, observation, and baseline convention.
- Verify adversarial linear training against the analytical H-infinity target
  under the selected adversary object.
- Audit frozen controllers with held-out adversary search using fresh
  initializations, multiple PGD step counts, and multiple restarts.
- Report per replicate, not only group means.

**Pass/fail criteria.**

- Policy/gain match, or an explicitly justified equivalent representation.
- Nominal trajectory match, including peak velocity, time-to-peak, terminal
  error, and full cost time course.
- Induced-gain or closed-loop robustness match.
- Held-out adversary loss close to the analytical/training reference.
- Delta-v within the predeclared absolute band around the analytical target.
- No endpoint, overshoot, or control-energy pathology.

**Associated issues.**

- `6ec6b19`: cost schedule must be frozen for the gate.
- `daa48c8`: replicate-conditioned reporting and bimodality handling.
- `63cec06`: analysis pipeline support for composite signature reporting.
- `1ad3c16`: standardized adversary amplitude/budget used for held-out audits.

**Exit criterion.** Linear controllers pass the same-game certificate. If LQR
warmup fails, the basic LQ stack is not working. If the H-infinity linear gate
fails, the training/evaluation pipeline is not yet the analytical game.

## Phase 4: GRU same-game test

**Purpose.** Test what changes when only the controller architecture becomes a
flexible recurrent controller.

**Experiment/task.**

- Hold plant, task, cost, horizon, state/observation access, disturbance
  channel, adversary object, budget, training schedule, evaluation suite, and
  reporting fixed from the certified linear game.
- Change only the controller architecture to GRU.
- Evaluate the same composite signature used in Phase 3.

**Outcome classification.**

- **GRU couples speed and gain.** The broad-epsilon game itself is sufficient to
  induce a cs2019-like composite signature in a flexible recurrent controller.
- **GRU dissociates gain from speed.** The game can be solved by a tracker-like
  or otherwise decoupled policy unavailable to the linear regulator; this is a
  substantive architectural result if robustness is verified.
- **GRU partially couples.** Report its location on the
  robustness-speed-gain frontier; do not force a binary label.
- **GRU shows neither gain nor speed.** Treat as training/adversary failure
  first, not as a scientific null.

**Associated issues.**

- `daa48c8`: replicate-conditioned reporting.
- `63cec06`: composite analysis pipeline.
- `8fcb6c7`: decoupling/decomposition analysis if the GRU dissociates gain and
  speed.
- `cf56e1e`: architecture ladder only after the same-game test is meaningful.

**Exit criterion.** The GRU result is classified under a fixed certified game,
with no silent changes to cost, adversary, perturbation budget, observation, or
evaluation.

## Phase 5: Perturbation-family transfer matrix

**Purpose.** Test the scientific contrast: whether broad adversarial training
transfers to restricted physical perturbations, and whether restricted physical
training transfers back to broader adversaries.

**Experiment/task.** Train and evaluate a matrix of formally labeled games.

Training arms:

- Broad additive epsilon H-infinity/minimax.
- E1-like restricted curl-field distribution, if selected by the reviewer/user.
- E2-like mixed perturbation distribution: step loads, curl fields, and
  orthogonal velocity-dependent fields, if selected by the reviewer/user.
- Optional worst-case restricted-field arm, explicitly labeled as a different
  question.
- Optional state-multiplicative `Delta-A` or force-channel arms if needed to
  locate what the adversary class changes.

Evaluation columns:

- Clean nominal reaches.
- Standardized step loads.
- Curl fields.
- Orthogonal velocity-dependent fields.
- Broad epsilon induced gain or equivalent.
- State-multiplicative / `Delta-A` perturbations.
- Held-out adversary searches.

**Interpretation rule.** "Generalization" means measured transfer to
perturbation families not used in training. Do not frame a single broad-vs-curl
comparison as the whole scientific claim.

**Associated issues.**

- `1ad3c16`: amplitude/budget standardization.
- `020a65b`: adversary family implementation.
- `63cec06`: transfer-matrix analysis/reporting.

**Exit criterion.** The comparison separates training family, evaluation family,
and perturbation budget so that any robustness transfer claim is interpretable.

## Phase 6: Bridges to production task and interpretation

**Purpose.** Move from the narrow cs_faithful gate to the production delayed
reach task without losing track of which added feature changes the solution.

**Bridge experiments.**

1. **Plant bridge.** Add production damping/time constants while keeping a
   hold-free single reach. Create a new game card and Riccati reference if
   tractable.
2. **Target bridge.** Add center-out directions while staying hold-free. Use
   per-direction references or a symmetry-justified reference.
3. **Delayed-reach bridge.** Add hold/target-on/movement epochs. Epsilon should
   be movement-epoch-gated unless a new analytical target explicitly includes
   preparation perturbations.
4. **Interpretability bridge.** Start dynamical-systems and affine-decomposition
   work only after the movement-epoch signature survives the relevant bridges.

**Associated issues.**

- `b6084c7`: comparability hygiene for production-vs-cs_faithful interpretation.
- `b41c940`: feedbax/GraphSpec migration if needed for clean bridge
  specification.
- `f7b1b17`: constrained-regulator RNN, later than the certified game and
  unconstrained GRU.
- `8fcb6c7`: decomposition/decoupling analysis after standardized perturbation
  gates exist.

**Exit criterion.** Each bridge either preserves the prior signature or
identifies the added task feature that changes it. Bridge failure is a
scientific result, not a reason to silently change the gate.

## Core issue map

- `020a65b`: full-state epsilon adversary; central for Phases 0-2 and later
  adversary-family arms.
- `6ec6b19`: cost schedule; first used to define the analytical game, later for
  frontier/sensitivity sweeps.
- `daa48c8`: replicate-conditioned reporting; mandatory once training runs
  begin.
- `1ad3c16`: perturbation amplitude/budget; central at adversary-equivalence and
  transfer-matrix points.
- `63cec06`: analysis pipeline; should report the composite signature and later
  transfer matrix.
- `8fcb6c7`: decoupling/decomposition; important after GRU or bridge results
  show dissociation or unexpected robustness.
- `cf56e1e`: architecture ladder; not a first blocker, but relevant after the
  certified GRU test.
- `f7b1b17`: constrained-regulator RNN; later, not before the certified game and
  unconstrained GRU.
- `b41c940`: GraphSpec/feedbax migration; infrastructure dependency when current
  feedbax state/perturbation APIs block precise game specification.
- `b6084c7`: comparability hygiene; relevant for production bridges, not the
  decisive first gate.

## Supplementary analyses and background issues

These issues are not the central skeleton, but they may become relevant once
the corresponding phase is active.

- `ac06736`: supplementary training/methodology analysis; revisit if Phase 4 or
  Phase 5 outcomes need additional method variants.
- `a5e1450`: supplementary analysis/statistics issue; keep available for
  deeper interpretation after the core composite signature is measurable.
- `31043a5`: supplementary methodology issue; review when expanding beyond the
  first certified game.
- `a3edc0c`: supplementary methodology issue; likely relevant only after the
  basic game-equivalence gate is stable.
- `6d62018`: supplementary methodology issue; not a first gate blocker.
- `65156e8`: supplementary methodology issue; defer until the phase it informs
  becomes active.
- `297260c`: prior Part 1 analysis umbrella/context; historical background for
  analysis choices, not controlling.
- `0af472c`: prior Part 2 analysis umbrella/context; historical background for
  analysis choices, not controlling.
- `bf71d86`: go-cue timing/stratification concern; important for delayed-reach
  interpretation, but not a first same-game blocker.

## Ops, hygiene, and artifact issues

These support the plan but are not scientific phases.

- `fdad09d`: artifact/repo hygiene.
- `e75ddd7`: artifact/repo hygiene.
- `2ef67ca`: legacy result/archive hygiene.
- `2092cb5`: artifact/repo hygiene.
- `6d5c906`: artifact/repo hygiene.
- `76d3a8e`: RunPod/ops support.
- `216b368`: RunPod/ops support.
- `a8ed10f`: RunPod/ops support.
- `3bd407b`: RunPod/ops support.
- `f7d40f1`: RunPod/ops support.
- `f350f58`: RunPod/ops support.

Use these when launching or preserving runs, but do not let them define the
scientific order of work.

## Superseded or historical issues

- `35f64be`: v1 plan. Superseded; do not use as canonical.
- `6f783fa`: v2 interim plan. Useful for the game-card and bridge-chain ideas,
  but superseded by `43e8728`.
- `f695729`: older broad phase skeleton. Superseded as active phase skeleton.
- `753508c`: historical BCS/DAI/PAI-ASF context.
- `84ee4ff`: historical APT/CVaR context.
- `ce34c2c`: historical GRU-laziness framing; not the central explanation.
- `83fc5b5`: output-feedback/Kalman gap; possible later bridge issue, not a
  first gate.

## Reporting discipline

Every substantive run or analysis should report:

- the phase it belongs to;
- the game card or bridge card it uses;
- the exact associated issue IDs;
- whether a stop-and-ask gate was encountered;
- replicate-level results, not only aggregate means;
- the composite signature, not only Delta-v;
- whether the result advances, blocks, or changes the next phase.
