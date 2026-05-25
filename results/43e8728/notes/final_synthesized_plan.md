# Final synthesized plan: cs2019-to-RNN game-equivalence programme

## Authority and framing

This issue (`43e8728`) is the central plan for the cs2019-to-RNN robustness
programme. It supersedes `35f64be` and `6f783fa` as plan drivers. The controlling
sources are the GPT-5.5 Pro review and `0b1f109`; older plans remain useful only
where they support this framing.

The first experiment is **game equivalence**, not behavioral reproduction. The
initial question is:

Can the rlrmp/feedbax training and evaluation stack reproduce the same
closed-loop input-output game solved by the finite-horizon H-infinity Riccati
target, under a fully specified plant, cost, horizon, observation structure,
delay treatment, disturbance channel, norm, and gamma convention?

Only after that is established should GRU behavior be interpreted. Delta-v alone
is not a success criterion. The signature is composite: nominal speed/trajectory,
feedback-response gain, induced gain or equivalent robustness metric,
perturbation-induced displacement, endpoint quality, cost decomposition, and
control-energy sanity.

## STOP/ASK gates

Future agents must stop, summarize the choice, ask the user for an explicit
decision, and suggest fresh strong-model review when the choice changes the
formal game or downstream interpretation.

- **Open-loop epsilon vs state-dependent H-infinity adversary.** Decide whether
  the first gate trains/evaluates (a) open-loop rollout epsilon only, (b) the
  Riccati-implied state-dependent adversary / closed-loop induced-gain object,
  or (c) both as separate arms. Default recommendation: run both and treat
  disagreement as a formal result, not a tuning nuisance.
- **Exact game card fields.** Stop if the state vector, observation structure,
  delay augmentation, `B_w`, coordinate scaling, epsilon norm, time-integration
  convention, gamma/gamma-star mapping, or gamma-to-training-budget mapping is
  not explicit. Do not fill gaps from current code defaults.
- **State-map feasibility.** Stop if feedbax cannot implement the chosen
  `B_w` exactly. If a 48D delay-augmented `B_w = I` is not representable,
  decide whether to expose/materialize that state, rebuild the Riccati target
  on a smaller state, or formally relabel the experiment as a different game.
- **Restricted-field family.** Stop before defining "restricted-field" as a
  single arm. Choose E1-like curl-field exposure, E2-like mixed perturbations,
  worst-case over restricted fields, or multiple labeled arms.
- **Pass/fail tolerances.** Set tolerances after the analytical target is
  materialized but before any GRU outcome is inspected. Include gain/policy
  match, trajectory match, cost match, induced-gain/held-out adversary match,
  Delta-v band, terminal error, and overshoot/control-energy fails.
- **Bridge transitions.** Stop before moving from cs_faithful to production,
  from single reach to center-out, or from hold-free movement to delayed-reach.
  Each bridge changes the interpretation and needs a stated criterion.

## Phase 0: Analytical game card

Create a compact, auditable game card before training. It should be one
reference object that every later training/evaluation comparison points back to.

Required contents:

- Discrete-time state vector, including eight-state/six-state variants if both
  are considered, delay augmentation, observation/information structure, and
  exact state ordering.
- Plant and task: cs2019-faithful point mass first; hold-free, movement-only,
  single canonical reach unless the user explicitly approves a different gate.
- Discretization: `dt`, horizon, update convention, and whether epsilon enters
  as `x[t+1] = A x[t] + B u[t] + B_w epsilon[t]` or with timestep scaling.
- Cost: running and terminal schedules, `Q_t`, `R_t`, control scaling,
  terminal weights, delay-distributed cost convention, and Delta-v baseline.
- Disturbance channel: exact `B_w`, affected coordinates, coordinate scaling,
  and whether lag buffers/integrator states are mutable in the simulator.
- Epsilon metric: rollout-integrated norm, timestep integration, coordinate
  weighting, and relation to the H-infinity gamma penalty.
- Gamma convention: `gamma_star`, chosen gamma values, and the mapping from
  gamma to any PGD epsilon budget.
- Analytical controls: LQR gains, H-infinity gains, nominal trajectories, cost
  time courses, induced gain, feedback-response metric, and Delta-v.
- Worst-case disturbance record: **both** the Riccati-implied state-feedback
  worst-case disturbance / closed-loop induced-gain characterization and any
  open-loop nominal-rollout epsilon characterization. This preserves the useful
  first-Claude point while keeping the GPT-5.5 Pro formal warning central.

If the eight-state-to-six-state simplification is used, demonstrate equivalence
on the H-infinity quantities of interest across multiple gamma values, not only
at one point.

## Phase 1: Formal adversary equivalence

Resolve or explicitly split the adversary object before training is trusted.

- Closed-loop object: compute the Riccati-implied state-dependent worst-case
  disturbance policy or a direct closed-loop induced-gain / DP equivalent.
- Open-loop object: optimize rollout-level epsilon trajectories under the exact
  training norm and budget, with independent restarts and step sweeps.
- Equivalence check: compare worst-case cost/loss, induced gain, nominal
  trajectory, Delta-v, feedback response, and cost components.

If open-loop and state-dependent objects disagree materially, stop. Options are:
make the training adversary state-dependent, relabel the first experiment as an
open-loop-surrogate game, or request a narrow external review focused on formal
equivalence.

## Phase 2: Feedbax state-map and regression gate

Before launching training, prove that feedbax implements the game card.

Required artifacts:

- A row-by-row map from epsilon coordinates to feedbax state leaves or buffers.
- A statement of whether delay buffers are explicit mutable state or implicit
  simulator history.
- A coordinate-scaling check between Riccati coordinates, feedbax leaves, and
  PGD projection coordinates.
- Basis-vector epsilon regression tests with zero control showing that the
  empirical update recovers the intended `B_w` row by row.
- Information-structure parity: the trained linear controller and GRU must see
  the same state/observation structure as the Riccati target.

Failure here blocks training. Changing the state representation requires a new
game card and new analytical target numbers.

## Phase 3: Linear same-game round trip

Train a time-varying linear controller only after Phases 0-2 pass.

Minimum gates:

- LQR warmup matches analytical LQR under the same plant, cost, state, horizon,
  observation, and baseline convention.
- Adversarial linear training matches the analytical H-infinity target under
  the selected adversary object.
- Held-out adversary audit uses fresh initializations, multiple PGD step counts,
  and multiple restarts against frozen controllers.
- Report per replicate; do not rely on group means.

Pass/fail is conjunctive:

- policy/gain match or explicitly justified equivalent representation;
- nominal trajectory match, including peak velocity, time-to-peak, terminal
  error, and full cost time course;
- induced-gain or closed-loop robustness match;
- held-out adversary loss close to the analytical/training reference;
- Delta-v within the predeclared absolute band around the analytical target;
- no endpoint, overshoot, or control-energy pathology.

If LQR warmup fails, stop: the basic LQ stack is not working. If the H-infinity
linear gate fails, stop: the training/evaluation pipeline is not yet the
analytical game. Do not proceed to GRU interpretation.

## Phase 4: GRU under the certified game

Launch the GRU arm only after the linear same-game gate passes. Hold plant,
task, cost, horizon, state/observation access, disturbance channel, adversary
object, budget, training schedule, evaluation suite, and reporting fixed; change
only the controller architecture.

Interpret outcomes as:

- **GRU couples speed and gain:** the broad-epsilon game itself is sufficient to
  induce the cs2019-like composite signature in a flexible recurrent controller.
- **GRU dissociates gain from speed:** the game can be solved by a tracker-like
  or otherwise decoupled policy unavailable to the linear regulator; this is a
  substantive architectural result if robustness is verified.
- **GRU partially couples:** report its location on the robustness-speed-gain
  frontier; do not force a binary label.
- **GRU shows neither gain nor speed:** treat as training/adversary failure, not
  a scientific null.

If dissociation occurs, prefer testing realistic constraints that might suppress
decoupling, such as nonlinear damping, two-link/muscle dynamics, or delay and
information restrictions, before forcing a human-like signature by architectural
fiat.

## Phase 5: Matched adversary-class transfer matrix

After the same-game certificate exists, test the scientific contrast as a
transfer matrix, not a single broad-vs-narrow comparison.

Training arms should be separately labeled formal games:

- broad additive epsilon H-infinity / minimax;
- E1-like restricted curl-field distribution, likely expected-cost or domain
  randomized unless worst-case restricted fields are explicitly the question;
- E2-like mixed perturbation distribution: step loads, curl fields, and
  orthogonal velocity-dependent fields;
- optional worst-case restricted-field arm, labeled as a different question;
- optional state-multiplicative Delta-A or force-channel arms if needed to
  locate what the adversary class changes.

Evaluate every trained controller on clean nominal reaches, standardized step
loads, curl fields, orthogonal velocity-dependent fields, broad epsilon induced
gain, state-multiplicative / Delta-A perturbations, and held-out adversary
searches. "Generalization" means measured transfer to perturbation families not
used in training.

## Phase 6: Bridges to production and interpretability

The cs_faithful gate certifies movement-epoch H-infinity behavior for a narrow
game. It does not directly certify the production delayed-reach task.

- **Bridge 1: plant.** Production damping/time constants, hold-free single
  reach, new game card and Riccati reference if tractable.
- **Bridge 2: targets.** Center-out directions, still hold-free, with
  per-direction or symmetry-justified references.
- **Bridge 3: delayed-reach.** Add hold/target-on/movement epochs. Epsilon must
  be movement-epoch-gated unless a new analytical target explicitly includes
  preparation perturbations.
- **Interpretability.** Start dynamical-systems / affine-decomposition work
  only after the movement-epoch signature survives the relevant bridges.

Each bridge is a consistency check against the prior step's signatures. A bridge
failure is a substantive result about which added task feature changes the
solution, not a reason to silently change the gate.

## Core child issue mapping

- `020a65b`: full-state epsilon adversary implementation; revise around the
  open-loop-vs-state-dependent gate and the final game card.
- `6ec6b19`: cost schedule; in Phase 0 the schedule is part of the game card,
  not an optional sweep. Later alpha/frontier sweeps are follow-up analyses.
- `daa48c8`: replicate-conditioned reporting and bimodality; mandatory for
  linear, GRU, and transfer-matrix reporting.
- `1ad3c16`: principled evaluation perturbation amplitude; applies to
  standardized step/curl/field probes.
- `63cec06`: deterministic/declarative analysis pipeline for repeatable
  multi-signature reports.
- `8fcb6c7`: decoupling decomposition; useful after a verified GRU outcome.
  Measure local feedback under standardized perturbations, not only clean
  nominal reaches.
- `cf56e1e`: capacity ladder after the GRU result, especially if dissociation
  or partial coupling appears.
- `f7b1b17`: constrained-regulator RNN; defer until decomposition shows a
  feedforward channel is load-bearing.
- `b41c940`: graph-spec dependency if new training entry points/artifacts need
  to survive Feedbax graph-template migration.

## Supplementary analyses

- `ac06736`: two-link/six-muscle plant; later test of whether biomechanical
  coupling suppresses GRU decoupling.
- `a5e1450`: cross-partial diagnostic; use if an adversarial training fix fails
  and the project needs to distinguish adversary reachability from absent
  behavioral signature.
- `31043a5`: adaptation-vs-robustness probes; relevant to restricted-field and
  trial-history claims, not the first static-policy gate.
- `a3edc0c`: BCS/DAI/PAI-ASF comparison; method taxonomy after the core
  adversary behavior is established.
- `6d62018`: motor-noise scaling; later production/delayed-reach calibration,
  unless motor noise is explicitly added to the game card.
- `65156e8`: variable reach length; generalization bridge after the canonical
  single-reach game is certified.
- `297260c`: Part 1 analyses; reuse existing perturbation/feedback tools where
  they fit the standardized evaluation suite.
- `0af472c`: Part 2 analyses; later SISU/hidden-state/adaptation tools for
  production and trial-history questions.

## Ops, hygiene, and dependencies

- `fdad09d`: training scripts must write `run.json`; prerequisite for any new
  matrix that may become evidence.
- `e75ddd7`: legacy asset cleanup; do not touch during this plan except through
  the dedicated issue.
- `2ef67ca`: legacy pre-migration runs; historical context only, not a layout
  model for new artifacts.
- `2092cb5`: legacy eval-script relocation; opportunistic unless a script is
  directly reused.
- `6d5c906`: author-name correction; fix before manuscript-facing text or
  polished figures.
- `76d3a8e`: RunPod operational lessons; canonical GPU runbook for long jobs.
- `216b368`: environment/image stability; relevant before large remote runs.
- `a8ed10f`: Modal integration; possible future compute path, not a blocker.
- `3bd407b`: RunPod path-patching hazard; known deploy risk when rsyncing local
  editable dependencies.
- `f7d40f1`: Docker Hub pull-rate failures; use validated templates or early
  boot-log/dashboard checks.
- `f350f58`: keep `runpodctl` path handling consistent with the runbook.

## Superseded and historical disposition

- `35f64be`: superseded as central plan driver by `43e8728`. Keep as historical
  v1 record; future work should reference this issue instead.
- `6f783fa`: closed/subsumed interim v2. Preserve useful state-map, game-card,
  and bridge material, but do not revive it as canonical.
- `753508c`: historical training-method analysis; background only.
- `84ee4ff`: historical worst-case-method issue; useful for CVaR/APT context,
  not the first game-equivalence experiment.
- `ce34c2c`: historical GRU-laziness framing; not the central explanation.
- `83fc5b5`: historical output-feedback gap; do not make output feedback the
  first gate unless the game card reopens that requirement.

After this plan is adopted, comment on still-open superseded issues with a short
disposition and point future agents to `43e8728`. Do not manually close issues
unless the user explicitly asks.

## Reporting discipline

- Report frontiers over gamma or epsilon budget, not only one chosen point.
- Report the composite signature, not Delta-v alone.
- Report the full transfer matrix for adversary-class comparisons.
- Lock pass/fail tolerances before GRU outputs are inspected.
- Keep claims narrow: "broad-epsilon training is sufficient to induce this
  signature under this plant/cost/controller" is allowed after evidence;
  "humans optimized over full-state epsilon" is not.
