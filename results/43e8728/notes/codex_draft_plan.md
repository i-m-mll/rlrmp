# Draft Plan: cs2019-to-RNN Game-Equivalence Programme

## Framing

Issue `43e8728` should replace `35f64be` / `6f783fa` as the central plan for
the cs2019-to-RNN robustness programme. The first experiment is not "train a GRU
and see whether Delta-v is positive." It is a same-game equivalence programme:
can the rlrmp/feedbax training and evaluation stack reproduce the same
closed-loop input-output game solved by the finite-horizon H-infinity Riccati
target?

The controlling sources are `0b1f109` and the GPT-5.5 Pro review. Older plans
are subordinate source material. Carry forward their useful gates and issue
mapping, but do not preserve their implicit assumption that an optimized
open-loop epsilon trajectory is automatically equivalent to the state-dependent
H-infinity adversary / closed-loop induced-gain game.

The scientific claim should stay narrow:

- The model-matched arm asks whether broad additive-epsilon H-infinity training
  is sufficient to induce the composite cs2019-like signature in trained
  controllers.
- The restricted-field arms ask whether narrower human-protocol-like
  distributions are sufficient under otherwise matched conditions.
- The plan does not claim humans explicitly solve H-infinity. It tests whether a
  broad-defense policy/game is a better explanation than narrow perturbation
  exposure alone.

## Phase 0: Game Card Before Training

Before implementing or launching training, produce a compact analytical "game
card" under `results/43e8728/` (or a child issue path chosen by the final plan).
This artifact is the contract for all later comparisons.

Required contents:

- State vector: physical state, disturbance/integrator states, delay
  augmentation, observation/information structure, and the exact state ordering.
- Discretization: `dt`, horizon, update convention, whether epsilon enters as
  `x[t+1] = A x[t] + B u[t] + B_w epsilon[t]` or with `dt` scaling.
- Plant: cs2019-faithful point mass first; production plant only as a later
  bridge/reference.
- Costs: running and terminal schedules, `Q_t`, `R_t`, control scaling, terminal
  weights, delay-distributed cost convention, and Delta-v baseline.
- Disturbance channel: `B_w`, coordinates affected by epsilon, coordinate
  scaling, and whether lag buffers/integrator states are actually mutable in the
  simulator.
- Epsilon metric: rollout-integrated norm, timestep integration convention,
  coordinate weighting, and relation to the H-infinity `gamma` penalty.
- Gamma convention: `gamma_star`, chosen `gamma` (for example `1.5 *
  gamma_star` only if confirmed), and whether stronger robustness means smaller
  admissible gamma in the implementation.
- Analytical controls: LQR gains, H-infinity gains, nominal trajectories, cost
  time courses, induced gain, feedback-gain/step-response metric, and Delta-v.
- Worst-case disturbance characterization: either the Riccati-implied
  state-dependent disturbance policy or an equivalent closed-loop induced-gain
  computation, plus any open-loop worst-case trajectory derived for the nominal
  rollout.

Decision gate: if any of these fields cannot be stated exactly, stop and ask the
user whether to narrow the gate, revise the analytical target, or request fresh
strong-model review. Do not proceed by filling gaps with implementation
defaults.

## Phase 1: Formal Adversary Equivalence Gate

The open-loop trajectory-epsilon training surrogate and the state-dependent
H-infinity adversary are a load-bearing uncertainty. Split them unless a smart
review explicitly certifies equivalence for the exact finite-horizon setting.

Run or derive two checks:

1. Closed-loop H-infinity object: compute the Riccati-implied worst-case
   disturbance policy or directly compute the closed-loop induced gain for the
   candidate controller.
2. Open-loop epsilon object: optimize a rollout-level epsilon trajectory under
   the same norm/budget used by training, with independent restarts and step
   sweeps.

Pass condition: the two objects agree on the quantities that matter for the
game-card target: induced gain / worst-case cost, nominal trajectory, Delta-v,
feedback response, and relevant cost components.

If they disagree materially, stop. This is not a tuning problem. Ask the user to
choose among:

- make the training adversary state-dependent / dynamic-programming faithful;
- keep the open-loop adversary but relabel the first experiment as an
  open-loop-surrogate game, not cs2019 H-infinity equivalence;
- request another strong-model review focused only on the formal equivalence
  question.

## Phase 2: Simulator State-Map And Regression Gate

Before training, prove that feedbax can implement the `B_w` in the game card.
The old v2 state-map gate remains valid but should now live under this broader
game-equivalence umbrella.

Required artifacts:

- A row-by-row state map from each epsilon coordinate to feedbax state leaves or
  buffers.
- A statement of whether delay buffers are explicit mutable state or implicit
  history.
- A coordinate-scaling check between Riccati state coordinates, feedbax state
  leaves, and PGD projection coordinates.
- Regression tests using basis-vector epsilon injections with zero control,
  verifying that empirical `B_w` matches the game card.

Decision gate: if the delay-augmented `B_w = I` target cannot be represented
cleanly in feedbax, stop and ask whether to change the Riccati target (for
example to an 8-state/no-delay target) or invest in feedbax state exposure. A
changed target requires a new game card and new analytical Delta-v numbers.

## Phase 3: Linear Same-Game Round Trip

Train the linear controller only after Phases 0-2 pass.

Minimum requirements:

- LQR warmup recovers analytical LQR on the same plant, cost, horizon,
  observation structure, and state coordinates.
- Adversarial linear training recovers the analytical H-infinity solution under
  the selected adversary object.
- Held-out adversary audit uses fresh initializations, multiple PGD steps, and
  restarts against frozen controllers.
- Report per replicate; do not rely on group means.

The pass criterion must be conjunctive, not sign-only:

- gain or equivalent policy match;
- nominal trajectory match;
- cost time-course match;
- induced-gain / robustness match;
- Delta-v within a predeclared absolute band around the analytical target;
- no endpoint, overshoot, or control-energy pathology.

Failure modes and action:

- LQR warmup fails: stop; the basic training stack is not solving the LQ game.
- H-infinity linear gate fails and held-out adversary is stronger than training
  adversary: strengthen or redesign the inner maximization before interpretation.
- H-infinity linear gate fails even with the closed-loop adversary object
  matched: stop; the rlrmp/feedbax implementation is not the analytical game.
- Delta-v positive but robustness/gain fails: fail the gate; this is likely
  urgency, cost misspecification, or optimization pathology.

## Phase 4: GRU Same-Game Test

Only launch the GRU arm after the linear same-game gate passes. Hold plant, task,
cost, disturbance channel, budget, observation access, training schedule,
evaluation suite, and reporting fixed; change only controller architecture.

Interpret outcomes as follows:

- GRU couples speed and gain: the broad-epsilon game itself is sufficient to
  induce the cs2019-like signature in a flexible recurrent controller.
- GRU dissociates gain from speed: the game can be solved by a tracker-like or
  otherwise decoupled policy unavailable to the linear regulator; this is a
  scientific finding, not an implementation failure if robustness is verified.
- GRU partially couples: report the robustness-speed-gain frontier; do not force
  the result into a binary coupled/decoupled label.
- GRU shows neither gain nor speed: treat as training/adversary failure, not a
  scientific null.

If a significant unresolved choice becomes relevant during Phase 4 (for example
baseline choice for Delta-v, perturbation family for feedback gain, or whether
to add an intermediate architecture), stop and ask the user before continuing.
Suggest fresh strong-model review when the choice changes the interpretation of
the experiment rather than merely the implementation.

## Phase 5: Transfer Matrix Against Restricted Perturbation Games

After the same-game certificate exists, run the central scientific contrast as a
transfer matrix rather than a single broad-vs-narrow comparison.

Training arms should be separate formal games:

- broad additive epsilon H-infinity / minimax;
- Experiment-1-like restricted curl-field distribution, probably expected-cost
  or domain-randomized unless worst-case restricted-field control is explicitly
  the question;
- Experiment-2-like mixed perturbation distribution: step loads, curl fields,
  and orthogonal velocity-dependent fields;
- optional worst-case restricted-field arm, labeled separately from
  human-protocol-like expected training.

Evaluate every trained controller on:

- clean nominal reach;
- standardized step-load feedback response;
- curl fields;
- orthogonal velocity-dependent fields;
- broad epsilon induced gain;
- state-multiplicative / Delta-A perturbations;
- held-out adversary searches.

Use "generalization" only for measured transfer to perturbation families not
used in training.

## Phase 6: Bridge To Production And Interpretability

The cs2019-faithful gate is movement-only and narrow by design. It certifies a
training/evaluation pipeline and reference signatures; it does not directly
certify the production delayed-reach task.

Bridge in separate steps:

- Plant bridge: production damping/time constants, hold-free single reach, new
  game card and Riccati reference if tractable.
- Target bridge: center-out directions, still hold-free, with per-direction or
  symmetry-justified references.
- Delayed-reach bridge: add hold/target-on/movement epochs; epsilon must be
  movement-epoch-gated unless a new analytical target includes preparation
  perturbations.
- Interpretability: only after the movement-epoch signature survives the bridge.

If GRU dissociation is the main outcome, prefer testing realistic constraints
that might suppress decoupling (nonlinear damping, two-link/muscle dynamics,
delay/information restrictions) before forcing a human-like signature by
architectural fiat.

## Subordinate Issue Mapping

- `020a65b`: child for full-state epsilon adversary implementation, but its spec
  must be revised to respect the Phase 1 open-loop-vs-feedback adversary gate.
- `6ec6b19`: child for cost-schedule effects; in the first gate the schedule is
  part of the game card, not an optional sweep. Later alpha/frontier sweeps are
  useful after the same-game target exists.
- `daa48c8`: child for replicate-conditioned reporting and bimodality handling;
  required for all trained-controller reports.
- `1ad3c16`: child for principled evaluation perturbation amplitude; apply to
  standardized step/curl/field probes in the transfer matrix.
- `63cec06`: child for deterministic/declarative analysis pipeline; should
  provide repeatable multi-signature reports for matrix experiments.
- `8fcb6c7`: child for decoupling decomposition, but only after a verified GRU
  outcome exists; measure local feedback under standardized perturbations, not
  only clean nominal trajectories.
- `cf56e1e`: child for capacity ladder after the GRU result; useful if
  dissociation or partial coupling appears.
- `f7b1b17`: child for constrained-regulator RNN, deferred until decomposition
  shows a feedforward channel is actually load-bearing.
- `b41c940`: dependency/child for graph-spec migration if new training entry
  points or artifacts need to survive Feedbax's graph-template transition.

## Supplementary Analyses

- `ac06736`: two-link/six-muscle plant is a later test of whether biomechanical
  coupling suppresses GRU decoupling. Do not start before point-mass same-game
  and GRU outcomes are known.
- `a5e1450`: cross-partial diagnostic can help distinguish adversary-gradient
  reachability from absence of a behavioral signature. Run when a candidate
  adversarial training fix unexpectedly fails.
- `31043a5`: adaptation-vs-robustness probes become important for
  human-protocol-like restricted-field arms and trial-history claims.
- `a3edc0c`: BCS/DAI/PAI-ASF comparison is supplementary; useful for method
  taxonomy, not a central driver of the first game-equivalence experiment.
- `6d62018`: motor-noise scaling may matter for speed-accuracy tradeoffs in
  later production/delayed-reach training, but should not perturb the first
  analytical gate unless motor noise is in the game card.
- `65156e8`: variable reach length is a generalization test after the
  single-reach game is certified.
- `297260c`: Part 1 analyses provide existing feedback/plant perturbation tools
  that can be reused for standardized evaluation.
- `0af472c`: Part 2 analyses provide SISU/hidden-state tools for later
  production and adaptation-vs-robustness questions.

## Ops, Hygiene, And Dependencies

- `fdad09d`: training scripts must write `run.json`; this is a prerequisite for
  any new matrix whose results might later become evidence.
- `e75ddd7`: repository history cleanup is not a blocker, but large legacy
  assets should not be touched during this plan except by the dedicated issue.
- `2ef67ca`: legacy pre-migration runs are historical context only; avoid using
  their layout as a model for new artifacts.
- `2092cb5`: legacy eval-script relocation is hygiene; do not block the
  same-game gate on it unless a script is directly reused.
- `6d5c906`: fix the cs2019 author-name misattribution before manuscript-facing
  text or polished figures are produced.
- `76d3a8e`: RunPod runbook is the canonical operational reference for GPU
  training. Use its current SSH, CUDA/JAX, parallelism, and cost-discipline
  lessons.
- `216b368`: persistent environment or image stability matters for long training
  runs; prefer reproducible setup over manual pod mutation.
- `a8ed10f`: Modal integration is a possible future compute path, not a blocker
  for the first same-game experiment.
- `3bd407b`: path-patching runbook bug should be treated as a known deployment
  hazard when rsyncing local editable dependencies.
- `f7d40f1`: Docker Hub pull-rate failures can mimic slow pod boot; prefer
  validated templates or early boot-log/dashboard checks.
- `f350f58`: keep `runpodctl` in `~/.local/bin`, not tracked dotfiles paths.

## Superseded And Historical Issue Dispositions

- `35f64be`: superseded as central plan driver by `43e8728`. Keep open only if
  project policy requires historical records to remain open; comment after final
  plan adoption that future work should reference `43e8728`, not `35f64be`.
- `6f783fa`: already closed and superseded. Treat as useful v2 source material,
  especially the game-card and state-map gates, but do not revive it as the
  implementation plan.
- `753508c`: historical training-method analysis. Keep as background; central
  claims about BCS/DAI/PAI-ASF should not override the broad-epsilon
  game-equivalence framing.
- `84ee4ff`: historical/wider worst-case-training methods issue. Keep for CVaR
  and APT context, but the first experiment is a specified game-equivalence
  test, not a generic worst-case-method comparison.
- `ce34c2c`: historical GRU-laziness framing. Keep as a possible explanatory
  thread, but later evidence already weakened GRU-specific laziness as the
  central cause.
- `83fc5b5`: historical/output-feedback gap; demoted by later recipe-bug audit.
  Do not make Kalman/output-feedback the first gate unless the new game card
  reopens that formal requirement.

After the final `43e8728` plan is adopted, comment on still-open superseded
issues with their disposition and point future agents to the new plan. Do not
close issues manually unless the user explicitly asks or the protected-branch
merge/auth request is intended to close them.

## Stop Points For Future Agents

Stop and ask the user before proceeding if:

- open-loop epsilon and closed-loop H-infinity adversary checks disagree;
- feedbax cannot implement the game-card `B_w` exactly;
- the game card changes plant, cost, state, delay, or observation structure;
- the linear same-game gate fails;
- the GRU result suggests a new scientific interpretation rather than an
  implementation fix;
- a proposed shortcut would collapse broad-epsilon, restricted-field, and
  state-multiplicative perturbations into one informal "robustness" label.

At these points, the implementing agent should remind the user what decision is
being made, state the consequences for interpretation, and suggest fresh
strong-model review when the decision changes the formal game.
