# External Review Packet: cs2019-to-RNN Robustness Plan

Date: 2026-05-25

Primary issue: `0b1f109` - "Synthesis: bridging cs2019 induction to RNN training methods"

Companion materials expected for review:

- `/Users/mll/Documents/Claude/Projects/clench/synthesis-5.md`
- `/Users/mll/Main/10 Projects/10 PhD/cs2019.pdf`
- Optional source-code access to `rlrmp` and `feedbax`

This packet summarizes the current rlrmp plan that grew out of issue `0b1f109`.
It is meant to support an external conceptual, formal, and methodological review.
It is not intended to absorb the separate `synthesis-5.md` backdrop, and it is not
an implementation ticket. The goal is to represent the plan as it currently stands
and make its assumptions, intended comparisons, and possible weak points explicit.

## Executive Summary

rlrmp is trying to connect a human motor-control finding from cs2019 to trained
recurrent neural controllers. In cs2019, humans exposed to unpredictable force
fields show a coupled robustness signature: increased movement speed and increased
feedback gain. The paper's analytical model is an H-infinity-style robust controller
that predicts this signature under a broad additive disturbance game.

rlrmp has trained RNN controllers under several adversarial or robustness-oriented
objectives. The current concern is that the existing trained adversarial condition,
especially the `LinearDynamicsAdversary` / Delta-A flavor-B setup, may not be the
same game as cs2019's analytical model and may therefore induce a different
behavioral signature. In particular, state-multiplicative disturbances can reward
keeping state excursions small, plausibly producing slower movement rather than the
velocity increase seen in cs2019.

The current plan is:

1. Treat `0b1f109` as the canonical synthesis and brainstorming home.
2. First certify the broad epsilon H-infinity training game against a known
   analytical target, using a linear regulator round-trip gate.
3. Only after the linear gate succeeds, train a GRU under the same broad epsilon
   game and ask whether it couples speed and feedback gain, behaviorally
   dissociates them, partially couples them, or collapses.
4. Treat curl-field / distributional training as an essential contrast to broad
   epsilon training, but design that contrast cleanly: same plant, task, cost,
   architecture, training budget, and evaluation suite, with only the uncertainty
   class changed.
5. Bridge gradually from the cs2019-faithful gate to the production delayed-reach
   setting before doing internal-dynamics interpretability work.

The plan is not claiming that humans explicitly solve the H-infinity game. The more
cautious framing is that humans may behave as if they have a broad-defense prior or
policy, while a GRU trained only on narrow curl-field exposure may not acquire that
broad policy unless the broader adversary class is supplied explicitly.

## Review Posture

The questions in this packet are guiding questions, not a checklist. A stronger
review might reject the framing, identify a better central question, or propose a
cleaner experimental design. Please treat the plan generously enough to see its
best version, but critically enough to identify hidden confounds, formal mismatches,
or overinterpretations.

## Project Context

The broader project studies trained neural controllers for reaching tasks under
uncertainty. rlrmp contains analysis, training, and experiment artifacts. Feedbax is
the lower-level model / task / intervention infrastructure on which rlrmp depends.

The scientific target is not only "can an RNN reproduce a behavioral profile." The
project is trying to relate:

- analytical robust-control predictions;
- human behavior under uncertainty;
- trained neural-network controllers;
- adversary class and cost-schedule choices;
- later internal-dynamics / interpretability analyses.

The `synthesis-5.md` document is a separate theoretical and motivational backdrop
for the broader research program. It should be reviewed as context, not treated as
the source of the concrete experimental plan in this packet.

## The cs2019 Reference Point

The most important empirical and formal reference is cs2019. The high-level facts
used by the current plan are:

- Humans are physically exposed to curl-field-like disturbances on a subset of
  trials.
- The cs2019 behavioral signature includes both increased movement speed and
  increased feedback gain under uncertainty.
- The analytical model is a robust-control model that predicts this coupled
  speed/gain signature.
- The model's disturbance game is broader than a single narrow curl-field
  distribution. In the current rlrmp framing, the relevant game is a free
  time-varying additive epsilon disturbance entering the current physical state,
  with an H-infinity budget.

This creates a methodological tension:

- The human experimental perturbation class is narrow and physical.
- The analytical model's robust disturbance class is broader.
- The human behavior matches the broad robust model better than a literal narrow
  "train only on the exact curl distribution" story might suggest.

The current project framing calls this a possible broad-defense prior or
adversary-class broadening. It should not be taken as proven human
"generalization" unless the evaluation defines and tests generalization as an
observable transfer property.

## Current Core Reframe From `0b1f109`

The initial worry was that a learned fixed adversary might not really be
adversarial because the controller and adversary could co-adapt. The synthesis
reframed that worry.

The core issue is not "fixed adversary versus non-fixed adversary." In linear
quadratic and H-infinity games, optimal policies and worst-case strategies can also
converge to fixed structures. The load-bearing questions are instead:

1. What uncertainty channel is being robustified?
2. What cost schedule makes speed useful rather than wasteful?
3. Does the controller architecture force nominal speed and feedback gain to be
   coupled, or can it separate them?
4. Has the inner maximization actually found a strong worst case?

This is why the plan treats the linear broad-epsilon round trip as decisive. If a
linear controller trained through the rlrmp/feedbax training pipeline cannot match
the corresponding Riccati solution, then the pipeline is not implementing the same
formal game and GRU behavior under that pipeline is not interpretable as evidence
about the H-infinity hypothesis.

## Important Distinctions

### Adversary Classes

The plan distinguishes at least four disturbance or uncertainty classes:

- Narrow stochastic curl-field exposure, close to the physical perturbations in
  cs2019 human experiments.
- Force-channel perturbations such as Gaussian bumps or fixed force fields.
- State-multiplicative dynamics perturbations, such as a Delta-A / flavor-B
  adversary.
- Broad additive epsilon disturbances entering the physical state, used as the
  cs2019 analytical H-infinity reference game.

The current plan does not treat these as interchangeable versions of the same
thing. They are different formal training games and should be expected to produce
different behavioral signatures.

### Behavioral Dissociation Versus Structural Separability

The word "decoupling" has been overloaded. The current plan distinguishes:

- Behavioral dissociation: feedback-gain modulation is present but nominal speed
  inflation is absent or much smaller.
- Structural separability: the learned policy internally separates something like
  a feedforward nominal trajectory generator from a local feedback law.

Behavioral dissociation does not prove structural separability. A narrow curl-field
training objective could produce high feedback gain and flat nominal speed for many
reasons unrelated to an explicit feedforward/feedback decomposition. Structural
separability is a later interpretability question, not something established by
Delta v alone.

### Model-Matched Versus Human-Protocol-Matched

Broad epsilon training is model-matched to the cs2019 analytical H-infinity
controller. It is not the literal human-protocol-matched exposure.

Curl-field distributional training is closer to the human experimental perturbation
class. It is not expected to automatically reproduce the analytical H-infinity
signature unless the trained controller acquires a broader defense policy than the
training distribution directly requires.

The contrast between these two training regimes is scientifically meaningful only
if it is designed cleanly.

## Existing Empirical Motivation

A prior rlrmp adversarial training condition using `LinearDynamicsAdversary`
produced a negative or mixed Delta-v pattern rather than the cs2019-like speed
increase. The issue thread records a rough interpretation:

- A Delta-A / state-multiplicative adversary applies stronger effective disturbance
  when the state is large.
- A controller can reduce adversary leverage by moving more slowly or keeping
  state excursions smaller.
- This can plausibly flip the sign of speed modulation relative to the broad
  additive epsilon H-infinity game.

This does not make the Delta-A adversary "wrong." It means it is a different
training method with its own signature. The current plan deprioritizes more
flavor-B sweeps until the model-matched broad-epsilon game is validated.

The issue thread also notes that group means can be misleading: some trained
replicates may show different qualitative solutions. Replicate-conditioned
reporting is therefore part of the plan.

## Plan Overview

The current plan has three layers:

1. A same-game certification gate: prove that the training pipeline can reproduce
   the analytical broad-epsilon H-infinity solution.
2. A GRU architecture test: train a GRU under the same certified broad-epsilon game
   and observe whether it couples speed and gain.
3. A bridge-and-contrast program: compare broad-epsilon and curl-field-trained GRUs
   under matched conditions, then bridge toward the production delayed-reach task
   where internal-dynamics analysis is meaningful.

The gate is first because a curl-field baseline or GRU interpretation is much less
informative without knowing whether the broad-epsilon machinery can reproduce the
known analytical game.

## Phase 0: Tier-0 Audit of Existing Comparisons

Issue: `b6084c7`

Before launching new training, the plan calls for a cheap audit of existing
controllers and artifacts. The reason is that several numbers have been compared
across different plants, horizons, costs, adversary classes, delay treatments,
inputs, and training objectives. Those comparisons are not clean unless their axes
are made explicit.

The audit should compare, at minimum:

- a baseline warmup-only GRU;
- a flavor-B adversarially trained GRU;
- a trained linear regulator from prior work;
- the analytical Riccati controller.

For each, the audit should document:

- plant configuration;
- state dimension and delay handling;
- horizon and cost schedule;
- adversary class;
- task input channels;
- active training losses;
- catch-trial or perturbation probabilities;
- evaluation protocol.

This audit is not the main experiment. Its role is to prevent another round of
interpretation based on non-comparable Delta-v values.

## Phase 0.5: Analytical Target Materialization

The gate requires an explicit analytical target before training starts. The output
should be an artifact that states exactly what the trained linear arm is supposed
to match.

The current target is cs2019-faithful in spirit:

- point-mass plant;
- cs2019-style cost schedule;
- movement-only finite horizon;
- sensorimotor delay represented in the analytical Riccati state;
- broad additive epsilon disturbance entering the current physical state;
- gamma chosen relative to the critical gamma star.

The target materialization should include:

- gamma star;
- chosen gamma, e.g. 1.5 gamma star unless revised;
- LQR and H-infinity gain time series;
- nominal trajectories;
- peak velocity, time-to-peak, terminal error;
- worst-case epsilon trajectory or equivalent disturbance characterization;
- rollout-integrated disturbance norm used as the adversary budget;
- cost time courses;
- induced-gain values;
- the analytical Delta-v signature.

The latest issue comment adds an important practical decision. cs2019's eight
physical-state model includes two disturbance-mediator / integrator states. rlrmp's
audit suggests those integrator pathways are dynamically inert for the H-infinity
worst-case solution because direct velocity attack is more efficient. The current
plan is therefore:

1. Materialize the cs2019-faithful eight-state analytical target with those
   integrator states.
2. Re-run the Riccati on a six-state version without the integrators.
3. Demonstrate approximate equivalence for the relevant H-infinity quantities.
4. Use the simpler six-state feedbax implementation as the training target if the
   equivalence demonstration holds.

This is a substantive part of the plan because it simplifies the feedbax
intervention while preserving the cs2019 analytical anchor.

The plan also treats no-delay variants as supplementary analysis rather than the
primary gate. The primary gate keeps delay because delay is part of the cs2019
reference.

## Phase 1: Full-State Epsilon Adversary and State Injection

Issue: `020a65b`

The new adversary class is a free time-varying epsilon disturbance rather than a
fixed Delta-A matrix. The intended high-level properties are:

- epsilon is per-timestep and time-varying;
- the budget is rollout-integrated L2, not a per-step cap;
- the projection rescales the full flattened epsilon trajectory when it exceeds
  the allowed norm;
- the default budget is calibrated from the analytical Riccati target;
- the same core class can later support variants, but the gate uses the integrated
  L2 version only.

The feedbax-side intervention should be a discrete state update, not a force-port
perturbation. In the simplified target, it writes epsilon into the current physical
state coordinates. The regression test should confirm the intended state map:

- one epsilon basis vector changes exactly the corresponding physical state
  coordinate under the chosen discrete-time convention;
- the empirical linearization recovers the intended disturbance channel;
- delay propagation is verified separately when the analytical target includes
  delayed state.

The plan no longer treats SISU as required for the gate. The first round-trip can
be a single-purpose epsilon-only experiment. SISU becomes a later experimental
axis if one wants a single network conditioned on robustness level.

Possible review-relevant detail: the implementation plan has described epsilon as
an optimized per-timestep disturbance trajectory. A reviewer may want to examine
whether that is fully aligned with the H-infinity game being invoked, or whether
the intended training/evaluation protocol needs a clearer statement about
open-loop versus state-dependent worst-case disturbances.

## Phase 2: Round-Trip Training Entry Point

The plan calls for a dedicated `scripts/train_round_trip.py` entry point, rather
than overloading the production `train_minimax.py` CLI. The entry point should
reuse library training primitives, but construct the cs2019-faithful task directly.

The gate task is intentionally narrow:

- movement-only;
- single canonical reach;
- no hold epoch;
- no catch trials;
- no go-cue structure;
- no mixed target directions;
- same observation/information structure as the analytical target.

This narrowness is deliberate. The first question is not whether the production
delayed-reach task can be made robust. The first question is whether the training
pipeline can reproduce a known H-infinity game.

## Phase 3: Linear Regulator Same-Game Gate

The linear arm is the load-bearing gate. It should run before any GRU arm.

The intended sequence:

1. Train a linear time-varying regulator without adversary.
2. Verify that this warmup-only controller matches the analytical LQR solution.
3. Add the broad epsilon adversary.
4. Train under the integrated L2 epsilon budget.
5. Freeze the trained controller and run strong held-out adversary searches with
   multiple PGD step counts and restarts.
6. Compare the trained controller to the analytical H-infinity target.

The success criterion should be a same-game certificate, not merely positive
Delta v. The current intended criteria include:

- gain matrix match or an explainable equivalent representation;
- LQR warmup sanity;
- nominal trajectory match;
- terminal error and cost time-course match;
- induced full-state epsilon gain / robustness match;
- held-out adversary loss close to trained-adversary loss;
- Delta v within a predeclared band around the analytical value;
- failure if large speed inflation is achieved through pathological trajectories
  or poor robustness match.

Exact tolerances should be declared after Phase 0.5 reveals the numerical scale and
training noise floor.

If this gate fails, the GRU arm should not be interpreted. The failure would mean
the training pipeline, task construction, disturbance channel, information
structure, optimization, or evaluation does not yet implement the intended
H-infinity game.

## Phase 4: GRU Under the Same Certified Game

Only after the linear gate succeeds should a GRU be trained under the same plant,
cost, adversary budget, task, observation structure, and evaluation protocol.
Architecture should be the main variable.

The planned outcome tree:

- GRU couples speed and feedback gain. Then broad epsilon training appears
  sufficient to induce the cs2019-like signature even in a flexible recurrent
  architecture. The earlier concern that a GRU can simply decouple may be wrong or
  weaker than expected.
- GRU behaviorally dissociates gain from speed. Then the GRU may be solving the
  broad game in a more tracker-like or structurally separable way than the linear
  regulator, and unlike the human behavioral phenotype in cs2019. This would be a
  substantive scientific result, not just a failure.
- GRU partially couples. Then coupling is quantitative rather than binary, and the
  relevant analysis is a robustness-versus-nominal-speed frontier.
- GRU collapses. If there is no gain increase and no speed increase, the model may
  be ignoring or failing to optimize against the adversary. That is a training or
  pipeline failure mode, not evidence for a scientific dissociation.

If dissociation occurs, the current preferred follow-up is not immediately to
restrict the architecture until it matches humans. The more scientifically
interesting path is to identify real constraints that might suppress decoupling in
human motor control, such as nonlinear damping, two-link arm dynamics, muscle
activation, or delay/biomechanical coupling.

## Curl-Field / Distributional Contrast

The issue thread later adds an important framing from critique: humans were exposed
to curl fields, not directly to full-state epsilon disturbances. Therefore the
broad-epsilon GRU is not the human-protocol-matched condition. It is the
model-matched broad-class condition.

A curl-field-trained GRU is therefore an essential contrast. Its role is to ask:
what does broadening the uncertainty class buy, holding everything else fixed?

The clean contrast should not be:

- curl-field GRU on the production delayed-reach task;
- versus epsilon GRU on the cs2019-faithful movement-only gate.

That comparison would confound adversary class with plant, task, cost, delay,
epoch structure, observation, and evaluation.

The cleaner contrast should be:

- same plant;
- same task;
- same cost;
- same architecture;
- same training budget;
- same evaluation suite;
- uncertainty class changed: curl-field / distributional exposure versus broad
  full-state epsilon exposure.

The curl arm should probably be distributional exposure to physical curl fields,
not necessarily a learned minimax Delta-A adversary, unless the explicit question
is "what happens if curl fields are made adversarial?"

This contrast is central to the "broad-defense prior" framing. If curl-field
training yields gain-only or no-speed behavior, while broad-epsilon training yields
the coupled cs2019 signature under matched conditions, that supports a narrower
claim:

> Broad adversary-class training is sufficient to induce the cs2019-like signature
> in GRUs under this task and objective; curl-distribution training is not.

It would not by itself prove that humans generalize in that exact way. It would
instead identify what the model needs in order to reproduce the human-like
signature.

## Bridge From Gate to Production Interpretability

The cs2019-faithful gate is not the final interpretability target. The eventual
neural-dynamics analysis should be on a production-like delayed-reach task because
that is closer to the trained rlrmp workflows and human task structure.

The bridge is a staged consistency chain:

| Stage | Plant | Task | Adversary scope | Analytical anchor | Purpose |
|---|---|---|---|---|---|
| Gate | cs2019-faithful point mass | movement-only, single reach | epsilon over movement rollout | cs2019-faithful Riccati | Certify same-game training |
| Bridge 1: plant | production point mass | movement-only, single reach | epsilon over movement rollout | new production Riccati | Test plant-parameter shift |
| Bridge 2: targets | production point mass | movement-only, multi-direction | per-trial epsilon | symmetry / per-direction target if available | Test target generality |
| Bridge 3: delayed reach | production point mass | hold + target-on + movement | epsilon active only during movement | movement-epoch anchor | Add task epochs |
| Interpretability | production point mass | delayed reach | as above | no direct Riccati anchor | Analyze internal dynamics |

Each bridge is a consistency check against the previous stage, not a fresh
scientific reset. If a signature disappears at a bridge stage, that disappearance
is itself informative.

The movement-epoch gating constraint is important. In delayed-reach tasks, the
epsilon adversary should be active only during the movement epoch if the goal is to
compare movement-epoch H-infinity behavior. Perturbing preparation state would be a
different game.

## Evaluation Suite

The planned evaluation should be multi-signature and replicate-conditioned. Bare
group means are considered unsafe because replicate populations can be bimodal.

Key measurements:

- nominal Delta v relative to the appropriate baseline;
- peak velocity and time-to-peak;
- terminal endpoint error;
- full nominal trajectory, not only scalar summaries;
- feedback-gain step responses;
- induced gain by perturbation channel;
- held-out adversary loss from independent searches;
- trained-adversary versus held-out-adversary comparison;
- local feedback Jacobians under standardized perturbations;
- cost time courses and loss-term breakdown;
- replicate-level scatter and cluster summaries.

The plan distinguishes baselines:

- analytical LQR is the mathematical baseline for the Riccati/H-infinity match;
- warmup-only trained models are empirical baselines for what optimization finds
  without adversarial training.

For later affine decomposition / interpretability work, the plan notes that local
feedback should be measured under standardized perturbations. Measuring only on a
clean nominal reach can be degenerate because the correction signal is zero by
construction.

## Current Interpretive Commitments

The plan currently commits to the following modest claims:

- The current Delta-A flavor-B adversary is a different game from cs2019's
  analytical broad-epsilon model.
- A linear same-game round trip is required before GRU behavior can be interpreted
  as evidence about the H-infinity hypothesis.
- Feedback-gain modulation is easier and broader than speed inflation; speed
  inflation is a narrower signature.
- Broad epsilon training is model-matched to cs2019's analytical controller.
- Curl-field distributional training is closer to the human perturbation protocol.
- The contrast between broad-epsilon and curl-field training should be designed
  cleanly and not interpreted from mismatched task settings.
- Matching the cs2019 Riccati model behaviorally does not prove that humans solve
  or represent the H-infinity game.

## Open Assumptions and Risks

The following assumptions are known to matter:

- The analytical target and training target must share the same state coordinates
  and disturbance metric.
- The disturbance budget norm must match between Riccati and PGD training.
- The controller's observation / information structure must match the analytical
  controller for the linear gate to be meaningful.
- The trained adversary may not be worst-case unless held-out searches confirm it.
- The six-state simplification must be justified by the eight-state equivalence
  demonstration.
- The curl-field contrast needs a precise protocol: perturbation probability,
  direction distribution, sign distribution, catch trials, block context, and
  whether evaluation is on clean, perturbed, or mixed trials.
- "Generalization" needs an observable definition, likely robustness transfer to
  perturbation families not seen during training.
- A speed increase can arise from robust control, urgency, cost misspecification,
  overshoot tolerance, optimization pathologies, or unstable dynamics. It should
  not be interpreted alone.

## Tentative Guiding Questions for Review

These questions are intentionally tentative. A useful review may decide that some
of them are the wrong questions, or that a more central issue has been missed.

1. What is the strongest version of the current plan?
2. What is the weakest hidden assumption in the plan?
3. Does the plan correctly distinguish cs2019's human perturbation protocol from
   cs2019's analytical model?
4. Is broad additive epsilon training the right model-matched intervention to test
   the H-infinity explanation in trained controllers?
5. Is the linear same-game gate sufficient, too strict, or missing a key criterion?
6. Is the proposed six-state simplification after an eight-state equivalence demo
   methodologically defensible?
7. Is the curl-field versus broad-epsilon contrast the right operational test of
   "broad-defense policy" or "adversary-class broadening"?
8. What would be a cleaner definition of generalization in this setting?
9. Does the bridge chain add complications in the right order?
10. Are there alternative experiments that would more decisively separate
    uncertainty-class effects, cost-schedule effects, and architecture effects?
11. Is the GRU outcome tree complete, or are there important outcomes that would be
    misclassified?
12. What observations would force us to abandon the current framing rather than
    merely patch the implementation?
13. Which parts of `synthesis-5.md`, if any, are mathematically or conceptually in
    tension with this plan?
14. If source-code review is useful, where should it focus: the Riccati solver, the
    disturbance intervention, the training inner loop, the evaluation metrics, or
    the task construction?

## Appendix A: Issue Lineage

- `0b1f109`: canonical synthesis / brainstorming home for the cs2019-to-RNN
  robustness plan.
- `35f64be`: v1 implementation plan. Still open as historical record, but not the
  final plan.
- `6f783fa`: richer v2 snapshot. Closed as premature and subsumed back into
  `0b1f109`; useful for details such as analytical target materialization,
  multi-criterion gate, and bridge chain.
- `b6084c7`: Tier-0 audit of existing non-comparable controller comparisons.
- `020a65b`: full-state epsilon adversary class.
- `89891ab`: adversary strategy / held-out worst-case verification.
- `6ec6b19`: cost-schedule sweep; folded into the gate insofar as the cs2019
  schedule defines the target game.
- `daa48c8`: bimodal-replicate analysis and cluster-conditioned reporting.
- `f695729`: broader phase umbrella for regulator-coupling and biomechanical
  decoupling.
- `0c95d6b`: later GRU affine decomposition / internal-structure analysis, to be
  updated so feedback is measured under standardized perturbations.
- `c723082`: existing LinearDynamicsAdversary / Delta-A flavor-B work.
- `b399efc`: production movement-ramp warmup schedule work; relevant to later
  production bridges, not the first cs2019-faithful gate.

## Appendix B: Technical Notes Believed Current

These notes are included to avoid confusion. They should not dominate the review
unless a reviewer thinks they affect the plan's substance.

- cs2019's implementation uses a disturbance matrix whose top-left physical-state
  block is identity and whose lagged-state block is zero. Equivalently, epsilon
  enters the current physical state and then propagates through the delay
  augmentation.
- The adversary does not "reach back in time" by directly writing lagged sensory
  states.
- Motor noise and sensory noise in cs2019 are separate from the H-infinity
  disturbance used in the Riccati calculation.
- The two cs2019 disturbance-mediator states appear to be dynamically inert for
  the worst-case H-infinity solution in rlrmp's audit, motivating the planned
  eight-state-to-six-state equivalence demonstration.
- SISU is not required for the first round-trip gate. It is a later conditioning
  axis if the project wants one network spanning robustness levels.

## Appendix C: What This Packet Does Not Try to Do

This packet does not:

- restate or absorb `synthesis-5.md`;
- prove the cs2019 interpretation;
- specify exact command-line invocations;
- settle the implementation details of feedbax interventions;
- choose final numerical tolerances;
- claim that humans solve H-infinity control;
- claim that broad epsilon training is the literal human experimental protocol.

Its purpose is to make the current plan clear enough that a stronger reviewer can
critique it without reconstructing the issue thread.
