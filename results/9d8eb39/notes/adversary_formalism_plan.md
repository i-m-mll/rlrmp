# Adversary Formalism and Same-Channel GRU Robustness Plan

Issue: [issue:9d8eb39].

This plan coordinates the adversary-formalism path that connects the analytical
Riccati H-infinity game, open-loop PGD epsilon optimization, and frozen-policy
same-channel robustness diagnostics for trained C&S GRUs. It is not itself a
training-run plan; training expansion lives on [issue:4d79e07] and full-state
epsilon implementation machinery primarily lives on [issue:020a65b].

## Motivation

The project now has three related but distinct robustness objects:

1. The formal C&S H-infinity game: additive full-state/broad epsilon through the
   declared C&S `B_w=[I_8;0]` channel, with gamma/budget semantics inherited from
   the analytical game card.
2. The analytical Riccati adversary: a state-dependent feedback disturbance
   policy, not merely a fixed disturbance sequence.
3. The practical PGD surrogate: an optimized open-loop `T x 8` epsilon sequence
   found against a particular frozen controller or during GRU training.

PGD is useful because it attacks the actual nonlinear GRU rollout. But the
strong claim that PGD recovers the same game object as the analytical Riccati
adversary needs an adequacy check. Until that check is passed, PGD-trained GRUs
should be described as "PGD broad-epsilon robust training" or "same-channel
open-loop adversarial training," not as direct H-infinity game solutions.

## Core References

- [issue:c99ad9d] - project training-methods coordination.
- [issue:4d38c15] - project analysis coordination.
- [issue:020a65b] - full-state epsilon adversary implementation and audits.
- [issue:a7dad8a] - closed Riccati-realized versus open-loop analytical gate.
- [issue:abe33da] - H-infinity phenotype sidecar.
- [issue:e4800d6] - robust-control strategy ladder.
- [issue:89891ab] - adversary strategy verification and PGD best-seen/restart
  discipline.
- `results/c99ad9d/notes/perturbation_taxonomy.md`
- `results/c99ad9d/notes/adversarial_epsilon_robustification_plan.md`
- `results/c99ad9d/notes/pgd_bank_alpha_response_plan_20260607.md`
- `docs/diagnostic_stack.md`

## Claim Boundaries

Use these labels consistently:

| Label | Meaning | Permitted claim |
|---|---|---|
| Riccati H-infinity adversary | State-dependent analytical disturbance policy from the finite-horizon game | Formal analytical reference |
| Open-loop PGD epsilon | Rollout-specific `T x 8` sequence optimized under a declared L2 budget | Practical same-channel surrogate or audit |
| Same-channel worst-case audit | Frozen-controller optimization over the same C&S epsilon channel | Evidence about the trained controller's vulnerability or robustness on that channel |
| Random broad epsilon | Sampled full-state epsilon with the right channel/budget but no inner maximization | Stress/noise exposure, not worst-case evidence |
| Calibrated perturbation bank | Restricted physical/sensory/task perturbation families | Feedback competence and transfer evidence |
| H-infinity phenotype sidecar | Interprets robustness-like behavior across diagnostics | Interpretation sidecar, not a formal certificate by itself |

## Phase A: Formal Contract Freeze

Before comparing PGD and Riccati adversaries, freeze the exact game contract:

- state basis and ordering for the 48D C&S augmented state;
- disturbance channel and shape: current `T x 8` epsilon through `B_w=[I_8;0]`;
- budget/gamma translation, including whether the radius is the output-feedback
  robust exact-audit radius or another declared value;
- time indexing, units, and whether any `dt` factor is included in the norm;
- information pattern: full-state analytical, output-feedback estimator-in-loop,
  or GRU delayed/partial observation;
- controller being attacked: extLQG, output-feedback robust analytical, trained
  GRU baseline, trained PGD GRU, or another frozen policy;
- objective being maximized: full Q/R/Q_f realized cost, excess over nominal,
  gamma-penalized game objective, or another declared objective;
- checkpoint policy for GRU rows, with validation-selected primary and
  feedback-selected audit-only unless a separate issue changes that.

Exit gate: a single JSON-like manifest can describe the above without relying on
conversation context.

## Phase B: PGD-vs-Riccati Equivalence on Analytical Controllers

Use analytical controllers as the calibration object before using PGD evidence
to make stronger claims about trained GRUs.

For a frozen analytical robust controller:

1. Generate or recover the Riccati-realized adversarial epsilon trajectory.
2. Run open-loop PGD on the same channel, budget, initial state, target, noise
   convention, and objective.
3. Compare:
   - achieved objective/cost;
   - epsilon trajectory distance to Riccati realization;
   - epsilon norm/radius usage and boundary fraction;
   - energy by time/component;
   - induced `delta x`, `delta u`, endpoint error, and cost deltas;
   - best-seen versus final PGD objective;
   - sensitivity to PGD initialization, steps, step size, and restarts.

The first target should be the output-feedback robust analytical controller if
that is the comparator used for current PGD GRU runs. ExtLQG/non-robust
analytical rows can be included as secondary controls.

Interpretation:

- If PGD recovers comparable cost and similar epsilon structure, then PGD is a
  strong surrogate for this setup.
- If PGD finds different epsilon with comparable or higher cost, PGD remains
  useful but is not the same object as Riccati feedback.
- If PGD fails to find the known Riccati-strength direction, do not use PGD
  training failures as evidence that the GRU cannot robustify.

## Phase C: Same-Channel GRU Frozen-Policy Audit

For trained GRUs, run same-channel worst-case epsilon audits as frozen-policy
diagnostics, not as formal certificates.

Minimum rows:

- matched force/filter-feedback non-PGD GRU;
- matched force/filter-feedback PGD GRU;
- extLQG analytical comparator where defined;
- output-feedback robust analytical comparator where defined.

Minimum metrics:

- worst-case full Q/R/Q_f cost and delta cost;
- nominal cost and worst-case/nominal ratio;
- achieved epsilon norm/radius and budget saturation;
- epsilon energy by component and time;
- peak/mean/AUC `delta x`;
- peak/mean/AUC `delta u`;
- endpoint error and terminal speed;
- attenuation-style metrics;
- GRU/extLQG and GRU/robust-analytical ratios where denominators are valid;
- PGD path diagnostics: initialization, step count, step size, restarts,
  best-seen objective, final objective, and monotonicity/projection behavior.

Report both absolute values and ratios. Ratios alone are not sufficient because
small denominators can make physically tiny responses look large.

## Phase D: Diagnostic Stack Integration

The same-channel audit should be integrated as an essential robustness sidecar
in the diagnostic stack, alongside but not replacing:

- standard certificate;
- objective comparator;
- perturbation-response bank;
- feedback ablation/lens bundle;
- map-error decomposition;
- H-infinity phenotype sidecar.

The H-infinity phenotype sidecar should consume the same-channel audit as
evidence, but it should keep "formal H-infinity equivalence" separate from
"robustness-like phenotype."

## Implementation Routing

Use this routing:

- [issue:020a65b] owns implementation of full-state epsilon PGD/audit machinery
  unless a narrower child issue is created.
- [issue:4d79e07] owns training replications and pressure/budget sweeps for the
  promising lr=3e-3 PGD row.
- [issue:6f1ffa5] owns Riccati-epsilon teacher/curriculum experiments.
- [issue:9d8eb39] owns the cross-cutting formalism plan and interpretation
  guardrails.
- [issue:4d38c15] receives analysis-method cross-reference comments.
- [issue:c99ad9d] receives training-method cross-reference comments when the
  training objective or adversary family changes.

## Stop Gates

Stop and report before using PGD results for robustness claims if:

- the disturbance channel, norm, budget, or time indexing differs between PGD
  and the analytical comparator;
- PGD diagnostics do not report best-seen objective, achieved radius, and
  projection behavior;
- random broad-epsilon rows are being interpreted as worst-case evidence;
- robust analytical comparator diagnostics are missing for the channel being
  claimed;
- validation-selected and feedback-selected checkpoint roles are conflated;
- the same-channel audit improves while behavioral perturbation-response and
  objective-comparator evidence degrade in a way that changes the scientific
  interpretation.

## Exclusions

Out of scope for this issue:

- launching the 3e-3 PGD replication matrix; see [issue:4d79e07];
- teacher/distillation implementation; see [issue:6f1ffa5];
- changing the standard certificate definition;
- relabeling calibrated perturbation-bank training as H-infinity training;
- moving to CVaR/top-k as the primary method before the PGD same-channel path is
  interpreted.

## Near-Term Next Actions

1. On [issue:020a65b], verify the same-channel worst-case audit can run on the
   frozen output-feedback robust analytical controller and report PGD-vs-Riccati
   comparators.
2. On [issue:4d79e07], replicate the promising lr=3e-3 PGD GRU result with a
   matched non-PGD force/filter-feedback baseline.
3. On [issue:4d38c15], classify same-channel worst-case epsilon audit as an
   essential robustness sidecar and point to this artifact.
4. On [issue:c99ad9d], keep the training-method vocabulary explicit: PGD
   pressure, calibrated-bank sampling, random broad epsilon, CVaR/top-k, and
   teacher/Riccati replay are separate knobs.

