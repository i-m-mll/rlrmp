# Perturbation And Robustness-Training Taxonomy

Issue: `c99ad9d` - Project training-methods coordination

Last updated: 2026-06-04

## Purpose

This note is the durable project reference for perturbation, adversary, and
robustness-training terminology. It is attached to the training-methods
coordination issue because the same vocabulary keeps recurring across phases:
H-infinity targets, epsilon channels, CVaR/top-k objectives, force-field
training, structural dynamics adversaries, and feedback-identification curricula.

Future issue comments can point here instead of reconstructing the taxonomy from
chronological discussion. Update this file when a new formal distinction becomes
project-canonical.

## Current Canon

The current project canon is:

1. The C&S H-infinity model is a broad/full-state additive-epsilon game, not a
   narrow curl-field game and not the `Delta-A` structural-adversary game.
2. The exact analytical adversary is state-dependent through the Riccati game.
   Open-loop epsilon PGD is a trainable/evaluation surrogate only after an
   explicit equivalence or adequacy check.
3. Restricted perturbation families, stochastic perturbation training, and
   CVaR/top-k objectives are useful robustness methods, but they should not be
   silently relabeled as H-infinity-equivalent.
4. Training axes and evaluation lenses must stay separate. For example, a
   nominally trained controller can be evaluated under a Riccati-epsilon lens,
   and a robustly trained controller can be evaluated on nominal-clean reaches.

## Taxonomy

| Family | What it is | H-infinity status | Main use |
|---|---|---|---|
| Full-state/broad epsilon Riccati | Finite-horizon additive epsilon game with gamma penalty and the C&S `B_w` channel | Canonical C&S H-infinity analytical target | Formal same-game target and reference behavior |
| Riccati feedback adversary | State-dependent worst-case disturbance, e.g. `epsilon_t = F_t x_t` | Exact analytical adversary object | Analytical reference, induced-gain/gamma checks |
| Open-loop epsilon PGD | Rollout-level L2-constrained epsilon sequence optimized over time | Surrogate, not automatically equivalent | Trainable adversary or first-gate audit after validation |
| Gamma/induced-gain audit | Frozen-policy check of H-infinity feasibility or worst-case gain | Formal evaluation for a fixed policy | Same-game certificate and robustness frontier |
| Robust Bellman/oracle objectives | Bellman or action-value objectives with closed-form robust inner pieces | Formal only when matched to the game and information pattern | Diagnostic/guided training lane, not generic perturbation training |
| CVaR/top-k/APT/PAI-ASF | Tail-risk or worst-sampled-trial optimization over a sampled distribution | Stochastic robustness, not H-infinity-equivalent by default | Behavioral robustness pressure and comparator arm |
| Restricted physical fields | Curl fields, step loads, orthogonal velocity fields, force pulses | Different game unless explicitly embedded in the H-infinity channel | Human-protocol contrasts and transfer matrix columns |
| Parametric force-profile minimax | Learned force/load profile adversary such as Gaussian bumps | Input-instance adversary, not the C&S full-state game | Held-out adversary tests and restricted-family minimax |
| Process/load epsilon perturbation bank | Mechanics/process epsilon pulses used in diagnostics or training mixtures | Diagnostic/training family; formal only if matched to declared `B_w` | Perturbation-response evaluation and support expansion |
| Command/sensory/delayed-observation perturbations | External channel offsets before controller/plant interfaces | Not H-infinity plant disturbance | Feedback-identification diagnostics and training support |
| Structural/model-class adversary | State-multiplicative dynamics changes such as `Delta-A x` | Different game from C&S Eq. 13 | Separate adversary-family arm; do not treat as C&S H-infinity |
| LEQG/risk-sensitive control | Exponential-quadratic or cumulant-tilted objective family | Formal cousin, not automatically equivalent | Later bridge or comparator if deliberately introduced |
| Analytical teacher/distillation | Action, response-map, local-Jacobian, or Bellman guidance from extLQG/H-infinity | Guided fidelity/capacity evidence | Calibration lane, not pure rollout-discovery evidence |

## Broad Epsilon Versus Restricted Perturbations

The phrase "broad perturbations" is ambiguous and should be avoided in run specs
unless the channel is explicit.

For the C&S H-infinity lane, "broad epsilon" means the declared additive
epsilon channel in the game card. In the C&S-faithful 48D delay-augmented state,
epsilon is 8D and enters the current physical-state block through
`B_w = [I_8; 0]`. It does not directly overwrite lag-buffer history.

That is distinct from:

- curl-field or load-field training;
- process/load pulses used as finite diagnostic rows;
- command-input perturbations;
- sensory or delayed-observation offsets;
- random initial-state offsets;
- structural `Delta-A` dynamics perturbations.

Those families are useful, but they answer different questions unless the game
card explicitly declares them to be the adversary's channel.

## Open-Loop Epsilon Caveat

The Riccati H-infinity adversary is a feedback object. An open-loop epsilon
sequence can be a realization of that object along a particular trajectory.

Therefore:

- open-loop PGD matching the Riccati-realized epsilon on a frozen analytical
  controller is a useful equivalence gate;
- it does not prove that open-loop PGD is equivalent for all off-trajectory
  states or for a changing nonlinear controller during training;
- GRU claims should state whether the training adversary is the formal feedback
  object, a validated open-loop surrogate, or a deliberately different game.

## Initial-State Offsets

Initial-state and early-position offsets are currently central for diagnosing
the C&S GRU feedback-map failure. The randomized fixed-target and static
target-relative screens showed that nominal behavior can remain near extLQG
while initial-position recovery remains a large stress-bank gap.

This does not make random initial-state training "the H-infinity method." It
means initial-state/early-position support is an important feedback-identifying
training or diagnostic family for the current GRU lane. It should be reported as
such.

## Strategy-Conditioning Ladder

When studying optimal-to-robust behavior in flexible recurrent controllers,
distinguish these increasingly ambitious regimes:

1. Separate networks for separate training games, such as one nominal optimal
   controller and one robust controller.
2. One scalar- or context-conditioned network implementing a spectrum of
   strategies across robustness levels, budgets, or task contexts.
3. A cross-trial controller with persistent hidden state that adapts its
   strategy from trial history, block context, perturbation experience, or
   instructions.

These are not interchangeable. The first is the cleanest discovery screen. The
second is the natural internal-dynamics/interpolation target once both endpoint
strategies are understood. The third is a meta-policy or trial-history learning
problem and should not be implied by single-trial training.

## Reporting Rules

Every robust-control or perturbation-training result should make these fields
explicit:

- training objective family: nominal rollout, stochastic expected cost,
  CVaR/top-k, minimax, H-infinity epsilon, teacher-guided, or other;
- adversary or perturbation channel: full-state epsilon, force/load field,
  process/load epsilon, command, sensory, delayed observation, initial state,
  structural dynamics, or mixed;
- whether the adversary is feedback/state-dependent, open-loop, sampled, or a
  fixed parametric family;
- whether gamma, epsilon energy, perturbation amplitude, or distribution
  parameters define the budget;
- validation-selection rule, keeping analytical action/I/O/map metrics
  audit-only unless a guided lane explicitly makes them training or selection
  objectives;
- evaluation lenses reported separately, especially nominal-clean,
  Riccati-epsilon, process/load, restricted-field, initial-state, and
  perturbation-family transfer rows.

## Issue Map

- `43e8728` - cs2019-to-RNN game-equivalence plan.
- `0b1f109` - synthesis connecting C&S induction to RNN training methods.
- `020a65b` - full-state epsilon adversary and C&S disturbance-channel parity.
- `a7dad8a` - adversary-equivalence gate for Riccati-realized versus open-loop
  epsilon on the frozen analytical controller.
- `cb98e58` - analytical game card.
- `1ad3c16` - perturbation amplitude/budget standardization.
- `3992394` - standard perturbation-response diagnostic bank.
- `aacb9ed` - task-breadth and feedback-identification diagnostics for C&S GRUs.
- `c314267` - guided feedback-response, response-map, and local-Jacobian
  supervision lane.
- `ddf7f43` - observation-action map-error decomposition sidecar.
- `ba82f3d` - target-relative static multi-target C&S GRU training contract.

## Revision Policy

Update this note when:

- the formal game card changes;
- a new adversary or perturbation family becomes part of the standard method
  menu;
- evidence changes whether a family is treated as H-infinity-equivalent,
  diagnostic-only, or a separate training game;
- the strategy-conditioning ladder advances from separate networks to
  context-conditioned or trial-history controllers.

Issue comments should summarize the change and point back to this file.
