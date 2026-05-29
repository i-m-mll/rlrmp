# Phase 1 Adversary Equivalence

Issue: `a7dad8a`. Umbrella: `43e8728`. Game card: `cb98e58`.

Rerun metadata:

- Discretization: `euler`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

This note compares the C&S-style deterministic Riccati state-dependent
disturbance against an open-loop epsilon surrogate under the Phase 0 game-card
budget.

## Fixed Contract

- Gamma factor: `1.05`.
- Gamma: `9625.1728`.
- Budget: `sum_t ||epsilon_t||^2 = 5.4218684e-06`.
- Budget L2 radius: `0.0023284906`.
- Disturbance channel: `B_w = [I_8; 0]`; epsilon is 8D and enters only the
  current physical state.
- Epsilon metric: unweighted discrete rollout sum, no extra `dt` scaling.

## Riccati Feedback Arm

- Total cost without disturbance penalty:
  `5305.5926`.
- Disturbance energy: `5.4218684e-06`.
- H-infinity objective: `4803.2893`.
- Peak forward velocity: `0.80624688`.
- Time to peak: step `16`.
- Terminal position error: `0.0046268192`.

## Open-Loop Surrogate Arm

| PGD steps | restarts | best cost | ratio to Riccati | energy | epsilon L2 distance |
|---:|---:|---:|---:|---:|---:|
| 50 | 8 | 5305.5926 | 1 | 5.4218684e-06 | 0 |
| 200 | 8 | 5305.5926 | 1 | 5.4218684e-06 | 0 |
| 800 | 8 | 5305.5926 | 1 | 5.4218684e-06 | 0 |

The open-loop arm maximizes the finite-horizon task cost under a projected
rollout-level L2 budget. The H-infinity penalty is not part of the ascent
objective because the energy is constrained directly; it is reported as a
diagnostic objective after each optimized rollout. Each restart retains the best
sequence seen over the whole projected-ascent path, so the Riccati-realized
epsilon incumbent cannot be discarded by a nonmonotone optimizer step.

## Predeclared Gates

- PGD convergence: best-cost relative improvement between the two longest
  sweeps should be below `0.001`.
- Restart stability: the top-three restart objectives should span less than
  `0.002` relative.
- Equivalence claim: open-loop and Riccati costs/trajectory metrics should be
  within `0.01` relative, with deterministic replay protected by
  `rtol=1e-09` and
  `atol=1e-11`.

## Interpretation

Open-loop optimization is initialized with the Riccati realized epsilon sequence
as one restart, plus independent random restarts projected to the same L2
budget. Matching or exceeding the Riccati realized sequence here establishes
trajectory-level replay equivalence for this fixed game card and initial
condition. It does not make the open-loop epsilon object equivalent to a
state-dependent feedback adversary across off-trajectory states or training-time
policy changes.
