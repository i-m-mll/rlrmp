# Euler Lane Rerun Plan

Issue: `dd232cd`. Umbrella: `43e8728`.

This note records the lane decision after the C&S fidelity audit.

## Decisions

The canonical C&S-aligned work should use the released-code discretization:
forward Euler with `dt = 0.01`. Zero-order hold remains a useful numerical
sensitivity variant, but it is not the released-code-fidelity card.

The project will keep two main lanes side by side:

1. Deterministic analytical lane.
2. Released-code stochastic forward-simulation lane.

Bellman diagnostics remain in the deterministic analytical lane for now. The
released-code stochastic lane should not claim Bellman parity until a separate
objective is derived for the stochastic `extLQG` / robust output-feedback setup.

## Phase 0 Reset

Rebuild the base C&S card with Euler as the canonical discretization. Preserve
ZOH as an explicit named variant for later sensitivity analysis.

The reset must recompute the discrete plant, cost schedule applied to the
augmented delay state, gamma boundary, controller gains, clean rollouts, costs,
velocity summaries, and any epsilon or disturbance-budget objects derived from
the card.

## Phase 1 Rerun

Phase 1 should again be organized around the four exact solution families:

1. LQR.
2. LQG.
3. Full-state H-infinity.
4. Output-feedback H-infinity.

For the deterministic analytical lane, recompute these exact objects and their
deterministic evaluations using the Euler plant. Conceptually this is the same
Phase 1 as before, but all matrix-dependent results must be regenerated because
the discrete plant changed.

For the released-code stochastic lane, evaluate the corresponding controller
families in a C&S-style Monte Carlo forward simulation. This lane needs the C&S
`extLQG` comparator for LQG, the robust released-code command law for
output-feedback H-infinity, sampled sensory noise, sampled motor/process noise,
signal-dependent control noise, shared random draws across comparator arms where
appropriate, and the fixed-step perturbation hook used by the released MATLAB
code.

## Phase 3 Rerun

For the deterministic analytical lane, redo Bellman and rollout-recovery work
using the Euler card:

- deterministic Bellman diagnostics/training;
- deterministic rollout recovery;
- optimizer-condition comparisons;
- finite-gamma/exact audits on recovered controllers.

For the released-code stochastic lane, do not do Bellman yet. Redo rollout
training/evaluation under Euler plus the C&S stochastic forward simulation, with
the `extLQG` comparator where relevant. This lane should answer whether learned
or recovered linear controllers reproduce C&S-style stochastic behavior, not
whether a Bellman objective has been derived for that stochastic game.

## Deferred Variants

ZOH should be retained as a named sensitivity variant because it is a more
accurate discretization of the continuous-time plant. It should not be the
canonical fidelity lane unless the project later decides to prioritize
continuous-time numerical accuracy over released-code matching.

Bellman for the released-code stochastic lane is deferred. If needed later, it
should be tracked as a separate formal-objective issue because `extLQG` and
signal-dependent noise make the objective different from the deterministic
LQR/H-infinity Bellman work already implemented.

