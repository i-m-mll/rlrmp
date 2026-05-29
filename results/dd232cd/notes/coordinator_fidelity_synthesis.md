# Coordinator Synthesis: C&S Game-Card Fidelity

Issue: `dd232cd`. Related umbrella: `43e8728`.

This synthesis summarizes the read-only audit outputs preserved in this
directory:

- `agy_fidelity_audit_summary.md`
- `agy_fidelity_audit_full.md`
- `claude_fidelity_audit.md`
- `fidelity_audit_prompt.md`

## Current Classification

The current rlrmp output-feedback work is a deterministic analytical lane, not
a fully released-code-faithful C&S forward simulation lane.

The analytical pieces are still useful: the Riccati recursion, robust estimator
covariance recursion, output-feedback information structure, and persistent
`M(:,:,k)` code-fidelity variant are meaningful and mostly well separated.
However, the current labels should not imply full C&S released MATLAB simulation
fidelity.

## Confirmed Gaps

1. Forward simulations are deterministic.

   C&S released code samples sensory noise, motor/process noise, and
   signal-dependent control noise during the forward rollout. rlrmp currently
   uses the corresponding covariances in estimator calculations, but the actual
   rollout equations use deterministic observations and dynamics unless an
   explicit deterministic epsilon sequence is supplied.

2. The released MATLAB code uses forward Euler discretization.

   rlrmp's current "faithful" physical plant uses zero-order-hold
   discretization. ZOH is mathematically reasonable, but it is not identical to
   the released C&S code. This affects the discrete plant, gamma boundary,
   gains, and downstream numerical results.

3. The LQG comparator is simplified.

   C&S uses the `extLQG` iteration with signal-dependent noise terms. rlrmp's
   current comparator is standard LQR gains with a delayed Kalman-style
   estimator, not full C&S `extLQG` parity.

4. The C&S perturbation hook is absent.

   The released forward loop can inject `simdata.pert` into the disturbance
   integrator at the fixed perturbation step. The default clean script sets this
   to zero, but perturbation/sensitivity panels need this hook.

5. Some tracked notes overclaim "C&S faithful" status.

   Existing deterministic analytical notes should be relabeled as deterministic
   analytical or C&S information-structure lanes. "Released-code forward
   simulation fidelity" should be reserved for a stochastic lane that includes
   the sampled noise and Euler discretization choices.

## Recommended Path

Do not run more Phase 1 or Phase 3 result matrices against the current card as
if they were C&S-released-code faithful.

The next implementation pass should create or explicitly select a canonical
contract:

- deterministic analytical card, likely retained for formal Bellman/H-infinity
  diagnostics;
- released-code Euler/stochastic forward-simulation card, needed for direct
  C&S phenotype and Monte Carlo simulation claims;
- optional mathematically cleaned variants, kept separate from the released-code
  label.

If released-code fidelity becomes canonical for the next phase reruns, rerun
Phase 0, output-feedback gamma sweep, robust Bellman diagnostics, and rollout
recovery after the Euler and cost-contract changes land.

## Immediate Issue Comments

The audit should be cross-linked from:

- `dd232cd`: authoritative audit preservation and synthesis.
- `3fb0891`: discrepancy ledger entry distinguishing deterministic analytical
  fidelity, C&S information-structure fidelity, and released stochastic
  forward-simulation fidelity.
- `cb98e58`, `83fc5b5`, `97604a8`, `583d764`, `7a459bb`: result-scope notes
  stating which artifacts are deterministic diagnostics and which require rerun
  under a corrected released-code card.

