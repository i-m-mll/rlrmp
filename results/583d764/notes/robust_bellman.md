# Robust Bellman Diagnostics

Issue: `583d764`. Umbrella: `43e8728`.
Gamma-sweep precursor: `97604a8`.

This diagnostic tests the robust Bellman/oracle lane before any rollout
retraining. The deterministic full-state rows fit time-varying gains against
the one-step finite-horizon H-infinity Bellman objective with the inner
disturbance maximized in closed form.

Gamma star: `9041.4439`.

## Deterministic Full-State Robust Bellman

| gamma factor | objective ratio | gain rel err | clean peak vel | status |
|---:|---:|---:|---:|---|
| 1.35 | 1 | 0.00011785518 | 0.74856791 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 0.00010322106 | 0.74634329 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 0.00018547661 | 0.74199484 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Joint Diagnostic

| gamma factor | lambda/gamma^2 | penalized feasible | clean peak vel |
|---:|---:|---|---:|
| 1.35 | 0.99475167 | true | 0.74869374 |
| 1.4 | 0.93722957 | true | 0.74641624 |
| 1.5 | 0.8341075 | true | 0.74254842 |

## Output-Feedback Joint Policy-Improvement Fit

| gamma factor | objective ratio | gain rel err | clean peak vel | status |
|---:|---:|---:|---:|---|
| 1.35 | 0.98690782 | 0.64958395 | 1.0838279 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 0.98741629 | 0.5507821 | 0.79727853 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 0.98811584 | 0.52656605 | 0.763054 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

Status: joint_policy_improvement_diagnostic; value sequence is policy-evaluated from the C&S released-code output-feedback gains.

The output-feedback fit is a diagnostic, not a proof of the C&S robust
separation theorem. It policy-evaluates the released-code-compatible
output-feedback gains into a joint value sequence over `z=[x,xhat]`, then asks
whether one-step robust Bellman fitting recovers those gains when control is
restricted to `u=-K xhat`. In this run it does not recover those gains: the
smallest output-feedback gain relative error is `0.52656605`, and
the fitted objectives are lower than the released-code-compatible reference
objectives. This suggests the C&S output-feedback command law is not a fixed
point of this simple joint policy-improvement objective, or that the objective
is still underconstrained relative to the released-code robust estimator law.
