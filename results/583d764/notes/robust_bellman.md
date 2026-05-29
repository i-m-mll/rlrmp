# Robust Bellman Diagnostics

Issue: `583d764`. Umbrella: `43e8728`.
Gamma-sweep precursor: `97604a8`.

This diagnostic tests the robust Bellman/oracle lane before any rollout
retraining. The deterministic full-state rows fit time-varying gains against
the one-step finite-horizon H-infinity Bellman objective with the inner
disturbance maximized in closed form.

Rerun metadata:

- Discretization: `zoh`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

Gamma star: `9041.4439`.

## Deterministic Full-State Robust Bellman

| gamma factor | objective ratio | gain rel err | clean peak vel | status |
|---:|---:|---:|---:|---|
| 1.35 | 1 | 0.00011785518 | 0.74856791 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 0.00010322106 | 0.74634329 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 0.00018547661 | 0.74199484 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Deterministic Numerical Min-Max Smoke

| gamma factor | t | objective ratio | gain rel err | margin | outer nfev | inner nfev | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.35 | 0 | 1 | 1.21946e-05 | 1.4875542e+08 | 24 | 233 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 1 | 1 | 3.2876022e-05 | 1.4872874e+08 | 24 | 236 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 10 | 1 | 1.1681114e-05 | 1.4822452e+08 | 23 | 277 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 30 | 1 | 0.00014553818 | 1.4023952e+08 | 24 | 379 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 59 | 1 | 8.2670822e-05 | 1.4881853e+08 | 24 | 114 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 0 | 1 | 9.1946327e-06 | 1.5999678e+08 | 24 | 219 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 1 | 4.1781544e-06 | 1.5997034e+08 | 22 | 220 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | 1 | 1.2147597e-05 | 1.5947397e+08 | 23 | 278 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | 1 | 2.4757589e-05 | 1.5205526e+08 | 30 | 374 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | 1 | 0.00026897644 | 1.6005884e+08 | 23 | 111 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 0 | 1 | 4.7652483e-05 | 1.8370538e+08 | 29 | 262 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 1 | 2.1082872e-05 | 1.8367933e+08 | 23 | 229 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 10 | 1 | 1.8047552e-05 | 1.8319584e+08 | 23 | 272 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 30 | 1 | 0.00010588003 | 1.7658008e+08 | 25 | 333 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 59 | 1 | 0.00019327896 | 1.8376568e+08 | 22 | 108 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

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

## Output-Feedback Information-State Numerical Min-Max

| gamma factor | target | recovers formal | max gain err | mean gain err | min margin | C&S persistent err |
|---:|---|---|---:|---:|---:|---:|
| 1.35 | formal_time_indexed_information_state | false | 1 | 0.32854558 | 4447647.9 | 0.37438561 |
| 1.4 | formal_time_indexed_information_state | false | 1 | 0.32841376 | 5214854.9 | 0.319575 |
| 1.5 | formal_time_indexed_information_state | false | 1 | 0.32763943 | 6666588.5 | 0.24411647 |

Status: formal_time_indexed_target; controller u=-K xhat and adversarial hidden-state selector x_adv=M xhat optimized by nested inner-outer L-BFGS-B.

Per-time fits:

| gamma factor | t | objective ratio | gain rel err | margin | outer nfev | inner nfev | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.35 | 0 | 1 | 5.0298594e-06 | 1.4898289e+10 | 24 | 119 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 1 | 1 | 2.5704363e-05 | 1.4723269e+08 | 29 | 731 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 10 | 0.99999998 | 0.00093966346 | 6909478.1 | 22 | 1612 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 30 | 1.0001436 | 0.64175748 | 4447647.9 | 12 | 977 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 59 | 1.0000082 | 1 | 6680890.5 | 8 | 710 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 0 | 1 | 6.1000076e-06 | 1.6022321e+10 | 23 | 115 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 1 | 1.3007256e-05 | 1.5836148e+08 | 30 | 655 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | 1.0000001 | 0.0046070861 | 7444513.3 | 22 | 1629 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | 1.0001185 | 0.63744262 | 5214854.9 | 12 | 972 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | 1.0000079 | 1 | 7219762.3 | 8 | 703 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 0 | 1 | 2.5768118e-05 | 1.8393006e+10 | 29 | 135 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 1 | 9.8301008e-05 | 1.818325e+08 | 24 | 695 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 10 | 1 | 0.0018415339 | 8570857.9 | 22 | 1607 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 30 | 1.000102 | 0.63623154 | 6666588.5 | 13 | 1037 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 59 | 1.0000083 | 1 | 8354659.5 | 8 | 710 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Information-State Exact Inner

| gamma factor | target | all feasible | recovers formal | max gain err | mean gain err | min margin | C&S persistent err |
|---:|---|---|---|---:|---:|---:|---:|
| 1.4 | formal_time_indexed_information_state_exact_hidden_state_inner | true | true | 0.00056880316 | 0.00013645885 | 5214854.9 | 0.319575 |

Status: formal_time_indexed_target; hidden true state is maximized analytically when gamma^2 Sigma^-1 - L is positive definite.

Per-time fits:

| gamma factor | t | feasible | objective ratio | gain rel err | margin | nfev | status |
|---:|---:|---|---:|---:|---:|---:|---|
| 1.4 | 0 | true | 1 | 5.7609668e-05 | 1.6022321e+10 | 19 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | true | 1 | 1.0379354e-05 | 1.5836148e+08 | 21 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | true | 1 | 3.2672997e-05 | 7444513.3 | 24 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | true | 1 | 1.282908e-05 | 5214854.9 | 23 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | true | 1 | 0.00056880316 | 7219762.3 | 19 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Information-State Exact Inner C&S Persistent-Index

| gamma factor | target | all feasible | recovers persistent | max err to persistent | mean err to persistent | max err to formal | formal-persistent ref err | min margin |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1.4 | cs_code_fidelity_persistent_index_exact_hidden_state_inner | true | true | 0.00035479325 | 8.7095144e-05 | 0.30951553 | 0.319575 | 7330367.2 |

Status: cs_code_fidelity_target; control Hessian/cross terms remain tied to P[t+1] but the hidden-state Schur complement uses the released persistent P[0] slice.

Per-time fits:

| gamma factor | t | feasible | objective ratio | err to persistent | err to formal | margin | nfev | status |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1.4 | 0 | true | 1 | 5.7609668e-05 | 5.7609668e-05 | 1.6022321e+10 | 19 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | true | 1 | 1.2824815e-05 | 0.00014943683 | 1.5838448e+08 | 23 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | true | 1 | 8.3451852e-06 | 0.019133361 | 7520900.2 | 27 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | true | 1 | 1.9028026e-06 | 0.30951553 | 7330367.2 | 27 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | true | 1 | 0.00035479325 | 0.0025902753 | 7422951 | 24 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Flattened Epsilon Exact Inner

| gamma factor | feasible | objective ratio | gain rel err | margin | lambda/gamma^2 | ref margin | ref lambda/gamma^2 | C&S persistent err | status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.4 | true | 1 | 8.2645264e-09 | 24611867 | 0.84639233 | 24611867 | 0.84639233 | 0.319575 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

Status: full_horizon_closed_loop_target; flattened epsilon trajectory is maximized analytically when gamma^2 I - H_epsilon is positive definite.

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

The information-state numerical min-max section is the formal time-indexed
target, not the C&S persistent-index target. It optimizes `u=-K xhat` and
`x_adv=M xhat` with nested inner-outer L-BFGS-B and reports the persistent-index
gain mismatch separately. On the default time grid it does not recover the full
formal target: early steps recover tightly, while later steps find nearly
reference-valued objectives with large gain mismatch, so this remains a
diagnostic rather than a success claim.

The exact-inner information-state section removes the numerical inner optimizer
and directly maximizes over the hidden true state. It is only meaningful when
the positive-definite margin is positive; otherwise the row is reported as
unbounded/infeasible. The flattened epsilon section is a separate whole-horizon
closed-loop objective. It checks `gamma^2 I - H_epsilon` directly and reports
infeasible/unbounded instead of coercing a finite value when the margin fails.

Interpretation for the exact-inner comparison at gamma factor
`1.4`: the GPT-style exact hidden-state inner
objective recovers the formal time-indexed output-feedback target on the tested
grid, with max gain relative error
`0.00056880316` and positive minimum margin
`5214854.9`. This is a strict improvement
over the numerical inner-outer optimizer, whose later time slices retained
large gain mismatch despite near-reference objective ratios. The Gemini-style
flattened epsilon objective is numerically stable and feasible
(`lambda/gamma^2 = 0.84639233`), but its
learned controller stays essentially at the formal time-indexed target
(`gain_relative_error = 8.2645264e-09`) rather than
moving toward the C&S persistent-index target
(`C&S persistent error = 0.319575`).

The persistent-index exact-inner section is intentionally labeled
C&S-code-fidelity. It changes the information-state Bellman block only enough
to make the hidden-state Schur complement use the released persistent Riccati
slice `P[0]`; the one-step control Hessian/cross terms still come from the
formal `P[t+1]` robust control step. At gamma factor
`1.4`, this code-fidelity objective
recovers the released persistent-index target with max gain relative error
`0.00035479325`. The same
fitted gains remain separated from the formal time-indexed target with max
relative error `0.30951553`,
matching the reference-level formal-vs-persistent split
`0.319575`.
