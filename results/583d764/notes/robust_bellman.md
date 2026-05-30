# Robust Bellman Diagnostics

Issue: `583d764`. Umbrella: `43e8728`.
Gamma-sweep precursor: `97604a8`.

This diagnostic tests the robust Bellman/oracle lane before any rollout
retraining. The deterministic full-state rows fit time-varying gains against
the one-step finite-horizon H-infinity Bellman objective with the inner
disturbance maximized in closed form.

Rerun metadata:

- Discretization: `euler`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

Gamma star: `9166.8313`.

## Deterministic Full-State Robust Bellman

| gamma factor | objective ratio | gain rel err | clean peak vel | status |
|---:|---:|---:|---:|---|
| 1.35 | 1 | 0.00011158621 | 0.7629764 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 9.9228663e-05 | 0.76059247 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 0.0001119579 | 0.75664949 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Deterministic Numerical Min-Max Smoke

| gamma factor | t | objective ratio | gain rel err | margin | outer nfev | inner nfev | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.35 | 0 | 1 | 5.461313e-05 | 1.529119e+08 | 24 | 235 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 1 | 1 | 2.0613996e-05 | 1.5288444e+08 | 24 | 233 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 10 | 1 | 2.2354735e-05 | 1.5236105e+08 | 24 | 282 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 30 | 1 | 3.3378986e-05 | 1.4406686e+08 | 33 | 425 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 59 | 1 | 0.86836778 | 1.5297946e+08 | 5 | 46 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 0 | 1 | 8.7908516e-06 | 1.6446719e+08 | 23 | 214 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 1 | 8.4457478e-06 | 1.6443996e+08 | 22 | 226 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | 1 | 1.8247137e-05 | 1.6392467e+08 | 23 | 270 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | 1 | 0.0001442662 | 1.5622162e+08 | 25 | 379 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | 1 | 0.86838734 | 1.6453369e+08 | 5 | 46 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 0 | 1 | 3.9172868e-05 | 1.8883789e+08 | 29 | 257 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 1 | 2.1648371e-05 | 1.8881105e+08 | 23 | 231 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 10 | 1 | 1.7230503e-05 | 1.8830904e+08 | 30 | 287 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 30 | 1 | 1.9691386e-05 | 1.8144371e+08 | 30 | 354 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 59 | 1 | 0.86841473 | 1.8890262e+08 | 5 | 44 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Joint Diagnostic

| gamma factor | lambda/gamma^2 | penalized feasible | clean peak vel |
|---:|---:|---|---:|
| 1.35 | 0.99045992 | true | 0.76309337 |
| 1.4 | 0.93312536 | true | 0.76076819 |
| 1.5 | 0.83036023 | true | 0.75682072 |

## Output-Feedback Joint Policy-Improvement Fit

| gamma factor | objective ratio | gain rel err | clean peak vel | status |
|---:|---:|---:|---:|---|
| 1.35 | 0.9869029 | 0.62658693 | 0.99622267 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 0.98743447 | 0.55026031 | 0.80744831 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 0.9881884 | 0.52668069 | 0.77396334 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

Status: joint_policy_improvement_diagnostic; value sequence is policy-evaluated from the C&S released-code output-feedback gains.

## Output-Feedback Information-State Numerical Min-Max

| gamma factor | target | recovers formal | max gain err | mean gain err | min margin | C&S persistent err |
|---:|---|---|---:|---:|---:|---:|
| 1.35 | formal_time_indexed_information_state | false | 1 | 0.32911612 | 4537013.7 | 0.37142188 |
| 1.4 | formal_time_indexed_information_state | false | 1 | 0.32810934 | 5329681.1 | 0.31676055 |
| 1.5 | formal_time_indexed_information_state | false | 1 | 0.32830765 | 6827031.7 | 0.24160533 |

Status: formal_time_indexed_target; controller u=-K xhat and adversarial hidden-state selector x_adv=M xhat optimized by nested inner-outer L-BFGS-B.

Per-time fits:

| gamma factor | t | objective ratio | gain rel err | margin | outer nfev | inner nfev | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.35 | 0 | 1 | 6.4438645e-06 | 1.5314377e+10 | 23 | 116 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 1 | 1 | 5.5109481e-05 | 1.5134641e+08 | 28 | 800 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 10 | 0.99999999 | 0.003591812 | 7100818.8 | 24 | 1748 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 30 | 1.0001484 | 0.64192725 | 4537013.7 | 13 | 1033 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.35 | 59 | 0.99999805 | 1 | 6880509.8 | 6 | 578 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 0 | 1 | 5.4855664e-06 | 1.6469802e+10 | 23 | 115 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | 1 | 3.2375389e-05 | 1.6278596e+08 | 29 | 713 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | 1 | 0.0024888264 | 7650832.5 | 23 | 1678 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | 1.0001164 | 0.63802002 | 5329681.1 | 12 | 967 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | 0.99999826 | 1 | 7434245.4 | 6 | 580 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 0 | 1 | 1.0586585e-05 | 1.8906697e+10 | 29 | 135 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 1 | 1 | 7.2980539e-06 | 1.869124e+08 | 23 | 680 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 10 | 1.0000001 | 0.0049677741 | 8808712 | 22 | 1602 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 30 | 1.0000998 | 0.63655257 | 6827031.7 | 13 | 1035 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.5 | 59 | 0.99999852 | 1 | 8600438.1 | 6 | 576 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Information-State Exact Inner

| gamma factor | target | all feasible | recovers formal | max gain err | mean gain err | min margin | C&S persistent err |
|---:|---|---|---|---:|---:|---:|---:|
| 1.4 | formal_time_indexed_information_state_exact_hidden_state_inner | true | false | 0.86830772 | 0.17366731 | 5329681.1 | 0.31676055 |

Status: formal_time_indexed_target; hidden true state is maximized analytically when gamma^2 Sigma^-1 - L is positive definite.

Per-time fits:

| gamma factor | t | feasible | objective ratio | gain rel err | margin | nfev | status |
|---:|---:|---|---:|---:|---:|---:|---|
| 1.4 | 0 | true | 1 | 1.0993908e-05 | 1.6469802e+10 | 26 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | true | 1 | 5.7203511e-06 | 1.6278596e+08 | 25 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | true | 1 | 5.4382452e-06 | 7650832.5 | 26 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | true | 1 | 6.671874e-06 | 5329681.1 | 24 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | true | 1 | 0.86830772 | 7434245.4 | 5 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Information-State Exact Inner C&S Persistent-Index

| gamma factor | target | all feasible | recovers persistent | max err to persistent | mean err to persistent | max err to formal | formal-persistent ref err | min margin |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1.4 | cs_code_fidelity_persistent_index_exact_hidden_state_inner | true | false | 0.86836825 | 0.1736778 | 0.86836815 | 0.31676055 | 7533750.6 |

Status: cs_code_fidelity_target; control Hessian/cross terms remain tied to P[t+1] but the hidden-state Schur complement uses the released persistent P[0] slice.

Per-time fits:

| gamma factor | t | feasible | objective ratio | err to persistent | err to formal | margin | nfev | status |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1.4 | 0 | true | 1 | 1.0993908e-05 | 1.0993908e-05 | 1.6469802e+10 | 26 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 1 | true | 1 | 6.3668434e-06 | 0.00014491142 | 1.6280962e+08 | 23 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 10 | true | 1 | 2.3729569e-06 | 0.019288935 | 7730802.3 | 26 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 30 | true | 1 | 9.9634309e-07 | 0.31337774 | 7533750.6 | 27 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 1.4 | 59 | true | 1 | 0.86836825 | 0.86836815 | 7634527.8 | 5 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Output-Feedback Flattened Epsilon Exact Inner

| gamma factor | feasible | objective ratio | gain rel err | margin | lambda/gamma^2 | ref margin | ref lambda/gamma^2 | C&S persistent err | status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.4 | true | 1 | 7.9155362e-09 | 25535424 | 0.8449583 | 25535424 | 0.8449583 | 0.31676055 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

Status: full_horizon_closed_loop_target; flattened epsilon trajectory is maximized analytically when gamma^2 I - H_epsilon is positive definite.

The output-feedback fit is a diagnostic, not a proof of the C&S robust
separation theorem. It policy-evaluates the released-code-compatible
output-feedback gains into a joint value sequence over `z=[x,xhat]`, then asks
whether one-step robust Bellman fitting recovers those gains when control is
restricted to `u=-K xhat`. In this run it does not recover those gains: the
smallest output-feedback gain relative error is `0.52668069`, and
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
`0.86830772` and positive minimum margin
`5329681.1`. This is a strict improvement
over the numerical inner-outer optimizer, whose later time slices retained
large gain mismatch despite near-reference objective ratios. The Gemini-style
flattened epsilon objective is numerically stable and feasible
(`lambda/gamma^2 = 0.8449583`), but its
learned controller stays essentially at the formal time-indexed target
(`gain_relative_error = 7.9155362e-09`) rather than
moving toward the C&S persistent-index target
(`C&S persistent error = 0.31676055`).

The persistent-index exact-inner section is intentionally labeled
C&S-code-fidelity. It changes the information-state Bellman block only enough
to make the hidden-state Schur complement use the released persistent Riccati
slice `P[0]`; the one-step control Hessian/cross terms still come from the
formal `P[t+1]` robust control step. At gamma factor
`1.4`, this code-fidelity objective
recovers the released persistent-index target with max gain relative error
`0.86836825`. The same
fitted gains remain separated from the formal time-indexed target with max
relative error `0.86836815`,
matching the reference-level formal-vs-persistent split
`0.31676055`.
