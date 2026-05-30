# Gamma-Penalized Output-Feedback Robust Feasibility Sweep

Issue: `97604a8`. Output-feedback lane: `83fc5b5`.
Umbrella: `43e8728`.

Rerun metadata:

- Discretization: `euler`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

This note extends the exact output-feedback Phase 1 audit from an L2-budget
trust-region check to a gamma-penalized H-infinity feasibility check. For each
gamma factor, the robust output-feedback controller is built in the C&S
estimator-in-loop lane, the coupled `[x, xhat]` closed-loop quadratic is
flattened over the whole epsilon trajectory, and the condition
`gamma^2 I - Q_epsilon` is checked.

`robust lambda/gamma^2` below is `lambda_max(Q_epsilon) / gamma^2` for the
analytical H-infinity robust controller. Values below 1 are finite for the
penalized maximization; values at or above 1 indicate an unbounded penalized
open-loop epsilon objective for that frozen controller.

Gamma star: `9166.8313`.

Selected working output-feedback gamma factor:
`1.4`.

| gamma factor | status | robust lambda/gamma^2 | robust penalized feasible | min estimator eig | min gain-correction eig | min fixed-policy eig | robust exact/LQR exact | H-inf peak velocity |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1.001 | ok | 1.4734828 | false | 0.061264737 | 0.9757718 | -20341291 | 0.81574102 | 0.79092692 |
| 1.005 | ok | 1.4683333 | false | 0.061291712 | 0.97599479 | -7745962.9 | 0.82424909 | 0.7904267 |
| 1.01 | ok | 1.4618204 | false | 0.061324918 | 0.97626902 | -1.3903895e+09 | 0.83371687 | 0.78981009 |
| 1.02 | ok | 1.4485414 | false | 0.061389658 | 0.97680296 | -1.6242085e+08 | 0.84919158 | 0.78860507 |
| 1.05 | ok | 1.4069118 | false | 0.061571284 | 0.97829667 | -2.9850742e+08 | 0.87997306 | 0.78520275 |
| 1.1 | ok | 1.3338082 | false | 0.061836586 | 0.98047307 | -21857092 | 0.90825515 | 0.78016469 |
| 1.2 | ok | 1.1870412 | false | 0.062254849 | 0.98392216 | -60373687 | 0.90786543 | 0.77199016 |
| 1.25 | ok | 1.1175745 | false | 0.062419977 | 0.98530454 | -3.76433e+08 | 0.84983531 | 0.76865004 |
| 1.3 | ok | 1.0519517 | false | 0.062562045 | 0.98651159 | -6.5996402e+08 | 0.69036982 | 0.76570435 |
| 1.32 | ok | 1.0268534 | false | 0.062613261 | 0.98695203 | -8399038 | 0.60883628 | 0.76462275 |
| 1.33 | ok | 1.0145549 | false | 0.062637773 | 0.98716401 | -2.6233097e+08 | 0.5855349 | 0.76410095 |
| 1.34 | ok | 1.0024238 | false | 0.062661587 | 0.98737072 | -27031112 | 0.58704891 | 0.76359128 |
| 1.345 | ok | 0.99642101 | true | 0.06267324 | 0.98747217 | 1.1814189e+08 | 0.58675473 | 0.76334088 |
| 1.35 | ok | 0.99045992 | true | 0.062684726 | 0.98757236 | 1.2271568e+08 | 0.58571079 | 0.76309337 |
| 1.4 | ok | 0.93312536 | true | 0.06279106 | 0.98851006 | 1.4436489e+08 | 0.72445575 | 0.76076819 |
| 1.45 | ok | 0.87982605 | true | 0.06288357 | 0.98934335 | 1.5982197e+08 | 0.82724009 | 0.75868846 |
| 1.5 | ok | 0.83036023 | true | 0.062964362 | 0.99008747 | 1.7421422e+08 | 0.87984836 | 0.75682072 |
| 2 | ok | 0.49405315 | true | 0.063403751 | 0.99457868 | 3.2671188e+08 | 0.98016743 | 0.74533849 |
| 3 | ok | 0.22738734 | true | 0.06364315 | 0.99763728 | 7.484535e+08 | 0.99679177 | 0.73731668 |

## Recommendation

The smallest swept passing gamma factor is `1.345` (`gamma=12329.388`). This identifies the boundary of feasibility for this sweep, not the mandatory working default. The working default for later output-feedback Phase 1/3 diagnostics is selected separately as `1.4` from this sweep. The current working output-feedback gamma factor is `1.4` because it keeps additional slack (`lambda/gamma^2=0.93312536`). The margin at `1.345` is thin (`lambda/gamma^2=0.99642101`), so `1.35` is the nearest more conservative swept fallback if we want numerical slack.

## Interpretation

The sweep is a Phase 1/0B certificate step. It does not train Phase 3
controllers and does not implement robust Bellman. Its purpose is to identify
which robust analytical output-feedback target is coherent enough for later
Phase 3 training and certification.
