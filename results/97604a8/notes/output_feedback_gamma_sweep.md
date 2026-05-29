# Gamma-Penalized Output-Feedback Robust Feasibility Sweep

Issue: `97604a8`. Output-feedback lane: `83fc5b5`.
Umbrella: `43e8728`.

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

Gamma star: `9041.4439`.

| gamma factor | status | robust lambda/gamma^2 | robust penalized feasible | min estimator eig | min gain-correction eig | min fixed-policy eig | robust exact/LQR exact | H-inf peak velocity |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1.001 | ok | 1.4797993 | false | 0.061169006 | 0.97549197 | -77036698 | 0.81229017 | 0.77654586 |
| 1.005 | ok | 1.4745808 | false | 0.061196888 | 0.97571834 | -49215957 | 0.82068953 | 0.77601311 |
| 1.01 | ok | 1.467987 | false | 0.061231212 | 0.9759967 | -25542181 | 0.82987187 | 0.77535675 |
| 1.02 | ok | 1.4545625 | false | 0.061298142 | 0.97653861 | -1.7595141e+09 | 0.84542402 | 0.77407513 |
| 1.05 | ok | 1.4126024 | false | 0.061485973 | 0.97805416 | -7315293.8 | 0.87673213 | 0.77046417 |
| 1.1 | ok | 1.3391502 | false | 0.061760532 | 0.98026113 | -97338654 | 0.90582459 | 0.76539217 |
| 1.2 | ok | 1.1919234 | false | 0.062193952 | 0.98375574 | -2.7906557e+08 | 0.90752552 | 0.75740188 |
| 1.25 | ok | 1.122255 | false | 0.062365289 | 0.98515536 | -25015406 | 0.85255038 | 0.75413385 |
| 1.3 | ok | 1.0564357 | false | 0.062512813 | 0.986377 | -82665783 | 0.69957369 | 0.75125044 |
| 1.32 | ok | 1.03126 | false | 0.062566023 | 0.98682267 | -7.4074927e+08 | 0.61533604 | 0.75019142 |
| 1.33 | ok | 1.0189231 | false | 0.062591494 | 0.98703713 | -6.5976121e+08 | 0.58569299 | 0.74968046 |
| 1.34 | ok | 1.0067537 | false | 0.062616243 | 0.98724626 | -2.777571e+09 | 0.57728675 | 0.74918136 |
| 1.345 | ok | 1.0007318 | false | 0.062628354 | 0.98734889 | -11035413 | 0.5833632 | 0.74893613 |
| 1.35 | ok | 0.99475167 | true | 0.062640293 | 0.98745025 | 1.172626e+08 | 0.57975025 | 0.74869374 |
| 1.4 | ok | 0.93722957 | true | 0.062750851 | 0.98839869 | 1.4004419e+08 | 0.70805681 | 0.74641624 |
| 1.45 | ok | 0.88374849 | true | 0.062847084 | 0.98924133 | 1.552079e+08 | 0.81746191 | 0.74437867 |
| 1.5 | ok | 0.8341075 | true | 0.062931158 | 0.98999362 | 1.692622e+08 | 0.8736949 | 0.74254842 |
| 2 | ok | 0.49645263 | true | 0.06338854 | 0.99453079 | 3.17707e+08 | 0.97942811 | 0.73129029 |
| 3 | ok | 0.22854633 | true | 0.063637068 | 0.99761741 | 7.2799808e+08 | 0.99668136 | 0.72341993 |

## Recommendation

Choose gamma factor `1.35` (`gamma=12205.949`) for the next robust linear target, subject to external-review interpretation. This is the smallest swept ratio with positive estimator, gain-correction, fixed-policy, and gamma-penalized exact-audit margins. The margin at `1.35` is thin (`lambda/gamma^2=0.99475167`), so `1.4` is the nearest more conservative swept fallback if we want numerical slack.

## Interpretation

The sweep is a Phase 1/0B certificate step. It does not train Phase 3
controllers and does not implement robust Bellman. Its purpose is to identify
which robust analytical output-feedback target is coherent enough for later
Phase 3 training and certification.
