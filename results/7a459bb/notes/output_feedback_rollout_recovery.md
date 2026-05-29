# Output-Feedback Rollout Recovery for the Linear Bridge

Issue: `7a459bb`. Umbrella: `43e8728`.

This note materializes the first `7a459bb` matrix: clean output-feedback LQR
rollout recovery with objective-preserving optimizer/conditioning changes only.
Each condition is run twice: from scratch and from a Bellman-initialized gain.

Scope: Objective-preserving clean output-feedback LQR rollout recovery only: clean, stronger optimizer, whitening/scaling, and stronger optimizer plus whitening/scaling; each from scratch and Bellman-initialized.

Non-goals: No weak Bellman/proximal anchor, coverage perturbations, robust rollout, or GRU training in this materialization.

Bellman initialization gain relative error:
`0.00016594268`.

Training-state scale condition:
`3.9276054`.

Initial-state ensemble effective rank:
`10.542073` entropy /
`3.5615011`
participation.

Reference clean LQR cost:
`4288.752`.

Reference LQR under Riccati epsilon cost:
`4960.8314`.

Analytical exact L2 audit costs:
- LQR: `5709.1549`
- H-infinity: `5005.3995`

## Run Matrix

| condition | init | objective ratio | gain rel err | clean cost | clean action mismatch | under-epsilon ratio | exact L2 ratio | lambda/gamma^2 | iters | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | scratch | 1.2669796 | 0.99493401 | 5069.2833 | 0.57450687 | 1.2084882 | 1.1966033 | 4.2489988 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| clean | bellman_init | 1 | 0.00015943043 | 4288.752 | 8.3468181e-06 | 0.99999805 | 0.99999731 | 2.8093428 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer | scratch | 1.081527 | 0.99462132 | 4409.8679 | 0.2526623 | 1.0728789 | 1.0826864 | 4.2675778 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer | bellman_init | 1 | 0.00015782398 | 4288.752 | 7.9247354e-07 | 0.9999982 | 0.9999974 | 2.8093288 | 1707 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| whitened | scratch | 1.0741972 | 0.98409645 | 4398.057 | 0.24556666 | 1.0646469 | 1.0672429 | 3.7690028 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| whitened | bellman_init | 1 | 0.00015955733 | 4288.752 | 9.1860496e-06 | 0.99999782 | 0.99999704 | 2.8093409 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened | scratch | 1.0128441 | 0.98311632 | 4288.9099 | 0.0090786096 | 1.0424876 | 1.0491853 | 3.7993729 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened | bellman_init | 1 | 0.00015774181 | 4288.752 | 1.1113103e-06 | 0.99999822 | 0.99999737 | 2.8093257 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |

## Current Verdict

This matrix separates discovery from preservation. From-scratch rows test whether clean rollout can discover the Riccati-like policy; Bellman-init rows test whether clean rollout preserves it once initialized there.

Best from-scratch gain error is `0.98311632` (strong_optimizer_whitened__scratch).
Best Bellman-initialized gain error is `0.00015774181` (strong_optimizer_whitened__bellman_init).
Bellman-initialized rollout preserves the analytical policy to a useful gain tolerance under at least one objective-preserving condition.
No from-scratch run in this matrix discovers the analytical policy to the same tolerance. If Bellman-init preserves but scratch fails, discovery is the remaining problem; if both fail, the clean objective itself is not identifying the feedback law.
