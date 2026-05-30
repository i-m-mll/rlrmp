# Output-Feedback Rollout Recovery for the Linear Bridge

Issue: `7a459bb`. Umbrella: `43e8728`.

This note extends the first `7a459bb` matrix with two rescue conditions:
objective-preserving block/time preconditioning and noncanonical Bellman-objective
auxiliary guidance. Objective-preserving rows are run from scratch and from a
Bellman-initialized gain; the Bellman-auxiliary row is scratch-only.

Rerun metadata:

- Discretization: `euler`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

Scope: Clean output-feedback LQR rollout recovery: clean, stronger optimizer, whitening/scaling, and stronger optimizer plus whitening/scaling are objective-preserving; block/time preconditioning is objective-preserving and unscaled before reporting; Bellman-auxiliary guidance is noncanonical, scratch-only, and annealed off before final clean-rollout continuation.

Non-goals: No weak Bellman/proximal anchor, no action/gain matching to the known controller, no coverage perturbations, no robust rollout, and no GRU training in this materialization.

Bellman initialization gain relative error:
`0.00013943432`.

Output-feedback certificate gamma factor:
`1.4`.

Training-state scale condition:
`3.9276054`.

Time/block preconditioner scale condition:
`1.3360532`.

Bellman auxiliary schedule:
`{'strong_optimizer_whitened_bellman_aux': (0.1, 0.03, 0.01, 0.0)}`.

Initial-state ensemble effective rank:
`10.542073` entropy /
`3.5615011`
participation.

Reference clean LQR cost:
`4363.5099`.

Reference LQR under Riccati epsilon cost:
`10810.221`.

Analytical exact L2 audit costs:
- LQR: `11245.972`
- H-infinity: `8147.209`

## Run Matrix

| condition | init | objective ratio | gain rel err | clean cost | clean action mismatch | under-epsilon ratio | exact L2 ratio | lambda/gamma^2 | iters | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| clean | scratch | 1.2624981 | 0.99509618 | 5144.3938 | 0.55574985 | 1.1880185 | 1.3119629 | 2.4310569 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| clean | bellman_init | 1 | 0.00013279667 | 4363.5099 | 7.0199537e-06 | 0.999995 | 0.99999159 | 1.5551168 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer | scratch | 1.0633372 | 0.9946863 | 4420.9592 | 0.17702762 | 1.0920076 | 1.2059417 | 2.4427811 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer | bellman_init | 1 | 0.00013073821 | 4363.5099 | 9.5254627e-07 | 0.99999554 | 0.99999267 | 1.5551101 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| whitened | scratch | 1.0728253 | 0.98329875 | 4469.8601 | 0.24134247 | 1.1046796 | 1.1769393 | 2.07751 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| whitened | bellman_init | 1 | 0.00013298907 | 4363.5099 | 8.245961e-06 | 0.99999535 | 0.99999189 | 1.5551163 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened | scratch | 1.01317 | 0.97947196 | 4363.7893 | 0.011977817 | 1.0831466 | 1.1558731 | 2.0967916 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened | bellman_init | 1 | 0.00013085276 | 4363.5099 | 1.2048652e-06 | 0.9999954 | 0.99999243 | 1.5551095 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened_block_time | scratch | 1.0154345 | 0.97960462 | 4370.3992 | 0.061795456 | 1.0859043 | 1.158585 | 2.063401 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened_block_time | bellman_init | 1 | 0.00013125818 | 4363.5099 | 1.0261665e-06 | 0.99999542 | 0.99999251 | 1.555109 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| strong_optimizer_whitened_bellman_aux | scratch | 1.0156766 | 0.98076856 | 4366.9939 | 0.044121808 | 1.0842369 | 1.1562768 | 2.0823415 | 2000 | bellman_weight=0.1: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; bellman_weight=0.03: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; bellman_weight=0.01: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT; bellman_weight=0: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |

## Current Verdict

This matrix separates discovery from preservation. From-scratch rows test whether clean rollout can discover the Riccati-like policy; Bellman-init rows test whether clean rollout preserves it once initialized there.

Best from-scratch gain error is `0.97947196` (strong_optimizer_whitened__scratch).
Best Bellman-initialized gain error is `0.00013073821` (strong_optimizer__bellman_init).
Bellman-initialized rollout preserves the analytical policy to a useful gain tolerance under at least one objective-preserving condition.
No from-scratch run in this matrix discovers the analytical policy to the same tolerance. If Bellman-init preserves but scratch fails, discovery is the remaining problem; if both fail, the clean objective itself is not identifying the feedback law.
