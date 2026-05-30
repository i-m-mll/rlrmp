# Phase 3 Linear Same-Game Round Trip

Issue: `6f5c79e`. Umbrella: `43e8728`.

Rerun metadata:

- Discretization: `euler`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

This note records the first local analytical Phase 3 certificate attempt for
the cs2019-to-RNN game-equivalence programme. It intentionally does not perform
the Feedbax GraphSpec execution conversion or the full `63cec06` matrix-analysis
generalization; those remain the next workup after the local certificate.

## Fixed Game

- Game-card issue: `cb98e58`.
- Adversary-equivalence issue: `a7dad8a`.
- State: 48D delay-augmented C&S state.
- Disturbance: 8D epsilon through `B_w = [I_8; 0]`.
- Cost: C&S 60-step `(t/T)^6` schedule from Phase 0.
- Primary robust target: `gamma = 1.05 * gamma_star`.

## Local Objective-Training Result

Status: `blocked_on_optimizer`.

The analytical replay/audit and teacher-fit representational paths are in place, but the tested objective-gradient optimizers for the clean LQR gain did not meet the predeclared gain/objective pass band from zero initialization.

The clean LQR trainers optimize time-varying full-state gains `K[t]` over a
deterministic full-rank initial-state ensemble. The full-rank ensemble is
necessary because a single reach trajectory can match behavior while leaving
many gain columns underdetermined.

Best objective-trained controller: `lbfgsb_after_adam_lqr_fit`.

| optimizer | objective ratio | gain rel err | clean cost | peak forward v | terminal err | iterations | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `adam_lqr_fit` | 1.2385067 | 0.99152168 | 5314.0355 | 0.75553691 | 0.0039979351 | 2500 | completed |
| `lbfgsb_after_adam_lqr_fit` | 1.1401984 | 0.99124537 | 4854.5904 | 0.74524869 | 0.0036003297 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |

## Teacher-Fit Representational Check

Status: `passed`.

The teacher-fit check trains the same gain tensor shape by gradient descent
against the analytical gain tensor directly. This is not the minimax objective
gate; it isolates representation and metric plumbing from objective-optimizer
quality.

| teacher fit | gain rel err | clean cost | peak forward v | terminal err |
|---|---:|---:|---:|---:|
| `teacher_lqr_fit` | 1.1691543e-16 | 4363.5099 | 0.73100941 | 0.00311078 |
| `teacher_hinf_fit` | 5.5544293e-17 | 4580.6313 | 0.78554543 | 6.5519591e-06 |

## Frozen-Controller Audits

| controller | clean cost | Delta-v vs LQR | gain rel err | held-out cost | held-out steps | terminal err |
|---|---:|---:|---:|---:|---:|---:|
| `analytical_lqr_reference` | 4363.5099 | +0.0000% | 0 | 4863.7157 | 50 | 0.00311078 |
| `adam_lqr_fit` | 5314.0355 | +3.3553% | 0.991522 | 6512.2101 | 50 | 0.00399794 |
| `lbfgsb_after_adam_lqr_fit` | 4854.5904 | +1.9479% | 0.991245 | 5922.6562 | 50 | 0.00360033 |
| `teacher_lqr_fit` | 4363.5099 | +0.0000% | 1.16915e-16 | 4862.4177 | 200 | 0.00311078 |
| `analytical_hinf_reference` | 4580.6313 | +7.4604% | 0 | 5305.5926 | 50 | 6.55196e-06 |
| `teacher_hinf_fit` | 4580.6313 | +7.4604% | 5.55443e-17 | 5305.5926 | 50 | 6.55196e-06 |

Held-out adversary audits use independent projected open-loop epsilon searches
with fresh seeds. Each inner search retains the best-seen objective, not only
the final endpoint, following the Phase 1 `89891ab`/`a7dad8a` lesson.

## Interpretation

This pass should not be treated as a successful Phase 3 exit certificate unless
`phase3_status` is `passed`. A failed clean LQR gain recovery means the local
linear certificate still needs work before GRU same-game interpretation.

The important positive result is narrower: the analytical replay, metric, and
held-out adversary audit surfaces now exist locally and are tied to the exact
Phase 0-2 game. Adam-warm-started L-BFGS-B can recover the clean objective and
canonical behavior, but not the raw analytical gain tensor under the current
full-rank ensemble certificate. The next decision is whether to replace the raw
gain-error gate with a behaviorally equivalent certificate or introduce a more
structured identifiable linear-policy optimization method before attempting the
GRU phase.
