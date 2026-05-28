# Phase 3 Linear Same-Game Round Trip

Issue: `6f5c79e`. Umbrella: `43e8728`.

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

Status: `blocked_on_gain_recovery`.

Adam-warm-started L-BFGS-B recovered the clean LQR objective/behavior gate, but the raw gain tensor remains far from the analytical Riccati gain. Phase 3 is therefore blocked on whether gain-relative-error is the right certificate or whether a structured/identifiable linear policy objective is needed.

The clean LQR trainers optimize time-varying full-state gains `K[t]` over a
deterministic full-rank initial-state ensemble. The full-rank ensemble is
necessary because a single reach trajectory can match behavior while leaving
many gain columns underdetermined.

Best objective-trained controller: `lbfgsb_after_adam_lqr_fit`.

| optimizer | objective ratio | gain rel err | clean cost | peak forward v | terminal err | iterations | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `adam_lqr_fit` | 1.0839928 | 0.98965567 | 4638.0186 | 0.7275051 | 0.0033697862 | 2500 | completed |
| `lbfgsb_after_adam_lqr_fit` | 1.0088325 | 0.9891224 | 4290.3905 | 0.7175249 | 0.0030968906 | 500 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |

## Teacher-Fit Representational Check

Status: `passed`.

The teacher-fit check trains the same gain tensor shape by gradient descent
against the analytical gain tensor directly. This is not the minimax objective
gate; it isolates representation and metric plumbing from objective-optimizer
quality.

| teacher fit | gain rel err | clean cost | peak forward v | terminal err |
|---|---:|---:|---:|---:|
| `teacher_lqr_fit` | 1.2225762e-16 | 4288.752 | 0.7172296 | 0.0030967012 |
| `teacher_hinf_fit` | 2.3142036e-17 | 4507.4441 | 0.7709339 | 7.0706828e-06 |

## Frozen-Controller Audits

| controller | clean cost | Delta-v vs LQR | gain rel err | held-out cost | held-out steps | terminal err |
|---|---:|---:|---:|---:|---:|---:|
| `analytical_lqr_reference` | 4288.752 | +0.0000% | 0 | 4789.8634 | 50 | 0.0030967 |
| `adam_lqr_fit` | 4638.0186 | +1.4327% | 0.989656 | 5654.4856 | 50 | 0.00336979 |
| `lbfgsb_after_adam_lqr_fit` | 4290.3905 | +0.0412% | 0.989122 | 5266.175 | 200 | 0.00309689 |
| `teacher_lqr_fit` | 4288.752 | +0.0000% | 1.22258e-16 | 4784.0777 | 200 | 0.0030967 |
| `analytical_hinf_reference` | 4507.4441 | +7.4877% | 0 | 5236.1406 | 50 | 7.07068e-06 |
| `teacher_hinf_fit` | 4507.4441 | +7.4877% | 2.3142e-17 | 5236.1406 | 50 | 7.07068e-06 |

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
