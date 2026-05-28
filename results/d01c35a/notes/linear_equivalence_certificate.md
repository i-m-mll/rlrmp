# State-Weighted Linear Equivalence Certificate

Issue: `d01c35a`. Phase 3 issue: `6f5c79e`.
Umbrella: `43e8728`.

This note applies the GPT 5.5 Pro critique imported under `6f5c79e` by testing
whether the objective-trained Phase 3 linear controllers are disturbance-
relevant equivalents of analytical LQR, not merely clean canonical reach
matches.

## Status

Overall status: `blocked_not_disturbance_equivalent`.

The richer certificate treats clean behavior as insufficient. Controllers must also match analytical LQR on state-weighted action, closed-loop transition, value, and held-out disturbance-relevant metrics.

## Controller Summary

| controller | classification | objective ratio | clean cost ratio | held-out cost ratio | raw gain err | value gap train cov | final grad norm |
|---|---|---:|---:|---:|---:|---:|---:|
| `adam_lqr_fit` | `optimizer_uncertain_not_disturbance_equivalent` | 1.08399 | 1.08144 | 1.18051 | 0.989656 | 0.0920786 | 0.533118 |
| `lbfgsb_after_adam_lqr_fit` | `optimizer_uncertain_not_disturbance_equivalent` | 1.00883 | 1.00038 | 1.09944 | 0.989122 | 0.0355763 | 0.705864 |

The held-out cost ratio compares each controller's held-out open-loop adversary
audit cost to the analytical LQR held-out audit cost. Values above 1 indicate
worse disturbance-relevant behavior under this audit.

## Distribution Metrics

Action mismatch is the R-weighted ratio of `(K_fit - K_ref)x` to `K_ref x`.
Transition mismatch compares `(A - B K_fit)x` to `(A - B K_ref)x`. Bellman
action mismatch uses `H_t = R_t + B^T P^*_(t+1) B`.

### `adam_lqr_fit`

| distribution | action rms delta/ref | action mismatch mean | transition rms delta/ref | transition mismatch mean | Bellman action mean | eff rank mean | parallel gain-error mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `canonical_clean_reference` | 4.5409/5.30607 | 33084.8 | 0.638437/8.14872 | 0.0532323 | 33084.8 | 1 | 0.00815452 |
| `training_ensemble_reference_rollouts` | 1.2745/0.707175 | 16.2787 | 0.179191/1.08865 | 0.0529448 | 16.2787 | 2.60691 | 0.0186469 |
| `validation_ensemble_reference_rollouts` | 1.79038/0.853622 | 26.1934 | 0.251722/1.31296 | 0.0727691 | 26.1934 | 2.73215 | 0.0186464 |
| `candidate_heldout_adversary_states` | 2.57178/5.87588 | 0.744796 | 0.361584/8.31574 | 0.0477683 | 0.744796 | 1 | 0.00807143 |
| `analytical_lqr_heldout_adversary_states` | 4.43573/5.31355 | 280.719 | 0.62365/8.18458 | 0.0529455 | 280.719 | 1 | 0.00816252 |

### `lbfgsb_after_adam_lqr_fit`

| distribution | action rms delta/ref | action mismatch mean | transition rms delta/ref | transition mismatch mean | Bellman action mean | eff rank mean | parallel gain-error mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `canonical_clean_reference` | 1.00721/5.30607 | 29352.4 | 0.14161/8.14872 | 0.0211916 | 29352.4 | 1 | 0.000128871 |
| `training_ensemble_reference_rollouts` | 1.35991/0.707175 | 15.087 | 0.1912/1.08865 | 0.0414321 | 15.087 | 2.60691 | 0.0185011 |
| `validation_ensemble_reference_rollouts` | 2.02645/0.853622 | 22.4376 | 0.284912/1.31296 | 0.0627896 | 22.4376 | 2.73215 | 0.0185005 |
| `candidate_heldout_adversary_states` | 1.41583/5.23928 | 1.06399 | 0.199061/8.11203 | 0.0641475 | 1.06399 | 1 | 0.000117392 |
| `analytical_lqr_heldout_adversary_states` | 1.10903/5.31355 | 173.824 | 0.155927/8.18458 | 0.0226271 | 173.824 | 1 | 0.000136266 |

## Interpretation

The certificate is designed to classify whether large raw gain error is
harmless off-subspace mismatch, optimizer uncertainty, or disturbance-relevant
non-equivalence. A controller that passes clean objective/behavior but fails
held-out cost and adversary-state action/transition metrics should not unlock
GRU same-game interpretation.
