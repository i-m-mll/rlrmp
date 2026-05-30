# State-Weighted Linear Equivalence Certificate

Issue: `d01c35a`. Phase 3 issue: `6f5c79e`.
Umbrella: `43e8728`.

Rerun metadata:

- Discretization: `euler`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

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
| `adam_lqr_fit` | `optimizer_uncertain_not_disturbance_equivalent` | 1.23851 | 1.21784 | 1.33894 | 0.991522 | 0.303869 | 10.7508 |
| `lbfgsb_after_adam_lqr_fit` | `optimizer_uncertain_not_disturbance_equivalent` | 1.1402 | 1.11254 | 1.21772 | 0.991245 | 0.227645 | 12.8576 |

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
| `canonical_clean_reference` | 19.3155/5.29358 | 240713 | 2.9266/8.37583 | 0.240381 | 240713 | 1 | 0.0107771 |
| `training_ensemble_reference_rollouts` | 3.55808/0.705935 | 97.8163 | 0.539103/1.11942 | 0.193145 | 97.8163 | 2.60728 | 0.019174 |
| `validation_ensemble_reference_rollouts` | 5.13809/0.852401 | 137.478 | 0.778498/1.35058 | 0.238113 | 137.478 | 2.73172 | 0.019174 |
| `candidate_heldout_adversary_states` | 3.97036/6.79581 | 108.769 | 0.601569/8.83608 | 0.152505 | 108.769 | 1 | 0.0116743 |
| `analytical_lqr_heldout_adversary_states` | 19.3055/5.3013 | 1358.55 | 2.92508/8.41166 | 0.218959 | 1358.55 | 1 | 0.0106271 |

### `lbfgsb_after_adam_lqr_fit`

| distribution | action rms delta/ref | action mismatch mean | transition rms delta/ref | transition mismatch mean | Bellman action mean | eff rank mean | parallel gain-error mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `canonical_clean_reference` | 17.5582/5.29358 | 172243 | 2.66033/8.37583 | 0.27511 | 172243 | 1 | 0.0106219 |
| `training_ensemble_reference_rollouts` | 3.33175/0.705935 | 75.3618 | 0.50481/1.11942 | 0.240933 | 75.3618 | 2.60728 | 0.0199746 |
| `validation_ensemble_reference_rollouts` | 4.73748/0.852401 | 99.583 | 0.717801/1.35058 | 0.291186 | 99.583 | 2.73172 | 0.0199745 |
| `candidate_heldout_adversary_states` | 3.03227/6.12486 | 41.238 | 0.459434/8.59786 | 0.122424 | 41.238 | 1 | 0.00950475 |
| `analytical_lqr_heldout_adversary_states` | 17.5451/5.3013 | 1325.15 | 2.65836/8.41166 | 0.255226 | 1325.15 | 1 | 0.0104736 |

## Interpretation

The certificate is designed to classify whether large raw gain error is
harmless off-subspace mismatch, optimizer uncertainty, or disturbance-relevant
non-equivalence. A controller that passes clean objective/behavior but fails
held-out cost and adversary-state action/transition metrics should not unlock
GRU same-game interpretation.
