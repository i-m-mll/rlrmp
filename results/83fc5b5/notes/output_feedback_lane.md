# C&S Output-Feedback / Estimator-In-Loop Lane

Issue: `83fc5b5`. Umbrella: `43e8728`.

This note adds the C&S information-structure lane while preserving the older
deterministic full augmented-state replay as Phase 0A. The canonical C&S
fidelity card is now Phase 0B:

```text
y_t = H x_aug,t
x_hat_aug,t+1 = estimator_update(x_hat_aug,t, u_t, y_t)
u_t = gain_t x_hat_aug,t
```

The gain still has full augmented-state support, but it acts on an estimated
augmented state, not directly on the true augmented state.

## Phase 0B Reference

- Observation: H selects delayed x_(t-5) physical block; gain acts on x_hat_aug.
- Initial condition: C&S-compatible repeated physical initial state in every delay-history block.
- Robust command indexing: MATLAB-compatible: released C&S code applies M(:,:,k) after the backward loop, so k is the first Riccati slice.
- LQR output-feedback cost: `4288.752`.
- LQR peak forward velocity: `0.7172296`.
- H-infinity output-feedback cost: `4496.785`.
- H-infinity peak forward velocity: `0.77046417`.
- H-infinity estimator RMS error: `4.5763423e-06`.

## Phase 1 Output-Feedback Adversary Equivalence

Riccati realized disturbance budget:
`3.1540521e-06` (`L2=0.0017759651`).

Riccati feedback cost:
`4944.7493`.

| PGD steps | best cost | ratio to Riccati | epsilon L2 distance |
|---:|---:|---:|---:|
| 50 | 4944.7493 | 1 | 0 |
| 200 | 4944.7493 | 1 | 0 |

## Phase 3 Output-Feedback Linear Gate

The output-feedback reference gate evaluates action and cost through
`x_hat_aug`. The existing fitted-controller certificate remains the Phase 0A
deterministic gate; this lane defines the C&S estimator-in-loop target and
disturbance card that later Feedbax/GraphSpec work should preserve.

LQR comparator scope: simplified delayed Kalman baseline, not a full extLQG parity implementation with signal-dependent estimator noise terms.

- LQR clean output-feedback cost:
  `4288.752`.
- LQR under Riccati epsilon cost:
  `4960.8314`.
- H-infinity under Riccati epsilon cost:
  `4944.7493`.
- H-infinity / LQR cost ratio under Riccati epsilon:
  `0.99675818`.
- H-infinity vs LQR peak-velocity delta under Riccati epsilon:
  `8.0541421%`.

Fitted deterministic Phase 3 controllers replayed through the output-feedback
estimator loop:

| controller | clean cost ratio | under-epsilon cost ratio | action mismatch ratio |
|---|---:|---:|---:|
| adam_lqr_fit | 1.0869923 | 1.1470837 | 0.45198345 |
| lbfgsb_after_adam_lqr_fit | 1.0010846 | 1.0627964 | 0.038177654 |

## Phase 4 Implication

Phase 4 should preserve the output-feedback information structure: Feedbax/GraphSpec should feed delayed observations and let hidden state serve as an implicit estimator, rather than exposing true x_aug. The deterministic and output-feedback linear certificates should be treated as separate gates.
