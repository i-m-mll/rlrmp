# C&S Output-Feedback / Estimator-In-Loop Lane

Issue: `83fc5b5`. Umbrella: `43e8728`.

This note adds the C&S information-structure lane while preserving the older
deterministic full augmented-state replay as Phase 0A. The canonical C&S
deterministic estimator-in-loop card is now Phase 0B:

Rerun metadata:

- Discretization: `zoh`.
- Lane: `deterministic_analytical`.
- Lane scope: Deterministic analytical lane: exact recursions and deterministic rollouts/audits with no sampled sensory, motor/process, or signal-dependent control noise.

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

Exact fixed-controller L2-budget audits:

| controller | estimator | exact cost | ratio to LQR exact | ratio to Riccati feedback | quadratic error |
|---|---|---:|---:|---:|---:|
| analytical_lqr_kalman | kalman | 5709.1549 | 1 | 1.1545894 | 1e-11 |
| analytical_hinf_robust | robust | 5005.3995 | 0.87673213 | 1.0122656 | 1.18e-11 |
| adam_lqr_fit_kalman | kalman | 6556.8816 | 1.1484855 | 1.3260291 | 1.64e-11 |
| lbfgsb_after_adam_lqr_fit_kalman | kalman | 6122.5098 | 1.0724021 | 1.2381841 | 5.73e-11 |

The projected-gradient rows below are retained as diagnostics. They include the
Riccati epsilon as an initial candidate, so they should not be read as an
independent proof that unseeded open-loop ascent recovered the same sequence.

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

Canonical output-feedback retraining starts from zero, not from the old
deterministic fit:

Clean estimator-in-loop training starts from zero. Because xhat_0=x_0 and clean innovations remain zero, the clean objective is algebraically equivalent to full-state clean training; the estimator-loop distinction is tested by the exact disturbance audits. The Bellman row is a diagnostic one-step LQR dynamic-programming objective using the analytical P[t+1] value matrices; it tests recoverability when the objective identifies the Riccati law, not robust/H-infinity training.

| optimizer | objective ratio | gain rel err | clean cost | exact cost ratio to LQR | exact cost ratio to H-inf |
|---|---:|---:|---:|---:|---:|
| of_adam_lqr_fit | 1.0845942 | 0.98936579 | 4661.8353 | 1.1479793 | 1.3093843 |
| of_lbfgsb_zero_lqr_fit | 1.4284263 | 0.99476252 | 5859.154 | 1.3712741 | 1.5640742 |
| of_lbfgsb_after_of_adam_lqr_fit | 1.0086626 | 0.98880891 | 4295.3733 | 1.0637975 | 1.2133666 |
| of_bellman_lbfgsb_lqr_fit | 1 | 0.00016594268 | 4288.8359 | 1.0002233 | 1.140854 |

Fitted deterministic Phase 3 controllers replayed through the output-feedback
estimator loop:

| controller | clean cost ratio | under-epsilon cost ratio | action mismatch ratio |
|---|---:|---:|---:|
| adam_lqr_fit | 1.0869923 | 1.1470837 | 0.45198345 |
| lbfgsb_after_adam_lqr_fit | 1.0010846 | 1.0627964 | 0.038177654 |

## Phase 4 Implication

Phase 4 should preserve the output-feedback information structure: Feedbax/GraphSpec should feed delayed observations and let hidden state serve as an implicit estimator, rather than exposing true x_aug. The deterministic and output-feedback linear certificates should be treated as separate gates.
