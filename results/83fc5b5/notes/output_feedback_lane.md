# C&S Output-Feedback / Estimator-In-Loop Lane

Issue: `83fc5b5`. Umbrella: `43e8728`.

This note adds the C&S information-structure lane while preserving the older
deterministic full augmented-state replay as Phase 0A. The canonical C&S
deterministic estimator-in-loop card is now Phase 0B:

Rerun metadata:

- Discretization: `euler`.
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

- Gamma factor: `1.4`.
- Observation: H selects delayed x_(t-5) physical block; gain acts on x_hat_aug.
- Initial condition: C&S-compatible repeated physical initial state in every delay-history block.
- Robust command indexing: MATLAB-compatible: released C&S code applies M(:,:,k) after the backward loop, so k is the first Riccati slice.
- LQR output-feedback cost: `4363.5099`.
- LQR peak forward velocity: `0.73100941`.
- H-infinity output-feedback cost: `4441.4882`.
- H-infinity peak forward velocity: `0.76076819`.
- H-infinity estimator RMS error: `4.130211e-06`.

## Phase 1 Output-Feedback Adversary Equivalence

Gamma factor: `1.4`.

Riccati realized disturbance budget:
`2.0661571e-05` (`L2=0.0045455001`).

Riccati feedback cost:
`8147.209`.

Exact fixed-controller L2-budget audits:

| controller | estimator | exact cost | ratio to LQR exact | ratio to Riccati feedback | quadratic error |
|---|---|---:|---:|---:|---:|
| analytical_lqr_kalman | kalman | 11245.972 | 1 | 1.3803466 | 9.09e-12 |
| analytical_hinf_robust | robust | 8147.209 | 0.72445575 | 1 | 5.46e-12 |
| adam_lqr_fit_kalman | kalman | 15493.838 | 1.3777234 | 1.9017357 | 1.22e-10 |
| lbfgsb_after_adam_lqr_fit_kalman | kalman | 15940.115 | 1.4174067 | 1.9565124 | 7.64e-11 |

The projected-gradient rows below are retained as diagnostics. They include the
Riccati epsilon as an initial candidate, so they should not be read as an
independent proof that unseeded open-loop ascent recovered the same sequence.

| PGD steps | best cost | ratio to Riccati | epsilon L2 distance |
|---:|---:|---:|---:|
| 50 | 8147.209 | 1 | 0 |
| 200 | 8147.209 | 1 | 0 |

## Phase 3 Output-Feedback Linear Gate

Gamma factor for disturbance/audit comparisons: `1.4`.

The output-feedback reference gate evaluates action and cost through
`x_hat_aug`. The existing fitted-controller certificate remains the Phase 0A
deterministic gate; this lane defines the C&S estimator-in-loop target and
disturbance card that later Feedbax/GraphSpec work should preserve.

LQR comparator scope: simplified delayed Kalman baseline, not a full extLQG parity implementation with signal-dependent estimator noise terms.

- LQR clean output-feedback cost:
  `4363.5099`.
- LQR under Riccati epsilon cost:
  `10810.221`.
- H-infinity under Riccati epsilon cost:
  `8147.209`.
- H-infinity / LQR cost ratio under Riccati epsilon:
  `0.75365798`.
- H-infinity vs LQR peak-velocity delta under Riccati epsilon:
  `4.370238%`.

Canonical output-feedback retraining starts from zero, not from the old
deterministic fit:

Clean estimator-in-loop training starts from zero. Because xhat_0=x_0 and clean innovations remain zero, the clean objective is algebraically equivalent to full-state clean training; the estimator-loop distinction is tested by the exact disturbance audits. The Bellman row is a diagnostic one-step LQR dynamic-programming objective using the analytical P[t+1] value matrices; it tests recoverability when the objective identifies the Riccati law, not robust/H-infinity training.

| optimizer | objective ratio | gain rel err | clean cost | exact cost ratio to LQR | exact cost ratio to H-inf |
|---|---:|---:|---:|---:|---:|
| of_adam_lqr_fit | 1.2424059 | 0.9896993 | 5318.8736 | 1.3572954 | 1.8735381 |
| of_lbfgsb_zero_lqr_fit | 1.5476996 | 0.99518567 | 6420.6724 | 1.3840333 | 1.9104456 |
| of_lbfgsb_after_of_adam_lqr_fit | 1.1263033 | 0.98928026 | 4827.5749 | 1.2504078 | 1.7259961 |
| of_bellman_lbfgsb_lqr_fit | 1 | 0.00013943432 | 4363.5903 | 1.00023 | 1.380664 |

Fitted deterministic Phase 3 controllers replayed through the output-feedback
estimator loop:

| controller | clean cost ratio | under-epsilon cost ratio | action mismatch ratio |
|---|---:|---:|---:|
| adam_lqr_fit | 1.2142155 | 1.2362602 | 0.69687479 |
| lbfgsb_after_adam_lqr_fit | 1.1350431 | 1.2828108 | 0.5121412 |

## Phase 4 Implication

Phase 4 should preserve the output-feedback information structure: Feedbax/GraphSpec should feed delayed observations and let hidden state serve as an implicit estimator, rather than exposing true x_aug. The deterministic and output-feedback linear certificates should be treated as separate gates.
