# Phase 1 Released Stochastic Evaluation

Issue: `dd232cd`. Umbrella: `43e8728`. Deterministic
Phase 1 comparator: `a7dad8a`.

Rerun metadata:

- Discretization: `euler`.
- Lane: `released_stochastic`.
- Lane scope: Released-code stochastic lane: Euler plant plus sampled sensory, motor/process, and signal-dependent control noise. Bellman parity is not claimed unless a separate stochastic objective is derived.
- Bellman claim: `none; stochastic Bellman parity is explicitly out of scope`.

This note materializes the Phase 1 released-code stochastic lane. All arms use
the Euler plant and sampled sensory, state-space motor/process, and
signal-dependent state noise. Each seed reuses the same noise bundle across
arms, so output-feedback LQG and robust comparisons use common random numbers.

## Comparator Scope

The output-feedback LQG arm uses the local port of the C&S
`extLQG -> computeOFC -> computeExtKalman` fixed-point comparator. The robust
arm uses the local C&S-style output-feedback H-infinity gains. No stochastic
Bellman objective or Bellman parity is claimed in this lane.

## Summary Metrics

Trials: `12`. Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`.

| Arm | Structure | Comparator status | Mean cost | Cost std | Peak v mean | Terminal error mean | Estimator RMS mean |
|---|---|---|---:|---:|---:|---:|---:|
| `full_state_lqr` | full_state | exact deterministic LQR gains under sampled stochastic plant | 4360.8734 | 17.339323 | 0.73115254 | 0.0031136322 | n/a |
| `full_state_hinf` | full_state | exact deterministic H-infinity gains under sampled stochastic plant | 4576.6909 | 19.314304 | 0.78569898 | 5.4246591e-05 | n/a |
| `output_feedback_lqg_extlqg` | output_feedback | fixed_point: local port of extLQG/computeOFC/computeExtKalman | 4379.075 | 21.002769 | 0.7263866 | 0.0033047811 | 0.0096668309 |
| `output_feedback_hinf` | output_feedback | C&S-style robust output-feedback gains under sampled stochastic plant | 7052.1616 | 597.66387 | 0.78604528 | 0.0008793739 | 0.014443941 |

## Noise Contract

- Motor covariance scale: `1e-10`.
- Process covariance scale: `1.0`.
- Signal-dependent tensor scale: `0.02`.
- Shared-noise policy: Each seed samples one draw bundle and reuses it for full-state LQR, full-state H-infinity, output-feedback extLQG, and output-feedback H-infinity arms.

## Interpretation

This is a released forward-simulation check for exact controller families where
local exact arrays exist. It should be read beside the deterministic analytical
Phase 1 result, not as a replacement for it. The output-feedback LQG row now
uses the local extLQG fixed-point path; remaining fidelity questions should be
treated as numerical/audit questions against the MATLAB code, not as a missing
comparator implementation.
