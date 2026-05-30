# Coverage and Noise Sweeps for Output-Feedback Rollout Recovery

Issue: `7a459bb`. Umbrella: `43e8728`.

This note records the first post-plan sweep pass for the free time-varying
output-feedback linear rollout bridge. All training rows here use the
`strong_optimizer_whitened` L-BFGS-B condition from scratch. Bellman-initialized
preservation rows are not repeated in these sweeps because the open question is
from-scratch discovery.

Raw manifests:

- `output_feedback_initial_state_variability_sweep_manifest.json`
- `output_feedback_process_noise_sweep_manifest.json`
- `output_feedback_eigenspectrum_coverage_sweep_manifest.json`

## Initial-State Variability

This sweep scales the existing synthetic basis/random augmented-state coverage
together while preserving the canonical reach state and its weight. The current
baseline is `1x` (`basis_scale=0.01`, `random_state_scale=0.02`).

| scale | iters | objective ratio | gain rel err | exact L2 ratio | lambda/gamma^2 |
|---:|---:|---:|---:|---:|---:|
| 0x | 1448 | 1.3492627 | 0.9957519 | 725.7452506 | 2396.6030779 |
| 0.3x | 2000 | 1.0238790 | 0.9841321 | 1.1713429 | 2.0710428 |
| 1x | 2000 | 1.0131700 | 0.9794720 | 1.1558731 | 2.0967916 |
| 3x | 2000 | 1.0060938 | 0.9892487 | 1.1659665 | 2.1142102 |

Interpretation: zero synthetic coverage is catastrophic under the exact audit.
The current `1x` setting remains the best row by gain error and exact-L2 among
these points. The `3x` row improves the training objective but worsens the
controller certificate, consistent with changing the effective training
distribution rather than revealing the analytical controller.

## Process-Noise Scale

This is a released-stochastic evaluation sweep over the C&S-shaped process
covariance multiplier. It does not retrain the deterministic controller.

Reported row below is `strong_optimizer_whitened__scratch`.

| process scale | stochastic cost ratio | action mismatch | peak v mean | terminal err mean |
|---:|---:|---:|---:|---:|
| 0.0 | 1.0004877 | 0.0473433 | 0.7331476 | 0.0031348 |
| 0.3 | 1.0010698 | 0.0482200 | 0.7330702 | 0.0031502 |
| 1.0 | 1.0028821 | 0.0499561 | 0.7330063 | 0.0032078 |
| 3.0 | 1.0083606 | 0.0544303 | 0.7329028 | 0.0033764 |

Interpretation: increasing process-noise scale monotonically worsens the
stochastic behavioral readout for this controller. Process-noise scale is not a
hidden rescue lever in this first pass.

## Eigenspectrum Coverage

Both eigenspectrum methods use signed leading eigenvectors from the exact-audit
epsilon quadratic of the analytical LQR controller. `trajectory` trains on full
perturbed reach trials. `state` trains on time-indexed states harvested from
those perturbed trials, with remaining-horizon suffix costs.

All rows use coverage weight `0.1`.

| objective | modes | scale | iters | objective ratio | gain rel err | exact L2 ratio | lambda/gamma^2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trajectory | 1 | 0.3 | 2000 | 1.2127364 | 0.9889602 | 8.5037980 | 26.3121474 |
| trajectory | 1 | 1.0 | 2000 | 0.9205185 | 0.9892990 | 12.0780638 | 38.0056075 |
| trajectory | 1 | 3.0 | 2000 | 0.5293511 | 0.9947077 | 30.7174771 | 99.4907871 |
| trajectory | 4 | 0.3 | 2000 | 1.2430678 | 0.9898954 | 1.2640804 | 2.2935836 |
| trajectory | 4 | 1.0 | 2000 | 1.2697387 | 0.9912117 | 2.5960191 | 6.7699879 |
| trajectory | 4 | 3.0 | 2000 | 1.0804448 | 0.9883803 | 32.1890426 | 104.5786319 |
| state | 1 | 0.3 | 2000 | 1.0081119 | 0.9908947 | 1.1009955 | 2.1319346 |
| state | 1 | 1.0 | 2000 | 0.9933853 | 0.9909953 | 1.1070289 | 2.1750162 |
| state | 1 | 3.0 | 2000 | 0.9663889 | 0.9919761 | 1.1193935 | 2.2150780 |
| state | 4 | 0.3 | 2000 | 1.0140665 | 0.9921147 | 1.1672881 | 2.1301904 |
| state | 4 | 1.0 | 2000 | 1.0166625 | 0.9935963 | 1.1420130 | 2.0408117 |
| state | 4 | 3.0 | 2000 | 0.9920972 | 0.9930904 | 1.1249597 | 1.9324805 |

Interpretation: eigenspectrum coverage does not restore from-scratch gain
recovery in this grid. Trajectory coverage, especially high-scale rows, changes
the training objective strongly and can severely worsen disturbance sidecars.
State coverage is less destructive and can improve exact-L2 or lambda/gamma^2
relative to the no-coverage baseline, but the gain remains far from the
analytical controller. This suggests the free time-varying rollout bridge is not
fixed by these coverage variants alone.

## Current Verdict

The no-coverage/current-coverage baseline remains blocked for from-scratch
discovery. Initial-state scale, process-noise scale, and the first eigenspectrum
coverage grid do not recover the analytical gain. The next decision point is
whether to try the observer-error coverage alternative or move to a different
objective/parameterization diagnostic rather than simply adding more coverage.
