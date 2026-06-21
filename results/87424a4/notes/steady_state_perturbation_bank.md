<!-- AUTO-GENERATED: steady_state_perturbation_bank -->
# Steady-State Perturbation Bank

This materialization probes GRU feedback sensitivity around a hold-at-target endpoint state. It does not rerun the reach-context perturbation bank.

## Wash-In Contract

- Fan-out policy: prefix_equivalent_batched_trials because the current Feedbax eval API does not expose a supported hidden-state resume hook.
- Delayed rows use a 10-step pre-go prefix followed by 30 post-go hold steps, then preserve a 50-step post-onset response window.
- Undelayed rows use a 30-step immediate hold prefix and extend short hold-at-target validation trials when needed, rather than shortening the 50-step post-onset window.
- Default pulse shape: 5 steps; position=0.1 m, velocity=0.5 m/s, force/filter=10.0.
- Output, position, and velocity rows show primary aligned traces plus lower-emphasis orthogonal companion traces. The orthogonal trace uses the same signed direction rotated +90 degrees in the right-handed x-y plane.

## Comparisons

### `delayed_sisu_effective_020a65b`

- Figure: `results/87424a4/figures/delayed_sisu_effective_020a65b/spec.json`
- Source run: `7c1f7ed/delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| SISU=0 | washin_endpoint_response | 1.567 | position=8.111, velocity=17.07, force_filter=19.17 |
| SISU=1 | washin_endpoint_response | 1.321 | position=9.602, velocity=18.96, force_filter=22.28 |

### `undelayed_targetfix_sisu_effective_020a65b`

- Figure: `results/87424a4/figures/undelayed_targetfix_sisu_effective_020a65b/spec.json`
- Source run: `e4800d6/cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| SISU=0 | washin_endpoint_response | 0.6067 | position=7.524, velocity=15.87, force_filter=19.4 |
| SISU=1 | washin_endpoint_response | 1.398 | position=19.89, velocity=21.63, force_filter=23.18 |

### `matched_020a65b_no_pgd_vs_pgd`

- Figure: `results/87424a4/figures/matched_020a65b_no_pgd_vs_pgd/spec.json`
- Source run: `020a65b/target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| No PGD | washin_endpoint_response | 1.637 | position=12.04, velocity=18.67, force_filter=26.18 |
| PGD | washin_endpoint_response | 0.9042 | position=23.48, velocity=23.43, force_filter=27.98 |

### `matched_020a65b_h0_no_pgd_vs_pgd`

- Figure: `results/87424a4/figures/matched_020a65b_h0_no_pgd_vs_pgd/spec.json`
- Source run: `020a65b/target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| No PGD H0 | washin_endpoint_response | 3.318 | position=2.594, velocity=10.58, force_filter=13.02 |
| PGD H0 | washin_endpoint_response | 0.8743 | position=17.99, velocity=19.43, force_filter=21.06 |

<!-- /AUTO-GENERATED -->
