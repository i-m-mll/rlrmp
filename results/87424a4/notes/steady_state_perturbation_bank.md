<!-- AUTO-GENERATED: steady_state_perturbation_bank -->
# Steady-State Perturbation Bank

This materialization probes GRU feedback sensitivity around a hold-at-target endpoint state. It does not rerun the reach-context perturbation bank.

## Wash-In Contract

- Fan-out policy: prefix_equivalent_batched_trials because the current Feedbax eval API does not expose a supported hidden-state resume hook.
- Delayed rows use a 10-step pre-go prefix followed by post-go hold wash-in.
- Undelayed rows use an immediate hold prefix on the validated trial horizon.
- Default pulse shape: 5 steps; position=0.1 m, velocity=0.5 m/s, force/filter=1.0.

## Comparisons

### `delayed_sisu_effective_020a65b`

- Figure: `results/87424a4/figures/delayed_sisu_effective_020a65b/spec.json`
- Source run: `7c1f7ed/delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| SISU=0 | washin_endpoint_response | 1.567 | position=8.111, velocity=17.07, force_filter=7.529 |
| SISU=1 | washin_endpoint_response | 1.321 | position=9.602, velocity=18.96, force_filter=7.952 |

### `undelayed_targetfix_sisu_effective_020a65b`

- Figure: `results/87424a4/figures/undelayed_targetfix_sisu_effective_020a65b/spec.json`
- Source run: `e4800d6/cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| SISU=0 | washin_endpoint_response | 1.307 | position=11.01, velocity=13.25, force_filter=1.704 |
| SISU=1 | washin_endpoint_response | 1.397 | position=16.21, velocity=17.37, force_filter=2.416 |

### `matched_020a65b_no_pgd_vs_pgd`

- Figure: `results/87424a4/figures/matched_020a65b_no_pgd_vs_pgd/spec.json`
- Source run: `020a65b/target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

| Condition | Response label | Baseline command | Peak output by family |
|---|---:|---:|---|
| No PGD | washin_endpoint_response | 3.957 | position=7.842, velocity=15.78, force_filter=2.861 |
| PGD | washin_endpoint_response | 1.384 | position=11.92, velocity=13.63, force_filter=2.024 |

<!-- /AUTO-GENERATED -->
