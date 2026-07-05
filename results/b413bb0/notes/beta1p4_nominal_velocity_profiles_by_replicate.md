<!-- AUTO-GENERATED: beta1p4_nominal_velocity_profiles_by_replicate -->
# Beta 1.4 Nominal Velocity Profiles by Replicate

- Scope: nominal velocity profiles only for the completed b413 beta 1.4 rows.
- Rows: `direct_epsilon`, `linear_no_bias`, and `affine`.
- GRU traces: one line per trained replicate, averaged over the fixed 64-trial validation rollout bank.
- Checkpoint policy: final trained checkpoint for all GRU traces.
- Analytical comparators: 6D no-integrator extLQG and 6D no-integrator output-feedback H-infinity.
- Figure spec: `results/b413bb0/figures/beta1p4_nominal_velocity_profiles_by_replicate/spec.json`.
- HTML artifact: `_artifacts/b413bb0/figures/beta1p4_nominal_velocity_profiles_by_replicate/figure.html`.
- PNG artifact: `_artifacts/b413bb0/figures/beta1p4_nominal_velocity_profiles_by_replicate/figure.png`.
- PNG renderer: `chrome_headless_html_screenshot_after_kaleido_block`.

| row | n replicates | peak forward velocity range (m/s) |
|---|---:|---:|
| `direct_epsilon` | 5 | 0.730531 - 0.733485 |
| `linear_no_bias` | 5 | 0.0573473 - 0.732558 |
| `affine` | 5 | 0.00198138 - 0.731246 |

<!-- /AUTO-GENERATED -->
