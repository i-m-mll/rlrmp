<!-- AUTO-GENERATED: post_run_analysis -->
# Post-run analysis: epsilon-scaled short row

This note compares `epsilon_scaled_short_3500to1000` against [issue:91a090c] `short_3500to1000` and `medium_3500to1000`, with the 6D analytical extLQG and output-feedback H-infinity nominal comparators reused from the 91a090c velocity-profile materialization.

## Headline

- The epsilon-scaled row reaches a nominal peak forward velocity of `0.766066 m/s` at `0.16 s`, nearly matching the 91a090c short intended checkpoint (`0.763901`) and medium near-intended checkpoint (`0.766532`).
- Its nominal peak is also close to the reused output-feedback H-infinity comparator (`0.760662 m/s`), while remaining above the reused extLQG nominal comparator.
- Damage control at the end of the run is near, but above, the 1000 target: full-strength mean damage `1233.432`, applied-scaled mean damage `1233.368`, and EMA `1091.923`.
- The controller-training exposure was epsilon-scaled: applied damage starts near zero when the scale is zero, then converges to the full-strength diagnostic once epsilon scale reaches one.

## Nominal Velocity And Quality

| Row / trace | Peak velocity (m/s) | Peak time (s) | Mean terminal error (m) | Endpoint spread (m) | Samples |
|---|---:|---:|---:|---:|---:|
| short_3500to1000 / 6D analytical extLQG nominal | 0.731475 | 0.16 | 0.00324721 | 0.000951497 | 320 |
| short_3500to1000 / 6D output-feedback H-infinity nominal | 0.760662 | 0.16 | 0.0012945 | 0.000648583 | 320 |
| short_3500to1000 / checkpoint 15500 | 0.763901 | 0.16 | 0.00111689 | 0.000566873 | 320 |
| medium_3500to1000 / checkpoint 17000 | 0.766532 | 0.16 | 0.00110246 | 0.000579415 | 320 |
| medium_3500to1000 / checkpoint 19000 | 0.766446 | 0.16 | 0.00108325 | 0.000575608 | 320 |
| epsilon_scaled_short_3500to1000 / checkpoint 16500 | 0.766066 | 0.16 | 0.00105285 | 0.000551325 | 320 |

## Damage, Lambda, And Epsilon Scale

| Row | Completed batches | Target near endpoint | Damage mean near endpoint | Damage mean final | Lambda final | Epsilon scale final | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| short_3500to1000 | 19500 | 1000 | 1110.56 | 1050.22 | 5.54381e+06 | NA | overran beyond intended 15500; data extend to 19499/19500. |
| medium_3500to1000 | 19000 | 1000 | 1086.85 | 1056.72 | 717147 | NA | continued past intended 17250 and was stopped at 19000. |
| epsilon_scaled_short_3500to1000 | 16500 | 1000 | 1233.43 | 1233.43 | 6.77957e+06 | 1 | completed the intended 16500-batch stop with a 1000-batch target hold. |

## Artifacts

- Velocity figure spec: `results/1ab1fef/figures/nominal_velocity_profiles/spec.json`
- Velocity render: `_artifacts/1ab1fef/figures/nominal_velocity_profiles/figure.html`
- Velocity summary: `_artifacts/1ab1fef/figures/nominal_velocity_profiles/summary.json`
- Damage/lambda figure spec: `results/1ab1fef/figures/adaptive_damage_lambda/spec.json`
- Damage/lambda render: `_artifacts/1ab1fef/figures/adaptive_damage_lambda/figure.html`
- Damage/lambda summary: `_artifacts/1ab1fef/figures/adaptive_damage_lambda/summary.json`

## Caveats

- The 91a090c short row overran its intended stop; this comparison uses checkpoint 15500 as the primary short-row endpoint.
- The 91a090c medium row was manually stopped at 19000; checkpoint 17000 is the near-intended comparator and checkpoint 19000 is retained as a later sidecar.
- This analysis does not claim a GRU standard-certificate pass. The analytical curves are nominal behavioral comparators only.

<!-- /AUTO-GENERATED -->
