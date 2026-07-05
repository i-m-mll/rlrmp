<!-- AUTO-GENERATED: nominal_velocity_profiles -->
## Nominal Velocity Profiles

- Figure: `_artifacts/91a090c/figures/nominal_velocity_profiles/figure.html`
- Figure spec: `results/91a090c/figures/nominal_velocity_profiles/spec.json`
- Profile CSV: `_artifacts/91a090c/figures/nominal_velocity_profiles/profiles.csv`
- Summary JSON: `_artifacts/91a090c/figures/nominal_velocity_profiles/summary.json`
- Figure contents: two subplots, each with only the 6D extLQG line, the 6D output-feedback H-infinity line, and the three requested row checkpoints.

| Row | Trace/checkpoint | Peak mean forward velocity (m/s) | Time to peak (s) | Mean terminal position error (m) | Endpoint spread (m) |
|---|---:|---:|---:|---:|---:|
| short_3500to1000 | 6D analytical extLQG nominal | 0.731475 | 0.16 | 0.00324721 | 0.000951497 |
| short_3500to1000 | 6D output-feedback H-infinity nominal | 0.760662 | 0.16 | 0.0012945 | 0.000648583 |
| short_3500to1000 | checkpoint_0013000 | 0.76137 | 0.16 | 0.0014821 | 0.000756427 |
| short_3500to1000 | checkpoint_0015500 | 0.763901 | 0.16 | 0.00111689 | 0.000566873 |
| short_3500to1000 | checkpoint_0017500 | 0.767168 | 0.16 | 0.00107975 | 0.000563476 |
| medium_3500to1000 | 6D analytical extLQG nominal | 0.731475 | 0.16 | 0.00324721 | 0.000951497 |
| medium_3500to1000 | 6D output-feedback H-infinity nominal | 0.760662 | 0.16 | 0.0012945 | 0.000648583 |
| medium_3500to1000 | checkpoint_0013500 | 0.761624 | 0.16 | 0.00119162 | 0.000627999 |
| medium_3500to1000 | checkpoint_0017000 | 0.766532 | 0.16 | 0.00110246 | 0.000579415 |
| medium_3500to1000 | checkpoint_0019000 | 0.766446 | 0.16 | 0.00108325 | 0.000575608 |

Caveats: the short row is summarized at the requested intermediate checkpoints, not the overrun endpoint; the medium row was user-stopped at checkpoint 19000 and has no `training_summary.json`, so all checkpoint metrics here come from direct nominal rollouts.
<!-- /AUTO-GENERATED -->
