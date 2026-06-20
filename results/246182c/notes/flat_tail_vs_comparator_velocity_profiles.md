<!-- AUTO-GENERATED: delayed_timing_hold_velocity_profiles -->
## Delayed timing / pre-go hold velocity profiles

Generated 2-row fixed delayed-bank target-radial velocity profiles.

| Bank | Aggregate HTML | By-replicate HTML |
|---|---|---|
| `no_catch` | `_artifacts/246182c/figures/flat_tail_vs_comparator_velocity_profiles/no_catch/forward_velocity_profiles_stochastic.html` | `_artifacts/246182c/figures/flat_tail_vs_comparator_velocity_profiles/no_catch/forward_velocity_profiles_by_replicate_stochastic.html` |
| `catch` | `_artifacts/246182c/figures/flat_tail_vs_comparator_velocity_profiles/catch/forward_velocity_profiles_stochastic.html` | `_artifacts/246182c/figures/flat_tail_vs_comparator_velocity_profiles/catch/forward_velocity_profiles_by_replicate_stochastic.html` |

Rows:
- `ef9c882/hold__start_pos_zero_vel_lr1e-2` - comparator canonical tail
- `246182c/hold__start_pos_zero_vel_lr1e-2_flat_tail` - flat tail after canonical horizon
<!-- /AUTO-GENERATED -->

<!-- AUTO-GENERATED: flat_tail_interpretation -->
## Interpretation

The flat post-canonical-horizon tail modestly increases the no-catch forward
velocity profile, but it does not change the main timing shape. On the fixed
no-catch bank, peak target-radial velocity rises from `0.6835 m/s` for the
canonical-tail comparator to `0.6968 m/s` for the flat-tail row, with both peaks
at `0.16 s` after the go cue. This is a `+0.0133 m/s` change, about `+1.9%`.

Catch behavior remains effectively stationary. Peak target-radial velocity is
`0.00747 m/s` for the comparator and `0.00728 m/s` for the flat-tail row, both
peaking near `0.06 s` after the cue. The flat tail therefore does not introduce
a visible catch-trial movement tendency in this fixed-bank projection.

Overall, the diagnostic tail helps the delayed reach move slightly faster, but
the effect is small relative to the gap to the extLQG/output-feedback sidecar
reference (`0.7308-0.7311 m/s` peak on the same no-catch figure). The result
supports treating the disappearing post-window running cost as a minor
contributor, not the main cause of the depressed delayed-reach velocity profile.

| Bank | Row | Peak target-radial velocity (m/s) | Time to peak (s) |
|---|---|---:|---:|
| no-catch | comparator canonical tail | 0.6835 | 0.16 |
| no-catch | flat tail after canonical horizon | 0.6968 | 0.16 |
| catch | comparator canonical tail | 0.00747 | 0.06 |
| catch | flat tail after canonical horizon | 0.00728 | 0.06 |
<!-- /AUTO-GENERATED -->
