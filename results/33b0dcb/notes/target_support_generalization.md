<!-- AUTO-GENERATED: target_support_generalization -->
# Target-Support Generalization Evaluation

Checkpoint policy: validation-selected per replicate from sparse logged validation.
Each target is evaluated with one stochastic rollout per replicate.

## Headline

Recommended verdict: `bracketed`.
This local pass quantifies the completed rows and separates target-support effects, but objective-comparator and repeat-averaged stochastic grids remain sidecars to run before calling the experiment fully answered.

- Old replay reproduces the failure: old-grid held-out endpoint error 0.0290472 m vs seen 0.00379808 m (gap 0.0252491 m).
- Dense all-angle constant-reach training removes the primary split: dense-all endpoint error is 0.00388743 m.
- Sparse8 constant-reach training still leaves a held-out direction penalty: held minus train endpoint gap 0.00905164 m.
- Held-out band rows are robust for small/moderate bands and degrade mildly for the 36-direction held-out stress band: D const_band8 -0.000135637 m; E const_band16 4.87559e-05 m; F const_band36 0.00110167 m.

## Primary Dense 0.15 m Grid

| row | split | n | endpoint mean m | endpoint p95 m | peak radial m/s | t_peak s | late neg radial m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A old_replicate` | `held_out_direction_interpolated_radius` | 4 | 0.026301 | 0.0334069 | 0.605605 | 0.157 | -0.000198424 |
| `A old_replicate` | `interpolation_unseen_grid` | 62 | 0.0182145 | 0.0313899 | 0.688581 | 0.159548 | -0.00105998 |
| `A old_replicate` | `train_support` | 6 | 0.00409631 | 0.0067691 | 0.729897 | 0.16 | -0.00276635 |
| `B const_dense_all` | `train_support` | 72 | 0.00388743 | 0.0065154 | 0.73019 | 0.16 | -0.00304464 |
| `C const_sparse8` | `held_out_support` | 64 | 0.012902 | 0.0207435 | 0.715474 | 0.160062 | -0.00234939 |
| `C const_sparse8` | `train_support` | 8 | 0.00385031 | 0.00641754 | 0.73331 | 0.16 | -0.00359761 |
| `D const_band8` | `held_out_support` | 8 | 0.00385051 | 0.00629127 | 0.730197 | 0.16 | -0.00278834 |
| `D const_band8` | `train_support` | 64 | 0.00398615 | 0.00641338 | 0.729437 | 0.16 | -0.0027773 |
| `E const_band16` | `held_out_support` | 16 | 0.00382881 | 0.00646092 | 0.730273 | 0.16 | -0.00318223 |
| `E const_band16` | `train_support` | 56 | 0.00378005 | 0.00610059 | 0.730545 | 0.16 | -0.00303239 |
| `F const_band36` | `held_out_support` | 36 | 0.00488843 | 0.00789169 | 0.732568 | 0.159889 | -0.00272332 |
| `F const_band36` | `train_support` | 36 | 0.00378676 | 0.00622353 | 0.731262 | 0.16 | -0.00357784 |

## Old 020a65b Validation Grid

| row | split | n | endpoint mean m | endpoint p95 m | peak radial m/s | t_peak s | late neg radial m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A old_replicate` | `old_held_out_support` | 8 | 0.0290472 | 0.0416916 | 0.591688 | 0.15775 | -0.000312504 |
| `A old_replicate` | `old_seen_support` | 11 | 0.00379808 | 0.00639845 | 0.597251 | 0.159818 | -0.00311042 |
| `A old_replicate` | `original_anchor` | 1 | 0.00198492 | 0.00287128 | 0.730194 | 0.16 | -0.000439536 |
| `B const_dense_all` | `old_held_out_support` | 8 | 0.00726015 | 0.0135849 | 0.726144 | 0.15975 | -0.0072837 |
| `B const_dense_all` | `old_seen_support` | 11 | 0.0060744 | 0.0101595 | 0.660806 | 0.156545 | -0.0159255 |
| `B const_dense_all` | `original_anchor` | 1 | 0.00139072 | 0.00180912 | 0.732444 | 0.16 | 0 |
| `C const_sparse8` | `old_held_out_support` | 8 | 0.0221987 | 0.0357254 | 0.708428 | 0.16025 | -0.00754194 |
| `C const_sparse8` | `old_seen_support` | 11 | 0.0189722 | 0.0381338 | 0.686484 | 0.158909 | -0.0177118 |
| `C const_sparse8` | `original_anchor` | 1 | 0.00164918 | 0.00196473 | 0.735348 | 0.16 | 0 |
| `D const_band8` | `old_held_out_support` | 8 | 0.00755045 | 0.0139933 | 0.726606 | 0.15925 | -0.00603513 |
| `D const_band8` | `old_seen_support` | 11 | 0.00646391 | 0.0114891 | 0.659875 | 0.156727 | -0.0143132 |
| `D const_band8` | `original_anchor` | 1 | 0.00192142 | 0.00218021 | 0.731576 | 0.16 | 0 |
| `E const_band16` | `old_held_out_support` | 8 | 0.00741952 | 0.0137708 | 0.726961 | 0.15925 | -0.00637695 |
| `E const_band16` | `old_seen_support` | 11 | 0.00641926 | 0.0105721 | 0.661559 | 0.155636 | -0.015032 |
| `E const_band16` | `original_anchor` | 1 | 0.00175288 | 0.00238821 | 0.726354 | 0.16 | 0 |
| `F const_band36` | `old_held_out_support` | 8 | 0.0079139 | 0.0135508 | 0.729504 | 0.15875 | -0.00591505 |
| `F const_band36` | `old_seen_support` | 11 | 0.00675805 | 0.0115951 | 0.664567 | 0.156364 | -0.0150979 |
| `F const_band36` | `original_anchor` | 1 | 0.0016069 | 0.00227058 | 0.729835 | 0.16 | -3.66911e-05 |

## Length Diagnostic

Diagnostic-only crossed 0.10/0.12/0.15/0.18 m by 72-direction grid.

| row | split | n | endpoint mean m | endpoint p95 m | peak radial m/s | t_peak s | late neg radial m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A old_replicate` | `held_out_direction__held_out_radius` | 8 | 0.0295785 | 0.0402743 | 0.581101 | 0.15775 | 0 |
| `A old_replicate` | `held_out_direction__seen_radius` | 8 | 0.0236498 | 0.0337196 | 0.51607 | 0.15875 | -0.00028563 |
| `A old_replicate` | `interpolation_direction__held_out_radius` | 124 | 0.0217741 | 0.0354779 | 0.630514 | 0.159226 | -0.000430724 |
| `A old_replicate` | `interpolation_direction__seen_radius` | 124 | 0.0153173 | 0.0284996 | 0.578308 | 0.159758 | -0.00144837 |
| `A old_replicate` | `seen_direction__held_out_radius` | 12 | 0.0128003 | 0.0198903 | 0.66466 | 0.159333 | -0.00120426 |
| `A old_replicate` | `seen_direction__seen_radius` | 12 | 0.00368757 | 0.00621731 | 0.607784 | 0.16 | -0.00258005 |
| `B const_dense_all` | `seen_direction__diagnostic_radius` | 216 | 0.00753504 | 0.013094 | 0.686308 | 0.157713 | -0.0142265 |
| `B const_dense_all` | `seen_direction__seen_radius` | 72 | 0.00395779 | 0.00656036 | 0.730015 | 0.16 | -0.00293827 |
| `C const_sparse8` | `held_out_direction__diagnostic_radius` | 192 | 0.0212219 | 0.0368904 | 0.695748 | 0.159448 | -0.0151216 |
| `C const_sparse8` | `held_out_direction__seen_radius` | 64 | 0.0131615 | 0.0212924 | 0.715878 | 0.160031 | -0.00239211 |
| `C const_sparse8` | `seen_direction__diagnostic_radius` | 24 | 0.0176071 | 0.0358169 | 0.713798 | 0.158583 | -0.0169834 |
| `C const_sparse8` | `seen_direction__seen_radius` | 8 | 0.00375393 | 0.00627778 | 0.733593 | 0.16 | -0.00348765 |
| `D const_band8` | `held_out_direction__diagnostic_radius` | 24 | 0.00789796 | 0.0135854 | 0.684801 | 0.15775 | -0.0117633 |
| `D const_band8` | `held_out_direction__seen_radius` | 8 | 0.00391195 | 0.00638404 | 0.729207 | 0.16 | -0.00285822 |
| `D const_band8` | `seen_direction__diagnostic_radius` | 192 | 0.00793738 | 0.0132923 | 0.685269 | 0.157625 | -0.0119012 |
| `D const_band8` | `seen_direction__seen_radius` | 64 | 0.00387438 | 0.00646096 | 0.729652 | 0.16 | -0.00266961 |
| `E const_band16` | `held_out_direction__diagnostic_radius` | 48 | 0.00761073 | 0.0133931 | 0.68551 | 0.157333 | -0.0126648 |
| `E const_band16` | `held_out_direction__seen_radius` | 16 | 0.0037292 | 0.00658361 | 0.731118 | 0.16 | -0.00314714 |
| `E const_band16` | `seen_direction__diagnostic_radius` | 168 | 0.00781036 | 0.0131573 | 0.686511 | 0.157619 | -0.0129552 |
| `E const_band16` | `seen_direction__seen_radius` | 56 | 0.00374409 | 0.00607702 | 0.730734 | 0.16 | -0.00327092 |
| `F const_band36` | `held_out_direction__diagnostic_radius` | 108 | 0.00796374 | 0.0140738 | 0.687644 | 0.156574 | -0.0116662 |
| `F const_band36` | `held_out_direction__seen_radius` | 36 | 0.00486598 | 0.0077717 | 0.732623 | 0.159778 | -0.00256872 |
| `F const_band36` | `seen_direction__diagnostic_radius` | 108 | 0.00823292 | 0.0132874 | 0.688228 | 0.158389 | -0.0135975 |
| `F const_band36` | `seen_direction__seen_radius` | 36 | 0.00389345 | 0.0062527 | 0.730943 | 0.16 | -0.00315462 |

## Output Files

- Summary CSV: `results/33b0dcb/notes/target_support_generalization_summary.csv`
- Normalized radial velocity profiles bulk CSV: `_artifacts/33b0dcb/target_support_eval/target_support_velocity_profiles.csv`
- Tracked velocity profile pointer: `results/33b0dcb/notes/target_support_velocity_profiles_pointer.json`

Objective-comparator note: this pass reports rollout kinematics and task-target split behavior. It does not materialize the heavier analytical objective comparator sidecar.
<!-- /AUTO-GENERATED -->
