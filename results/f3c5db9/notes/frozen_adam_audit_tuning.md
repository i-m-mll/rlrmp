<!-- AUTO-GENERATED: frozen_adam_audit_tuning -->
# Frozen Adam audit reliability tuning

Issue: `f3c5db9`. Reference issue: `1697bdc`.

This audit uses the same frozen c92 no-PGD rows, lambda regions, mechanisms, and strict validity rule as the 1697bdc frozen-audit reference. A match means Adam found a finite, useful, interior policy at the reference lambda multiplier; objective equality with the reference solver is not required.

## Recommendation

Stage 1 zero-start Adam matches all direct, linear, and affine reference regions. The conservative common setting is steps=12 and lr=1.0e-05; use that for training-facing smoke tests before considering more aggressive per-row settings.

Common Stage 1 matching settings: steps=12, lr=1.0e-05

## Headline

| row | mechanism | lambda multiplier/range | Adam settings | finite/useful/interior | norm/cap | cap-bound | gain | optimizer status | match | recommendation |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| `open_loop_small` | `direct_epsilon` | 2.59368x-2.82843x; reference=2.82843x | stage1_grid; steps=64; lr=1.0e-05; init=zero | finite/true/true | 0.832647 | 0.000% | 774.863 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_small` | `linear_no_bias` | 1.83401x-2x; reference=2x | stage1_grid; steps=12; lr=1.0e-05; init=zero | finite/true/true | 0.557692 | 0.000% | 585.001 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_small` | `affine` | 2x-2.18102x; reference=2.18102x | stage1_grid; steps=12; lr=1.0e-05; init=zero | finite/true/true | 0.612946 | 0.000% | 634.628 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_moderate` | `direct_epsilon` | 2.82843x-3.08442x; reference=3.08442x | stage1_grid; steps=128; lr=3.0e-05; init=zero | finite/true/true | 0.712648 | 0.000% | 449.151 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_moderate` | `linear_no_bias` | 1.83401x-2x; reference=2x | stage1_grid; steps=12; lr=1.0e-05; init=zero | finite/true/true | 0.53014 | 0.000% | 458.095 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_moderate` | `affine` | 2x-2.18102x; reference=2.18102x | stage1_grid; steps=12; lr=1.0e-05; init=zero | finite/true/true | 0.568103 | 0.000% | 510.621 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_stress` | `direct_epsilon` | 2x-2.18102x; reference=2.18102x | stage1_grid; steps=128; lr=1.0e-04; init=zero | finite/true/true | 0.875142 | 0.000% | 566.839 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_stress` | `linear_no_bias` | 1.29684x-1.41421x; reference=1.41421x | stage1_grid; steps=12; lr=1.0e-05; init=zero | finite/true/true | 0.521252 | 0.000% | 475.582 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |
| `open_loop_stress` | `affine` | 1.54221x-1.68179x; reference=1.68179x | stage1_grid; steps=12; lr=1.0e-05; init=zero | finite/true/true | 0.592677 | 0.000% | 506.178 | adam_all_finite | match | Adam stage 1 is reliable enough for this frozen-audit row. |

## Stage 1 grid

| row | mechanism | matching settings | best matching gain | reference solver | reference gain |
|---|---|---:|---:|---|---:|
| `open_loop_small` | `direct_epsilon` | 5 / 16 | 774.863 | `pgd_projected_epsilon` | 700.408 |
| `open_loop_small` | `linear_no_bias` | 1 / 16 | 585.001 | `lbfgsb` | 733.725 |
| `open_loop_small` | `affine` | 1 / 16 | 634.628 | `lbfgsb` | 608.22 |
| `open_loop_moderate` | `direct_epsilon` | 15 / 16 | 449.151 | `pgd_projected_epsilon` | 383.93 |
| `open_loop_moderate` | `linear_no_bias` | 1 / 16 | 458.095 | `lbfgsb` | 539.381 |
| `open_loop_moderate` | `affine` | 1 / 16 | 510.621 | `lbfgsb` | 583.271 |
| `open_loop_stress` | `direct_epsilon` | 14 / 16 | 566.839 | `pgd_projected_epsilon` | 489.912 |
| `open_loop_stress` | `linear_no_bias` | 1 / 16 | 475.582 | `lbfgsb` | 541.794 |
| `open_loop_stress` | `affine` | 1 / 16 | 506.178 | `lbfgsb` | 524.961 |

## Machine-readable artifacts

- `results/f3c5db9/frozen_adam_audit_tuning.json`
- `results/f3c5db9/frozen_adam_audit_tuning.csv`
<!-- /AUTO-GENERATED -->
