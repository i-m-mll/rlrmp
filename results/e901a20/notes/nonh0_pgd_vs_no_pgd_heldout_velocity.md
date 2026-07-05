<!-- AUTO-GENERATED: nonh0_pgd_vs_no_pgd_heldout_velocity -->
## Non-H0 PGD vs no-PGD held-out velocity

Source rows:
- no-PGD: `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`
- PGD: `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64`

GRU curves use validation-selected checkpoints, 64 stochastic repeats per target condition per replicate, and target-radial effector velocity divided by reach length. extLQG references are stochastic forward velocity for the 15 cm released C&S reference divided by 0.15 m.

Result: the non-H0 PGD row largely removes the held-out direction/length degradation visible in the paired non-H0 no-PGD row. The no-PGD held-out peak drops by 0.5500 1/s and its RMSE vs 8D extLQG rises from 0.0156 to 0.2620 1/s. The PGD held-out peak differs from its seen peak by 0.0236 1/s, with RMSE moving from 0.1622 to 0.1309 1/s.

Runtime provenance: generated with current Feedbax commit `4642e20f` plus the same scoped non-H0 dtype/shape compatibility patches used by the no-PGD diagnostic.

| Row | Split | Peak (1/s) | Late mean (1/s) | Mean band (1/s) | Band ratio | RMSE vs 8D extLQG (1/s) | Shape corr vs 8D |
|---|---|---:|---:|---:|---:|---:|---:|
| `no_pgd` | `seen direction + seen length` | 4.8881 | 0.0054 | 0.0446 | -- | 0.0156 | 1.0000 |
| `no_pgd` | `held-out direction/length` | 4.3381 | 0.0926 | 0.1207 | 2.71 | 0.2620 | 0.9989 |
| `no_pgd` | `all validation targets` | 4.6681 | 0.0403 | 0.1335 | 2.99 | 0.1033 | 0.9998 |
| `pgd` | `seen direction + seen length` | 5.1996 | 0.0015 | 0.0731 | -- | 0.1622 | 0.9992 |
| `pgd` | `held-out direction/length` | 5.1759 | -0.0001 | 0.0921 | 1.26 | 0.1309 | 0.9996 |
| `pgd` | `all validation targets` | 5.1901 | 0.0009 | 0.0837 | 1.15 | 0.1491 | 0.9994 |

- Figure: `_artifacts/e901a20/figures/nonh0_pgd_vs_no_pgd_heldout_velocity/nonh0_pgd_vs_no_pgd_heldout_velocity.html`
- Data: `_artifacts/e901a20/figures/nonh0_pgd_vs_no_pgd_heldout_velocity/nonh0_pgd_vs_no_pgd_heldout_velocity.npz`
- Summary CSV: `_artifacts/e901a20/figures/nonh0_pgd_vs_no_pgd_heldout_velocity/nonh0_pgd_vs_no_pgd_heldout_velocity_summary.csv`
- Manifest: `_artifacts/e901a20/figures/nonh0_pgd_vs_no_pgd_heldout_velocity/manifest.json`
<!-- /AUTO-GENERATED -->
