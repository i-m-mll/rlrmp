<!-- AUTO-GENERATED: nonh0_no_pgd_extlqg_heldout_velocity -->
## Non-H0 no-PGD held-out velocity vs extLQG

Source row: `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`. This is the non-H0 020a65b no-PGD GRU row.

GRU curves use validation-selected checkpoints, 64 stochastic repeats per target condition per replicate, and target-radial effector velocity divided by reach length. extLQG references are stochastic forward velocity for the 15 cm released C&S reference divided by 0.15 m.

The 8D extLQG output-feedback curve is the primary reference; the 4D pos+vel reference is included as a sidecar because the GRU observes 6D target-relative force/filter feedback rather than exactly either analytical observation channel.

Runtime provenance: generated with `current Python import path`.

Compatibility caveat: this was generated with current Feedbax commit `4642e20f` plus scoped dtype/shape patches for the non-H0 legacy checkpoint. The exact Feedbax `3add27d7` runtime used for the H0 MaskedLinear old-compatible figures was attempted separately but blocked before evaluation because the current RLRMP graph builder passes `dtype` to the older `SimpleStagedNetwork` constructor.

| Split | Peak (1/s) | Late mean (1/s) | Mean band (1/s) | Band ratio | RMSE vs 8D extLQG (1/s) | Shape corr vs 8D |
|---|---:|---:|---:|---:|---:|---:|
| `seen direction + seen length` | 4.8881 | 0.0054 | 0.0446 | -- | 0.0156 | 1.0000 |
| `held-out direction/length` | 4.3381 | 0.0926 | 0.1207 | 2.71 | 0.2620 | 0.9989 |
| `all validation targets` | 4.6681 | 0.0403 | 0.1335 | 2.99 | 0.1033 | 0.9998 |

- Figure: `_artifacts/e901a20/figures/nonh0_no_pgd_extlqg_heldout_velocity/nonh0_no_pgd_extlqg_heldout_velocity.html`
- Data: `_artifacts/e901a20/figures/nonh0_no_pgd_extlqg_heldout_velocity/nonh0_no_pgd_extlqg_heldout_velocity.npz`
- Summary CSV: `_artifacts/e901a20/figures/nonh0_no_pgd_extlqg_heldout_velocity/nonh0_no_pgd_extlqg_heldout_velocity_summary.csv`
- Manifest: `_artifacts/e901a20/figures/nonh0_no_pgd_extlqg_heldout_velocity/manifest.json`
<!-- /AUTO-GENERATED -->
