# GRU Worst-Case Epsilon Audit

Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.

| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |
|---|---:|---:|---:|---:|---:|---:|
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | evaluated | 0.0023284906 | 4607.0831 | 5587.5551 | 980.47198 | 4630.8992 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | evaluated | 0.0045455001 | 4767.3072 | 7883.0731 | 3115.7659 | 4817.5651 |

Limits: this is open-loop projected ascent over one declared `T x 8` epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke one b8aa38e row; pass explicit `--run-id` values for a broader matrix.
