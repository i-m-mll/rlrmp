# GRU Worst-Case Epsilon Audit

Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.

| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | evaluated | 0.0012324306 | 4388.7899 | 5236.7687 | 847.97878 | 4419.696 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | evaluated | 0.0012324306 | 4755.9394 | 5055.925 | 299.98558 | 4764.0119 |

Limits: this is open-loop projected ascent over one declared `T x 8` epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke one b8aa38e row; pass explicit `--run-id` values for a broader matrix.
