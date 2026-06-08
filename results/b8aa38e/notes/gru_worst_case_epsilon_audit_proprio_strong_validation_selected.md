# GRU Worst-Case Epsilon Audit

Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.

| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4659.8982 | 6889.6831 | 2229.7849 | 4729.7763 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4642.4971 | 6854.1796 | 2211.6825 | 4709.1561 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4629.0014 | 6784.5565 | 2155.555 | 4694.0551 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4664.9001 | 6893.589 | 2228.6889 | 4730.286 |

Limits: this is open-loop projected ascent over one declared `T x 8` epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke one b8aa38e row; pass explicit `--run-id` values for a broader matrix.
