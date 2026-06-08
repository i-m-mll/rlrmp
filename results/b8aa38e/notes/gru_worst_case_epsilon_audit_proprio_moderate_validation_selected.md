# GRU Worst-Case Epsilon Audit

Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.

| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4659.8982 | 5494.0094 | 834.11122 | 4694.5204 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4642.4971 | 5483.2316 | 840.73458 | 4675.4496 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4629.0014 | 5454.0479 | 825.04652 | 4661.1444 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4664.9001 | 5544.0689 | 879.16882 | 4697.3199 |

Limits: this is open-loop projected ascent over one declared `T x 8` epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke one b8aa38e row; pass explicit `--run-id` values for a broader matrix.
