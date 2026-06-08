# GRU Worst-Case Epsilon Audit

Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.

| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | evaluated | 0.0045455003 | 4625.4296 | 11546.269 | 6920.8392 | 4790.6241 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | evaluated | 0.0045455003 | 5585.5008 | 9692.4656 | 4106.9648 | 5658.2055 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | evaluated | 0.0045455003 | 4418.9576 | 10592.697 | 6173.7392 | 4558.8134 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | evaluated | 0.0045455003 | 4742.781 | 7874.8856 | 3132.1046 | 4814.0595 |

Limits: this is open-loop projected ascent over one declared `T x 8` epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke one b8aa38e row; pass explicit `--run-id` values for a broader matrix.
