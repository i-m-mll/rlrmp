# GRU Worst-Case Epsilon Audit

Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.

| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4610.357 | 5423.4259 | 813.06888 | 4646.1852 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4653.2309 | 5512.5229 | 859.292 | 4685.8785 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4653.3262 | 5497.9624 | 844.63619 | 4684.5605 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | evaluated | 0.0012324306 | 4667.2302 | 5459.5593 | 792.32905 | 4697.6654 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4616.526 | 6733.8626 | 2117.3366 | 4688.2422 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4656.914 | 6869.5631 | 2212.6491 | 4722.3558 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4650.7825 | 6829.7882 | 2179.0057 | 4715.6787 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | evaluated | 0.0023284907 | 4681.7462 | 6844.8283 | 2163.0821 | 4741.361 |

Limits: this is open-loop projected ascent over one declared `T x 8` epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke one b8aa38e row; pass explicit `--run-id` values for a broader matrix.
