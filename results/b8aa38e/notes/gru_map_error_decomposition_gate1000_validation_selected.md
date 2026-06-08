# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b8aa38e`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean | 0.162806 | 0.00356506 | 0.000580412 | 0.0265054 | 3493.59 | available | low_norm, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean | 0.128388 | 0.0159764 | 0.00205118 | 0.0164792 | 1850.87 | available | low_norm, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3493.59 | 0.158267 | 5 | px | 6 | ux | 2.45393e-05 |
| 2 | 3479.48 | 0.156991 | 5 | py | 6 | uy | 2.58963e-05 |
| 3 | 2928.8 | 0.111231 | 8 | vx | 35 | ux | 3.95055e-05 |
| 4 | 2917.29 | 0.110359 | 8 | vy | 35 | uy | 4.9626e-05 |
| 5 | 1649.27 | 0.0352718 | 5 | py | 6 | uy | 5.24898e-05 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1850.87 | 0.123447 | 1 | vy | 35 | uy | 0.0248769 |
| 2 | 1846.7 | 0.122891 | 1 | vx | 35 | ux | 0.0824595 |
| 3 | 1741.98 | 0.109349 | 4 | vy | 38 | uy | 0.0219241 |
| 4 | 1741.82 | 0.109329 | 4 | vx | 38 | ux | 0.048929 |
| 5 | 867.519 | 0.0271197 | 1 | vy | 51 | uy | 0.00928881 |
