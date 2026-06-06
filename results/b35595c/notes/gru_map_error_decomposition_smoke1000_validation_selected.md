# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b35595c`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean | 0.14557 | -0.000319736 | -4.65438e-05 | 0.0211905 | 3492.89 | available | low_norm, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean | 0.371175 | 0.0386743 | 0.014355 | 0.137565 | 3529.24 | available | low_norm, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3492.89 | 0.158833 | 5 | px | 6 | ux | 3.79265e-05 |
| 2 | 3481.93 | 0.157837 | 5 | py | 6 | uy | 3.92496e-05 |
| 3 | 2930.47 | 0.111801 | 8 | vx | 35 | ux | 5.74923e-05 |
| 4 | 2918.7 | 0.110904 | 8 | vy | 35 | uy | 6.91508e-05 |
| 5 | 1641.04 | 0.0350597 | 5 | py | 6 | uy | 3.59311e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3529.24 | 0.149322 | 5 | px | 6 | ux | 3.07562e-05 |
| 2 | 3501.6 | 0.146992 | 5 | py | 6 | uy | 1.93236e-05 |
| 3 | 3027.3 | 0.109868 | 8 | vy | 35 | uy | 4.67572e-05 |
| 4 | 2988.08 | 0.10704 | 8 | vx | 35 | ux | 3.92116e-05 |
| 5 | 1820.49 | 0.0397316 | 5 | px | 6 | ux | 3.0784e-05 |
