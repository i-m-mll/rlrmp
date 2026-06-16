# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `020a65b`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64__nominal_clean | 0.428053 | 0.140852 | 0.060292 | 0.179594 | 3361.45 | available | low_norm, unexcited_directions |
| target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64__nominal_clean | 1.96377 | 0.150584 | 0.295713 | 3.76894 | 7106.6 | available | unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3361.45 | 0.141378 | 5 | px | 6 | ux | 2.14428e-05 |
| 2 | 3355.45 | 0.140874 | 5 | py | 6 | uy | 1.51693e-05 |
| 3 | 2783.58 | 0.0969471 | 8 | vx | 35 | ux | 1.87811e-05 |
| 4 | 2777.06 | 0.0964934 | 8 | vy | 35 | uy | 2.02816e-05 |
| 5 | 1590.02 | 0.0316326 | 5 | px | 6 | ux | 2.32121e-05 |

### `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 7106.6 | 0.157444 | 31 | py | 40 | uy | 1.34809e-05 |
| 2 | 7046.68 | 0.1548 | 29 | py | 38 | uy | 2.24858e-05 |
| 3 | 6909.1 | 0.148814 | 30 | px | 39 | ux | 1.59116e-05 |
| 4 | 6882.12 | 0.147654 | 28 | px | 41 | ux | 1.7154e-05 |
| 5 | 3513.58 | 0.038486 | 5 | py | 31 | uy | 1.90179e-05 |
