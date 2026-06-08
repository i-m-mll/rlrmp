# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `020a65b`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64__nominal_clean | 0.202715 | -0.0102115 | -0.00207003 | 0.0410889 | 1858.33 | available | low_norm |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__pgd_moderate_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1858.33 | 0.120532 | 1 | vx | 35 | ux | 0.0356751 |
| 2 | 1857.41 | 0.120414 | 1 | vy | 35 | uy | 0.00724087 |
| 3 | 1753.29 | 0.107292 | 4 | vy | 38 | uy | 0.11561 |
| 4 | 1752.88 | 0.107242 | 4 | vx | 38 | ux | 0.0781369 |
| 5 | 903.696 | 0.0285038 | 0 | vx | 51 | ux | 0.0197648 |
