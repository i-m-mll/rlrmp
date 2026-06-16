# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `020a65b`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000__nominal_clean | 0.123345 | 0.00506122 | 0.000624279 | 0.0152137 | 1849.85 | available | low_norm |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1849.85 | 0.123118 | 1 | vy | 35 | uy | 0.0194029 |
| 2 | 1846.84 | 0.122718 | 1 | vx | 35 | ux | 0.00471236 |
| 3 | 1741.73 | 0.109147 | 5 | vy | 38 | uy | 0.00722697 |
| 4 | 1741.53 | 0.109121 | 5 | vx | 38 | ux | 0.0618063 |
| 5 | 867.062 | 0.0270488 | 0 | vy | 51 | uy | 0.00927681 |
