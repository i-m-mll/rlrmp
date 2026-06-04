# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `ba82f3d`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean | 0.677121 | -0.30817 | -0.208668 | 0.41495 | 4193.33 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean | 1.07005 | -0.0937121 | -0.100277 | 1.13496 | 4200.09 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 4193.33 | 0.124635 | 5 | px | 6 | ux | 1.49294e-05 |
| 2 | 4156.61 | 0.122462 | 5 | py | 6 | uy | 1.44895e-05 |
| 3 | 3705.95 | 0.0973471 | 8 | vx | 35 | ux | 1.49087e-05 |
| 4 | 3659.86 | 0.0949408 | 8 | vy | 35 | uy | 1.69282e-05 |
| 5 | 2242.57 | 0.0356464 | 5 | px | 6 | ux | 1.91688e-05 |

### `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 4200.09 | 0.0999966 | 5 | px | 6 | ux | 1.78204e-05 |
| 2 | 4103.43 | 0.0954471 | 5 | py | 6 | uy | 1.60554e-05 |
| 3 | 3704.46 | 0.0777888 | 8 | vx | 35 | ux | 2.01495e-05 |
| 4 | 3592.22 | 0.0731464 | 8 | vy | 35 | uy | 2.05923e-05 |
| 5 | 2243.52 | 0.0285316 | 5 | px | 6 | ux | 2.267e-05 |
