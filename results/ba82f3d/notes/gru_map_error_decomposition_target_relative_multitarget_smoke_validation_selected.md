# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `ba82f3d`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_smoke__nominal_clean | 0.00235768 | 0.116398 | 0.000274429 | 5.48334e-06 | 1555.15 | available | low_norm, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_smoke__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1555.15 | 0.160866 | 5 | py | 6 | uy | 6.10428e-06 |
| 2 | 1555.05 | 0.160846 | 5 | px | 6 | ux | 1.30979e-05 |
| 3 | 1302.01 | 0.112759 | 8 | vy | 35 | uy | 8.6924e-06 |
| 4 | 1301.82 | 0.112726 | 8 | vx | 35 | ux | 1.98241e-05 |
| 5 | 726.231 | 0.035081 | 5 | px | 6 | ux | 2.16162e-05 |
