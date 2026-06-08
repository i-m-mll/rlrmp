# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b8aa38e`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| smoke__broad_strong_cal_small__nominal_clean | 0.00105296 | 0.0319685 | 3.36614e-05 | 1.10758e-06 | 1555.42 | available | low_norm, unexcited_directions |
| smoke__proprio_cal_stress__nominal_clean | 0.00257116 | -0.00605136 | -1.55591e-05 | 6.61064e-06 | 826.455 | available | low_norm, unexcited_directions |

## Top Singular Directions

### `smoke__broad_strong_cal_small__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1555.42 | 0.160846 | 5 | py | 6 | uy | 6.46797e-06 |
| 2 | 1555.24 | 0.160808 | 5 | px | 6 | ux | 1.28022e-05 |
| 3 | 1302.31 | 0.112758 | 8 | vx | 35 | ux | 8.6495e-06 |
| 4 | 1302.09 | 0.112719 | 8 | vy | 35 | uy | 1.98325e-05 |
| 5 | 726.289 | 0.0350698 | 5 | py | 6 | uy | 2.30409e-05 |

### `smoke__proprio_cal_stress__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 826.455 | 0.124584 | 1 | vy | 35 | uy | 2.6327e-05 |
| 2 | 826.424 | 0.124575 | 1 | vx | 35 | ux | 2.1232e-05 |
| 3 | 778.696 | 0.110602 | 5 | vy | 38 | uy | 2.22562e-05 |
| 4 | 778.627 | 0.110582 | 5 | vx | 38 | ux | 3.0227e-05 |
| 5 | 386.626 | 0.0272651 | 25 | vy | 51 | uy | 4.04297e-05 |
