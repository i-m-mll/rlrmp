# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `ae9f30f`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| direct_epsilon_b1p05__nominal_clean | 0.631415 | 0.159694 | 0.100833 | 0.388518 | 3244.26 | available | wrong_timing_or_channel, unexcited_directions |
| direct_epsilon_b1p4__nominal_clean | 0.557853 | 0.152074 | 0.0848348 | 0.304003 | 3259.73 | available | wrong_timing_or_channel, unexcited_directions |
| linear_no_bias_b1p05__nominal_clean | 0.861945 | 0.114563 | 0.0987472 | 0.733197 | 3381.73 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `direct_epsilon_b1p05__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3244.26 | 0.116909 | 5 | px | 6 | ux | 2.52215e-05 |
| 2 | 3227.25 | 0.115686 | 5 | py | 6 | uy | 1.55645e-05 |
| 3 | 2678.38 | 0.0796818 | 8 | vx | 34 | ux | 1.6628e-05 |
| 4 | 2667.08 | 0.0790108 | 8 | vy | 34 | uy | 2.01465e-05 |
| 5 | 1580.15 | 0.0277341 | 5 | px | 6 | ux | 1.68125e-05 |

### `direct_epsilon_b1p4__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3259.73 | 0.123763 | 5 | px | 6 | ux | 2.51241e-05 |
| 2 | 3247.48 | 0.122835 | 5 | py | 6 | uy | 1.52126e-05 |
| 3 | 2693.48 | 0.0844997 | 8 | vx | 34 | ux | 1.64648e-05 |
| 4 | 2685.12 | 0.0839761 | 8 | vy | 34 | uy | 1.91587e-05 |
| 5 | 1564.7 | 0.0285161 | 5 | px | 6 | ux | 1.73546e-05 |

### `linear_no_bias_b1p05__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3381.73 | 0.0983873 | 37 | px | 59 | ux | 2.43949e-05 |
| 2 | 3319.23 | 0.0947838 | 5 | px | 6 | ux | 2.66555e-05 |
| 3 | 3293.69 | 0.093331 | 5 | py | 6 | uy | 1.72457e-05 |
| 4 | 3042.42 | 0.0796339 | 39 | px | 57 | ux | 2.69645e-05 |
| 5 | 2936.11 | 0.074166 | 40 | py | 59 | uy | 2.26251e-05 |
