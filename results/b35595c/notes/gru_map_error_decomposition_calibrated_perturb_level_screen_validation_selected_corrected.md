# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b35595c`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean | 0.621665 | 0.239233 | 0.148723 | 0.364348 | 3223.15 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64__nominal_clean | 0.832606 | 0.14772 | 0.122993 | 0.678106 | 3246.64 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64__nominal_clean | 0.680306 | 0.239449 | 0.162899 | 0.43628 | 3237.34 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean | 0.839718 | 0.187425 | 0.157384 | 0.680356 | 3187.92 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64__nominal_clean | 0.807545 | 0.263584 | 0.212856 | 0.606821 | 3260.15 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64__nominal_clean | 0.868284 | 0.241754 | 0.209911 | 0.709855 | 3115.83 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3223.15 | 0.126836 | 5 | py | 6 | uy | 1.77726e-05 |
| 2 | 3201.6 | 0.125145 | 5 | px | 6 | ux | 2.88395e-05 |
| 3 | 2687.89 | 0.0882067 | 8 | vy | 34 | uy | 2.86923e-05 |
| 4 | 2662.65 | 0.0865583 | 8 | vx | 34 | ux | 2.70284e-05 |
| 5 | 1591.83 | 0.0309367 | 5 | py | 6 | uy | 2.41816e-05 |

### `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3246.64 | 0.0968372 | 5 | py | 6 | uy | 2.03854e-05 |
| 2 | 3216.49 | 0.0950468 | 5 | px | 6 | ux | 2.91715e-05 |
| 3 | 2711.52 | 0.0675459 | 8 | vy | 34 | uy | 3.07136e-05 |
| 4 | 2680.71 | 0.0660194 | 8 | vx | 34 | ux | 2.64206e-05 |
| 5 | 1662.16 | 0.0253817 | 5 | py | 6 | uy | 2.67472e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3237.34 | 0.122554 | 5 | py | 6 | uy | 1.70632e-05 |
| 2 | 3223.61 | 0.121516 | 5 | px | 6 | ux | 2.25166e-05 |
| 3 | 2712.91 | 0.0860638 | 8 | vy | 34 | uy | 2.27184e-05 |
| 4 | 2704.92 | 0.0855575 | 8 | vx | 34 | ux | 2.36346e-05 |
| 5 | 1615.39 | 0.0305144 | 5 | px | 6 | ux | 2.46916e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3187.92 | 0.0971863 | 5 | py | 6 | uy | 1.67788e-05 |
| 2 | 3170.63 | 0.096135 | 5 | px | 6 | ux | 2.20538e-05 |
| 3 | 2642.4 | 0.0667711 | 8 | vy | 34 | uy | 2.16067e-05 |
| 4 | 2615.08 | 0.0653974 | 8 | vx | 34 | ux | 2.28617e-05 |
| 5 | 1623.47 | 0.0252046 | 5 | py | 6 | uy | 1.9794e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3260.15 | 0.115227 | 5 | py | 6 | uy | 1.76357e-05 |
| 2 | 3259.78 | 0.1152 | 5 | px | 6 | ux | 2.37653e-05 |
| 3 | 2789.61 | 0.0843659 | 8 | vx | 34 | ux | 2.71019e-05 |
| 4 | 2777.87 | 0.0836568 | 8 | vy | 34 | uy | 2.22848e-05 |
| 5 | 1699.56 | 0.031315 | 5 | px | 6 | ux | 1.79369e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3115.83 | 0.0967557 | 5 | py | 6 | uy | 1.54013e-05 |
| 2 | 3091.49 | 0.0952503 | 5 | px | 6 | ux | 1.82442e-05 |
| 3 | 2549.21 | 0.0647648 | 8 | vy | 34 | uy | 2.16902e-05 |
| 4 | 2514.27 | 0.0630019 | 8 | vx | 34 | ux | 2.14648e-05 |
| 5 | 1589.77 | 0.0251883 | 5 | py | 6 | uy | 1.86801e-05 |
