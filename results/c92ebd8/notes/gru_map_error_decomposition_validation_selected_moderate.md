# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `c92ebd8`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| open_loop_small__nominal_clean | 0.332403 | 0.103047 | 0.034253 | 0.109318 | 3347.33 | available | low_norm, unexcited_directions |
| open_loop_moderate__nominal_clean | 0.445897 | 0.127715 | 0.0569479 | 0.195581 | 3298.4 | available | low_norm, unexcited_directions |
| open_loop_stress__nominal_clean | 14.1171 | 0.0089086 | 0.125764 | 199.278 | 71640.7 | available | unexcited_directions |
| closed_loop_small__nominal_clean | 0.261089 | 0.0791104 | 0.0206548 | 0.0677407 | 3431.13 | available | low_norm, unexcited_directions |
| closed_loop_moderate__nominal_clean | 0.382235 | 0.116512 | 0.0445349 | 0.14412 | 3377.28 | available | low_norm, unexcited_directions |
| closed_loop_stress__nominal_clean | 0.83085 | 0.174055 | 0.144614 | 0.669398 | 3271.41 | available | wrong_timing_or_channel, unexcited_directions |
| closed_loop_cmd_lateral_small__nominal_clean | 0.338302 | 0.101249 | 0.0342529 | 0.113275 | 3411.42 | available | low_norm, unexcited_directions |
| closed_loop_cmd_lateral_moderate__nominal_clean | 0.362617 | 0.104059 | 0.0377336 | 0.130068 | 3382.64 | available | low_norm, unexcited_directions |
| closed_loop_cmd_lateral_stress__nominal_clean | 1.41944 | 0.109343 | 0.155206 | 1.99072 | 7931.38 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `open_loop_small__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3347.33 | 0.142972 | 5 | px | 6 | ux | 2.79017e-05 |
| 2 | 3337.58 | 0.142141 | 5 | py | 6 | uy | 1.67404e-05 |
| 3 | 2781.98 | 0.0987558 | 8 | vx | 35 | ux | 1.8493e-05 |
| 4 | 2769.78 | 0.0978915 | 8 | vy | 35 | uy | 1.93813e-05 |
| 5 | 1635.71 | 0.0341405 | 0 | py | 0 | uy | 2.69387e-05 |

### `open_loop_moderate__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3298.4 | 0.133328 | 5 | px | 6 | ux | 2.67334e-05 |
| 2 | 3287.81 | 0.132473 | 5 | py | 6 | uy | 1.55672e-05 |
| 3 | 2730.58 | 0.0913742 | 8 | vx | 34 | ux | 1.61511e-05 |
| 4 | 2720.83 | 0.0907233 | 8 | vy | 34 | uy | 1.91378e-05 |
| 5 | 1583.98 | 0.0307479 | 0 | py | 0 | uy | 2.12591e-05 |

### `open_loop_stress__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 71640.7 | 0.341126 | 18 | py | 59 | uy | 1.6848e-05 |
| 2 | 63142 | 0.264991 | 20 | py | 57 | uy | 2.57264e-05 |
| 3 | 52281.3 | 0.181672 | 18 | px | 59 | ux | 3.03945e-05 |
| 4 | 45806.6 | 0.139461 | 20 | px | 57 | ux | 2.51185e-05 |
| 5 | 16523.4 | 0.0181464 | 18 | py | 59 | uy | 2.10873e-05 |

### `closed_loop_small__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3431.13 | 0.152433 | 5 | px | 6 | ux | 2.83775e-05 |
| 2 | 3426.83 | 0.152052 | 5 | py | 6 | uy | 1.92531e-05 |
| 3 | 2870.34 | 0.106677 | 8 | vy | 35 | uy | 2.55371e-05 |
| 4 | 2867.23 | 0.106447 | 8 | vx | 35 | ux | 1.801e-05 |
| 5 | 2032.49 | 0.0534888 | 0 | py | 0 | uy | 4.28784e-05 |

### `closed_loop_moderate__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3377.28 | 0.14347 | 5 | py | 6 | uy | 1.83354e-05 |
| 2 | 3374.22 | 0.14321 | 5 | px | 6 | ux | 2.75289e-05 |
| 3 | 2820.86 | 0.10009 | 8 | vx | 34 | ux | 1.91146e-05 |
| 4 | 2813.62 | 0.0995766 | 8 | vy | 34 | uy | 2.38251e-05 |
| 5 | 1693.22 | 0.0360626 | 0 | py | 0 | uy | 2.93622e-05 |

### `closed_loop_stress__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3271.41 | 0.10156 | 5 | py | 32 | uy | 1.76939e-05 |
| 2 | 3251.68 | 0.100339 | 5 | px | 31 | ux | 2.69912e-05 |
| 3 | 2972.65 | 0.0838569 | 42 | vx | 56 | ux | 3.31584e-05 |
| 4 | 2846.59 | 0.0768959 | 40 | vx | 48 | ux | 2.04685e-05 |
| 5 | 2785.02 | 0.0736051 | 8 | vy | 35 | uy | 2.75322e-05 |

### `closed_loop_cmd_lateral_small__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3411.42 | 0.147938 | 5 | px | 6 | ux | 2.98142e-05 |
| 2 | 3405.13 | 0.147393 | 5 | py | 6 | uy | 1.80065e-05 |
| 3 | 2849.34 | 0.103204 | 8 | vx | 35 | ux | 1.98813e-05 |
| 4 | 2845.37 | 0.102916 | 8 | vy | 35 | uy | 2.25207e-05 |
| 5 | 2478.61 | 0.0780956 | 0 | py | 0 | uy | 3.11136e-05 |

### `closed_loop_cmd_lateral_moderate__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3382.64 | 0.144064 | 5 | px | 6 | ux | 2.7109e-05 |
| 2 | 3378.16 | 0.143682 | 5 | py | 6 | uy | 2.17235e-05 |
| 3 | 2836.36 | 0.101289 | 8 | vy | 35 | uy | 1.73465e-05 |
| 4 | 2827.14 | 0.100632 | 8 | vx | 35 | ux | 1.92158e-05 |
| 5 | 2537.37 | 0.0810609 | 0 | py | 0 | uy | 2.35333e-05 |

### `closed_loop_cmd_lateral_stress__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 7931.38 | 0.309274 | 33 | px | 59 | uy | 2.21431e-05 |
| 2 | 6846.34 | 0.230443 | 32 | px | 57 | uy | 2.36151e-05 |
| 3 | 3303.93 | 0.053667 | 5 | py | 32 | uy | 2.12861e-05 |
| 4 | 3272.9 | 0.0526638 | 5 | px | 6 | ux | 3.12935e-05 |
| 5 | 2803.34 | 0.0386366 | 8 | vy | 22 | uy | 2.7547e-05 |
