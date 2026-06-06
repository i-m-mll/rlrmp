# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b35595c`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean | 0.588954 | 0.264486 | 0.15577 | 0.322603 | 3184.11 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64__nominal_clean | 0.707947 | 0.181619 | 0.128576 | 0.484658 | 3193.55 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64__nominal_clean | 0.667868 | 0.252411 | 0.168577 | 0.417629 | 3228.24 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean | 0.727262 | 0.26104 | 0.189845 | 0.49287 | 3113.52 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64__nominal_clean | 0.789444 | 0.282849 | 0.223294 | 0.573361 | 3245.98 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64__nominal_clean | 0.93333 | 0.212548 | 0.198377 | 0.831751 | 3142.72 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3184.11 | 0.130201 | 5 | py | 6 | uy | 1.70159e-05 |
| 2 | 3174.22 | 0.129394 | 5 | px | 6 | ux | 3.09807e-05 |
| 3 | 2630.26 | 0.0888457 | 8 | vy | 34 | uy | 2.80458e-05 |
| 4 | 2612.7 | 0.087663 | 8 | vx | 34 | ux | 2.58967e-05 |
| 5 | 1556.3 | 0.0311048 | 5 | py | 6 | uy | 2.56371e-05 |

### `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3193.55 | 0.109001 | 5 | py | 6 | uy | 1.79139e-05 |
| 2 | 3162.16 | 0.106868 | 5 | px | 6 | ux | 3.01002e-05 |
| 3 | 2637.78 | 0.0743639 | 8 | vy | 34 | uy | 2.79417e-05 |
| 4 | 2591.61 | 0.0717834 | 8 | vx | 34 | ux | 2.45093e-05 |
| 5 | 1591.61 | 0.0270743 | 5 | py | 6 | uy | 2.42438e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3228.24 | 0.124956 | 5 | py | 6 | uy | 1.75816e-05 |
| 2 | 3214.73 | 0.123913 | 5 | px | 6 | ux | 2.33512e-05 |
| 3 | 2701.23 | 0.0874883 | 8 | vy | 34 | uy | 2.27599e-05 |
| 4 | 2691.55 | 0.0868623 | 8 | vx | 34 | ux | 2.30776e-05 |
| 5 | 1606.89 | 0.0309597 | 5 | px | 6 | ux | 2.50396e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3113.52 | 0.112155 | 5 | py | 6 | uy | 1.49863e-05 |
| 2 | 3080.16 | 0.109763 | 5 | px | 6 | ux | 2.12634e-05 |
| 3 | 2544.2 | 0.0748885 | 8 | vy | 34 | uy | 2.06075e-05 |
| 4 | 2493.21 | 0.0719168 | 8 | vx | 34 | ux | 2.14446e-05 |
| 5 | 1537.05 | 0.0273332 | 5 | py | 6 | uy | 1.97209e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3245.98 | 0.11906 | 5 | py | 6 | uy | 1.78014e-05 |
| 2 | 3240.35 | 0.118647 | 5 | px | 6 | ux | 2.87501e-05 |
| 3 | 2768.07 | 0.0865822 | 8 | vx | 34 | ux | 2.606e-05 |
| 4 | 2763.07 | 0.08627 | 8 | vy | 34 | uy | 2.17558e-05 |
| 5 | 1690.98 | 0.0323111 | 5 | px | 6 | ux | 1.75876e-05 |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3142.72 | 0.0890689 | 5 | py | 6 | uy | 1.50751e-05 |
| 2 | 3118.43 | 0.0876978 | 5 | px | 6 | ux | 2.03957e-05 |
| 3 | 2585.5 | 0.0602843 | 8 | vy | 34 | uy | 2.25192e-05 |
| 4 | 2551.56 | 0.0587119 | 8 | vx | 34 | ux | 2.13534e-05 |
| 5 | 1635.31 | 0.0241166 | 5 | py | 6 | uy | 1.89196e-05 |
