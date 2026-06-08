# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b8aa38e`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64__nominal_clean | 0.624886 | 0.233782 | 0.146087 | 0.369141 | 3235.79 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64__nominal_clean | 0.682374 | 0.240051 | 0.163804 | 0.438803 | 3232.59 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64__nominal_clean | 0.808199 | 0.26603 | 0.215005 | 0.606958 | 3255.75 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64__nominal_clean | 1.31688 | 0.265412 | 0.349517 | 1.61202 | 3634.43 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64__nominal_clean | 0.621284 | 0.242213 | 0.150483 | 0.363348 | 3229.14 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64__nominal_clean | 0.683709 | 0.242234 | 0.165617 | 0.440029 | 3234.81 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64__nominal_clean | 0.808859 | 0.266427 | 0.215502 | 0.607812 | 3252.31 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean | 1.33168 | 0.268282 | 0.357266 | 1.64573 | 3666 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64__nominal_clean | 0.231806 | -0.0110758 | -0.00256743 | 0.0537274 | 1856.7 | available | low_norm |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64__nominal_clean | 0.284822 | -0.0207282 | -0.00590384 | 0.0810887 | 1861.18 | available | low_norm |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64__nominal_clean | 0.357945 | -0.0168431 | -0.00602889 | 0.128089 | 1864.27 | available | low_norm |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean | 0.907932 | 0.000857452 | 0.000778508 | 0.824341 | 2708.12 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3235.79 | 0.126751 | 5 | py | 6 | uy | 1.80988e-05 |
| 2 | 3210.92 | 0.124811 | 5 | px | 6 | ux | 2.92548e-05 |
| 3 | 2698.66 | 0.0881632 | 8 | vy | 34 | uy | 2.88697e-05 |
| 4 | 2673.74 | 0.0865426 | 8 | vx | 34 | ux | 2.74875e-05 |
| 5 | 1592.49 | 0.0307006 | 5 | py | 6 | uy | 2.41723e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3232.59 | 0.122086 | 5 | py | 6 | uy | 1.8062e-05 |
| 2 | 3218.34 | 0.121012 | 5 | px | 6 | ux | 2.16402e-05 |
| 3 | 2713.47 | 0.0860228 | 8 | vy | 34 | uy | 2.31202e-05 |
| 4 | 2700.16 | 0.0851808 | 8 | vx | 34 | ux | 2.31251e-05 |
| 5 | 1613.46 | 0.0304146 | 5 | px | 6 | ux | 1.80469e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3255.75 | 0.11522 | 5 | py | 6 | uy | 1.77688e-05 |
| 2 | 3253.49 | 0.11506 | 5 | px | 6 | ux | 2.30315e-05 |
| 3 | 2789.16 | 0.0845618 | 8 | vx | 34 | ux | 2.41041e-05 |
| 4 | 2779.28 | 0.0839637 | 8 | vy | 34 | uy | 2.36384e-05 |
| 5 | 1695.75 | 0.0312573 | 5 | px | 6 | ux | 1.75136e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3634.43 | 0.0862965 | 5 | px | 44 | ux | 1.86113e-05 |
| 2 | 3556.02 | 0.0826131 | 5 | py | 44 | uy | 1.60995e-05 |
| 3 | 3447.97 | 0.0776686 | 42 | px | 58 | ux | 1.87544e-05 |
| 4 | 3332.61 | 0.0725586 | 42 | py | 58 | uy | 1.49987e-05 |
| 5 | 2461.19 | 0.0395741 | 5 | py | 59 | uy | 1.38354e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3229.14 | 0.127776 | 5 | py | 6 | uy | 1.7838e-05 |
| 2 | 3207.68 | 0.126084 | 5 | px | 6 | ux | 2.91755e-05 |
| 3 | 2691.01 | 0.0887371 | 8 | vy | 34 | uy | 2.88674e-05 |
| 4 | 2667.94 | 0.0872221 | 8 | vx | 34 | ux | 2.82482e-05 |
| 5 | 1586.48 | 0.030842 | 5 | py | 6 | uy | 2.46842e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3234.81 | 0.122447 | 5 | py | 6 | uy | 1.78423e-05 |
| 2 | 3214.36 | 0.120904 | 5 | px | 6 | ux | 2.20382e-05 |
| 3 | 2715.06 | 0.0862605 | 8 | vy | 34 | uy | 2.28161e-05 |
| 4 | 2696.11 | 0.0850607 | 8 | vx | 34 | ux | 2.34108e-05 |
| 5 | 1615.25 | 0.0305304 | 5 | py | 6 | uy | 2.27658e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3252.31 | 0.11497 | 5 | py | 6 | uy | 1.68913e-05 |
| 2 | 3250.09 | 0.114813 | 5 | px | 6 | ux | 2.41739e-05 |
| 3 | 2784.5 | 0.0842745 | 8 | vx | 34 | ux | 2.41472e-05 |
| 4 | 2774.86 | 0.083692 | 8 | vy | 34 | uy | 2.30666e-05 |
| 5 | 1698.58 | 0.0313597 | 5 | px | 6 | ux | 1.76249e-05 |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3666 | 0.0867922 | 5 | px | 56 | ux | 1.83467e-05 |
| 2 | 3563.98 | 0.0820285 | 5 | py | 44 | uy | 1.66649e-05 |
| 3 | 3535.89 | 0.0807408 | 42 | px | 58 | ux | 1.81723e-05 |
| 4 | 3375.67 | 0.0735892 | 42 | py | 58 | uy | 1.5667e-05 |
| 5 | 2502.71 | 0.0404497 | 5 | px | 59 | ux | 1.4421e-05 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1856.7 | 0.118771 | 1 | vx | 35 | ux | 0.0218561 |
| 2 | 1855.43 | 0.118609 | 1 | vy | 35 | uy | 0.00543229 |
| 3 | 1752.62 | 0.105829 | 4 | vx | 38 | ux | 0.108682 |
| 4 | 1751.65 | 0.105712 | 4 | vy | 38 | uy | 0.0480154 |
| 5 | 908.261 | 0.0284218 | 0 | vx | 51 | ux | 0.00794312 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1861.18 | 0.115626 | 1 | vx | 35 | ux | 0.020488 |
| 2 | 1860.56 | 0.11555 | 1 | vy | 35 | uy | 0.00471467 |
| 3 | 1758.56 | 0.103227 | 4 | vy | 38 | uy | 0.0086554 |
| 4 | 1758.15 | 0.103179 | 4 | vx | 38 | ux | 0.00865192 |
| 5 | 940.669 | 0.029536 | 31 | vy | 51 | uy | 0.00338103 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1864.27 | 0.111203 | 1 | vx | 35 | ux | 0.0244414 |
| 2 | 1863.8 | 0.111146 | 1 | vy | 35 | uy | 0.00785992 |
| 3 | 1762.54 | 0.099398 | 4 | vy | 38 | uy | 0.0130428 |
| 4 | 1761.84 | 0.0993189 | 4 | vx | 38 | ux | 0.0110417 |
| 5 | 1033.01 | 0.0341434 | 31 | vy | 52 | uy | 0.00344601 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 2708.12 | 0.146782 | 25 | vx | 56 | ux | 0.00061796 |
| 2 | 2472.15 | 0.122317 | 25 | vy | 56 | uy | 0.000777332 |
| 3 | 2082.54 | 0.0868004 | 24 | vy | 58 | uy | 0.00337088 |
| 4 | 1924.55 | 0.0741304 | 24 | vx | 54 | ux | 0.0104912 |
| 5 | 1834.03 | 0.0673208 | 1 | vx | 41 | ux | 0.0205434 |
