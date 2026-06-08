# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `020a65b`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64__nominal_clean | 0.290476 | -0.0189366 | -0.00550064 | 0.0843459 | 1860.99 | available | low_norm |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64__nominal_clean | 0.221441 | -0.00389014 | -0.000861436 | 0.0490353 | 1894.11 | available | low_norm |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64__nominal_clean | 0.372881 | -0.0041993 | -0.00156584 | 0.139038 | 1860.51 | available | low_norm |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64__nominal_clean | 0.674305 | -0.00266789 | -0.00179897 | 0.454684 | 1880.4 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1860.99 | 0.115344 | 1 | vx | 35 | ux | 0.0141867 |
| 2 | 1860.52 | 0.115286 | 1 | vy | 35 | uy | 0.0080747 |
| 3 | 1758.83 | 0.103028 | 4 | vx | 38 | ux | 0.00862551 |
| 4 | 1758.41 | 0.102979 | 4 | vy | 38 | uy | 0.010588 |
| 5 | 941.875 | 0.0295457 | 31 | vy | 51 | uy | 0.00370223 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1894.11 | 0.12456 | 1 | vx | 35 | ux | 0.165489 |
| 2 | 1885.92 | 0.123485 | 1 | vy | 35 | uy | 0.0214477 |
| 3 | 1790.88 | 0.111353 | 5 | vx | 38 | ux | 0.203305 |
| 4 | 1783.36 | 0.11042 | 5 | vy | 38 | uy | 0.0832724 |
| 5 | 880.902 | 0.0269416 | 1 | vx | 28 | ux | 0.0698229 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1860.51 | 0.110561 | 1 | vy | 35 | uy | 0.00946628 |
| 2 | 1859.43 | 0.110434 | 1 | vx | 35 | ux | 0.014248 |
| 3 | 1759.35 | 0.0988656 | 4 | vy | 38 | uy | 0.021521 |
| 4 | 1758.4 | 0.0987583 | 4 | vx | 38 | ux | 0.0293077 |
| 5 | 942.738 | 0.0283872 | 31 | vy | 51 | uy | 0.00661713 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 1880.4 | 0.0884564 | 1 | vy | 34 | uy | 0.0489185 |
| 2 | 1870.84 | 0.0875597 | 1 | vx | 34 | ux | 0.0371686 |
| 3 | 1799.09 | 0.0809722 | 5 | vy | 50 | uy | 0.0422645 |
| 4 | 1772.99 | 0.0786401 | 5 | vx | 50 | ux | 0.0613735 |
| 5 | 1647.13 | 0.067871 | 17 | vy | 38 | uy | 0.0121255 |
