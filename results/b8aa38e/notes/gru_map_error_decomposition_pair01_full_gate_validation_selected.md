# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `b8aa38e`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean | 1.33168 | 0.268282 | 0.357266 | 1.64573 | 3666 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean | 0.907932 | 0.000857452 | 0.000778508 | 0.824341 | 2708.12 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3666 | 0.0867922 | 5 | px | 56 | ux | 1.83467e-05 |
| 2 | 3563.98 | 0.0820285 | 5 | py | 44 | uy | 1.66649e-05 |
| 3 | 3535.89 | 0.0807408 | 42 | px | 58 | ux | 1.81723e-05 |
| 4 | 3375.67 | 0.0735892 | 42 | py | 58 | uy | 1.5667e-05 |
| 5 | 2502.71 | 0.0404497 | 5 | px | 59 | ux | 1.4421e-05 |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 2708.12 | 0.146782 | 25 | vx | 56 | ux | 0.00061796 |
| 2 | 2472.15 | 0.122317 | 25 | vy | 56 | uy | 0.000777332 |
| 3 | 2082.54 | 0.0868004 | 24 | vy | 58 | uy | 0.00337088 |
| 4 | 1924.55 | 0.0741304 | 24 | vx | 54 | ux | 0.0104912 |
| 5 | 1834.03 | 0.0673208 | 1 | vx | 41 | ux | 0.0205434 |
