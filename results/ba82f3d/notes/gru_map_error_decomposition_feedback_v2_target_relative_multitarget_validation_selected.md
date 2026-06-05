# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `ba82f3d`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean | 0.677121 | 0.30817 | 0.208668 | 0.41495 | 3174.06 | available | wrong_timing_or_channel, unexcited_directions |
| target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean | 1.07005 | 0.0937121 | 0.100277 | 1.13496 | 3352.82 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3174.06 | 0.128656 | 5 | py | 6 | uy | 1.78989e-05 |
| 2 | 3133.41 | 0.125382 | 5 | px | 6 | ux | 2.24121e-05 |
| 3 | 2653.63 | 0.0899253 | 8 | vy | 34 | uy | 2.20347e-05 |
| 4 | 2618.36 | 0.0875505 | 8 | vx | 34 | ux | 2.46332e-05 |
| 5 | 1583.35 | 0.0320151 | 5 | py | 6 | uy | 2.59648e-05 |

### `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3352.82 | 0.0768665 | 5 | py | 6 | uy | 1.93081e-05 |
| 2 | 3329.91 | 0.0758198 | 5 | px | 6 | ux | 2.40868e-05 |
| 3 | 2836.33 | 0.0550084 | 7 | vy | 34 | uy | 2.57806e-05 |
| 4 | 2814.64 | 0.0541706 | 8 | vx | 34 | ux | 2.70928e-05 |
| 5 | 1844.13 | 0.0232542 | 5 | py | 6 | uy | 2.0947e-05 |
