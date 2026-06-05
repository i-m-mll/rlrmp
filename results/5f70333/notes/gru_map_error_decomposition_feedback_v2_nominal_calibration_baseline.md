# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `5f70333`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean | 0.162183 | 0.220122 | 0.0357001 | 0.025029 | 3406.02 | available | low_norm, unexcited_directions |

## Top Singular Directions

### `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3406.02 | 0.16153 | 5 | py | 6 | uy | 2.30485e-05 |
| 2 | 3315.63 | 0.153069 | 5 | px | 6 | ux | 2.75558e-05 |
| 3 | 2825.26 | 0.111141 | 8 | vy | 35 | uy | 2.35745e-05 |
| 4 | 2732.34 | 0.10395 | 8 | vx | 35 | ux | 2.22201e-05 |
| 5 | 1609.53 | 0.0360709 | 5 | py | 6 | uy | 2.01363e-05 |
