# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `aacb9ed`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean | 0.313647 | 0.334215 | 0.104826 | 0.0873861 | 3345.92 | available | low_norm, unexcited_directions |
| fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean | 0.425669 | 0.2944 | 0.125317 | 0.16549 | 3247.02 | available | low_norm, unexcited_directions |

## Top Singular Directions

### `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3345.92 | 0.167486 | 5 | p_lateral | 6 | u_lateral | 1.65813e-05 |
| 2 | 3063.74 | 0.140428 | 5 | p_parallel | 6 | u_parallel | 2.05089e-05 |
| 3 | 2756.05 | 0.113638 | 8 | v_lateral | 35 | u_lateral | 2.08162e-05 |
| 4 | 2430.09 | 0.0883475 | 8 | v_parallel | 34 | u_parallel | 2.20103e-05 |
| 5 | 1597.68 | 0.0381882 | 5 | p_lateral | 6 | u_lateral | 1.89921e-05 |

### `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3247.02 | 0.150641 | 5 | p_lateral | 6 | u_lateral | 1.55098e-05 |
| 2 | 3132.96 | 0.140243 | 5 | p_parallel | 6 | u_parallel | 2.12279e-05 |
| 3 | 2633.15 | 0.0990653 | 8 | v_lateral | 34 | u_lateral | 2.12109e-05 |
| 4 | 2428.52 | 0.0842664 | 8 | v_parallel | 34 | u_parallel | 2.27616e-05 |
| 5 | 1576.65 | 0.0355175 | 5 | p_lateral | 6 | u_lateral | 1.8543e-05 |
