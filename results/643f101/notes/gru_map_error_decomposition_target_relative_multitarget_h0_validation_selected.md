# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `643f101`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean | 0.438193 | 0.122515 | 0.0536851 | 0.189131 | 3486.45 | available | low_norm, unexcited_directions |
| target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean | 0.592092 | 0.402951 | 0.238584 | 0.29365 | 3043.48 | available | wrong_timing_or_channel, unexcited_directions |

## Top Singular Directions

### `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3486.45 | 0.149003 | 5 | px | 6 | ux | 2.08768e-05 |
| 2 | 3431.76 | 0.144365 | 5 | py | 6 | uy | 2.04267e-05 |
| 3 | 2971.02 | 0.108203 | 8 | vx | 35 | ux | 1.92449e-05 |
| 4 | 2905.7 | 0.103498 | 8 | vy | 35 | uy | 2.38748e-05 |
| 5 | 1964.11 | 0.047289 | 0 | py | 0 | uy | 1.18415e-05 |

### `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3043.48 | 0.141008 | 5 | py | 6 | uy | 1.75471e-05 |
| 2 | 2988.06 | 0.135919 | 5 | px | 6 | ux | 1.62255e-05 |
| 3 | 2444.2 | 0.0909438 | 8 | vy | 35 | uy | 2.61663e-05 |
| 4 | 2400.37 | 0.0877119 | 8 | vx | 34 | ux | 3.27664e-05 |
| 5 | 1558.52 | 0.0369766 | 0 | px | 0 | ux | 2.34902e-05 |
