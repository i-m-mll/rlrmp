# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `e4800d6`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64__nominal_clean | 0.881835 | 0.176797 | 0.155906 | 0.753327 | 3274.07 | available | wrong_timing_or_channel, unexcited_directions |
| cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64__nominal_clean | 2.25111 | 0.122066 | 0.274784 | 4.99197 | 8650.97 | available | unexcited_directions |

## Top Singular Directions

### `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 3274.07 | 0.0972323 | 5 | px | 6 | ux | 2.08987e-05 |
| 2 | 3264.72 | 0.096678 | 5 | py | 6 | uy | 1.73133e-05 |
| 3 | 2734.65 | 0.0678325 | 8 | vy | 46 | uy | 1.81648e-05 |
| 4 | 2711.79 | 0.0667032 | 8 | vx | 35 | ux | 1.83229e-05 |
| 5 | 2575.87 | 0.0601845 | 39 | py | 47 | uy | 1.57242e-05 |

### `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64__nominal_clean`

| rank | singular value | energy fraction | obs time | obs channel | action time | action channel | covariance projection |
|---:|---:|---:|---|---|---|---|---:|
| 1 | 8650.97 | 0.180331 | 32 | py | 45 | uy | 1.89285e-05 |
| 2 | 8565.13 | 0.176771 | 30 | py | 43 | uy | 1.86038e-05 |
| 3 | 8159.15 | 0.16041 | 30 | px | 43 | ux | 1.46012e-05 |
| 4 | 8035.87 | 0.155599 | 32 | px | 41 | ux | 1.79489e-05 |
| 5 | 3568.27 | 0.0306802 | 5 | px | 31 | ux | 2.48793e-05 |
