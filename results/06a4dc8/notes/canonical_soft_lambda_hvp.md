<!-- AUTO-GENERATED: canonical_soft_lambda_hvp -->
# Canonical soft-lambda HVP estimate

Issue: `06a4dc8`. Source experiment: `c92ebd8`.

## Method

The materializer estimates each trial's largest algebraic Hessian eigenvalue with HVP-backed Lanczos and reports `lambda_star_i = 0.5 * eigmax_i` under the ordinary Hessian convention `J ~= J0 + grad.T eps + 0.5 eps.T M eps`.

The objective is the corrected per-trial `J_i(delta_i)` for the 6D no-integrator process-epsilon channel. The corresponding soft training convention is `mean_i[J_i - lambda * E_i]`; cap/interiority is recorded only as provenance and is not used as a criterion.

## Run provenance

Command: `results/06a4dc8/scripts/materialize_canonical_soft_lambda_hvp.py --batch-size 2 --max-trials-per-run 2 --lanczos-steps 4`

Lanczos steps: `4`. Lanczos seed: `60931`. The JSON sidecar records per-trial Ritz values and residual estimates.

## Distribution summary

| substrate | n | lambda median | lambda p75 | lambda p90 | lambda max | eigmax p90 |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 2 | 2.54905e+08 | 2.55537e+08 | **2.55916e+08** | 2.56169e+08 | 5.11833e+08 |
| `open_loop_moderate` | 2 | 2.29983e+08 | 2.30561e+08 | **2.30908e+08** | 2.31139e+08 | 4.61815e+08 |
| `open_loop_stress` | 2 | 2.12186e+08 | 2.12406e+08 | **2.12539e+08** | 2.12627e+08 | 4.25077e+08 |
| `pooled` | 6 | 2.29983e+08 | 2.48016e+08 | **2.54905e+08** | 2.56169e+08 | 5.0981e+08 |

Primary continuity summary: p90 of `lambda_star_i`.

## Beta mapping

| source | beta | role | lambda |
|---|---:|---|---:|
| `open_loop_small` | 0.95 | diagnostic_only | 2.30965e+08 |
| `open_loop_small` | 1.05 | candidate_training_scale | 2.82148e+08 |
| `open_loop_small` | 1.2 | candidate_training_scale | 3.6852e+08 |
| `open_loop_small` | 1.4 | candidate_training_scale | 5.01596e+08 |
| `open_loop_small` | 1.8 | candidate_training_scale | 8.29169e+08 |
| `open_loop_moderate` | 0.95 | diagnostic_only | 2.08394e+08 |
| `open_loop_moderate` | 1.05 | candidate_training_scale | 2.54576e+08 |
| `open_loop_moderate` | 1.2 | candidate_training_scale | 3.32507e+08 |
| `open_loop_moderate` | 1.4 | candidate_training_scale | 4.52579e+08 |
| `open_loop_moderate` | 1.8 | candidate_training_scale | 7.48141e+08 |
| `open_loop_stress` | 0.95 | diagnostic_only | 1.91816e+08 |
| `open_loop_stress` | 1.05 | candidate_training_scale | 2.34324e+08 |
| `open_loop_stress` | 1.2 | candidate_training_scale | 3.06056e+08 |
| `open_loop_stress` | 1.4 | candidate_training_scale | 4.16576e+08 |
| `open_loop_stress` | 1.8 | candidate_training_scale | 6.88626e+08 |
| `pooled` | 0.95 | diagnostic_only | 2.30052e+08 |
| `pooled` | 1.05 | candidate_training_scale | 2.81033e+08 |
| `pooled` | 1.2 | candidate_training_scale | 3.67064e+08 |
| `pooled` | 1.4 | candidate_training_scale | 4.99614e+08 |
| `pooled` | 1.8 | candidate_training_scale | 8.25893e+08 |

Beta `0.95` is diagnostic only; beta values at or above `1.0` are candidate training scales pending later lanes.

## Analytical gamma comparison

`gamma_star = 9166.83128547`, so the previous `lambda = gamma^2` convention corresponds to `8.40308e+07`. The pooled p90 GRU-local conversion factor is `3.03347`.

## Finite-difference validation

| substrate | trial | step | central curvature | HVP eigmax | rel error |
|---|---:|---:|---:|---:|---:|
| `open_loop_small` | 0 | 1.0e-07 | 3.75173e+11 | 5.07282e+08 | 739 |
| `open_loop_small` | 0 | 3.0e-07 | -1.31315e+10 | 5.07282e+08 | 26.9 |
| `open_loop_small` | 0 | 1.0e-06 | 2.77413e+09 | 5.07282e+08 | 4.47 |
| `open_loop_small` | 0 | 3.0e-06 | 6.86302e+08 | 5.07282e+08 | 0.353 |
| `open_loop_small` | 0 | 1.0e-05 | 5.29361e+08 | 5.07282e+08 | 0.0435 |
| `open_loop_small` | 0 | 3.0e-05 | 5.08775e+08 | 5.07282e+08 | 0.00294 |
| `open_loop_moderate` | 0 | 1.0e-07 | 2.40652e+11 | 4.62277e+08 | 520 |
| `open_loop_moderate` | 0 | 3.0e-07 | -6.86203e+09 | 4.62277e+08 | 15.8 |
| `open_loop_moderate` | 0 | 1.0e-06 | -4.74718e+09 | 4.62277e+08 | 11.3 |
| `open_loop_moderate` | 0 | 3.0e-06 | 6.03505e+08 | 4.62277e+08 | 0.306 |
| `open_loop_moderate` | 0 | 1.0e-05 | 4.59588e+08 | 4.62277e+08 | 0.00582 |
| `open_loop_moderate` | 0 | 3.0e-05 | 4.60303e+08 | 4.62277e+08 | 0.00427 |
| `open_loop_stress` | 0 | 1.0e-07 | 6.18183e+11 | 4.23489e+08 | 1.46e+03 |
| `open_loop_stress` | 0 | 3.0e-07 | 1.8515e+10 | 4.23489e+08 | 42.7 |
| `open_loop_stress` | 0 | 1.0e-06 | 4.50156e+09 | 4.23489e+08 | 9.63 |
| `open_loop_stress` | 0 | 3.0e-06 | 4.08726e+08 | 4.23489e+08 | 0.0349 |
| `open_loop_stress` | 0 | 1.0e-05 | 4.57623e+08 | 4.23489e+08 | 0.0806 |
| `open_loop_stress` | 0 | 3.0e-05 | 4.26728e+08 | 4.23489e+08 | 0.00765 |

## Deterministic smoke command

```bash
PYTHONPATH=src uv run --no-sync python results/06a4dc8/scripts/materialize_canonical_soft_lambda_hvp.py \
  --run-ids open_loop_small --batch-size 1 --max-trials-per-run 1 --lanczos-steps 2 \
  --output-json results/06a4dc8/smoke/canonical_soft_lambda_hvp.json \
  --output-csv results/06a4dc8/smoke/canonical_soft_lambda_hvp_trials.csv \
  --output-md results/06a4dc8/smoke/canonical_soft_lambda_hvp.md
```
<!-- /AUTO-GENERATED -->
