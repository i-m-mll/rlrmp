<!-- AUTO-GENERATED: canonical_soft_lambda_hvp -->
# Canonical soft-lambda HVP estimate

Issue: `06a4dc8`. Source experiment: `c92ebd8`.

## Method

The materializer estimates each trial's largest algebraic Hessian eigenvalue with HVP-backed Lanczos and reports `lambda_star_i = 0.5 * eigmax_i` under the ordinary Hessian convention `J ~= J0 + grad.T eps + 0.5 eps.T M eps`.

The objective is the corrected per-trial `J_i(delta_i)` for the 6D no-integrator process-epsilon channel. The corresponding soft training convention is `mean_i[J_i - lambda * E_i]`; cap/interiority is recorded only as provenance and is not used as a criterion.

## Run provenance

Command: `results/06a4dc8/scripts/materialize_canonical_soft_lambda_hvp.py --run-ids open_loop_small --batch-size 1 --max-trials-per-run 1 --lanczos-steps 2 --output-json results/06a4dc8/smoke/canonical_soft_lambda_hvp.json --output-csv results/06a4dc8/smoke/canonical_soft_lambda_hvp_trials.csv --output-md results/06a4dc8/smoke/canonical_soft_lambda_hvp.md`

Lanczos steps: `2`. Lanczos seed: `60931`. The JSON sidecar records per-trial Ritz values and residual estimates.

## Distribution summary

| substrate | n | lambda median | lambda p75 | lambda p90 | lambda max | eigmax p90 |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 1 | 2.50914e+08 | 2.50914e+08 | **2.50914e+08** | 2.50914e+08 | 5.01829e+08 |
| `pooled` | 1 | 2.50914e+08 | 2.50914e+08 | **2.50914e+08** | 2.50914e+08 | 5.01829e+08 |

Primary continuity summary: p90 of `lambda_star_i`.

## Beta mapping

| source | beta | role | lambda |
|---|---:|---|---:|
| `open_loop_small` | 0.95 | diagnostic_only | 2.2645e+08 |
| `open_loop_small` | 1.05 | candidate_training_scale | 2.76633e+08 |
| `open_loop_small` | 1.2 | candidate_training_scale | 3.61317e+08 |
| `open_loop_small` | 1.4 | candidate_training_scale | 4.91792e+08 |
| `open_loop_small` | 1.8 | candidate_training_scale | 8.12963e+08 |
| `pooled` | 0.95 | diagnostic_only | 2.2645e+08 |
| `pooled` | 1.05 | candidate_training_scale | 2.76633e+08 |
| `pooled` | 1.2 | candidate_training_scale | 3.61317e+08 |
| `pooled` | 1.4 | candidate_training_scale | 4.91792e+08 |
| `pooled` | 1.8 | candidate_training_scale | 8.12963e+08 |

Beta `0.95` is diagnostic only; beta values at or above `1.0` are candidate training scales pending later lanes.

## Analytical gamma comparison

`gamma_star = 9166.83128547`, so the previous `lambda = gamma^2` convention corresponds to `8.40308e+07`. The pooled p90 GRU-local conversion factor is `2.98598`.

## Finite-difference validation

| substrate | trial | step | central curvature | HVP eigmax | rel error |
|---|---:|---:|---:|---:|---:|
| `open_loop_small` | 0 | 1.0e-07 | 3.15509e+11 | 5.01829e+08 | 628 |
| `open_loop_small` | 0 | 3.0e-07 | 4.55007e+09 | 5.01829e+08 | 8.07 |
| `open_loop_small` | 0 | 1.0e-06 | 1.26451e+09 | 5.01829e+08 | 1.52 |
| `open_loop_small` | 0 | 3.0e-06 | 5.58718e+08 | 5.01829e+08 | 0.113 |
| `open_loop_small` | 0 | 1.0e-05 | 4.95595e+08 | 5.01829e+08 | 0.0124 |
| `open_loop_small` | 0 | 3.0e-05 | 5.0311e+08 | 5.01829e+08 | 0.00255 |

## Deterministic smoke command

```bash
PYTHONPATH=src uv run --no-sync python results/06a4dc8/scripts/materialize_canonical_soft_lambda_hvp.py \
  --run-ids open_loop_small --batch-size 1 --max-trials-per-run 1 --lanczos-steps 2 \
  --output-json results/06a4dc8/smoke/canonical_soft_lambda_hvp.json \
  --output-csv results/06a4dc8/smoke/canonical_soft_lambda_hvp_trials.csv \
  --output-md results/06a4dc8/smoke/canonical_soft_lambda_hvp.md
```
<!-- /AUTO-GENERATED -->
