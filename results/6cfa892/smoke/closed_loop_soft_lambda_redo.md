<!-- AUTO-GENERATED: closed_loop_soft_lambda_redo -->
# Closed-loop soft-lambda redo

Issue: `6cfa892`. Source no-PGD substrates: `c92ebd8`.

No training was launched and no controller weights were updated. This deterministic local materializer loads the frozen c92 substrates and evaluates closed-loop linear no-bias and affine mechanisms at beta-scaled lambda values from the corrected HVP/Lanczos p90 source.

## Source contract

HVP source: `results/06a4dc8/canonical_soft_lambda_hvp.json` (`rlrmp.canonical_soft_lambda_hvp.v1`). Primary scale: `lambda_star_p90`.

Beta mapping: `lambda(beta) = beta^2 * substrate_p90(lambda_star_i)`. Beta `0.95` is diagnostic only. Cap/interiority is not used as a criterion; old-cap ratios below are sidecars only.

## HVP/p90 beta mapping

| substrate | beta | role | lambda_star p90 | lambda | source |
|---|---:|---|---:|---:|---|
| `open_loop_small` | 0.95 | diagnostic_only | 2.55916e+08 | 2.30965e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_small` | 1.4 | candidate_training_scale | 2.55916e+08 | 5.01596e+08 | copied_from_hvp_source_beta_mapping |

## Objective-level rows

| substrate | mechanism | optimizer | beta | finite | grad | class | objective success | penalized gain | task gain | energy | penalty | norm | old-cap ratio |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1352.59 | 6420.45 | 2.19421e-05 | 5067.86 | 0.00471963 | 1.03831 |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 107.296 | 269.613 | 3.23599e-07 | 162.316 | 0.000569615 | 0.125314 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1405.53 | 5904.5 | 1.94791e-05 | 4498.97 | 0.00441351 | 0.970962 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 101.199 | 253.865 | 3.0436e-07 | 152.666 | 0.000551689 | 0.12137 |

## Best objective rows

| substrate | mechanism | optimizer | beta | class | penalized gain | task gain | energy penalty | norm | old-cap ratio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1352.59 | 6420.45 | 5067.86 | 0.00471963 | 1.03831 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1405.53 | 5904.5 | 4498.97 | 0.00441351 | 0.970962 |

## Classification counts

| substrate | counts |
|---|---|
| `open_loop_small` | `nonzero_positive_penalized_and_task_gain`: 4 |

## Interpretation

Closed-loop objective-level rows were finite on 4/4 evaluations and produced positive nonzero penalized-gain behavior on 4/4 evaluations. Of those successes, 2 were beta<1 diagnostic rows and 2 were beta>=1 candidate-scale rows. Old-cap ratios were reported only as sidecars and did not enter classification.

The old hard cap is retained only as `old_cap_*_sidecar` provenance. It is not used to select lambda, to define success, or to classify any row.

## Reproduction

```bash
uv run --no-sync python results/6cfa892/scripts/materialize_closed_loop_soft_lambda_redo.py
```

For a fast local smoke:

```bash
uv run --no-sync python results/6cfa892/scripts/materialize_closed_loop_soft_lambda_redo.py \
  --run-ids open_loop_small --betas 0.95 1.4 --optimizers line_search_known_direction \
  --output-json results/6cfa892/smoke/closed_loop_soft_lambda_redo.json \
  --output-csv results/6cfa892/smoke/closed_loop_soft_lambda_redo.csv \
  --output-md results/6cfa892/smoke/closed_loop_soft_lambda_redo.md
```
<!-- /AUTO-GENERATED -->
