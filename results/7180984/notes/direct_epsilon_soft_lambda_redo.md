<!-- AUTO-GENERATED: direct_epsilon_soft_lambda_redo -->
# Direct-epsilon soft-lambda redo

Issue: `7180984`. Source no-PGD substrates: `c92ebd8`.

No controller weights were updated. This deterministic local materializer loads frozen c92 no-PGD substrates and evaluates direct-epsilon optima at beta-scaled lambda values from the corrected HVP/Lanczos p90 source.

## Source contract

HVP source: `results/06a4dc8/canonical_soft_lambda_hvp.json` (`rlrmp.canonical_soft_lambda_hvp.v1`). Primary scale: `lambda_star_p90`.

Beta mapping: `lambda(beta) = beta^2 * substrate_p90(lambda_star_i)`. Beta `0.95` is diagnostic only. Cap/interiority is not used as a criterion; old-cap ratios below are sidecars only.

## HVP/p90 beta mapping

| substrate | beta | role | lambda_star p90 | lambda | source |
|---|---:|---|---:|---:|---|
| `open_loop_small` | 0.95 | diagnostic_only | 2.55916e+08 | 2.30965e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_small` | 1.05 | candidate_training_scale | 2.55916e+08 | 2.82148e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_small` | 1.2 | candidate_training_scale | 2.55916e+08 | 3.6852e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_small` | 1.4 | candidate_training_scale | 2.55916e+08 | 5.01596e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_small` | 1.8 | candidate_training_scale | 2.55916e+08 | 8.29169e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_moderate` | 0.95 | diagnostic_only | 2.30908e+08 | 2.08394e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_moderate` | 1.05 | candidate_training_scale | 2.30908e+08 | 2.54576e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_moderate` | 1.2 | candidate_training_scale | 2.30908e+08 | 3.32507e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_moderate` | 1.4 | candidate_training_scale | 2.30908e+08 | 4.52579e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_moderate` | 1.8 | candidate_training_scale | 2.30908e+08 | 7.48141e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_stress` | 0.95 | diagnostic_only | 2.12539e+08 | 1.91816e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_stress` | 1.05 | candidate_training_scale | 2.12539e+08 | 2.34324e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_stress` | 1.2 | candidate_training_scale | 2.12539e+08 | 3.06056e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_stress` | 1.4 | candidate_training_scale | 2.12539e+08 | 4.16576e+08 | copied_from_hvp_source_beta_mapping |
| `open_loop_stress` | 1.8 | candidate_training_scale | 2.12539e+08 | 6.88626e+08 | copied_from_hvp_source_beta_mapping |

## Direct-epsilon objective rows

| substrate | beta | finite | selected | class | penalized gain | task gain | energy | penalty | norm | old-cap ratio |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 0.95 | finite | true | `nonzero_positive_penalized_and_task_gain` | 1977.38 | 6749.47 | 2.06616e-05 | 4772.09 | 0.0045455 | 1 |
| `open_loop_small` | 1.05 | finite | true | `nonzero_positive_penalized_and_task_gain` | 842.467 | 6672.08 | 2.06616e-05 | 5829.62 | 0.0045455 | 1 |
| `open_loop_small` | 1.2 | finite | true | `nonzero_positive_penalized_and_task_gain` | 208.816 | 684.703 | 1.29135e-06 | 475.887 | 0.00113638 | 0.25 |
| `open_loop_small` | 1.4 | finite | true | `nonzero_positive_penalized_and_task_gain` | 36.968 | 684.703 | 1.29135e-06 | 647.735 | 0.00113638 | 0.25 |
| `open_loop_small` | 1.8 | finite | false | `zero_selected_no_positive_penalized_gain` | 3.11346e-06 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | 0.95 | finite | true | `nonzero_positive_penalized_and_task_gain` | 2618.58 | 6924.33 | 2.06616e-05 | 4305.75 | 0.0045455 | 1 |
| `open_loop_moderate` | 1.05 | finite | true | `nonzero_positive_penalized_and_task_gain` | 1662.29 | 6922.22 | 2.06616e-05 | 5259.93 | 0.0045455 | 1 |
| `open_loop_moderate` | 1.2 | finite | true | `nonzero_positive_penalized_and_task_gain` | 530.831 | 2033.25 | 4.51845e-06 | 1502.41 | 0.00222918 | 0.490414 |
| `open_loop_moderate` | 1.4 | finite | true | `nonzero_positive_penalized_and_task_gain` | 246.269 | 830.706 | 1.29135e-06 | 584.437 | 0.00113638 | 0.25 |
| `open_loop_moderate` | 1.8 | finite | false | `zero_selected_no_positive_penalized_gain` | -0.00012885 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_stress` | 0.95 | finite | true | `nonzero_positive_penalized_and_task_gain` | 2309.04 | 6272.26 | 2.06616e-05 | 3963.22 | 0.0045455 | 1 |
| `open_loop_stress` | 1.05 | finite | true | `nonzero_positive_penalized_and_task_gain` | 1425.14 | 6266.64 | 2.06616e-05 | 4841.5 | 0.0045455 | 1 |
| `open_loop_stress` | 1.2 | finite | true | `nonzero_positive_penalized_and_task_gain` | 452.88 | 1744.79 | 4.22117e-06 | 1291.91 | 0.00218854 | 0.481474 |
| `open_loop_stress` | 1.4 | finite | true | `nonzero_positive_penalized_and_task_gain` | 209.612 | 747.556 | 1.29135e-06 | 537.945 | 0.00113638 | 0.25 |
| `open_loop_stress` | 1.8 | finite | false | `zero_selected_no_positive_penalized_gain` | -7.66348e-06 | 0 | 0 | 0 | 0 | 0 |

## Classification counts

| substrate | counts |
|---|---|
| `open_loop_small` | `nonzero_positive_penalized_and_task_gain`: 4, `zero_selected_no_positive_penalized_gain`: 1 |
| `open_loop_moderate` | `nonzero_positive_penalized_and_task_gain`: 4, `zero_selected_no_positive_penalized_gain`: 1 |
| `open_loop_stress` | `nonzero_positive_penalized_and_task_gain`: 4, `zero_selected_no_positive_penalized_gain`: 1 |

## Deterministic local audit command

`PYTHONPATH=src uv run --no-sync python results/7180984/scripts/materialize_direct_epsilon_soft_lambda_redo.py`

Focused smoke example:

`PYTHONPATH=src uv run --no-sync python results/7180984/scripts/materialize_direct_epsilon_soft_lambda_redo.py --run-ids open_loop_small --pgd-steps 2 --output-json results/7180984/smoke/direct_epsilon_soft_lambda_redo.json --output-csv results/7180984/smoke/direct_epsilon_soft_lambda_redo.csv --output-md results/7180984/smoke/direct_epsilon_soft_lambda_redo.md`
<!-- /AUTO-GENERATED -->
