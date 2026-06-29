<!-- AUTO-GENERATED: critical_lambda_search -->
# Critical lambda frozen adversary search

Issue: `1697bdc`. Source frozen no-PGD runs: `c92ebd8`.

This audit defines practical `lambda_crit` as the smallest tested lambda multiplier where the optimized adversary is both interior and useful. Interior means `cap_bound_fraction = 0.0` and `max_norm_over_cap <= 0.99`; useful means finite objective/gradients and positive soft-energy objective gain over zero. It is not an analytical H-infinity threshold.

Closed-loop policy rows optimize the raw objective `J(raw_epsilon) - lambda * E(raw_epsilon)`. Cap behavior is computed afterward as a diagnostic.

## Summary

| row | mechanism | optimizer | bracket | lowest valid multiplier | gain | max norm/cap | cap-bound | finite | reliability | failure mode |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `open_loop_small` | `direct_epsilon` | `pgd_projected_epsilon` | yes | 2.82843 | 700.408 | 0.745786 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | yes | 2.37841 | 799.612 | 0.634226 | 0.000% | finite | reference_direction_only | bracketed_or_valid |
| `open_loop_small` | `linear_no_bias` | `adam` | yes | 2.37841 | 589.443 | 0.806749 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | yes | 2 | 733.725 | 0.911264 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_small` | `affine` | `line_search_known_direction` | no | 0.5 | 1491.3 | 0.542296 | 0.000% | finite | reference_direction_only | valid_at_lowest_probe_threshold_below_range |
| `open_loop_small` | `affine` | `adam` | yes | 2.59368 | 510.177 | 0.97554 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_small` | `affine` | `lbfgsb` | yes | 2.18102 | 608.22 | 0.751913 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_moderate` | `direct_epsilon` | `pgd_projected_epsilon` | yes | 3.08442 | 383.93 | 0.492094 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | yes | 2.37841 | 589.424 | 0.619236 | 0.000% | finite | reference_direction_only | bracketed_or_valid |
| `open_loop_moderate` | `linear_no_bias` | `adam` | yes | 2.37841 | 421.641 | 0.68372 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | yes | 2 | 539.381 | 0.893489 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | no | 0.5 | 2091.62 | 0.682757 | 0.000% | finite | reference_direction_only | valid_at_lowest_probe_threshold_below_range |
| `open_loop_moderate` | `affine` | `adam` | no | n/a | n/a | n/a | n/a | n/a | no_valid_point | mixed_invalid_or_nonmonotone |
| `open_loop_moderate` | `affine` | `lbfgsb` | yes | 2.18102 | 583.271 | 0.865387 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | yes | 2.18102 | 489.912 | 0.745454 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | yes | 1.68179 | 684.198 | 0.622951 | 0.000% | finite | reference_direction_only | bracketed_or_valid |
| `open_loop_stress` | `linear_no_bias` | `adam` | yes | 1.68179 | 449.573 | 0.745197 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | yes | 1.41421 | 541.794 | 0.83068 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_stress` | `affine` | `line_search_known_direction` | yes | 1.54221 | 309.866 | 0.613466 | 0.000% | finite | reference_direction_only | bracketed_or_valid |
| `open_loop_stress` | `affine` | `adam` | yes | 1.83401 | 401.221 | 0.909578 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |
| `open_loop_stress` | `affine` | `lbfgsb` | yes | 1.68179 | 524.961 | 0.853252 | 0.000% | finite | bounded_optimizer_materialized_valid_point | bracketed_or_valid |

## Probe and bisection rows

| row | mechanism | optimizer | phase | multiplier | gain | max norm/cap | cap-bound | useful | interior | valid | failure |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| `open_loop_small` | `direct_epsilon` | `pgd_projected_epsilon` | bracket | 2 | 2607.36 | 1 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `direct_epsilon` | `pgd_projected_epsilon` | bracket | 4 | 231.314 | 0.25 | 0.000% | True | True | True | valid |
| `open_loop_small` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.82843 | 700.408 | 0.745786 | 0.000% | True | True | True | valid |
| `open_loop_small` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.37841 | 1733.73 | 1 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.59368 | 1187.46 | 1 | 75.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | probe | 0.5 | 30087.7 | 4.6823 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | probe | 1 | 14099 | 4.6823 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | probe | 2 | 1329.41 | 1.56702 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | probe | 4 | 216.501 | 0.288986 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | probe | 8 | 37.4631 | 0.138955 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | bisection | 2.82843 | 493.331 | 0.634226 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | bisection | 2.37841 | 799.612 | 0.634226 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | bisection | 2.18102 | 625.726 | 1.56702 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `adam` | probe | 0.5 | 49522.9 | 6.38889 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `adam` | probe | 1 | 23775.6 | 5.40754 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `adam` | probe | 2 | 1400.46 | 1.46459 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `adam` | probe | 4 | 82.979 | 0.316696 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `adam` | probe | 8 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_small` | `linear_no_bias` | `adam` | bisection | 2.82843 | 318.223 | 0.569874 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `adam` | bisection | 2.37841 | 589.443 | 0.806749 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `adam` | bisection | 2.18102 | 854.584 | 1.15254 | 12.500% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | probe | 0.5 | 2.79691e+08 | 1967.98 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | probe | 1 | 252079 | 21.9496 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | probe | 2 | 733.725 | 0.911264 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | probe | 4 | 131.557 | 0.172565 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | probe | 8 | 50.3265 | 0.0702666 | 0.000% | True | True | True | valid |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | bisection | 1.41421 | 29253.9 | 10.1196 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | bisection | 1.68179 | 2642.03 | 3.07479 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `linear_no_bias` | `lbfgsb` | bisection | 1.83401 | 1195.88 | 1.48717 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `line_search_known_direction` | probe | 0.5 | 1491.3 | 0.542296 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `line_search_known_direction` | probe | 1 | 1152.87 | 0.542296 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `line_search_known_direction` | probe | 2 | 475.993 | 0.542296 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `line_search_known_direction` | probe | 4 | 6.13142 | 0.135574 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `line_search_known_direction` | probe | 8 | 5.08116 | 0.0338935 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `adam` | probe | 0.5 | 58811.9 | 6.60636 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `adam` | probe | 1 | 31530.2 | 5.06932 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `adam` | probe | 2 | 3239.39 | 2.43344 | 87.500% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `adam` | probe | 4 | 62.2622 | 0.310731 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `adam` | probe | 8 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_small` | `affine` | `adam` | bisection | 2.82843 | 333.116 | 0.768443 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `adam` | bisection | 2.37841 | 829.955 | 1.59382 | 75.000% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `adam` | bisection | 2.59368 | 510.177 | 0.97554 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `lbfgsb` | probe | 0.5 | 2.7301e+08 | 1951.48 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `lbfgsb` | probe | 1 | 252148 | 21.5225 | 100.000% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `lbfgsb` | probe | 2 | 878.897 | 1.07059 | 62.500% | True | False | False | cap_bound |
| `open_loop_small` | `affine` | `lbfgsb` | probe | 4 | 143.186 | 0.192763 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `lbfgsb` | probe | 8 | 53.515 | 0.0745938 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `lbfgsb` | bisection | 2.82843 | 281.354 | 0.353509 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `lbfgsb` | bisection | 2.37841 | 446.588 | 0.531806 | 0.000% | True | True | True | valid |
| `open_loop_small` | `affine` | `lbfgsb` | bisection | 2.18102 | 608.22 | 0.751913 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `direct_epsilon` | `pgd_projected_epsilon` | bracket | 2 | 2240.4 | 1 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `direct_epsilon` | `pgd_projected_epsilon` | bracket | 4 | 177.819 | 0.25 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.82843 | 552.458 | 0.990948 | 0.000% | True | False | False | near_cap |
| `open_loop_moderate` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 3.36359 | 262.412 | 0.25 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 3.08442 | 383.93 | 0.492094 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | probe | 0.5 | 36678.2 | 4.5086 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | probe | 1 | 20834.5 | 4.5086 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | probe | 2 | 1194.73 | 1.54592 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | probe | 4 | 145.354 | 0.278663 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | probe | 8 | 19.9574 | 0.133055 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | bisection | 2.82843 | 305.844 | 0.619236 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | bisection | 2.37841 | 589.424 | 0.619236 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | bisection | 2.18102 | 507.195 | 1.54592 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `adam` | probe | 0.5 | 35762 | 4.80528 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `adam` | probe | 1 | 17752.6 | 3.74386 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `adam` | probe | 2 | 1207 | 1.42396 | 75.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `adam` | probe | 4 | 44.1888 | 0.257195 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `adam` | probe | 8 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_moderate` | `linear_no_bias` | `adam` | bisection | 2.82843 | 211.084 | 0.468479 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `adam` | bisection | 2.37841 | 421.641 | 0.68372 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `adam` | bisection | 2.18102 | 659.977 | 1.13009 | 25.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | probe | 0.5 | 4.5647e+08 | 2720.79 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | probe | 1 | 74470 | 11.654 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | probe | 2 | 539.381 | 0.893489 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | probe | 4 | 108.444 | 0.160888 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | probe | 8 | 43.1426 | 0.0625014 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | bisection | 1.41421 | 39440.8 | 9.49319 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | bisection | 1.68179 | 10756.6 | 7.99214 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `linear_no_bias` | `lbfgsb` | bisection | 1.83401 | 942.185 | 1.63401 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | probe | 0.5 | 2091.62 | 0.682757 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | probe | 1 | 1595.92 | 0.682757 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | probe | 2 | 604.53 | 0.682757 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | probe | 4 | 8.2372 | 0.170689 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | probe | 8 | 8.58269 | 0.0426723 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `adam` | probe | 0.5 | 45192 | 5.75856 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `adam` | probe | 1 | 24400.4 | 4.70976 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `adam` | probe | 2 | 2395.25 | 1.87679 | 87.500% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `adam` | probe | 4 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_moderate` | `affine` | `adam` | probe | 8 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_moderate` | `affine` | `lbfgsb` | probe | 0.5 | 3.41288e+08 | 2426.28 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `lbfgsb` | probe | 1 | 164200 | 12.7856 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `lbfgsb` | probe | 2 | 1073.92 | 1.68473 | 100.000% | True | False | False | cap_bound |
| `open_loop_moderate` | `affine` | `lbfgsb` | probe | 4 | 125.088 | 0.169207 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `lbfgsb` | probe | 8 | 47.5965 | 0.0630569 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `lbfgsb` | bisection | 2.82843 | 248.77 | 0.35798 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `lbfgsb` | bisection | 2.37841 | 407.039 | 0.586054 | 0.000% | True | True | True | valid |
| `open_loop_moderate` | `affine` | `lbfgsb` | bisection | 2.18102 | 583.271 | 0.865387 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bracket | 1 | 3391.14 | 1 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bracket | 2 | 732.953 | 0.995226 | 0.000% | True | False | False | near_cap |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bracket_expand | 4 | 22.2769 | 0.25 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2 | 732.953 | 0.995226 | 0.000% | True | False | False | near_cap |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.82843 | 227.738 | 0.25 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.37841 | 348.546 | 0.745191 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `direct_epsilon` | `pgd_projected_epsilon` | bisection | 2.18102 | 489.912 | 0.745454 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | probe | 0.5 | 33887.8 | 4.76864 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | probe | 1 | 7806.39 | 4.76864 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | probe | 2 | 394.364 | 0.622951 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | probe | 4 | 118.224 | 0.133621 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | probe | 8 | 46.8803 | 0.0654223 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | bisection | 1.41421 | 1292.7 | 1.57258 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | bisection | 1.68179 | 684.198 | 0.622951 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | bisection | 1.54221 | 554.769 | 1.57258 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `adam` | probe | 0.5 | 23740 | 4.57594 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `adam` | probe | 1 | 6686.99 | 2.98054 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `adam` | probe | 2 | 234.486 | 0.530064 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `adam` | probe | 4 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_stress` | `linear_no_bias` | `adam` | probe | 8 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_stress` | `linear_no_bias` | `adam` | bisection | 1.41421 | 1212.39 | 1.48489 | 87.500% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `adam` | bisection | 1.68179 | 449.573 | 0.745197 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `adam` | bisection | 1.54221 | 680 | 1.16554 | 50.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | probe | 0.5 | 58411.3 | 9.43126 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | probe | 1 | 5874.05 | 5.74572 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | probe | 2 | 222.76 | 0.314563 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | probe | 4 | 78.3591 | 0.107138 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | probe | 8 | 34.5258 | 0.0468017 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | bisection | 1.41421 | 541.794 | 0.83068 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | bisection | 1.18921 | 1939.06 | 3.01203 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `linear_no_bias` | `lbfgsb` | bisection | 1.29684 | 836.436 | 1.35534 | 62.500% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `line_search_known_direction` | probe | 0.5 | 4128.9 | 1.22693 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `line_search_known_direction` | probe | 1 | 2016.91 | 1.22693 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `line_search_known_direction` | probe | 2 | 141.041 | 0.306733 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `line_search_known_direction` | probe | 4 | 44.6061 | 0.0766833 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `line_search_known_direction` | probe | 8 | 17.5695 | 0.0383416 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `line_search_known_direction` | bisection | 1.41421 | 267.279 | 1.22693 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `line_search_known_direction` | bisection | 1.68179 | 162.468 | 0.613466 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `line_search_known_direction` | bisection | 1.54221 | 309.866 | 0.613466 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `adam` | probe | 0.5 | 41942.9 | 5.49852 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `adam` | probe | 1 | 15349.8 | 4.14912 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `adam` | probe | 2 | 244.803 | 0.692617 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `adam` | probe | 4 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_stress` | `affine` | `adam` | probe | 8 | 0 | 0 | 0.000% | False | True | False | not_useful |
| `open_loop_stress` | `affine` | `adam` | bisection | 1.41421 | 3392.5 | 2.59638 | 87.500% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `adam` | bisection | 1.68179 | 711.202 | 1.15713 | 75.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `adam` | bisection | 1.83401 | 401.221 | 0.909578 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `lbfgsb` | probe | 0.5 | 120520 | 11.05 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `lbfgsb` | probe | 1 | 32495.9 | 7.45575 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `lbfgsb` | probe | 2 | 294.437 | 0.451795 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `lbfgsb` | probe | 4 | 89.2974 | 0.12189 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `lbfgsb` | probe | 8 | 39.4004 | 0.0497359 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `lbfgsb` | bisection | 1.41421 | 2234.98 | 3.49723 | 100.000% | True | False | False | cap_bound |
| `open_loop_stress` | `affine` | `lbfgsb` | bisection | 1.68179 | 524.961 | 0.853252 | 0.000% | True | True | True | valid |
| `open_loop_stress` | `affine` | `lbfgsb` | bisection | 1.54221 | 929.611 | 1.37475 | 100.000% | True | False | False | cap_bound |

## Interpretation

Direct epsilon produced practical lambda_crit estimates on 3/3 substrate rows. Closed-loop optimizer/mechanism searches produced valid practical points on 17/18 optimizer-specific rows; missing brackets are reported as outside-range, nonfinite, or optimizer/parameterization failures rather than interpreted as exact critical values.

Rows without a valid bracket are deliberately not assigned an exact lambda. `likely_above_probe_range` means useful adversaries remained too close to or over the cap through the tested range. `optimizer_or_parameterization_not_useful` means the optimizer did not find positive soft-energy gain over zero. `line_search_known_direction` is included as a reference optimizer comparison; the full-parameter Adam and L-BFGS-B rows are the closed-loop optimization rows.
<!-- /AUTO-GENERATED -->
