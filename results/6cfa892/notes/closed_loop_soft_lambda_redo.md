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

## Objective-level rows

| substrate | mechanism | optimizer | beta | finite | grad | class | objective success | penalized gain | task gain | energy | penalty | norm | old-cap ratio |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1352.59 | 6420.45 | 2.19421e-05 | 5067.86 | 0.00471963 | 1.03831 |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 490.667 | 1978.24 | 5.27232e-06 | 1487.57 | 0.00230591 | 0.507295 |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 208.594 | 687.875 | 1.30056e-06 | 479.281 | 0.00114314 | 0.251488 |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 107.296 | 269.613 | 3.23599e-07 | 162.316 | 0.000569615 | 0.125314 |
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 1.8 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 49.4009 | 116.357 | 8.07503e-08 | 66.9557 | 0.000284383 | 0.0625637 |
| `open_loop_small` | `linear_no_bias` | `adam` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 683.194 | 3767.32 | 1.33532e-05 | 3084.13 | 0.00396783 | 0.872913 |
| `open_loop_small` | `linear_no_bias` | `adam` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 255.012 | 1658.26 | 4.97344e-06 | 1403.25 | 0.00236115 | 0.519448 |
| `open_loop_small` | `linear_no_bias` | `adam` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 48.1887 | 702.004 | 1.77417e-06 | 653.816 | 0.00139922 | 0.307824 |
| `open_loop_small` | `linear_no_bias` | `adam` | 1.4 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_small` | `linear_no_bias` | `adam` | 1.8 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1405.53 | 5904.5 | 1.94791e-05 | 4498.97 | 0.00441351 | 0.970962 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 470.422 | 1844.42 | 4.86976e-06 | 1373.99 | 0.00220675 | 0.485481 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 197.142 | 645.793 | 1.21744e-06 | 448.651 | 0.00110338 | 0.242741 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 101.199 | 253.865 | 3.0436e-07 | 152.666 | 0.000551689 | 0.12137 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 1.8 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 46.5941 | 109.686 | 7.60901e-08 | 63.0916 | 0.000275844 | 0.0606851 |
| `open_loop_small` | `affine` | `adam` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 2183.68 | 11961.8 | 4.23359e-05 | 9778.1 | 0.00695319 | 1.52969 |
| `open_loop_small` | `affine` | `adam` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 577.403 | 4900.16 | 1.53209e-05 | 4322.75 | 0.00407965 | 0.897515 |
| `open_loop_small` | `affine` | `adam` | 1.2 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_small` | `affine` | `adam` | 1.4 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_small` | `affine` | `adam` | 1.8 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1078.8 | 2617.75 | 7.38481e-06 | 1538.95 | 0.00277452 | 0.610388 |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 737.755 | 2617.75 | 7.38481e-06 | 1879.99 | 0.00277452 | 0.610388 |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 396.317 | 889.001 | 1.48173e-06 | 492.684 | 0.00122978 | 0.270549 |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 218.403 | 889.001 | 1.48173e-06 | 670.598 | 0.00122978 | 0.270549 |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | 1.8 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 102.744 | 356.037 | 3.38564e-07 | 253.294 | 0.000584757 | 0.128645 |
| `open_loop_moderate` | `linear_no_bias` | `adam` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1462.89 | 6200.41 | 2.27335e-05 | 4737.52 | 0.00511852 | 1.12606 |
| `open_loop_moderate` | `linear_no_bias` | `adam` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 658.667 | 2981.68 | 9.12504e-06 | 2323.01 | 0.00325682 | 0.716492 |
| `open_loop_moderate` | `linear_no_bias` | `adam` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 225.94 | 1336.57 | 3.34016e-06 | 1110.63 | 0.00198734 | 0.437211 |
| `open_loop_moderate` | `linear_no_bias` | `adam` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 14.2881 | 711.801 | 1.5412e-06 | 697.513 | 0.00135711 | 0.298562 |
| `open_loop_moderate` | `linear_no_bias` | `adam` | 1.8 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1651.58 | 5003.22 | 1.60832e-05 | 3351.64 | 0.00401038 | 0.882275 |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 908.837 | 5003.22 | 1.60832e-05 | 4094.38 | 0.00401038 | 0.882275 |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 350.609 | 1687.55 | 4.02079e-06 | 1336.94 | 0.00200519 | 0.441138 |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 185.703 | 640.634 | 1.0052e-06 | 454.931 | 0.0010026 | 0.220569 |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | 1.8 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 81.5856 | 269.593 | 2.51299e-07 | 188.007 | 0.000501298 | 0.110284 |
| `open_loop_moderate` | `affine` | `adam` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 3858.84 | 17187.4 | 6.39585e-05 | 13328.6 | 0.00840703 | 1.84953 |
| `open_loop_moderate` | `affine` | `adam` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1338.58 | 8154.2 | 2.67725e-05 | 6815.62 | 0.00543154 | 1.19493 |
| `open_loop_moderate` | `affine` | `adam` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 203.874 | 2691.21 | 7.48055e-06 | 2487.33 | 0.0028714 | 0.631702 |
| `open_loop_moderate` | `affine` | `adam` | 1.4 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `affine` | `adam` | 1.8 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1021.16 | 2406.63 | 7.22293e-06 | 1385.48 | 0.00272112 | 0.598639 |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 714.127 | 2406.63 | 7.22293e-06 | 1692.51 | 0.00272112 | 0.598639 |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 361.141 | 817.771 | 1.49198e-06 | 456.63 | 0.00123242 | 0.271129 |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 196.246 | 817.771 | 1.49198e-06 | 621.524 | 0.00123242 | 0.271129 |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | 1.8 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 91.9677 | 329.123 | 3.44389e-07 | 237.155 | 0.000589765 | 0.129747 |
| `open_loop_stress` | `linear_no_bias` | `adam` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1473.4 | 6679.97 | 2.71436e-05 | 5206.58 | 0.00536822 | 1.181 |
| `open_loop_stress` | `linear_no_bias` | `adam` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 617.374 | 3194.27 | 1.09971e-05 | 2576.89 | 0.00347649 | 0.76482 |
| `open_loop_stress` | `linear_no_bias` | `adam` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 209.063 | 1293.64 | 3.54373e-06 | 1084.58 | 0.00200005 | 0.440008 |
| `open_loop_stress` | `linear_no_bias` | `adam` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 9.03979 | 684.89 | 1.62239e-06 | 675.85 | 0.0013454 | 0.295985 |
| `open_loop_stress` | `linear_no_bias` | `adam` | 1.8 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `affine` | `line_search_known_direction` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1458.74 | 4680.45 | 1.67958e-05 | 3221.71 | 0.00409827 | 0.90161 |
| `open_loop_stress` | `affine` | `line_search_known_direction` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 744.785 | 4680.45 | 1.67958e-05 | 3935.66 | 0.00409827 | 0.90161 |
| `open_loop_stress` | `affine` | `line_search_known_direction` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 309.522 | 1594.64 | 4.19895e-06 | 1285.11 | 0.00204913 | 0.450805 |
| `open_loop_stress` | `affine` | `line_search_known_direction` | 1.4 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 172.634 | 609.93 | 1.04974e-06 | 437.296 | 0.00102457 | 0.225403 |
| `open_loop_stress` | `affine` | `line_search_known_direction` | 1.8 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 77.3258 | 258.045 | 2.62435e-07 | 180.719 | 0.000512284 | 0.112701 |
| `open_loop_stress` | `affine` | `adam` | 0.95 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 3543.17 | 16664.3 | 6.84047e-05 | 13121.1 | 0.00842608 | 1.85372 |
| `open_loop_stress` | `affine` | `adam` | 1.05 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 1191.18 | 8152.02 | 2.97061e-05 | 6960.84 | 0.00561991 | 1.23637 |
| `open_loop_stress` | `affine` | `adam` | 1.2 | finite | finite | `nonzero_positive_penalized_and_task_gain` | true | 169.218 | 2412.39 | 7.32928e-06 | 2243.17 | 0.00284851 | 0.626667 |
| `open_loop_stress` | `affine` | `adam` | 1.4 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |
| `open_loop_stress` | `affine` | `adam` | 1.8 | finite | finite | `zero_selected_no_positive_penalized_gain` | false | 0 | 0 | 0 | 0 | 0 | 0 |

## Best objective rows

| substrate | mechanism | optimizer | beta | class | penalized gain | task gain | energy penalty | norm | old-cap ratio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `open_loop_small` | `linear_no_bias` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1352.59 | 6420.45 | 5067.86 | 0.00471963 | 1.03831 |
| `open_loop_small` | `linear_no_bias` | `adam` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 683.194 | 3767.32 | 3084.13 | 0.00396783 | 0.872913 |
| `open_loop_small` | `affine` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1405.53 | 5904.5 | 4498.97 | 0.00441351 | 0.970962 |
| `open_loop_small` | `affine` | `adam` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 2183.68 | 11961.8 | 9778.1 | 0.00695319 | 1.52969 |
| `open_loop_moderate` | `linear_no_bias` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1078.8 | 2617.75 | 1538.95 | 0.00277452 | 0.610388 |
| `open_loop_moderate` | `linear_no_bias` | `adam` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1462.89 | 6200.41 | 4737.52 | 0.00511852 | 1.12606 |
| `open_loop_moderate` | `affine` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1651.58 | 5003.22 | 3351.64 | 0.00401038 | 0.882275 |
| `open_loop_moderate` | `affine` | `adam` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 3858.84 | 17187.4 | 13328.6 | 0.00840703 | 1.84953 |
| `open_loop_stress` | `linear_no_bias` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1021.16 | 2406.63 | 1385.48 | 0.00272112 | 0.598639 |
| `open_loop_stress` | `linear_no_bias` | `adam` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1473.4 | 6679.97 | 5206.58 | 0.00536822 | 1.181 |
| `open_loop_stress` | `affine` | `line_search_known_direction` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 1458.74 | 4680.45 | 3221.71 | 0.00409827 | 0.90161 |
| `open_loop_stress` | `affine` | `adam` | 0.95 | `nonzero_positive_penalized_and_task_gain` | 3543.17 | 16664.3 | 13121.1 | 0.00842608 | 1.85372 |

## Classification counts

| substrate | counts |
|---|---|
| `open_loop_small` | `nonzero_positive_penalized_and_task_gain`: 15, `zero_selected_no_positive_penalized_gain`: 5 |
| `open_loop_moderate` | `nonzero_positive_penalized_and_task_gain`: 17, `zero_selected_no_positive_penalized_gain`: 3 |
| `open_loop_stress` | `nonzero_positive_penalized_and_task_gain`: 17, `zero_selected_no_positive_penalized_gain`: 3 |

## Interpretation

Closed-loop objective-level rows were finite on 60/60 evaluations and produced positive nonzero penalized-gain behavior on 49/60 evaluations. Of those successes, 12 were beta<1 diagnostic rows and 37 were beta>=1 candidate-scale rows. Old-cap ratios were reported only as sidecars and did not enter classification.

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
