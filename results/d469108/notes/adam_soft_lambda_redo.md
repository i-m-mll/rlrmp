<!-- AUTO-GENERATED: adam_soft_lambda_redo -->
# Adam soft-lambda redo

Issue: `d469108`. Source no-PGD substrates: `c92ebd8`.

This materializer evaluates zero-start frozen Adam inner solves against the corrected HVP/p90 lambda mapping and corrected direct-epsilon / closed-loop reference rows. It does not launch training and does not update controller weights.

## Source contract

HVP source: `results/06a4dc8/canonical_soft_lambda_hvp.json` (`rlrmp.canonical_soft_lambda_hvp.v1`). Primary scale: `lambda_star_p90`.

Reference rows: direct-epsilon Adam is compared to corrected PGD direct-epsilon rows from `7180984`; closed-loop Adam is compared to corrected line-search-known-direction rows from `6cfa892`.

Agreement is objective-level only: finite Adam status, finite gradient, selected nonzero perturbation, positive penalized gain over zero, task-loss gain class, energy/penalty relation, and sidecar norm diagnostics. Old cap/interiority ratios are not criteria.

## Headline

Adam rows were finite with finite gradients on 405/405 bounded evaluations. Across beta>=1 candidate reference groups, Adam matched the corrected reference classification in 33/36 groups and matched the objective-success flag in 33/36 groups. Beta 0.95 is diagnostic only. Old hard-cap ratios are sidecars and do not enter selection, success, or failure labels.

## Per-reference summary

| substrate | mechanism | beta | role | ref class | any class agreement | any success agreement | representative Adam setting | representative class | penalized gain | task gain | energy penalty | norm | old-cap ratio |
|---|---|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| `open_loop_moderate` | `affine` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 852.864 | 2252.27 | 1399.41 | 0.00271885 | 0.598141 |
| `open_loop_moderate` | `affine` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 602.646 | 1809.78 | 1207.14 | 0.00228545 | 0.502795 |
| `open_loop_moderate` | `affine` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 359.914 | 1199.83 | 839.92 | 0.00166409 | 0.366096 |
| `open_loop_moderate` | `affine` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 196.395 | 669.912 | 473.517 | 0.00106907 | 0.235193 |
| `open_loop_moderate` | `affine` | 1.8 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 78.7815 | 258.76 | 179.978 | 0.000527542 | 0.116058 |
| `open_loop_moderate` | `direct_epsilon` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 326.211 | 441.661 | 115.45 | 0.000768843 | 0.169144 |
| `open_loop_moderate` | `direct_epsilon` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 300.537 | 432.845 | 132.308 | 0.000750062 | 0.165012 |
| `open_loop_moderate` | `direct_epsilon` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 260.774 | 413.793 | 153.018 | 0.000728178 | 0.160198 |
| `open_loop_moderate` | `direct_epsilon` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 210.318 | 384.663 | 174.345 | 0.000699744 | 0.153942 |
| `open_loop_moderate` | `direct_epsilon` | 1.8 | candidate_training_scale | `zero_selected_no_positive_penalized_gain` | false | false | steps=128; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 117.46 | 283.843 | 166.383 | 0.0005385 | 0.118469 |
| `open_loop_moderate` | `linear_no_bias` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 525.125 | 1392.8 | 867.672 | 0.00215712 | 0.474562 |
| `open_loop_moderate` | `linear_no_bias` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 382.358 | 1129.89 | 747.529 | 0.00181456 | 0.399199 |
| `open_loop_moderate` | `linear_no_bias` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 246.151 | 772.351 | 526.2 | 0.00133201 | 0.293039 |
| `open_loop_moderate` | `linear_no_bias` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 155.13 | 450.28 | 295.15 | 0.000855206 | 0.188144 |
| `open_loop_moderate` | `linear_no_bias` | 1.8 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 77.5195 | 196.656 | 119.136 | 0.000438011 | 0.0963615 |
| `open_loop_small` | `affine` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 550.33 | 1652.73 | 1102.4 | 0.00242772 | 0.534093 |
| `open_loop_small` | `affine` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 351.075 | 1230.3 | 879.23 | 0.00189918 | 0.417816 |
| `open_loop_small` | `affine` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 190.506 | 734.115 | 543.609 | 0.00125467 | 0.276026 |
| `open_loop_small` | `affine` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 92.9948 | 369.211 | 276.217 | 0.000748535 | 0.164676 |
| `open_loop_small` | `affine` | 1.8 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 30.7013 | 131.122 | 100.42 | 0.000353088 | 0.0776786 |
| `open_loop_small` | `direct_epsilon` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 216.747 | 342.839 | 126.093 | 0.000744305 | 0.163746 |
| `open_loop_small` | `direct_epsilon` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 188.664 | 332.903 | 144.24 | 0.000728551 | 0.16028 |
| `open_loop_small` | `direct_epsilon` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 148.947 | 297.031 | 148.084 | 0.00069153 | 0.152135 |
| `open_loop_small` | `direct_epsilon` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 104.62 | 253.633 | 149.013 | 0.000598736 | 0.131721 |
| `open_loop_small` | `direct_epsilon` | 1.8 | candidate_training_scale | `zero_selected_no_positive_penalized_gain` | false | false | steps=128; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 50.1287 | 120.804 | 70.6753 | 0.000304648 | 0.067022 |
| `open_loop_small` | `linear_no_bias` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 325.271 | 962.377 | 637.106 | 0.00186507 | 0.410312 |
| `open_loop_small` | `linear_no_bias` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 219.676 | 745.258 | 525.582 | 0.00147279 | 0.32401 |
| `open_loop_small` | `linear_no_bias` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 130.322 | 472.344 | 342.023 | 0.000994437 | 0.218774 |
| `open_loop_small` | `linear_no_bias` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 76.9245 | 248.62 | 171.696 | 0.000585746 | 0.128863 |
| `open_loop_small` | `linear_no_bias` | 1.8 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 35.2143 | 101.055 | 65.8402 | 0.000282141 | 0.0620704 |
| `open_loop_stress` | `affine` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 811.713 | 2167.87 | 1356.16 | 0.00266292 | 0.585837 |
| `open_loop_stress` | `affine` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 576.01 | 1709.26 | 1133.25 | 0.00222168 | 0.488764 |
| `open_loop_stress` | `affine` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 352.694 | 1121.03 | 768.336 | 0.00162612 | 0.357743 |
| `open_loop_stress` | `affine` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 193.293 | 638.745 | 445.452 | 0.00108056 | 0.237721 |
| `open_loop_stress` | `affine` | 1.8 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 77.9035 | 270.989 | 193.085 | 0.000561617 | 0.123554 |
| `open_loop_stress` | `direct_epsilon` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 294.442 | 401.459 | 107.017 | 0.000749883 | 0.164973 |
| `open_loop_stress` | `direct_epsilon` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 270.23 | 396.451 | 126.221 | 0.000738155 | 0.162392 |
| `open_loop_stress` | `direct_epsilon` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 231.4 | 385.21 | 153.81 | 0.000717062 | 0.157752 |
| `open_loop_stress` | `direct_epsilon` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 179.537 | 358.671 | 179.134 | 0.000673448 | 0.148157 |
| `open_loop_stress` | `direct_epsilon` | 1.8 | candidate_training_scale | `zero_selected_no_positive_penalized_gain` | false | false | steps=128; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 105.701 | 252.956 | 147.255 | 0.000520186 | 0.11444 |
| `open_loop_stress` | `linear_no_bias` | 0.95 | diagnostic_only | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 513.398 | 1396.08 | 882.687 | 0.00214939 | 0.472862 |
| `open_loop_stress` | `linear_no_bias` | 1.05 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 371.396 | 1094.86 | 723.466 | 0.00177771 | 0.391092 |
| `open_loop_stress` | `linear_no_bias` | 1.2 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 244.357 | 727.087 | 482.73 | 0.00129578 | 0.285068 |
| `open_loop_stress` | `linear_no_bias` | 1.4 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 154.611 | 428.605 | 273.995 | 0.000859941 | 0.189185 |
| `open_loop_stress` | `linear_no_bias` | 1.8 | candidate_training_scale | `nonzero_positive_penalized_and_task_gain` | true | true | steps=8; lr=1.0e-05 | `nonzero_positive_penalized_and_task_gain` | 76.2491 | 206.786 | 130.537 | 0.000468892 | 0.103155 |

## Mismatch groups

| substrate | mechanism | beta | role | reference class | Adam representative class | representative setting | representative gain | reference gain | note |
|---|---|---:|---|---|---|---|---:|---:|---|
| `open_loop_moderate` | `direct_epsilon` | 1.8 | candidate_training_scale | `zero_selected_no_positive_penalized_gain` | `nonzero_positive_penalized_and_task_gain` | steps=128; lr=1.0e-05 | 117.46 | -0.00012885 | Adam found positive objective behavior where the reference selected zero. |
| `open_loop_small` | `direct_epsilon` | 1.8 | candidate_training_scale | `zero_selected_no_positive_penalized_gain` | `nonzero_positive_penalized_and_task_gain` | steps=128; lr=1.0e-05 | 50.1287 | 3.11346e-06 | Adam found positive objective behavior where the reference selected zero. |
| `open_loop_stress` | `direct_epsilon` | 1.8 | candidate_training_scale | `zero_selected_no_positive_penalized_gain` | `nonzero_positive_penalized_and_task_gain` | steps=128; lr=1.0e-05 | 105.701 | -7.66348e-06 | Adam found positive objective behavior where the reference selected zero. |

Full per-setting Adam rows are tracked in `results/d469108/adam_soft_lambda_redo.csv` and `results/d469108/adam_soft_lambda_redo.json`.

## Representative Adam rows

| substrate | mechanism | beta | steps | lr | Adam class | ref class | agreement | penalized gain | task gain | norm | old-cap ratio |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| `open_loop_moderate` | `affine` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 852.864 | 2252.27 | 0.00271885 | 0.598141 |
| `open_loop_moderate` | `affine` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 602.646 | 1809.78 | 0.00228545 | 0.502795 |
| `open_loop_moderate` | `affine` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 359.914 | 1199.83 | 0.00166409 | 0.366096 |
| `open_loop_moderate` | `affine` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 196.395 | 669.912 | 0.00106907 | 0.235193 |
| `open_loop_moderate` | `affine` | 1.8 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 78.7815 | 258.76 | 0.000527542 | 0.116058 |
| `open_loop_moderate` | `direct_epsilon` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 326.211 | 441.661 | 0.000768843 | 0.169144 |
| `open_loop_moderate` | `direct_epsilon` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 300.537 | 432.845 | 0.000750062 | 0.165012 |
| `open_loop_moderate` | `direct_epsilon` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 260.774 | 413.793 | 0.000728178 | 0.160198 |
| `open_loop_moderate` | `direct_epsilon` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 210.318 | 384.663 | 0.000699744 | 0.153942 |
| `open_loop_moderate` | `direct_epsilon` | 1.8 | 128 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `zero_selected_no_positive_penalized_gain` | `fails_reference_objective_success` | 117.46 | 283.843 | 0.0005385 | 0.118469 |
| `open_loop_moderate` | `linear_no_bias` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 525.125 | 1392.8 | 0.00215712 | 0.474562 |
| `open_loop_moderate` | `linear_no_bias` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 382.358 | 1129.89 | 0.00181456 | 0.399199 |
| `open_loop_moderate` | `linear_no_bias` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 246.151 | 772.351 | 0.00133201 | 0.293039 |
| `open_loop_moderate` | `linear_no_bias` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 155.13 | 450.28 | 0.000855206 | 0.188144 |
| `open_loop_moderate` | `linear_no_bias` | 1.8 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 77.5195 | 196.656 | 0.000438011 | 0.0963615 |
| `open_loop_small` | `affine` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 550.33 | 1652.73 | 0.00242772 | 0.534093 |
| `open_loop_small` | `affine` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 351.075 | 1230.3 | 0.00189918 | 0.417816 |
| `open_loop_small` | `affine` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 190.506 | 734.115 | 0.00125467 | 0.276026 |
| `open_loop_small` | `affine` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 92.9948 | 369.211 | 0.000748535 | 0.164676 |
| `open_loop_small` | `affine` | 1.8 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 30.7013 | 131.122 | 0.000353088 | 0.0776786 |
| `open_loop_small` | `direct_epsilon` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 216.747 | 342.839 | 0.000744305 | 0.163746 |
| `open_loop_small` | `direct_epsilon` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 188.664 | 332.903 | 0.000728551 | 0.16028 |
| `open_loop_small` | `direct_epsilon` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 148.947 | 297.031 | 0.00069153 | 0.152135 |
| `open_loop_small` | `direct_epsilon` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 104.62 | 253.633 | 0.000598736 | 0.131721 |
| `open_loop_small` | `direct_epsilon` | 1.8 | 128 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `zero_selected_no_positive_penalized_gain` | `fails_reference_objective_success` | 50.1287 | 120.804 | 0.000304648 | 0.067022 |
| `open_loop_small` | `linear_no_bias` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 325.271 | 962.377 | 0.00186507 | 0.410312 |
| `open_loop_small` | `linear_no_bias` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 219.676 | 745.258 | 0.00147279 | 0.32401 |
| `open_loop_small` | `linear_no_bias` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 130.322 | 472.344 | 0.000994437 | 0.218774 |
| `open_loop_small` | `linear_no_bias` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 76.9245 | 248.62 | 0.000585746 | 0.128863 |
| `open_loop_small` | `linear_no_bias` | 1.8 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 35.2143 | 101.055 | 0.000282141 | 0.0620704 |
| `open_loop_stress` | `affine` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 811.713 | 2167.87 | 0.00266292 | 0.585837 |
| `open_loop_stress` | `affine` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 576.01 | 1709.26 | 0.00222168 | 0.488764 |
| `open_loop_stress` | `affine` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 352.694 | 1121.03 | 0.00162612 | 0.357743 |
| `open_loop_stress` | `affine` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 193.293 | 638.745 | 0.00108056 | 0.237721 |
| `open_loop_stress` | `affine` | 1.8 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 77.9035 | 270.989 | 0.000561617 | 0.123554 |
| `open_loop_stress` | `direct_epsilon` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 294.442 | 401.459 | 0.000749883 | 0.164973 |
| `open_loop_stress` | `direct_epsilon` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 270.23 | 396.451 | 0.000738155 | 0.162392 |
| `open_loop_stress` | `direct_epsilon` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 231.4 | 385.21 | 0.000717062 | 0.157752 |
| `open_loop_stress` | `direct_epsilon` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 179.537 | 358.671 | 0.000673448 | 0.148157 |
| `open_loop_stress` | `direct_epsilon` | 1.8 | 128 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `zero_selected_no_positive_penalized_gain` | `fails_reference_objective_success` | 105.701 | 252.956 | 0.000520186 | 0.11444 |
| `open_loop_stress` | `linear_no_bias` | 0.95 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 513.398 | 1396.08 | 0.00214939 | 0.472862 |
| `open_loop_stress` | `linear_no_bias` | 1.05 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 371.396 | 1094.86 | 0.00177771 | 0.391092 |
| `open_loop_stress` | `linear_no_bias` | 1.2 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 244.357 | 727.087 | 0.00129578 | 0.285068 |
| `open_loop_stress` | `linear_no_bias` | 1.4 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 154.611 | 428.605 | 0.000859941 | 0.189185 |
| `open_loop_stress` | `linear_no_bias` | 1.8 | 8 | 1.0e-05 | `nonzero_positive_penalized_and_task_gain` | `nonzero_positive_penalized_and_task_gain` | `agrees_with_reference_classification` | 76.2491 | 206.786 | 0.000468892 | 0.103155 |

## Counts

- Row classifications: `nonzero_positive_penalized_and_task_gain`: 361, `zero_selected_no_positive_penalized_gain`: 44
- Agreement labels: `agrees_with_reference_classification`: 334, `fails_reference_objective_success`: 71

## Reproduction

```bash
PYTHONPATH=src uv run --no-sync python \
  results/d469108/scripts/materialize_adam_soft_lambda_redo.py
```

Fast smoke:

```bash
PYTHONPATH=src uv run --no-sync python \
  results/d469108/scripts/materialize_adam_soft_lambda_redo.py \
  --run-ids open_loop_small --mechanisms direct_epsilon linear_no_bias \
  --betas 0.95 1.4 --adam-steps 2 --adam-learning-rates 5e-5 \
  --output-json results/d469108/smoke/adam_soft_lambda_redo.json \
  --output-csv results/d469108/smoke/adam_soft_lambda_redo.csv \
  --output-md results/d469108/smoke/adam_soft_lambda_redo.md
```
<!-- /AUTO-GENERATED -->
