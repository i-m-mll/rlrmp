# GRU Feedback Ablation Diagnostic

- Issue: `ae9f30f`
- Source experiment: `ae9f30f`
- Scope: `soft_lambda_wave1_trained_final`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `direct_epsilon_b1p05` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `direct_epsilon_b1p4` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `linear_no_bias_b1p05` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `direct_epsilon_b1p05` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p05` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 283.09 | 0.00327892 | 0.00644488 |
| `direct_epsilon_b1p05` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 283.09 | 0.00327892 | 0.00644488 |
| `direct_epsilon_b1p05` | `nominal` | `shuffled_observation_history` | evaluated | 0.55341 | 330.214 | 0.00384048 | 0.00825217 |
| `direct_epsilon_b1p05` | `nominal` | `lagged_observation_history` | evaluated | 0.61806 | 23833.3 | 0.0502323 | 0.0908181 |
| `direct_epsilon_b1p05` | `nominal` | `position_only_observation` | evaluated | 13.287 | 2.90268e+06 | 0.508431 | 1.45427 |
| `direct_epsilon_b1p05` | `nominal` | `velocity_only_observation` | evaluated | 3.30134 | 162795 | 0.144557 | 0.0358589 |
| `direct_epsilon_b1p05` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p05` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.175382 | 977.093 | 0.00682056 | 0.00641079 |
| `direct_epsilon_b1p05` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.175382 | 977.093 | 0.00682056 | 0.00641079 |
| `direct_epsilon_b1p05` | `initial_state` | `shuffled_observation_history` | evaluated | 0.552184 | 331.08 | 0.00402607 | 0.00823172 |
| `direct_epsilon_b1p05` | `initial_state` | `lagged_observation_history` | evaluated | 0.0595609 | 1.03087 | -1.17873e-05 | 5.72808e-06 |
| `direct_epsilon_b1p05` | `initial_state` | `position_only_observation` | evaluated | 13.0553 | 2.709e+06 | 0.484056 | 1.51003 |
| `direct_epsilon_b1p05` | `initial_state` | `velocity_only_observation` | evaluated | 3.2483 | 146678 | 0.137203 | 0.0358248 |
| `direct_epsilon_b1p05` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p05` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.579611 | 850.567 | 0.00602639 | 0.0175173 |
| `direct_epsilon_b1p05` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.579611 | 850.567 | 0.00602639 | 0.0175173 |
| `direct_epsilon_b1p05` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.56343 | 231.79 | 0.0032157 | 0.00670643 |
| `direct_epsilon_b1p05` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.623004 | 23194.8 | 0.0500904 | 0.0871449 |
| `direct_epsilon_b1p05` | `process_epsilon` | `position_only_observation` | evaluated | 13.3877 | 2.95277e+06 | 0.514818 | 1.44128 |
| `direct_epsilon_b1p05` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.3536 | 159541 | 0.143905 | 0.036811 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.159678 | 215.008 | 0.0034089 | 0.00637195 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.159678 | 215.008 | 0.0034089 | 0.00637195 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.567695 | 818.102 | 0.00651857 | 0.0226212 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.64933 | 17706.1 | 0.0439291 | 0.0651827 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `position_only_observation` | evaluated | 13.2281 | 2.90262e+06 | 0.508561 | 1.4542 |
| `direct_epsilon_b1p05` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.27373 | 162727 | 0.144687 | 0.0357859 |
| `direct_epsilon_b1p05` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p05` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 283.09 | 0.00327892 | 0.00644488 |
| `direct_epsilon_b1p05` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0598239 | 0.584104 | -1.09122e-05 | 5.98577e-06 |
| `direct_epsilon_b1p05` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.55341 | 330.214 | 0.00384048 | 0.00825217 |
| `direct_epsilon_b1p05` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.61806 | 23833.3 | 0.0502323 | 0.0908181 |
| `direct_epsilon_b1p05` | `delayed_observation` | `position_only_observation` | evaluated | 13.287 | 2.90268e+06 | 0.508431 | 1.45427 |
| `direct_epsilon_b1p05` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.30134 | 162795 | 0.144557 | 0.0358589 |
| `direct_epsilon_b1p4` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p4` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 268.384 | 0.0029441 | 0.00633128 |
| `direct_epsilon_b1p4` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 268.384 | 0.0029441 | 0.00633128 |
| `direct_epsilon_b1p4` | `nominal` | `shuffled_observation_history` | evaluated | 0.481551 | 317.861 | 0.00352055 | 0.007993 |
| `direct_epsilon_b1p4` | `nominal` | `lagged_observation_history` | evaluated | 0.58709 | 22960.4 | 0.0486358 | 0.090203 |
| `direct_epsilon_b1p4` | `nominal` | `position_only_observation` | evaluated | 12.1499 | 3.01544e+06 | 0.548125 | 0.977185 |
| `direct_epsilon_b1p4` | `nominal` | `velocity_only_observation` | evaluated | 3.21629 | 166067 | 0.146027 | 0.0453758 |
| `direct_epsilon_b1p4` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p4` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.160274 | 894.475 | 0.00631939 | 0.00631181 |
| `direct_epsilon_b1p4` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.160274 | 894.475 | 0.00631939 | 0.00631181 |
| `direct_epsilon_b1p4` | `initial_state` | `shuffled_observation_history` | evaluated | 0.480394 | 320.783 | 0.0039174 | 0.00797149 |
| `direct_epsilon_b1p4` | `initial_state` | `lagged_observation_history` | evaluated | 0.0561038 | 0.921309 | -2.11206e-05 | 6.272e-05 |
| `direct_epsilon_b1p4` | `initial_state` | `position_only_observation` | evaluated | 11.9179 | 2.806e+06 | 0.526751 | 1.02332 |
| `direct_epsilon_b1p4` | `initial_state` | `velocity_only_observation` | evaluated | 3.17348 | 149802 | 0.138887 | 0.0453563 |
| `direct_epsilon_b1p4` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p4` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.50345 | 852.486 | 0.00563344 | 0.0172557 |
| `direct_epsilon_b1p4` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.50345 | 852.486 | 0.00563344 | 0.0172557 |
| `direct_epsilon_b1p4` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.487447 | 222.193 | 0.00309823 | 0.00624052 |
| `direct_epsilon_b1p4` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.59001 | 22482.9 | 0.0488209 | 0.0858787 |
| `direct_epsilon_b1p4` | `process_epsilon` | `position_only_observation` | evaluated | 12.2624 | 3.06917e+06 | 0.553858 | 0.97076 |
| `direct_epsilon_b1p4` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.26226 | 163228 | 0.146062 | 0.0425011 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.130312 | 218.886 | 0.00340478 | 0.0063654 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.130312 | 218.886 | 0.00340478 | 0.0063654 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.493773 | 652.99 | 0.00615303 | 0.0193512 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.614646 | 17703.5 | 0.0436067 | 0.0674783 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `position_only_observation` | evaluated | 12.1036 | 3.01539e+06 | 0.548586 | 0.977219 |
| `direct_epsilon_b1p4` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.18624 | 166017 | 0.146487 | 0.0454099 |
| `direct_epsilon_b1p4` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `direct_epsilon_b1p4` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 268.384 | 0.0029441 | 0.00633128 |
| `direct_epsilon_b1p4` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0563987 | 0.282695 | -2.09891e-05 | 5.66726e-05 |
| `direct_epsilon_b1p4` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.481551 | 317.861 | 0.00352055 | 0.007993 |
| `direct_epsilon_b1p4` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.58709 | 22960.4 | 0.0486358 | 0.090203 |
| `direct_epsilon_b1p4` | `delayed_observation` | `position_only_observation` | evaluated | 12.1499 | 3.01544e+06 | 0.548125 | 0.977185 |
| `direct_epsilon_b1p4` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.21629 | 166067 | 0.146027 | 0.0453758 |
| `linear_no_bias_b1p05` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias_b1p05` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 203.514 | 0.00122371 | 0.00667929 |
| `linear_no_bias_b1p05` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 203.514 | 0.00122371 | 0.00667929 |
| `linear_no_bias_b1p05` | `nominal` | `shuffled_observation_history` | evaluated | 0.443745 | 214.336 | 0.00146892 | 0.00707078 |
| `linear_no_bias_b1p05` | `nominal` | `lagged_observation_history` | evaluated | 0.578028 | 16926 | 0.0390384 | 0.0779088 |
| `linear_no_bias_b1p05` | `nominal` | `position_only_observation` | evaluated | 6.6919 | 4.38313e+06 | 0.6717 | 1.04535 |
| `linear_no_bias_b1p05` | `nominal` | `velocity_only_observation` | evaluated | 3.74411 | 168479 | 0.146958 | 0.0297177 |
| `linear_no_bias_b1p05` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias_b1p05` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.164289 | 657.229 | 0.00335493 | 0.00669437 |
| `linear_no_bias_b1p05` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.164289 | 657.229 | 0.00335493 | 0.00669437 |
| `linear_no_bias_b1p05` | `initial_state` | `shuffled_observation_history` | evaluated | 0.445304 | 216.438 | 0.00202365 | 0.00701735 |
| `linear_no_bias_b1p05` | `initial_state` | `lagged_observation_history` | evaluated | 0.0522433 | -2.97012 | -1.07271e-05 | 2.53166e-05 |
| `linear_no_bias_b1p05` | `initial_state` | `position_only_observation` | evaluated | 6.49668 | 4.13297e+06 | 0.655509 | 1.00269 |
| `linear_no_bias_b1p05` | `initial_state` | `velocity_only_observation` | evaluated | 3.70693 | 152083 | 0.140232 | 0.0297328 |
| `linear_no_bias_b1p05` | `process_epsilon` | `normal` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `process_epsilon` | `frozen_nominal_observation_tape` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `process_epsilon` | `shuffled_observation_history` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `process_epsilon` | `lagged_observation_history` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `process_epsilon` | `position_only_observation` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `process_epsilon` | `velocity_only_observation` | not_available | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias_b1p05` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0940261 | 196.722 | 0.00200733 | 0.00666969 |
| `linear_no_bias_b1p05` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0940261 | 196.722 | 0.00200733 | 0.00666969 |
| `linear_no_bias_b1p05` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.453338 | 381.876 | 0.00435423 | 0.0131038 |
| `linear_no_bias_b1p05` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.598256 | 13929.4 | 0.0362144 | 0.062638 |
| `linear_no_bias_b1p05` | `sensory_feedback` | `position_only_observation` | evaluated | 6.67548 | 4.38312e+06 | 0.672483 | 1.04534 |
| `linear_no_bias_b1p05` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.74158 | 168472 | 0.147741 | 0.0297081 |
| `linear_no_bias_b1p05` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `linear_no_bias_b1p05` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 203.514 | 0.00122371 | 0.00667929 |
| `linear_no_bias_b1p05` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.052945 | -4.50707 | -1.67113e-05 | 2.80425e-05 |
| `linear_no_bias_b1p05` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.443745 | 214.336 | 0.00146892 | 0.00707078 |
| `linear_no_bias_b1p05` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.578028 | 16926 | 0.0390384 | 0.0779088 |
| `linear_no_bias_b1p05` | `delayed_observation` | `position_only_observation` | evaluated | 6.6919 | 4.38313e+06 | 0.6717 | 1.04535 |
| `linear_no_bias_b1p05` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.74411 | 168479 | 0.146958 | 0.0297177 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `direct_epsilon_b1p05` | available | 2.05266 | 3.91205 | 0.193282 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `direct_epsilon_b1p4` | available | 1.9234 | 3.66559 | 0.181209 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `linear_no_bias_b1p05` | available | 1.10006 | 2.05857 | 0.141546 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `direct_epsilon_b1p05` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `direct_epsilon_b1p4` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `linear_no_bias_b1p05` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `direct_epsilon_b1p05` | 0 | 10000 | 2000 | -8000 | 128.623 | 6 |
| `direct_epsilon_b1p05` | 1 | 9500 | 1500 | -8000 | 110.316 | 6 |
| `direct_epsilon_b1p05` | 2 | 12000 | 1500 | -10500 | 136.949 | 6 |
| `direct_epsilon_b1p05` | 3 | 7500 | 2000 | -5500 | 124.126 | 6 |
| `direct_epsilon_b1p05` | 4 | 10000 | 2000 | -8000 | 124.493 | 6 |
| `direct_epsilon_b1p4` | 0 | 10000 | 1500 | -8500 | 137.448 | 6 |
| `direct_epsilon_b1p4` | 1 | 10000 | 1500 | -8500 | 110.919 | 6 |
| `direct_epsilon_b1p4` | 2 | 10000 | 1500 | -8500 | 122.383 | 6 |
| `direct_epsilon_b1p4` | 3 | 10000 | 2000 | -8000 | 126.196 | 6 |
| `direct_epsilon_b1p4` | 4 | 10000 | 2000 | -8000 | 124.288 | 6 |
| `linear_no_bias_b1p05` | 0 | 12000 | 1500 | -10500 | 160.29 | 4 |
| `linear_no_bias_b1p05` | 1 | 11500 | 1500 | -10000 | 109.814 | 4 |
| `linear_no_bias_b1p05` | 2 | 11500 | 1000 | -10500 | 152.978 | 4 |
| `linear_no_bias_b1p05` | 3 | 10500 | 2000 | -8500 | 186.943 | 4 |
| `linear_no_bias_b1p05` | 4 | 11000 | 1000 | -10000 | 138.892 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
