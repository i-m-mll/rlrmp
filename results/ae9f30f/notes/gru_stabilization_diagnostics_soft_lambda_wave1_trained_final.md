# ae9f30f Stabilization Diagnostics

This is a reader-oriented reshaping of existing perturbation-response and feedback-ablation outputs. It does not rerun diagnostics or training. `linear_no_bias_b1p4` remains stopped context and is not included.

## Baseline Choice

The primary comparison row in this note is now labeled `020a65b_h0_no_pgd_calibrated`. That is the actual source of the current perturbation-response and feedback-use baseline values: `_artifacts/020a65b/perturbation_response/gru_h0_pgd_bank_two_rows_validation_selected_calibrated/gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected_calibrated_manifest_detail.json` plus the 020a65b feedback-ablation note.

I did not swap the primary baseline to the explicit `const_band16` run, because the available 3244f1a/33b0dcb artifact is a separate perturbation-response artifact and does not provide the feedback-use baseline used here. It is the right family and order of magnitude, so a compact side check is included below as `33b0dcb_const_band16_moderate`.

## What Counts as Stabilization Here

The perturbation bank is calibrated relative to a 15 cm reach, but the evaluated rows here are stabilization tests: the model is disturbed and the question is how well it returns to the intended movement. `initial_state` offsets test recovery from a wrong starting position or velocity. `process_epsilon` and `command_input` rows inject plant-side or command-path disturbances during movement. `sensory_feedback` rows perturb what the controller senses.

The only target/reach-goal perturbation family in this bank is `target_stream_jump`. It is not applicable for these fixed-target checkpoints, so it should not be read as failed stabilization evidence.

## Units and Columns

- endpoint delta mm: terminal position change in millimeters; source values are meters.
- endpoint/reach: terminal position change divided by the 0.15 m reach length, shown as percent.
- AUC dx mm*s: area under the position-deviation curve; source values are m*s.
- eval / blocked / n/a: evaluated rows, blocked rows, and not-applicable rows in the group.
- dFull-QRF: extra realized full Q/R/Q_f cost caused by the perturbation; lower is better.
- feedback-use score: diagnostic index combining feedback-ablation dependence with perturbation-response rescue, not a pass/fail gate.

## Sensory vs Legacy Non-Sensory Overview

This keeps the previous sensory/non-sensory grouping for continuity. The legacy non-sensory aggregate combines initial-state, command-input, and process-epsilon rows, so use the channel-split table for interpretation.

| row | group | eval / blocked / n/a | endpoint delta mm | endpoint/reach | AUC dx mm*s | dFull-QRF |
| --- | --- | --- | --- | --- | --- | --- |
| `direct_epsilon_b1p05` | sensory | 108 / 0 / 0 | 1.628 | 1.09% | 0.879 | 426.7 |
| `direct_epsilon_b1p05` | legacy non-sensory | 186 / 36 / 0 | 3.371 | 2.25% | 2.144 | 952.7 |
| `direct_epsilon_b1p4` | sensory | 108 / 0 / 0 | 1.685 | 1.12% | 0.902 | 395.7 |
| `direct_epsilon_b1p4` | legacy non-sensory | 186 / 36 / 0 | 3.715 | 2.48% | 2.289 | 978.4 |
| `linear_no_bias_b1p05` | sensory | 108 / 0 / 0 | 1.162 | 0.77% | 0.931 | 386.4 |
| `linear_no_bias_b1p05` | legacy non-sensory | 78 / 144 / 0 | 0.990 | 0.66% | 1.739 | 380.8 |
| `020a65b_h0_no_pgd_calibrated` | sensory | 108 / 0 / 0 | 0.987 | 0.66% | 0.809 | 422.3 |
| `020a65b_h0_no_pgd_calibrated` | legacy non-sensory | 222 / 0 / 0 | 4.549 | 3.03% | 2.697 | 1595.0 |

## Stabilization Channel Split

This is the safer table to read. It separates initial-state correction from plant/command disturbances and shows where blocked process-epsilon rows limit the evidence. The `linear_no_bias_b1p05` plant/command row has only 54 evaluated rows and 144 blocked rows, so that comparison is fragile.

| row | channel | eval / blocked / n/a | endpoint delta mm | endpoint/reach | AUC dx mm*s | peak dx/open-loop | dFull-QRF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_epsilon_b1p05` | initial state | 24 / 0 / 0 | 0.129 | 0.09% | 3.046 | 0.607 | 145.6 |
| `direct_epsilon_b1p05` | plant/command | 162 / 36 / 0 | 3.851 | 2.57% | 2.010 | 0.423 | 1072.2 |
| `direct_epsilon_b1p05` | sensory | 108 / 0 / 0 | 1.628 | 1.09% | 0.879 | 0.169 | 426.7 |
| `direct_epsilon_b1p05` | target/reach | 0 / 0 / 1 | n/a | n/a | n/a | n/a | n/a |
| `direct_epsilon_b1p4` | initial state | 24 / 0 / 0 | 0.368 | 0.25% | 3.219 | 0.606 | 153.0 |
| `direct_epsilon_b1p4` | plant/command | 162 / 36 / 0 | 4.211 | 2.81% | 2.151 | 0.427 | 1100.7 |
| `direct_epsilon_b1p4` | sensory | 108 / 0 / 0 | 1.685 | 1.12% | 0.902 | 0.147 | 395.7 |
| `direct_epsilon_b1p4` | target/reach | 0 / 0 / 1 | n/a | n/a | n/a | n/a | n/a |
| `linear_no_bias_b1p05` | initial state | 24 / 0 / 0 | 0.161 | 0.11% | 3.433 | 0.612 | 163.8 |
| `linear_no_bias_b1p05` | plant/command | 54 / 144 / 0 | 1.358 | 0.91% | 0.987 | 0.204 | 477.2 |
| `linear_no_bias_b1p05` | sensory | 108 / 0 / 0 | 1.162 | 0.77% | 0.931 | 0.122 | 386.4 |
| `linear_no_bias_b1p05` | target/reach | 0 / 0 / 1 | n/a | n/a | n/a | n/a | n/a |
| `020a65b_h0_no_pgd_calibrated` | initial state | 24 / 0 / 0 | 3.696 | 2.46% | 5.337 | 0.616 | 763.5 |
| `020a65b_h0_no_pgd_calibrated` | plant/command | 198 / 0 / 0 | 4.652 | 3.10% | 2.377 | 0.447 | 1695.8 |
| `020a65b_h0_no_pgd_calibrated` | sensory | 108 / 0 / 0 | 0.987 | 0.66% | 0.809 | 0.068 | 422.3 |
| `020a65b_h0_no_pgd_calibrated` | target/reach | 0 / 0 / 1 | n/a | n/a | n/a | n/a | n/a |

## Moderate Plant/Command Disturbances by Timing

These are stabilization diagnostics during the reach, not target/reach-goal perturbations. Counts include blocked process-epsilon rows.

| row | timing | eval / blocked / n/a | endpoint delta mm | endpoint/reach | AUC dx mm*s | dFull-QRF |
| --- | --- | --- | --- | --- | --- | --- |
| `direct_epsilon_b1p05` | early | 18 / 4 / 0 | 0.344 | 0.23% | 1.467 | 81.8 |
| `direct_epsilon_b1p05` | mid | 18 / 4 / 0 | 1.445 | 0.96% | 1.602 | 216.3 |
| `direct_epsilon_b1p05` | late | 18 / 4 / 0 | 6.497 | 4.33% | 1.468 | 957.1 |
| `direct_epsilon_b1p4` | early | 18 / 4 / 0 | 0.650 | 0.43% | 1.637 | 91.1 |
| `direct_epsilon_b1p4` | mid | 18 / 4 / 0 | 1.861 | 1.24% | 1.742 | 227.2 |
| `direct_epsilon_b1p4` | late | 18 / 4 / 0 | 6.378 | 4.25% | 1.476 | 974.6 |
| `linear_no_bias_b1p05` | early | 6 / 16 / 0 | 0.128 | 0.09% | 0.764 | 21.6 |
| `linear_no_bias_b1p05` | mid | 6 / 16 / 0 | 0.390 | 0.26% | 0.833 | 53.4 |
| `linear_no_bias_b1p05` | late | 6 / 16 / 0 | 1.368 | 0.91% | 0.629 | 432.1 |
| `020a65b_h0_no_pgd_calibrated` | early | 22 / 0 / 0 | 1.571 | 1.05% | 2.054 | 205.4 |
| `020a65b_h0_no_pgd_calibrated` | mid | 22 / 0 / 0 | 2.221 | 1.48% | 1.915 | 323.3 |
| `020a65b_h0_no_pgd_calibrated` | late | 22 / 0 / 0 | 5.072 | 3.38% | 1.381 | 1427.9 |

## Explicit Const-Band16 Side Check

This side check uses the existing explicit const-band16 artifact from 3244f1a/33b0dcb: `_artifacts/3244f1a/perturbation_response/gru_targetsupport_const_band16_calibrated_moderate/gru_perturbation_response_const_band16_calibrated_moderate_manifest_detail.json`. It is not the primary baseline above because it is a separate run/scope and does not include the feedback-use baseline. It does show that the baseline family is broadly comparable in scale.

| channel | eval / blocked / n/a | endpoint delta mm | endpoint/reach | AUC dx mm*s | dFull-QRF |
| --- | --- | --- | --- | --- | --- |
| initial state | 8 / 0 / 0 | 1.135 | 0.76% | 3.158 | 137.0 |
| plant/command | 66 / 0 / 0 | 3.692 | 2.46% | 1.741 | 692.0 |
| sensory | 36 / 0 / 0 | 0.841 | 0.56% | 0.621 | 134.7 |
| target/reach | 0 / 0 / 1 | n/a | n/a | n/a | n/a |

Moderate plant/command timing side values:

| timing | eval / blocked / n/a | endpoint delta mm | endpoint/reach | AUC dx mm*s | dFull-QRF |
| --- | --- | --- | --- | --- | --- |
| early | 22 / 0 / 0 | 1.742 | 1.16% | 1.953 | 171.4 |
| mid | 22 / 0 / 0 | 2.725 | 1.82% | 1.857 | 321.8 |
| late | 22 / 0 / 0 | 6.608 | 4.41% | 1.412 | 1582.8 |

## Feedback-Ablation Status

Feedback ablation ran for the three completed ae9f30f rows. The baseline feedback-use row is from 020a65b, matching the primary baseline label above. The feedback-ablation pipeline emits paired action, full-QRF, endpoint, terminal-speed deltas, and normalized feedback-use indices; it does not emit AUC. AUC values in this note come from perturbation-response diagnostics.

| row | feedback-use score | ablation dependence | perturbation rescue | status |
| --- | --- | --- | --- | --- |
| `direct_epsilon_b1p05` | 2.053 | 3.912 | 0.193 | available |
| `direct_epsilon_b1p4` | 1.923 | 3.666 | 0.181 | available |
| `linear_no_bias_b1p05` | 1.100 | 2.059 | 0.142 | available |
| `020a65b_h0_no_pgd_calibrated` | 0.031 | 0.015 | 0.046 | available |

## Plain-Language Interpretation

The direct-epsilon rows show stronger feedback-use scores than the 020a65b calibrated H0/no-PGD baseline, but they are not a uniform stabilization win across every channel. Their plant/command endpoint deltas and dFull-QRF are lower than the 020a65b primary baseline in the aggregate, while sensory endpoint/reach is higher than that baseline. `linear_no_bias_b1p05` has small endpoint deltas in the evaluated plant/command rows, but those values rest on only 54 evaluated rows with 144 process-epsilon rows blocked, and the row ended training with a zero adversary. Treat it as fragile evidence, not as a solved robust-training result.

The tiny endpoint and AUC values are expected once displayed in human-scale units: sub-millimeter to few-millimeter endpoint changes and mm*s AUCs. The issue was presentation and provenance, not an aggregation-unit bug.

## Caveats

- The primary baseline is `020a65b_h0_no_pgd_calibrated`, not the explicit 33b0dcb const-band16 run.
- The explicit const-band16 side check is perturbation-response only and should not be mixed into the feedback-use score.
- `target_stream_jump` is not applicable for fixed-target checkpoints.
- `linear_no_bias_b1p05` plant/command comparisons are fragile because the process-epsilon family is blocked.

## Source Files

- Summary input: `results/ae9f30f/notes/gru_perturbation_feedback_summary_soft_lambda_wave1_trained_final.json`
- ae9f30f perturbation detail: `_artifacts/ae9f30f/perturbation_response/gru_soft_lambda_wave1_trained_final/gru_perturbation_response_soft_lambda_wave1_trained_final_manifest_detail.json`
- primary baseline perturbation detail: `_artifacts/020a65b/perturbation_response/gru_h0_pgd_bank_two_rows_validation_selected_calibrated/gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected_calibrated_manifest_detail.json`
- explicit const-band16 side detail: `_artifacts/3244f1a/perturbation_response/gru_targetsupport_const_band16_calibrated_moderate/gru_perturbation_response_const_band16_calibrated_moderate_manifest_detail.json`
- companion JSON: `results/ae9f30f/notes/gru_stabilization_diagnostics_soft_lambda_wave1_trained_final.json`
