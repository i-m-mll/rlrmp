# GRU Feedback Ablation Diagnostic

- Issue: `e4800d6`
- Source experiment: `e4800d6`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 259.602 | 0.00337337 | 0.00527238 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0711659 | -1.0042 | -4.64179e-06 | -4.87033e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0711659 | -1.0042 | -4.64179e-06 | -4.87033e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.664544 | 26203.7 | 0.0536301 | 0.0926378 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0711659 | -1.0042 | -4.64179e-06 | -4.87033e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 7.06542 | 293582 | 0.146232 | 0.526205 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0715135 | -0.808853 | -5.78179e-06 | -6.8206e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0715135 | -0.808853 | -5.78179e-06 | -6.8206e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0715135 | -0.808853 | -5.78179e-06 | -6.8206e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0715135 | -0.808853 | -5.78179e-06 | -6.8206e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 14.8701 | 3.73805e+06 | 0.568635 | 1.62495 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 7.07487 | 271841 | 0.13646 | 0.526321 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0707527 | 1.94441 | 1.54789e-07 | -1.76591e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0707527 | 1.94441 | 1.54789e-07 | -1.76591e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0707527 | 1.94441 | 1.54789e-07 | -1.76591e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0707527 | 1.94441 | 1.54789e-07 | -1.76591e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0707527 | 1.94441 | 1.54789e-07 | -1.76591e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0707527 | 1.94441 | 1.54789e-07 | -1.76591e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.306532 | 73.9388 | 0.00336537 | 0.00567571 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.306532 | 73.9388 | 0.00336537 | 0.00567571 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 15.1151 | 4.09735e+06 | 0.607915 | 1.51759 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0700438 | -1.09117 | 2.81211e-06 | -1.01433e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 276.22 | 0.00339919 | 0.0059424 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0848576 | -1.7827 | -1.1807e-06 | 2.30351e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0848576 | -1.7827 | -1.1807e-06 | 2.30351e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.894568 | 30747.7 | 0.0583614 | 0.102432 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0848576 | -1.7827 | -1.1807e-06 | 2.30351e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 4.62765 | 175731 | 0.153612 | 0.0660025 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0846034 | -1.61636 | -6.5704e-07 | -9.47367e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0846034 | -1.61636 | -6.5704e-07 | -9.47367e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0846034 | -1.61636 | -6.5704e-07 | -9.47367e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0846034 | -1.61636 | -6.5704e-07 | -9.47367e-07 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 19.2847 | 4.57464e+06 | 0.34376 | 3.704 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 4.55269 | 153799 | 0.14364 | 0.0660267 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.085086 | 1.70557 | -8.97111e-06 | -1.63572e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.085086 | 1.70557 | -8.97111e-06 | -1.63572e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.085086 | 1.70557 | -8.97111e-06 | -1.63572e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.085086 | 1.70557 | -8.97111e-06 | -1.63572e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.085086 | 1.70557 | -8.97111e-06 | -1.63572e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.085086 | 1.70557 | -8.97111e-06 | -1.63572e-05 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.561606 | 37.2249 | 0.0033783 | 0.00587111 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.561606 | 37.2249 | 0.0033783 | 0.00587111 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 19.7634 | 4.76119e+06 | 0.396219 | 3.62626 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0844495 | -1.45488 | 3.75136e-06 | 2.09574e-06 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | available | 2.07178 | 4.12839 | 0.0151623 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | available | 2.49246 | 4.97752 | 0.00739381 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | 0 | 11000 | 2000 | -9000 | 92.3287 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | 1 | 11000 | 2500 | -8500 | 101.985 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | 2 | 12000 | 2500 | -9500 | 99.7673 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | 3 | 8500 | 2500 | -6000 | 87.2327 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | 4 | 11500 | 2500 | -9000 | 92.8869 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | 0 | 10000 | 2000 | -8000 | 95.8325 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | 1 | 10500 | 1500 | -9000 | 87.9571 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | 2 | 7500 | 2500 | -5000 | 97.5581 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | 3 | 11000 | 2000 | -9000 | 108.527 | 8 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | 4 | 12000 | 1500 | -10500 | 110.407 | 8 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
