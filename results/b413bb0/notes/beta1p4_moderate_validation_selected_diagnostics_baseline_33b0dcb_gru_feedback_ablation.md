# GRU Feedback Ablation Diagnostic

- Issue: `b413bb0`
- Source experiment: `33b0dcb`
- Scope: `beta1p4_moderate_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 241.965 | 0.00294318 | 0.00425508 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 241.965 | 0.00294318 | 0.00425508 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0437957 | -2.06248 | 7.91507e-06 | 1.11702e-05 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.511607 | 19026.5 | 0.0432488 | 0.0792021 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 9.35707 | 4.39294e+06 | 0.679232 | 0.380274 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.36784 | 165966 | 0.143922 | 0.0624688 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.167384 | 2018.3 | 0.0106109 | 0.00685776 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.167384 | 2018.3 | 0.0106109 | 0.00685776 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.357152 | 250.097 | 0.00176307 | 0.00579142 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.513228 | 20671.2 | 0.0456788 | 0.0767418 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 9.18745 | 3.88331e+06 | 0.642962 | 0.237317 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.3433 | 134299 | 0.126919 | 0.0650714 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.415385 | 3285.18 | 0.0125873 | 0.041389 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.415385 | 3285.18 | 0.0125873 | 0.041389 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.355795 | 211.878 | 0.0026356 | 0.00417396 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.511943 | 20061.8 | 0.0454179 | 0.0784699 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 9.56128 | 4.53082e+06 | 0.690287 | 0.383245 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0417839 | 2.15496 | 1.20519e-05 | -3.10402e-05 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0433782 | -2.64396 | -2.6034e-05 | 5.24499e-06 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.11184 | 201.13 | 0.00319864 | 0.00572555 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.37707 | 458.676 | 0.00640353 | 0.014892 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.544603 | 13711.3 | 0.0374752 | 0.0539497 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 9.33875 | 4.3929e+06 | 0.679487 | 0.381744 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.34632 | 165925 | 0.144177 | 0.0639392 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0 | 241.965 | 0.00294318 | 0.00425508 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0 | 241.965 | 0.00294318 | 0.00425508 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.35829 | 240.021 | 0.00284968 | 0.00420957 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.511607 | 19026.5 | 0.0432488 | 0.0792021 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0437957 | -2.06248 | 7.91507e-06 | 1.11702e-05 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.36784 | 165966 | 0.143922 | 0.0624688 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | available | 1.67721 | 2.93826 | 0.416169 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | 0 | 7500 | 3500 | -4000 | 289.202 | 6 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | 1 | 6000 | 11000 | 5000 | 308.219 | 6 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | 2 | 11000 | 12000 | 1000 | 311.066 | 6 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | 3 | 10500 | 3500 | -7000 | 290.141 | 6 |
| `h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64` | 4 | 9500 | 2500 | -7000 | 300.51 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
