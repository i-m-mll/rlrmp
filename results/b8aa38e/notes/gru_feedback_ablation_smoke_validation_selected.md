# GRU Feedback Ablation Diagnostic

- Issue: `b8aa38e`
- Source experiment: `b8aa38e`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `smoke__broad_strong_cal_small` | `motor_tape_like` | all evaluated feedback ablations produced small action changes |
| `smoke__proprio_cal_stress` | `motor_tape_like` | all evaluated feedback ablations produced small action changes |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `smoke__broad_strong_cal_small` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__broad_strong_cal_small` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.3503e-06 | 38.5369 | 2.45486e-05 | -6.1597e-05 |
| `smoke__broad_strong_cal_small` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.3503e-06 | 38.5369 | 2.45486e-05 | -6.1597e-05 |
| `smoke__broad_strong_cal_small` | `nominal` | `shuffled_observation_history` | evaluated | 1.3503e-06 | 38.5369 | 2.45486e-05 | -6.1597e-05 |
| `smoke__broad_strong_cal_small` | `nominal` | `lagged_observation_history` | evaluated | 1.3503e-06 | 38.5369 | 2.45486e-05 | -6.1597e-05 |
| `smoke__broad_strong_cal_small` | `nominal` | `position_only_observation` | evaluated | 1.3503e-06 | 38.5369 | 2.45486e-05 | -6.1597e-05 |
| `smoke__broad_strong_cal_small` | `nominal` | `velocity_only_observation` | evaluated | 1.3503e-06 | 38.5369 | 2.45486e-05 | -6.1597e-05 |
| `smoke__broad_strong_cal_small` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__broad_strong_cal_small` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 1.30786e-06 | 34.8419 | 2.38415e-05 | -5.93297e-05 |
| `smoke__broad_strong_cal_small` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 1.30786e-06 | 34.8419 | 2.38415e-05 | -5.93297e-05 |
| `smoke__broad_strong_cal_small` | `initial_state` | `shuffled_observation_history` | evaluated | 1.30786e-06 | 34.8419 | 2.38415e-05 | -5.93297e-05 |
| `smoke__broad_strong_cal_small` | `initial_state` | `lagged_observation_history` | evaluated | 1.30786e-06 | 34.8419 | 2.38415e-05 | -5.93297e-05 |
| `smoke__broad_strong_cal_small` | `initial_state` | `position_only_observation` | evaluated | 1.30786e-06 | 34.8419 | 2.38415e-05 | -5.93297e-05 |
| `smoke__broad_strong_cal_small` | `initial_state` | `velocity_only_observation` | evaluated | 1.30786e-06 | 34.8419 | 2.38415e-05 | -5.93297e-05 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 1.34854e-06 | 37.9528 | 2.45628e-05 | -6.27111e-05 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 1.34854e-06 | 37.9528 | 2.45628e-05 | -6.27111e-05 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.34854e-06 | 37.9528 | 2.45628e-05 | -6.27111e-05 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `lagged_observation_history` | evaluated | 1.34854e-06 | 37.9528 | 2.45628e-05 | -6.27111e-05 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `position_only_observation` | evaluated | 1.34854e-06 | 37.9528 | 2.45628e-05 | -6.27111e-05 |
| `smoke__broad_strong_cal_small` | `process_epsilon` | `velocity_only_observation` | evaluated | 1.34854e-06 | 37.9528 | 2.45628e-05 | -6.27111e-05 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 1.34507e-06 | -39.94 | -2.6701e-05 | 9.63703e-05 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 1.34507e-06 | -39.94 | -2.6701e-05 | 9.63703e-05 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.34507e-06 | -39.94 | -2.6701e-05 | 9.63703e-05 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `lagged_observation_history` | evaluated | 1.34507e-06 | -39.94 | -2.6701e-05 | 9.63703e-05 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `position_only_observation` | evaluated | 1.34507e-06 | -39.94 | -2.6701e-05 | 9.63703e-05 |
| `smoke__broad_strong_cal_small` | `sensory_feedback` | `velocity_only_observation` | evaluated | 1.34507e-06 | -39.94 | -2.6701e-05 | 9.63703e-05 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 1.34509e-06 | -39.9577 | -2.67279e-05 | 9.67919e-05 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 1.34509e-06 | -39.9577 | -2.67279e-05 | 9.67919e-05 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.34509e-06 | -39.9577 | -2.67279e-05 | 9.67919e-05 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `lagged_observation_history` | evaluated | 1.34509e-06 | -39.9577 | -2.67279e-05 | 9.67919e-05 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `position_only_observation` | evaluated | 1.34509e-06 | -39.9577 | -2.67279e-05 | 9.67919e-05 |
| `smoke__broad_strong_cal_small` | `delayed_observation` | `velocity_only_observation` | evaluated | 1.34509e-06 | -39.9577 | -2.67279e-05 | 9.67919e-05 |
| `smoke__proprio_cal_stress` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__proprio_cal_stress` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 4.23914e-05 | 34.7606 | 2.20128e-05 | -5.48373e-05 |
| `smoke__proprio_cal_stress` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 4.23914e-05 | 34.7606 | 2.20128e-05 | -5.48373e-05 |
| `smoke__proprio_cal_stress` | `nominal` | `shuffled_observation_history` | evaluated | 4.23914e-05 | 34.7606 | 2.20128e-05 | -5.48373e-05 |
| `smoke__proprio_cal_stress` | `nominal` | `lagged_observation_history` | evaluated | 4.23914e-05 | 34.7606 | 2.20128e-05 | -5.48373e-05 |
| `smoke__proprio_cal_stress` | `nominal` | `position_only_observation` | evaluated | 4.23914e-05 | 34.7606 | 2.20128e-05 | -5.48373e-05 |
| `smoke__proprio_cal_stress` | `nominal` | `velocity_only_observation` | evaluated | 4.23914e-05 | 34.7606 | 2.20128e-05 | -5.48373e-05 |
| `smoke__proprio_cal_stress` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__proprio_cal_stress` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 4.13472e-05 | 31.3581 | 2.13652e-05 | -5.31277e-05 |
| `smoke__proprio_cal_stress` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 4.13472e-05 | 31.3581 | 2.13652e-05 | -5.31277e-05 |
| `smoke__proprio_cal_stress` | `initial_state` | `shuffled_observation_history` | evaluated | 4.13472e-05 | 31.3581 | 2.13652e-05 | -5.31277e-05 |
| `smoke__proprio_cal_stress` | `initial_state` | `lagged_observation_history` | evaluated | 4.13472e-05 | 31.3581 | 2.13652e-05 | -5.31277e-05 |
| `smoke__proprio_cal_stress` | `initial_state` | `position_only_observation` | evaluated | 4.13472e-05 | 31.3581 | 2.13652e-05 | -5.31277e-05 |
| `smoke__proprio_cal_stress` | `initial_state` | `velocity_only_observation` | evaluated | 4.13472e-05 | 31.3581 | 2.13652e-05 | -5.31277e-05 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 4.11249e-05 | 32.3696 | 2.06132e-05 | -4.98584e-05 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 4.11249e-05 | 32.3696 | 2.06132e-05 | -4.98584e-05 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `shuffled_observation_history` | evaluated | 4.11249e-05 | 32.3696 | 2.06132e-05 | -4.98584e-05 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `lagged_observation_history` | evaluated | 4.11249e-05 | 32.3696 | 2.06132e-05 | -4.98584e-05 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `position_only_observation` | evaluated | 4.11249e-05 | 32.3696 | 2.06132e-05 | -4.98584e-05 |
| `smoke__proprio_cal_stress` | `process_epsilon` | `velocity_only_observation` | evaluated | 4.11249e-05 | 32.3696 | 2.06132e-05 | -4.98584e-05 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 4.98088e-05 | -34.9855 | -2.383e-05 | 9.32157e-05 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 4.98088e-05 | -34.9855 | -2.383e-05 | 9.32157e-05 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 4.98088e-05 | -34.9855 | -2.383e-05 | 9.32157e-05 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `lagged_observation_history` | evaluated | 4.98088e-05 | -34.9855 | -2.383e-05 | 9.32157e-05 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `position_only_observation` | evaluated | 4.98088e-05 | -34.9855 | -2.383e-05 | 9.32157e-05 |
| `smoke__proprio_cal_stress` | `sensory_feedback` | `velocity_only_observation` | evaluated | 4.98088e-05 | -34.9855 | -2.383e-05 | 9.32157e-05 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 5.02469e-05 | -34.8118 | -2.37602e-05 | 9.39636e-05 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 5.02469e-05 | -34.8118 | -2.37602e-05 | 9.39636e-05 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `shuffled_observation_history` | evaluated | 5.02469e-05 | -34.8118 | -2.37602e-05 | 9.39636e-05 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `lagged_observation_history` | evaluated | 5.02469e-05 | -34.8118 | -2.37602e-05 | 9.39636e-05 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `position_only_observation` | evaluated | 5.02469e-05 | -34.8118 | -2.37602e-05 | 9.39636e-05 |
| `smoke__proprio_cal_stress` | `delayed_observation` | `velocity_only_observation` | evaluated | 5.02469e-05 | -34.8118 | -2.37602e-05 | 9.39636e-05 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `smoke__broad_strong_cal_small` | available | 0.000142015 | 3.83097e-05 | 0.000245721 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `smoke__proprio_cal_stress` | available | 0.000741598 | 0.00126075 | 0.000222444 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `smoke__broad_strong_cal_small` | 0 | 20 | 20 | 0 | -5789.15 | 4 |
| `smoke__proprio_cal_stress` | 0 | 20 | 10 | -10 | -5734.16 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
