# GRU Feedback Ablation Diagnostic

- Issue: `c92ebd8`
- Source experiment: `c92ebd8`
- Scope: `output_feedback_budget_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `open_loop_moderate` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `moderate_pgd_ofb1p05` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `moderate_pgd_ofb1p4` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `open_loop_moderate` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `shuffled_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `lagged_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `position_only_observation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `nominal` | `velocity_only_observation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `lagged_observation_history` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `position_only_observation` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `initial_state` | `velocity_only_observation` | evaluated | 0.0468019 | -1.1159 | -2.00392e-05 | -2.11419e-05 |
| `open_loop_moderate` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `position_only_observation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0453868 | 1.60192 | -1.3905e-05 | -5.07907e-07 |
| `open_loop_moderate` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0474215 | -2.43589 | 7.73012e-06 | -1.48474e-05 |
| `open_loop_moderate` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0474215 | -2.43589 | 7.73012e-06 | -1.48474e-05 |
| `open_loop_moderate` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0474215 | -2.43589 | 7.73012e-06 | -1.48474e-05 |
| `open_loop_moderate` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0474215 | -2.43589 | 7.73012e-06 | -1.48474e-05 |
| `open_loop_moderate` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0474215 | -2.43589 | 7.73012e-06 | -1.48474e-05 |
| `open_loop_moderate` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0474215 | -2.43589 | 7.73012e-06 | -1.48474e-05 |
| `open_loop_moderate` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `open_loop_moderate` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `position_only_observation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `open_loop_moderate` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0478094 | -1.82372 | 1.14131e-05 | 6.10791e-06 |
| `moderate_pgd_ofb1p05` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p05` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `nominal` | `shuffled_observation_history` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `nominal` | `lagged_observation_history` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `nominal` | `position_only_observation` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `nominal` | `velocity_only_observation` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p05` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0613196 | -0.110586 | 9.45609e-07 | 2.24163e-06 |
| `moderate_pgd_ofb1p05` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0613196 | -0.110586 | 9.45609e-07 | 2.24163e-06 |
| `moderate_pgd_ofb1p05` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0613196 | -0.110586 | 9.45609e-07 | 2.24163e-06 |
| `moderate_pgd_ofb1p05` | `initial_state` | `lagged_observation_history` | evaluated | 0.0613196 | -0.110586 | 9.45609e-07 | 2.24163e-06 |
| `moderate_pgd_ofb1p05` | `initial_state` | `position_only_observation` | evaluated | 0.0613196 | -0.110586 | 9.45609e-07 | 2.24163e-06 |
| `moderate_pgd_ofb1p05` | `initial_state` | `velocity_only_observation` | evaluated | 0.0613196 | -0.110586 | 9.45609e-07 | 2.24163e-06 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0598915 | 2.36982 | 5.33892e-06 | -6.77352e-06 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0598915 | 2.36982 | 5.33892e-06 | -6.77352e-06 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0598915 | 2.36982 | 5.33892e-06 | -6.77352e-06 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0598915 | 2.36982 | 5.33892e-06 | -6.77352e-06 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `position_only_observation` | evaluated | 0.0598915 | 2.36982 | 5.33892e-06 | -6.77352e-06 |
| `moderate_pgd_ofb1p05` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.0598915 | 2.36982 | 5.33892e-06 | -6.77352e-06 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0613769 | -2.50166 | -2.87633e-06 | -9.91761e-06 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0613769 | -2.50166 | -2.87633e-06 | -9.91761e-06 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0613769 | -2.50166 | -2.87633e-06 | -9.91761e-06 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0613769 | -2.50166 | -2.87633e-06 | -9.91761e-06 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0613769 | -2.50166 | -2.87633e-06 | -9.91761e-06 |
| `moderate_pgd_ofb1p05` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0613769 | -2.50166 | -2.87633e-06 | -9.91761e-06 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `position_only_observation` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p05` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0624946 | -1.30036 | -8.62654e-06 | -5.12175e-06 |
| `moderate_pgd_ofb1p4` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p4` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `nominal` | `shuffled_observation_history` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `nominal` | `lagged_observation_history` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `nominal` | `position_only_observation` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `nominal` | `velocity_only_observation` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p4` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.07259 | 0.75782 | 5.99692e-06 | 3.80154e-06 |
| `moderate_pgd_ofb1p4` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.07259 | 0.75782 | 5.99692e-06 | 3.80154e-06 |
| `moderate_pgd_ofb1p4` | `initial_state` | `shuffled_observation_history` | evaluated | 0.07259 | 0.75782 | 5.99692e-06 | 3.80154e-06 |
| `moderate_pgd_ofb1p4` | `initial_state` | `lagged_observation_history` | evaluated | 0.07259 | 0.75782 | 5.99692e-06 | 3.80154e-06 |
| `moderate_pgd_ofb1p4` | `initial_state` | `position_only_observation` | evaluated | 0.07259 | 0.75782 | 5.99692e-06 | 3.80154e-06 |
| `moderate_pgd_ofb1p4` | `initial_state` | `velocity_only_observation` | evaluated | 0.07259 | 0.75782 | 5.99692e-06 | 3.80154e-06 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.071429 | 0.878771 | -2.04815e-06 | 1.45811e-05 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.071429 | 0.878771 | -2.04815e-06 | 1.45811e-05 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.071429 | 0.878771 | -2.04815e-06 | 1.45811e-05 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.071429 | 0.878771 | -2.04815e-06 | 1.45811e-05 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `position_only_observation` | evaluated | 0.071429 | 0.878771 | -2.04815e-06 | 1.45811e-05 |
| `moderate_pgd_ofb1p4` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.071429 | 0.878771 | -2.04815e-06 | 1.45811e-05 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.073263 | -0.548369 | 1.02998e-05 | 2.33664e-05 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.073263 | -0.548369 | 1.02998e-05 | 2.33664e-05 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.073263 | -0.548369 | 1.02998e-05 | 2.33664e-05 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.073263 | -0.548369 | 1.02998e-05 | 2.33664e-05 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `position_only_observation` | evaluated | 0.073263 | -0.548369 | 1.02998e-05 | 2.33664e-05 |
| `moderate_pgd_ofb1p4` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.073263 | -0.548369 | 1.02998e-05 | 2.33664e-05 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `position_only_observation` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |
| `moderate_pgd_ofb1p4` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0736447 | -0.233008 | 7.75584e-06 | 5.23734e-06 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `open_loop_moderate` | available | 0.00764209 | 0.0149277 | 0.000356476 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `moderate_pgd_ofb1p05` | available | 0.00940861 | 0.018312 | 0.00050519 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `moderate_pgd_ofb1p4` | available | 0.0102814 | 0.0203724 | 0.000190355 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback Pass Audit

| Run | Overall | Nominal gate | Dependence | Small perturbation | Sensory/delayed | Command | Warnings |
|---|---|---|---|---|---|---|---|
| `open_loop_moderate` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `moderate_pgd_ofb1p05` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |
| `moderate_pgd_ofb1p4` | `pass` | `pass` | `pass` | `pass` | `pass` | `pass` | none |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop_moderate` | 0 | 10500 | 2000 | -8500 | 278.379 | 6 |
| `open_loop_moderate` | 1 | 11500 | 4000 | -7500 | 297.101 | 6 |
| `open_loop_moderate` | 2 | 12000 | 2500 | -9500 | 279.278 | 6 |
| `open_loop_moderate` | 3 | 12000 | 2500 | -9500 | 301.124 | 6 |
| `open_loop_moderate` | 4 | 10000 | 2500 | -7500 | 307.908 | 6 |
| `moderate_pgd_ofb1p05` | 0 | 8500 | 3000 | -5500 | 306.46 | 6 |
| `moderate_pgd_ofb1p05` | 1 | 10000 | 2500 | -7500 | 293.669 | 6 |
| `moderate_pgd_ofb1p05` | 2 | 5500 | 2000 | -3500 | 288.363 | 6 |
| `moderate_pgd_ofb1p05` | 3 | 10000 | 2000 | -8000 | 276.631 | 6 |
| `moderate_pgd_ofb1p05` | 4 | 8500 | 2000 | -6500 | 267.833 | 6 |
| `moderate_pgd_ofb1p4` | 0 | 4000 | 1500 | -2500 | 278.648 | 6 |
| `moderate_pgd_ofb1p4` | 1 | 3500 | 1500 | -2000 | 263.26 | 6 |
| `moderate_pgd_ofb1p4` | 2 | 3500 | 2500 | -1000 | 302.239 | 6 |
| `moderate_pgd_ofb1p4` | 3 | 7500 | 1500 | -6000 | 304.085 | 6 |
| `moderate_pgd_ofb1p4` | 4 | 5000 | 3000 | -2000 | 323.568 | 6 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
