# GRU Feedback Ablation Diagnostic

- Issue: `5f70333`
- Source experiment: `5f70333`
- Scope: `feedback_v2_nominal_calibration_baseline`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.25115e-15 | 206.988 | 0.00243413 | 0.00527607 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.25115e-15 | 206.988 | 0.00243413 | 0.00527607 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.25115e-15 | 206.988 | 0.00243413 | 0.00527607 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.33504 | 2025.46 | 0.00962593 | 0.00679737 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 2.48382 | 1.55531e+06 | 0.37878 | 0.993699 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.246228 | 9288.19 | 0.0217482 | 0.0805304 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.02498 | 259.964 | 0.000860055 | 0.00166099 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.02498 | 259.964 | 0.000860055 | 0.00166099 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.20965e-15 | 45.1484 | -0.000586306 | 0.00398476 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.334651 | 3563.27 | 0.0156948 | 0.00062895 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 2.46597 | 1.52647e+06 | 0.378717 | 0.961917 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.253244 | 12572.7 | 0.0304896 | 0.0769153 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0613806 | -849.288 | -0.000915786 | -0.0161882 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0613806 | -849.288 | -0.000915786 | -0.0161882 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.25047e-15 | -833.717 | -0.00184937 | 0.00341224 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.339134 | 11708.9 | 0.0158697 | -0.0153352 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 2.47006 | 1.75285e+06 | 0.384945 | 0.953828 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.299123 | 23619.7 | 0.028732 | 0.0590661 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0521983 | -1.00298 | 0.000146148 | -0.000343171 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0558352 | 139.118 | 0.00249806 | 0.00616887 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0521983 | -1.00298 | 0.000146148 | -0.000343171 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0521983 | -1.00298 | 0.000146148 | -0.000343171 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 2.46649 | 1.56305e+06 | 0.379715 | 0.995882 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0521983 | -1.00298 | 0.000146148 | -0.000343171 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.147727 | 131.92 | 0.00230527 | 0.00486266 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.147727 | 131.92 | 0.00230527 | 0.00486266 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.17625e-15 | 116.265 | 0.000236721 | 0.00195041 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.330747 | 2261.65 | 0.0103492 | 0.00810184 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 2.46405 | 1.56103e+06 | 0.379299 | 0.993222 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.235116 | 9118.22 | 0.0212016 | 0.0797262 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | available | 0.416825 | 0.777155 | 0.0564951 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 0 | 11000 | 500 | -10500 | -592.969 | 4 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 1 | 11000 | 1000 | -10000 | 2560.63 | 4 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 2 | 10500 | 500 | -10000 | 547.918 | 4 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 3 | 11500 | 500 | -11000 | 2570.3 | 4 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 4 | 11000 | 3500 | -7500 | 2871.57 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
