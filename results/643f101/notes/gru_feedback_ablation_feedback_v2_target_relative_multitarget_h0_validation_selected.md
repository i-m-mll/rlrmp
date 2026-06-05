# GRU Feedback Ablation Diagnostic

- Issue: `643f101`
- Source experiment: `643f101`
- Scope: `feedback_v2_target_relative_multitarget_h0_validation_selected`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0354041 | -15.6581 | 6.0809e-05 | -8.08393e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.09524e-15 | 101.954 | 8.40935e-05 | 0.00527726 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.09524e-15 | 101.954 | 8.40935e-05 | 0.00527726 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.403708 | 3934.54 | 0.0157949 | 0.0226423 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0354041 | -15.6581 | 6.0809e-05 | -8.08393e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0354041 | -15.6581 | 6.0809e-05 | -8.08393e-05 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0355142 | -5.49294 | 7.25767e-05 | -0.000165708 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0355142 | -5.49294 | 7.25767e-05 | -0.000165708 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.05215e-15 | 118.579 | 0.00139513 | 0.00320068 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0355142 | -5.49294 | 7.25767e-05 | -0.000165708 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0355142 | -5.49294 | 7.25767e-05 | -0.000165708 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.19354 | 147014 | 0.138711 | -0.00356638 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.239364 | 240.366 | 0.00402421 | -0.0716928 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0354969 | 1.60687 | -2.73239e-05 | 0.000577828 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.15256e-15 | 666.451 | 0.00100112 | -0.00104836 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.448843 | 16463.8 | 0.0228764 | -0.0372015 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0354969 | 1.60687 | -2.73239e-05 | 0.000577828 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.40851 | 69160 | 0.0565663 | -0.0784576 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0385476 | 36.239 | -0.000288528 | 0.00045019 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0385476 | 36.239 | -0.000288528 | 0.00045019 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0385476 | 36.239 | -0.000288528 | 0.00045019 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.505731 | 854.941 | 0.00402814 | 0.00306388 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.8142 | 1.1136e+06 | 0.335474 | 0.484489 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.35812 | 168787 | 0.148067 | -0.00263769 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0362802 | 25.0416 | -0.000422749 | 0.000272656 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0362802 | 25.0416 | -0.000422749 | 0.000272656 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 9.62414e-16 | 207.081 | 0.0015756 | 0.00537811 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0362802 | 25.0416 | -0.000422749 | 0.000272656 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.70497 | 1.11911e+06 | 0.335704 | 0.476892 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0362802 | 25.0416 | -0.000422749 | 0.000272656 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0.0351336 | -18.9507 | -5.3394e-05 | -0.000457371 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.06556e-15 | 102.948 | 0.000160863 | 0.00491916 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.06556e-15 | 102.948 | 0.000160863 | 0.00491916 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.416579 | 6310.19 | 0.021748 | 0.033851 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0351336 | -18.9507 | -5.3394e-05 | -0.000457371 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0351336 | -18.9507 | -5.3394e-05 | -0.000457371 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0348105 | -13.7474 | -5.00346e-05 | -0.000409781 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0348105 | -13.7474 | -5.00346e-05 | -0.000409781 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.12681e-15 | 120.932 | 0.00164084 | 0.0035995 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.0348105 | -13.7474 | -5.00346e-05 | -0.000409781 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0348105 | -13.7474 | -5.00346e-05 | -0.000409781 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.19615 | 146425 | 0.138396 | -0.00536422 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.262709 | -340.431 | 0.00498286 | -0.0844859 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0355043 | 9.45922 | -5.30357e-05 | 0.000924122 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.12843e-15 | 574.746 | 0.000801768 | -0.00027614 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.469906 | 20750.9 | 0.028103 | -0.0454237 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 0.0355043 | 9.45922 | -5.30357e-05 | 0.000924122 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.44299 | 67826.5 | 0.0568733 | -0.0921063 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0387999 | 43.3437 | 3.73835e-05 | 0.000571853 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0387999 | 43.3437 | 3.73835e-05 | 0.000571853 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0387999 | 43.3437 | 3.73835e-05 | 0.000571853 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.4958 | 2548.9 | 0.011685 | 0.0133491 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.26895 | 1.12471e+06 | 0.343124 | 0.366293 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.34624 | 168183 | 0.147989 | -0.003805 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0362893 | 31.043 | -0.00024169 | 0.00043338 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0362893 | 31.043 | -0.00024169 | 0.00043338 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.14739e-15 | 282.512 | 0.00229659 | 0.00244426 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0362893 | 31.043 | -0.00024169 | 0.00043338 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 4.16182 | 1.12934e+06 | 0.343297 | 0.360395 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0362893 | 31.043 | -0.00024169 | 0.00043338 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | available | 0.575889 | 1.1395 | 0.0122801 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | available | 0.648536 | 1.28748 | 0.00958979 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 0 | 5500 | 500 | -5000 | 1427.27 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 1 | 5500 | 500 | -5000 | 121.968 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 2 | 5000 | 500 | -4500 | -525.946 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 3 | 6000 | 500 | -5500 | 125.767 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 4 | 5500 | 500 | -5000 | 297.039 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 0 | 3000 | 500 | -2500 | 983.643 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 1 | 4500 | 2000 | -2500 | 2293.87 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 2 | 3000 | 500 | -2500 | 1422.32 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 3 | 2000 | 500 | -1500 | 815.64 | 4 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 4 | 3000 | 500 | -2500 | 1965.27 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
