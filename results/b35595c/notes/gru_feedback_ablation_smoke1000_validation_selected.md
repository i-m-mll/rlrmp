# GRU Feedback Ablation Diagnostic

- Issue: `b35595c`
- Source experiment: `b35595c`
- Scope: `postrun_feedback_ablation`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.15179e-15 | -92.089 | -0.000669411 | 0.000102657 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.15179e-15 | -92.089 | -0.000669411 | 0.000102657 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.15179e-15 | -92.089 | -0.000669411 | 0.000102657 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.330341 | 2510.97 | 0.00195234 | 0.00377972 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 2.40841 | 682430 | 0.213311 | 0.652341 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.42279 | 161430 | 0.140072 | -0.00744422 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0711146 | 270.846 | -0.00234192 | -0.00938891 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0711146 | 270.846 | -0.00234192 | -0.00938891 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.00706e-15 | -98.0494 | -0.00106472 | -0.000182202 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.330664 | 2515.28 | 0.00663409 | -0.0135873 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 2.41678 | 637573 | 0.212655 | 0.598945 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.43633 | 140275 | 0.131797 | -0.0169358 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.00964448 | -114.609 | -0.00126924 | -0.00384603 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.00964448 | -114.609 | -0.00126924 | -0.00384603 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.15128e-15 | -96.2385 | -0.000748538 | 1.10197e-05 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.329688 | 2663.2 | 0.00286371 | 0.00133229 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 2.40728 | 687754 | 0.214729 | 0.653179 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.42157 | 160120 | 0.139831 | -0.00868317 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0537827 | 39.6392 | 0.0013717 | -0.00164256 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.030738 | 11.8223 | 0.000481834 | 0.000430358 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.030738 | 11.8223 | 0.000481834 | 0.000430358 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.030738 | 11.8223 | 0.000481834 | 0.000430358 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 2.37358 | 687029 | 0.21557 | 0.653248 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.030738 | 11.8223 | 0.000481834 | 0.000430358 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.118276 | 18.4965 | 0.00193818 | -0.00126995 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.118276 | 18.4965 | 0.00193818 | -0.00126995 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.08279e-15 | 224.276 | 0.0012424 | 0.000925778 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.325616 | 3350.16 | 0.0051328 | 0.00437964 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 2.36953 | 692554 | 0.216813 | 0.659862 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.39199 | 162191 | 0.14228 | -0.0085407 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.17945e-15 | 203.796 | 0.00128988 | 0.00371749 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0424337 | -45.5649 | -0.000350946 | -7.8143e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0424337 | -45.5649 | -0.000350946 | -7.8143e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.426744 | 2610.44 | 0.00858926 | 0.0160564 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0424337 | -45.5649 | -0.000350946 | -7.8143e-05 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 3.32807 | 160446 | 0.142141 | 0.000664566 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0925764 | 812.364 | 0.00305244 | 0.00623033 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0925764 | 812.364 | 0.00305244 | 0.00623033 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.19022e-15 | 227.233 | 0.00131458 | 0.00652316 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.431648 | 2824.96 | 0.0111158 | 0.0143167 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.54983 | 947509 | 0.298723 | 0.472298 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 3.34766 | 139242 | 0.132533 | 0.00317741 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.0424208 | -48.4689 | -0.000372853 | -0.000235406 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.010697 | 209.029 | 0.00107008 | 0.00485801 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.14764e-15 | 203.865 | 0.00131972 | 0.00411808 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0424208 | -48.4689 | -0.000372853 | -0.000235406 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.54253 | 1.03239e+06 | 0.307239 | 0.525911 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.32733 | 159509 | 0.141837 | 0.00109736 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.117457 | 437.896 | 0.00309825 | 0.0124214 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0389314 | 12.4051 | -1.87828e-05 | -0.00167533 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.116895 | 104.655 | 0.00128911 | 0.00762466 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0389314 | 12.4051 | -1.87828e-05 | -0.00167533 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.4789 | 1.02777e+06 | 0.307474 | 0.524619 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0389314 | 12.4051 | -1.87828e-05 | -0.00167533 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.291006 | 413.866 | 0.00243455 | 0.0109958 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0382282 | -15.8854 | -0.000429892 | -0.00192203 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.21995e-15 | 170.833 | -0.000322179 | 0.0100185 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.424703 | 4513.15 | 0.0148719 | 0.0297927 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.58248 | 1.02546e+06 | 0.307103 | 0.516972 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 3.37134 | 161172 | 0.142719 | 0.000890856 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | available | 0.540997 | 1.04191 | 0.0400809 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | available | 0.594396 | 1.05214 | 0.136651 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 0 | 1000 | 1000 | 0 | -192.554 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 1 | 1000 | 1000 | 0 | -72.2498 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 2 | 1000 | 1000 | 0 | -193.655 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 3 | 1000 | 1000 | 0 | -221.22 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 4 | 1000 | 1000 | 0 | -305.531 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 0 | 1000 | 1000 | 0 | -250.779 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 1 | 1000 | 1000 | 0 | -108.554 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 2 | 1000 | 1000 | 0 | -38.7237 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 3 | 1000 | 1000 | 0 | -167.541 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 4 | 1000 | 1000 | 0 | -122.186 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
