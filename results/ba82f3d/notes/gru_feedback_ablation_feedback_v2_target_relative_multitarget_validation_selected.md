# GRU Feedback Ablation Diagnostic

- Issue: `ba82f3d`
- Source experiment: `ba82f3d`
- Scope: `feedback_v2_target_relative_multitarget_validation_selected`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.08064e-15 | 190.958 | 0.000866298 | 0.00612882 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.08064e-15 | 190.958 | 0.000866298 | 0.00612882 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.08064e-15 | 190.958 | 0.000866298 | 0.00612882 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0344115 | -24.6716 | 8.26112e-05 | -0.000227532 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 4.18306 | 1.02527e+06 | 0.326836 | 0.30776 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0344115 | -24.6716 | 8.26112e-05 | -0.000227532 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.034042 | -15.7575 | 0.000106666 | -7.06471e-05 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.15077 | 1140.59 | 0.00695338 | 0.00482004 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.1142e-15 | 208.49 | 0.00166344 | 0.00489137 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.477451 | 6648.45 | 0.024931 | 0.0293632 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.98639 | 919375 | 0.313084 | 0.276422 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.034042 | -15.7575 | 0.000106666 | -7.06471e-05 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.281097 | -267.186 | 0.00651235 | -0.0829842 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0351307 | 2.05251 | -8.10107e-05 | 0.0010098 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0351307 | 2.05251 | -8.10107e-05 | 0.0010098 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0351307 | 2.05251 | -8.10107e-05 | 0.0010098 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 4.51097 | 1.18127e+06 | 0.331383 | 0.0590766 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.5326 | 66628.3 | 0.0577741 | -0.0911799 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0344115 | 683.003 | 0.00346055 | 0.011109 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0308308 | 16.6204 | -0.000485893 | 0.000203365 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.10145e-15 | 210.677 | 0.00122561 | 0.0043665 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.489631 | 6297.06 | 0.0214956 | 0.0354018 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 4.19524 | 1.02311e+06 | 0.326504 | 0.307221 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.31396 | 166745 | 0.146592 | -0.00182488 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0310325 | 7.89186 | -0.000466006 | 0.000668676 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0310325 | 7.89186 | -0.000466006 | 0.000668676 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0310325 | 7.89186 | -0.000466006 | 0.000668676 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.0310325 | 7.89186 | -0.000466006 | 0.000668676 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.0310325 | 7.89186 | -0.000466006 | 0.000668676 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0310325 | 7.89186 | -0.000466006 | 0.000668676 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.20794e-15 | 141.734 | 0.000319658 | 0.00575337 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.20794e-15 | 141.734 | 0.000319658 | 0.00575337 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.20794e-15 | 141.734 | 0.000319658 | 0.00575337 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0416259 | -20.1851 | 5.03638e-06 | -0.000365247 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 6.07794 | 1.3162e+06 | 0.371766 | 0.168953 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0416259 | -20.1851 | 5.03638e-06 | -0.000365247 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0407934 | -16.7074 | 2.59211e-06 | -0.000334622 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.159547 | 1084.98 | 0.00698982 | 0.00495469 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.18902e-15 | 153.225 | 0.00148074 | 0.00534467 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.825879 | 10047.8 | 0.0311613 | 0.0449178 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 5.74565 | 1.1741e+06 | 0.353861 | 0.163571 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0407934 | -16.7074 | 2.59211e-06 | -0.000334622 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.250101 | -429.732 | 0.00673079 | -0.0777308 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0425558 | -12.767 | -2.65459e-05 | 0.000459383 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.0425558 | -12.767 | -2.65459e-05 | 0.000459383 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.0425558 | -12.767 | -2.65459e-05 | 0.000459383 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 6.5758 | 1.51487e+06 | 0.374479 | 0.179569 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 3.57123 | 67011.3 | 0.0581362 | -0.0862689 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0416259 | 581.832 | 0.00282075 | 0.0104075 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0428786 | 17.4236 | -0.000401137 | 8.21277e-05 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 1.26155e-15 | 255.878 | 0.00181025 | 0.00381159 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.838744 | 10010 | 0.0289964 | 0.0508237 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 6.08051 | 1.31277e+06 | 0.371357 | 0.168139 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 3.40195 | 167435 | 0.146974 | -0.00237349 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.043484 | 9.61723 | -0.000387459 | 0.000261848 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.043484 | 9.61723 | -0.000387459 | 0.000261848 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.043484 | 9.61723 | -0.000387459 | 0.000261848 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.043484 | 9.61723 | -0.000387459 | 0.000261848 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 0.043484 | 9.61723 | -0.000387459 | 0.000261848 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.043484 | 9.61723 | -0.000387459 | 0.000261848 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | available | 0.743882 | 1.27176 | 0.216004 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | available | 1.02707 | 1.83873 | 0.215405 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | 0 | 8000 | 500 | -7500 | -373.458 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | 1 | 10500 | 500 | -10000 | -937.922 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | 2 | 10000 | 500 | -9500 | -2587.97 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | 3 | 11500 | 500 | -11000 | 1326.66 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | 4 | 12000 | 500 | -11500 | -356.16 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | 0 | 6500 | 500 | -6000 | 416.362 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | 1 | 7000 | 500 | -6500 | 2151.97 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | 2 | 6500 | 500 | -6000 | 1571.6 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | 3 | 5500 | 500 | -5000 | 2536.19 | 4 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | 4 | 5000 | 2500 | -2500 | 3345.1 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
