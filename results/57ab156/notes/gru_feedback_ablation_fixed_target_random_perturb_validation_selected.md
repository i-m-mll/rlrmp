# GRU Feedback Ablation Diagnostic

- Issue: `57ab156`
- Source experiment: `aacb9ed`
- Scope: `fixed_target_random_perturb_validation_selected`
- Checkpoint policy: `validation_selected_per_replicate`

## Interpretation

| Run | Label | Reason |
|---|---|---|
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `feedback_sensitive` | observation tape ablations changed actions above the diagnostic threshold |

## Ablation Deltas

| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | dEndpoint mean | dTerminal speed mean |
|---|---|---|---|---:|---:|---:|---:|
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 203.148 | 0.00151507 | 0.00578572 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0632616 | -3.87685 | 4.60227e-06 | 2.25092e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0632616 | -3.87685 | 4.60227e-06 | 2.25092e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0632616 | -3.87685 | 4.60227e-06 | 2.25092e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0632616 | -3.87685 | 4.60227e-06 | 2.25092e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0632616 | -3.87685 | 4.60227e-06 | 2.25092e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0636687 | -8.27658 | -2.67305e-05 | -7.35394e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0636687 | -8.27658 | -2.67305e-05 | -7.35394e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0636687 | -8.27658 | -2.67305e-05 | -7.35394e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.375065 | 5062.15 | 0.0216952 | 0.0115882 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0636687 | -8.27658 | -2.67305e-05 | -7.35394e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0636687 | -8.27658 | -2.67305e-05 | -7.35394e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.121054 | 85.3448 | 0.00205462 | -0.0367253 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.121054 | 85.3448 | 0.00205462 | -0.0367253 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.22549 | 223.058 | 1.23965e-05 | 0.000180244 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.39429 | 16453.3 | 0.0222088 | -0.0294958 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.35295 | 2.2241e+06 | 0.444432 | 0.923961 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.654263 | 57385.9 | 0.0544469 | 0.0989263 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0632616 | 336.131 | 0.00180837 | 0.00607572 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0586687 | -3.60715 | 6.97577e-05 | -7.37024e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0586687 | -3.60715 | 6.97577e-05 | -7.37024e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0586687 | -3.60715 | 6.97577e-05 | -7.37024e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.30195 | 2.01362e+06 | 0.438718 | 1.01833 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0586687 | -3.60715 | 6.97577e-05 | -7.37024e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0586416 | -0.867845 | 6.07719e-05 | -0.000117248 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0586416 | -0.867845 | 6.07719e-05 | -0.000117248 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0586416 | -0.867845 | 6.07719e-05 | -0.000117248 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.37833 | 3921.28 | 0.0157542 | 0.0236763 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.29696 | 2.00611e+06 | 0.437751 | 1.01772 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0586416 | -0.867845 | 6.07719e-05 | -0.000117248 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 0 | 215.013 | 0.0019352 | 0.00545842 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0634881 | -4.55938 | 2.50672e-05 | -1.91734e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 0.0634881 | -4.55938 | 2.50672e-05 | -1.91734e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.0634881 | -4.55938 | 2.50672e-05 | -1.91734e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 0.0634881 | -4.55938 | 2.50672e-05 | -1.91734e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.0634881 | -4.55938 | 2.50672e-05 | -1.91734e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0640599 | -9.3486 | -3.6198e-05 | -1.31689e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0640599 | -9.3486 | -3.6198e-05 | -1.31689e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 0.0640599 | -9.3486 | -3.6198e-05 | -1.31689e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.420384 | 5248.07 | 0.0222487 | 0.0107635 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 0.0640599 | -9.3486 | -3.6198e-05 | -1.31689e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.0640599 | -9.3486 | -3.6198e-05 | -1.31689e-05 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.127075 | 15.5467 | 0.00252064 | -0.0440949 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.127075 | 15.5467 | 0.00252064 | -0.0440949 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 0.303272 | 232.175 | 1.35315e-05 | 0.000158093 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.44101 | 17004.2 | 0.0230336 | -0.031917 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.26983 | 2.00931e+06 | 0.423787 | 0.843126 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.747056 | 66208.9 | 0.0598705 | 0.100135 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0634881 | 351.526 | 0.00213148 | 0.00642917 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0569572 | -4.609 | 3.26635e-05 | -0.00013867 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.0569572 | -4.609 | 3.26635e-05 | -0.00013867 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.0569572 | -4.609 | 3.26635e-05 | -0.00013867 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 3.24695 | 1.80615e+06 | 0.417978 | 0.938581 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.0569572 | -4.609 | 3.26635e-05 | -0.00013867 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.0570341 | -1.61496 | 3.27021e-05 | -9.74617e-06 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0570341 | -1.61496 | 3.27021e-05 | -9.74617e-06 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 0.0570341 | -1.61496 | 3.27021e-05 | -9.74617e-06 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.429044 | 4316.83 | 0.0169653 | 0.0234527 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.2386 | 1.79601e+06 | 0.416655 | 0.936197 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.0570341 | -1.61496 | 3.27021e-05 | -9.74617e-06 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
