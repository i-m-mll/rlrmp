# GRU Feedback Ablation Diagnostic

- Issue: `aacb9ed`
- Source experiment: `aacb9ed`
- Scope: `feedback_v2_fixed_target_random_perturb_validation_selected`
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
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.10403e-15 | 49.7494 | -0.000826178 | 0.00258253 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.10403e-15 | 49.7494 | -0.000826178 | 0.00258253 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.10403e-15 | 49.7494 | -0.000826178 | 0.00258253 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.371513 | 3815.56 | 0.0154972 | 0.0225356 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.2988 | 2.00558e+06 | 0.437629 | 1.01339 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.537682 | 29762.9 | 0.0456554 | 0.142051 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0595784 | 634.442 | 0.00587184 | -0.00152973 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0595784 | 634.442 | 0.00587184 | -0.00152973 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.1117e-15 | 109.561 | 0.00155728 | 0.000568251 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.372829 | 5157.74 | 0.0227151 | 0.0133896 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.24951 | 1.90311e+06 | 0.432888 | 0.961086 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.568301 | 36199.9 | 0.0578345 | 0.137939 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.122348 | 560.733 | 0.00312002 | -0.0396077 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.122348 | 560.733 | 0.00312002 | -0.0396077 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.11673e-15 | 650.38 | 0.00106597 | -0.00147024 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.392934 | 17098.3 | 0.023294 | -0.0316206 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.35039 | 2.22257e+06 | 0.444962 | 0.920205 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.645003 | 57972.9 | 0.0550968 | 0.099861 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0596339 | 9.04997 | -0.000320754 | 0.000415447 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0498921 | 399.6 | 0.00158727 | 0.00691014 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.00302366 | 192.162 | 0.000762402 | 0.00307095 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.374515 | 4289.24 | 0.0164918 | 0.0273952 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0596339 | 9.04997 | -0.000320754 | 0.000415447 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.544247 | 30005.2 | 0.0459229 | 0.142232 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.217714 | 401.459 | 0.0021693 | 0.0077389 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.217714 | 401.459 | 0.0021693 | 0.0077389 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.2071e-15 | 208.179 | 0.00140215 | 0.00362897 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.385873 | 4118.01 | 0.0165279 | 0.0264718 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.29589 | 2.00234e+06 | 0.437536 | 1.0174 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.567502 | 30393.2 | 0.0469182 | 0.143045 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `frozen_nominal_observation_tape` | evaluated | 1.01095e-15 | 71.0924 | -0.000490052 | 0.00222176 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `zeroed_perturbation_observation_deviation` | evaluated | 1.01095e-15 | 71.0924 | -0.000490052 | 0.00222176 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `shuffled_observation_history` | evaluated | 1.01095e-15 | 71.0924 | -0.000490052 | 0.00222176 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `lagged_observation_history` | evaluated | 0.424551 | 4028.83 | 0.0163412 | 0.0245897 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `position_only_observation` | evaluated | 3.24018 | 1.79767e+06 | 0.416742 | 0.936509 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `nominal` | `velocity_only_observation` | evaluated | 0.651764 | 35544.4 | 0.0508304 | 0.150983 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `frozen_nominal_observation_tape` | evaluated | 0.0657495 | 671.167 | 0.00607927 | -0.00278283 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0657495 | 671.167 | 0.00607927 | -0.00278283 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `shuffled_observation_history` | evaluated | 1.02469e-15 | 118.733 | 0.00178729 | 0.000395911 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `lagged_observation_history` | evaluated | 0.425722 | 5237.65 | 0.0230683 | 0.0136327 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `position_only_observation` | evaluated | 3.17799 | 1.68789e+06 | 0.410303 | 0.880274 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `initial_state` | `velocity_only_observation` | evaluated | 0.675982 | 42594.8 | 0.0629374 | 0.145979 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `frozen_nominal_observation_tape` | evaluated | 0.128151 | 462.86 | 0.0035605 | -0.0489148 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `zeroed_perturbation_observation_deviation` | evaluated | 0.128151 | 462.86 | 0.0035605 | -0.0489148 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `shuffled_observation_history` | evaluated | 1.02064e-15 | 625.083 | 0.00100667 | -0.00140109 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `lagged_observation_history` | evaluated | 0.44618 | 17422.2 | 0.0239037 | -0.0335429 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `position_only_observation` | evaluated | 3.26381 | 2.00638e+06 | 0.424341 | 0.839306 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon` | `velocity_only_observation` | evaluated | 0.741652 | 66476.9 | 0.0604615 | 0.0998469 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `frozen_nominal_observation_tape` | evaluated | 0.0569843 | 13.4527 | -0.000360057 | 0.00029139 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `zeroed_perturbation_observation_deviation` | evaluated | 0.0535206 | 454.42 | 0.00220218 | 0.00718243 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `shuffled_observation_history` | evaluated | 0.00205438 | 259.754 | 0.0016878 | 0.00457774 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `lagged_observation_history` | evaluated | 0.428716 | 4629.42 | 0.0173601 | 0.0281657 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `position_only_observation` | evaluated | 0.0569843 | 13.4527 | -0.000360057 | 0.00029139 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `sensory_feedback` | `velocity_only_observation` | evaluated | 0.670233 | 36090.3 | 0.0513735 | 0.151108 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `normal` | evaluated | 0 | 0 | 0 | 0 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `frozen_nominal_observation_tape` | evaluated | 0.305129 | 449.89 | 0.00278445 | 0.00594607 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `zeroed_perturbation_observation_deviation` | evaluated | 0.305129 | 449.89 | 0.00278445 | 0.00594607 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `shuffled_observation_history` | evaluated | 1.15069e-15 | 260.578 | 0.00206733 | 0.00402841 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `lagged_observation_history` | evaluated | 0.441978 | 4552.54 | 0.0176995 | 0.0246307 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `position_only_observation` | evaluated | 3.24219 | 1.79206e+06 | 0.416462 | 0.934586 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `delayed_observation` | `velocity_only_observation` | evaluated | 0.688075 | 36698.9 | 0.0524961 | 0.150851 |

## Normalized Feedback-Use Indices

| Run | Status | Score | Ablation dependence | Perturbation rescue | Correction vs open-loop | Warnings |
|---|---|---:|---:|---:|---:|---|
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | available | 0.585195 | 1.03674 | 0.133645 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | available | 0.575771 | 1.01158 | 0.139958 | n/a | correction_index_vs_open_loop not available: open-loop data not supplied |

## Feedback-Selected Checkpoint Audit

- Status: `materialized`
- Selection use: `audit_only_not_primary_checkpoint_selection`
- Primary checkpoint policy: `validation_selected_per_replicate`

| Run | Replicate | Validation checkpoint | Feedback checkpoint | Feedback - validation | Feedback score | Bins |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | 0 | 10500 | 1000 | -9500 | 482.848 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | 1 | 11000 | 1500 | -9500 | 2308.19 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | 2 | 8500 | 2000 | -6500 | 2851.15 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | 3 | 10500 | 1500 | -9000 | 2650.52 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | 4 | 12000 | 1000 | -11000 | 1870.03 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | 0 | 8500 | 500 | -8000 | -1807.13 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | 1 | 5000 | 1500 | -3500 | 2671.51 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | 2 | 8500 | 5500 | -3000 | 3508.56 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | 3 | 10500 | 500 | -10000 | 553.071 | 4 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | 4 | 12000 | 2000 | -10000 | 2988.54 | 4 |

## Notes

- `normal` is the per-bin baseline; all other rows are paired deltas against it.
- `frozen_nominal_observation_tape` and `zeroed_perturbation_observation_deviation` are separate diagnostic lenses that share the nominal feedback tape for perturbed bins.
- Validation-selected checkpoints are used for model loading. Analytical action/I/O metrics, where present in adjacent diagnostics, remain audit-only.
