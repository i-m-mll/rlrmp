# Paired broad-epsilon attribution diagnostic

Active uses the run's actual training sampler, including target sampling, calibrated graph-channel perturbations, and broad/full-state epsilon. The paired condition replays the same sampler branches and rollout PRNG seed but removes only the broad-epsilon draw. The manifest separates `paired_without_broad`, `broad_delta`, and `active_total` epsilon arrays.

- Schema: `rlrmp.gru_broad_epsilon_attribution.v1`
- Rows: 8
- CSV summary: `results/b8aa38e/notes/gru_broad_epsilon_attribution_validation_selected.csv`
- Bulk arrays: `_artifacts/b8aa38e/broad_epsilon_attribution/gru_broad_epsilon_attribution_validation_selected`

| run | level | n | broad L2 mean | active loss | without-broad loss | delta | grad |
|---|---:|---:|---:|---:|---:|---:|---|
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64 | moderate | 8 | 0.000924323 | 2782.87 | 2778.52 | 4.3479 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64 | moderate | 8 | 0.000924323 | 2755.89 | 2752.84 | 3.05048 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64 | moderate | 8 | 0.000924323 | 3018.23 | 3011.12 | 7.1115 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64 | moderate | 8 | 0.00102703 | 3383.45 | 3379.81 | 3.64281 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64 | strong | 8 | 0.00174637 | 2788.6 | 2778.33 | 10.2628 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64 | strong | 8 | 0.00174637 | 2761.62 | 2754.02 | 7.59326 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64 | strong | 8 | 0.00174637 | 3050.31 | 3033.84 | 16.4712 | evaluated |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64 | strong | 8 | 0.00194041 | 3393.9 | 3384.64 | 9.25924 | evaluated |

Gradient attribution is raw pre-optimizer trainable-parameter gradient direction on the bounded replicate subset. Optimizer update direction is not materialized because validation-selected models are assembled from per-replicate checkpoints without a synchronized optimizer state.
