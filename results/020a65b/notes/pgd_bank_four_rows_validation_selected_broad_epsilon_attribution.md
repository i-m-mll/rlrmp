# Paired broad-epsilon attribution diagnostic

Active uses the run's actual training sampler, including target sampling, calibrated graph-channel perturbations, and broad/full-state epsilon. The paired condition replays the same sampler branches and rollout PRNG seed but removes only the broad-epsilon draw. The manifest separates `paired_without_broad`, `broad_delta`, and `active_total` epsilon arrays.

- Schema: `rlrmp.gru_broad_epsilon_attribution.v1`
- Rows: 4
- CSV summary: `results/020a65b/notes/pgd_bank_four_rows_validation_selected_broad_epsilon_attribution.csv`
- Bulk arrays: `_artifacts/020a65b/broad_epsilon_attribution/pgd_bank_four_rows_validation_selected_broad_epsilon_attribution`

| run | level | n | broad L2 mean | active loss | without-broad loss | delta | grad |
|---|---:|---:|---:|---:|---:|---:|---|
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64 | moderate |  |  |  |  |  | not_applicable: paired broad-epsilon attribution requires a BroadFullStateEpsilonTrainingTaskAdapter in the training task stack |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64 | moderate |  |  |  |  |  | not_applicable: paired broad-epsilon attribution requires a BroadFullStateEpsilonTrainingTaskAdapter in the training task stack |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64 | moderate |  |  |  |  |  | not_applicable: paired broad-epsilon attribution requires a BroadFullStateEpsilonTrainingTaskAdapter in the training task stack |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64 | moderate |  |  |  |  |  | not_applicable: paired broad-epsilon attribution requires a BroadFullStateEpsilonTrainingTaskAdapter in the training task stack |

Gradient attribution is raw pre-optimizer trainable-parameter gradient direction on the bounded replicate subset. Optimizer update direction is not materialized because validation-selected models are assembled from per-replicate checkpoints without a synchronized optimizer state.
