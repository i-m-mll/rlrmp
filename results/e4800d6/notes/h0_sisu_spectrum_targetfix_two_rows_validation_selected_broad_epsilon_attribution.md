# Paired broad-epsilon attribution diagnostic

Active uses the run's actual training sampler, including target sampling, calibrated graph-channel perturbations, and broad/full-state epsilon. The paired condition replays the same sampler branches and rollout PRNG seed but removes only the broad-epsilon draw. The manifest separates `paired_without_broad`, `broad_delta`, and `active_total` epsilon arrays.

- Schema: `rlrmp.gru_broad_epsilon_attribution.v1`
- Rows: 2
- CSV summary: `results/e4800d6/notes/h0_sisu_spectrum_targetfix_two_rows_validation_selected_broad_epsilon_attribution.csv`
- Bulk arrays: `_artifacts/e4800d6/broad_epsilon_attribution/h0_sisu_spectrum_targetfix_two_rows_validation_selected_broad_epsilon_attribution`

| run | level | n | broad L2 mean | active loss | without-broad loss | delta | grad |
|---|---:|---:|---:|---:|---:|---:|---|
| cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64 | moderate |  |  |  |  |  | not_applicable: paired broad-epsilon attribution requires a BroadFullStateEpsilonTrainingTaskAdapter in the training task stack |
| cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64 | moderate |  |  |  |  |  | not_applicable: paired broad-epsilon attribution requires a BroadFullStateEpsilonTrainingTaskAdapter in the training task stack |

Gradient attribution is raw pre-optimizer trainable-parameter gradient direction on the bounded replicate subset. Optimizer update direction is not materialized because validation-selected models are assembled from per-replicate checkpoints without a synchronized optimizer state.
