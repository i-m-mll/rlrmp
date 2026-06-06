# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

## Component Status

| Component | Status | Source |
|---|---:|---|
| evaluation_diagnostics | available | results/b35595c/notes/gru_evaluation_diagnostics_calibrated_perturb_level_screen_validation_selected.json |
| exact_audit | missing | source not provided |
| feedback_ablation | available | results/b35595c/notes/gru_feedback_ablation_calibrated_perturb_level_screen_validation_selected.json |
| induced_gain | missing | source not provided |
| map_error_decomposition | available | results/b35595c/notes/gru_map_error_decomposition_calibrated_perturb_level_screen_validation_selected.json |
| objective_comparator | available | results/b35595c/notes/objective_comparator_calibrated_perturb_level_screen_validation_selected.json |
| perturbation_response | available | _artifacts/b35595c/notes/gru_perturbation_response_calibrated_perturb_level_screen_validation_selected_manifest.json |
| standard_certificate | available | results/b35595c/notes/gru_standard_certificates_calibrated_perturb_level_screen_validation_selected_manifest.json |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
