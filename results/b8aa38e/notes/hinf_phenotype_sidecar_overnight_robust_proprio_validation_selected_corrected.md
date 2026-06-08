# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

## Component Status

| Component | Status | Source |
|---|---:|---|
| evaluation_diagnostics | available | results/b8aa38e/notes/gru_evaluation_diagnostics_overnight_robust_proprio_validation_selected_corrected.json |
| exact_audit | missing | source not provided |
| feedback_ablation | available | results/b8aa38e/notes/gru_feedback_ablation_overnight_robust_proprio_validation_selected_corrected.json |
| induced_gain | missing | source not provided |
| map_error_decomposition | available | results/b8aa38e/notes/gru_map_error_decomposition_overnight_robust_proprio_validation_selected_corrected.json |
| objective_comparator | available | results/b8aa38e/notes/objective_comparator_overnight_robust_proprio_validation_selected_corrected.json |
| perturbation_response | available | _artifacts/b8aa38e/notes/gru_perturbation_response_overnight_robust_proprio_validation_selected_corrected_manifest.json |
| standard_certificate | available | results/b8aa38e/notes/gru_standard_certificates_overnight_robust_proprio_validation_selected_corrected_manifest.json |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
