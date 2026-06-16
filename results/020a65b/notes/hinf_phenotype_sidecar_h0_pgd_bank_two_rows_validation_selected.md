# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

Regeneration spec: `results/020a65b/notes/hinf_phenotype_sidecar_h0_pgd_bank_two_rows_validation_selected_regeneration_spec.json`

## Component Status

| Component | Status | Source |
|---|---:|---|
| broad_epsilon_attribution | available | results/020a65b/notes/h0_pgd_bank_two_rows_validation_selected_broad_epsilon_attribution.json |
| evaluation_diagnostics | available | results/020a65b/notes/gru_evaluation_diagnostics_h0_pgd_bank_two_rows_validation_selected.json |
| exact_audit | available | results/020a65b/notes/gru_worst_case_epsilon_audit_h0_pgd_bank_two_rows_validation_selected.json |
| feedback_ablation | available | results/020a65b/notes/gru_feedback_ablation_h0_pgd_bank_two_rows_validation_selected.json |
| induced_gain | missing | source not provided |
| map_error_decomposition | available | results/020a65b/notes/gru_map_error_decomposition_h0_pgd_bank_two_rows_validation_selected.json |
| objective_comparator | available | results/020a65b/notes/objective_comparator_h0_pgd_bank_two_rows_validation_selected.json |
| perturbation_response | available | _artifacts/020a65b/notes/gru_perturbation_response_h0_pgd_bank_two_rows_validation_selected_calibrated_manifest.json |
| standard_certificate | available | results/020a65b/notes/gru_standard_certificates_h0_pgd_bank_two_rows_validation_selected_manifest.json |
| worst_case_epsilon_audit | missing | source not provided |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
