# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

Regeneration spec: `results/e4800d6/notes/hinf_phenotype_sidecar_h0_sisu_spectrum_targetfix_two_rows_validation_selected_regeneration_spec.json`

## Component Status

| Component | Status | Source |
|---|---:|---|
| broad_epsilon_attribution | available | results/e4800d6/notes/h0_sisu_spectrum_targetfix_two_rows_validation_selected_broad_epsilon_attribution.json |
| evaluation_diagnostics | available | results/e4800d6/notes/gru_evaluation_diagnostics_h0_sisu_spectrum_targetfix_two_rows_validation_selected.json |
| exact_audit | available | results/e4800d6/notes/gru_worst_case_epsilon_audit_h0_sisu_spectrum_targetfix_two_rows_validation_selected.json |
| feedback_ablation | available | results/e4800d6/notes/gru_feedback_ablation_h0_sisu_spectrum_targetfix_two_rows_validation_selected.json |
| induced_gain | missing | source not provided |
| map_error_decomposition | available | results/e4800d6/notes/gru_map_error_decomposition_h0_sisu_spectrum_targetfix_two_rows_validation_selected.json |
| objective_comparator | available | results/e4800d6/notes/objective_comparator_h0_sisu_spectrum_targetfix_two_rows_validation_selected.json |
| perturbation_response | available | _artifacts/e4800d6/notes/gru_perturbation_response_h0_sisu_spectrum_targetfix_two_rows_validation_selected_calibrated_manifest.json |
| standard_certificate | available | results/e4800d6/notes/gru_standard_certificates_h0_sisu_spectrum_targetfix_two_rows_validation_selected_manifest.json |
| worst_case_epsilon_audit | missing | source not provided |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |
| cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64 | not_claimed | available | available | available | available | 1 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
