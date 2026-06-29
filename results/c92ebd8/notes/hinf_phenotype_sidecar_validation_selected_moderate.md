# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

Regeneration spec: `results/c92ebd8/notes/hinf_phenotype_sidecar_validation_selected_moderate_regeneration_spec.json`

## Component Status

| Component | Status | Source |
|---|---:|---|
| broad_epsilon_attribution | missing | source not provided |
| evaluation_diagnostics | available | results/c92ebd8/notes/gru_evaluation_diagnostics_validation_selected_moderate.json |
| exact_audit | missing | source not provided |
| feedback_ablation | available | results/c92ebd8/notes/gru_feedback_ablation_validation_selected_moderate_manifest.json |
| induced_gain | missing | source not provided |
| map_error_decomposition | available | results/c92ebd8/notes/gru_map_error_decomposition_validation_selected_moderate.json |
| objective_comparator | available | results/c92ebd8/notes/objective_comparator_validation_selected_moderate.json |
| perturbation_response | available | results/c92ebd8/notes/gru_perturbation_response_validation_selected_moderate_manifest.json |
| standard_certificate | available | results/c92ebd8/notes/gru_standard_certificates_validation_selected_moderate_manifest.json |
| worst_case_epsilon_audit | missing | source not provided |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| closed_loop_moderate | not_claimed | available | available | available | available | 1 |
| closed_loop_cmd_lateral_moderate | not_claimed | available | available | available | available | 1 |
| closed_loop_cmd_lateral_small | not_claimed | available | available | available | available | 1 |
| closed_loop_cmd_lateral_stress | not_claimed | available | available | available | available | 1 |
| closed_loop_small | not_claimed | available | available | available | available | 1 |
| closed_loop_stress | not_claimed | available | available | available | available | 1 |
| open_loop_moderate | not_claimed | available | available | available | available | 1 |
| open_loop_small | not_claimed | available | available | available | available | 1 |
| open_loop_stress | not_claimed | available | available | available | available | 1 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
