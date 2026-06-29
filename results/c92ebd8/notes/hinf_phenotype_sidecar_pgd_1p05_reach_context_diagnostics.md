# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

Regeneration spec: `results/c92ebd8/notes/hinf_phenotype_sidecar_pgd_1p05_reach_context_diagnostics_regeneration_spec.json`

## Component Status

| Component | Status | Source |
|---|---:|---|
| broad_epsilon_attribution | missing | source not provided |
| evaluation_diagnostics | available | results/c92ebd8/notes/gru_evaluation_diagnostics_pgd_1p05_reach_context_diagnostics.json |
| exact_audit | missing | source not provided |
| feedback_ablation | available | results/c92ebd8/notes/gru_feedback_ablation_pgd_1p05_reach_context_diagnostics_manifest.json |
| induced_gain | missing | source not provided |
| map_error_decomposition | missing | source not provided |
| objective_comparator | missing | source not provided |
| perturbation_response | available | results/c92ebd8/notes/gru_perturbation_response_pgd_1p05_reach_context_diagnostics_manifest.json |
| standard_certificate | missing | source not provided |
| worst_case_epsilon_audit | missing | source not provided |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| moderate | not_claimed | available | available | missing | available | 3 |
| open_loop_moderate | not_claimed | available | available | missing | available | 3 |
| open_loop_small | not_claimed | available | available | missing | available | 3 |
| open_loop_stress | not_claimed | available | available | missing | available | 3 |
| small | not_claimed | available | available | missing | available | 3 |
| stress | not_claimed | available | available | missing | available | 3 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
