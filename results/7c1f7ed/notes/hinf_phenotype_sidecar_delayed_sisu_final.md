# H-infinity Phenotype Sidecar

Interpretive robustness phenotype report. This is not a standard certificate and is not a checkpoint-selection input.

Regeneration spec: `results/7c1f7ed/notes/hinf_phenotype_sidecar_delayed_sisu_final_regeneration_spec.json`

## Component Status

| Component | Status | Source |
|---|---:|---|
| broad_epsilon_attribution | missing | source not provided |
| evaluation_diagnostics | available | results/7c1f7ed/notes/gru_evaluation_diagnostics_delayed_sisu_final.json |
| exact_audit | missing | source not provided |
| feedback_ablation | missing | source not provided |
| induced_gain | missing | source not provided |
| map_error_decomposition | missing | source not provided |
| objective_comparator | missing | source not provided |
| perturbation_response | available | results/7c1f7ed/notes/delayed_sisu_perturbation_class_comparison.json |
| standard_certificate | available | results/7c1f7ed/notes/gru_standard_certificates_delayed_sisu_final_manifest.json |
| worst_case_epsilon_audit | missing | source not provided |

## Rows

| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | Local feedback law | H-inf markers | Warnings |
|---|---|---|---|---|---|---:|
| delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64 | not_claimed | available | available | missing | available | 3 |
| delayed_sisu_spectrum__raw_strong_gamma_1p05_radius_lr1e-2_clip5_b64 | not_claimed | available | available | missing | available | 3 |

## Caveats

- Formal H-infinity claims remain separate from phenotype evidence.
- Missing components are explicit; omitted evidence should not be inferred as pass.
- Paired baseline-vs-robust comparisons are reported only when matching pairs are present.
- Delayed contract caveat: Current standard certificate reports 6D delayed feedback/force-filter GraphSpec versus 8D output-feedback analytical reference response-map mismatch, so this sidecar is interpretive phenotype evidence only. Formal H-infinity equivalence is not_claimed.
