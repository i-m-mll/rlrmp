# PGD 1.05 reach-context diagnostics

Rows compare calibrated no-PGD open-loop comparators against PGD 1.05 rows at the same physical perturbation level. Values are computed from the materialized sidecars listed below, not copied from the issue comment.

| Row | Training condition | Physical level | Peak velocity | fb Δu | Ablation idx | Sensory AUC Δx | Non-sensory AUC Δx | Peak dx/OL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | no-PGD open-loop calibrated | `small` | 0.73068 | 0.508 | 2.710 | 0.675 | 2.197 | 0.485 |
| `open_loop_moderate` | no-PGD open-loop calibrated | `moderate` | 0.73109 | 0.585 | 3.289 | 0.683 | 1.972 | 0.456 |
| `open_loop_stress` | no-PGD open-loop calibrated | `stress` | 0.73242 | 1.972 | 4.891 | 0.561 | 1.623 | 0.415 |
| `small` | PGD 1.05 open-loop calibrated | `small` | 0.75773 | 0.569 | 3.052 | 0.598 | 2.000 | 0.478 |
| `moderate` | PGD 1.05 open-loop calibrated | `moderate` | 0.76293 | 0.630 | 3.393 | 0.608 | 1.839 | 0.461 |
| `stress` | PGD 1.05 open-loop calibrated | `stress` | 0.76233 | 1.653 | 5.078 | 0.504 | 1.420 | 0.419 |

## Interpretation

PGD 1.05 does not read as a simple global control-gain increase: nominal peak velocity stays close to the no-PGD rows, while feedback dependence and perturbation-response metrics move unevenly by physical level.

PGD 1.05 does not consistently increase robustness beyond the calibrated no-PGD open-loop rows on these reach-context diagnostics.

No formal H-infinity claim is made; the phenotype sidecar is diagnostic-only unless a standard certificate passes.

## Source outputs

- `checkpoint_manifest`: `results/c92ebd8/notes/pgd_1p05_reach_context_diagnostics_validation_selected_checkpoints.json`
- `evaluation`: `results/c92ebd8/notes/gru_evaluation_diagnostics_pgd_1p05_reach_context_diagnostics.json`
- `evaluation_regeneration_spec`: `results/c92ebd8/notes/gru_evaluation_diagnostics_pgd_1p05_reach_context_diagnostics_regeneration_spec.json`
- `perturbation`: `results/c92ebd8/notes/gru_perturbation_response_pgd_1p05_reach_context_diagnostics_manifest.json`
- `perturbation_note`: `results/c92ebd8/notes/gru_perturbation_response_pgd_1p05_reach_context_diagnostics.md`
- `perturbation_regeneration_spec`: `results/c92ebd8/notes/gru_perturbation_response_pgd_1p05_reach_context_diagnostics_manifest_regeneration_spec.json`
- `feedback`: `results/c92ebd8/notes/gru_feedback_ablation_pgd_1p05_reach_context_diagnostics.json`
- `feedback_note`: `results/c92ebd8/notes/gru_feedback_ablation_pgd_1p05_reach_context_diagnostics.md`
- `feedback_regeneration_spec`: `results/c92ebd8/notes/gru_feedback_ablation_pgd_1p05_reach_context_diagnostics_regeneration_spec.json`
- `phenotype`: `results/c92ebd8/notes/hinf_phenotype_sidecar_pgd_1p05_reach_context_diagnostics.json`
- `phenotype_note`: `results/c92ebd8/notes/hinf_phenotype_sidecar_pgd_1p05_reach_context_diagnostics.md`
- `phenotype_regeneration_spec`: `results/c92ebd8/notes/hinf_phenotype_sidecar_pgd_1p05_reach_context_diagnostics_regeneration_spec.json`
- `summary_json`: `results/c92ebd8/notes/pgd_1p05_reach_context_diagnostics.json`
- `summary_markdown`: `results/c92ebd8/notes/pgd_1p05_reach_context_diagnostics.md`
- `summary_csv`: `results/c92ebd8/notes/pgd_1p05_reach_context_diagnostics.csv`
- `perturbation_detail_manifest`: `{'path': '_artifacts/c92ebd8/perturbation_response/pgd_1p05_reach_context_diagnostics/perturbation_response/gru_perturbation_response_pgd_1p05_reach_context_diagnostics_manifest_detail.json', 'format': 'json', 'contains': 'full per-run perturbation rows and row-level metric summaries'}`

## Aggregation contract

- `peak_velocity_m_s`: mean_profile_peak_forward_velocity_m_s from evaluation diagnostics.
- `fb_delta_u`: feedback_ablation.interpretation.max_feedback_delta_action_norm_mean.
- `ablation_idx`: feedback_pass_audit.components.feedback_ablation_dependence.ablation_dependence_index.
- `sensory_auc_dx_mm_s`: sensory_feedback/sensory_feedback_offset class mean delta_position_response_m.auc, converted to mm*s.
- `non_sensory_auc_dx_mm_s`: unweighted mean of available non-sensory/non-target class means for delta_position_response_m.auc, converted to mm*s.
- `peak_dx_over_open_loop`: mean available attenuation_metrics.closed_loop_peak_dx_over_open_loop_peak_dx over evaluated non-sensory/non-target perturbation rows.
