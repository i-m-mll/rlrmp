<!-- AUTO-GENERATED: output_feedback_budget_diagnostics -->
# Output-feedback-budget PGD diagnostics

This note compares exactly the c92 no-PGD open-loop moderate baseline against the two new open-loop moderate rows trained with output-feedback rollout PGD budgets. The older raw/full-state PGD row is intentionally not part of the main table.

| Row | Training condition | Active L2 radius | Peak velocity | fb delta u | Ablation idx | Sensory AUC dx | Non-sensory AUC dx | Peak dx/OL | Stab feedback AUC | Stab mechanical AUC | Stab command AUC | Stab process-force AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `open_loop_moderate` | no-PGD open-loop calibrated moderate | not applicable | 0.73109 | 0.048 | 0.015 | 0.688 | 1.972 | 0.456 | 7.250 | 0.7400 | 1.012 | 0.4681 |
| `moderate_pgd_ofb1p05` | PGD output-feedback-budget gamma 1.05 | 0.00175133 | 0.75790 | 0.062 | 0.018 | 0.640 | 1.606 | 0.446 | 7.307 | 0.7483 | 0.916 | 0.5809 |
| `moderate_pgd_ofb1p4` | PGD output-feedback-budget gamma 1.4 | 0.00454501 | 0.76534 | 0.074 | 0.020 | 0.540 | 1.527 | 0.446 | 7.918 | 0.8334 | 1.067 | 0.5995 |

## Interpretation

Both output-feedback-budget PGD rows improve the reach-context displacement diagnostics versus the no-PGD open-loop moderate baseline, but both worsen the stabilization endpoint feedback/mechanical AUCs. That is not a clean across-task robustness improvement.

The larger OFB 1.4 budget is stronger on reach-context attenuation but worse on stabilization endpoint feedback/mechanical AUCs than the OFB 1.05 budget.

No formal H-infinity evidence is claimed. The rows carry OFB rollout budget provenance, but these diagnostics are empirical phenotype checks.

Budget provenance:
- `moderate_pgd_ofb1p05`: `ofb_6d_no_integrator_gamma_1p05_rollout_radius`, active_l2_radius_15cm=0.001751332497496124.
- `moderate_pgd_ofb1p4`: `ofb_6d_no_integrator_gamma_1p4_rollout_radius`, active_l2_radius_15cm=0.004545011406169036.

## Outputs

- `checkpoint_manifest`: `results/c92ebd8/notes/output_feedback_budget_diagnostics_validation_selected_checkpoints.json`
- `evaluation`: `results/c92ebd8/notes/gru_evaluation_diagnostics_output_feedback_budget_diagnostics.json`
- `evaluation_regeneration_spec`: `results/c92ebd8/notes/gru_evaluation_diagnostics_output_feedback_budget_diagnostics_regeneration_spec.json`
- `perturbation`: `results/c92ebd8/notes/gru_perturbation_response_output_feedback_budget_diagnostics_manifest.json`
- `perturbation_note`: `results/c92ebd8/notes/gru_perturbation_response_output_feedback_budget_diagnostics.md`
- `perturbation_regeneration_spec`: `results/c92ebd8/notes/gru_perturbation_response_output_feedback_budget_diagnostics_manifest_regeneration_spec.json`
- `feedback`: `results/c92ebd8/notes/gru_feedback_ablation_output_feedback_budget_diagnostics.json`
- `feedback_note`: `results/c92ebd8/notes/gru_feedback_ablation_output_feedback_budget_diagnostics.md`
- `feedback_regeneration_spec`: `results/c92ebd8/notes/gru_feedback_ablation_output_feedback_budget_diagnostics_regeneration_spec.json`
- `stabilization_detail`: `_artifacts/c92ebd8/stabilization_diagnostics/output_feedback_budget_diagnostics/per_probe_detail.json`
- `summary_json`: `results/c92ebd8/notes/output_feedback_budget_diagnostics.json`
- `summary_markdown`: `results/c92ebd8/notes/output_feedback_budget_diagnostics.md`
- `summary_csv`: `results/c92ebd8/notes/output_feedback_budget_diagnostics.csv`
- `perturbation_detail_manifest`: `{'contains': 'full per-run perturbation rows and row-level metric summaries', 'format': 'json', 'path': '_artifacts/c92ebd8/perturbation_response/output_feedback_budget_diagnostics/gru_perturbation_response_output_feedback_budget_diagnostics_manifest_detail.json'}`

## Aggregation contract

- `peak_velocity_m_s`: mean_profile_peak_forward_velocity_m_s from evaluation diagnostics.
- `fb_delta_u`: feedback_ablation.interpretation.max_feedback_delta_action_norm_mean.
- `ablation_idx`: feedback_pass_audit.components.feedback_ablation_dependence.ablation_dependence_index.
- `sensory_auc_dx_mm_s`: sensory_feedback/sensory_feedback_offset class mean delta_position_response_m.auc, converted to mm*s.
- `non_sensory_auc_dx_mm_s`: unweighted mean of available non-sensory/non-target class means for delta_position_response_m.auc, converted to mm*s.
- `peak_dx_over_open_loop`: mean available closed_loop_peak_dx_over_open_loop_peak_dx over evaluated non-sensory/non-target reach perturbation rows.
- `stabilization_*_auc_mm_s`: stabilization-task endpoint mean signed-direction-aligned absolute hand-position displacement over the post-onset window.
<!-- /AUTO-GENERATED -->
