<!-- AUTO-GENERATED: soft_pgd_feedback_robustness_diagnostics -->
# Soft-PGD Feedback Robustness Diagnostics

This note compares only the d55 first-batch soft-constraint PGD rows. It intentionally does not read or overwrite c92 OFB-budget outputs.

| Row | Gamma factor | Peak velocity | fb delta u | Ablation idx | Sensory AUC dx | Non-sensory AUC dx | Peak dx/OL | Stab feedback AUC | Stab mechanical AUC | Stab command AUC | Stab process-force AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `soft_pgd_ofb1p05` | 1.05 | 0.72887 | 0.542 | 1.007 | 0.671 | 1.978 | 0.456 | 7.432 | 0.7444 | 1.007 | 0.4821 |
| `soft_pgd_ofb1p4` | 1.40 | 0.72914 | 0.541 | 1.267 | 0.668 | 1.983 | 0.457 | 7.377 | 0.7875 | 1.061 | 0.5140 |
| `soft_pgd_ofb1p8` | 1.80 | 0.73029 | 0.544 | 1.247 | 0.670 | 1.976 | 0.457 | 7.501 | 0.8278 | 1.129 | 0.5263 |

## Soft-gamma deltas

- `soft_pgd_ofb1p4_minus_soft_pgd_ofb1p05`:
  - `peak_velocity_m_s`: 0.000266091
  - `fb_delta_u`: -0.00113971
  - `ablation_idx`: 0.259634
  - `sensory_auc_dx_mm_s`: -0.00323292
  - `non_sensory_auc_dx_mm_s`: 0.00513538
  - `peak_dx_over_open_loop`: 0.00104407
  - `stabilization_feedback_auc_mm_s`: -0.0552754
  - `stabilization_mechanical_auc_mm_s`: 0.0430996
  - `stabilization_command_auc_mm_s`: 0.0542958
  - `stabilization_process_force_auc_mm_s`: 0.0319033
- `soft_pgd_ofb1p8_minus_soft_pgd_ofb1p4`:
  - `peak_velocity_m_s`: 0.0011554
  - `fb_delta_u`: 0.002965
  - `ablation_idx`: -0.0203193
  - `sensory_auc_dx_mm_s`: 0.0020249
  - `non_sensory_auc_dx_mm_s`: -0.00724051
  - `peak_dx_over_open_loop`: -0.000601167
  - `stabilization_feedback_auc_mm_s`: 0.123696
  - `stabilization_mechanical_auc_mm_s`: 0.0402884
  - `stabilization_command_auc_mm_s`: 0.0682741
  - `stabilization_process_force_auc_mm_s`: 0.0123028

## Outputs

- `checkpoint_manifest`: `results/d55c5f0/notes/soft_pgd_feedback_robustness_diagnostics_validation_selected_checkpoints.json`
- `evaluation`: `results/d55c5f0/notes/gru_evaluation_diagnostics_soft_pgd_feedback_robustness_diagnostics.json`
- `evaluation_regeneration_spec`: `results/d55c5f0/notes/gru_evaluation_diagnostics_soft_pgd_feedback_robustness_diagnostics_regeneration_spec.json`
- `perturbation`: `results/d55c5f0/notes/gru_perturbation_response_soft_pgd_feedback_robustness_diagnostics_manifest.json`
- `perturbation_note`: `results/d55c5f0/notes/gru_perturbation_response_soft_pgd_feedback_robustness_diagnostics.md`
- `perturbation_regeneration_spec`: `results/d55c5f0/notes/gru_perturbation_response_soft_pgd_feedback_robustness_diagnostics_manifest_regeneration_spec.json`
- `feedback`: `results/d55c5f0/notes/gru_feedback_ablation_soft_pgd_feedback_robustness_diagnostics.json`
- `feedback_note`: `results/d55c5f0/notes/gru_feedback_ablation_soft_pgd_feedback_robustness_diagnostics.md`
- `feedback_regeneration_spec`: `results/d55c5f0/notes/gru_feedback_ablation_soft_pgd_feedback_robustness_diagnostics_regeneration_spec.json`
- `stabilization_detail`: `_artifacts/d55c5f0/stabilization_diagnostics/soft_pgd_feedback_robustness_diagnostics/per_probe_detail.json`
- `summary_json`: `results/d55c5f0/notes/soft_pgd_feedback_robustness_diagnostics.json`
- `summary_markdown`: `results/d55c5f0/notes/soft_pgd_feedback_robustness_diagnostics.md`
- `summary_csv`: `results/d55c5f0/notes/soft_pgd_feedback_robustness_diagnostics.csv`
- `perturbation_detail_manifest`: `{'path': '_artifacts/d55c5f0/perturbation_response/soft_pgd_feedback_robustness_diagnostics/gru_perturbation_response_soft_pgd_feedback_robustness_diagnostics_manifest_detail.json', 'format': 'json', 'contains': 'full per-run perturbation rows and row-level metric summaries'}`

## Aggregation contract

- `peak_velocity_m_s`: mean_profile_peak_forward_velocity_m_s from evaluation diagnostics.
- `fb_delta_u`: feedback_ablation.interpretation.max_feedback_delta_action_norm_mean.
- `ablation_idx`: feedback_pass_audit.components.feedback_ablation_dependence.ablation_dependence_index.
- `sensory_auc_dx_mm_s`: sensory_feedback/sensory_feedback_offset class mean delta_position_response_m.auc, converted to mm*s.
- `non_sensory_auc_dx_mm_s`: unweighted mean of available non-sensory/non-target class means for delta_position_response_m.auc, converted to mm*s.
- `peak_dx_over_open_loop`: mean available closed_loop_peak_dx_over_open_loop_peak_dx over evaluated non-sensory/non-target reach perturbation rows.
- `stabilization_*_auc_mm_s`: stabilization-task endpoint mean signed-direction-aligned absolute hand-position displacement over the post-onset window.
<!-- /AUTO-GENERATED -->
