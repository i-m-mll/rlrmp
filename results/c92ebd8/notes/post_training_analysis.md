<!-- AUTO-GENERATED: c92_post_training_analysis -->
# c92 Post-Training Analysis

- Scope: nine no-PGD calibrated perturbation rows after 12000/12000 batches.
- Physical level for perturbation profiles: `moderate`.
- Analytical comparators: 6D no-integrator extLQG and output-feedback H-infinity.
- Robustness phenotype outputs are interpretive diagnostics, not formal H-infinity certificates.

## Moderate Perturbation Profiles

- Status: `materialized`.
- Figure spec: `results/c92ebd8/figures/moderate_perturbation_profiles/spec.json`.
- HTML render directory: `_artifacts/c92ebd8/figures/moderate_perturbation_profiles`.
- Figure count: `288`.

## Nominal Velocity Profiles

- Status: `materialized`.
- Figure spec: `results/c92ebd8/figures/nominal_velocity_profiles/spec.json`.
- Navigable HTML link: `results/c92ebd8/figures/nominal_velocity_profiles/figure.html`.
- H-infinity comparator: 6D no-integrator output-feedback path (`state_dim=None`, `physical_dim=None`, `disturbance_integrators_exposed=none`).

## Diagnostic Inputs

- Evaluation diagnostics: `results/c92ebd8/notes/gru_evaluation_diagnostics_validation_selected_moderate.json`.
- Feedback-quality diagnostics: `results/c92ebd8/notes/gru_feedback_ablation_validation_selected_moderate.json`.
- Perturbation response manifest: `results/c92ebd8/notes/gru_perturbation_response_validation_selected_moderate_manifest.json`.
- Robustness phenotype sidecar: `results/c92ebd8/notes/hinf_phenotype_sidecar_validation_selected_moderate.json`.
- Post-run materialization manifest: `results/c92ebd8/notes/gru_postrun_materialization_validation_selected_moderate.json`.

<!-- /AUTO-GENERATED -->
