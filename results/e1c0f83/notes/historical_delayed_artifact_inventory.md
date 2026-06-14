# Historical Delayed Artifact Inventory

Source branch tip: `3b4eb9c` (`feature/ffff699-no-integrator-delayed`).
Base audited against current local `main` at migration time.

## Disposition Summary

The audited path set contains 87 tracked artifact files across `results/6c36536`,
`results/ffff699`, `results/3b2af27`, and `results/ba82f3d`.

| Artifact class | Count | Disposition | Rationale |
|---|---:|---|---|
| Active delayed run specs | 16 | Migrated in place under `results/<issue>/runs/<run>/run.json` | Specs contain full training recipe metadata and load through `rlrmp.runtime.run_specs.validate_nominal_gru_run_spec_file`. Absolute `/workspace/rlrmp/...` artifact paths were normalized to repo-relative `_artifacts/...`. |
| Active graph manifests | 16 | Migrated beside each run spec as `model.graph.manifest.json` | Historical C&S LSS runs declare `feedbax_graph.graph_export_status=unavailable`, so the manifest is the current active custody pointer; no full GraphSpec sidecar exists for these runs. |
| Generated Markdown narratives | 21 | Provenance-only, not restored as active reports | The notes summarize historical analysis outputs whose bulk artifacts and generator state were not part of this lane. Regeneration should use current analysis lanes before treating them as active results. |
| Analysis/regeneration JSON specs | 20 | Provenance-only | These are old ad hoc regeneration inputs; current Feedbax-native analysis/regeneration manifests should replace them before active use. |
| Analysis output manifests | 6 | Provenance-only | They describe generated analysis outputs, not run custody. Source bulk inputs were not sufficient here to validate them as active manifests. |
| Figure specs | 2 | Provenance-only | The delayed diagnostic figure spec and related no-delay comparison spec depend on generated outputs; keep the source commit as provenance until regenerated with current `feedbax.plot.save_figure`. |
| Related full GraphSpec JSON sidecars | 2 | Rejected as stale active sidecars | Current `main` already records that these `3b2af27` sidecars serialized the legacy FirstOrderFilter -> PointMass compatibility path and did not represent the CS-LSS saved model. They were not restored as active GraphSpec custody. |
| Related modified non-delayed specs/manifests | 4 | Rejected in favor of current `main` custody | Current `main` already carries the active `3b2af27` run specs/manifests with `graph_export_status=unavailable` and the stale-sidecar rationale. This lane does not overwrite that correction with older branch metadata. |

## Migrated Active Run Specs

| Experiment | Run | Run spec | Graph manifest |
|---|---|---|---|
| `6c36536` | `delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego0_lr3e-3_clip5_b64_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego0_lr3e-3_clip5_b64_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego0_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1_lr3e-3_clip5_b64_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1_lr3e-3_clip5_b64_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e3_lr3e-3_clip5_b8_h16_smoke500_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e3_lr3e-3_clip5_b8_h16_smoke500_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e3_lr3e-3_clip5_b8_h16_smoke500_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b8_h16_smoke500_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b8_h16_smoke500_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b8_h16_smoke500_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b64_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b64_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b8_h16_smoke500_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b8_h16_smoke500_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b8_h16_smoke500_seed42/model.graph.manifest.json` |
| `6c36536` | `delayed_8d_no_pgd_catch0p5_prego1e5_normloss_smoke500_lr3e-3_clip5_b8_h16_seed42` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_normloss_smoke500_lr3e-3_clip5_b8_h16_seed42/run.json` | `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e5_normloss_smoke500_lr3e-3_clip5_b8_h16_seed42/model.graph.manifest.json` |
| `6c36536` | `normloss_3e3_s42` | `results/6c36536/runs/normloss_3e3_s42/run.json` | `results/6c36536/runs/normloss_3e3_s42/model.graph.manifest.json` |
| `ffff699` | `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42` | `results/ffff699/runs/delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42/run.json` | `results/ffff699/runs/delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `ffff699` | `delayed_8d_pgd_moderate_lr3e-3_clip5_b64_seed42` | `results/ffff699/runs/delayed_8d_pgd_moderate_lr3e-3_clip5_b64_seed42/run.json` | `results/ffff699/runs/delayed_8d_pgd_moderate_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `ffff699` | `delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed42` | `results/ffff699/runs/delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed42/run.json` | `results/ffff699/runs/delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |
| `ffff699` | `delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed43` | `results/ffff699/runs/delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed43/run.json` | `results/ffff699/runs/delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed43/model.graph.manifest.json` |
| `ffff699` | `delayed_no_integrator_pgd_moderate_lr3e-3_clip5_b64_seed42` | `results/ffff699/runs/delayed_no_integrator_pgd_moderate_lr3e-3_clip5_b64_seed42/run.json` | `results/ffff699/runs/delayed_no_integrator_pgd_moderate_lr3e-3_clip5_b64_seed42/model.graph.manifest.json` |

## Provenance-Only Historical Files

These files were audited from `3b4eb9c` but intentionally not restored as active contracts.

### Generated Markdown narratives

- `results/6c36536/notes/delayed_nominal_peak_recovery_plan_20260609.md`
- `results/6c36536/notes/delayed_peak_decay_diagnostics.md`
- `results/6c36536/notes/delayed_peak_decay_diagnostics_delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42.md`
- `results/6c36536/notes/delayed_peak_decay_diagnostics_delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42.md`
- `results/6c36536/notes/delayed_peak_decay_diagnostics_delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42.md`
- `results/6c36536/notes/delayed_peak_decay_diagnostics_normloss_3e3_s42.md`
- `results/6c36536/notes/no_pgd_catch_prego_runpod_plan.md`
- `results/6c36536/notes/runpod_delayed_8d_no_pgd_catch_prego_20260609.md`
- `results/6c36536/notes/runpod_delayed_normloss_20260609.md`
- `results/ffff699/notes/gru_feedback_ablation_delayed8d_nopgd_diagnostics.md`
- `results/ffff699/notes/gru_feedback_ablation_delayed8d_nopgd_validation_selected.md`
- `results/ffff699/notes/gru_map_error_decomposition_delayed8d_nopgd_diagnostics.md`
- `results/ffff699/notes/gru_map_error_decomposition_delayed8d_nopgd_validation_selected.md`
- `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_diagnostics.md`
- `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_validation_selected.md`
- `results/ffff699/notes/gru_perturbation_response_norm_plots_delayed8d_nopgd_diagnostics.md`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_core_smoke.md`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_diagnostics.md`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_validation_selected.md`
- `results/ffff699/notes/objective_comparator_delayed8d_nopgd_validation_selected.md`
- `results/ba82f3d/notes/no_delay_direction_aligned_velocity_lr3e-3_validation_selected.md`

### Analysis/regeneration JSON specs

- `results/ffff699/notes/gru_evaluation_diagnostics_delayed8d_nopgd_core_smoke_regeneration_spec.json`
- `results/ffff699/notes/gru_evaluation_diagnostics_delayed8d_nopgd_diagnostics_regeneration_spec.json`
- `results/ffff699/notes/gru_evaluation_diagnostics_delayed8d_nopgd_validation_selected_regeneration_spec.json`
- `results/ffff699/notes/gru_feedback_ablation_delayed8d_nopgd_diagnostics_regeneration_spec.json`
- `results/ffff699/notes/gru_feedback_ablation_delayed8d_nopgd_validation_selected_regeneration_spec.json`
- `results/ffff699/notes/gru_map_error_decomposition_delayed8d_nopgd_diagnostics_regeneration_spec.json`
- `results/ffff699/notes/gru_map_error_decomposition_delayed8d_nopgd_validation_selected_regeneration_spec.json`
- `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_diagnostics_manifest_regeneration_spec.json`
- `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_validation_selected_manifest_regeneration_spec.json`
- `results/ffff699/notes/gru_pilot_figures_delayed8d_nopgd_core_smoke_regeneration_spec.json`
- `results/ffff699/notes/gru_pilot_figures_delayed8d_nopgd_diagnostics_regeneration_spec.json`
- `results/ffff699/notes/gru_pilot_figures_delayed8d_nopgd_validation_selected_regeneration_spec.json`
- `results/ffff699/notes/gru_postrun_materialization_delayed8d_nopgd_core_smoke_regeneration_spec.json`
- `results/ffff699/notes/gru_postrun_materialization_delayed8d_nopgd_diagnostics_regeneration_spec.json`
- `results/ffff699/notes/gru_postrun_materialization_delayed8d_nopgd_validation_selected_regeneration_spec.json`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_core_smoke_manifest_regeneration_spec.json`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_diagnostics_manifest_regeneration_spec.json`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_validation_selected_manifest_regeneration_spec.json`
- `results/ffff699/notes/objective_comparator_delayed8d_nopgd_diagnostics_regeneration_spec.json`
- `results/ffff699/notes/objective_comparator_delayed8d_nopgd_validation_selected_regeneration_spec.json`

### Analysis output manifests

- `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_diagnostics_manifest.json`
- `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_validation_selected_manifest.json`
- `results/ffff699/notes/gru_perturbation_response_norm_plots_delayed8d_nopgd_diagnostics_manifest.json`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_core_smoke_manifest.json`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_diagnostics_manifest.json`
- `results/ffff699/notes/gru_standard_certificates_delayed8d_nopgd_validation_selected_manifest.json`

### Figure specs

- `results/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/spec.json`
- `results/ba82f3d/figures/no_delay_direction_aligned_velocity_lr3e-3_validation_selected/spec.json`

### Related non-delayed GraphSpec sidecars and modified specs

- `results/3b2af27/runs/lss_12k__hidden_penalty/model.graph.json`
- `results/3b2af27/runs/lss_12k__no_hidden_penalty/model.graph.json`
- `results/3b2af27/runs/lss_12k__hidden_penalty/run.json`
- `results/3b2af27/runs/lss_12k__hidden_penalty/model.graph.manifest.json`
- `results/3b2af27/runs/lss_12k__no_hidden_penalty/run.json`
- `results/3b2af27/runs/lss_12k__no_hidden_penalty/model.graph.manifest.json`

## Validation Contract

Active run specs must parse as JSON and pass `rlrmp.runtime.run_specs.validate_nominal_gru_run_spec_file`. Because these historical C&S LSS specs explicitly declare `feedbax_graph.graph_export_status=unavailable`, validation requires the adjacent `model.graph.manifest.json` pointer but does not require a full `model.graph.json` sidecar.

Residual provenance-only files require current-generator regeneration before becoming active results. The missing inputs are the historical bulk analysis outputs/checkpoints and current Feedbax-native analysis manifests for those pipelines. The rejected `3b2af27` full graph sidecars would also need regenerated CS-LSS GraphSpecs rather than legacy point-mass compatibility exports.
