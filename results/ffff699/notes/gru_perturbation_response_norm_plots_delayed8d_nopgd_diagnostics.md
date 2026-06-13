# GRU Perturbation-Response Norm Plots

This sidecar materializes Plotly response-curve HTML from existing calibrated perturbation-response bulk arrays. It does not rerun GRU diagnostics.

- Source manifest: `results/ffff699/notes/gru_perturbation_response_delayed8d_nopgd_diagnostics_manifest.json`
- Selector: `overnight_robust_proprio_validation_selected_corrected`
- Spec: `results/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/spec.json`
- Manifest: `results/ffff699/notes/gru_perturbation_response_norm_plots_delayed8d_nopgd_diagnostics_manifest.json`
- HTML inventory: 24 files under `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets`
- Aggregation: target-relative/sign-equalized xy responses are converted to Euclidean norms before pooling. Mean-norm panels show mean +/- SEM over pooled replicate x eval-seed samples; max-norm panels are unbanded pooled extreme-response curves.
- ExtLQG: deterministic dotted traces are reconstructed for command-input, initial-state, process-epsilon, sensory-feedback, and delayed-observation rows.
- ExtLQG trace status counts: disabled: 64.

## Interpretation Caveats

- Graph-adapter rows are paired against base rows evaluated on the same adapter-augmented graph with a zero payload, so pre-window differences reflect the declared perturbation path rather than a graph-topology change.
- Initial-state extLQG traces use a nominal estimator/controller initial state while perturbing the plant initial state, matching the delayed visibility contract used by the GRU rows.

## Inventory

- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__command_input_pulse__early.html` - class_a / delta_position / command_input_pulse / early
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__command_input_pulse__mid.html` - class_a / delta_position / command_input_pulse / mid
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__command_input_pulse__late.html` - class_a / delta_position / command_input_pulse / late
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__initial_position_offset__initial_condition.html` - class_a / delta_position / initial_position_offset / initial_condition
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__initial_velocity_offset__initial_condition.html` - class_a / delta_position / initial_velocity_offset / initial_condition
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__target_aligned_lateral_command_load_pulse__early.html` - class_a / delta_position / target_aligned_lateral_command_load_pulse / early
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__target_aligned_lateral_command_load_pulse__mid.html` - class_a / delta_position / target_aligned_lateral_command_load_pulse / mid
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_position__target_aligned_lateral_command_load_pulse__late.html` - class_a / delta_position / target_aligned_lateral_command_load_pulse / late
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_position__command_input_pulse__default.html` - class_b / delta_position / command_input_pulse / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_position__initial_position_offset__default.html` - class_b / delta_position / initial_position_offset / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_position__initial_velocity_offset__default.html` - class_b / delta_position / initial_velocity_offset / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_position__target_aligned_lateral_command_load_pulse__default.html` - class_b / delta_position / target_aligned_lateral_command_load_pulse / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__command_input_pulse__early.html` - class_a / delta_action / command_input_pulse / early
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__command_input_pulse__mid.html` - class_a / delta_action / command_input_pulse / mid
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__command_input_pulse__late.html` - class_a / delta_action / command_input_pulse / late
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__initial_position_offset__initial_condition.html` - class_a / delta_action / initial_position_offset / initial_condition
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__initial_velocity_offset__initial_condition.html` - class_a / delta_action / initial_velocity_offset / initial_condition
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__target_aligned_lateral_command_load_pulse__early.html` - class_a / delta_action / target_aligned_lateral_command_load_pulse / early
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__target_aligned_lateral_command_load_pulse__mid.html` - class_a / delta_action / target_aligned_lateral_command_load_pulse / mid
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_a__delta_action__target_aligned_lateral_command_load_pulse__late.html` - class_a / delta_action / target_aligned_lateral_command_load_pulse / late
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_action__command_input_pulse__default.html` - class_b / delta_action / command_input_pulse / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_action__initial_position_offset__default.html` - class_b / delta_action / initial_position_offset / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_action__initial_velocity_offset__default.html` - class_b / delta_action / initial_velocity_offset / default
- `_artifacts/ffff699/figures/perturbation_response_norms_delayed8d_nopgd_diagnostics/_assets/class_b__delta_action__target_aligned_lateral_command_load_pulse__default.html` - class_b / delta_action / target_aligned_lateral_command_load_pulse / default
