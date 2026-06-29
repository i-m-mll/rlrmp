<!-- AUTO-GENERATED: soft_pgd_stabilization_perturbation_responses -->
# Soft-PGD Stabilization Perturbation Responses

- Scope: stabilization-task endpoint perturbation response figures only.
- Rows: `soft_pgd_ofb1p05`, `soft_pgd_ofb1p4`, and `soft_pgd_ofb1p8`.
- Figure family: one figure per perturbation family (`command_input_pulse`, `feedback_position`, `feedback_velocity`, `feedback_force_filter`, `process_epsilon_force_state_xy`).
- Per-figure layout: response-state rows (`command`, `position`, `velocity`) by soft-PGD row columns.
- Perturbation timing: each subplot uses a shaded onset-to-offset band; duration comes from adapter provenance with summary timing fallback.
- Analytical comparators: 6D extLQG is rendered for all five families. 6D output-feedback H-infinity is rendered for `command_input_pulse` and `process_epsilon_force_state_xy`; sensory-feedback families are annotated as unsupported rather than faked.
- Tracked spec: `results/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/spec.json`.
- Bulk figures: `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses`.
- Bulk detail: `_artifacts/d55c5f0/stabilization_diagnostics/soft_pgd_stabilization_perturbation_responses/per_probe_detail.json`.
- Figure count: `5` HTML and `5` PNG.

| Perturbation family | Layout | HTML | PNG | H-inf unsupported panels |
|---|---|---|---|---:|
| `command_input_pulse` | 3x3 | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/command_input_pulse.html` | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/command_input_pulse.png` | 0 |
| `feedback_position` | 3x3 | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/feedback_position.html` | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/feedback_position.png` | 9 |
| `feedback_velocity` | 3x3 | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/feedback_velocity.html` | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/feedback_velocity.png` | 9 |
| `feedback_force_filter` | 3x3 | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/feedback_force_filter.html` | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/feedback_force_filter.png` | 9 |
| `process_epsilon_force_state_xy` | 3x3 | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/process_epsilon_force_state_xy.html` | `_artifacts/d55c5f0/figures/soft_pgd_stabilization_perturbation_responses/process_epsilon_force_state_xy.png` | 0 |

Total unsupported H-infinity sensory-feedback subplot entries: `27`.
<!-- /AUTO-GENERATED -->
