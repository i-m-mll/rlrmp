<!-- AUTO-GENERATED: pgd_1p05_stabilization_perturbation_responses -->
# PGD 1.05 Stabilization Task Perturbation Responses

- Scope: stabilization-task endpoint perturbation response figures.
- Figure family: one figure per perturbation family (`command_input_pulse`, `feedback_position`, `feedback_velocity`, `feedback_force_filter`, `process_epsilon_force_state_xy`).
- Per-figure layout: 3 rows of response variables (`command`, `position`, `velocity`) by 3 physical-level columns (`small`, `moderate`, `stress`).
- Contract: later stabilization-task isolation figures should use the same perturbation-family figure set and response-variable-by-level layout unless a later tracked spec supersedes it.
- Row pairing: each subplot overlays the no-PGD open-loop calibrated row with its PGD 1.05 counterpart.
- Timing: single `steady_state_endpoint` pulse timing; no early/mid/late split.
- Perturbation marker: shaded vertical band from perturbation onset to offset; duration comes from adapter provenance with summary timing as fallback.
- Trace contract: signed direction-aligned response residual with a lower-emphasis orthogonal companion trace. Command, hand-position, and hand-velocity traces come directly from the diagnostic detail payload and are residuals relative to the unperturbed endpoint rollout.
- Figure spec: `results/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/spec.json`.
- Figure count: `5` HTML, `5` PNG.

| Perturbation family | Layout | Event marker | HTML | PNG |
|---|---|---|---|---:|
| `command_input_pulse` | 3x3 | duration_band (5 steps) | `_artifacts/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/command_input_pulse.html` | `written` |
| `feedback_position` | 3x3 | duration_band (5 steps) | `_artifacts/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/feedback_position.html` | `written` |
| `feedback_velocity` | 3x3 | duration_band (5 steps) | `_artifacts/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/feedback_velocity.html` | `written` |
| `feedback_force_filter` | 3x3 | duration_band (5 steps) | `_artifacts/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/feedback_force_filter.html` | `written` |
| `process_epsilon_force_state_xy` | 3x3 | duration_band (5 steps) | `_artifacts/c92ebd8/figures/pgd_1p05_stabilization_perturbation_responses/process_epsilon_force_state_xy.html` | `written` |
<!-- /AUTO-GENERATED -->
