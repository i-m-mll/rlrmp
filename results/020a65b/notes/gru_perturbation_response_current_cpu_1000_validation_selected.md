# GRU perturbation-response bank

Issue: `020a65b`. Source experiment: `020a65b`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 18 |
| `delayed_observation` | 24 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 24 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 24 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 24 |
| `target_aligned_lateral_command_load_pulse` | 6 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64_current_cpu_1000`

- Evaluated: 122
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.10615 | 0.00723752 | 0.00189282 | 0.0369455 | 0.00823147 | 0.413688 | 0.0636897 | 0.554758 | NA | 0.00193598 | 0.00113526 | 607.322 | 8.31792 | 8.02805 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.108274 | 0.00644655 | 0.00170257 | 0.0363516 | 0.0075587 | 0.452345 | 0.0649642 | 0.544578 | NA | 0.00216833 | 0.00101405 | 497.254 | 6.81043 | 6.57309 | none |
| `delayed_observation/delayed_observation_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.0186346 | 0.000826236 | 0.000175091 | 0.00496229 | 0.00092178 | 0.0875816 | 0.0111807 | 0.570691 | NA | 3.92643e-05 | 0.00027881 | 13.2513 | 0.252387 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0261405 | 0.01 | 0.00537253 | 0.0121286 | 0.00652545 | 0.080228 | 0.0156843 | 0.0147891 | NA | 0.00189235 | 0.00165992 | 506.182 | 13.575 | 11.394 | inflated_ratio; inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.118034 | 0.0157458 | 0.00646097 | 0.0499525 | 0.0180846 | 0.272507 | 0.0708206 | 0.564586 | NA | 0.00824855 | 0.000665337 | 1863.95 | 50.8962 | 40.2065 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00700642 | 0.000477699 | 0.000124947 | 0.00243829 | 0.000543185 | 0.0273668 | 0.00420385 | 0.555169 | NA | 7.98652e-06 | 4.8395e-06 | 2.64619 | 8.32011 | 8.03017 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0176973 | 0.00328327 | 0.00055443 | 0.0130973 | 0.00346331 | 0.0812262 | 0.0106184 | 0.59 | NA | 0.00054272 | 0.00335579 | 161.468 | 146.709 | 147.211 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.0871488 | 0.05 | 0.0189938 | 0.05963 | 0.0211135 | 0.2991 | 0.0522893 | 0.229247 | NA | 0.0353747 | 0.0261381 | 16805.2 | 1.43124 | 1.89627 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.0845426 | 0.0132962 | 0.00345893 | 0.0499101 | 0.0145177 | 0.257392 | 0.0507256 | 0.58524 | NA | 0.00640427 | 0.00688278 | 1738.86 | 1.81834 | 7.70259 | none |
| `sensory_feedback/sensory_feedback_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.0186346 | 0.000826236 | 0.000175091 | 0.00496229 | 0.00092178 | 0.0875816 | 0.0111807 | 0.570691 | NA | 3.92643e-05 | 0.00027881 | 13.2513 | 0.252387 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.114797 | 0.00957325 | 0.00320893 | 748.904 | 31.4576 | 27.8169 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.107417 | 0.00575816 | 0.000630059 | 726.54 | 4.72909 | 4.73766 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0962338 | 0.00638115 | 0.00183947 | 346.521 | 8.32941 | 7.42447 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.121381 | 0.00749105 | 0.00268668 | 445.556 | 18.7155 | 16.5495 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.106393 | 0.00576281 | 0.000630252 | 725.936 | 4.72516 | 4.73372 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.0970475 | 0.00608578 | 0.00179077 | 320.271 | 7.69843 | 6.86204 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.0254295 | 0.000993058 | 0.000303688 | 10.9739 | 0.122837 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.0117973 | 0.000447742 | 2.61388e-05 | 12.4078 | 0.893381 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.0186769 | 0.00103791 | 0.000195445 | 16.3723 | 0.301591 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0261405 | 0.01 | 0.00537253 | 506.182 | 13.575 | 11.394 | inflated_ratio; inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.118034 | 0.0157458 | 0.00646097 | 1863.95 | 50.8962 | 40.2065 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00757446 | 0.000632498 | 0.000211921 | 3.26825 | 31.5157 | 27.8683 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00709609 | 0.000379972 | 4.15805e-05 | 3.1636 | 4.72728 | 4.73585 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00634872 | 0.000420627 | 0.000121339 | 1.50672 | 8.31436 | 7.41105 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0294719 | 0.00509994 | 0.00101092 | 270.314 | 375.453 | 373.328 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.00384817 | 0.00112481 | 8.82646e-05 | 39.1518 | 22.5117 | 22.6535 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0197719 | 0.00362506 | 0.000564109 | 174.937 | 207.603 | 208.71 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.0969812 | 0.05 | 0.0248468 | 15329.9 | 3.08073 | 6.41725 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.068793 | 0.05 | 0.0114603 | 17939 | 1.00414 | 1.05709 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0956722 | 0.05 | 0.0206744 | 17146.8 | 1.3846 | 2.37235 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.102076 | 0.0156597 | 0.00538422 | 1860.06 | 5.61577 | 29.1968 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.0579855 | 0.0100943 | 0.00122772 | 1763.19 | 1.08165 | 3.67044 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0935665 | 0.0141346 | 0.00376486 | 1593.32 | 1.75563 | 11.9649 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.0254295 | 0.000993058 | 0.000303688 | 10.9739 | 0.122837 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.0117973 | 0.000447742 | 2.61388e-05 | 12.4078 | 0.893381 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.0186769 | 0.00103791 | 0.000195445 | 16.3723 | 0.301591 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
