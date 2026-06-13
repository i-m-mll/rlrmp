# GRU perturbation-response bank

Issue: `ffff699`. Source experiment: `ffff699`.

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

### `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42`

- Evaluated: 122
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.130344 | 0.00333011 | 0.000805854 | 0.0318005 | 0.00503327 | 1.00061 | 0.0782066 | 0.38282 | NA | 0.000850608 | 0.00199104 | 136.153 | 1.86476 | 1.79978 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.131144 | 0.00333639 | 0.000806107 | 0.0318878 | 0.00504458 | 1.00694 | 0.0786865 | 0.381641 | NA | 0.000872727 | 0.00202071 | 136.984 | 1.87614 | 1.81076 | none |
| `delayed_observation/delayed_observation_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.151862 | 0.00193813 | 0.000402709 | 0.0202893 | 0.00273071 | 0.929465 | 0.0911173 | 0.42868 | NA | 0.00042841 | 0.000527188 | 83.6519 | 1.59325 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0361401 | 0.00716833 | 0.00156743 | 0.0255632 | 0.00686487 | 0.101282 | 0.021684 | 0 | 0.520359 | 6.69124e-05 | 0.000265247 | 16.937 | 0.454224 | 0.381249 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.0331265 | 0.00587144 | 0.00147354 | 0.0167356 | 0.00560556 | 0.12259 | 0.0198759 | 0 | 0.559017 | 7.52964e-05 | 0.000284869 | 18.3703 | 0.501611 | 0.396259 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00860652 | 0.00021959 | 5.31267e-05 | 0.00209881 | 0.000331989 | 0.066123 | 0.00516391 | 0.382734 | NA | 4.77789e-06 | 9.17087e-06 | 0.592013 | 1.8614 | 1.79653 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0327371 | 0.00161712 | 0.000342692 | 0.00643434 | 0.00171927 | 0.0788304 | 0.0196423 | 0.59 | NA | 0.000631149 | 0.000838024 | 30.001 | 27.2589 | 27.3522 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.426585 | 0.05 | 0.0149816 | 0.139759 | 0.0352488 | 2.59525 | 0.255951 | 0.230068 | NA | 0.0211268 | 0.078275 | 14118 | 1.20238 | 1.59304 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.156461 | 0.00645004 | 0.00166972 | 0.0499101 | 0.00977216 | 1.11862 | 0.0938765 | 0.377995 | NA | 0.0023219 | 0.00613803 | 426.297 | 0.445784 | 1.88836 | none |
| `sensory_feedback/sensory_feedback_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.15183 | 0.00193753 | 0.000402561 | 0.0202846 | 0.00272951 | 0.929519 | 0.091098 | 0.42871 | NA | 0.000450917 | 0.000558796 | 83.6067 | 1.59239 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.140553 | 0.00342392 | 0.00105086 | 58.1926 | 2.44437 | 2.16148 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.118573 | 0.00329026 | 0.000463644 | 252.903 | 1.64616 | 1.64914 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.131907 | 0.00327615 | 0.000903057 | 97.3635 | 2.34035 | 2.08608 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.141717 | 0.00342967 | 0.00104939 | 58.3528 | 2.4511 | 2.16742 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.119428 | 0.00329835 | 0.000464447 | 254.529 | 1.65674 | 1.65975 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.132287 | 0.00328114 | 0.000904486 | 98.0692 | 2.35731 | 2.1012 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.138923 | 0.00179311 | 0.000520679 | 27.5381 | 0.30825 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.150962 | 0.00200839 | 0.000184815 | 166.062 | 11.9568 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.165701 | 0.0020129 | 0.000502632 | 57.3554 | 1.05653 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0361401 | 0.00716833 | 0.00156743 | 16.937 | 0.454224 | 0.381249 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.0331265 | 0.00587144 | 0.00147354 | 18.3703 | 0.501611 | 0.396259 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00928139 | 0.000225779 | 6.92635e-05 | 0.252675 | 2.43654 | 2.15455 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0078275 | 0.000216955 | 3.05837e-05 | 1.10027 | 1.64411 | 1.64709 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00871069 | 0.000216037 | 5.95327e-05 | 0.42309 | 2.33469 | 2.08104 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0462736 | 0.0020707 | 0.000572396 | 33.3048 | 46.2587 | 45.9968 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.015196 | 0.000946332 | 8.12021e-05 | 24.9691 | 14.3569 | 14.4473 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0367418 | 0.00183434 | 0.000374479 | 31.7293 | 37.6542 | 37.855 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.499337 | 0.05 | 0.0177854 | 7176.06 | 1.44212 | 3.00398 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.323227 | 0.05 | 0.0109253 | 21486.3 | 1.20269 | 1.26612 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.45719 | 0.05 | 0.0162342 | 13691.6 | 1.1056 | 1.89431 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.172409 | 0.00660316 | 0.00214782 | 198.832 | 0.600303 | 3.12102 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.136385 | 0.00638848 | 0.00100646 | 745.581 | 0.457387 | 1.55208 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.160588 | 0.00635849 | 0.00185487 | 334.477 | 0.368549 | 2.51172 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.138875 | 0.00179267 | 0.00052042 | 27.4936 | 0.307752 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.15103 | 0.00200834 | 0.000184806 | 166.084 | 11.9583 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.165585 | 0.00201159 | 0.000502455 | 57.243 | 1.05446 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
