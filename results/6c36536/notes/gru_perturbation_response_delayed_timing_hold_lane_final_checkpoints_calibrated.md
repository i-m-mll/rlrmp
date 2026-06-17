# GRU perturbation-response bank

Issue: `6c36536`. Source experiment: `6c36536`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 54 |
| `delayed_observation` | 108 |
| `initial_state` | 24 |
| `process_epsilon` | 144 |
| `sensory_feedback` | 108 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 36 |
| `delayed_observation_offset` | 108 |
| `initial_position_offset` | 12 |
| `initial_velocity_offset` | 12 |
| `process_epsilon_force_state_xy` | 36 |
| `process_epsilon_integrator_xy` | 36 |
| `process_epsilon_position_xy` | 36 |
| `process_epsilon_velocity_xy` | 36 |
| `sensory_feedback_offset` | 108 |
| `target_aligned_lateral_command_load_pulse` | 18 |
| `target_stream_jump` | 1 |

## Evaluation

### `baseline__delayed_repeat`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.427723 | 0.0203567 | 0.00839485 | 0.119463 | 0.0324146 | 2.82588 | 0.384951 | 0.643514 | 0.614183 | 0.0150109 | 0.0074387 | 8949.94 | 17.3012 | 17.1864 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.473682 | 0.0214134 | 0.00882595 | 0.127235 | 0.0340781 | 3.278 | 0.426313 | 0.661205 | 0.56 | 0.0166297 | 0.00916473 | 11361.6 | 21.9633 | 21.8175 | inflated_ratio; inflated_ratio |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.0960249 | 0.00444738 | 0.00170907 | 0.0223857 | 0.00614025 | 0.693671 | 0.0864224 | 0.717599 | 0.671233 | 0.00215807 | 0.0014387 | 687.294 | 0.00829473 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.145487 | 0.0200196 | 0.0149409 | 0.046505 | 0.0204777 | 1.28164 | 0.130939 | 0.193734 | 0.620597 | 0.00805527 | 0.00125626 | 2068.41 | 9.86156 | 8.27721 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.370837 | 0.0172171 | 0.00886384 | 0.0946864 | 0.0323089 | 2.86787 | 0.333754 | 0.518565 | 0.76625 | 0.0117802 | 0.00433556 | 6065.03 | 249.861 | 197.383 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.42837 | 0.0203473 | 0.00838942 | 0.119404 | 0.0324242 | 2.82201 | 0.385533 | 0.642372 | 0.666511 | 0.0150362 | 0.00746381 | 8929.69 | 17.2621 | 17.1475 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.271882 | 0.0350061 | 0.00959244 | 0.0993593 | 0.0389623 | 1.26414 | 0.244694 | 0.827931 | 0.605707 | 0.0285737 | 0.0579098 | 23520.2 | 91.3316 | 91.8903 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.0851064 | 0.0200104 | 0.0120655 | 0.0334764 | 0.0149725 | 0.733897 | 0.0765957 | 0.313563 | 0.621446 | 0.00970871 | 0.00129638 | 2765 | 1.0466 | 1.38666 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.321094 | 0.0184394 | 0.00786746 | 0.0969692 | 0.0297217 | 2.30108 | 0.288984 | 0.620785 | 0.716474 | 0.012502 | 0.00408368 | 5501.88 | 1.78727 | 6.56705 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.0960249 | 0.00444738 | 0.00170907 | 0.0223857 | 0.00614025 | 0.693671 | 0.0864224 | 0.717599 | 0.671233 | 0.00215807 | 0.0014387 | 687.294 | 0.00829473 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.47923 | 0.0195116 | 0.00865039 | 9554.09 | 351.447 | 310.773 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.257314 | 0.0175578 | 0.00636422 | 4019.09 | 2.77684 | 2.78187 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.546626 | 0.0240007 | 0.0101699 | 13276.6 | 171.63 | 152.984 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.489864 | 0.0207969 | 0.00928031 | 11667.4 | 429.185 | 379.514 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.292036 | 0.0163286 | 0.00581348 | 4139.8 | 2.86024 | 2.86542 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.639145 | 0.0271149 | 0.011384 | 18277.7 | 236.28 | 210.609 | inflated_ratio; inflated_ratio |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.0721228 | 0.00288697 | 0.00129425 | 443.136 | 0.00258477 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.0688928 | 0.00325311 | 0.000947676 | 259.155 | 0.0839411 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.147059 | 0.00720207 | 0.00288528 | 1359.59 | 0.0183607 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.145487 | 0.0200196 | 0.0149409 | 2068.41 | 9.86156 | 8.27721 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.370837 | 0.0172171 | 0.00886384 | 6065.03 | 249.861 | 197.383 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.481583 | 0.0194989 | 0.0086354 | 9537.41 | 350.833 | 310.23 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.257321 | 0.0175575 | 0.00636403 | 4021.71 | 2.77865 | 2.78368 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.546207 | 0.0239856 | 0.0101688 | 13230 | 171.027 | 152.446 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.191944 | 0.0181295 | 0.00677684 | 4472.38 | 470.265 | 467.603 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.429776 | 0.0615156 | 0.013796 | 57558 | 78.1126 | 78.6047 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.193925 | 0.0253733 | 0.00820447 | 8530.14 | 325.512 | 327.249 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.135785 | 0.0200107 | 0.0136468 | 2094.27 | 1.87053 | 3.89637 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.022339 | 0.0200191 | 0.0102469 | 3802.22 | 0.945908 | 0.995791 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0971949 | 0.0200013 | 0.0123027 | 2398.52 | 0.860802 | 1.47489 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.409623 | 0.0187037 | 0.00894249 | 6905.02 | 23.8119 | 123.8 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.209323 | 0.0207453 | 0.00741844 | 4956.27 | 0.640413 | 2.17316 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.344335 | 0.0158691 | 0.00724145 | 4644.33 | 3.85121 | 26.2466 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.0721228 | 0.00288697 | 0.00129425 | 443.136 | 0.00258477 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.0688928 | 0.00325311 | 0.000947676 | 259.155 | 0.0839411 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00778234, 0.0155647, 0.035747, 0.0389117, 0.0714941, 0.178735, 0.341719, 0.683438, 1.7086 | 0.147059 | 0.00720207 | 0.00288528 | 1359.59 | 0.0183607 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
