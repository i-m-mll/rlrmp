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

### `delayed_movement_bank`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.363008 | 0.0125559 | 0.00531696 | 0.0873486 | 0.0233568 | 2.06324 | 0.326707 | 0.546533 | 0.55618 | 0.00765543 | 0.00224172 | 1739.83 | 3.36328 | 3.34096 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.380859 | 0.0116496 | 0.00499331 | 0.08491 | 0.022487 | 2.09572 | 0.342773 | 0.533526 | 0.578884 | 0.0067681 | 0.00273494 | 1485.06 | 2.87078 | 2.85172 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.0882993 | 0.00290179 | 0.00115736 | 0.0177481 | 0.00430426 | 0.463846 | 0.0794693 | 0.604406 | 0.68398 | 0.00113138 | 0.000185463 | 171.084 | 0.00211109 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.251605 | 0.0200772 | 0.013819 | 0.0642333 | 0.0223462 | 1.42139 | 0.226445 | 0.0259115 | 0.689984 | 0.00698536 | 0.000763495 | 1319.51 | 6.29102 | 5.28031 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.432303 | 0.0129896 | 0.00683713 | 0.0954941 | 0.0303061 | 2.39378 | 0.389073 | 0.439781 | 0.590458 | 0.00774388 | 0.00145093 | 2014.62 | 82.9963 | 65.5648 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.364271 | 0.0125583 | 0.00531545 | 0.0873141 | 0.023374 | 2.06348 | 0.327844 | 0.547372 | 0.554626 | 0.00770483 | 0.00220414 | 1725.7 | 3.33598 | 3.31383 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.298293 | 0.0271862 | 0.0083484 | 0.0743259 | 0.0304593 | 1.18465 | 0.268463 | 0.882697 | 0.550278 | 0.023685 | 0.0297573 | 11248.1 | 43.6776 | 43.9448 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.158642 | 0.0200278 | 0.0112097 | 0.0456649 | 0.0163081 | 0.885174 | 0.142778 | 0.23457 | 0.658175 | 0.00944444 | 0.00175516 | 2110.91 | 0.799019 | 1.05863 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.376391 | 0.0141385 | 0.00627431 | 0.0954367 | 0.027145 | 2.17116 | 0.338752 | 0.531353 | 0.549705 | 0.00879363 | 0.00272615 | 2319.18 | 0.753379 | 2.76817 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.0882993 | 0.00290179 | 0.00115736 | 0.0177481 | 0.00430426 | 0.463846 | 0.0794693 | 0.604406 | 0.68398 | 0.00113138 | 0.000185463 | 171.084 | 0.00211109 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.432151 | 0.0116543 | 0.00550296 | 1560.9 | 57.4177 | 50.7726 | inflated_ratio; inflated_ratio |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.302351 | 0.0159291 | 0.00584251 | 2552.73 | 1.76371 | 1.76691 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.354522 | 0.0100843 | 0.0046054 | 1105.86 | 14.2957 | 12.7426 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.45152 | 0.0112003 | 0.00532492 | 1497.43 | 55.083 | 48.7081 | inflated_ratio; inflated_ratio |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.312022 | 0.0132791 | 0.00492266 | 1633.18 | 1.12838 | 1.13042 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.379034 | 0.0104693 | 0.00473234 | 1324.56 | 17.1229 | 15.2626 | inflated_ratio; inflated_ratio |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.0512787 | 0.00161264 | 0.00074538 | 43.2243 | 0.000257783 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.072873 | 0.00189519 | 0.0006172 | 64.1632 | 0.021245 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.140746 | 0.00519754 | 0.00210949 | 405.864 | 0.005604 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.251605 | 0.0200772 | 0.013819 | 1319.51 | 6.29102 | 5.28031 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.432303 | 0.0129896 | 0.00683713 | 2014.62 | 82.9963 | 65.5648 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.433221 | 0.0116664 | 0.00550321 | 1543.29 | 56.7698 | 50.1997 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.304531 | 0.0159321 | 0.00583854 | 2554.76 | 1.76512 | 1.76831 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.355062 | 0.0100764 | 0.00460459 | 1079.06 | 13.9493 | 12.4338 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.217483 | 0.0127592 | 0.0055026 | 1718.32 | 180.679 | 179.656 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.478856 | 0.0493262 | 0.012297 | 28158.1 | 38.2137 | 38.4544 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.198539 | 0.0194731 | 0.00724561 | 3867.78 | 147.595 | 148.383 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.245285 | 0.0200747 | 0.0125147 | 1336.59 | 1.1938 | 2.48673 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0320971 | 0.02 | 0.0100093 | 3479.45 | 0.865609 | 0.911257 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.198544 | 0.0200087 | 0.0111051 | 1516.7 | 0.544326 | 0.932641 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.473064 | 0.0137824 | 0.00673255 | 2558.04 | 8.82138 | 45.863 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.236384 | 0.0163138 | 0.0064254 | 2378.93 | 0.307388 | 1.04308 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.419726 | 0.0123193 | 0.00566497 | 2020.55 | 1.6755 | 11.4188 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.0512787 | 0.00161264 | 0.00074538 | 43.2243 | 0.000257783 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.072873 | 0.00189519 | 0.0006172 | 64.1632 | 0.021245 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00778017, 0.0155603, 0.0356049, 0.0389008, 0.0712098, 0.178025, 0.337942, 0.675884, 1.68971 | 0.140746 | 0.00519754 | 0.00210949 | 405.864 | 0.005604 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
