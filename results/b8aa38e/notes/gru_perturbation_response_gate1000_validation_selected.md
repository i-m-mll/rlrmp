# GRU perturbation-response bank

Issue: `b8aa38e`. Source experiment: `b8aa38e`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 12 |
| `delayed_observation` | 3 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 3 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 3 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 3 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 2
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.099684 | 0.00585595 | 0.00134883 | 0.0315814 | 0.00728778 | 0.367297 | 0.0598104 | 0.373917 | 0 | 0.000394442 | 0.00516002 | 632.436 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.11963 | 0.00250001 | 0.000813423 | 0.0128843 | 0.00427151 | 0.330049 | 0.0717781 | 0.457 | NA | -0.000421109 | 0.00174518 | 72.4908 | 3.41234 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0635684 | 0.01 | 0.00455691 | 0.0207579 | 0.00869807 | 0.182288 | 0.038141 | 0.014 | NA | 0.000415707 | 0.00238144 | 316.869 | 16.339 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.198389 | 0.0091434 | 0.00353485 | 0.0499525 | 0.0156496 | 0.657611 | 0.119034 | 0.239 | NA | 0.00043091 | 0.00257144 | 373.774 | 26.1568 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00920858 | 0.000530124 | 0.000131601 | 0.00278576 | 0.000665716 | 0.032241 | 0.00552515 | 0.4875 | NA | 5.99146e-06 | 2.50709e-05 | 3.51744 | 11.0595 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0277685 | 0.00228407 | 0.000451725 | 0.00907578 | 0.00242102 | 0.0991909 | 0.0166611 | 0.59 | NA | 0.000172798 | 0.000920306 | 63.004 | 57.2452 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.25306 | 0.05 | 0.0175039 | 0.0989677 | 0.0276721 | 0.931655 | 0.151836 | 0.229333 | NA | 0.0246734 | 0.0564507 | 15931.9 | 1.35687 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.169717 | 0.00896142 | 0.0024346 | 0.0499101 | 0.012182 | 0.621384 | 0.10183 | 0.445417 | NA | 0.00141725 | 0.00307232 | 870.499 | 0.910291 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0537102 | 0.00170879 | 0.000496972 | 0.00898005 | 0.00241063 | 0.176782 | 0.0322261 | 0.465 | 0.51 | -0.000815836 | 0.00137329 | 60.9849 | 2.87072 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.130548 | 0.00643706 | 0.00219739 | 529.784 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0866846 | 0.0068051 | 0.000686924 | 1070.63 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0818192 | 0.0043257 | 0.00116217 | 296.897 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.12491 | 0.00260187 | 0.000850695 | 101.381 | 2.30425 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.118596 | 0.00248794 | 0.000801533 | 35.0606 | 44.8732 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.115384 | 0.00241024 | 0.00078804 | 81.0306 | 4.27549 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0635684 | 0.01 | 0.00455691 | 316.869 | 16.339 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.198389 | 0.0091434 | 0.00353485 | 373.774 | 26.1568 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0109932 | 0.000571661 | 0.00019633 | 2.78606 | 26.8659 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00572178 | 0.000449142 | 4.53371e-05 | 4.66377 | 6.96894 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0109108 | 0.000569569 | 0.000153137 | 3.10249 | 17.1201 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0421702 | 0.00317747 | 0.000788635 | 83.3286 | 115.739 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0100485 | 0.001068 | 8.63599e-05 | 34.0806 | 19.5959 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0310868 | 0.00260674 | 0.00048018 | 71.6027 | 84.9732 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.271062 | 0.05 | 0.0218004 | 12129.8 | 2.43764 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.198814 | 0.05 | 0.0113503 | 18355.9 | 1.02747 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.289305 | 0.05 | 0.0193611 | 17310.1 | 1.39779 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.182799 | 0.0090885 | 0.00327015 | 597.537 | 1.80405 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.132115 | 0.00829233 | 0.00115146 | 1186.78 | 0.728046 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.194236 | 0.00950343 | 0.00288219 | 827.181 | 0.911443 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0530737 | 0.00163773 | 0.000527388 | 85.0442 | 1.93293 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0559142 | 0.0016701 | 0.000446807 | 34.6819 | 44.3887 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0521428 | 0.00181854 | 0.00051672 | 63.2284 | 3.33618 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 2
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.131282 | 0.0046656 | 0.00120372 | 0.0265753 | 0.00558967 | 0.625056 | 0.0787693 | 0.390167 | 0 | 0.00131711 | 0.00572275 | 385.733 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.112868 | 0.00265676 | 0.000847707 | 0.0141393 | 0.00371909 | 0.363041 | 0.0677205 | 0.472333 | NA | 0.000909224 | 0.00247102 | 214.885 | 10.1152 | inflated_ratio |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.029496 | 0.01 | 0.00537518 | 0.012353 | 0.00656973 | 0.0770026 | 0.0176976 | 0.0125 | NA | 0.00265907 | 0.00247242 | 504.237 | 26.0005 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.156503 | 0.0140348 | 0.00609708 | 0.0499525 | 0.0168717 | 0.429995 | 0.0939021 | 0.521 | NA | 0.00754481 | 0.00570978 | 1518.18 | 106.242 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0111803 | 0.000395813 | 0.000108096 | 0.00233051 | 0.000489534 | 0.0536847 | 0.00670817 | 0.49925 | NA | 1.0397e-05 | 2.49816e-05 | 1.96671 | 6.1837 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0308721 | 0.00309945 | 0.000542083 | 0.0120265 | 0.00325297 | 0.228744 | 0.0185233 | 0.59 | NA | 0.000789257 | 0.00236481 | 137.645 | 125.064 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.121366 | 0.05 | 0.0189926 | 0.0643512 | 0.0213398 | 0.642106 | 0.0728196 | 0.228333 | NA | 0.0355618 | 0.0353053 | 16888.8 | 1.43836 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.130812 | 0.0123723 | 0.00334501 | 0.0499101 | 0.0138514 | 0.676141 | 0.0784871 | 0.557583 | NA | 0.00639198 | 0.00638393 | 1546.49 | 1.61718 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0331062 | 0.00154492 | 0.000480941 | 0.00635457 | 0.0019491 | 0.0857168 | 0.0198637 | 0.496333 | NA | -0.000542412 | 0.00093919 | -5.85474 | -0.275599 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.108747 | 0.00679437 | 0.00227072 | 480.968 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.216723 | 0.00456344 | 0.000569761 | 510.796 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0683764 | 0.00263899 | 0.000770669 | 165.436 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.115364 | 0.00269361 | 0.000859773 | 219.779 | 4.99527 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.110866 | 0.00264743 | 0.000842185 | 213.844 | 273.694 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.112373 | 0.00262923 | 0.000841163 | 211.032 | 11.1349 | inflated_ratio |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.029496 | 0.01 | 0.00537518 | 504.237 | 26.0005 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.156503 | 0.0140348 | 0.00609708 | 1518.18 | 106.242 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00970353 | 0.00054222 | 0.000185685 | 2.59265 | 25.0008 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0147644 | 0.000300981 | 3.75849e-05 | 2.23406 | 3.3383 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00907294 | 0.000344238 | 0.000101017 | 1.07342 | 5.92334 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0492275 | 0.00476061 | 0.000985794 | 225.504 | 313.214 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.00762756 | 0.00111295 | 8.79463e-05 | 38.0269 | 21.8649 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0357614 | 0.00342478 | 0.000552509 | 149.405 | 177.304 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.115721 | 0.05 | 0.0248557 | 15399.4 | 3.09471 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.126502 | 0.05 | 0.0114529 | 17967.7 | 1.00574 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.121874 | 0.05 | 0.0206692 | 17299.3 | 1.39691 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.138509 | 0.0143768 | 0.00516986 | 1613.85 | 4.87242 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.118499 | 0.00974779 | 0.00121652 | 1635.41 | 1.00327 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.135427 | 0.0129925 | 0.00364865 | 1390.2 | 1.53182 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0352869 | 0.00167618 | 0.000522066 | -7.66532 | -0.174222 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0294134 | 0.00146691 | 0.000453884 | -9.70644 | -12.4231 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0346183 | 0.00149167 | 0.000466874 | -0.192472 | -0.0101556 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
