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

### `smoke__broad_strong_cal_small`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 2
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.000477614 | 0.00434194 | 0.000819179 | 0.0127584 | 0.00455888 | 0.000965797 | 0.000286569 | 0.1475 | 0 | -0.00128089 | 0.00945946 | -1533.38 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.000750948 | 1.9266e-05 | 4.11368e-06 | 7.81585e-05 | 2.46001e-05 | 0.00266989 | 0.000450569 | 0.59 | NA | 1.72805e-05 | -3.88062e-05 | 27.2304 | 1.28181 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.000689927 | 0.01 | 0.00598671 | 0.01 | 0.00598814 | 0.000850628 | 0.000413956 | 0.00375 | NA | 0.000172398 | 5.50264e-07 | 758.596 | 39.1162 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.00286942 | 0.0290905 | 0.00896772 | 0.0551146 | 0.0308155 | 0.00371924 | 0.00172165 | 0.59 | NA | 0.00143201 | 0.0305846 | 6499.15 | 454.812 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 9.76409e-05 | 0.00104048 | 0.000204149 | 0.00334831 | 0.0010989 | 0.000221178 | 5.85846e-05 | 0.59 | NA | 2.13208e-06 | 0.000135577 | 12.9028 | 40.5687 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.00034278 | 0.00397597 | 0.000620434 | 0.019365 | 0.0042193 | 0.00105198 | 0.000205668 | 0.59 | NA | 3.55884e-05 | 0.00628947 | 275.684 | 250.486 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.00212206 | 0.05 | 0.0198137 | 0.05 | 0.0198158 | 0.0041058 | 0.00127324 | 0.229167 | NA | 0.00420282 | 4.53932e-06 | 18718.1 | 1.59416 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.00182111 | 0.0189246 | 0.00416526 | 0.051938 | 0.0200314 | 0.00369797 | 0.00109267 | 0.59 | NA | 0.000672852 | 0.0317499 | 3737.77 | 3.90862 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 1.3503e-06 | 2.47334e-05 | 4.70405e-06 | 8.89812e-05 | 2.67491e-05 | 2.96012e-06 | 8.10182e-07 | 0.59 | NA | 2.45486e-05 | -6.1597e-05 | 38.5369 | 1.81404 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0 | 0 | 0 | 0 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0 | 0 | 0 | 0 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.00143284 | 0.0130258 | 0.00245754 | -4600.15 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.000750948 | 1.9266e-05 | 4.11368e-06 | 27.2304 | 0.618908 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.000750948 | 1.9266e-05 | 4.11368e-06 | 27.2304 | 34.8516 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.000750948 | 1.9266e-05 | 4.11368e-06 | 27.2304 | 1.43678 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.000689927 | 0.01 | 0.00598671 | 758.596 | 39.1162 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.00286942 | 0.0290905 | 0.00896772 | 6499.15 | 454.812 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.000143082 | 0.00146388 | 0.000348572 | 18.6229 | 179.581 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 3.92941e-05 | 0.000509977 | 4.74879e-05 | 6.38848 | 9.54614 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.000110547 | 0.00114759 | 0.000216388 | 13.6969 | 75.582 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.000588184 | 0.00652341 | 0.00115916 | 514.426 | 714.512 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 7.31307e-05 | 0.00115222 | 8.91971e-05 | 41.8049 | 24.0372 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.000367026 | 0.00425229 | 0.000612943 | 270.822 | 321.393 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.00295177 | 0.05 | 0.0264602 | 19033 | 3.82492 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.00108577 | 0.05 | 0.0114988 | 18031.8 | 1.00933 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00232864 | 0.05 | 0.0214821 | 19089.5 | 1.54147 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.00249092 | 0.025324 | 0.00677672 | 5197.79 | 15.6928 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.000958347 | 0.0108844 | 0.00126107 | 2145.21 | 1.31601 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00201406 | 0.0205654 | 0.004458 | 3870.29 | 4.26455 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 1.3503e-06 | 2.47334e-05 | 4.70405e-06 | 38.5369 | 0.875889 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 1.3503e-06 | 2.47334e-05 | 4.70405e-06 | 38.5369 | 49.3225 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 1.3503e-06 | 2.47334e-05 | 4.70405e-06 | 38.5369 | 2.03336 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `smoke__proprio_cal_stress`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 2
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.00278776 | 0.00405282 | 0.000777686 | 0.0116968 | 0.00425697 | 0.0145343 | 0.00167266 | 0.1475 | 0 | -0.00112859 | 0.00837006 | -1395.54 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.0014604 | 4.93006e-05 | 1.08388e-05 | 0.00017168 | 6.00341e-05 | 0.00386032 | 0.000876237 | 0.59 | NA | 4.84916e-05 | -0.00015075 | 72.2138 | 3.3993 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.00117036 | 0.0100322 | 0.0059954 | 0.0100447 | 0.00599852 | 0.00142424 | 0.000702215 | 0.30625 | NA | 0.00017589 | 1.46028e-06 | 763.97 | 39.3933 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.00881859 | 0.0298957 | 0.00910225 | 0.0584628 | 0.0316414 | 0.0124237 | 0.00529115 | 0.59 | NA | 0.00146589 | 0.0325512 | 6928.59 | 484.865 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.000806744 | 0.00096401 | 0.000191693 | 0.00304369 | 0.00101836 | 0.00454964 | 0.000484046 | 0.59 | NA | 2.08373e-06 | 0.000110575 | 11.0847 | 34.8522 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.00107914 | 0.0040114 | 0.000623955 | 0.0196698 | 0.00425779 | 0.00329874 | 0.000647484 | 0.59 | NA | 3.59622e-05 | 0.00549851 | 282.552 | 256.726 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.00355106 | 0.0500601 | 0.0198265 | 0.050084 | 0.019831 | 0.00705897 | 0.00213064 | 0.409167 | NA | 0.00424371 | 1.38405e-05 | 18763.2 | 1.598 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.00564852 | 0.0192302 | 0.00420389 | 0.0537828 | 0.0203513 | 0.0122358 | 0.00338911 | 0.59 | NA | 0.000683739 | 0.0321944 | 3894.02 | 4.07202 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 4.23914e-05 | 2.21024e-05 | 4.16694e-06 | 8.30163e-05 | 2.39427e-05 | 0.000113073 | 2.54349e-05 | 0.59 | NA | 2.20128e-05 | -5.48373e-05 | 34.7606 | 1.63628 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0 | 0 | 0 | 0 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0 | 0 | 0 | 0 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.00836328 | 0.0121585 | 0.00233306 | -4186.63 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0014604 | 4.93006e-05 | 1.08388e-05 | 72.2138 | 1.64132 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0014604 | 4.93006e-05 | 1.08388e-05 | 72.2138 | 92.4248 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0014604 | 4.93006e-05 | 1.08388e-05 | 72.2138 | 3.81028 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.00117036 | 0.0100322 | 0.0059954 | 763.97 | 39.3933 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.00881859 | 0.0298957 | 0.00910225 | 6928.59 | 484.865 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.000925233 | 0.00134623 | 0.000325264 | 15.9294 | 153.607 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.000653533 | 0.000486727 | 4.62303e-05 | 5.69639 | 8.51196 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.000841467 | 0.00105907 | 0.000203584 | 11.6282 | 64.1668 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0018424 | 0.00659675 | 0.00116705 | 529.502 | 735.452 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.000239746 | 0.00115377 | 8.92551e-05 | 41.957 | 24.1247 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00115527 | 0.00428369 | 0.000615564 | 276.197 | 327.771 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.00497501 | 0.0501111 | 0.0264862 | 19120.9 | 3.8426 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.00178399 | 0.0500074 | 0.0114996 | 18035.2 | 1.00952 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0038942 | 0.0500617 | 0.0214938 | 19133.6 | 1.54503 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.00768192 | 0.0258828 | 0.00685588 | 5484.49 | 16.5584 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.00305776 | 0.0109259 | 0.00126304 | 2167.69 | 1.3298 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00620589 | 0.0208821 | 0.00449274 | 4029.88 | 4.44039 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 4.23914e-05 | 2.21024e-05 | 4.16694e-06 | 34.7606 | 0.79006 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 4.23914e-05 | 2.21024e-05 | 4.16694e-06 | 34.7606 | 44.4894 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 4.23914e-05 | 2.21024e-05 | 4.16694e-06 | 34.7606 | 1.83411 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
