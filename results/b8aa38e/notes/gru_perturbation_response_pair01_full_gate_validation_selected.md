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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.112319 | 0.00290934 | 0.000723126 | 0.0239777 | 0.00412062 | 0.620816 | 0.0673916 | 0.269583 | 0 | 0.000234451 | 0.00143757 | 133.716 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.901293 | 0.00255272 | 0.000816474 | 0.0214227 | 0.00610964 | 2.39913 | 0.540776 | 0.371 | 0.52 | -0.00018937 | 0.000933678 | -9.50709 | -0.447525 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.204694 | 0.01 | 0.0027699 | 0.0446211 | 0.0102039 | 1.08565 | 0.122816 | 0.0175 | 0.5295 | 9.20225e-05 | 0.000241149 | 46.7627 | 2.41127 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.466458 | 0.00510429 | 0.00156055 | 0.0592436 | 0.0144594 | 2.91459 | 0.279875 | 0.113 | 0.23062 | 0.000371153 | 0.00116798 | 124.585 | 8.71848 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0111183 | 0.000288789 | 7.3357e-05 | 0.00237764 | 0.000409147 | 0.0623293 | 0.00667099 | 0.400167 | NA | 2.62747e-06 | 1.30899e-05 | 0.899521 | 2.82826 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0351495 | 0.00119512 | 0.000273834 | 0.00526556 | 0.00127639 | 0.113673 | 0.0210897 | 0.59 | NA | 0.000111694 | 0.000328181 | 15.403 | 13.9952 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.559885 | 0.05 | 0.0148313 | 0.137909 | 0.0322426 | 2.75851 | 0.335931 | 0.228083 | NA | 0.0214287 | 0.0495471 | 11912.5 | 1.01455 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.262163 | 0.00533806 | 0.00130545 | 0.0499101 | 0.00852163 | 1.57273 | 0.157298 | 0.34425 | NA | 0.00050178 | 0.00266403 | 278.699 | 0.291439 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.15255 | 0.00160727 | 0.000490837 | 0.0141776 | 0.002965 | 1.00429 | 0.09153 | 0.344667 | 0.395556 | -0.000721782 | 0.000663455 | 26.472 | 1.24611 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.138189 | 0.00361589 | 0.00114861 | 63.2221 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.113917 | 0.00289206 | 0.000391349 | 262.487 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0848517 | 0.00222007 | 0.000629418 | 75.4376 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.901293 | 0.00255272 | 0.000816474 | -9.50709 | -0.216083 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.901293 | 0.00255272 | 0.000816474 | -9.50709 | -12.1679 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.901293 | 0.00255272 | 0.000816474 | -9.50709 | -0.501631 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.204694 | 0.01 | 0.0027699 | 46.7627 | 2.41127 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.466458 | 0.00510429 | 0.00156055 | 124.585 | 8.71848 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0121691 | 0.000321296 | 0.000102559 | 0.46984 | 4.53066 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0102525 | 0.000253497 | 3.43427e-05 | 1.57879 | 2.35915 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0109334 | 0.000291573 | 8.31695e-05 | 0.64993 | 3.58643 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.047242 | 0.00154047 | 0.000456546 | 18.0776 | 25.1089 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0200886 | 0.000720766 | 7.03534e-05 | 12.7772 | 7.34672 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0381179 | 0.00132412 | 0.000294604 | 15.3542 | 18.2214 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.666633 | 0.05 | 0.0173271 | 5007.67 | 1.00636 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.434722 | 0.05 | 0.0107486 | 20380.6 | 1.1408 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.5783 | 0.05 | 0.016418 | 10349.2 | 0.835698 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.249854 | 0.00565614 | 0.00160122 | 82.8593 | 0.250164 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.289983 | 0.00487514 | 0.000782179 | 587.019 | 0.360115 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.246652 | 0.0054829 | 0.00153295 | 166.22 | 0.183153 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.118743 | 0.00186914 | 0.000591298 | 7.99333 | 0.181677 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.219862 | 0.00138336 | 0.00038631 | 61.5239 | 78.7431 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.119045 | 0.00156932 | 0.000494903 | 9.89886 | 0.522302 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.117998 | 0.00162906 | 0.00039041 | 0.019226 | 0.00261115 | 0.946466 | 0.0707989 | 0.2335 | 0 | -5.90845e-05 | 0.000102429 | 38.2507 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.667061 | 0.00240023 | 0.000862638 | 0.0216259 | 0.00587375 | 1.84893 | 0.400237 | 0.406 | NA | -0.000495455 | -0.00068626 | -30.5988 | -1.44037 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.252721 | 0.01 | 0.00263148 | 0.0512269 | 0.0103177 | 0.959659 | 0.151633 | 0.0185 | 0.461762 | 1.60809e-05 | 3.64778e-05 | 41.767 | 2.15367 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.38013 | 0.00561433 | 0.00118637 | 0.0500234 | 0.0121142 | 1.71228 | 0.228078 | 0.131 | 0.30225 | -4.17077e-05 | 1.97001e-05 | 51.4002 | 3.59699 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0119625 | 0.000162321 | 3.89431e-05 | 0.00189979 | 0.000267399 | 0.0914278 | 0.00717748 | 0.346417 | NA | 2.16692e-07 | 2.80372e-06 | 0.325039 | 1.02198 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0354034 | 0.00151336 | 0.000327311 | 0.00580221 | 0.00160215 | 0.113663 | 0.021242 | 0.59 | NA | 0.000169168 | 0.000254908 | 25.9062 | 23.5383 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.519488 | 0.05 | 0.0151174 | 0.126063 | 0.0306329 | 2.6057 | 0.311693 | 0.229583 | NA | 0.0231434 | 0.0391942 | 11378.8 | 0.969097 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.202959 | 0.00602534 | 0.00160497 | 0.0499101 | 0.0089307 | 1.20975 | 0.121776 | 0.365417 | NA | 0.00101474 | 0.00438793 | 330.121 | 0.345211 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.153016 | 0.00111506 | 0.000355968 | 0.0116977 | 0.00227024 | 0.794926 | 0.0918097 | 0.333667 | 0.36625 | -0.000607736 | 9.92474e-05 | 0.559248 | 0.0263253 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.130643 | 0.00224633 | 0.000654946 | 6.62163 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.139347 | 0.00147477 | 0.000220983 | 89.3688 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.084004 | 0.0011661 | 0.000295301 | 18.7619 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.667061 | 0.00240023 | 0.000862638 | -30.5988 | -0.695468 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.667061 | 0.00240023 | 0.000862638 | -30.5988 | -39.1628 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.667061 | 0.00240023 | 0.000862638 | -30.5988 | -1.61451 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.252721 | 0.01 | 0.00263148 | 41.767 | 2.15367 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.38013 | 0.00561433 | 0.00118637 | 51.4002 | 3.59699 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0114159 | 0.000200313 | 5.90407e-05 | 0.144533 | 1.39372 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0129161 | 0.00012877 | 1.84773e-05 | 0.6625 | 0.989956 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0115555 | 0.000157881 | 3.93114e-05 | 0.168086 | 0.927529 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0480308 | 0.00201786 | 0.000550765 | 32.6016 | 45.2819 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0194262 | 0.000812368 | 7.52273e-05 | 17.1223 | 9.84511 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0387532 | 0.00170986 | 0.000355941 | 27.9946 | 33.2221 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.664031 | 0.05 | 0.0180118 | 5385.96 | 1.08238 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.271826 | 0.05 | 0.0109465 | 18864.4 | 1.05594 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.622607 | 0.05 | 0.0163937 | 9886.16 | 0.798306 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.211377 | 0.00663067 | 0.0021929 | 151.954 | 0.45877 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.182585 | 0.00561327 | 0.000910828 | 642.181 | 0.393955 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.214915 | 0.00583208 | 0.00171118 | 196.228 | 0.216217 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.118036 | 0.0014422 | 0.000481272 | -11.5172 | -0.26177 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.181384 | 0.000793595 | 0.000232189 | 20.5744 | 26.3327 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.159629 | 0.00110937 | 0.000354444 | -7.37939 | -0.389365 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
