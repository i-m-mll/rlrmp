# GRU perturbation-response bank

Issue: `b35595c`. Source experiment: `b35595c`.

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

### `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.122451 | 0.00872538 | 0.00212994 | 0.0432027 | 0.0109785 | 0.460162 | 0.0734705 | 0.500833 | NA | 0.00193843 | 0.0104109 | 996.823 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.110557 | 0.0026903 | 0.000907554 | 0.0147693 | 0.00482925 | 0.284614 | 0.0663344 | 0.430667 | NA | -0.00170866 | -0.000835354 | -19.0739 | -0.897862 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0691091 | 0.01 | 0.00456582 | 0.0218489 | 0.00925987 | 0.183122 | 0.0414655 | 0.012 | NA | 0.000365737 | 0.00206272 | 372.012 | 19.1824 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.203396 | 0.0102708 | 0.0039962 | 0.0499525 | 0.0181852 | 0.631266 | 0.122037 | 0.271 | 0.5525 | 0.000712145 | 0.00439504 | 715.339 | 50.0595 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00807799 | 0.000575865 | 0.000140577 | 0.00285138 | 0.000724567 | 0.0303637 | 0.00484679 | 0.5005 | NA | 7.62469e-06 | 4.01288e-05 | 4.34176 | 13.6513 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0268307 | 0.00247915 | 0.000478392 | 0.0100126 | 0.0026225 | 0.0955359 | 0.0160984 | 0.59 | NA | 0.000226166 | 0.000688817 | 77.0123 | 69.9732 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.257605 | 0.05 | 0.0174433 | 0.101783 | 0.0288067 | 0.865915 | 0.154563 | 0.228333 | NA | 0.0247926 | 0.0593017 | 16792.2 | 1.43013 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.156601 | 0.009819 | 0.00258735 | 0.0499101 | 0.0134189 | 0.550005 | 0.0939607 | 0.459 | NA | 0.00200239 | 0.00643557 | 1142.69 | 1.19492 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0521253 | 0.00199171 | 0.000511983 | 0.0111193 | 0.00314206 | 0.166383 | 0.0312752 | 0.471333 | 0.49 | -0.0011501 | -0.000955615 | -43.2345 | -2.03517 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.161644 | 0.00947054 | 0.00320374 | 868.962 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0612015 | 0.00718759 | 0.000701004 | 1219.18 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.144507 | 0.009518 | 0.00248508 | 902.326 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.103664 | 0.00252191 | 0.000886241 | -18.1784 | -0.41317 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.109732 | 0.00273724 | 0.000906111 | -36.8495 | -47.1628 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118276 | 0.00281175 | 0.000930308 | -2.19382 | -0.115754 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0691091 | 0.01 | 0.00456582 | 372.012 | 19.1824 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.203396 | 0.0102708 | 0.0039962 | 715.339 | 50.0595 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0106658 | 0.000625057 | 0.000211443 | 3.78464 | 36.4952 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00403817 | 0.000474388 | 4.62664e-05 | 5.31092 | 7.93597 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00953003 | 0.00062815 | 0.000164023 | 3.92973 | 21.685 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0425322 | 0.00348264 | 0.000840712 | 103.795 | 144.166 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.00673343 | 0.00110367 | 8.76033e-05 | 37.1837 | 21.3801 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0312265 | 0.00285113 | 0.000506859 | 90.0583 | 106.875 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.318009 | 0.05 | 0.0215019 | 13340.3 | 2.68091 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.175895 | 0.05 | 0.0113933 | 18109.4 | 1.01367 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.278911 | 0.05 | 0.0194346 | 18926.8 | 1.52833 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.187991 | 0.00970174 | 0.00343922 | 853.524 | 2.57691 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.109759 | 0.00925325 | 0.00119507 | 1452.73 | 0.891197 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.172053 | 0.010502 | 0.00312778 | 1121.81 | 1.23608 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0549939 | 0.00193985 | 0.000510518 | -21.0877 | -0.479292 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0475991 | 0.00188252 | 0.000485485 | -85.2793 | -109.147 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0537827 | 0.00215276 | 0.000539946 | -23.3365 | -1.23132 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 1
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.100582 | 0.00372347 | 0.000945424 | 0.0234527 | 0.00554768 | 0.393673 | 0.0603494 | 0.240833 | 0 | 0.000572859 | 0.0015296 | 296.097 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.303163 | 0.00364987 | 0.00107359 | 0.0260668 | 0.00748395 | 0.878956 | 0.181898 | 0.380667 | 0.57 | -0.000386967 | 0.000631397 | 29.019 | 1.36601 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.104085 | 0.01 | 0.00393479 | 0.0283395 | 0.0100234 | 0.321025 | 0.0624507 | 0.012 | NA | 0.000305143 | 0.00159835 | 212.523 | 10.9585 | inflated_ratio |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.237317 | 0.00809157 | 0.00266678 | 0.0499525 | 0.0150767 | 0.838819 | 0.14239 | 0.202 | 0.535 | 0.000163278 | 0.000928211 | 191.576 | 13.4065 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0104741 | 0.000415114 | 9.82132e-05 | 0.00265417 | 0.000583622 | 0.0467386 | 0.00628446 | 0.4415 | NA | 6.7912e-06 | 1.74823e-05 | 2.27163 | 7.14243 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0330689 | 0.00170365 | 0.000361217 | 0.00718674 | 0.00181764 | 0.105837 | 0.0198414 | 0.59 | NA | 0.000272776 | 0.000532729 | 33.7458 | 30.6614 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.47698 | 0.05 | 0.0155185 | 0.143978 | 0.0346369 | 2.03639 | 0.286188 | 0.229667 | NA | 0.0199136 | 0.0768487 | 14848.7 | 1.26462 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.219851 | 0.00722196 | 0.0017769 | 0.0499101 | 0.0112053 | 0.992484 | 0.131911 | 0.391833 | 0.565 | 0.00141616 | 0.00548094 | 549.633 | 0.574758 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.107833 | 0.00150364 | 0.000416443 | 0.013054 | 0.00295984 | 0.416337 | 0.0646998 | 0.416667 | 0.445 | -0.000702942 | -0.000151135 | -1.73562 | -0.0817004 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.204297 | 0.00660927 | 0.0020864 | 333.704 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0560159 | 0.00303784 | 0.000328358 | 417.914 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0414347 | 0.0015233 | 0.000421513 | 136.673 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.300543 | 0.00373275 | 0.00110059 | 39.1912 | 0.890761 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.317941 | 0.00343808 | 0.00105472 | 35.0018 | 44.7981 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.291006 | 0.00377879 | 0.00106544 | 12.8641 | 0.678758 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.104085 | 0.01 | 0.00393479 | 212.523 | 10.9585 | inflated_ratio |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.237317 | 0.00809157 | 0.00266678 | 191.576 | 13.4065 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0134782 | 0.000436172 | 0.000137703 | 1.45237 | 14.0052 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0073576 | 0.000400331 | 4.33165e-05 | 3.6285 | 5.42198 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0105865 | 0.000408839 | 0.00011362 | 1.73402 | 9.56867 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0477785 | 0.00217602 | 0.000608212 | 36.8471 | 51.1787 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0145012 | 0.000995335 | 8.35782e-05 | 28.4127 | 16.3369 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0369271 | 0.00193959 | 0.00039186 | 35.9776 | 42.6958 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.550493 | 0.05 | 0.0183891 | 8529.78 | 1.71417 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.312314 | 0.05 | 0.0111278 | 20350 | 1.13909 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.568133 | 0.05 | 0.0170386 | 15666.5 | 1.26506 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.247285 | 0.00735051 | 0.00219438 | 275.057 | 0.830434 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.154679 | 0.00703169 | 0.00106471 | 904.036 | 0.554593 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.257589 | 0.00728367 | 0.0020716 | 469.807 | 0.517665 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0902004 | 0.00134242 | 0.000391818 | -41.3283 | -0.939334 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.115842 | 0.0016857 | 0.000433597 | 47.2872 | 60.5219 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.117457 | 0.00148279 | 0.000423916 | -11.1657 | -0.589148 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
