# GRU perturbation-response bank

Issue: `c92ebd8`. Source experiment: `c92ebd8`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 18 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 36 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 36 |
| `target_aligned_lateral_command_load_pulse` | 6 |
| `target_stream_jump` | 1 |

## Evaluation

### `open_loop_moderate`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.19434 | 0.00344287 | 0.000783612 | 0.0351807 | 0.00456031 | 1.27803 | 0.116604 | 0.37756 | NA | 0.000927556 | 0.00142526 | 244.795 | 1.1821 | 1.17519 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.197845 | 0.003337 | 0.000764894 | 0.0349544 | 0.00437835 | 1.30749 | 0.118707 | 0.371099 | NA | 0.00113306 | 0.00107079 | 241.417 | 1.16579 | 1.15897 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.279811 | 0.015 | 0.00466758 | 0.0643806 | 0.0147705 | 0.961548 | 0.167886 | 0.0308672 | NA | 0.000341783 | 0.00120136 | 148.833 | 1.77398 | 1.48897 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.191514 | 0.00312371 | 0.000765965 | 0.02832 | 0.00697027 | 0.72382 | 0.114908 | 0.143898 | 0.298921 | 6.94278e-05 | 0.00023833 | 21.3802 | 2.202 | 1.73952 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.194435 | 0.00344246 | 0.000783487 | 0.0351812 | 0.00456036 | 1.27811 | 0.116661 | 0.377201 | NA | 0.000918322 | 0.00134225 | 244.797 | 1.18211 | 1.1752 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.125807 | 0.015 | 0.00471317 | 0.0337034 | 0.00849598 | 0.532666 | 0.0754841 | 0.232487 | NA | 0.00625878 | 0.00840153 | 1058.09 | 1.00128 | 1.32659 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.154377 | 0.00577401 | 0.00132517 | 0.0449007 | 0.00756741 | 0.875938 | 0.0926265 | 0.38587 | NA | 0.00233486 | 0.0015599 | 514.582 | 0.411642 | 1.53551 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152661, 0.0738732, 0.741371 | 0.243209 | 0.00284426 | 0.000688351 | 0.0291872 | 0.00378744 | 1.12652 | 0.145926 | 0.418773 | NA | 0.000544851 | 0.00159129 | 146.08 | 0.554578 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.112286 | 0.00254409 | 0.000820517 | 26.9787 | 2.48102 | 2.19389 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.322267 | 0.00520612 | 0.000778282 | 658.214 | 1.13595 | 1.13898 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.148468 | 0.00257839 | 0.000752038 | 49.1929 | 1.5898 | 1.4171 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.111657 | 0.00233707 | 0.000770636 | 23.2689 | 2.13985 | 1.89221 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.320181 | 0.00515934 | 0.000777465 | 651.75 | 1.1248 | 1.12779 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.161698 | 0.00251458 | 0.000746581 | 49.2338 | 1.59112 | 1.41828 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.279811 | 0.015 | 0.00466758 | 148.833 | 1.77398 | 1.48897 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.191514 | 0.00312371 | 0.000765965 | 21.3802 | 2.202 | 1.73952 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.112288 | 0.00254372 | 0.0008201 | 26.9441 | 2.47783 | 2.19107 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.322579 | 0.00520477 | 0.000778137 | 658.234 | 1.13599 | 1.13901 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.148439 | 0.0025789 | 0.000752225 | 49.2134 | 1.59046 | 1.41769 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.187375 | 0.015 | 0.00552287 | 518.916 | 1.15879 | 2.4136 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.049956 | 0.015 | 0.00336195 | 1625.21 | 1.01078 | 1.0641 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.14009 | 0.015 | 0.0052547 | 1030.14 | 0.924288 | 1.58362 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.1435 | 0.00380148 | 0.00116897 | 43.3984 | 0.366524 | 1.94522 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.174897 | 0.00895187 | 0.00139342 | 1363.98 | 0.434829 | 1.49515 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.144735 | 0.00456867 | 0.0014131 | 136.364 | 0.275491 | 1.9266 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.222786 | 0.00320351 | 0.000995593 | 77.9652 | 0.390897 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.243596 | 0.00230163 | 0.000245401 | 254.756 | 0.76478 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.263246 | 0.00302765 | 0.000824059 | 105.521 | 0.409529 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `moderate_pgd_ofb1p05`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.225004 | 0.00320688 | 0.00059661 | 0.0348771 | 0.00464861 | 1.43608 | 0.135002 | 0.364021 | 0.485829 | 0.00133107 | 0.000488793 | 220.834 | 1.06639 | 1.06016 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.227325 | 0.00316236 | 0.000578017 | 0.034732 | 0.00457678 | 1.45399 | 0.136395 | 0.361156 | 0.506749 | 0.00132594 | 0.000578051 | 222.894 | 1.07634 | 1.07005 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.364844 | 0.015 | 0.00380053 | 0.079112 | 0.0163091 | 1.14921 | 0.218907 | 0.0302656 | 0.350885 | 0.000151477 | 0.000111049 | 106.556 | 1.27007 | 1.06602 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.169178 | 0.0032047 | 0.000700605 | 0.0258152 | 0.00654376 | 0.603553 | 0.101507 | 0.151844 | 0.331664 | 5.47684e-05 | 4.7686e-05 | 14.363 | 1.47928 | 1.16859 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.224972 | 0.00320729 | 0.000596504 | 0.0348777 | 0.0046488 | 1.43543 | 0.134983 | 0.363995 | 0.484041 | 0.00134099 | 0.000450637 | 220.757 | 1.06602 | 1.05979 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.275424 | 0.015 | 0.00389713 | 0.0560138 | 0.0105104 | 1.06357 | 0.165254 | 0.232714 | 0.483867 | 0.00563003 | 0.00249106 | 884.916 | 0.837404 | 1.10947 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.205764 | 0.00547359 | 0.00106999 | 0.0449102 | 0.00784078 | 1.09552 | 0.123458 | 0.375318 | 0.447887 | 0.00272413 | 0.000772353 | 465.471 | 0.372356 | 1.38897 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152661, 0.0738732, 0.741371 | 0.363563 | 0.00325515 | 0.000640005 | 0.0362807 | 0.00491548 | 1.59709 | 0.218138 | 0.40552 | 0.530478 | 0.00101741 | 0.000694246 | 170.476 | 0.647192 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.153671 | 0.00220736 | 0.000475064 | 15.342 | 1.41088 | 1.2476 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.319383 | 0.00493356 | 0.000752579 | 606.267 | 1.0463 | 1.04909 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.201956 | 0.00247971 | 0.000562188 | 40.893 | 1.32157 | 1.178 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.153779 | 0.00208695 | 0.000434668 | 13.8372 | 1.27249 | 1.12523 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.314788 | 0.00497576 | 0.000757038 | 614.049 | 1.05973 | 1.06256 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.213407 | 0.00242438 | 0.000542346 | 40.7966 | 1.31845 | 1.17523 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.364844 | 0.015 | 0.00380053 | 106.556 | 1.27007 | 1.06602 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.169178 | 0.0032047 | 0.000700605 | 14.363 | 1.47928 | 1.16859 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.153704 | 0.00220715 | 0.00047454 | 15.3199 | 1.40885 | 1.2458 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.319441 | 0.00493339 | 0.000752465 | 606.091 | 1.046 | 1.04879 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.201771 | 0.00248132 | 0.000562507 | 40.8598 | 1.3205 | 1.17705 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.363835 | 0.015 | 0.00409363 | 293.004 | 0.654305 | 1.36283 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0824318 | 0.015 | 0.00335058 | 1578.05 | 0.981445 | 1.03322 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.380005 | 0.015 | 0.00424719 | 783.696 | 0.703169 | 1.20476 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.195674 | 0.00361305 | 0.00080039 | 34.983 | 0.295451 | 1.56802 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.182492 | 0.00853538 | 0.00135787 | 1257.76 | 0.400966 | 1.37871 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.239125 | 0.00427234 | 0.00105169 | 103.672 | 0.209444 | 1.4647 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.363145 | 0.00353331 | 0.000788142 | 70.9275 | 0.355612 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.294208 | 0.00247934 | 0.000273072 | 287.055 | 0.861745 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.433336 | 0.0037528 | 0.0008588 | 153.444 | 0.595521 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `moderate_pgd_ofb1p4`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.245884 | 0.00326359 | 0.00054987 | 0.0352173 | 0.004848 | 1.44581 | 0.147531 | 0.359622 | 0.466266 | 0.00125972 | 0.00156192 | 266.124 | 1.2851 | 1.27758 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.247204 | 0.00320509 | 0.000527323 | 0.035023 | 0.00475844 | 1.47703 | 0.148322 | 0.356995 | 0.46668 | 0.00127603 | 0.00146144 | 265.619 | 1.28266 | 1.27516 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.388251 | 0.015 | 0.00383356 | 0.0794781 | 0.0166238 | 1.1984 | 0.23295 | 0.0307031 | 0.36368 | 4.59569e-05 | 6.11363e-05 | 116.732 | 1.39136 | 1.16783 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.186426 | 0.00315563 | 0.000674295 | 0.0257881 | 0.00651158 | 0.660517 | 0.111856 | 0.146187 | 0.325008 | 3.0534e-05 | 8.09834e-06 | 14.2445 | 1.46708 | 1.15895 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.245991 | 0.00326295 | 0.000550211 | 0.0352165 | 0.00484406 | 1.44679 | 0.147595 | 0.35968 | 0.454552 | 0.00124085 | 0.00162299 | 266.522 | 1.28702 | 1.27949 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.406503 | 0.015 | 0.00359926 | 0.0699369 | 0.0114061 | 1.64307 | 0.243902 | 0.232766 | 0.476148 | 0.00455156 | 0.00259358 | 906.325 | 0.857664 | 1.13631 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.261595 | 0.00535111 | 0.000954833 | 0.0452423 | 0.00797484 | 1.31337 | 0.156957 | 0.36412 | 0.449787 | 0.00235848 | 0.00147333 | 478.124 | 0.382478 | 1.42673 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152661, 0.0738732, 0.741371 | 0.504193 | 0.00343714 | 0.00054009 | 0.0423438 | 0.00580388 | 2.23329 | 0.302516 | 0.39345 | 0.467674 | 0.000730863 | 0.0011897 | 249.59 | 0.947538 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.178177 | 0.00205914 | 0.000408489 | 13.5796 | 1.2488 | 1.10428 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.30257 | 0.0054951 | 0.0008072 | 743.027 | 1.28232 | 1.28574 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.256906 | 0.00223652 | 0.000433923 | 41.7672 | 1.34982 | 1.20319 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.179971 | 0.00191677 | 0.000349513 | 14.0618 | 1.29315 | 1.14349 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.29876 | 0.00551368 | 0.00080921 | 739.471 | 1.27619 | 1.27959 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.26288 | 0.00218483 | 0.000423246 | 43.324 | 1.40013 | 1.24803 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.388251 | 0.015 | 0.00383356 | 116.732 | 1.39136 | 1.16783 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.186426 | 0.00315563 | 0.000674295 | 14.2445 | 1.46708 | 1.15895 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.178412 | 0.00205843 | 0.000407813 | 13.4892 | 1.24049 | 1.09693 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.302341 | 0.00549404 | 0.000807042 | 744.177 | 1.28431 | 1.28773 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.25722 | 0.00223638 | 0.00043578 | 41.9012 | 1.35415 | 1.20705 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.472495 | 0.015 | 0.00377902 | 294.323 | 0.65725 | 1.36896 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0981334 | 0.015 | 0.00333798 | 1591.76 | 0.989976 | 1.0422 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.64888 | 0.015 | 0.00368076 | 832.888 | 0.747306 | 1.28039 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.24361 | 0.00347742 | 0.000701072 | 33.2198 | 0.28056 | 1.48899 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.199734 | 0.00860698 | 0.00136421 | 1296.43 | 0.413293 | 1.4211 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.34144 | 0.00396892 | 0.000799216 | 104.727 | 0.211576 | 1.47961 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.519342 | 0.00350735 | 0.000642292 | 85.1687 | 0.427013 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.303706 | 0.00298283 | 0.000309491 | 407.543 | 1.22345 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152661, 0.0738732, 0.741371 | 0.68953 | 0.00382125 | 0.000668487 | 256.057 | 0.993766 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
