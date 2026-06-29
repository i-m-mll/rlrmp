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

### `open_loop_small`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.151412 | 0.00414963 | 0.000910553 | 0.0370126 | 0.00539334 | 0.922814 | 0.0908472 | 0.39457 | NA | 0.00138261 | 0.0034417 | 372.338 | 1.79943 | 1.78748 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.149044 | 0.00411024 | 0.000906205 | 0.0368742 | 0.00531827 | 0.928824 | 0.0894264 | 0.394016 | NA | 0.00159009 | 0.00368447 | 371.431 | 1.79504 | 1.78313 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.207473 | 0.015 | 0.0053595 | 0.0521488 | 0.0140679 | 0.663431 | 0.124484 | 0.0306875 | NA | 0.00102725 | 0.00357254 | 265.521 | 3.16481 | 2.65636 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.183426 | 0.00331958 | 0.000992216 | 0.0305368 | 0.00745452 | 0.63549 | 0.110056 | 0.155008 | 0.403184 | 0.000178921 | 0.000555716 | 31.3596 | 3.22981 | 2.55146 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.151458 | 0.0041491 | 0.000910473 | 0.0370148 | 0.00539343 | 0.922924 | 0.0908748 | 0.39451 | NA | 0.00137437 | 0.00334305 | 372.261 | 1.79905 | 1.78711 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.088294 | 0.015 | 0.00487578 | 0.0311769 | 0.00862187 | 0.403266 | 0.0529764 | 0.232135 | NA | 0.00621012 | 0.0145761 | 1183.7 | 1.12013 | 1.48407 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.13038 | 0.0061011 | 0.00142622 | 0.0449007 | 0.00801481 | 0.709597 | 0.0782278 | 0.391953 | NA | 0.00248918 | 0.0039692 | 589.36 | 0.478631 | 1.75866 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.16091 | 0.00286778 | 0.000664244 | 0.0249961 | 0.00378283 | 0.795607 | 0.0965461 | 0.444227 | NA | 0.000526429 | 0.00158607 | 142.629 | 0.581102 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.0946387 | 0.00275578 | 0.00092233 | 37.1424 | 3.41571 | 3.0204 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.242522 | 0.00664552 | 0.000923722 | 1009.49 | 1.74367 | 1.74682 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.117075 | 0.00304757 | 0.000885607 | 70.3862 | 2.27475 | 2.02761 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.090797 | 0.00265816 | 0.00090404 | 35.9949 | 3.31018 | 2.92709 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.239882 | 0.00663788 | 0.000922776 | 1008.12 | 1.7413 | 1.74445 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.116453 | 0.00303468 | 0.000891799 | 70.1815 | 2.26813 | 2.02171 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.207473 | 0.015 | 0.0053595 | 265.521 | 3.16481 | 2.65636 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.183426 | 0.00331958 | 0.000992216 | 31.3596 | 3.22981 | 2.55146 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.0946486 | 0.00275568 | 0.000922115 | 37.1396 | 3.41545 | 3.02017 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.242652 | 0.00664375 | 0.000923577 | 1009.21 | 1.74319 | 1.74634 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.117073 | 0.00304788 | 0.000885726 | 70.4348 | 2.27632 | 2.02901 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.118194 | 0.015 | 0.00590377 | 694.344 | 1.55041 | 3.22956 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0588315 | 0.015 | 0.00334654 | 1677.61 | 1.04338 | 1.09841 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.0878566 | 0.015 | 0.00537702 | 1179.14 | 1.05795 | 1.81267 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.113553 | 0.00397914 | 0.00130579 | 60.5004 | 0.521587 | 2.71177 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.171004 | 0.00940055 | 0.00143839 | 1530.63 | 0.494442 | 1.67782 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.106582 | 0.0049236 | 0.00153448 | 176.953 | 0.366835 | 2.50004 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.179699 | 0.00336462 | 0.00104265 | 97.7288 | 0.521959 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.147124 | 0.00246269 | 0.000222055 | 240.903 | 0.776402 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.155908 | 0.00277604 | 0.000728024 | 89.2539 | 0.373731 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.19434 | 0.00344287 | 0.000783612 | 0.0351807 | 0.00456031 | 1.27803 | 0.116604 | 0.37756 | NA | 0.000927556 | 0.00142526 | 244.795 | 1.18304 | 1.17519 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.197845 | 0.003337 | 0.000764894 | 0.0349544 | 0.00437835 | 1.30749 | 0.118707 | 0.371099 | NA | 0.00113306 | 0.00107079 | 241.417 | 1.16672 | 1.15897 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.279811 | 0.015 | 0.00466758 | 0.0643806 | 0.0147705 | 0.961548 | 0.167886 | 0.0308672 | NA | 0.000341783 | 0.00120136 | 148.833 | 1.77398 | 1.48897 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.191514 | 0.00312371 | 0.000765965 | 0.02832 | 0.00697027 | 0.72382 | 0.114908 | 0.143898 | 0.298921 | 6.94278e-05 | 0.00023833 | 21.3802 | 2.202 | 1.73952 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.194435 | 0.00344246 | 0.000783487 | 0.0351812 | 0.00456036 | 1.27811 | 0.116661 | 0.377201 | NA | 0.000918322 | 0.00134225 | 244.797 | 1.18305 | 1.1752 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.125807 | 0.015 | 0.00471317 | 0.0337034 | 0.00849598 | 0.532666 | 0.0754841 | 0.232487 | NA | 0.00625878 | 0.00840153 | 1058.09 | 1.00126 | 1.32659 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.154377 | 0.00577401 | 0.00132517 | 0.0449007 | 0.00756741 | 0.875938 | 0.0926265 | 0.38587 | NA | 0.00233486 | 0.0015599 | 514.582 | 0.417901 | 1.53551 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.237387 | 0.00277991 | 0.000673321 | 0.0285136 | 0.00370187 | 1.10004 | 0.142432 | 0.41885 | NA | 0.000517448 | 0.00152857 | 138.479 | 0.564194 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.112286 | 0.00254409 | 0.000820517 | 26.9787 | 2.48103 | 2.19389 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.322267 | 0.00520612 | 0.000778282 | 658.214 | 1.13692 | 1.13898 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.148468 | 0.00257839 | 0.000752038 | 49.1929 | 1.58982 | 1.4171 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.111657 | 0.00233707 | 0.000770636 | 23.2689 | 2.13986 | 1.89221 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.320181 | 0.00515934 | 0.000777465 | 651.75 | 1.12575 | 1.12779 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.161698 | 0.00251458 | 0.000746581 | 49.2338 | 1.59115 | 1.41828 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.279811 | 0.015 | 0.00466758 | 148.833 | 1.77398 | 1.48897 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.191514 | 0.00312371 | 0.000765965 | 21.3802 | 2.202 | 1.73952 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.112288 | 0.00254372 | 0.0008201 | 26.9441 | 2.47784 | 2.19107 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.322579 | 0.00520477 | 0.000778137 | 658.234 | 1.13696 | 1.13901 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.148439 | 0.0025789 | 0.000752225 | 49.2134 | 1.59048 | 1.41769 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.187375 | 0.015 | 0.00552287 | 518.916 | 1.1587 | 2.4136 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.049956 | 0.015 | 0.00336195 | 1625.21 | 1.01079 | 1.0641 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.14009 | 0.015 | 0.0052547 | 1030.14 | 0.924261 | 1.58362 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.1435 | 0.00380148 | 0.00116897 | 43.3984 | 0.374147 | 1.94522 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.174897 | 0.00895187 | 0.00139342 | 1363.98 | 0.440609 | 1.49515 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.144735 | 0.00456867 | 0.0014131 | 136.364 | 0.282693 | 1.9266 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.218041 | 0.00313714 | 0.000975087 | 74.4977 | 0.397884 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.236932 | 0.00224226 | 0.000239079 | 240.499 | 0.7751 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.257188 | 0.00296034 | 0.000805796 | 100.439 | 0.420565 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `open_loop_stress`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.242259 | 0.00246786 | 0.000561039 | 0.0318196 | 0.00353612 | 2.18507 | 0.145355 | 0.348974 | NA | 0.00031128 | 0.000206625 | 128.638 | 0.621678 | 0.617552 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.239913 | 0.00239438 | 0.000544287 | 0.0316303 | 0.0034051 | 2.22587 | 0.143948 | 0.346865 | NA | 0.000488693 | 0.000151436 | 127.047 | 0.613991 | 0.609915 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.378739 | 0.015 | 0.00370536 | 0.0778155 | 0.0157556 | 1.97414 | 0.227243 | 0.0293828 | 0.369609 | 1.7532e-05 | 2.31612e-05 | 83.5826 | 0.996245 | 0.836189 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.156754 | 0.00289831 | 0.000597061 | 0.0257205 | 0.00562709 | 0.867148 | 0.0940524 | 0.133813 | 0.340172 | -5.04605e-06 | -2.45611e-07 | 9.37528 | 0.965584 | 0.762784 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.242486 | 0.00246679 | 0.000561204 | 0.0318204 | 0.00353349 | 2.1851 | 0.145492 | 0.34893 | NA | 0.000309712 | 0.000205604 | 128.578 | 0.621389 | 0.617264 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.233648 | 0.015 | 0.00424852 | 0.0449572 | 0.00905589 | 0.984432 | 0.140189 | 0.232255 | NA | 0.00563205 | 0.00105557 | 866.63 | 0.820088 | 1.08654 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.205465 | 0.00507522 | 0.00114216 | 0.0449007 | 0.00684416 | 1.23531 | 0.123279 | 0.37537 | NA | 0.00184208 | 0.000222293 | 385.762 | 0.313285 | 1.15112 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.345589 | 0.00233651 | 0.000553198 | 0.0298547 | 0.00330751 | 2.19817 | 0.207354 | 0.387449 | 0.537815 | 0.000278835 | 0.000406338 | 118.008 | 0.480794 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.124829 | 0.00211747 | 0.000591254 | 14.2059 | 1.30641 | 1.15521 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.445036 | 0.00321713 | 0.000519478 | 343.895 | 0.594002 | 0.595078 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.156913 | 0.00206898 | 0.000572384 | 27.8135 | 0.89888 | 0.801222 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.120013 | 0.00196848 | 0.000557309 | 12.5107 | 1.15051 | 1.01736 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.44696 | 0.00320107 | 0.00051889 | 341.952 | 0.590647 | 0.591717 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.152766 | 0.00201358 | 0.000556662 | 26.6786 | 0.862203 | 0.768529 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.378739 | 0.015 | 0.00370536 | 83.5826 | 0.996245 | 0.836189 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.156754 | 0.00289831 | 0.000597061 | 9.37528 | 0.965584 | 0.762784 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.124864 | 0.00211778 | 0.000591916 | 14.2152 | 1.30726 | 1.15597 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.445288 | 0.00321368 | 0.000519423 | 343.679 | 0.593629 | 0.594705 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.157306 | 0.0020689 | 0.000572274 | 27.8402 | 0.899744 | 0.801992 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.328321 | 0.015 | 0.00462794 | 295.909 | 0.660741 | 1.37634 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0660439 | 0.015 | 0.00338669 | 1568.95 | 0.975801 | 1.02726 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.306579 | 0.015 | 0.00473092 | 735.028 | 0.659482 | 1.12995 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.161985 | 0.00353818 | 0.000982094 | 29.6522 | 0.255638 | 1.32908 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.260265 | 0.00766069 | 0.00127093 | 1043.11 | 0.336958 | 1.14342 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.194146 | 0.00402677 | 0.00117345 | 84.5257 | 0.175228 | 1.1942 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.274198 | 0.00273897 | 0.000772144 | 43.1557 | 0.23049 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.411949 | 0.00176236 | 0.000215901 | 236.987 | 0.763781 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.35062 | 0.0025082 | 0.000671548 | 73.8822 | 0.309365 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `closed_loop_small`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.154019 | 0.00487274 | 0.00108747 | 0.0384602 | 0.00610556 | 0.804199 | 0.0924117 | 0.422344 | NA | 0.00185074 | 0.00510557 | 487.895 | 2.35789 | 2.34224 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.161776 | 0.00467375 | 0.000976324 | 0.0382167 | 0.00610367 | 0.839239 | 0.0970659 | 0.406641 | NA | 0.00188071 | 0.00546662 | 474.79 | 2.29456 | 2.27933 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.193476 | 0.015 | 0.0059275 | 0.0453366 | 0.0133304 | 0.561275 | 0.116086 | 0.0294063 | NA | 0.00194336 | 0.00528022 | 392.741 | 4.6812 | 3.92912 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.152165 | 0.00390698 | 0.00142014 | 0.0265599 | 0.00677782 | 0.481201 | 0.0912991 | 0.193789 | 0.328104 | 0.000350743 | 0.000817745 | 60.4378 | 6.22464 | 4.91729 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.154177 | 0.0048708 | 0.00108709 | 0.0384603 | 0.00610602 | 0.804969 | 0.0925063 | 0.422216 | NA | 0.00181486 | 0.00521065 | 487.483 | 2.35589 | 2.34026 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.0551009 | 0.0150485 | 0.00530761 | 0.0219289 | 0.00724113 | 0.188315 | 0.0330605 | 0.288693 | NA | 0.00836681 | 0.00724311 | 1298.06 | 1.22835 | 1.62745 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.0764959 | 0.00872828 | 0.00204609 | 0.0449007 | 0.00988339 | 0.299924 | 0.0458975 | 0.513411 | NA | 0.00571011 | 0.0152367 | 1287.68 | 1.04575 | 3.84243 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.105601 | 0.0026557 | 0.000667168 | 0.0177954 | 0.00309957 | 0.459398 | 0.0633608 | 0.53513 | NA | 0.000861281 | 0.00227164 | 118.454 | 0.482611 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.0813297 | 0.00339655 | 0.00128005 | 76.717 | 7.05508 | 6.23858 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.248679 | 0.00768657 | 0.000993361 | 1281.95 | 2.21428 | 2.21829 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.132049 | 0.00353509 | 0.000988992 | 105.024 | 3.39418 | 3.02543 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.0900648 | 0.0030017 | 0.00106393 | 50.2416 | 4.62034 | 4.08562 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.24292 | 0.00769229 | 0.00099403 | 1276.41 | 2.20472 | 2.20871 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.152345 | 0.00332725 | 0.000871014 | 97.7201 | 3.15813 | 2.81502 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.193476 | 0.015 | 0.0059275 | 392.741 | 4.6812 | 3.92912 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.152165 | 0.00390698 | 0.00142014 | 60.4378 | 6.22464 | 4.91729 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.0813449 | 0.00339598 | 0.00127989 | 76.6621 | 7.05004 | 6.23411 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.249059 | 0.00768242 | 0.000992931 | 1280.73 | 2.21219 | 2.21619 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.132128 | 0.00353401 | 0.000988449 | 105.052 | 3.39509 | 3.02624 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.0697305 | 0.015 | 0.00667285 | 977.639 | 2.18299 | 4.54723 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0143901 | 0.0151456 | 0.00345674 | 1634.81 | 1.01676 | 1.07038 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.081182 | 0.015 | 0.00579325 | 1281.73 | 1.14999 | 1.97038 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.0726274 | 0.00559831 | 0.00230007 | 223.153 | 1.92385 | 10.0022 | inflated_ratio |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.0556351 | 0.0137892 | 0.00168111 | 3278.88 | 1.05918 | 3.59421 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.101225 | 0.00679733 | 0.00215709 | 360.99 | 0.748356 | 5.10017 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.155929 | 0.00404006 | 0.00131007 | 144.011 | 0.769149 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.0647245 | 0.00131766 | 0.000105695 | 132.435 | 0.426823 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.0961507 | 0.00260938 | 0.000585738 | 78.9166 | 0.330446 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `closed_loop_moderate`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.231552 | 0.0036287 | 0.000892284 | 0.035977 | 0.00484932 | 1.45922 | 0.138931 | 0.386542 | NA | 0.000757253 | 0.000819377 | 230.146 | 1.11225 | 1.10486 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.234089 | 0.00346119 | 0.000810653 | 0.0356508 | 0.00473706 | 1.48177 | 0.140453 | 0.378021 | NA | 0.000935852 | 0.00067189 | 222.336 | 1.0745 | 1.06737 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.276801 | 0.015 | 0.0050307 | 0.0565393 | 0.0143493 | 0.809022 | 0.16608 | 0.0303906 | NA | 0.000680243 | 0.000225871 | 189.092 | 2.25384 | 1.89174 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.176795 | 0.00340979 | 0.000934196 | 0.0258844 | 0.00638286 | 0.589645 | 0.106077 | 0.163914 | 0.327033 | 3.14113e-05 | 1.84535e-05 | 17.7891 | 1.83215 | 1.44735 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.231477 | 0.00362862 | 0.000891869 | 0.0359786 | 0.00485148 | 1.45899 | 0.138886 | 0.38687 | NA | 0.000758051 | 0.000818431 | 230.29 | 1.11294 | 1.10555 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.140832 | 0.0150013 | 0.00497154 | 0.0293227 | 0.00773623 | 0.740667 | 0.0844993 | 0.253193 | NA | 0.00725011 | 0.00276941 | 1106.18 | 1.04678 | 1.38688 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.16967 | 0.00662666 | 0.00170208 | 0.0449007 | 0.00801132 | 1.05893 | 0.101802 | 0.42144 | NA | 0.00321386 | 0.00135983 | 638.508 | 0.518545 | 1.90531 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.157548 | 0.00192953 | 0.000480684 | 0.0175223 | 0.0024183 | 0.798037 | 0.0945291 | 0.460619 | NA | 0.000329877 | 0.000852134 | 89.9238 | 0.36637 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.114422 | 0.00318718 | 0.00112617 | 51.3696 | 4.72407 | 4.17734 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.401363 | 0.0046347 | 0.000717009 | 573.768 | 0.991058 | 0.992854 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.17887 | 0.00306422 | 0.000833675 | 65.3009 | 2.1104 | 1.88112 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.106205 | 0.00286983 | 0.000995014 | 37.3964 | 3.43906 | 3.04105 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.393492 | 0.00465223 | 0.00071978 | 572.858 | 0.989487 | 0.991279 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.202569 | 0.0028615 | 0.000717164 | 56.7524 | 1.83413 | 1.63486 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.276801 | 0.015 | 0.0050307 | 189.092 | 2.25384 | 1.89174 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.176795 | 0.00340979 | 0.000934196 | 17.7891 | 1.83215 | 1.44735 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.114295 | 0.00318618 | 0.00112536 | 51.3009 | 4.71775 | 4.17175 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.400848 | 0.00463519 | 0.000717095 | 574.249 | 0.991888 | 0.993685 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.179289 | 0.00306448 | 0.000833156 | 65.3202 | 2.11103 | 1.88168 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.165559 | 0.015 | 0.00602785 | 645.876 | 1.44219 | 3.00412 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0530828 | 0.0150038 | 0.00341054 | 1602.05 | 0.996389 | 1.04893 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.203855 | 0.015 | 0.00547624 | 1070.62 | 0.960581 | 1.64584 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.116844 | 0.0046463 | 0.00181304 | 112.901 | 0.973346 | 5.0605 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.209479 | 0.00957029 | 0.00145264 | 1573.45 | 0.508274 | 1.72476 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.182686 | 0.00566341 | 0.00184057 | 229.174 | 0.475094 | 3.23784 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.150511 | 0.00256177 | 0.000837553 | 46.1881 | 0.246686 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.19561 | 0.00151841 | 0.000150762 | 175.815 | 0.566629 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.126525 | 0.00170841 | 0.000453736 | 47.7686 | 0.20002 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `closed_loop_stress`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.26015 | 0.00259253 | 0.000587876 | 0.0331722 | 0.00377603 | 2.60207 | 0.15609 | 0.349328 | NA | 0.0003063 | 0.000271347 | 130.557 | 0.630951 | 0.626763 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.253879 | 0.00251429 | 0.000570041 | 0.0328414 | 0.00364417 | 2.62643 | 0.152327 | 0.347953 | NA | 0.000437012 | 0.00024471 | 129.544 | 0.626058 | 0.621902 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.39285 | 0.015 | 0.00390141 | 0.074987 | 0.015556 | 1.68199 | 0.23571 | 0.030125 | 0.416296 | 0.000115481 | 1.06056e-05 | 90.4385 | 1.07796 | 0.904777 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.169332 | 0.00292439 | 0.000658198 | 0.0257286 | 0.00558057 | 0.95464 | 0.101599 | 0.134484 | 0.372637 | 2.78243e-05 | 1.35164e-06 | 10.1967 | 1.05018 | 0.829612 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.259711 | 0.00259331 | 0.000587337 | 0.0331745 | 0.00377722 | 2.60207 | 0.155827 | 0.349607 | NA | 0.000319753 | 0.000286852 | 130.632 | 0.631314 | 0.627123 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.248165 | 0.015 | 0.00447067 | 0.0435881 | 0.00881063 | 1.10893 | 0.148899 | 0.238466 | NA | 0.00602483 | 0.000283494 | 925.227 | 0.875538 | 1.16001 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.225846 | 0.00538301 | 0.00128187 | 0.0449012 | 0.00723192 | 1.74866 | 0.135508 | 0.388562 | NA | 0.00184969 | 0.000842172 | 370.385 | 0.300797 | 1.10523 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.267566 | 0.0017173 | 0.000302927 | 0.0231246 | 0.00215594 | 2.12442 | 0.160539 | 0.408441 | 0.390875 | 0.000423066 | 0.000857686 | 217.587 | 0.886501 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.146692 | 0.00225801 | 0.000615085 | 15.9423 | 1.46609 | 1.29642 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.431364 | 0.00306218 | 0.000504963 | 337.556 | 0.583055 | 0.584111 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.202394 | 0.00245742 | 0.00064358 | 38.1714 | 1.23363 | 1.0996 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.137026 | 0.00212709 | 0.000591803 | 14.4866 | 1.33223 | 1.17804 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.434385 | 0.00305709 | 0.000504185 | 338.829 | 0.585253 | 0.586313 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.190226 | 0.00235868 | 0.000614135 | 35.3167 | 1.14137 | 1.01737 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.39285 | 0.015 | 0.00390141 | 90.4385 | 1.07796 | 0.904777 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.169332 | 0.00292439 | 0.000658198 | 10.1967 | 1.05018 | 0.829612 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.146651 | 0.00225752 | 0.00061399 | 15.9864 | 1.47015 | 1.30001 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.430903 | 0.00306486 | 0.00050522 | 337.765 | 0.583414 | 0.584471 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.20158 | 0.00245756 | 0.000642802 | 38.1441 | 1.23275 | 1.09881 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.316592 | 0.015 | 0.00511829 | 394.351 | 0.880553 | 1.83422 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0880999 | 0.015 | 0.00338148 | 1554.19 | 0.966621 | 1.0176 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.339805 | 0.0150001 | 0.00491226 | 827.139 | 0.742126 | 1.27155 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.176763 | 0.00415336 | 0.00126992 | 50.2317 | 0.433059 | 2.2515 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.266319 | 0.00733501 | 0.00122091 | 939.604 | 0.303522 | 1.02996 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.234455 | 0.00466065 | 0.00135478 | 121.321 | 0.251506 | 1.71405 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.10526 | 0.00100874 | 0.000287562 | 9.65577 | 0.0515704 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.537846 | 0.00308344 | 0.000358328 | 621.592 | 2.00331 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.159591 | 0.00105972 | 0.00026289 | 21.514 | 0.0900851 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `closed_loop_cmd_lateral_small`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.213473 | 0.00351713 | 0.00081031 | 0.0353274 | 0.00465001 | 1.41557 | 0.128084 | 0.39406 | NA | 0.000886346 | 0.00184382 | 241.799 | 1.16856 | 1.16081 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.245594 | 0.00317635 | 0.000636608 | 0.0348276 | 0.0044145 | 1.52217 | 0.147356 | 0.368865 | NA | 0.000937558 | 0.00111225 | 223.596 | 1.08059 | 1.07341 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.256054 | 0.015 | 0.00580875 | 0.0499885 | 0.0133053 | 0.712459 | 0.153632 | 0.0290938 | NA | 0.00198857 | 0.00323183 | 355.409 | 4.23622 | 3.55563 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.209062 | 0.00372033 | 0.00118163 | 0.0296031 | 0.00684745 | 0.577031 | 0.125437 | 0.18125 | 0.292618 | 0.000121584 | 0.000232363 | 39.2333 | 4.04074 | 3.19207 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.213407 | 0.00351663 | 0.00081 | 0.0353263 | 0.00464911 | 1.41522 | 0.128044 | 0.393719 | NA | 0.000873981 | 0.00186951 | 241.69 | 1.16803 | 1.16028 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.0730395 | 0.0150035 | 0.00519296 | 0.0232872 | 0.00726884 | 0.298214 | 0.0438237 | 0.26363 | NA | 0.00811098 | 0.00535431 | 1218.15 | 1.15273 | 1.52726 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.11677 | 0.00731197 | 0.0017999 | 0.0449007 | 0.00849522 | 0.606989 | 0.070062 | 0.44119 | NA | 0.00420816 | 0.00585588 | 856.501 | 0.695581 | 2.5558 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.185911 | 0.0027354 | 0.000739871 | 0.0229444 | 0.00340225 | 0.808837 | 0.111546 | 0.488708 | NA | 0.000787246 | 0.00159069 | 125.616 | 0.511787 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.14267 | 0.00270673 | 0.000916713 | 45.2605 | 4.16226 | 3.68055 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.334027 | 0.00509763 | 0.000764194 | 626.283 | 1.08177 | 1.08373 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.163722 | 0.00274701 | 0.000750022 | 53.8544 | 1.74047 | 1.55138 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.198832 | 0.00206148 | 0.000520456 | 15.606 | 1.43517 | 1.26907 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.338969 | 0.00497952 | 0.000757687 | 613.491 | 1.05967 | 1.06159 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.198981 | 0.00248805 | 0.00063168 | 41.6894 | 1.34732 | 1.20094 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.256054 | 0.015 | 0.00580875 | 355.409 | 4.23622 | 3.55563 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.209062 | 0.00372033 | 0.00118163 | 39.2333 | 4.04074 | 3.19207 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.142745 | 0.0027058 | 0.000915969 | 45.221 | 4.15863 | 3.67734 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.333742 | 0.00509699 | 0.000764167 | 625.943 | 1.08118 | 1.08314 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.163733 | 0.0027471 | 0.000749863 | 53.9046 | 1.7421 | 1.55283 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.0972719 | 0.015 | 0.00642029 | 825.523 | 1.84333 | 3.83971 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0176559 | 0.0150104 | 0.0034424 | 1619.53 | 1.00726 | 1.06037 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.104191 | 0.015 | 0.00571619 | 1209.39 | 1.08509 | 1.85917 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.0944913 | 0.00458573 | 0.00188282 | 130.408 | 1.12428 | 5.8452 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.121974 | 0.0115205 | 0.00156542 | 2170.28 | 0.701069 | 2.37899 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.133844 | 0.00582965 | 0.00195144 | 268.817 | 0.557276 | 3.79792 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.223008 | 0.00417108 | 0.0013862 | 143.758 | 0.767798 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.132253 | 0.0014034 | 0.000137892 | 140.112 | 0.451565 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.202471 | 0.00263172 | 0.000695519 | 92.9765 | 0.389318 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `closed_loop_cmd_lateral_moderate`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.215635 | 0.00294546 | 0.000676554 | 0.0331536 | 0.00419281 | 1.50834 | 0.129381 | 0.362685 | NA | 0.000475129 | 0.0030185 | 183.8 | 0.888266 | 0.88237 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.239477 | 0.00259253 | 0.000508528 | 0.0323805 | 0.00388035 | 1.65782 | 0.143686 | 0.341552 | NA | 0.000514147 | 0.00251816 | 168.806 | 0.8158 | 0.810385 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.253357 | 0.015 | 0.00531375 | 0.0543247 | 0.0141654 | 0.725619 | 0.152014 | 0.0299687 | NA | 0.0010069 | 0.00158965 | 237.054 | 2.82552 | 2.37157 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.173273 | 0.00362685 | 0.00109085 | 0.0260828 | 0.00674705 | 0.512625 | 0.103964 | 0.174555 | 0.345242 | 0.000118892 | 0.000294933 | 26.1764 | 2.69598 | 2.12974 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.21558 | 0.00294542 | 0.000676019 | 0.033157 | 0.00419541 | 1.50847 | 0.129348 | 0.362581 | NA | 0.000459827 | 0.00301157 | 183.649 | 0.887533 | 0.881642 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.115563 | 0.0150325 | 0.00497959 | 0.0283782 | 0.00771783 | 0.501335 | 0.0693381 | 0.279318 | NA | 0.00751722 | 0.00521914 | 1123.81 | 1.06346 | 1.40898 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.135937 | 0.00758841 | 0.00173503 | 0.0449007 | 0.00889539 | 0.625607 | 0.0815622 | 0.433536 | NA | 0.00444717 | 0.00885538 | 1044.04 | 0.847882 | 3.11541 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.203217 | 0.00263256 | 0.000644665 | 0.0243252 | 0.0033329 | 0.997159 | 0.12193 | 0.467156 | 0.552857 | 0.000855204 | 0.00256791 | 204.017 | 0.83121 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.142099 | 0.00243124 | 0.000811371 | 33.5638 | 3.08661 | 2.72938 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.330572 | 0.00415079 | 0.000639484 | 481.18 | 0.831133 | 0.832638 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.174236 | 0.00225435 | 0.000578807 | 36.6575 | 1.1847 | 1.05599 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.171587 | 0.00183699 | 0.000461334 | 12.9447 | 1.19043 | 1.05266 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.348659 | 0.00401724 | 0.000618857 | 467.468 | 0.807449 | 0.808911 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.198186 | 0.00192336 | 0.000445394 | 26.0041 | 0.840404 | 0.749099 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.253357 | 0.015 | 0.00531375 | 237.054 | 2.82552 | 2.37157 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.173273 | 0.00362685 | 0.00109085 | 26.1764 | 2.69598 | 2.12974 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.142114 | 0.00243072 | 0.000810629 | 33.5343 | 3.0839 | 2.72699 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.330445 | 0.00414971 | 0.000639574 | 480.6 | 0.830131 | 0.831635 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.174181 | 0.00225584 | 0.000577853 | 36.8117 | 1.18968 | 1.06043 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.166334 | 0.015 | 0.00593765 | 620.357 | 1.38521 | 2.88543 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.027858 | 0.0150974 | 0.00345411 | 1635.24 | 1.01703 | 1.07067 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.152498 | 0.015 | 0.00554702 | 1115.83 | 1.00114 | 1.71534 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.128884 | 0.00427121 | 0.00167304 | 96.6747 | 0.833454 | 4.33319 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.123046 | 0.0128141 | 0.00164171 | 2785.68 | 0.899863 | 3.05357 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.155882 | 0.00567997 | 0.00189034 | 249.758 | 0.517765 | 3.52865 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.198131 | 0.00291793 | 0.000994696 | 71.8418 | 0.383699 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.175467 | 0.00232486 | 0.00022328 | 420.788 | 1.35615 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.236054 | 0.0026549 | 0.000716018 | 119.42 | 0.500042 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `closed_loop_cmd_lateral_stress`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.285787 | 0.00209308 | 0.000466389 | 0.0290546 | 0.00319752 | 2.42488 | 0.171472 | 0.338279 | 0.4925 | 0.000141402 | 0.000523338 | 113.942 | 0.550658 | 0.547003 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.295537 | 0.00191435 | 0.000418005 | 0.0285522 | 0.00291611 | 2.5514 | 0.177322 | 0.331875 | NA | 0.000202955 | 0.000414047 | 110.463 | 0.533844 | 0.5303 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.416752 | 0.015 | 0.00388983 | 0.0747732 | 0.0156921 | 1.81879 | 0.250051 | 0.0299297 | 0.403665 | 7.7069e-05 | 2.80195e-05 | 94.3439 | 1.12451 | 0.943849 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.198485 | 0.00299811 | 0.000634154 | 0.0257205 | 0.0058528 | 0.950507 | 0.119091 | 0.135937 | 0.34929 | -4.40566e-06 | 4.03409e-06 | 11.2009 | 1.15361 | 0.911316 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.285713 | 0.00209453 | 0.000466489 | 0.0290611 | 0.00319879 | 2.4223 | 0.171428 | 0.338521 | 0.5 | 0.000145614 | 0.000540655 | 114.068 | 0.551265 | 0.547606 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.245581 | 0.015 | 0.00440961 | 0.0432405 | 0.00880045 | 1.08193 | 0.147349 | 0.232237 | NA | 0.00589808 | 0.00112773 | 917.316 | 0.868052 | 1.15009 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.234274 | 0.00532376 | 0.00124893 | 0.0449007 | 0.00718828 | 1.54416 | 0.140564 | 0.363755 | NA | 0.00178182 | 0.000567091 | 410.852 | 0.333661 | 1.22599 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.015267, 0.0718666, 0.718049 | 0.439877 | 0.00244008 | 0.000517651 | 0.0327264 | 0.00335446 | 3.21623 | 0.263926 | 0.402211 | 0.461431 | 0.000432634 | 0.000921354 | 253.774 | 1.03393 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.169118 | 0.00189774 | 0.000537412 | 14.151 | 1.30136 | 1.15075 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.478603 | 0.00274354 | 0.000421922 | 305.5 | 0.527684 | 0.52864 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.20964 | 0.00163796 | 0.000439834 | 22.176 | 0.716688 | 0.638824 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.176994 | 0.00155677 | 0.000440071 | 10.8753 | 1.00012 | 0.884373 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.479024 | 0.00271929 | 0.000421077 | 298.976 | 0.516416 | 0.517351 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.230593 | 0.00146698 | 0.000392869 | 21.5379 | 0.696064 | 0.62044 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.416752 | 0.015 | 0.00388983 | 94.3439 | 1.12451 | 0.943849 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.198485 | 0.00299811 | 0.000634154 | 11.2009 | 1.15361 | 0.911316 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.169658 | 0.00189604 | 0.00053613 | 14.0766 | 1.29452 | 1.1447 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.478151 | 0.00274664 | 0.000422322 | 305.825 | 0.528246 | 0.529202 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.209331 | 0.00164089 | 0.000441015 | 22.3026 | 0.720777 | 0.642469 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.361433 | 0.015 | 0.004853 | 340.912 | 0.761229 | 1.58566 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0408947 | 0.015 | 0.00340673 | 1582.12 | 0.983993 | 1.03588 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.334417 | 0.015 | 0.00496912 | 828.915 | 0.743719 | 1.27428 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.197041 | 0.00385354 | 0.00111253 | 39.1166 | 0.337233 | 1.7533 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.280914 | 0.00757218 | 0.00126884 | 1075.31 | 0.347359 | 1.17872 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.224866 | 0.00454555 | 0.0013654 | 118.132 | 0.244896 | 1.66901 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.350258 | 0.00200912 | 0.000581175 | 48.993 | 0.261667 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.562955 | 0.00292784 | 0.000353512 | 596.174 | 1.92139 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.015267, 0.0718666, 0.718049 | 0.406419 | 0.00238328 | 0.000618266 | 116.155 | 0.486372 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
