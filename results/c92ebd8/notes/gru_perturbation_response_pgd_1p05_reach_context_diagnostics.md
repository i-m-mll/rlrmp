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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.151412 | 0.00414963 | 0.000910553 | 0.0370126 | 0.00539334 | 0.922814 | 0.0908472 | 0.39457 | NA | 0.00138261 | 0.0034417 | 372.338 | 1.798 | 1.78748 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.149044 | 0.00411024 | 0.000906205 | 0.0368742 | 0.00531827 | 0.928824 | 0.0894264 | 0.394016 | NA | 0.00159009 | 0.00368447 | 371.431 | 1.79362 | 1.78313 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.207473 | 0.015 | 0.0053595 | 0.0521488 | 0.0140679 | 0.663431 | 0.124484 | 0.0306875 | NA | 0.00102725 | 0.00357254 | 265.521 | 3.16481 | 2.65636 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.183426 | 0.00331958 | 0.000992216 | 0.0305368 | 0.00745452 | 0.63549 | 0.110056 | 0.155008 | 0.403184 | 0.000178921 | 0.000555716 | 31.3596 | 3.22981 | 2.55146 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.151458 | 0.0041491 | 0.000910473 | 0.0370148 | 0.00539343 | 0.922924 | 0.0908748 | 0.39451 | NA | 0.00137437 | 0.00334305 | 372.261 | 1.79763 | 1.78711 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.088294 | 0.015 | 0.00487578 | 0.0311769 | 0.00862187 | 0.403266 | 0.0529764 | 0.232135 | NA | 0.00621012 | 0.0145761 | 1183.7 | 1.12014 | 1.48407 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.13038 | 0.0061011 | 0.00142622 | 0.0449007 | 0.00801481 | 0.709597 | 0.0782278 | 0.391953 | NA | 0.00248918 | 0.0039692 | 589.36 | 0.471462 | 1.75866 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152664, 0.0733525, 0.731248 | 0.163476 | 0.00291334 | 0.000674677 | 0.0253941 | 0.00384278 | 0.808302 | 0.0980854 | 0.444228 | NA | 0.00054667 | 0.00163163 | 147.871 | 0.574229 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.0946387 | 0.00275578 | 0.00092233 | 37.1424 | 3.41569 | 3.0204 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.242522 | 0.00664552 | 0.000923722 | 1009.49 | 1.74218 | 1.74682 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.117075 | 0.00304757 | 0.000885607 | 70.3862 | 2.27472 | 2.02761 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.090797 | 0.00265816 | 0.00090404 | 35.9949 | 3.31017 | 2.92709 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.239882 | 0.00663788 | 0.000922776 | 1008.12 | 1.73982 | 1.74445 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.116453 | 0.00303468 | 0.000891799 | 70.1815 | 2.2681 | 2.02171 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.207473 | 0.015 | 0.0053595 | 265.521 | 3.16481 | 2.65636 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.183426 | 0.00331958 | 0.000992216 | 31.3596 | 3.22981 | 2.55146 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.0946486 | 0.00275568 | 0.000922115 | 37.1396 | 3.41543 | 3.02017 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.242652 | 0.00664375 | 0.000923577 | 1009.21 | 1.74171 | 1.74634 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.117073 | 0.00304788 | 0.000885726 | 70.4348 | 2.27629 | 2.02901 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.118194 | 0.015 | 0.00590377 | 694.344 | 1.55054 | 3.22956 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0588315 | 0.015 | 0.00334654 | 1677.61 | 1.04337 | 1.09841 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.0878566 | 0.015 | 0.00537702 | 1179.14 | 1.05798 | 1.81267 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.113553 | 0.00397914 | 0.00130579 | 60.5004 | 0.51096 | 2.71177 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.171004 | 0.00940055 | 0.00143839 | 1530.63 | 0.487955 | 1.67782 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.106582 | 0.0049236 | 0.00153448 | 176.953 | 0.35749 | 2.50004 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.182485 | 0.00341647 | 0.00105869 | 101.216 | 0.515686 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.14953 | 0.00250322 | 0.000225705 | 249.88 | 0.770444 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.158411 | 0.00282033 | 0.000739632 | 92.5186 | 0.367236 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.19434 | 0.00344287 | 0.000783612 | 0.0351807 | 0.00456031 | 1.27803 | 0.116604 | 0.37756 | NA | 0.000927556 | 0.00142526 | 244.795 | 1.1821 | 1.17519 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.197845 | 0.003337 | 0.000764894 | 0.0349544 | 0.00437835 | 1.30749 | 0.118707 | 0.371099 | NA | 0.00113306 | 0.00107079 | 241.417 | 1.16579 | 1.15897 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.279811 | 0.015 | 0.00466758 | 0.0643806 | 0.0147705 | 0.961548 | 0.167886 | 0.0308672 | NA | 0.000341783 | 0.00120136 | 148.833 | 1.77398 | 1.48897 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.191514 | 0.00312371 | 0.000765965 | 0.02832 | 0.00697027 | 0.72382 | 0.114908 | 0.143898 | 0.298921 | 6.94278e-05 | 0.00023833 | 21.3802 | 2.202 | 1.73952 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.194435 | 0.00344246 | 0.000783487 | 0.0351812 | 0.00456036 | 1.27811 | 0.116661 | 0.377201 | NA | 0.000918322 | 0.00134225 | 244.797 | 1.18211 | 1.1752 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.125807 | 0.015 | 0.00471317 | 0.0337034 | 0.00849598 | 0.532666 | 0.0754841 | 0.232487 | NA | 0.00625878 | 0.00840153 | 1058.09 | 1.00128 | 1.32659 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.154377 | 0.00577401 | 0.00132517 | 0.0449007 | 0.00756741 | 0.875938 | 0.0926265 | 0.38587 | NA | 0.00233486 | 0.0015599 | 514.582 | 0.411642 | 1.53551 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152664, 0.0733525, 0.731248 | 0.241208 | 0.00282305 | 0.00068348 | 0.0289615 | 0.00375916 | 1.11743 | 0.144725 | 0.418791 | NA | 0.000536333 | 0.00156993 | 143.58 | 0.557565 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.221239 | 0.00318262 | 0.000989178 | 76.9651 | 0.39213 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.241214 | 0.00228105 | 0.000243195 | 249.838 | 0.770314 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.261172 | 0.0030055 | 0.000818066 | 103.938 | 0.412562 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.242259 | 0.00246786 | 0.000561039 | 0.0318196 | 0.00353612 | 2.18507 | 0.145355 | 0.348974 | NA | 0.00031128 | 0.000206625 | 128.638 | 0.621185 | 0.617552 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.239913 | 0.00239438 | 0.000544287 | 0.0316303 | 0.0034051 | 2.22587 | 0.143948 | 0.346865 | NA | 0.000488693 | 0.000151436 | 127.047 | 0.613503 | 0.609915 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.378739 | 0.015 | 0.00370536 | 0.0778155 | 0.0157556 | 1.97414 | 0.227243 | 0.0293828 | 0.369609 | 1.7532e-05 | 2.31612e-05 | 83.5826 | 0.996245 | 0.836189 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.156754 | 0.00289831 | 0.000597061 | 0.0257205 | 0.00562709 | 0.867148 | 0.0940524 | 0.133813 | 0.340172 | -5.04605e-06 | -2.45611e-07 | 9.37528 | 0.965584 | 0.762784 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.242486 | 0.00246679 | 0.000561204 | 0.0318204 | 0.00353349 | 2.1851 | 0.145492 | 0.34893 | NA | 0.000309712 | 0.000205604 | 128.578 | 0.620896 | 0.617264 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.233648 | 0.015 | 0.00424852 | 0.0449572 | 0.00905589 | 0.984432 | 0.140189 | 0.232255 | NA | 0.00563205 | 0.00105557 | 866.63 | 0.820099 | 1.08654 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.205465 | 0.00507522 | 0.00114216 | 0.0449007 | 0.00684416 | 1.23531 | 0.123279 | 0.37537 | NA | 0.00184208 | 0.000222293 | 385.762 | 0.308593 | 1.15112 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152664, 0.0733525, 0.731248 | 0.350859 | 0.00236857 | 0.000560562 | 0.0302851 | 0.00335404 | 2.23271 | 0.210515 | 0.387378 | 0.537815 | 0.000287378 | 0.000418888 | 121.887 | 0.473324 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.124829 | 0.00211747 | 0.000591254 | 14.2059 | 1.3064 | 1.15521 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.445036 | 0.00321713 | 0.000519478 | 343.895 | 0.593497 | 0.595078 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.156913 | 0.00206898 | 0.000572384 | 27.8135 | 0.898868 | 0.801222 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.120013 | 0.00196848 | 0.000557309 | 12.5107 | 1.1505 | 1.01736 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.44696 | 0.00320107 | 0.00051889 | 341.952 | 0.590146 | 0.591717 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.152766 | 0.00201358 | 0.000556662 | 26.6786 | 0.862191 | 0.768529 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.378739 | 0.015 | 0.00370536 | 83.5826 | 0.996245 | 0.836189 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.156754 | 0.00289831 | 0.000597061 | 9.37528 | 0.965584 | 0.762784 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.124864 | 0.00211778 | 0.000591916 | 14.2152 | 1.30726 | 1.15597 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.445288 | 0.00321368 | 0.000519423 | 343.679 | 0.593125 | 0.594705 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.157306 | 0.0020689 | 0.000572274 | 27.8402 | 0.899731 | 0.801992 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.328321 | 0.015 | 0.00462794 | 295.909 | 0.660793 | 1.37634 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0660439 | 0.015 | 0.00338669 | 1568.95 | 0.975788 | 1.02726 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.306579 | 0.015 | 0.00473092 | 735.028 | 0.659501 | 1.12995 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.161985 | 0.00353818 | 0.000982094 | 29.6522 | 0.250429 | 1.32908 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.260265 | 0.00766069 | 0.00127093 | 1043.11 | 0.332537 | 1.14342 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.194146 | 0.00402677 | 0.00117345 | 84.5257 | 0.170764 | 1.1942 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.277928 | 0.00277579 | 0.000782486 | 44.3964 | 0.226196 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.419146 | 0.00178926 | 0.000219196 | 245.282 | 0.756268 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.355504 | 0.00254066 | 0.000680005 | 75.9825 | 0.301599 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `small`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.196241 | 0.00399264 | 0.000768884 | 0.0371553 | 0.00566373 | 1.04804 | 0.117744 | 0.383055 | 0.561141 | 0.00167786 | 0.00242964 | 365.654 | 1.76572 | 1.7554 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.199661 | 0.00391659 | 0.000751104 | 0.0369671 | 0.00548834 | 1.06554 | 0.119796 | 0.380099 | 0.561141 | 0.00173301 | 0.00256559 | 363.098 | 1.75338 | 1.74312 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.221374 | 0.015 | 0.00508503 | 0.0533787 | 0.0149593 | 0.657802 | 0.132825 | 0.0307031 | 0.472422 | 0.0010687 | 0.00219184 | 265.819 | 3.16838 | 2.65935 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.138177 | 0.00348471 | 0.000958942 | 0.0257283 | 0.00681565 | 0.473634 | 0.0829064 | 0.168852 | 0.441582 | 4.92503e-05 | 0.000137097 | 23.311 | 2.40086 | 1.89661 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.196457 | 0.00399188 | 0.000768744 | 0.0371571 | 0.00566492 | 1.04831 | 0.117874 | 0.382911 | 0.558769 | 0.00168669 | 0.00247954 | 365.807 | 1.76646 | 1.75613 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.169968 | 0.015 | 0.00441415 | 0.0439211 | 0.0100569 | 0.655072 | 0.101981 | 0.23243 | NA | 0.00589507 | 0.00902163 | 1098.32 | 1.03935 | 1.37702 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.178843 | 0.00596511 | 0.00125331 | 0.0449007 | 0.00848217 | 0.846911 | 0.107306 | 0.384672 | 0.50392 | 0.00282779 | 0.00217958 | 571.603 | 0.457257 | 1.70567 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152664, 0.0733525, 0.731248 | 0.235595 | 0.00302069 | 0.000597684 | 0.0284624 | 0.00438157 | 0.957846 | 0.141357 | 0.429743 | 0.48748 | 0.00088566 | 0.000738342 | 166.785 | 0.647679 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.130619 | 0.00262544 | 0.000717425 | 31.7627 | 2.92096 | 2.58292 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.273725 | 0.00645846 | 0.000904073 | 998.847 | 1.72382 | 1.72841 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.184378 | 0.00289403 | 0.000685155 | 66.3533 | 2.14439 | 1.91144 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.132666 | 0.00243623 | 0.000679914 | 26.7029 | 2.45565 | 2.17146 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.269523 | 0.00648682 | 0.000907463 | 999.969 | 1.72576 | 1.73035 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.196792 | 0.00282671 | 0.000665935 | 62.6227 | 2.02382 | 1.80397 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.221374 | 0.015 | 0.00508503 | 265.819 | 3.16838 | 2.65935 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.138177 | 0.00348471 | 0.000958942 | 23.311 | 2.40086 | 1.89661 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.130679 | 0.00262529 | 0.000717203 | 31.8324 | 2.92738 | 2.58859 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.273972 | 0.00645573 | 0.00090382 | 999.062 | 1.72419 | 1.72879 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.18472 | 0.00289463 | 0.000685208 | 66.5266 | 2.14999 | 1.91643 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.202572 | 0.015 | 0.0051446 | 553.591 | 1.23622 | 2.57488 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0868804 | 0.015 | 0.00333285 | 1688.7 | 1.05026 | 1.10566 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.220453 | 0.015 | 0.00476499 | 1052.67 | 0.944503 | 1.61825 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.151233 | 0.00400812 | 0.00110142 | 64.6103 | 0.54567 | 2.89598 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.193563 | 0.00920577 | 0.00142174 | 1492.23 | 0.475715 | 1.63574 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.191734 | 0.00468145 | 0.00123676 | 157.967 | 0.319133 | 2.2318 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.244 | 0.00301138 | 0.000751882 | 64.8485 | 0.330398 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.193766 | 0.00264299 | 0.000241245 | 289.299 | 0.891986 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.269019 | 0.00340771 | 0.000799926 | 146.208 | 0.580349 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `moderate`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.221637 | 0.00358904 | 0.000673156 | 0.0360437 | 0.00536264 | 1.20593 | 0.132982 | 0.368438 | 0.492573 | 0.00132607 | 0.00309051 | 310.81 | 1.50088 | 1.4921 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.227689 | 0.0034822 | 0.000664912 | 0.0358105 | 0.00511794 | 1.24774 | 0.136613 | 0.36199 | 0.475234 | 0.00132732 | 0.00355014 | 309.688 | 1.49547 | 1.48672 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.287034 | 0.015 | 0.00471216 | 0.0621488 | 0.0153173 | 0.88835 | 0.17222 | 0.0292266 | 0.395925 | 0.000798126 | 0.00103548 | 208.591 | 2.48625 | 2.08681 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.174379 | 0.00329273 | 0.000761856 | 0.0258117 | 0.00668821 | 0.607735 | 0.104627 | 0.15432 | 0.386645 | 3.57964e-05 | 2.55466e-05 | 16.4719 | 1.69648 | 1.34017 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.221757 | 0.00358774 | 0.000672442 | 0.0360433 | 0.00536277 | 1.20595 | 0.133054 | 0.368391 | 0.491423 | 0.00133097 | 0.00310025 | 310.699 | 1.50035 | 1.49157 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.205073 | 0.015 | 0.00424929 | 0.0498956 | 0.0106581 | 0.811528 | 0.123044 | 0.232482 | 0.445556 | 0.00534851 | 0.0113748 | 1104.73 | 1.04542 | 1.38506 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.203131 | 0.00571606 | 0.00113993 | 0.0449059 | 0.00843256 | 0.968063 | 0.121879 | 0.373648 | 0.4954 | 0.00256006 | 0.00316971 | 536.126 | 0.428877 | 1.5998 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152664, 0.0733525, 0.731248 | 0.303182 | 0.00322774 | 0.000608083 | 0.0328979 | 0.00486844 | 1.23155 | 0.181909 | 0.418932 | 0.519578 | 0.000934102 | 0.000755597 | 194.645 | 0.755865 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.149818 | 0.00240316 | 0.000611072 | 26.8062 | 2.46515 | 2.17986 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.303927 | 0.00574363 | 0.000827699 | 851.701 | 1.46988 | 1.47379 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.211166 | 0.00262034 | 0.000580699 | 53.9217 | 1.74262 | 1.55332 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.149697 | 0.00214416 | 0.00059498 | 22.1201 | 2.03421 | 1.79879 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.304485 | 0.00574945 | 0.00082783 | 856.009 | 1.47731 | 1.48125 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.228885 | 0.00255298 | 0.000571927 | 50.935 | 1.6461 | 1.46728 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.287034 | 0.015 | 0.00471216 | 208.591 | 2.48625 | 2.08681 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.174379 | 0.00329273 | 0.000761856 | 16.4719 | 1.69648 | 1.34017 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.149924 | 0.00240288 | 0.000609979 | 26.7857 | 2.46327 | 2.1782 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.304363 | 0.00573988 | 0.000827193 | 851.242 | 1.46908 | 1.473 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.210985 | 0.00262047 | 0.000580154 | 54.0697 | 1.74741 | 1.55758 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.240562 | 0.015 | 0.00490011 | 515.251 | 1.15061 | 2.39656 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0960396 | 0.015 | 0.00330784 | 1780.29 | 1.10723 | 1.16564 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.278616 | 0.015 | 0.00453992 | 1018.65 | 0.913981 | 1.56596 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.171024 | 0.003812 | 0.000962535 | 53.7419 | 0.45388 | 2.40884 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.197092 | 0.00887234 | 0.00139136 | 1414.88 | 0.451055 | 1.55094 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.241276 | 0.00446384 | 0.00106589 | 139.757 | 0.282346 | 1.97453 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.318566 | 0.00334592 | 0.000795879 | 83.1044 | 0.42341 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.228706 | 0.00290315 | 0.000277362 | 350.021 | 1.07921 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.362275 | 0.00343416 | 0.000751007 | 150.809 | 0.598611 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `stress`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.235982 | 0.00258663 | 0.00046561 | 0.0322588 | 0.00388907 | 1.8667 | 0.141589 | 0.348589 | 0.465495 | 0.000890135 | 0.000314278 | 156.663 | 0.756519 | 0.752094 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.234026 | 0.0025577 | 0.000459345 | 0.0321309 | 0.00381606 | 1.8839 | 0.140416 | 0.349562 | 0.475533 | 0.000913246 | 0.000318702 | 155.822 | 0.752455 | 0.748054 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.413361 | 0.015 | 0.00338974 | 0.0851651 | 0.016486 | 2.06105 | 0.248017 | 0.0315156 | 0.319977 | 3.83584e-06 | 1.26601e-05 | 90.1698 | 1.07476 | 0.90209 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.162851 | 0.00289318 | 0.000579913 | 0.0257323 | 0.00575944 | 0.867811 | 0.0977104 | 0.133039 | 0.318063 | 6.87543e-06 | 1.89656e-05 | 10.3999 | 1.07111 | 0.846149 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.236215 | 0.0025858 | 0.000465362 | 0.0322586 | 0.00388712 | 1.86865 | 0.141729 | 0.348648 | 0.463015 | 0.000906299 | 0.000303638 | 156.752 | 0.756947 | 0.75252 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.344926 | 0.015 | 0.00362824 | 0.0637846 | 0.0110011 | 1.58433 | 0.206956 | 0.232766 | 0.417843 | 0.0050723 | 0.00213581 | 845.163 | 0.799785 | 1.05963 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.224061 | 0.00503202 | 0.000954337 | 0.0449014 | 0.00732062 | 1.37846 | 0.134437 | 0.360495 | 0.444165 | 0.00234873 | 0.000572444 | 402.275 | 0.321802 | 1.20039 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152664, 0.0733525, 0.731248 | 0.424942 | 0.00279257 | 0.00050442 | 0.0372358 | 0.00443173 | 2.46049 | 0.254965 | 0.382414 | 0.450519 | 0.000649156 | 0.000779364 | 157.289 | 0.6108 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.152489 | 0.00193238 | 0.000394364 | 11.5882 | 1.06567 | 0.942343 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.358068 | 0.00391731 | 0.000623261 | 431.173 | 0.744124 | 0.746106 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.197388 | 0.00191019 | 0.000379205 | 27.2288 | 0.879973 | 0.78438 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.149823 | 0.00185047 | 0.000377195 | 10.8528 | 0.998045 | 0.882542 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.352694 | 0.0039301 | 0.000627992 | 429.074 | 0.740501 | 0.742473 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.199562 | 0.00189253 | 0.000372849 | 27.5391 | 0.89 | 0.793318 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.413361 | 0.015 | 0.00338974 | 90.1698 | 1.07476 | 0.90209 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.162851 | 0.00289318 | 0.000579913 | 10.3999 | 1.07111 | 0.846149 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.152637 | 0.00193205 | 0.000393897 | 11.582 | 1.0651 | 0.941837 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.358324 | 0.00391599 | 0.000622994 | 431.442 | 0.744589 | 0.746572 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.197685 | 0.00190935 | 0.000379194 | 27.2322 | 0.88008 | 0.784475 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.472374 | 0.015 | 0.00370376 | 252.78 | 0.564483 | 1.17574 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.080063 | 0.015 | 0.00335148 | 1573.73 | 0.978759 | 1.03039 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.482342 | 0.015 | 0.00382947 | 708.98 | 0.636129 | 1.0899 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.209236 | 0.00339643 | 0.000706462 | 26.1481 | 0.220835 | 1.17202 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.212267 | 0.0078679 | 0.00129406 | 1101.91 | 0.351281 | 1.20787 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.250681 | 0.00383174 | 0.000862494 | 78.7699 | 0.159136 | 1.11288 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.382457 | 0.00283212 | 0.000569835 | 49.8427 | 0.253944 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.369086 | 0.00233615 | 0.000278964 | 285.14 | 0.879162 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152664, 0.0733525, 0.731248 | 0.523284 | 0.00320944 | 0.000664461 | 136.884 | 0.543335 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
