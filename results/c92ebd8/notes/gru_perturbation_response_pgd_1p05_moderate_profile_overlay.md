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
