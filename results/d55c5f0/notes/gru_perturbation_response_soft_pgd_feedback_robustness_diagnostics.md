# GRU perturbation-response bank

Issue: `d55c5f0`. Source experiment: `d55c5f0`.

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

### `soft_pgd_ofb1p05`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.195481 | 0.00343574 | 0.00078592 | 0.0351318 | 0.00453606 | 1.28213 | 0.117288 | 0.381237 | NA | 0.000906344 | 0.0013085 | 242.564 | 1.17133 | 1.16447 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.199274 | 0.00334171 | 0.000768591 | 0.0349409 | 0.00438764 | 1.30868 | 0.119564 | 0.374167 | NA | 0.00109927 | 0.000771253 | 241.125 | 1.16438 | 1.15757 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.278601 | 0.015 | 0.00469303 | 0.0639582 | 0.0146843 | 0.95911 | 0.16716 | 0.0296797 | NA | 0.000362539 | 0.000958684 | 148.37 | 1.76846 | 1.48434 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.191882 | 0.00311596 | 0.000757523 | 0.0281604 | 0.00695051 | 0.728869 | 0.115129 | 0.14557 | 0.296391 | 6.46901e-05 | 0.000197413 | 20.794 | 2.14163 | 1.69183 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.19557 | 0.00343518 | 0.000785892 | 0.0351328 | 0.00453638 | 1.28218 | 0.117342 | 0.381057 | NA | 0.000898039 | 0.00127489 | 242.585 | 1.17143 | 1.16458 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.124688 | 0.015 | 0.00472893 | 0.0332877 | 0.00840014 | 0.52393 | 0.0748126 | 0.232198 | NA | 0.00631507 | 0.00747482 | 1054.65 | 0.99802 | 1.32227 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.155258 | 0.00576553 | 0.00132497 | 0.0449007 | 0.007543 | 0.883442 | 0.0931548 | 0.385672 | NA | 0.00230956 | 0.00155849 | 510.881 | 0.408682 | 1.52447 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152671, 0.0717183, 0.716459 | 0.240151 | 0.00276481 | 0.000670783 | 0.0285144 | 0.00367774 | 1.10408 | 0.14409 | 0.416828 | NA | 0.000507852 | 0.0013942 | 137.381 | 0.554803 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.109755 | 0.00254884 | 0.000828525 | 27.1318 | 2.49509 | 2.20634 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.327129 | 0.0051888 | 0.000775832 | 651.899 | 1.12506 | 1.12805 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.149558 | 0.00256958 | 0.000753404 | 48.6599 | 1.57258 | 1.40174 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.108732 | 0.00236579 | 0.000785677 | 24.2632 | 2.23129 | 1.97307 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.325806 | 0.00514798 | 0.000772831 | 650.105 | 1.12196 | 1.12495 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.163283 | 0.00251136 | 0.000747264 | 49.0061 | 1.58376 | 1.41172 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.278601 | 0.015 | 0.00469303 | 148.37 | 1.76846 | 1.48434 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.191882 | 0.00311596 | 0.000757523 | 20.794 | 2.14163 | 1.69183 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.109758 | 0.00254882 | 0.000828314 | 27.116 | 2.49365 | 2.20506 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.327356 | 0.00518653 | 0.00077571 | 651.929 | 1.12511 | 1.1281 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.149596 | 0.00257018 | 0.000753653 | 48.7108 | 1.57422 | 1.40321 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.183147 | 0.015 | 0.00555127 | 521.359 | 1.16424 | 2.42496 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0510714 | 0.015 | 0.00336491 | 1617.26 | 1.00583 | 1.05889 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.139845 | 0.015 | 0.00527062 | 1025.32 | 0.919961 | 1.5762 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.141388 | 0.0037998 | 0.00117246 | 43.4043 | 0.366574 | 1.94548 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.178237 | 0.00893654 | 0.00139104 | 1354.58 | 0.431833 | 1.48485 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.146149 | 0.00456025 | 0.00141139 | 134.655 | 0.272038 | 1.90245 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.217848 | 0.00311663 | 0.000973114 | 73.2758 | 0.386903 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.240062 | 0.00222634 | 0.000238535 | 239.467 | 0.769255 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.262543 | 0.00295145 | 0.0008007 | 99.4003 | 0.410446 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `soft_pgd_ofb1p4`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.195535 | 0.0034632 | 0.000787441 | 0.035192 | 0.0045779 | 1.26399 | 0.117321 | 0.381529 | NA | 0.000925691 | 0.00169243 | 249.512 | 1.20488 | 1.19783 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.198472 | 0.00336965 | 0.000772013 | 0.0349695 | 0.00439958 | 1.28685 | 0.119083 | 0.376922 | NA | 0.00114819 | 0.00147817 | 247.841 | 1.19681 | 1.18981 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.27616 | 0.015 | 0.00472398 | 0.0634691 | 0.0146427 | 0.963113 | 0.165696 | 0.0295234 | NA | 0.00038368 | 0.00110981 | 152.937 | 1.8229 | 1.53003 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.191361 | 0.00311826 | 0.000757958 | 0.0281647 | 0.00693424 | 0.734183 | 0.114817 | 0.144828 | 0.29925 | 6.53319e-05 | 0.000205478 | 20.7485 | 2.13694 | 1.68812 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.195662 | 0.00346263 | 0.000787233 | 0.0351924 | 0.00457829 | 1.26398 | 0.117397 | 0.381281 | NA | 0.000914721 | 0.00162016 | 249.55 | 1.20506 | 1.19801 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.123845 | 0.015 | 0.00472754 | 0.0334349 | 0.00845455 | 0.529556 | 0.0743068 | 0.232146 | NA | 0.00622774 | 0.00815488 | 1062.3 | 1.00527 | 1.33187 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.155192 | 0.0057768 | 0.00132463 | 0.0449007 | 0.00757044 | 0.874068 | 0.0931154 | 0.384745 | NA | 0.00230589 | 0.00183501 | 515.8 | 0.412617 | 1.53915 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152671, 0.0717183, 0.716459 | 0.239547 | 0.0027635 | 0.00066755 | 0.0284434 | 0.00368397 | 1.08176 | 0.143728 | 0.416269 | NA | 0.000493349 | 0.00145665 | 137.351 | 0.554682 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.109723 | 0.00255062 | 0.000832914 | 27.9148 | 2.5671 | 2.27001 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.325551 | 0.00528563 | 0.000785521 | 672.404 | 1.16044 | 1.16353 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.151331 | 0.00255333 | 0.000743889 | 48.2163 | 1.55824 | 1.38896 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.108873 | 0.00234625 | 0.000790696 | 24.5566 | 2.25827 | 1.99692 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.320452 | 0.00527381 | 0.000787438 | 670.974 | 1.15798 | 1.16106 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.16609 | 0.00248889 | 0.000737905 | 47.9918 | 1.55099 | 1.3825 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.27616 | 0.015 | 0.00472398 | 152.937 | 1.8229 | 1.53003 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.191361 | 0.00311826 | 0.000757958 | 20.7485 | 2.13694 | 1.68812 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.109814 | 0.00255006 | 0.000832334 | 27.875 | 2.56344 | 2.26677 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.325804 | 0.00528396 | 0.000785382 | 672.51 | 1.16063 | 1.16372 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.151369 | 0.00255387 | 0.000743981 | 48.2646 | 1.5598 | 1.39036 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.181767 | 0.015 | 0.00555852 | 528.133 | 1.17937 | 2.45647 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0539441 | 0.015 | 0.00335936 | 1625.81 | 1.01115 | 1.06449 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.135823 | 0.015 | 0.00526475 | 1032.96 | 0.926821 | 1.58796 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.14145 | 0.00379646 | 0.00117179 | 43.6479 | 0.368631 | 1.9564 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.179241 | 0.00897619 | 0.00139459 | 1368.25 | 0.436188 | 1.49983 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.144886 | 0.00455776 | 0.00140752 | 135.508 | 0.273762 | 1.9145 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.218694 | 0.00308369 | 0.000963239 | 72.6195 | 0.383438 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.237046 | 0.00224193 | 0.000238551 | 239.141 | 0.768208 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.262902 | 0.00296488 | 0.000800861 | 100.293 | 0.414132 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `soft_pgd_ofb1p8`

- Evaluated: 98
- Blocked: 12
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 0.675841, 0.86242, 1.94123 | 0.19643 | 0.00344654 | 0.000786022 | 0.0351792 | 0.0045409 | 1.27777 | 0.117858 | 0.382708 | NA | 0.000970085 | 0.00158117 | 244.718 | 1.18173 | 1.17482 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 0.675841, 0.86242, 1.94123 | 0.199754 | 0.00334971 | 0.00076715 | 0.0349443 | 0.00436059 | 1.30903 | 0.119852 | 0.379766 | NA | 0.00117678 | 0.00154435 | 241.788 | 1.16758 | 1.16075 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.015 | 0.279364 | 0.015 | 0.00468748 | 0.0642042 | 0.0146869 | 0.949309 | 0.167619 | 0.0295078 | NA | 0.000405298 | 0.00103625 | 148.965 | 1.77555 | 1.49029 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.025745 | 0.189909 | 0.00312469 | 0.000753033 | 0.0281068 | 0.00690482 | 0.716414 | 0.113945 | 0.144875 | 0.299969 | 5.96222e-05 | 0.000153261 | 20.2023 | 2.08069 | 1.64368 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.1024, 0.13067, 0.294126 | 0.196539 | 0.00344591 | 0.000785902 | 0.0351794 | 0.00454138 | 1.27771 | 0.117923 | 0.382802 | NA | 0.000961091 | 0.00152705 | 244.753 | 1.1819 | 1.17498 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | blocked=12 | 0.0229864, 0.0352696, 0.130182 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (12); channel 'process_epsilon' is not part of the robust output-feedback comparator (12) |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.003 | 0.127419 | 0.015 | 0.00472367 | 0.0334623 | 0.0083869 | 0.531194 | 0.0764515 | 0.232023 | NA | 0.00643802 | 0.00737676 | 1049.56 | 0.993206 | 1.31589 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.00591776, 0.00729051, 0.0137807 | 0.155649 | 0.00577141 | 0.00132685 | 0.0449007 | 0.00753494 | 0.878687 | 0.0933892 | 0.385818 | NA | 0.00239098 | 0.00151356 | 512.398 | 0.409895 | 1.529 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.0152671, 0.0717183, 0.716459 | 0.240805 | 0.00276165 | 0.000669575 | 0.0285025 | 0.00367206 | 1.09914 | 0.144483 | 0.416549 | NA | 0.00052981 | 0.00133689 | 136.637 | 0.551797 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 0.675841 | 0.111874 | 0.00254415 | 0.000826133 | 26.8617 | 2.47025 | 2.18437 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1.94123 | 0.326956 | 0.00522749 | 0.00078049 | 659.209 | 1.13767 | 1.1407 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 0.86242 | 0.150462 | 0.00256798 | 0.000751444 | 48.0836 | 1.55395 | 1.38514 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 0.675841 | 0.111914 | 0.00233686 | 0.000772822 | 22.8386 | 2.10028 | 1.85722 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1.94123 | 0.324549 | 0.00520472 | 0.000779975 | 654.822 | 1.1301 | 1.13311 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 0.86242 | 0.162797 | 0.00250755 | 0.000748651 | 47.7049 | 1.54171 | 1.37423 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.015 | 0.279364 | 0.015 | 0.00468748 | 148.965 | 1.77555 | 1.49029 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.025745 | 0.189909 | 0.00312469 | 0.000753033 | 20.2023 | 2.08069 | 1.64368 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.1024 | 0.111881 | 0.00254401 | 0.000825776 | 26.8605 | 2.47015 | 2.18428 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.294126 | 0.327236 | 0.00522511 | 0.00078032 | 659.272 | 1.13778 | 1.14081 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.13067 | 0.150501 | 0.00256861 | 0.000751612 | 48.1261 | 1.55532 | 1.38637 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | blocked=4 | 0.0229864 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | blocked=4 | 0.130182 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | blocked=4 | 0.0352696 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; analytical comparator is only defined for evaluated rows with supported external analytical adapters (4); channel 'process_epsilon' is not part of the robust output-feedback comparator (4) |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.003 | 0.187522 | 0.015 | 0.00554086 | 514.552 | 1.14904 | 2.3933 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.003 | 0.0519298 | 0.015 | 0.00336332 | 1615.86 | 1.00496 | 1.05797 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.003 | 0.142806 | 0.015 | 0.00526683 | 1018.26 | 0.913634 | 1.56536 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.00591776 | 0.142135 | 0.00379931 | 0.0011763 | 43.0036 | 0.36319 | 1.92752 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.0137807 | 0.178043 | 0.00895555 | 0.00139283 | 1360.32 | 0.433662 | 1.49114 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.00729051 | 0.146769 | 0.00455938 | 0.00141143 | 133.87 | 0.270453 | 1.89136 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.218313 | 0.0031095 | 0.00097181 | 72.0765 | 0.380571 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.241474 | 0.00222692 | 0.000238415 | 239.1 | 0.768078 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.0152671, 0.0717183, 0.716459 | 0.262627 | 0.00294853 | 0.0007985 | 98.7334 | 0.407692 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_and_sensory_feedback - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, and sensory_feedback. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator after converting target-relative GRU feedback signs into raw analytical observation signs. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
