# GRU perturbation-response bank

Issue: `020a65b`. Source experiment: `020a65b`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 54 |
| `delayed_observation` | 72 |
| `initial_state` | 24 |
| `process_epsilon` | 144 |
| `sensory_feedback` | 72 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 36 |
| `delayed_observation_offset` | 72 |
| `initial_position_offset` | 12 |
| `initial_velocity_offset` | 12 |
| `process_epsilon_force_state_xy` | 36 |
| `process_epsilon_integrator_xy` | 36 |
| `process_epsilon_position_xy` | 36 |
| `process_epsilon_velocity_xy` | 36 |
| `sensory_feedback_offset` | 72 |
| `target_aligned_lateral_command_load_pulse` | 18 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64`

- Evaluated: 366
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.214006 | 0.00473496 | 0.00101467 | 0.0472414 | 0.00638794 | 1.57493 | 0.128403 | 0.377727 | NA | 0.00149744 | 0.0042847 | 699.212 | 1.35165 | 1.34268 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.219628 | 0.00464162 | 0.000957039 | 0.0470159 | 0.00632778 | 1.60987 | 0.131777 | 0.373255 | NA | 0.00170204 | 0.00403099 | 694.673 | 1.34288 | 1.33397 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.276567 | 0.00362071 | 0.000817373 | 0.0372001 | 0.00492469 | 1.48324 | 0.16594 | 0.425459 | NA | 0.000781211 | 0.00177788 | 371.439 | 1.30994 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (72) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.237346 | 0.02 | 0.0073726 | 0.0614114 | 0.0181567 | 0.804634 | 0.142408 | 0.0147656 | NA | 0.00233243 | 0.00580751 | 725.849 | 3.46064 | 2.90466 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.142895 | 0.00462409 | 0.00156383 | 0.0342941 | 0.00803848 | 0.585211 | 0.085737 | 0.171094 | NA | 0.000148375 | 0.000617345 | 60.4205 | 2.48915 | 1.96635 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.214006 | 0.00473496 | 0.00101467 | 0.0472414 | 0.00638794 | 1.57493 | 0.128403 | 0.377727 | NA | 0.00149744 | 0.0042847 | 699.212 | 1.35165 | 1.34268 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.186839 | 0.0103294 | 0.00161346 | 0.0510272 | 0.0109727 | 0.628631 | 0.112103 | 0.59 | NA | 0.0071936 | 0.0376876 | 3686.57 | 14.3154 | 14.403 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.157157 | 0.02 | 0.00623298 | 0.0465479 | 0.012001 | 0.8496 | 0.0942945 | 0.228245 | NA | 0.00799289 | 0.0179982 | 2750.14 | 1.04098 | 1.3792 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.190318 | 0.00741291 | 0.00177653 | 0.0598677 | 0.0100569 | 1.28296 | 0.114191 | 0.375333 | NA | 0.00292204 | 0.00685779 | 1191.96 | 0.387205 | 1.42272 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.276567 | 0.00362071 | 0.000817373 | 0.0372001 | 0.00492469 | 1.48324 | 0.16594 | 0.425459 | NA | 0.000781211 | 0.00177788 | 371.439 | 1.30994 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (72) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.15181 | 0.00310917 | 0.000953185 | 56.0828 | 2.063 | 1.82425 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.320262 | 0.00759728 | 0.00110485 | 1917.46 | 1.3248 | 1.3272 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.169944 | 0.00349844 | 0.000985985 | 124.093 | 1.60418 | 1.42989 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.164656 | 0.00288208 | 0.000820237 | 42.6216 | 1.56783 | 1.38638 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.318464 | 0.00761992 | 0.00110713 | 1924.03 | 1.32934 | 1.33174 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.175764 | 0.00342286 | 0.000943751 | 117.365 | 1.51721 | 1.35237 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.246251 | 0.00369082 | 0.00114163 | 134.19 | 0.277495 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.286036 | 0.00342351 | 0.000356575 | 744.215 | 10.0128 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.297413 | 0.00374778 | 0.000953916 | 235.913 | 0.805819 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.237346 | 0.02 | 0.0073726 | 725.849 | 3.46064 | 2.90466 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.142895 | 0.00462409 | 0.00156383 | 60.4205 | 2.48915 | 1.96635 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.15181 | 0.00310917 | 0.000953185 | 56.0828 | 2.063 | 1.82425 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.320262 | 0.00759728 | 0.00110485 | 1917.46 | 1.3248 | 1.3272 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.169944 | 0.00349844 | 0.000985985 | 124.093 | 1.60418 | 1.42989 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.135141 | 0.00657282 | 0.00174306 | 479.494 | 50.4181 | 50.1327 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.261407 | 0.015789 | 0.00137355 | 9578.74 | 12.9994 | 13.0813 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.163969 | 0.00862637 | 0.00172378 | 1001.47 | 38.2165 | 38.4204 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.208777 | 0.02 | 0.00746624 | 1414.13 | 1.26305 | 2.63097 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0973849 | 0.02 | 0.00438933 | 4270.82 | 1.06248 | 1.11852 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.16531 | 0.02 | 0.00684336 | 2565.49 | 0.920725 | 1.57756 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.15576 | 0.00521406 | 0.00174006 | 141.824 | 0.489078 | 2.54275 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.248253 | 0.0110036 | 0.00177572 | 3093.55 | 0.399726 | 1.35642 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.166941 | 0.00602109 | 0.00181379 | 340.501 | 0.282352 | 1.92428 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.246251 | 0.00369082 | 0.00114163 | 134.19 | 0.277495 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.286036 | 0.00342351 | 0.000356575 | 744.215 | 10.0128 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.297413 | 0.00374778 | 0.000953916 | 235.913 | 0.805819 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64`

- Evaluated: 366
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.340784 | 0.00589984 | 0.000997406 | 0.0527036 | 0.00884729 | 1.53117 | 0.20447 | 0.390185 | 0.45681 | 0.00326031 | 0.00313154 | 1213.72 | 2.34625 | 2.33067 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.351955 | 0.00569385 | 0.000957452 | 0.051894 | 0.0087035 | 1.56441 | 0.211173 | 0.385771 | 0.422449 | 0.00326334 | 0.00307421 | 1198.73 | 2.31727 | 2.30189 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.505267 | 0.00458413 | 0.000804153 | 0.0489838 | 0.00785403 | 2.03352 | 0.30316 | 0.428725 | 0.462424 | 0.00210745 | 0.00508637 | 929.808 | 3.27911 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (72) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.477531 | 0.02 | 0.00626678 | 0.086869 | 0.0226298 | 1.47632 | 0.286519 | 0.0143568 | 0.449904 | 0.000639514 | 0.00390763 | 777.553 | 3.70715 | 3.11156 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.340597 | 0.00498753 | 0.00135795 | 0.0414571 | 0.011471 | 1.07933 | 0.204358 | 0.183109 | 0.371726 | 0.000330984 | 0.000644842 | 165.351 | 6.81196 | 5.38126 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.340784 | 0.00589984 | 0.000997406 | 0.0527036 | 0.00884729 | 1.53117 | 0.20447 | 0.390185 | 0.45681 | 0.00326031 | 0.00313154 | 1213.72 | 2.34625 | 2.33067 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.202813 | 0.0110054 | 0.0014894 | 0.0668748 | 0.0121147 | 0.869201 | 0.121688 | 0.59 | NA | 0.0096993 | 0.0548283 | 5125.73 | 19.9038 | 20.0256 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.481609 | 0.02 | 0.00528973 | 0.0888926 | 0.0157686 | 1.92957 | 0.288966 | 0.228313 | 0.474968 | 0.0070508 | 0.0135366 | 2836.21 | 1.07356 | 1.42237 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.411977 | 0.00843986 | 0.00150517 | 0.0685089 | 0.0133867 | 1.71986 | 0.247186 | 0.383951 | 0.427971 | 0.00470758 | 0.00307059 | 1836.47 | 0.596574 | 2.19202 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.505267 | 0.00458413 | 0.000804153 | 0.0489838 | 0.00785403 | 2.03352 | 0.30316 | 0.428725 | 0.462424 | 0.00210745 | 0.00508637 | 929.808 | 3.27911 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (72) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.339514 | 0.00382341 | 0.000842582 | 163.526 | 6.01531 | 5.31914 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.303294 | 0.0100883 | 0.00132375 | 3234.66 | 2.23487 | 2.23891 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.379544 | 0.00378782 | 0.000825889 | 242.961 | 3.14082 | 2.79959 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.33406 | 0.00345059 | 0.000757268 | 136.416 | 5.01807 | 4.43732 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.308053 | 0.00995791 | 0.00131662 | 3193.03 | 2.2061 | 2.2101 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.413753 | 0.00367305 | 0.000798472 | 266.736 | 3.44816 | 3.07354 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.48687 | 0.00375782 | 0.000811829 | 270.155 | 0.558662 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.21494 | 0.00380545 | 0.00030102 | 946.753 | 12.7377 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.813993 | 0.00618911 | 0.00129961 | 1572.52 | 5.37131 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.477531 | 0.02 | 0.00626678 | 777.553 | 3.70715 | 3.11156 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.340597 | 0.00498753 | 0.00135795 | 165.351 | 6.81196 | 5.38126 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.339514 | 0.00382341 | 0.000842582 | 163.526 | 6.01531 | 5.31914 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.303294 | 0.0100883 | 0.00132375 | 3234.66 | 2.23487 | 2.23891 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.379544 | 0.00378782 | 0.000825889 | 242.961 | 3.14082 | 2.79959 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.245742 | 0.00593199 | 0.00138733 | 633.023 | 66.5615 | 66.1847 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.173208 | 0.0179275 | 0.00146868 | 13271.1 | 18.0104 | 18.1238 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.18949 | 0.00915657 | 0.00161217 | 1473.05 | 56.2122 | 56.512 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.637146 | 0.02 | 0.00597041 | 1539.54 | 1.37507 | 2.86431 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0911435 | 0.02 | 0.00447748 | 4164.53 | 1.03604 | 1.09068 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.716538 | 0.02 | 0.00542132 | 2804.57 | 1.00653 | 1.72457 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.464247 | 0.00555529 | 0.0013144 | 348.976 | 1.20344 | 6.25677 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.220633 | 0.0138218 | 0.002006 | 4570.46 | 0.590561 | 2.00399 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.551053 | 0.0059425 | 0.00119512 | 589.991 | 0.489237 | 3.33423 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.48687 | 0.00375782 | 0.000811829 | 270.155 | 0.558662 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.21494 | 0.00380545 | 0.00030102 | 946.753 | 12.7377 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.813993 | 0.00618911 | 0.00129961 | 1572.52 | 5.37131 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

- Evaluated: 366
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.288587 | 0.00393198 | 0.000809367 | 0.0449894 | 0.00570274 | 2.01088 | 0.173152 | 0.351158 | NA | 0.00111363 | 0.00300537 | 587.064 | 1.13486 | 1.12733 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.299746 | 0.00385244 | 0.000775555 | 0.0448288 | 0.00560627 | 2.06133 | 0.179847 | 0.347415 | NA | 0.00121717 | 0.00261736 | 584.623 | 1.13014 | 1.12264 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.386914 | 0.00404761 | 0.000969546 | 0.0449709 | 0.00554392 | 1.8764 | 0.232149 | 0.410456 | NA | 0.00134504 | 0.00247359 | 444.062 | 1.56605 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (72) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.330271 | 0.02 | 0.00664181 | 0.0757516 | 0.0182497 | 1.61581 | 0.198162 | 0.0144271 | NA | 0.00254964 | 0.00230228 | 478.025 | 2.27909 | 1.91293 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.212266 | 0.00392308 | 0.000943834 | 0.0342941 | 0.00741962 | 0.952443 | 0.127359 | 0.139117 | 0.43006 | 5.71125e-05 | 4.75231e-05 | 28.7885 | 1.186 | 0.936906 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.288587 | 0.00393198 | 0.000809367 | 0.0449894 | 0.00570274 | 2.01088 | 0.173152 | 0.351158 | NA | 0.00111363 | 0.00300537 | 587.064 | 1.13486 | 1.12733 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.19141 | 0.00992244 | 0.00153844 | 0.0479023 | 0.0105173 | 0.670748 | 0.114846 | 0.59 | NA | 0.00769664 | 0.0355797 | 3376.64 | 13.1119 | 13.1921 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.177057 | 0.02 | 0.00617753 | 0.0464972 | 0.0112484 | 0.903484 | 0.106234 | 0.228031 | NA | 0.00967401 | 0.0113899 | 2529.39 | 0.957419 | 1.26849 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.229688 | 0.0068787 | 0.00158235 | 0.0598677 | 0.00939879 | 1.48524 | 0.137813 | 0.35832 | NA | 0.00312529 | 0.00524081 | 1064.78 | 0.345891 | 1.27092 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.386914 | 0.00404761 | 0.000969546 | 0.0449709 | 0.00554392 | 1.8764 | 0.232149 | 0.410456 | NA | 0.00134504 | 0.00247359 | 444.062 | 1.56605 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (72) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.193968 | 0.00258373 | 0.000645406 | 32.0165 | 1.17773 | 1.04143 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.481519 | 0.00614155 | 0.000901992 | 1636.34 | 1.13057 | 1.13261 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.190274 | 0.00307066 | 0.000880702 | 92.837 | 1.20013 | 1.06974 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.206433 | 0.00244048 | 0.000577996 | 29.9086 | 1.10019 | 0.972858 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.49027 | 0.00612185 | 0.000898911 | 1635.23 | 1.1298 | 1.13185 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.202533 | 0.002995 | 0.000849757 | 88.7338 | 1.14708 | 1.02246 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.390846 | 0.00468171 | 0.00145992 | 211.23 | 0.436808 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.355192 | 0.00328221 | 0.000360179 | 842.251 | 11.3317 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.414705 | 0.00417893 | 0.00108854 | 278.706 | 0.951988 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.330271 | 0.02 | 0.00664181 | 478.025 | 2.27909 | 1.91293 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.212266 | 0.00392308 | 0.000943834 | 28.7885 | 1.186 | 0.936906 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.193968 | 0.00258373 | 0.000645406 | 32.0165 | 1.17773 | 1.04143 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.481519 | 0.00614155 | 0.000901992 | 1636.34 | 1.13056 | 1.13261 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.190274 | 0.00307066 | 0.000880702 | 92.837 | 1.20013 | 1.06974 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.133733 | 0.00626096 | 0.00161906 | 448.541 | 47.1634 | 46.8965 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.27704 | 0.015222 | 0.0013465 | 8748.84 | 11.8731 | 11.9479 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.163457 | 0.00828436 | 0.00164977 | 932.536 | 35.5858 | 35.7757 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.285343 | 0.02 | 0.00713309 | 1127.91 | 1.00742 | 2.09848 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0727964 | 0.02 | 0.00444312 | 4054.66 | 1.00871 | 1.0619 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.173032 | 0.02 | 0.00695637 | 2405.59 | 0.863337 | 1.47923 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.20915 | 0.00460975 | 0.00133088 | 78.4286 | 0.27046 | 1.40614 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.292557 | 0.0104254 | 0.00170864 | 2853.11 | 0.368658 | 1.25099 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.187355 | 0.00560095 | 0.00170753 | 262.801 | 0.217922 | 1.48517 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.390846 | 0.00468171 | 0.00145992 | 211.23 | 0.436808 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.355192 | 0.00328221 | 0.000360179 | 842.251 | 11.3317 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.414705 | 0.00417893 | 0.00108854 | 278.706 | 0.951988 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64`

- Evaluated: 366
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.348823 | 0.00364989 | 0.000569318 | 0.0445134 | 0.00576163 | 2.23733 | 0.209294 | 0.340201 | 0.412793 | 0.00122389 | 0.000668221 | 489.357 | 0.945981 | 0.939702 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.345781 | 0.0036246 | 0.000557186 | 0.0444487 | 0.0057376 | 2.25814 | 0.207469 | 0.338884 | 0.40557 | 0.0012114 | 0.000538942 | 488.075 | 0.943502 | 0.93724 | none |
| `delayed_observation/delayed_observation_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.80997 | 0.00503625 | 0.000752568 | 0.067008 | 0.00861271 | 3.85011 | 0.485982 | 0.37744 | 0.462292 | 0.00105808 | 0.0039219 | 851.826 | 3.0041 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (72) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.547832 | 0.02 | 0.00455954 | 0.109272 | 0.0221419 | 2.01966 | 0.328699 | 0.0142318 | 0.321992 | 1.22737e-05 | 2.08349e-05 | 228.015 | 1.08711 | 0.912456 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.212621 | 0.00408686 | 0.000852832 | 0.0342941 | 0.00802187 | 0.829141 | 0.127573 | 0.145708 | 0.341799 | 5.02754e-07 | 2.77544e-06 | 26.8622 | 1.10664 | 0.874217 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.348823 | 0.00364989 | 0.000569318 | 0.0445134 | 0.00576163 | 2.23733 | 0.209294 | 0.340201 | 0.412793 | 0.00122389 | 0.000668221 | 489.357 | 0.945981 | 0.939702 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.20198 | 0.00848941 | 0.00121341 | 0.0448821 | 0.00921352 | 0.844392 | 0.121188 | 0.59 | NA | 0.00727994 | 0.0388239 | 2985.45 | 11.5929 | 11.6638 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.633486 | 0.02 | 0.00435602 | 0.105694 | 0.0158352 | 2.79587 | 0.380092 | 0.228373 | 0.417783 | 0.0057836 | 0.00115614 | 2164.87 | 0.819441 | 1.08569 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.373393 | 0.0062929 | 0.00109393 | 0.0599804 | 0.00979848 | 2.09344 | 0.224036 | 0.341609 | 0.424487 | 0.00250752 | 0.00103206 | 919.736 | 0.298774 | 1.0978 | none |
| `sensory_feedback/sensory_feedback_offset` | 72 | evaluated=72 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.80997 | 0.00503625 | 0.000752568 | 0.067008 | 0.00861271 | 3.85011 | 0.485982 | 0.37744 | 0.462292 | 0.00105808 | 0.0039219 | 851.826 | 3.0041 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (72) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.23267 | 0.0024831 | 0.000408356 | 31.1688 | 1.14654 | 1.01385 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.492331 | 0.00573708 | 0.000867862 | 1344.43 | 0.928881 | 0.930564 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.321468 | 0.00272948 | 0.000431736 | 92.4744 | 1.19544 | 1.06556 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.230149 | 0.00242745 | 0.00038431 | 31.5109 | 1.15913 | 1.02498 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.489533 | 0.00573325 | 0.000867124 | 1340.08 | 0.925876 | 0.927553 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.317662 | 0.0027131 | 0.000420124 | 92.6354 | 1.19752 | 1.06742 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.6561 | 0.00480639 | 0.000818435 | 202.428 | 0.418605 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.730351 | 0.00450396 | 0.000524064 | 1651.75 | 22.2229 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 1.04346 | 0.0057984 | 0.000915205 | 701.297 | 2.39545 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.547832 | 0.02 | 0.00455954 | 228.015 | 1.08711 | 0.912456 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.212621 | 0.00408686 | 0.000852832 | 26.8622 | 1.10664 | 0.874217 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.23267 | 0.0024831 | 0.000408356 | 31.1688 | 1.14654 | 1.01385 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.492331 | 0.00573708 | 0.000867862 | 1344.43 | 0.928881 | 0.930563 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.321468 | 0.00272948 | 0.000431736 | 92.4744 | 1.19544 | 1.06556 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.161147 | 0.00433532 | 0.00109538 | 295.548 | 31.0765 | 30.9006 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.265422 | 0.0146752 | 0.00131355 | 7979.68 | 10.8293 | 10.8975 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.179371 | 0.00645776 | 0.0012313 | 681.132 | 25.9922 | 26.1308 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.71342 | 0.02 | 0.00446873 | 676.788 | 0.604485 | 1.25916 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.235499 | 0.02 | 0.00434938 | 3921.07 | 0.975475 | 1.02692 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.951539 | 0.02 | 0.00424996 | 1896.74 | 0.680718 | 1.16633 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.305764 | 0.00447366 | 0.000838657 | 73.46 | 0.253326 | 1.31706 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.374254 | 0.00948425 | 0.00159342 | 2454.15 | 0.317108 | 1.07606 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.440162 | 0.0049208 | 0.000849714 | 231.595 | 0.192045 | 1.30882 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.6561 | 0.00480639 | 0.000818435 | 202.428 | 0.418605 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 0.730351 | 0.00450396 | 0.000524064 | 1651.75 | 22.2229 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 24 | evaluated=24 | 0.0075, 0.015, 0.0365505, 0.0375, 0.0731009, 0.182752 | 1.04346 | 0.0057984 | 0.000915205 | 701.297 | 2.39545 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
