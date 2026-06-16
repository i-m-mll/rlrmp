# GRU perturbation-response bank

Issue: `020a65b`. Source experiment: `020a65b`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 54 |
| `delayed_observation` | 108 |
| `initial_state` | 24 |
| `process_epsilon` | 144 |
| `sensory_feedback` | 108 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 36 |
| `delayed_observation_offset` | 108 |
| `initial_position_offset` | 12 |
| `initial_velocity_offset` | 12 |
| `process_epsilon_force_state_xy` | 36 |
| `process_epsilon_integrator_xy` | 36 |
| `process_epsilon_position_xy` | 36 |
| `process_epsilon_velocity_xy` | 36 |
| `sensory_feedback_offset` | 108 |
| `target_aligned_lateral_command_load_pulse` | 18 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.265013 | 0.00484328 | 0.0011097 | 0.0481641 | 0.00656601 | 1.56846 | 0.159008 | 0.379289 | NA | 0.00153875 | 0.0057329 | 773.828 | 1.49589 | 1.48597 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.272256 | 0.00476435 | 0.00106277 | 0.0480099 | 0.00657911 | 1.59352 | 0.163354 | 0.376089 | NA | 0.00169399 | 0.00607895 | 775.582 | 1.49929 | 1.48933 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.283822 | 0.00347072 | 0.000808961 | 0.0349189 | 0.00462027 | 1.41373 | 0.170293 | 0.424707 | NA | 0.000986515 | 0.00289234 | 422.266 | 0.00432404 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.139006 | 0.02 | 0.0091057 | 0.0392199 | 0.0145227 | 0.412649 | 0.0834039 | 0.0144167 | NA | 0.00709262 | 0.00376587 | 1471.92 | 7.01771 | 5.89025 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.154295 | 0.00463098 | 0.00156855 | 0.0342941 | 0.00782264 | 0.562106 | 0.0925768 | 0.173 | NA | 0.000300229 | 0.000113114 | 55.1259 | 2.27102 | 1.79404 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.264873 | 0.00484268 | 0.00110955 | 0.0481607 | 0.00656744 | 1.56746 | 0.158924 | 0.379215 | NA | 0.00156257 | 0.00573911 | 774.065 | 1.49635 | 1.48642 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.206381 | 0.00983043 | 0.0016107 | 0.0487896 | 0.0104237 | 0.740607 | 0.123829 | 0.59 | NA | 0.00704923 | 0.0289425 | 3188.39 | 12.3809 | 12.4566 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.100889 | 0.02 | 0.00689806 | 0.0325362 | 0.00950659 | 0.463326 | 0.0605332 | 0.227739 | NA | 0.0116167 | 0.00914651 | 2981.71 | 1.12863 | 1.49533 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.236623 | 0.00733914 | 0.00181657 | 0.0598677 | 0.0100126 | 1.35547 | 0.141974 | 0.373311 | NA | 0.00297412 | 0.0086253 | 1221.25 | 0.396721 | 1.45769 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.283822 | 0.00347072 | 0.000808961 | 0.0349189 | 0.00462027 | 1.41373 | 0.170293 | 0.424707 | NA | 0.000986515 | 0.00289234 | 422.266 | 0.00432404 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.147821 | 0.00337696 | 0.00113308 | 67.9162 | 2.49829 | 2.20916 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.466547 | 0.00724771 | 0.00104868 | 2105.84 | 1.45495 | 1.45758 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.180672 | 0.00390517 | 0.00114733 | 147.732 | 1.90977 | 1.70228 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.155351 | 0.00323751 | 0.00104794 | 57.6742 | 2.12154 | 1.87601 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.472099 | 0.00722226 | 0.00104384 | 2128.5 | 1.47061 | 1.47327 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.189317 | 0.00383329 | 0.00109652 | 140.57 | 1.81719 | 1.61976 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.228442 | 0.00386884 | 0.00125901 | 161.42 | 0.000798862 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.365693 | 0.00324124 | 0.000341061 | 914.601 | 0.251639 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.257331 | 0.0033021 | 0.000826808 | 190.777 | 0.00218607 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.139006 | 0.02 | 0.0091057 | 1471.92 | 7.01771 | 5.89025 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.154295 | 0.00463098 | 0.00156855 | 55.1259 | 2.27102 | 1.79404 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.147733 | 0.00337698 | 0.00113314 | 67.9969 | 2.50127 | 2.21179 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.466445 | 0.00724645 | 0.00104849 | 2106.47 | 1.45538 | 1.45802 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.180442 | 0.00390461 | 0.00114701 | 147.731 | 1.90976 | 1.70227 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.141411 | 0.00651837 | 0.00178141 | 469.025 | 49.3174 | 49.0382 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.300564 | 0.0148157 | 0.00133298 | 8225.16 | 11.1624 | 11.2328 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.177167 | 0.00815724 | 0.00171772 | 870.975 | 33.2366 | 33.4139 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.130568 | 0.02 | 0.00866204 | 2051.43 | 1.83227 | 3.81667 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.0576623 | 0.02 | 0.00447999 | 4020.76 | 1.00028 | 1.05303 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.114436 | 0.02 | 0.00755216 | 2872.94 | 1.03106 | 1.76661 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.16081 | 0.00525927 | 0.00182215 | 139.876 | 0.482359 | 2.50782 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.370165 | 0.0104806 | 0.00169291 | 3175.26 | 0.410284 | 1.39224 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.178896 | 0.00627755 | 0.00193466 | 348.625 | 0.289089 | 1.97019 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.228442 | 0.00386884 | 0.00125901 | 161.42 | 0.000798862 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.365693 | 0.00324124 | 0.000341061 | 914.601 | 0.251639 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.257331 | 0.0033021 | 0.000826808 | 190.777 | 0.00218607 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.325785 | 0.00410176 | 0.000647642 | 0.0461684 | 0.00621994 | 2.04149 | 0.195471 | 0.352612 | 0.428619 | 0.00160514 | 0.000745446 | 576.401 | 1.11425 | 1.10685 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.321261 | 0.00407033 | 0.000633972 | 0.0460293 | 0.00616438 | 2.04985 | 0.192757 | 0.35162 | 0.411947 | 0.00163608 | 0.000859665 | 576.517 | 1.11447 | 1.10707 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.711959 | 0.00465143 | 0.000679015 | 0.0597325 | 0.00786062 | 3.32839 | 0.427175 | 0.384711 | 0.463208 | 0.00105972 | 0.00264909 | 715.927 | 0.00733115 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.531252 | 0.02 | 0.00484369 | 0.108227 | 0.0223331 | 1.63995 | 0.318751 | 0.0147708 | 0.335583 | 3.46586e-05 | 5.70236e-05 | 263.092 | 1.25435 | 1.05282 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.231138 | 0.00427837 | 0.000893876 | 0.0342941 | 0.00851482 | 0.819782 | 0.138683 | 0.149732 | 0.336852 | 4.81345e-06 | 6.27195e-06 | 30.5352 | 1.25796 | 0.993753 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.326134 | 0.00410018 | 0.00064764 | 0.046163 | 0.00622046 | 2.04109 | 0.195681 | 0.352484 | 0.42745 | 0.00159445 | 0.000786875 | 576.384 | 1.11421 | 1.10682 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.191119 | 0.00916483 | 0.00127178 | 0.0513025 | 0.00994585 | 0.7992 | 0.114671 | 0.59 | NA | 0.00766412 | 0.0452158 | 3574.04 | 13.8784 | 13.9633 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.613093 | 0.02 | 0.00450453 | 0.103644 | 0.0157699 | 2.69127 | 0.367856 | 0.228414 | 0.426262 | 0.00582629 | 0.000720826 | 2236.32 | 0.846487 | 1.12152 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.362281 | 0.00669777 | 0.00116426 | 0.0605523 | 0.0102205 | 1.97931 | 0.217368 | 0.353641 | 0.42786 | 0.00285609 | 0.000832546 | 1024.76 | 0.332892 | 1.22316 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.711959 | 0.00465143 | 0.000679015 | 0.0597325 | 0.00786062 | 3.32839 | 0.427175 | 0.384711 | 0.463208 | 0.00105972 | 0.00264909 | 715.927 | 0.00733115 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.253329 | 0.00271617 | 0.000463816 | 35.5395 | 1.30732 | 1.15602 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.392031 | 0.00672951 | 0.00100941 | 1592.3 | 1.10014 | 1.10213 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.331993 | 0.0028596 | 0.000469698 | 101.359 | 1.31029 | 1.16794 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.252394 | 0.00262928 | 0.000432854 | 35.7243 | 1.31412 | 1.16203 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.3832 | 0.00675956 | 0.00101451 | 1594.69 | 1.10179 | 1.10379 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.32819 | 0.00282216 | 0.000454552 | 99.1367 | 1.28157 | 1.14233 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.640218 | 0.00454669 | 0.000773949 | 195.304 | 0.000966552 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.543666 | 0.00430084 | 0.000480717 | 1290.41 | 0.355037 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.951992 | 0.00510676 | 0.000782378 | 662.066 | 0.00758645 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.531252 | 0.02 | 0.00484369 | 263.092 | 1.25435 | 1.05282 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.231138 | 0.00427837 | 0.000893876 | 30.5352 | 1.25796 | 0.993753 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.253772 | 0.00271635 | 0.000464395 | 35.7821 | 1.31624 | 1.16391 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.393065 | 0.00672327 | 0.0010085 | 1592.53 | 1.1003 | 1.10229 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.331566 | 0.00286091 | 0.000470026 | 100.84 | 1.30359 | 1.16196 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.15547 | 0.00471961 | 0.00115161 | 367.309 | 38.622 | 38.4034 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.240926 | 0.0157269 | 0.00136493 | 9497.48 | 12.8891 | 12.9703 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.17696 | 0.00704798 | 0.0012988 | 857.342 | 32.7164 | 32.8909 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.721738 | 0.02 | 0.0046991 | 729.194 | 0.651293 | 1.35666 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.199318 | 0.02 | 0.00438651 | 3933.26 | 0.97851 | 1.03011 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.918223 | 0.02 | 0.00442799 | 2046.49 | 0.734461 | 1.25842 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.337505 | 0.00465296 | 0.000880604 | 83.1609 | 0.286779 | 1.49099 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.30325 | 0.0103427 | 0.00170888 | 2741.42 | 0.354226 | 1.20202 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.446087 | 0.00509768 | 0.000903306 | 249.711 | 0.207067 | 1.4112 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.640218 | 0.00454669 | 0.000773949 | 195.304 | 0.000966552 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.543666 | 0.00430084 | 0.000480717 | 1290.41 | 0.355037 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00763867, 0.0152773, 0.0371213, 0.0381934, 0.0742425, 0.185606, 0.371022, 0.742044, 1.85511 | 0.951992 | 0.00510676 | 0.000782378 | 662.066 | 0.00758645 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
