# GRU perturbation-response bank

Issue: `e4800d6`. Source experiment: `e4800d6`.

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

### `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.334467 | 0.00429872 | 0.000725359 | 0.047073 | 0.00682835 | 1.90056 | 0.20068 | 0.355536 | 0.420745 | 0.00166697 | 0.00279902 | 682.582 | 1.31951 | 1.31075 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.343744 | 0.00423255 | 0.000698412 | 0.0469222 | 0.0068901 | 1.93931 | 0.206246 | 0.352111 | 0.409821 | 0.00162122 | 0.00311609 | 686.989 | 1.32803 | 1.31921 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.596225 | 0.00468389 | 0.000778576 | 0.0543654 | 0.00776037 | 2.51713 | 0.357735 | 0.397107 | 0.462471 | 0.00145837 | 0.00419249 | 737.926 | 0.00704502 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.406019 | 0.02 | 0.00564963 | 0.0886713 | 0.0214257 | 1.15298 | 0.243612 | 0.0146406 | 0.418917 | 0.000582097 | 0.000308754 | 331.525 | 1.58062 | 1.32667 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.227179 | 0.00436549 | 0.00100663 | 0.0343453 | 0.00903028 | 0.735701 | 0.136307 | 0.154638 | 0.34782 | 8.59744e-05 | 5.02957e-05 | 38.6112 | 1.59067 | 1.25658 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.334537 | 0.00429872 | 0.000725899 | 0.0470699 | 0.00682374 | 1.901 | 0.200722 | 0.355493 | 0.423605 | 0.0016653 | 0.00280862 | 682.961 | 1.32024 | 1.31148 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.20383 | 0.00910701 | 0.00132821 | 0.0484435 | 0.00987382 | 0.815758 | 0.122298 | 0.59 | NA | 0.00799564 | 0.0402998 | 3361.8 | 13.0542 | 13.1341 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.445609 | 0.02 | 0.00499273 | 0.0837324 | 0.0149994 | 1.74087 | 0.267365 | 0.228234 | 0.489788 | 0.00690001 | 0.00277592 | 2332.12 | 0.882748 | 1.16956 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.358698 | 0.00682476 | 0.00125849 | 0.0610281 | 0.0109557 | 1.75902 | 0.215219 | 0.354379 | 0.424257 | 0.00291624 | 0.00425143 | 1138.93 | 0.36998 | 1.35943 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.596225 | 0.00468389 | 0.000778576 | 0.0543654 | 0.00776037 | 2.51713 | 0.357735 | 0.397107 | 0.462471 | 0.00145837 | 0.00419249 | 737.926 | 0.00704502 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.243066 | 0.00280642 | 0.00054437 | 50.1374 | 1.8443 | 1.63086 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.45981 | 0.00683472 | 0.00100259 | 1873.61 | 1.2945 | 1.29685 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.300525 | 0.00325504 | 0.000629114 | 123.995 | 1.60292 | 1.42877 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.25131 | 0.00270221 | 0.000534818 | 50.9204 | 1.87311 | 1.65633 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.462227 | 0.00681656 | 0.000999114 | 1882.16 | 1.30041 | 1.30276 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.317695 | 0.00317888 | 0.000561306 | 127.889 | 1.65325 | 1.47363 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.547388 | 0.00477244 | 0.000904478 | 224.082 | 0.00103392 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.525925 | 0.00425065 | 0.000459946 | 1429.7 | 0.366776 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.715361 | 0.00502857 | 0.000971303 | 559.992 | 0.00598255 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.406019 | 0.02 | 0.00564963 | 331.525 | 1.58062 | 1.32667 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.227179 | 0.00436549 | 0.00100663 | 38.6112 | 1.59067 | 1.25658 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.243203 | 0.00280638 | 0.000543862 | 50.1031 | 1.84304 | 1.62974 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.459905 | 0.00683468 | 0.00100262 | 1875.16 | 1.29557 | 1.29792 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.300502 | 0.00325508 | 0.000631214 | 123.618 | 1.59805 | 1.42443 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.164858 | 0.00469504 | 0.00122199 | 325.94 | 34.2722 | 34.0782 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.269135 | 0.015396 | 0.0013543 | 8971.13 | 12.1748 | 12.2515 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.177497 | 0.00722996 | 0.00140833 | 788.318 | 30.0824 | 30.2429 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.522489 | 0.02 | 0.00532717 | 833.764 | 0.744691 | 1.55121 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.172528 | 0.02 | 0.00440076 | 4033.13 | 1.00335 | 1.05627 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.641809 | 0.02 | 0.00525026 | 2129.46 | 0.764237 | 1.30943 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.307154 | 0.00472972 | 0.00101894 | 118.578 | 0.408913 | 2.12597 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.362199 | 0.0103061 | 0.00168294 | 2977.54 | 0.384736 | 1.30555 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.406743 | 0.00543843 | 0.00107358 | 320.684 | 0.26592 | 1.81229 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.547388 | 0.00477244 | 0.000904478 | 224.082 | 0.00103392 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.525925 | 0.00425065 | 0.000459946 | 1429.7 | 0.366776 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.715361 | 0.00502857 | 0.000971303 | 559.992 | 0.00598255 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64`

- Evaluated: 438
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.348908 | 0.00398875 | 0.00062921 | 0.0460874 | 0.0061393 | 2.17146 | 0.209345 | 0.347419 | 0.430691 | 0.00148365 | 0.000594924 | 564.131 | 1.09053 | 1.08329 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 0.33792, 0.43121, 0.675841, 0.86242, 0.970616, 1.6896, 1.94123, 2.15605, 4.85308 | 0.351924 | 0.00396191 | 0.000615283 | 0.0459609 | 0.00611315 | 2.17522 | 0.211154 | 0.346769 | 0.412216 | 0.00148686 | 0.000605037 | 564.032 | 1.09034 | 1.0831 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.76185 | 0.00469101 | 0.000693984 | 0.0611564 | 0.0079528 | 3.4636 | 0.45711 | 0.381546 | 0.46949 | 0.00103756 | 0.00384389 | 777.037 | 0.00741841 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (108) |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.518812 | 0.02 | 0.00490962 | 0.105515 | 0.0221923 | 1.53485 | 0.311287 | 0.0151016 | 0.339701 | 1.36154e-05 | 1.48396e-05 | 270.492 | 1.28963 | 1.08244 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.228859 | 0.00426673 | 0.000903817 | 0.0342941 | 0.00843349 | 0.757341 | 0.137315 | 0.150953 | 0.342612 | -5.57161e-07 | -4.61557e-07 | 29.8008 | 1.2277 | 0.969852 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 0.0512, 0.0653349, 0.1024, 0.13067, 0.147063, 0.256, 0.294126, 0.326674, 0.735316 | 0.349083 | 0.00398859 | 0.000628809 | 0.0460855 | 0.00614165 | 2.17103 | 0.20945 | 0.347496 | 0.431054 | 0.00148066 | 0.000649278 | 564.947 | 1.0921 | 1.08486 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 0.0114932, 0.0176348, 0.0229864, 0.0352696, 0.0574661, 0.065091, 0.088174, 0.130182, 0.325455 | 0.19208 | 0.00889878 | 0.00125806 | 0.048213 | 0.00962877 | 0.815132 | 0.115248 | 0.59 | NA | 0.00761887 | 0.0431717 | 3296.61 | 12.8011 | 12.8794 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 0.0015, 0.003, 0.0075 | 0.628269 | 0.02 | 0.00447939 | 0.104364 | 0.0157241 | 2.69905 | 0.376961 | 0.228351 | 0.432819 | 0.00587428 | 0.000721094 | 2227.98 | 0.843329 | 1.11733 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 0.00295888, 0.00364525, 0.00591776, 0.00689036, 0.00729051, 0.0137807, 0.0147944, 0.0182263, 0.0344518 | 0.373462 | 0.00653846 | 0.00114531 | 0.0601961 | 0.0100293 | 2.05868 | 0.224077 | 0.347944 | 0.438919 | 0.00273306 | 0.000795935 | 980.106 | 0.318385 | 1.16986 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.76185 | 0.00469101 | 0.000693984 | 0.0611564 | 0.0079528 | 3.4636 | 0.45711 | 0.381546 | 0.46949 | 0.00103756 | 0.00384389 | 777.037 | 0.00741841 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (108) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | evaluated=12 | 0.33792, 0.675841, 1.6896 | 0.257685 | 0.0026751 | 0.00044635 | 37.0988 | 1.36468 | 1.20674 | none |
| `command_input/command_input_pulse/late` | 12 | evaluated=12 | 0.970616, 1.94123, 4.85308 | 0.449748 | 0.00639297 | 0.000962024 | 1550.93 | 1.07156 | 1.0735 | none |
| `command_input/command_input_pulse/mid` | 12 | evaluated=12 | 0.43121, 0.86242, 2.15605 | 0.339291 | 0.00289818 | 0.000479257 | 104.362 | 1.34912 | 1.20254 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | evaluated=6 | 0.33792, 0.675841, 1.6896 | 0.264502 | 0.00262203 | 0.00042136 | 36.9509 | 1.35924 | 1.20193 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | evaluated=6 | 0.970616, 1.94123, 4.85308 | 0.445399 | 0.00639739 | 0.000962984 | 1550.01 | 1.07092 | 1.07286 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | evaluated=6 | 0.43121, 0.86242, 2.15605 | 0.34587 | 0.00286631 | 0.000461506 | 105.133 | 1.35908 | 1.21142 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.645475 | 0.00453382 | 0.000766042 | 193.505 | 0.000892835 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.654787 | 0.00434289 | 0.000489639 | 1485.75 | 0.381155 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.985289 | 0.00519634 | 0.000826272 | 651.853 | 0.00696392 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset/initial_condition` | 12 | evaluated=12 | 0.0075, 0.015, 0.0375 | 0.518812 | 0.02 | 0.00490962 | 270.492 | 1.28963 | 1.08244 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | evaluated=12 | 0.0128725, 0.025745, 0.0643625 | 0.228859 | 0.00426673 | 0.000903817 | 29.8008 | 1.2277 | 0.969852 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | evaluated=12 | 0.0512, 0.1024, 0.256 | 0.257453 | 0.0026751 | 0.00044676 | 37.2037 | 1.36854 | 1.21015 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | evaluated=12 | 0.147063, 0.294126, 0.735316 | 0.450648 | 0.0063924 | 0.000961794 | 1553.38 | 1.07325 | 1.07519 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | evaluated=12 | 0.0653349, 0.13067, 0.326674 | 0.339148 | 0.00289828 | 0.000477872 | 104.258 | 1.34777 | 1.20134 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | evaluated=12 | 0.0114932, 0.0229864, 0.0574661 | 0.150652 | 0.00459846 | 0.00114474 | 334.138 | 35.1341 | 34.9353 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | evaluated=12 | 0.065091, 0.130182, 0.325455 | 0.249598 | 0.0152534 | 0.00134201 | 8779.51 | 11.9148 | 11.9898 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | evaluated=12 | 0.0176348, 0.0352696, 0.088174 | 0.175989 | 0.00684452 | 0.00128744 | 776.191 | 29.6196 | 29.7776 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.739477 | 0.02 | 0.00464328 | 729.053 | 0.651167 | 1.3564 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.220781 | 0.02 | 0.004382 | 3917.99 | 0.974709 | 1.02611 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | evaluated=12 | 0.0015, 0.003, 0.0075 | 0.924548 | 0.02 | 0.00441289 | 2036.88 | 0.731013 | 1.25251 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | evaluated=12 | 0.00295888, 0.00591776, 0.0147944 | 0.33118 | 0.00459542 | 0.000860837 | 80.4964 | 0.277591 | 1.44322 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | evaluated=12 | 0.00689036, 0.0137807, 0.0344518 | 0.347437 | 0.00992796 | 0.00165803 | 2614.16 | 0.337783 | 1.14622 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | evaluated=12 | 0.00364525, 0.00729051, 0.0182263 | 0.441769 | 0.00509199 | 0.000917062 | 245.662 | 0.203709 | 1.38831 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.645475 | 0.00453382 | 0.000766042 | 193.505 | 0.000892835 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.654787 | 0.00434289 | 0.000489639 | 1485.75 | 0.381155 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | evaluated=36 | 0.00763768, 0.0152754, 0.0381884, 0.0382898, 0.0765797, 0.191449, 0.384261, 0.768521, 1.9213 | 0.985289 | 0.00519634 | 0.000826272 | 651.853 | 0.00696392 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
