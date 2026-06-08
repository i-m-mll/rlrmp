# GRU perturbation-response bank

Issue: `020a65b`. Source experiment: `020a65b`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 18 |
| `delayed_observation` | 24 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 24 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 24 |
| `initial_position_offset` | 4 |
| `initial_velocity_offset` | 4 |
| `process_epsilon_force_state_xy` | 12 |
| `process_epsilon_integrator_xy` | 12 |
| `process_epsilon_position_xy` | 12 |
| `process_epsilon_velocity_xy` | 12 |
| `sensory_feedback_offset` | 24 |
| `target_aligned_lateral_command_load_pulse` | 6 |
| `target_stream_jump` | 1 |

## Evaluation

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64`

- Evaluated: 122
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.142836 | 0.00315886 | 0.000783234 | 0.0308642 | 0.0045804 | 0.991758 | 0.0857017 | 0.380404 | NA | 0.00023602 | 0.000766044 | 105.322 | 1.4425 | 1.39223 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.149023 | 0.00305626 | 0.000721957 | 0.0306447 | 0.00451359 | 1.03248 | 0.0894137 | 0.376172 | NA | 0.000446058 | 0.000668457 | 100.4 | 1.37508 | 1.32716 | none |
| `delayed_observation/delayed_observation_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.139557 | 0.00182558 | 0.000414288 | 0.0187815 | 0.00247982 | 0.753896 | 0.083734 | 0.426223 | NA | 0.000121705 | 0.0003698 | 64.04 | 1.21972 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.11871 | 0.01 | 0.00368639 | 0.030711 | 0.00907697 | 0.402355 | 0.0712263 | 0.0159375 | NA | 0.000423737 | 0.00155091 | 129.011 | 3.45987 | 2.90401 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.208143 | 0.0067354 | 0.00227783 | 0.0499525 | 0.0117088 | 0.852425 | 0.124886 | 0.171141 | NA | 0.000226608 | 0.00094392 | 91.1438 | 2.48874 | 1.96603 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0094018 | 0.000208671 | 5.1729e-05 | 0.00203699 | 0.00030209 | 0.0655131 | 0.00564108 | 0.381167 | NA | 1.07058e-06 | 3.28709e-06 | 0.458787 | 1.44251 | 1.39224 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0312484 | 0.00163053 | 0.000338199 | 0.00619798 | 0.00172881 | 0.0799163 | 0.0187491 | 0.59 | NA | 0.000192645 | 0.00124471 | 30.4265 | 27.6455 | 27.7401 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.396101 | 0.05 | 0.01557 | 0.117352 | 0.0301021 | 2.12362 | 0.237661 | 0.229003 | NA | 0.0242436 | 0.0519842 | 12252.8 | 1.04353 | 1.38258 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.166021 | 0.00626835 | 0.00168068 | 0.0499101 | 0.00904083 | 1.03757 | 0.0996127 | 0.375799 | NA | 0.00125627 | 0.00353358 | 354.826 | 0.371045 | 1.57177 | none |
| `sensory_feedback/sensory_feedback_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.139557 | 0.00182558 | 0.000414288 | 0.0187815 | 0.00247982 | 0.753896 | 0.083734 | 0.426223 | NA | 0.000121705 | 0.0003698 | 64.04 | 1.21972 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.168474 | 0.00344979 | 0.00105782 | 49.0897 | 2.062 | 1.82336 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.112546 | 0.00298515 | 0.000432959 | 200.226 | 1.30328 | 1.30564 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.147489 | 0.00304165 | 0.000858926 | 66.6507 | 1.6021 | 1.42804 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.18281 | 0.00319761 | 0.000909871 | 37.2817 | 1.56601 | 1.38477 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.111803 | 0.00299582 | 0.000433915 | 200.926 | 1.30784 | 1.31021 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.152456 | 0.00297534 | 0.000822085 | 62.9916 | 1.51414 | 1.34964 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.125173 | 0.00187508 | 0.000580421 | 24.7368 | 0.276893 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.142809 | 0.00170208 | 0.000177474 | 124.203 | 8.94281 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.150688 | 0.00189958 | 0.000484968 | 43.1804 | 0.795417 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.11871 | 0.01 | 0.00368639 | 129.011 | 3.45987 | 2.90401 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.208143 | 0.0067354 | 0.00227783 | 91.1438 | 2.48874 | 1.96603 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0111197 | 0.00022765 | 6.98193e-05 | 0.213765 | 2.06133 | 1.82276 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00736131 | 0.000197645 | 2.86286e-05 | 0.872385 | 1.30358 | 1.30594 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00972439 | 0.000200718 | 5.67392e-05 | 0.290212 | 1.60144 | 1.42745 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0440819 | 0.00214559 | 0.000568802 | 36.3504 | 50.4889 | 50.2031 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0148176 | 0.000910005 | 7.91353e-05 | 22.6336 | 13.014 | 13.096 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0348457 | 0.00183599 | 0.00036666 | 32.2956 | 38.3262 | 38.5307 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.522371 | 0.05 | 0.018653 | 6278.19 | 1.26168 | 2.62812 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.250266 | 0.05 | 0.010967 | 19079.2 | 1.06796 | 1.12428 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.415665 | 0.05 | 0.01709 | 11401 | 0.920628 | 1.57739 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.197397 | 0.00660813 | 0.00220551 | 162.012 | 0.489138 | 2.54306 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.12912 | 0.00600285 | 0.000969421 | 646.348 | 0.396511 | 1.34551 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.171547 | 0.00619407 | 0.00186711 | 256.117 | 0.282206 | 1.92328 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.125173 | 0.00187508 | 0.000580421 | 24.7368 | 0.276893 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.142809 | 0.00170208 | 0.000177474 | 124.203 | 8.94281 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.150688 | 0.00189958 | 0.000484968 | 43.1804 | 0.795417 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64`

- Evaluated: 122
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.277733 | 0.00379982 | 0.000714695 | 0.0351385 | 0.00661555 | 1.11883 | 0.16664 | 0.389943 | 0.456273 | 0.00114817 | 0.00110092 | 201.077 | 2.75397 | 2.65799 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.285598 | 0.00360681 | 0.000672143 | 0.0342061 | 0.00647138 | 1.13351 | 0.171359 | 0.384969 | 0.420913 | 0.00117585 | 0.000916915 | 195.167 | 2.67302 | 2.57987 | none |
| `delayed_observation/delayed_observation_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.262639 | 0.00227429 | 0.000387411 | 0.0243793 | 0.00386702 | 1.09244 | 0.157583 | 0.42331 | 0.451282 | 0.000640009 | 0.00138289 | 135.325 | 2.57741 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.234442 | 0.01 | 0.00313008 | 0.0432579 | 0.0112126 | 0.748903 | 0.140665 | 0.0141328 | 0.463241 | 0.000122899 | 0.00061618 | 132.674 | 3.5581 | 2.98646 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.496444 | 0.00726475 | 0.00197761 | 0.0604276 | 0.0167047 | 1.57565 | 0.297867 | 0.183172 | 0.37189 | 0.000507505 | 0.00102905 | 249.032 | 6.79997 | 5.37179 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0185644 | 0.000250541 | 4.68063e-05 | 0.00232864 | 0.000435442 | 0.0751053 | 0.0111386 | 0.388477 | 0.458073 | 7.44792e-06 | 4.86951e-06 | 0.868761 | 2.73155 | 2.63636 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0441636 | 0.00164528 | 0.000293371 | 0.00807719 | 0.00186219 | 0.161294 | 0.0264982 | 0.59 | NA | 0.000699844 | 0.00185415 | 43.0549 | 39.1196 | 39.2535 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 1.16077 | 0.05 | 0.0133584 | 0.216538 | 0.039956 | 4.36061 | 0.696463 | 0.228828 | 0.474552 | 0.0195109 | 0.0432866 | 12964.1 | 1.10411 | 1.46284 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.433493 | 0.006898 | 0.00131642 | 0.0607229 | 0.012497 | 1.72528 | 0.260096 | 0.384177 | 0.431785 | 0.00262822 | 0.00236681 | 602.326 | 0.629859 | 2.66812 | none |
| `sensory_feedback/sensory_feedback_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.262639 | 0.00227429 | 0.000387411 | 0.0243793 | 0.00386702 | 1.09244 | 0.157583 | 0.42331 | 0.451282 | 0.000640009 | 0.00138289 | 135.325 | 2.57741 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.381002 | 0.00424129 | 0.000927966 | 141.825 | 5.95732 | 5.26786 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.115731 | 0.00388639 | 0.000510856 | 339.207 | 2.20792 | 2.21192 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.336466 | 0.00327178 | 0.000705263 | 122.199 | 2.93733 | 2.61821 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.373007 | 0.00382719 | 0.000834725 | 119.144 | 5.00462 | 4.42542 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.1173 | 0.00383565 | 0.000508193 | 334.419 | 2.17675 | 2.1807 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.366486 | 0.00315758 | 0.00067351 | 131.937 | 3.1714 | 2.82684 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.245538 | 0.00189432 | 0.000394297 | 44.2239 | 0.495024 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.108233 | 0.00190437 | 0.000151268 | 165.283 | 11.9006 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.434145 | 0.00302419 | 0.000616667 | 196.467 | 3.61908 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.234442 | 0.01 | 0.00313008 | 132.674 | 3.5581 | 2.98646 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.496444 | 0.00726475 | 0.00197761 | 249.032 | 6.79997 | 5.37179 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0255469 | 0.000279811 | 6.07434e-05 | 0.612209 | 5.90352 | 5.22028 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00763183 | 0.000256458 | 3.3714e-05 | 1.47664 | 2.20651 | 2.2105 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0225144 | 0.000215355 | 4.59617e-05 | 0.517433 | 2.85529 | 2.54508 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0822058 | 0.00194586 | 0.000452067 | 49.5826 | 68.8678 | 68.478 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.00987765 | 0.0010331 | 8.46194e-05 | 31.3507 | 18.0262 | 18.1398 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0404074 | 0.00195688 | 0.000343426 | 48.2315 | 57.2379 | 57.5432 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 1.63564 | 0.05 | 0.015134 | 7303.45 | 1.46772 | 3.05731 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.229963 | 0.05 | 0.011192 | 18527.4 | 1.03707 | 1.09176 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 1.61671 | 0.05 | 0.0137492 | 13061.4 | 1.05471 | 1.80712 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.59395 | 0.00704068 | 0.00165455 | 394.154 | 1.19 | 6.18692 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.117586 | 0.00754876 | 0.00109292 | 967.03 | 0.593237 | 2.01308 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.588942 | 0.00610456 | 0.00120179 | 445.794 | 0.491205 | 3.34764 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.245538 | 0.00189432 | 0.000394297 | 44.2239 | 0.495024 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.108233 | 0.00190437 | 0.000151268 | 165.283 | 11.9006 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.434145 | 0.00302419 | 0.000616667 | 196.467 | 3.61908 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64`

- Evaluated: 122
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.177322 | 0.00264668 | 0.000616343 | 0.0294722 | 0.004008 | 1.24666 | 0.106393 | 0.35201 | NA | 0.000273442 | 0.000915352 | 77.4284 | 1.06046 | 1.02351 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.186035 | 0.00256726 | 0.000581643 | 0.0293102 | 0.00391319 | 1.30057 | 0.111621 | 0.348099 | NA | 0.000383362 | 0.000793845 | 75.4971 | 1.03401 | 0.99798 | none |
| `delayed_observation/delayed_observation_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.187664 | 0.00204312 | 0.000493797 | 0.0225816 | 0.00276103 | 0.944615 | 0.112598 | 0.41223 | NA | 0.000313823 | 0.000291396 | 75.6176 | 1.44023 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.164412 | 0.01 | 0.00332197 | 0.0378335 | 0.00912291 | 0.808321 | 0.098647 | 0.0159141 | NA | 0.000642487 | 0.000527664 | 85.0884 | 2.28193 | 1.91532 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.309149 | 0.00571435 | 0.00137491 | 0.0499525 | 0.0108073 | 1.38733 | 0.18549 | 0.139102 | 0.430917 | 8.74405e-05 | 7.21344e-05 | 43.4437 | 1.18625 | 0.937108 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0115847 | 0.0001748 | 4.07733e-05 | 0.0019451 | 0.000264131 | 0.0822545 | 0.0069508 | 0.352831 | NA | 1.17782e-06 | 4.37055e-06 | 0.336317 | 1.05744 | 1.02059 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.031439 | 0.00156099 | 0.000318923 | 0.00575009 | 0.00165156 | 0.086205 | 0.0188634 | 0.59 | NA | 0.000320718 | 0.00135443 | 28.2241 | 25.6444 | 25.7321 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.4473 | 0.05 | 0.0154258 | 0.116856 | 0.0282448 | 2.24769 | 0.26838 | 0.228771 | NA | 0.0272751 | 0.0338949 | 11256.1 | 0.95864 | 1.27011 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.20244 | 0.0057623 | 0.00145985 | 0.0499101 | 0.0083866 | 1.24015 | 0.121464 | 0.358539 | NA | 0.00164214 | 0.00259969 | 294.947 | 0.308429 | 1.30652 | none |
| `sensory_feedback/sensory_feedback_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.187664 | 0.00204312 | 0.000493797 | 0.0225816 | 0.00276103 | 0.944615 | 0.112598 | 0.41223 | NA | 0.000313823 | 0.000291396 | 75.6176 | 1.44023 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.214808 | 0.00286653 | 0.000716821 | 27.9681 | 1.1748 | 1.03883 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.154972 | 0.00239777 | 0.000361387 | 154.003 | 1.00242 | 1.00423 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.162185 | 0.00267573 | 0.00077082 | 50.3139 | 1.20941 | 1.07801 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.228396 | 0.00270772 | 0.000640935 | 26.0839 | 1.09565 | 0.968845 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.157576 | 0.00238415 | 0.000359851 | 152.376 | 0.991823 | 0.99362 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.172132 | 0.00260993 | 0.000744143 | 48.0317 | 1.15455 | 1.02911 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.195098 | 0.00237758 | 0.000747583 | 39.2264 | 0.439084 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.165408 | 0.00164358 | 0.000181234 | 137.531 | 9.90248 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.202486 | 0.00210819 | 0.000552573 | 50.0952 | 0.922792 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.164412 | 0.01 | 0.00332197 | 85.0884 | 2.28193 | 1.91532 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.309149 | 0.00571435 | 0.00137491 | 43.4437 | 1.18625 | 0.937108 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.014145 | 0.000189137 | 4.73509e-05 | 0.121625 | 1.17283 | 1.0371 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0100156 | 0.000158482 | 2.39339e-05 | 0.66751 | 0.997442 | 0.999249 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0105934 | 0.000176782 | 5.10352e-05 | 0.219816 | 1.21299 | 1.0812 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0436072 | 0.00204376 | 0.000528346 | 34.006 | 47.2326 | 46.9652 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0159829 | 0.000876753 | 7.75645e-05 | 20.6324 | 11.8633 | 11.9381 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.034727 | 0.00176244 | 0.00035086 | 30.0341 | 35.6424 | 35.8325 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.715227 | 0.05 | 0.0178015 | 5003.2 | 1.00546 | 2.0944 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.188028 | 0.05 | 0.0111035 | 18062.3 | 1.01104 | 1.06435 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.438644 | 0.05 | 0.0173724 | 10702.7 | 0.864241 | 1.48078 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.264984 | 0.00584202 | 0.00168777 | 89.5072 | 0.270235 | 1.40497 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.150465 | 0.00568188 | 0.000932375 | 597.234 | 0.366381 | 1.24327 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.191872 | 0.005763 | 0.0017594 | 198.099 | 0.218279 | 1.48761 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.195098 | 0.00237758 | 0.000747583 | 39.2264 | 0.439084 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.165408 | 0.00164358 | 0.000181234 | 137.531 | 9.90248 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.202486 | 0.00210819 | 0.000552573 | 50.0952 | 0.922792 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64`

- Evaluated: 122
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.234923 | 0.00245016 | 0.000389697 | 0.0291288 | 0.00425414 | 1.41237 | 0.140954 | 0.341068 | 0.412582 | 0.00032611 | 5.87529e-05 | 69.7301 | 0.955028 | 0.921747 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.232647 | 0.00242568 | 0.000377539 | 0.0290569 | 0.00422694 | 1.4304 | 0.139588 | 0.339911 | 0.40428 | 0.000321401 | 5.8381e-05 | 69.8685 | 0.956924 | 0.923577 | none |
| `delayed_observation/delayed_observation_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.39974 | 0.0024786 | 0.000359171 | 0.0335813 | 0.00423691 | 2.05763 | 0.239844 | 0.377685 | 0.458914 | 0.00033653 | 0.000464575 | 133.679 | 2.54606 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (24) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.273693 | 0.01 | 0.00228134 | 0.0544906 | 0.0110688 | 1.01013 | 0.164216 | 0.0146797 | 0.322313 | 2.11721e-06 | 4.55927e-06 | 40.6374 | 1.08983 | 0.914738 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.309701 | 0.00595289 | 0.00124223 | 0.0499525 | 0.0116846 | 1.20773 | 0.185821 | 0.145695 | 0.341766 | 7.58954e-07 | 4.37811e-06 | 40.5279 | 1.10664 | 0.874212 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0154742 | 0.000161705 | 2.56901e-05 | 0.00192246 | 0.000280624 | 0.0933334 | 0.00928454 | 0.34126 | 0.410961 | 1.67148e-06 | 2.82545e-07 | 0.303264 | 0.953519 | 0.920291 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0353678 | 0.00121088 | 0.000231467 | 0.00537655 | 0.00134685 | 0.132095 | 0.0212207 | 0.59 | NA | 0.000403455 | 0.00212298 | 21.0639 | 19.1386 | 19.2041 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 1.54683 | 0.05 | 0.0110721 | 0.256526 | 0.039576 | 6.68983 | 0.9281 | 0.228958 | 0.444861 | 0.0154885 | 0.00787655 | 9762.78 | 0.831463 | 1.10161 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.345074 | 0.00530073 | 0.000933727 | 0.0500168 | 0.00902856 | 1.76543 | 0.207045 | 0.341836 | 0.423114 | 0.00122152 | 0.000333335 | 256.914 | 0.268658 | 1.13805 | none |
| `sensory_feedback/sensory_feedback_offset` | 24 | evaluated=24 | 0.01, 0.05 | 0.39974 | 0.0024786 | 0.000359171 | 0.0335813 | 0.00423691 | 2.05763 | 0.239844 | 0.377685 | 0.458914 | 0.00033653 | 0.000464575 | 133.679 | 2.54606 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (24) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.258313 | 0.00275498 | 0.000452538 | 27.3041 | 1.1469 | 1.01417 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.165406 | 0.00222429 | 0.000344108 | 131.864 | 0.858309 | 0.859863 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.28105 | 0.00237121 | 0.000372444 | 50.0224 | 1.2024 | 1.07177 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.255486 | 0.00269319 | 0.00042608 | 27.5983 | 1.15926 | 1.0251 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.164456 | 0.00222653 | 0.00034457 | 131.701 | 0.857248 | 0.858801 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.277999 | 0.00235733 | 0.000361968 | 50.3065 | 1.20923 | 1.07785 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.333432 | 0.00243266 | 0.000410906 | 36.9854 | 0.413999 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.326871 | 0.0021437 | 0.00025922 | 232.666 | 16.7524 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.538918 | 0.00285942 | 0.000407388 | 131.384 | 2.42019 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (8) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.273693 | 0.01 | 0.00228134 | 40.6374 | 1.08983 | 0.914738 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.309701 | 0.00595289 | 0.00124223 | 40.5279 | 1.10664 | 0.874212 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0170577 | 0.000181786 | 2.98228e-05 | 0.118967 | 1.1472 | 1.01443 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0107682 | 0.000146909 | 2.27668e-05 | 0.572443 | 0.855387 | 0.856937 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0185968 | 0.000156421 | 2.44807e-05 | 0.218383 | 1.20508 | 1.07415 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0526401 | 0.00141388 | 0.000357141 | 22.3823 | 31.0879 | 30.9119 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0152478 | 0.000846162 | 7.56977e-05 | 18.884 | 10.858 | 10.9264 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0382155 | 0.0013726 | 0.000261563 | 21.9253 | 26.0195 | 26.1583 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 1.78522 | 0.05 | 0.0112329 | 3006.76 | 0.604247 | 1.25867 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.620368 | 0.05 | 0.0107908 | 18140.9 | 1.01544 | 1.06899 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 2.23491 | 0.05 | 0.0111924 | 8140.7 | 0.65736 | 1.12631 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.387402 | 0.00566977 | 0.00106266 | 83.8752 | 0.253231 | 1.31657 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.191832 | 0.00517319 | 0.000872448 | 511.003 | 0.313482 | 1.06376 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.455989 | 0.00505924 | 0.000866071 | 175.863 | 0.193778 | 1.32063 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.333432 | 0.00243266 | 0.000410906 | 36.9854 | 0.413999 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.326871 | 0.0021437 | 0.00025922 | 232.666 | 16.7524 | NA | inflated_ratio; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 8 | evaluated=8 | 0.01, 0.05 | 0.538918 | 0.00285942 | 0.000407388 | 131.384 | 2.42019 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (8) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
