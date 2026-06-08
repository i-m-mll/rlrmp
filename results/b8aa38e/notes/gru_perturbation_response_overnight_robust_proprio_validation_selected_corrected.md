# GRU perturbation-response bank

Issue: `b8aa38e`. Source experiment: `b8aa38e`.

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

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.112963 | 0.00577075 | 0.00142791 | 0.0380246 | 0.00761333 | 0.550004 | 0.0677775 | 0.446435 | NA | 0.00184656 | 0.0050787 | 411.931 | 5.64184 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.101074 | 0.00136013 | 0.000311966 | 0.0133581 | 0.00194828 | 0.644322 | 0.0606442 | 0.436417 | NA | -0.000805877 | 0.00121156 | 32.6035 | 1.53473 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.135595 | 0.01 | 0.00326978 | 0.0354618 | 0.0098552 | 0.60688 | 0.0813571 | 0.0155625 | NA | 0.000187315 | 0.000830168 | 86.1731 | 2.31102 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.231511 | 0.00646461 | 0.00174757 | 0.0499525 | 0.012233 | 1.07694 | 0.138907 | 0.161359 | 0.511258 | 2.93456e-05 | 0.000156729 | 56.335 | 1.53826 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00745028 | 0.000380905 | 9.42772e-05 | 0.00250963 | 0.000502384 | 0.0362961 | 0.00447017 | 0.446589 | NA | 7.54638e-06 | 2.49217e-05 | 1.79461 | 5.64257 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0308457 | 0.00174635 | 0.000352795 | 0.00648751 | 0.00184804 | 0.0722222 | 0.0185074 | 0.59 | NA | 0.000274887 | 0.00119142 | 35.8083 | 32.5353 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.40676 | 0.05 | 0.0148941 | 0.129476 | 0.0333162 | 2.52328 | 0.244056 | 0.228969 | NA | 0.0209591 | 0.0616599 | 12577.5 | 1.07118 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.147322 | 0.00644304 | 0.00165799 | 0.0499101 | 0.00948082 | 0.963351 | 0.0883929 | 0.387206 | NA | 0.0016627 | 0.00329744 | 402.702 | 0.42111 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.101075 | 0.0013606 | 0.000311962 | 0.013361 | 0.00194884 | 0.643989 | 0.0606453 | 0.436448 | NA | -0.000808181 | 0.00117488 | 32.2555 | 1.51836 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.13936 | 0.00572611 | 0.00192995 | 210.67 | 8.84916 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0936483 | 0.0057215 | 0.000631519 | 711.138 | 4.62884 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105879 | 0.00586464 | 0.00172225 | 313.985 | 7.54733 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.117353 | 0.00149141 | 0.000395117 | 25.0635 | 0.569659 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0913614 | 0.00100881 | 0.00010965 | 25.2954 | 32.375 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0945063 | 0.00158017 | 0.000431131 | 47.4514 | 2.50372 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.135595 | 0.01 | 0.00326978 | 86.1731 | 2.31102 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.231511 | 0.00646461 | 0.00174757 | 56.335 | 1.53826 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00918506 | 0.00037795 | 0.000127452 | 0.917645 | 8.84883 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00617838 | 0.000377638 | 4.16808e-05 | 3.09796 | 4.6292 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00698739 | 0.000387127 | 0.000113698 | 1.36821 | 7.55006 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.044139 | 0.00226803 | 0.00058706 | 41.3801 | 57.4749 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0136392 | 0.00095201 | 8.10728e-05 | 25.3841 | 14.5955 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0347588 | 0.002019 | 0.000390252 | 40.6606 | 48.2532 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.517519 | 0.05 | 0.0174498 | 5827.01 | 1.17101 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.25839 | 0.05 | 0.0108949 | 19987.4 | 1.11879 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.44437 | 0.05 | 0.0163376 | 11918.1 | 0.962382 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.182141 | 0.00643238 | 0.00202645 | 163.192 | 0.492699 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.110803 | 0.00651298 | 0.00101621 | 731.221 | 0.448577 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.149021 | 0.00638375 | 0.00193131 | 313.693 | 0.345648 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.117425 | 0.0014904 | 0.000394455 | 23.6912 | 0.538468 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0913113 | 0.00100935 | 0.000109694 | 27.2776 | 34.912 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0944899 | 0.00158206 | 0.000431738 | 45.7978 | 2.41647 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.114046 | 0.00559738 | 0.00138183 | 0.0377485 | 0.00743014 | 0.574644 | 0.0684278 | 0.44149 | NA | 0.00168955 | 0.00467435 | 386.077 | 5.28774 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.108591 | 0.00137313 | 0.000317332 | 0.0140362 | 0.00197956 | 0.705793 | 0.0651545 | 0.430042 | NA | -0.000812555 | 0.00113698 | 33.3497 | 1.56986 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.142595 | 0.01 | 0.00318097 | 0.0368353 | 0.00993386 | 0.656116 | 0.0855569 | 0.0156016 | NA | 0.000145893 | 0.000717469 | 76.849 | 2.06097 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.243774 | 0.00626039 | 0.00155527 | 0.0499525 | 0.0120415 | 1.17563 | 0.146264 | 0.154656 | 0.44532 | 3.92428e-07 | 5.27844e-05 | 47.3799 | 1.29374 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00751973 | 0.000369456 | 9.12347e-05 | 0.0024914 | 0.00049029 | 0.0379237 | 0.00451184 | 0.441521 | NA | 6.80141e-06 | 2.45018e-05 | 1.68195 | 5.28835 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0311154 | 0.00169295 | 0.000343813 | 0.00630579 | 0.00179203 | 0.0731094 | 0.0186693 | 0.59 | NA | 0.000249932 | 0.00118741 | 33.5053 | 30.4428 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.429246 | 0.05 | 0.0147589 | 0.132568 | 0.0335981 | 2.70245 | 0.257547 | 0.228914 | NA | 0.020546 | 0.0616096 | 12465.4 | 1.06164 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.150746 | 0.0062657 | 0.00159672 | 0.0499101 | 0.0092978 | 1.03289 | 0.0904474 | 0.381146 | NA | 0.00151584 | 0.00331484 | 376.485 | 0.393695 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.108591 | 0.00137313 | 0.000317332 | 0.0140362 | 0.00197956 | 0.705793 | 0.0651545 | 0.430042 | NA | -0.000812555 | 0.00113698 | 33.3497 | 1.56986 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.141654 | 0.00554347 | 0.00185476 | 190.55 | 8.00403 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0948471 | 0.005594 | 0.000624763 | 676.772 | 4.40515 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105638 | 0.00565468 | 0.00166597 | 290.909 | 6.99265 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.123127 | 0.00153734 | 0.000412493 | 24.1431 | 0.548738 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.100304 | 0.00102496 | 0.00011407 | 31.9727 | 40.9211 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.102342 | 0.0015571 | 0.000425432 | 43.9332 | 2.31809 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.142595 | 0.01 | 0.00318097 | 76.849 | 2.06097 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.243774 | 0.00626039 | 0.00155527 | 47.3799 | 1.29374 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00933782 | 0.000365883 | 0.000122484 | 0.830028 | 8.00394 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00625697 | 0.000369224 | 4.12349e-05 | 2.94825 | 4.40549 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0069644 | 0.000373261 | 0.000109986 | 1.26756 | 6.99465 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0442822 | 0.00218624 | 0.000569997 | 38.3127 | 53.2144 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0140797 | 0.000938069 | 8.0419e-05 | 24.4503 | 14.0586 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0349844 | 0.00195454 | 0.000381023 | 37.7529 | 44.8026 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.540739 | 0.05 | 0.0172248 | 5541.63 | 1.11366 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.280874 | 0.05 | 0.0108609 | 20213.8 | 1.13147 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.466125 | 0.05 | 0.016191 | 11640.8 | 0.939995 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.186097 | 0.00625066 | 0.00193393 | 143.403 | 0.432954 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.110775 | 0.00633182 | 0.0010003 | 697.161 | 0.427683 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.155365 | 0.00621463 | 0.00185594 | 288.891 | 0.318319 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.123127 | 0.00153734 | 0.000412493 | 24.1431 | 0.548738 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.100304 | 0.00102496 | 0.00011407 | 31.9727 | 40.9211 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.102342 | 0.0015571 | 0.000425432 | 43.9332 | 2.31809 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.117916 | 0.00525516 | 0.00129348 | 0.0372757 | 0.00705677 | 0.639963 | 0.0707496 | 0.432932 | NA | 0.00142416 | 0.00330543 | 335.358 | 4.59308 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.124494 | 0.00138578 | 0.000327357 | 0.0152883 | 0.00201308 | 0.838967 | 0.0746965 | 0.417365 | NA | -0.000802332 | 0.00110449 | 36.5005 | 1.71818 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.154876 | 0.01 | 0.0030437 | 0.0390604 | 0.0100217 | 0.769552 | 0.0929256 | 0.0157891 | 0.588919 | 0.000108673 | 0.000487773 | 64.2482 | 1.72303 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.274761 | 0.00589919 | 0.00128357 | 0.0499525 | 0.0118935 | 1.43695 | 0.164857 | 0.141164 | 0.340297 | -6.4437e-06 | 9.06925e-05 | 43.1794 | 1.17904 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00777493 | 0.000346855 | 8.54001e-05 | 0.00246019 | 0.000465634 | 0.0422395 | 0.00466496 | 0.433148 | NA | 5.65717e-06 | 1.58468e-05 | 1.46088 | 4.59329 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0316896 | 0.00158271 | 0.000325439 | 0.00595143 | 0.00167634 | 0.0761006 | 0.0190138 | 0.59 | NA | 0.000214558 | 0.00102604 | 29.0147 | 26.3627 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.466476 | 0.05 | 0.0146166 | 0.135522 | 0.033709 | 3.04146 | 0.279886 | 0.228635 | NA | 0.0202426 | 0.0595093 | 12275.2 | 1.04544 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.15997 | 0.005925 | 0.00148515 | 0.0499101 | 0.00894146 | 1.19778 | 0.095982 | 0.368214 | NA | 0.0012655 | 0.00305311 | 329.403 | 0.344461 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.124494 | 0.00138578 | 0.000327357 | 0.0152883 | 0.00201308 | 0.838967 | 0.0746965 | 0.417365 | NA | -0.000802332 | 0.00110449 | 36.5005 | 1.71818 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.146339 | 0.00522638 | 0.00172117 | 156.332 | 6.56671 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0972261 | 0.00530021 | 0.000609411 | 604.384 | 3.93397 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.110183 | 0.0052389 | 0.00154987 | 245.357 | 5.89771 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.133646 | 0.00162435 | 0.000452723 | 27.5994 | 0.627295 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.124017 | 0.00104483 | 0.000121592 | 40.7206 | 52.1174 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.115819 | 0.00148815 | 0.000407756 | 41.1814 | 2.17289 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.154876 | 0.01 | 0.0030437 | 64.2482 | 1.72303 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.274761 | 0.00589919 | 0.00128357 | 43.1794 | 1.17904 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00964799 | 0.000344954 | 0.000113658 | 0.681048 | 6.56733 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00641342 | 0.000349825 | 4.02212e-05 | 2.63276 | 3.93406 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0072634 | 0.000345788 | 0.000102321 | 1.06885 | 5.89811 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.044559 | 0.00202374 | 0.000536261 | 32.6001 | 45.2799 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0150669 | 0.000906215 | 7.89426e-05 | 22.3976 | 12.8783 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0354429 | 0.00181818 | 0.000361112 | 32.0466 | 38.0307 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.559123 | 0.05 | 0.0170299 | 5207.22 | 1.04646 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.337002 | 0.05 | 0.0107963 | 20592.6 | 1.15267 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.503304 | 0.05 | 0.0160237 | 11025.8 | 0.890331 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.192073 | 0.00594352 | 0.0017787 | 114.417 | 0.345442 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.118436 | 0.00594254 | 0.000963678 | 633.088 | 0.388376 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.1694 | 0.00588895 | 0.00171307 | 240.705 | 0.265225 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.133646 | 0.00162435 | 0.000452723 | 27.5994 | 0.627295 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.124017 | 0.00104483 | 0.000121592 | 40.7206 | 52.1174 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.115819 | 0.00148815 | 0.000407756 | 41.1814 | 2.17289 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.171096 | 0.00434274 | 0.00110185 | 0.0359676 | 0.00613538 | 0.946215 | 0.102658 | 0.398943 | NA | 0.000636254 | 0.0021284 | 202.2 | 2.76935 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.127658 | 0.00122862 | 0.000305918 | 0.0132868 | 0.00178636 | 0.995456 | 0.0765948 | 0.403865 | NA | -0.000732177 | 0.000418589 | 29.8225 | 1.40383 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.202614 | 0.01 | 0.00274819 | 0.0448009 | 0.0102402 | 1.07161 | 0.121569 | 0.0170781 | 0.50489 | 9.38235e-05 | 9.89334e-05 | 44.7519 | 1.20017 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.458412 | 0.00509227 | 0.00156394 | 0.0585771 | 0.0142924 | 2.9201 | 0.275047 | 0.112734 | 0.230284 | 0.000391322 | 0.000809217 | 119.052 | 3.25079 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0113638 | 0.000286257 | 7.26924e-05 | 0.00237384 | 0.000404786 | 0.0628009 | 0.00681825 | 0.398591 | NA | 2.64747e-06 | 1.04053e-05 | 0.879073 | 2.76397 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0350394 | 0.0011889 | 0.000272023 | 0.00523244 | 0.00126898 | 0.114391 | 0.0210236 | 0.59 | NA | 0.000109893 | 0.000269193 | 15.2252 | 13.8336 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.570057 | 0.05 | 0.0148502 | 0.136177 | 0.0317039 | 2.73719 | 0.342034 | 0.228784 | NA | 0.0221274 | 0.0450375 | 11620.6 | 0.989685 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.268477 | 0.00530473 | 0.0012869 | 0.0499101 | 0.0084645 | 1.61003 | 0.161086 | 0.342732 | NA | 0.000492789 | 0.00214057 | 274.543 | 0.287093 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.127658 | 0.00122862 | 0.000305918 | 0.0132868 | 0.00178636 | 0.995456 | 0.0765948 | 0.403865 | NA | -0.000732177 | 0.000418589 | 29.8225 | 1.40383 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.190373 | 0.00481772 | 0.0015332 | 102.693 | 4.31359 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.154378 | 0.00381731 | 0.000517861 | 358.166 | 2.33132 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.168539 | 0.00439318 | 0.00125447 | 145.742 | 3.50323 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.110411 | 0.00160959 | 0.000500991 | 23.4082 | 0.532034 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.169522 | 0.000894416 | 0.00010392 | 42.0514 | 53.8207 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.103041 | 0.00118185 | 0.000312843 | 24.008 | 1.26676 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.202614 | 0.01 | 0.00274819 | 44.7519 | 1.20017 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.458412 | 0.00509227 | 0.00156394 | 119.052 | 3.25079 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0125563 | 0.000317937 | 0.000101221 | 0.446989 | 4.31031 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0103963 | 0.000251014 | 3.40862e-05 | 1.55671 | 2.32614 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0111387 | 0.000289821 | 8.277e-05 | 0.633525 | 3.49591 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0471339 | 0.00153728 | 0.000453124 | 18.0584 | 25.0823 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0200189 | 0.00071277 | 6.99487e-05 | 12.4477 | 7.15725 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0379654 | 0.00131666 | 0.000292997 | 15.1694 | 18.002 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.698924 | 0.05 | 0.0172631 | 4825.02 | 0.969649 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.410486 | 0.05 | 0.0107897 | 19804.9 | 1.10858 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.600761 | 0.05 | 0.0164976 | 10231.8 | 0.826216 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.255515 | 0.00562622 | 0.00155967 | 78.6692 | 0.237513 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.299296 | 0.00484628 | 0.000776092 | 583.084 | 0.357701 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.250619 | 0.00544168 | 0.00152493 | 161.876 | 0.178366 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.110411 | 0.00160959 | 0.000500991 | 23.4082 | 0.532034 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.169522 | 0.000894416 | 0.00010392 | 42.0514 | 53.8207 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.103041 | 0.00118185 | 0.000312843 | 24.008 | 1.26676 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.112746 | 0.00577045 | 0.00142845 | 0.0380414 | 0.00759767 | 0.552785 | 0.0676476 | 0.44625 | NA | 0.00193689 | 0.0048856 | 411.032 | 5.62953 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.101074 | 0.0013478 | 0.000309761 | 0.013352 | 0.00192532 | 0.647245 | 0.0606443 | 0.435437 | NA | -0.000805556 | 0.00122605 | 34.6382 | 1.63051 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.135014 | 0.01 | 0.00327564 | 0.0353674 | 0.00984544 | 0.60341 | 0.0810083 | 0.0163906 | NA | 0.000196793 | 0.00077373 | 86.2334 | 2.31264 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.231488 | 0.00647612 | 0.00174897 | 0.0499525 | 0.0122703 | 1.06922 | 0.138893 | 0.161211 | 0.513368 | 2.99363e-05 | 0.000129675 | 56.3735 | 1.53931 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00743586 | 0.000380884 | 9.43143e-05 | 0.00251073 | 0.000501345 | 0.0364798 | 0.00446152 | 0.446385 | NA | 7.9759e-06 | 2.28759e-05 | 1.7907 | 5.63028 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.03073 | 0.00175131 | 0.000353072 | 0.00648989 | 0.00185316 | 0.0720261 | 0.018438 | 0.59 | NA | 0.000291849 | 0.00116459 | 36.0619 | 32.7657 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.409045 | 0.05 | 0.014901 | 0.129261 | 0.0332138 | 2.52626 | 0.245427 | 0.228573 | NA | 0.0212095 | 0.0605318 | 12519.7 | 1.06626 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.147485 | 0.00644441 | 0.00165761 | 0.0499101 | 0.00946691 | 0.968998 | 0.0884911 | 0.387263 | NA | 0.0017326 | 0.00305435 | 401.329 | 0.419675 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.101076 | 0.00134834 | 0.000309794 | 0.0133552 | 0.00192592 | 0.64694 | 0.0606456 | 0.435531 | 0.57 | -0.000807116 | 0.00119354 | 34.3716 | 1.61797 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.139673 | 0.00571757 | 0.00192918 | 207.891 | 8.73244 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0931071 | 0.00572748 | 0.000631793 | 712.541 | 4.63797 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105458 | 0.00586631 | 0.00172438 | 312.665 | 7.51559 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.115662 | 0.00146094 | 0.000388412 | 25.7055 | 0.58425 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0921796 | 0.000998616 | 0.0001089 | 26.4845 | 33.8969 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0953804 | 0.00158386 | 0.000431971 | 51.7245 | 2.72918 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.135014 | 0.01 | 0.00327564 | 86.2334 | 2.31264 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.231488 | 0.00647612 | 0.00174897 | 56.3735 | 1.53931 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00920575 | 0.000377379 | 0.000127405 | 0.905537 | 8.73207 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00614254 | 0.000378033 | 4.16988e-05 | 3.10408 | 4.63834 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00695931 | 0.000387241 | 0.00011384 | 1.36247 | 7.51838 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0439919 | 0.00227603 | 0.000587407 | 41.7643 | 58.0085 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0135765 | 0.000952607 | 8.10991e-05 | 25.4231 | 14.618 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0346215 | 0.00202528 | 0.000390709 | 40.9982 | 48.6539 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.51975 | 0.05 | 0.0174587 | 5783.4 | 1.16225 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.259671 | 0.05 | 0.0108988 | 19936.3 | 1.11593 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.447713 | 0.05 | 0.0163455 | 11839.4 | 0.956027 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.183284 | 0.00643087 | 0.00202395 | 160.621 | 0.484936 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.110042 | 0.00652237 | 0.00101704 | 732.469 | 0.449343 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.14913 | 0.00637999 | 0.00193186 | 310.899 | 0.342569 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.115726 | 0.00146005 | 0.000387834 | 24.5245 | 0.557406 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0921263 | 0.000999002 | 0.000108931 | 28.293 | 36.2115 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0953758 | 0.00158597 | 0.000432618 | 50.2974 | 2.65389 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.113976 | 0.00559981 | 0.00138347 | 0.037755 | 0.00742172 | 0.575135 | 0.0683857 | 0.441703 | NA | 0.00169024 | 0.00460142 | 385.796 | 5.28388 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.108691 | 0.00136594 | 0.000317046 | 0.0140165 | 0.00196827 | 0.708891 | 0.0652144 | 0.429302 | NA | -0.000807842 | 0.00116817 | 32.9827 | 1.55259 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.142587 | 0.01 | 0.00318342 | 0.0368808 | 0.00992148 | 0.656092 | 0.0855525 | 0.0160156 | NA | 0.000146743 | 0.000702641 | 76.6387 | 2.05533 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.244229 | 0.0062663 | 0.00155056 | 0.0499525 | 0.0120551 | 1.17666 | 0.146537 | 0.154672 | 0.439344 | -3.81235e-06 | 5.45169e-05 | 47.418 | 1.29478 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00751481 | 0.000369614 | 9.13423e-05 | 0.00249183 | 0.000489734 | 0.0379561 | 0.00450889 | 0.441703 | NA | 6.80019e-06 | 2.3877e-05 | 1.68071 | 5.28447 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0310437 | 0.00169671 | 0.000344123 | 0.00630814 | 0.00179594 | 0.0730882 | 0.0186262 | 0.59 | NA | 0.000249644 | 0.00119843 | 33.6914 | 30.6119 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.429756 | 0.05 | 0.0147762 | 0.132046 | 0.0334722 | 2.70464 | 0.257854 | 0.22894 | NA | 0.0206579 | 0.0609226 | 12426.6 | 1.05833 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.150701 | 0.00626846 | 0.00159865 | 0.0499101 | 0.00928858 | 1.03323 | 0.0904204 | 0.381273 | NA | 0.00151779 | 0.00321106 | 376.194 | 0.39339 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.108691 | 0.00136594 | 0.000317046 | 0.0140165 | 0.00196827 | 0.708891 | 0.0652144 | 0.429302 | NA | -0.000807842 | 0.00116817 | 32.9827 | 1.55259 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.141824 | 0.00554744 | 0.00185796 | 190.179 | 7.98844 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0946112 | 0.00559669 | 0.000624859 | 677.33 | 4.40878 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105493 | 0.00565531 | 0.0016676 | 289.878 | 6.96786 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.123053 | 0.00153857 | 0.000414073 | 25.3659 | 0.57653 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.100708 | 0.00100726 | 0.000112767 | 29.8745 | 38.2357 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.102312 | 0.001552 | 0.000424297 | 43.7079 | 2.3062 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.142587 | 0.01 | 0.00318342 | 76.6387 | 2.05533 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.244229 | 0.0062663 | 0.00155056 | 47.418 | 1.29478 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00934951 | 0.000366144 | 0.000122694 | 0.828419 | 7.98842 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0062414 | 0.000369401 | 4.12412e-05 | 2.95068 | 4.40912 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00695353 | 0.000373298 | 0.000110092 | 1.26304 | 6.9697 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0441832 | 0.00219275 | 0.000570617 | 38.6091 | 53.6261 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0140517 | 0.00093832 | 8.04278e-05 | 24.4667 | 14.068 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.034896 | 0.00195905 | 0.000381325 | 37.9983 | 45.0938 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.540501 | 0.05 | 0.0172552 | 5529.27 | 1.11118 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.281232 | 0.05 | 0.0108629 | 20166.8 | 1.12884 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.467536 | 0.05 | 0.0162105 | 11583.9 | 0.935394 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.185871 | 0.00625202 | 0.00193766 | 143.25 | 0.43249 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.110394 | 0.00633581 | 0.00100063 | 697.522 | 0.427904 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.155837 | 0.00621755 | 0.00185766 | 287.81 | 0.317128 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.123053 | 0.00153857 | 0.000414073 | 25.3659 | 0.57653 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.100708 | 0.00100726 | 0.000112767 | 29.8745 | 38.2357 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.102312 | 0.001552 | 0.000424297 | 43.7079 | 2.3062 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.117825 | 0.00525302 | 0.00129541 | 0.0372726 | 0.00703723 | 0.640689 | 0.0706951 | 0.432794 | NA | 0.0014333 | 0.00332833 | 334.509 | 4.58145 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.124972 | 0.00137805 | 0.000328654 | 0.015304 | 0.00199654 | 0.844092 | 0.0749833 | 0.415896 | NA | -0.000811251 | 0.00108581 | 36.371 | 1.71208 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.154923 | 0.01 | 0.00305181 | 0.0389784 | 0.0100049 | 0.763285 | 0.092954 | 0.0163828 | 0.59 | 0.000116944 | 0.000488247 | 64.6143 | 1.73285 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.275082 | 0.00590047 | 0.00127999 | 0.0499525 | 0.0119083 | 1.42694 | 0.165049 | 0.142148 | 0.339922 | -3.4703e-06 | 8.44587e-05 | 43.1272 | 1.17761 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00776966 | 0.000346713 | 8.55263e-05 | 0.00245999 | 0.000464351 | 0.0422864 | 0.0046618 | 0.432885 | NA | 5.73267e-06 | 1.52293e-05 | 1.45719 | 4.58168 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0316531 | 0.00158533 | 0.000325669 | 0.00594966 | 0.001679 | 0.0761258 | 0.0189919 | 0.59 | NA | 0.000215611 | 0.00105116 | 29.1254 | 26.4633 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.465397 | 0.05 | 0.0146501 | 0.134628 | 0.0335031 | 3.03556 | 0.279238 | 0.228846 | NA | 0.0204698 | 0.0588277 | 12224.4 | 1.04111 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.160222 | 0.00592282 | 0.00148742 | 0.0499101 | 0.00891524 | 1.20022 | 0.096133 | 0.368365 | NA | 0.0012739 | 0.00300791 | 328.109 | 0.343107 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.124972 | 0.00137805 | 0.000328654 | 0.015304 | 0.00199654 | 0.844092 | 0.0749833 | 0.415896 | NA | -0.000811251 | 0.00108581 | 36.371 | 1.71208 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.145798 | 0.0052229 | 0.00172628 | 155.315 | 6.52399 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.096968 | 0.0052977 | 0.000609165 | 603.427 | 3.92774 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.110709 | 0.00523845 | 0.00155079 | 244.784 | 5.88392 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.13215 | 0.0016299 | 0.000463083 | 28.5083 | 0.647952 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.125946 | 0.00103161 | 0.000120609 | 39.7623 | 50.8909 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.116821 | 0.00147264 | 0.000402269 | 40.8425 | 2.15501 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.154923 | 0.01 | 0.00305181 | 64.6143 | 1.73285 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.275082 | 0.00590047 | 0.00127999 | 43.1272 | 1.17761 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00961291 | 0.000344723 | 0.000113994 | 0.676645 | 6.52487 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00639648 | 0.000349658 | 4.0205e-05 | 2.62859 | 3.92783 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00729959 | 0.000345758 | 0.00010238 | 1.06634 | 5.88426 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0445103 | 0.0020301 | 0.000536918 | 32.8554 | 45.6344 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0150479 | 0.00090585 | 7.89175e-05 | 22.3741 | 12.8648 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0354011 | 0.00182005 | 0.000361172 | 32.1467 | 38.1495 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.559358 | 0.05 | 0.0170786 | 5193.94 | 1.04379 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.336082 | 0.05 | 0.0108003 | 20498.9 | 1.14743 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.500752 | 0.05 | 0.0160715 | 10980.5 | 0.886672 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.19288 | 0.0059415 | 0.00178234 | 113.126 | 0.341542 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.118505 | 0.00593705 | 0.000963305 | 631.582 | 0.387453 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.16928 | 0.00588991 | 0.00171661 | 239.618 | 0.264027 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.13215 | 0.0016299 | 0.000463083 | 28.5083 | 0.647952 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.125946 | 0.00103161 | 0.000120609 | 39.7623 | 50.8909 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.116821 | 0.00147264 | 0.000402269 | 40.8425 | 2.15501 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.167287 | 0.00437913 | 0.00111211 | 0.0360228 | 0.00620201 | 0.935045 | 0.100372 | 0.400513 | NA | 0.000644579 | 0.00204768 | 206.596 | 2.82956 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.13023 | 0.00123088 | 0.000301077 | 0.0134971 | 0.00180403 | 1.02661 | 0.0781378 | 0.403531 | NA | -0.000672658 | 0.00015818 | 28.476 | 1.34045 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.204561 | 0.01 | 0.00276294 | 0.0447109 | 0.0102128 | 1.09292 | 0.122736 | 0.0164375 | 0.518903 | 9.17748e-05 | 0.000147687 | 45.8328 | 1.22916 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.467763 | 0.00510203 | 0.00157433 | 0.0594687 | 0.0145196 | 2.93175 | 0.280658 | 0.113398 | 0.230808 | 0.000370068 | 0.000961903 | 124.297 | 3.39401 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0111225 | 0.00028863 | 7.33672e-05 | 0.00237749 | 0.000409139 | 0.0620738 | 0.0066735 | 0.400138 | NA | 2.62537e-06 | 1.18972e-05 | 0.897793 | 2.82283 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.035082 | 0.00119379 | 0.000273824 | 0.00526252 | 0.0012751 | 0.11397 | 0.0210492 | 0.59 | NA | 0.000109417 | 0.000258633 | 15.3565 | 13.9529 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.560406 | 0.05 | 0.0148356 | 0.137573 | 0.0322125 | 2.76383 | 0.336244 | 0.228643 | NA | 0.0215038 | 0.0484392 | 11893.6 | 1.01294 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.262278 | 0.00533584 | 0.00130684 | 0.0499101 | 0.00850497 | 1.57546 | 0.157367 | 0.344326 | NA | 0.000510445 | 0.0021571 | 276.785 | 0.289437 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.13023 | 0.00123088 | 0.000301077 | 0.0134971 | 0.00180403 | 1.02661 | 0.0781378 | 0.403531 | NA | -0.000672658 | 0.00015818 | 28.476 | 1.34045 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.184078 | 0.00486761 | 0.00155473 | 107.84 | 4.5298 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.152804 | 0.00385188 | 0.000521205 | 362.177 | 2.35743 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.164981 | 0.0044179 | 0.0012604 | 149.771 | 3.60009 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.107577 | 0.00156622 | 0.000482762 | 17.1488 | 0.389768 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.179931 | 0.000959388 | 0.000111132 | 49.2723 | 63.0625 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.103181 | 0.00116701 | 0.000309336 | 19.007 | 1.00289 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.204561 | 0.01 | 0.00276294 | 45.8328 | 1.22916 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.467763 | 0.00510203 | 0.00157433 | 124.297 | 3.39401 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0121455 | 0.000321242 | 0.000102645 | 0.469507 | 4.52744 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0103105 | 0.000253219 | 3.43061e-05 | 1.57288 | 2.35031 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0109115 | 0.000291431 | 8.31505e-05 | 0.650996 | 3.59232 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0472095 | 0.00154014 | 0.000456664 | 18.0768 | 25.1077 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0199802 | 0.000718854 | 7.02796e-05 | 12.6883 | 7.29561 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0380562 | 0.00132236 | 0.000294528 | 15.3044 | 18.1622 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.667604 | 0.05 | 0.017335 | 4996.74 | 1.00416 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.432079 | 0.05 | 0.0107497 | 20329 | 1.13792 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.581535 | 0.05 | 0.0164222 | 10355.1 | 0.836171 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.249407 | 0.00565796 | 0.00160428 | 82.6994 | 0.249681 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.291597 | 0.00487146 | 0.000782503 | 581.764 | 0.356891 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.24583 | 0.0054781 | 0.00153374 | 165.892 | 0.182791 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.107577 | 0.00156622 | 0.000482762 | 17.1488 | 0.389768 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.179931 | 0.000959388 | 0.000111132 | 49.2723 | 63.0625 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.103181 | 0.00116701 | 0.000309336 | 19.007 | 1.00289 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.133181 | 0.00354951 | 0.000895919 | 0.0317896 | 0.00514107 | 0.838689 | 0.0799089 | 0.392055 | NA | 0.0004363 | 0.00165752 | 141.213 | 1.93406 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.0703521 | 0.0010814 | 0.000243009 | 0.010158 | 0.00147404 | 0.373829 | 0.0422113 | 0.435656 | NA | -0.000730334 | 0.000629077 | 22.9885 | 1.08213 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0943705 | 0.01 | 0.00392059 | 0.0264831 | 0.0089771 | 0.320211 | 0.0566223 | 0.0167578 | NA | 0.000635812 | 0.00250006 | 174.564 | 4.68151 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.177641 | 0.00728411 | 0.00273092 | 0.0499525 | 0.0123411 | 0.681061 | 0.106585 | 0.193062 | NA | 0.000561824 | 0.00237102 | 165.427 | 4.51709 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00877759 | 0.000234288 | 5.9134e-05 | 0.00209807 | 0.000339169 | 0.055393 | 0.00526655 | 0.392495 | NA | 1.83525e-06 | 9.5629e-06 | 0.61496 | 1.93355 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0315269 | 0.00171122 | 0.000355628 | 0.00658242 | 0.00181502 | 0.0787744 | 0.0189161 | 0.59 | NA | 0.000265608 | 0.00122765 | 33.7123 | 30.6309 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.351306 | 0.05 | 0.0158365 | 0.115046 | 0.0305232 | 1.77977 | 0.210784 | 0.228615 | NA | 0.024686 | 0.0606735 | 13114.4 | 1.1169 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.153409 | 0.00666218 | 0.00181239 | 0.0499101 | 0.00957848 | 0.88792 | 0.0920454 | 0.388836 | NA | 0.00176534 | 0.00500406 | 426.862 | 0.446374 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0703679 | 0.00108092 | 0.000242929 | 0.0101555 | 0.0014736 | 0.373959 | 0.0422208 | 0.435552 | NA | -0.000727281 | 0.000618191 | 22.946 | 1.08013 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.14964 | 0.0039423 | 0.00126927 | 81.2995 | 3.41497 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.113976 | 0.00333072 | 0.000466009 | 251.812 | 1.63906 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.135928 | 0.00337552 | 0.000952478 | 90.5283 | 2.17605 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0620005 | 0.00111461 | 0.000345414 | 19.9423 | 0.45326 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0700721 | 0.000980554 | 9.49856e-05 | 15.397 | 19.7062 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0789837 | 0.00114904 | 0.000288627 | 33.6263 | 1.77425 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0943705 | 0.01 | 0.00392059 | 174.564 | 4.68151 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.177641 | 0.00728411 | 0.00273092 | 165.427 | 4.51709 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00987489 | 0.000260155 | 8.37549e-05 | 0.353889 | 3.41254 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0074961 | 0.000219964 | 3.07673e-05 | 1.09694 | 1.63914 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00896177 | 0.000222744 | 6.28796e-05 | 0.394045 | 2.17441 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0447179 | 0.00226037 | 0.00060167 | 40.3616 | 56.0602 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0145883 | 0.000940909 | 8.07017e-05 | 24.6772 | 14.1891 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0352744 | 0.00193238 | 0.000384513 | 36.0981 | 42.8388 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.433429 | 0.05 | 0.0192415 | 7447.96 | 1.49676 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.252007 | 0.05 | 0.011013 | 19384 | 1.08502 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.368483 | 0.05 | 0.0172549 | 12511.1 | 1.01027 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.173923 | 0.0070481 | 0.00244335 | 229.72 | 0.693555 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.129969 | 0.00638881 | 0.00100544 | 728.188 | 0.446716 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.156335 | 0.00654962 | 0.00198837 | 322.678 | 0.355548 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0619886 | 0.00111452 | 0.000345417 | 19.3089 | 0.438863 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.070079 | 0.000979664 | 9.49305e-05 | 16.9185 | 21.6536 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0790362 | 0.00114857 | 0.000288439 | 32.6108 | 1.72067 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.14121 | 0.00319819 | 0.000796091 | 0.0309755 | 0.00462659 | 0.97521 | 0.084726 | 0.383273 | NA | 0.000295745 | 0.000767389 | 108.401 | 1.48466 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.0832446 | 0.00111612 | 0.000265956 | 0.0112152 | 0.00153531 | 0.447563 | 0.0499468 | 0.429927 | NA | -0.000748013 | 0.000543408 | 20.5212 | 0.965991 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.118361 | 0.01 | 0.00367515 | 0.0308113 | 0.00915849 | 0.403442 | 0.0710164 | 0.0157891 | NA | 0.000459454 | 0.00155691 | 128.631 | 3.44967 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.207475 | 0.00683861 | 0.00232003 | 0.0499525 | 0.011906 | 0.824753 | 0.124485 | 0.174484 | NA | 0.00028174 | 0.00103526 | 96.5152 | 2.6354 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00928921 | 0.000211395 | 5.25708e-05 | 0.00204434 | 0.000305314 | 0.0644146 | 0.00557353 | 0.384935 | NA | 1.31258e-06 | 3.22982e-06 | 0.47289 | 1.48685 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0309653 | 0.00164009 | 0.000340429 | 0.00624803 | 0.00173962 | 0.0790555 | 0.0185792 | 0.59 | NA | 0.000219295 | 0.00129015 | 30.7789 | 27.9656 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.39711 | 0.05 | 0.0155577 | 0.117834 | 0.0304047 | 2.09116 | 0.238266 | 0.229042 | NA | 0.0242801 | 0.0528077 | 12354 | 1.05214 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.164095 | 0.00632687 | 0.0017007 | 0.0499101 | 0.00913142 | 1.01625 | 0.0984569 | 0.378141 | NA | 0.00145451 | 0.00348771 | 364.041 | 0.380682 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0832446 | 0.00111612 | 0.000265956 | 0.0112152 | 0.00153531 | 0.447563 | 0.0499468 | 0.429927 | NA | -0.000748013 | 0.000543408 | 20.5212 | 0.965991 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.162364 | 0.00350008 | 0.00109398 | 51.7366 | 2.17319 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.112834 | 0.00302995 | 0.000436834 | 205.388 | 1.33688 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.148432 | 0.00306453 | 0.000857462 | 68.0773 | 1.63639 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0746301 | 0.0013221 | 0.000420233 | 21.9964 | 0.499948 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0828609 | 0.000906855 | 9.75024e-05 | 14.1019 | 18.0487 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0922429 | 0.00111941 | 0.000280134 | 25.4653 | 1.34365 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.118361 | 0.01 | 0.00367515 | 128.631 | 3.44967 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.207475 | 0.00683861 | 0.00232003 | 96.5152 | 2.6354 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0107153 | 0.000230959 | 7.21901e-05 | 0.225212 | 2.17171 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00736782 | 0.000200985 | 2.88932e-05 | 0.896946 | 1.34028 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00978457 | 0.00020224 | 5.66292e-05 | 0.296512 | 1.63621 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0438178 | 0.00215862 | 0.000573357 | 36.7177 | 50.999 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0145359 | 0.000913363 | 7.93009e-05 | 22.8418 | 13.1337 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0345423 | 0.0018483 | 0.000368629 | 32.7772 | 38.8977 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.51288 | 0.05 | 0.0186789 | 6396.61 | 1.28548 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.259368 | 0.05 | 0.010965 | 19159.2 | 1.07244 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.419083 | 0.05 | 0.0170292 | 11506 | 0.929109 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.191361 | 0.00668254 | 0.00225027 | 172.339 | 0.520314 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.129651 | 0.00604598 | 0.000974235 | 654.518 | 0.401523 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.171273 | 0.00625209 | 0.00187759 | 265.267 | 0.292289 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0746301 | 0.0013221 | 0.000420233 | 21.9964 | 0.499948 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0828609 | 0.000906855 | 9.75024e-05 | 14.1019 | 18.0487 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0922429 | 0.00111941 | 0.000280134 | 25.4653 | 1.34365 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.158223 | 0.00284443 | 0.000698085 | 0.0300139 | 0.00417926 | 1.15639 | 0.0949336 | 0.370195 | NA | 0.000203799 | 0.0006341 | 84.9569 | 1.16358 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.106063 | 0.0012012 | 0.000301992 | 0.0130337 | 0.00167024 | 0.600222 | 0.0636376 | 0.422427 | NA | -0.000794691 | 0.000524063 | 21.4364 | 1.00907 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.16632 | 0.01 | 0.00326911 | 0.0391251 | 0.00959604 | 0.576101 | 0.0997921 | 0.0162812 | NA | 0.000244078 | 0.000751671 | 80.9521 | 2.171 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.277001 | 0.00630633 | 0.00165489 | 0.0499525 | 0.0117063 | 1.1231 | 0.166201 | 0.152023 | 0.434173 | 2.55434e-05 | 0.000197782 | 49.6562 | 1.35589 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0104139 | 0.000188028 | 4.61006e-05 | 0.00198086 | 0.000275796 | 0.0764127 | 0.00624836 | 0.37137 | NA | 9.21096e-07 | 2.58173e-06 | 0.370603 | 1.16524 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0314302 | 0.00161386 | 0.000332709 | 0.00604109 | 0.00170951 | 0.0822183 | 0.0188581 | 0.59 | NA | 0.000206447 | 0.00130774 | 29.9626 | 27.224 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.448221 | 0.05 | 0.0153129 | 0.122027 | 0.030358 | 2.45054 | 0.268933 | 0.228953 | NA | 0.0239736 | 0.0484411 | 11791 | 1.0042 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.178746 | 0.00611144 | 0.00162478 | 0.0499101 | 0.00883128 | 1.16343 | 0.107248 | 0.3715 | NA | 0.00132071 | 0.00299296 | 330.339 | 0.345439 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.106063 | 0.0012012 | 0.000301992 | 0.0130337 | 0.00167024 | 0.600222 | 0.0636376 | 0.422427 | NA | -0.000794691 | 0.000524063 | 21.4364 | 1.00907 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.181742 | 0.00315682 | 0.000936217 | 40.6241 | 1.70641 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.128081 | 0.00264444 | 0.00039791 | 161.603 | 1.05188 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.164845 | 0.00273203 | 0.000760129 | 52.6436 | 1.26541 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0986399 | 0.00160729 | 0.000516321 | 28.4282 | 0.646133 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.10081 | 0.000827565 | 9.53501e-05 | 13.5093 | 17.2903 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118738 | 0.00116873 | 0.000294305 | 22.3715 | 1.18041 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.16632 | 0.01 | 0.00326911 | 80.9521 | 2.171 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.277001 | 0.00630633 | 0.00165489 | 49.6562 | 1.35589 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0119961 | 0.000208279 | 6.17703e-05 | 0.176878 | 1.70563 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00837947 | 0.000175519 | 2.63449e-05 | 0.705797 | 1.05465 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0108663 | 0.000180285 | 5.01866e-05 | 0.229135 | 1.26441 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0441557 | 0.00212491 | 0.000558357 | 36.0396 | 50.0572 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0152041 | 0.000894326 | 7.83287e-05 | 21.6579 | 12.453 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0349308 | 0.00182234 | 0.000361442 | 32.1903 | 38.2013 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.607066 | 0.05 | 0.0181091 | 5662.12 | 1.13787 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.268695 | 0.05 | 0.0109462 | 18850.4 | 1.05515 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.468901 | 0.05 | 0.0168833 | 10860.6 | 0.87699 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.212853 | 0.00644695 | 0.00210484 | 144.066 | 0.434955 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.135621 | 0.00584597 | 0.000954696 | 615.408 | 0.377531 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.187764 | 0.0060414 | 0.0018148 | 231.542 | 0.255128 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0986399 | 0.00160729 | 0.000516321 | 28.4282 | 0.646133 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.10081 | 0.000827565 | 9.53501e-05 | 13.5093 | 17.2903 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118738 | 0.00116873 | 0.000294305 | 22.3715 | 1.18041 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.179925 | 0.00246292 | 0.000593791 | 0.02879 | 0.00395993 | 1.38058 | 0.107955 | 0.347273 | 0.53875 | 7.9067e-05 | 0.000542004 | 66.8844 | 0.916053 | none |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.112313 | 0.000978737 | 0.000249768 | 0.0116077 | 0.00144597 | 0.729197 | 0.0673878 | 0.396156 | NA | -0.000574377 | 2.37331e-05 | 9.88708 | 0.465412 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.253373 | 0.01 | 0.00263254 | 0.0512799 | 0.0103111 | 0.95851 | 0.152024 | 0.0162266 | 0.467819 | 1.49159e-05 | 3.87246e-05 | 41.6063 | 1.11581 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.381907 | 0.00561304 | 0.00118215 | 0.0500293 | 0.0120958 | 1.71113 | 0.229144 | 0.130859 | 0.298564 | -4.40545e-05 | 5.80864e-05 | 50.898 | 1.3898 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0120642 | 0.000162125 | 3.88042e-05 | 0.00190006 | 0.000266758 | 0.0913736 | 0.00723853 | 0.346305 | NA | 2.98177e-07 | 4.34464e-06 | 0.321726 | 1.01157 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0353751 | 0.00151638 | 0.000327469 | 0.00580707 | 0.00160506 | 0.113535 | 0.0212251 | 0.59 | NA | 0.000172385 | 0.000335305 | 25.9946 | 23.6186 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.521356 | 0.05 | 0.0151155 | 0.126168 | 0.0306298 | 2.6112 | 0.312813 | 0.228836 | NA | 0.0232766 | 0.0397056 | 11368.2 | 0.968191 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.202558 | 0.00602446 | 0.0016058 | 0.0499101 | 0.00891675 | 1.2125 | 0.121535 | 0.365586 | NA | 0.0010333 | 0.0044711 | 328.863 | 0.343895 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.112313 | 0.000978737 | 0.000249768 | 0.0116077 | 0.00144597 | 0.729197 | 0.0673878 | 0.396156 | NA | -0.000574377 | 2.37331e-05 | 9.88708 | 0.465412 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.173308 | 0.00303461 | 0.000897931 | 33.4789 | 1.40627 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.187947 | 0.00196572 | 0.000296569 | 128.124 | 0.833966 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.178521 | 0.00238843 | 0.000586872 | 39.0503 | 0.938661 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.102138 | 0.00138401 | 0.00041818 | 3.13375 | 0.0712256 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.116347 | 0.00058592 | 7.12832e-05 | 23.1343 | 29.609 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118453 | 0.000966285 | 0.000259842 | 3.39323 | 0.17904 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.253373 | 0.01 | 0.00263254 | 41.6063 | 1.11581 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.381907 | 0.00561304 | 0.00118215 | 50.898 | 1.3898 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0114471 | 0.000200218 | 5.91782e-05 | 0.145373 | 1.40183 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0129902 | 0.000128634 | 1.85677e-05 | 0.650529 | 0.972068 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0117553 | 0.000157522 | 3.86667e-05 | 0.169277 | 0.934105 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0479898 | 0.00202132 | 0.000550933 | 32.6881 | 45.4022 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0194332 | 0.000813872 | 7.52838e-05 | 17.202 | 9.89091 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0387023 | 0.00171396 | 0.00035619 | 28.0937 | 33.3397 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.665534 | 0.05 | 0.0180181 | 5388.54 | 1.08289 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.273618 | 0.05 | 0.0109481 | 18843.6 | 1.05477 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.624916 | 0.05 | 0.0163802 | 9872.42 | 0.797196 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.210605 | 0.00662833 | 0.00219634 | 152.234 | 0.459616 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.180411 | 0.00561734 | 0.000912386 | 638.639 | 0.391782 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.216658 | 0.00582771 | 0.00170866 | 195.715 | 0.215651 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.102138 | 0.00138401 | 0.00041818 | 3.13375 | 0.0712256 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.116347 | 0.00058592 | 7.12832e-05 | 23.1343 | 29.609 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118453 | 0.000966285 | 0.000259842 | 3.39323 | 0.17904 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
