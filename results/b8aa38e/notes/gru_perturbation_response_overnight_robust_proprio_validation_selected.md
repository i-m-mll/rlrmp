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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.112963 | 0.00577075 | 0.00142791 | 0.0380246 | 0.00761333 | 0.550004 | 0.0677775 | 0.446435 | NA | 0.00184656 | 0.0050787 | 411.931 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.504603 | 0.00264404 | 0.000898907 | 0.0209159 | 0.00600029 | 1.29596 | 0.302762 | 0.359031 | 0.482154 | -0.000785329 | 0.000916173 | 31.4726 | 1.4815 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.135595 | 0.01 | 0.00326978 | 0.0354618 | 0.0098552 | 0.60688 | 0.0813571 | 0.0155625 | NA | 0.000187315 | 0.000830168 | 86.1731 | 4.44343 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.231511 | 0.00646461 | 0.00174757 | 0.0499525 | 0.012233 | 1.07694 | 0.138907 | 0.161359 | 0.511258 | 2.93456e-05 | 0.000156729 | 56.335 | 3.94234 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00745028 | 0.000380905 | 9.42772e-05 | 0.00250963 | 0.000502384 | 0.0362961 | 0.00447017 | 0.446589 | NA | 7.54638e-06 | 2.49217e-05 | 1.79461 | 5.64257 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0308457 | 0.00174635 | 0.000352795 | 0.00648751 | 0.00184804 | 0.0722222 | 0.0185074 | 0.59 | NA | 0.000274887 | 0.00119142 | 35.8083 | 32.5353 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.40676 | 0.05 | 0.0148941 | 0.129476 | 0.0333162 | 2.52328 | 0.244056 | 0.228969 | NA | 0.0209591 | 0.0616599 | 12577.5 | 1.07118 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.147322 | 0.00644304 | 0.00165799 | 0.0499101 | 0.00948082 | 0.963351 | 0.0883929 | 0.387206 | NA | 0.0016627 | 0.00329744 | 402.702 | 0.42111 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.117033 | 0.0016267 | 0.000477594 | 0.0140629 | 0.00305246 | 0.656383 | 0.0702198 | 0.380635 | 0.42211 | -0.000791105 | 0.00108169 | 32.0273 | 1.50761 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.13936 | 0.00572611 | 0.00192995 | 210.67 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0936483 | 0.0057215 | 0.000631519 | 711.138 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105879 | 0.00586464 | 0.00172225 | 313.985 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.505122 | 0.00267757 | 0.000902068 | 23.9327 | 0.543957 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.505546 | 0.00250402 | 0.000846114 | 24.1646 | 30.9277 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.503141 | 0.00275053 | 0.000948539 | 46.3206 | 2.44406 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.135595 | 0.01 | 0.00326978 | 86.1731 | 4.44343 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.231511 | 0.00646461 | 0.00174757 | 56.335 | 3.94234 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.120301 | 0.00165599 | 0.00048217 | 23.463 | 0.533281 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.120588 | 0.00143858 | 0.000400202 | 27.0494 | 34.62 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.11021 | 0.00178553 | 0.00055041 | 45.5695 | 2.40443 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0705339 | 0.00326104 | 0.00092735 | 0.0220332 | 0.00461325 | 0.341512 | 0.0423203 | 0.232482 | 0 | 0.000709172 | 0.00246616 | 167.457 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.53837 | 0.00262997 | 0.000879171 | 0.0209676 | 0.00606652 | 1.35426 | 0.323022 | 0.345104 | 0.468668 | -0.000511315 | 0.000577601 | 19.929 | 0.938111 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.142595 | 0.01 | 0.00318097 | 0.0368353 | 0.00993386 | 0.656116 | 0.0855569 | 0.0156016 | NA | 0.000145893 | 0.000717469 | 76.849 | 3.96264 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.243774 | 0.00626039 | 0.00155527 | 0.0499525 | 0.0120415 | 1.17563 | 0.146264 | 0.154656 | 0.44532 | 3.92428e-07 | 5.27844e-05 | 47.3799 | 3.31566 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00751973 | 0.000369456 | 9.12347e-05 | 0.0024914 | 0.00049029 | 0.0379237 | 0.00451184 | 0.441521 | NA | 6.80141e-06 | 2.45018e-05 | 1.68195 | 5.28835 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0311154 | 0.00169295 | 0.000343813 | 0.00630579 | 0.00179203 | 0.0731094 | 0.0186693 | 0.59 | NA | 0.000249932 | 0.00118741 | 33.5053 | 30.4428 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.429246 | 0.05 | 0.0147589 | 0.132568 | 0.0335981 | 2.70245 | 0.257547 | 0.228914 | NA | 0.020546 | 0.0616096 | 12465.4 | 1.06164 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.150746 | 0.0062657 | 0.00159672 | 0.0499101 | 0.0092978 | 1.03289 | 0.0904474 | 0.381146 | NA | 0.00151584 | 0.00331484 | 376.485 | 0.393695 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.125889 | 0.00167343 | 0.000490329 | 0.0148023 | 0.00313166 | 0.714792 | 0.0755333 | 0.381125 | 0.430643 | -0.000816211 | 0.00117048 | 32.2426 | 1.51775 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.109063 | 0.00414924 | 0.00137548 | 129.151 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.023787 | 0.00139723 | 0.000156123 | 171.241 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0787518 | 0.00423665 | 0.00125045 | 201.978 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.549481 | 0.00282351 | 0.000933409 | 24.5042 | 0.556947 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.551292 | 0.00259473 | 0.000866564 | 35.2304 | 45.0906 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.514337 | 0.00247166 | 0.00083754 | 0.0522867 | 0.00275885 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.142595 | 0.01 | 0.00318097 | 76.849 | 3.96264 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.243774 | 0.00626039 | 0.00155527 | 47.3799 | 3.31566 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.12816 | 0.001775 | 0.000515646 | 23.3631 | 0.531009 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.131101 | 0.00142028 | 0.000389838 | 30.5603 | 39.1134 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118405 | 0.00182503 | 0.000565501 | 42.8045 | 2.25853 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0730617 | 0.00305827 | 0.000866118 | 0.0217632 | 0.00438868 | 0.376225 | 0.043837 | 0.226297 | 0 | 0.000536704 | 0.00184964 | 140.67 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.625702 | 0.00264061 | 0.000882548 | 0.0218118 | 0.00620663 | 1.55078 | 0.375421 | 0.345094 | 0.466529 | -0.000541041 | 0.000760933 | 24.39 | 1.14811 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.154876 | 0.01 | 0.0030437 | 0.0390604 | 0.0100217 | 0.769552 | 0.0929256 | 0.0157891 | 0.588919 | 0.000108673 | 0.000487773 | 64.2482 | 3.31289 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.274761 | 0.00589919 | 0.00128357 | 0.0499525 | 0.0118935 | 1.43695 | 0.164857 | 0.141164 | 0.340297 | -6.4437e-06 | 9.06925e-05 | 43.1794 | 3.02171 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00777493 | 0.000346855 | 8.54001e-05 | 0.00246019 | 0.000465634 | 0.0422395 | 0.00466496 | 0.433148 | NA | 5.65717e-06 | 1.58468e-05 | 1.46088 | 4.59329 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0316896 | 0.00158271 | 0.000325439 | 0.00595143 | 0.00167634 | 0.0761006 | 0.0190138 | 0.59 | NA | 0.000214558 | 0.00102604 | 29.0147 | 26.3627 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.466476 | 0.05 | 0.0146166 | 0.135522 | 0.033709 | 3.04146 | 0.279886 | 0.228635 | NA | 0.0202426 | 0.0595093 | 12275.2 | 1.04544 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.15997 | 0.005925 | 0.00148515 | 0.0499101 | 0.00894146 | 1.19778 | 0.095982 | 0.368214 | NA | 0.0012655 | 0.00305311 | 329.403 | 0.344461 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.143041 | 0.00164659 | 0.000481107 | 0.0158736 | 0.00311902 | 0.857997 | 0.0858244 | 0.372792 | 0.419598 | -0.000804419 | 0.00113458 | 35.2275 | 1.65825 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.112197 | 0.00392108 | 0.00128078 | 105.768 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0244569 | 0.00132197 | 0.000152206 | 148.19 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0825308 | 0.00393175 | 0.00116537 | 168.051 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.636912 | 0.0028636 | 0.000950673 | 28.5806 | 0.649596 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.642697 | 0.00259585 | 0.000865085 | 44.4321 | 56.8676 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.597496 | 0.00246238 | 0.000831885 | 0.157372 | 0.00830357 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.154876 | 0.01 | 0.0030437 | 64.2482 | 3.31289 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.274761 | 0.00589919 | 0.00128357 | 43.1794 | 3.02171 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.13929 | 0.0018189 | 0.00053073 | 26.6038 | 0.604667 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.156708 | 0.0013857 | 0.000375579 | 39.2049 | 50.1775 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.133124 | 0.00173517 | 0.000537013 | 39.8737 | 2.10389 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.102434 | 0.00263372 | 0.000746408 | 0.0211023 | 0.00389822 | 0.508149 | 0.0614603 | 0.208461 | 0 | 0.000285825 | 0.000591303 | 87.9897 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.917535 | 0.00280325 | 0.000935824 | 0.0234818 | 0.0067218 | 2.40121 | 0.550521 | 0.34626 | 0.464395 | -0.000579725 | 0.000474469 | 21.6801 | 1.02054 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.202614 | 0.01 | 0.00274819 | 0.0448009 | 0.0102402 | 1.07161 | 0.121569 | 0.0170781 | 0.50489 | 9.38235e-05 | 9.89334e-05 | 44.7519 | 2.30759 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.458412 | 0.00509227 | 0.00156394 | 0.0585771 | 0.0142924 | 2.9201 | 0.275047 | 0.112734 | 0.230284 | 0.000391322 | 0.000809217 | 119.052 | 8.3313 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0113638 | 0.000286257 | 7.26924e-05 | 0.00237384 | 0.000404786 | 0.0628009 | 0.00681825 | 0.398591 | NA | 2.64747e-06 | 1.04053e-05 | 0.879073 | 2.76397 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0350394 | 0.0011889 | 0.000272023 | 0.00523244 | 0.00126898 | 0.114391 | 0.0210236 | 0.59 | NA | 0.000109893 | 0.000269193 | 15.2252 | 13.8336 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.570057 | 0.05 | 0.0148502 | 0.136177 | 0.0317039 | 2.73719 | 0.342034 | 0.228784 | NA | 0.0221274 | 0.0450375 | 11620.6 | 0.989685 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.268477 | 0.00530473 | 0.0012869 | 0.0499101 | 0.0084645 | 1.61003 | 0.161086 | 0.342732 | NA | 0.000492789 | 0.00214057 | 274.543 | 0.287093 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.152138 | 0.00147991 | 0.000445986 | 0.0142919 | 0.00284603 | 0.995431 | 0.091283 | 0.35175 | 0.386788 | -0.000730997 | 0.000481747 | 29.7422 | 1.40005 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.144258 | 0.00365195 | 0.00116519 | 73.7454 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0381815 | 0.000955333 | 0.000129618 | 88.9436 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.124862 | 0.00329389 | 0.000944414 | 101.28 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.914332 | 0.00304564 | 0.00102383 | 23.6765 | 0.538133 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.942348 | 0.00272889 | 0.000903834 | 42.9408 | 54.959 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.895925 | 0.00263521 | 0.000879812 | -1.57708 | -0.0832131 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.202614 | 0.01 | 0.00274819 | 44.7519 | 2.30759 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.458412 | 0.00509227 | 0.00156394 | 119.052 | 8.3313 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.123173 | 0.0017862 | 0.000559677 | 23.3937 | 0.531706 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.209688 | 0.00121253 | 0.000335093 | 41.99 | 53.7421 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.123554 | 0.001441 | 0.000443188 | 23.8429 | 1.25804 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.112746 | 0.00577045 | 0.00142845 | 0.0380414 | 0.00759767 | 0.552785 | 0.0676476 | 0.44625 | NA | 0.00193689 | 0.0048856 | 411.032 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.502112 | 0.00265432 | 0.000904083 | 0.0209908 | 0.00601046 | 1.29059 | 0.301267 | 0.357323 | 0.471877 | -0.000784691 | 0.000916623 | 33.3459 | 1.56968 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.135014 | 0.01 | 0.00327564 | 0.0353674 | 0.00984544 | 0.60341 | 0.0810083 | 0.0163906 | NA | 0.000196793 | 0.00077373 | 86.2334 | 4.44654 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.231488 | 0.00647612 | 0.00174897 | 0.0499525 | 0.0122703 | 1.06922 | 0.138893 | 0.161211 | 0.513368 | 2.99363e-05 | 0.000129675 | 56.3735 | 3.94503 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00743586 | 0.000380884 | 9.43143e-05 | 0.00251073 | 0.000501345 | 0.0364798 | 0.00446152 | 0.446385 | NA | 7.9759e-06 | 2.28759e-05 | 1.7907 | 5.63028 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.03073 | 0.00175131 | 0.000353072 | 0.00648989 | 0.00185316 | 0.0720261 | 0.018438 | 0.59 | NA | 0.000291849 | 0.00116459 | 36.0619 | 32.7657 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.409045 | 0.05 | 0.014901 | 0.129261 | 0.0332138 | 2.52626 | 0.245427 | 0.228573 | NA | 0.0212095 | 0.0605318 | 12519.7 | 1.06626 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.147485 | 0.00644441 | 0.00165761 | 0.0499101 | 0.00946691 | 0.968998 | 0.0884911 | 0.387263 | NA | 0.0017326 | 0.00305435 | 401.329 | 0.419675 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.117065 | 0.00162182 | 0.000477085 | 0.0140662 | 0.00303524 | 0.659494 | 0.0702388 | 0.379698 | 0.420079 | -0.000789679 | 0.00109905 | 34.0381 | 1.60227 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.139673 | 0.00571757 | 0.00192918 | 207.891 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0931071 | 0.00572748 | 0.000631793 | 712.541 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105458 | 0.00586631 | 0.00172438 | 312.665 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.50188 | 0.0026772 | 0.00090462 | 24.4133 | 0.554879 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.503374 | 0.0025203 | 0.000852729 | 25.1923 | 32.243 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.501081 | 0.00276545 | 0.000954899 | 50.4322 | 2.661 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.135014 | 0.01 | 0.00327564 | 86.2334 | 4.44654 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.231488 | 0.00647612 | 0.00174897 | 56.3735 | 3.94503 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.118707 | 0.00163745 | 0.000478825 | 24.191 | 0.549826 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.121456 | 0.0014372 | 0.000400719 | 27.9595 | 35.7847 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.111031 | 0.00179083 | 0.000551711 | 49.9639 | 2.63629 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.070595 | 0.00326259 | 0.000928551 | 0.0220386 | 0.00460679 | 0.341849 | 0.042357 | 0.232628 | 0 | 0.000704374 | 0.0023928 | 166.281 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.538629 | 0.00262632 | 0.000879647 | 0.020927 | 0.00604885 | 1.35639 | 0.323177 | 0.345573 | 0.465424 | -0.000509399 | 0.000650375 | 20.0695 | 0.944729 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.142587 | 0.01 | 0.00318342 | 0.0368808 | 0.00992148 | 0.656092 | 0.0855525 | 0.0160156 | NA | 0.000146743 | 0.000702641 | 76.6387 | 3.9518 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.244229 | 0.0062663 | 0.00155056 | 0.0499525 | 0.0120551 | 1.17666 | 0.146537 | 0.154672 | 0.439344 | -3.81235e-06 | 5.45169e-05 | 47.418 | 3.31832 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00751481 | 0.000369614 | 9.13423e-05 | 0.00249183 | 0.000489734 | 0.0379561 | 0.00450889 | 0.441703 | NA | 6.80019e-06 | 2.3877e-05 | 1.68071 | 5.28447 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0310437 | 0.00169671 | 0.000344123 | 0.00630814 | 0.00179594 | 0.0730882 | 0.0186262 | 0.59 | NA | 0.000249644 | 0.00119843 | 33.6914 | 30.6119 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.429756 | 0.05 | 0.0147762 | 0.132046 | 0.0334722 | 2.70464 | 0.257854 | 0.22894 | NA | 0.0206579 | 0.0609226 | 12426.6 | 1.05833 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.150701 | 0.00626846 | 0.00159865 | 0.0499101 | 0.00928858 | 1.03323 | 0.0904204 | 0.381273 | NA | 0.00151779 | 0.00321106 | 376.194 | 0.39339 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.125991 | 0.00167124 | 0.000490519 | 0.0147868 | 0.00312349 | 0.717566 | 0.0755945 | 0.380271 | 0.428096 | -0.00081104 | 0.00120673 | 31.9757 | 1.50518 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.109391 | 0.00415378 | 0.00137779 | 129.034 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0237548 | 0.00139685 | 0.000156092 | 168.874 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0786392 | 0.00423715 | 0.00125177 | 200.933 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.549771 | 0.0028238 | 0.000934927 | 26.179 | 0.595012 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.551621 | 0.002587 | 0.000866342 | 33.5355 | 42.9213 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.514493 | 0.00246816 | 0.000837673 | 0.494109 | 0.0260711 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.142587 | 0.01 | 0.00318342 | 76.6387 | 3.9518 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.244229 | 0.0062663 | 0.00155056 | 47.418 | 3.31832 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.128155 | 0.00177792 | 0.00051732 | 24.6952 | 0.561286 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.131532 | 0.00141349 | 0.000389403 | 28.5644 | 36.559 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.118285 | 0.00182231 | 0.000564835 | 42.6674 | 2.2513 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0730204 | 0.00305585 | 0.00086715 | 0.021762 | 0.00437327 | 0.377043 | 0.0438123 | 0.226141 | 0 | 0.000567181 | 0.00187002 | 141.753 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.625543 | 0.00263423 | 0.000883382 | 0.0217802 | 0.00618004 | 1.55347 | 0.375326 | 0.344292 | 0.468701 | -0.000553468 | 0.000797294 | 24.5824 | 1.15716 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.154923 | 0.01 | 0.00305181 | 0.0389784 | 0.0100049 | 0.763285 | 0.092954 | 0.0163828 | 0.59 | 0.000116944 | 0.000488247 | 64.6143 | 3.33177 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.275082 | 0.00590047 | 0.00127999 | 0.0499525 | 0.0119083 | 1.42694 | 0.165049 | 0.142148 | 0.339922 | -3.4703e-06 | 8.44587e-05 | 43.1272 | 3.01805 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00776966 | 0.000346713 | 8.55263e-05 | 0.00245999 | 0.000464351 | 0.0422864 | 0.0046618 | 0.432885 | NA | 5.73267e-06 | 1.52293e-05 | 1.45719 | 4.58168 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0316531 | 0.00158533 | 0.000325669 | 0.00594966 | 0.001679 | 0.0761258 | 0.0189919 | 0.59 | NA | 0.000215611 | 0.00105116 | 29.1254 | 26.4633 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.465397 | 0.05 | 0.0146501 | 0.134628 | 0.0335031 | 3.03556 | 0.279238 | 0.228846 | NA | 0.0204698 | 0.0588277 | 12224.4 | 1.04111 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.160222 | 0.00592282 | 0.00148742 | 0.0499101 | 0.00891524 | 1.20022 | 0.096133 | 0.368365 | NA | 0.0012739 | 0.00300791 | 328.109 | 0.343107 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.143644 | 0.00164146 | 0.000481399 | 0.0158847 | 0.00310292 | 0.863073 | 0.0861866 | 0.371677 | 0.410196 | -0.000812322 | 0.00111797 | 35.1321 | 1.65376 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.111734 | 0.00391696 | 0.00128424 | 105.852 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0243877 | 0.00132274 | 0.000152215 | 149.802 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.0829396 | 0.00392784 | 0.001165 | 169.606 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.636683 | 0.00286349 | 0.000955322 | 29.8164 | 0.677685 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.64281 | 0.00258685 | 0.000863905 | 43.6358 | 55.8485 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.597137 | 0.00245236 | 0.00083092 | 0.294784 | 0.0155539 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.154923 | 0.01 | 0.00305181 | 64.6143 | 3.33177 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.275082 | 0.00590047 | 0.00127999 | 43.1272 | 3.01805 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.138078 | 0.0018258 | 0.00053861 | 27.5292 | 0.6257 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.158778 | 0.00137736 | 0.000374232 | 38.3203 | 49.0453 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.134076 | 0.00172123 | 0.000531354 | 39.5466 | 2.08663 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.09968 | 0.00265469 | 0.000752304 | 0.0211293 | 0.00394835 | 0.503441 | 0.059808 | 0.209148 | 0 | 0.000198649 | 0.000696245 | 85.4089 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.932454 | 0.00281714 | 0.000941737 | 0.0236338 | 0.00676772 | 2.41824 | 0.559472 | 0.350719 | 0.49046 | -0.000545926 | 0.000189397 | 23.9853 | 1.12905 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.204561 | 0.01 | 0.00276294 | 0.0447109 | 0.0102128 | 1.09292 | 0.122736 | 0.0164375 | 0.518903 | 9.17748e-05 | 0.000147687 | 45.8328 | 2.36332 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.467763 | 0.00510203 | 0.00157433 | 0.0594687 | 0.0145196 | 2.93175 | 0.280658 | 0.113398 | 0.230808 | 0.000370068 | 0.000961903 | 124.297 | 8.69836 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0111225 | 0.00028863 | 7.33672e-05 | 0.00237749 | 0.000409139 | 0.0620738 | 0.0066735 | 0.400138 | NA | 2.62537e-06 | 1.18972e-05 | 0.897793 | 2.82283 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.035082 | 0.00119379 | 0.000273824 | 0.00526252 | 0.0012751 | 0.11397 | 0.0210492 | 0.59 | NA | 0.000109417 | 0.000258633 | 15.3565 | 13.9529 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.560406 | 0.05 | 0.0148356 | 0.137573 | 0.0322125 | 2.76383 | 0.336244 | 0.228643 | NA | 0.0215038 | 0.0484392 | 11893.6 | 1.01294 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.262278 | 0.00533584 | 0.00130684 | 0.0499101 | 0.00850497 | 1.57546 | 0.157367 | 0.344326 | NA | 0.000510445 | 0.0021571 | 276.785 | 0.289437 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.154324 | 0.00148495 | 0.000444064 | 0.0144777 | 0.00286992 | 1.02733 | 0.0925944 | 0.353427 | 0.38732 | -0.000676621 | 0.000216215 | 28.233 | 1.329 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.139667 | 0.00368809 | 0.00117838 | 75.6193 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0377527 | 0.000963956 | 0.000130408 | 80.9291 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.12162 | 0.00331204 | 0.000948122 | 99.6783 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.927782 | 0.00303666 | 0.00102022 | 18.7907 | 0.427086 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.959847 | 0.00275559 | 0.000916221 | 53.0186 | 67.8573 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.909733 | 0.00265917 | 0.000888764 | 0.146578 | 0.00773401 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.204561 | 0.01 | 0.00276294 | 45.8328 | 2.36332 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.467763 | 0.00510203 | 0.00157433 | 124.297 | 8.69836 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.120367 | 0.00175276 | 0.000544789 | 16.9758 | 0.385836 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.219472 | 0.00126141 | 0.000344604 | 49.1623 | 62.9217 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.123133 | 0.00144069 | 0.000442801 | 18.5608 | 0.979339 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.133181 | 0.00354951 | 0.000895919 | 0.0317896 | 0.00514107 | 0.838689 | 0.0799089 | 0.392055 | NA | 0.0004363 | 0.00165752 | 141.213 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.373947 | 0.00194477 | 0.000644209 | 0.0166382 | 0.00467293 | 1.05532 | 0.224368 | 0.38601 | 0.450554 | -0.000728782 | 0.000446181 | 23.9202 | 1.12599 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.0943705 | 0.01 | 0.00392059 | 0.0264831 | 0.0089771 | 0.320211 | 0.0566223 | 0.0167578 | NA | 0.000635812 | 0.00250006 | 174.564 | 9.00119 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.177641 | 0.00728411 | 0.00273092 | 0.0499525 | 0.0123411 | 0.681061 | 0.106585 | 0.193062 | NA | 0.000561824 | 0.00237102 | 165.427 | 11.5766 | inflated_ratio |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00877759 | 0.000234288 | 5.9134e-05 | 0.00209807 | 0.000339169 | 0.055393 | 0.00526655 | 0.392495 | NA | 1.83525e-06 | 9.5629e-06 | 0.61496 | 1.93355 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0315269 | 0.00171122 | 0.000355628 | 0.00658242 | 0.00181502 | 0.0787744 | 0.0189161 | 0.59 | NA | 0.000265608 | 0.00122765 | 33.7123 | 30.6309 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.351306 | 0.05 | 0.0158365 | 0.115046 | 0.0305232 | 1.77977 | 0.210784 | 0.228615 | NA | 0.024686 | 0.0606735 | 13114.4 | 1.1169 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.153409 | 0.00666218 | 0.00181239 | 0.0499101 | 0.00957848 | 0.88792 | 0.0920454 | 0.388836 | NA | 0.00176534 | 0.00500406 | 426.862 | 0.446374 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0945124 | 0.00122591 | 0.000363481 | 0.0111808 | 0.00236712 | 0.404382 | 0.0567074 | 0.384 | 0.387809 | -0.000721708 | 0.000622281 | 24.4888 | 1.15275 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.14964 | 0.0039423 | 0.00126927 | 81.2995 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.113976 | 0.00333072 | 0.000466009 | 251.812 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.135928 | 0.00337552 | 0.000952478 | 90.5283 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.368109 | 0.00197729 | 0.000669144 | 20.874 | 0.474437 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.375402 | 0.00188305 | 0.000603341 | 16.3287 | 20.8988 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.378331 | 0.00197396 | 0.00066014 | 34.558 | 1.82341 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.0943705 | 0.01 | 0.00392059 | 174.564 | 9.00119 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.177641 | 0.00728411 | 0.00273092 | 165.427 | 11.5766 | inflated_ratio |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.0760363 | 0.0012742 | 0.000409757 | 20.8516 | 0.473927 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.107406 | 0.00112195 | 0.000291003 | 18.4612 | 23.6281 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.100095 | 0.00128157 | 0.000389684 | 34.1535 | 1.80207 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0860527 | 0.00191966 | 0.000538174 | 0.0182612 | 0.00293634 | 0.542316 | 0.0516316 | 0.195221 | 0 | 0.00012586 | 0.000180192 | 45.7804 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.454985 | 0.00198034 | 0.000656489 | 0.0171627 | 0.00485361 | 1.22225 | 0.272991 | 0.368771 | 0.443225 | -0.000639807 | 0.000324791 | 13.9353 | 0.655971 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.118361 | 0.01 | 0.00367515 | 0.0308113 | 0.00915849 | 0.403442 | 0.0710164 | 0.0157891 | NA | 0.000459454 | 0.00155691 | 128.631 | 6.63271 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.207475 | 0.00683861 | 0.00232003 | 0.0499525 | 0.011906 | 0.824753 | 0.124485 | 0.174484 | NA | 0.00028174 | 0.00103526 | 96.5152 | 6.75415 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00928921 | 0.000211395 | 5.25708e-05 | 0.00204434 | 0.000305314 | 0.0644146 | 0.00557353 | 0.384935 | NA | 1.31258e-06 | 3.22982e-06 | 0.47289 | 1.48685 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0309653 | 0.00164009 | 0.000340429 | 0.00624803 | 0.00173962 | 0.0790555 | 0.0185792 | 0.59 | NA | 0.000219295 | 0.00129015 | 30.7789 | 27.9656 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.39711 | 0.05 | 0.0155577 | 0.117834 | 0.0304047 | 2.09116 | 0.238266 | 0.229042 | NA | 0.0242801 | 0.0528077 | 12354 | 1.05214 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.164095 | 0.00632687 | 0.0017007 | 0.0499101 | 0.00913142 | 1.01625 | 0.0984569 | 0.378141 | NA | 0.00145451 | 0.00348771 | 364.041 | 0.380682 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.109822 | 0.00124245 | 0.000371249 | 0.011999 | 0.002383 | 0.483512 | 0.065893 | 0.380948 | 0.355088 | -0.000749851 | 0.000566048 | 19.9576 | 0.939458 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.118948 | 0.00268362 | 0.000855579 | 39.3692 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0279752 | 0.000769738 | 0.00011005 | 51.5516 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.111235 | 0.00230562 | 0.000648892 | 46.4203 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.458613 | 0.0021941 | 0.000744978 | 24.1014 | 0.547791 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.46552 | 0.00194788 | 0.000626866 | 17.0427 | 21.8126 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.440822 | 0.00179906 | 0.000597623 | 0.66167 | 0.0349123 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.118361 | 0.01 | 0.00367515 | 128.631 | 6.63271 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.207475 | 0.00683861 | 0.00232003 | 96.5152 | 6.75415 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.090569 | 0.0014451 | 0.000469792 | 21.5065 | 0.488813 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.124529 | 0.00103912 | 0.000266308 | 13.4775 | 17.2495 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.114367 | 0.00124314 | 0.000377645 | 24.8887 | 1.31323 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.0954333 | 0.00171444 | 0.000471724 | 0.0177308 | 0.00266649 | 0.641673 | 0.05726 | 0.186794 | 0 | 7.97287e-05 | 0.000115382 | 35.0424 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.546388 | 0.00215004 | 0.000723983 | 0.0190988 | 0.00523514 | 1.46561 | 0.327833 | 0.355344 | 0.455508 | -0.000661195 | 0.000375871 | 16.0581 | 0.755897 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.16632 | 0.01 | 0.00326911 | 0.0391251 | 0.00959604 | 0.576101 | 0.0997921 | 0.0162812 | NA | 0.000244078 | 0.000751671 | 80.9521 | 4.17421 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.277001 | 0.00630633 | 0.00165489 | 0.0499525 | 0.0117063 | 1.1231 | 0.166201 | 0.152023 | 0.434173 | 2.55434e-05 | 0.000197782 | 49.6562 | 3.47495 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0104139 | 0.000188028 | 4.61006e-05 | 0.00198086 | 0.000275796 | 0.0764127 | 0.00624836 | 0.37137 | NA | 9.21096e-07 | 2.58173e-06 | 0.370603 | 1.16524 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0314302 | 0.00161386 | 0.000332709 | 0.00604109 | 0.00170951 | 0.0822183 | 0.0188581 | 0.59 | NA | 0.000206447 | 0.00130774 | 29.9626 | 27.224 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.448221 | 0.05 | 0.0153129 | 0.122027 | 0.030358 | 2.45054 | 0.268933 | 0.228953 | NA | 0.0239736 | 0.0484411 | 11791 | 1.0042 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.178746 | 0.00611144 | 0.00162478 | 0.0499101 | 0.00883128 | 1.16343 | 0.107248 | 0.3715 | NA | 0.00132071 | 0.00299296 | 330.339 | 0.345439 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.136173 | 0.00128916 | 0.000392605 | 0.0135677 | 0.00246136 | 0.637099 | 0.0817037 | 0.380292 | 0.332084 | -0.000798707 | 0.000550514 | 21.1405 | 0.995143 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.130822 | 0.00242431 | 0.000743302 | 31.4856 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0317619 | 0.000664688 | 9.98488e-05 | 37.8586 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.123716 | 0.00205432 | 0.00057202 | 35.7829 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.554321 | 0.00246393 | 0.000843129 | 31.1542 | 0.708092 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.556971 | 0.00205441 | 0.000677102 | 16.2517 | 20.8001 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.527872 | 0.00193179 | 0.000651716 | 0.768247 | 0.0405357 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.16632 | 0.01 | 0.00326911 | 80.9521 | 4.17421 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.277001 | 0.00630633 | 0.00165489 | 49.6562 | 3.47495 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.11709 | 0.00168823 | 0.000558264 | 28.2736 | 0.642618 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.147448 | 0.000934714 | 0.000240448 | 13.1058 | 16.7738 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.143981 | 0.00124453 | 0.000379103 | 22.0422 | 1.16303 | none |
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
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.104008 | 0.00153756 | 0.000403119 | 0.0170723 | 0.00255794 | 0.723462 | 0.0624047 | 0.178523 | 0.0897917 | 1.36566e-07 | -8.0874e-05 | 25.9324 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.706703 | 0.00266147 | 0.000914022 | 0.0229848 | 0.00626254 | 1.98801 | 0.424022 | 0.33825 | 0.463571 | -0.000462302 | -0.000199436 | 10.08 | 0.474495 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.253373 | 0.01 | 0.00263254 | 0.0512799 | 0.0103111 | 0.95851 | 0.152024 | 0.0162266 | 0.467819 | 1.49159e-05 | 3.87246e-05 | 41.6063 | 2.14538 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.381907 | 0.00561304 | 0.00118215 | 0.0500293 | 0.0120958 | 1.71113 | 0.229144 | 0.130859 | 0.298564 | -4.40545e-05 | 5.80864e-05 | 50.898 | 3.56185 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0120642 | 0.000162125 | 3.88042e-05 | 0.00190006 | 0.000266758 | 0.0913736 | 0.00723853 | 0.346305 | NA | 2.98177e-07 | 4.34464e-06 | 0.321726 | 1.01157 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0353751 | 0.00151638 | 0.000327469 | 0.00580707 | 0.00160506 | 0.113535 | 0.0212251 | 0.59 | NA | 0.000172385 | 0.000335305 | 25.9946 | 23.6186 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.521356 | 0.05 | 0.0151155 | 0.126168 | 0.0306298 | 2.6112 | 0.312813 | 0.228836 | NA | 0.0232766 | 0.0397056 | 11368.2 | 0.968191 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.202558 | 0.00602446 | 0.0016058 | 0.0499101 | 0.00891675 | 1.2125 | 0.121535 | 0.365586 | NA | 0.0010333 | 0.0044711 | 328.863 | 0.343895 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.153259 | 0.0010917 | 0.000332965 | 0.0121946 | 0.0022196 | 0.77511 | 0.0919557 | 0.344052 | 0.331996 | -0.000579095 | 3.44301e-05 | 10.4017 | 0.489637 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.129169 | 0.00231433 | 0.000691901 | 22.7801 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0475405 | 0.000489887 | 7.31008e-05 | 28.5174 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.135314 | 0.00180846 | 0.000444356 | 26.4996 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.711139 | 0.00285911 | 0.00097823 | 4.72657 | 0.107428 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.719387 | 0.00258542 | 0.000887626 | 25.0665 | 32.0821 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.689583 | 0.00253987 | 0.000876211 | 0.447038 | 0.0235874 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.253373 | 0.01 | 0.00263254 | 41.6063 | 2.14538 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.381907 | 0.00561304 | 0.00118215 | 50.898 | 3.56185 | none |
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
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.131797 | 0.00145328 | 0.000458101 | 3.67012 | 0.0834166 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.174977 | 0.000747532 | 0.00020393 | 23.5803 | 30.1799 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.153004 | 0.00107428 | 0.000336865 | 3.95468 | 0.208664 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
