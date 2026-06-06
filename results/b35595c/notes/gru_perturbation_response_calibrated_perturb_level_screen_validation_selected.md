# GRU perturbation-response bank

Issue: `b35595c`. Source experiment: `b35595c`.

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

### `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.113539 | 0.00575215 | 0.00142161 | 0.0380047 | 0.00760624 | 0.5541 | 0.0681234 | 0.445549 | NA | 0.00176559 | 0.00519423 | 410.014 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.502905 | 0.00265343 | 0.000901877 | 0.0210629 | 0.00603505 | 1.29145 | 0.301743 | 0.358344 | 0.46972 | -0.000798545 | 0.000920586 | 31.0828 | 1.46315 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.135365 | 0.01 | 0.00326498 | 0.0354225 | 0.00986935 | 0.612008 | 0.0812191 | 0.0161797 | NA | 0.000171697 | 0.000883246 | 86.2506 | 4.44743 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.231915 | 0.00642355 | 0.00171688 | 0.0499525 | 0.0121858 | 1.0956 | 0.139149 | 0.160117 | 0.501551 | 2.34996e-05 | 0.000150194 | 54.9128 | 3.84281 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0074884 | 0.000379676 | 9.38612e-05 | 0.00250831 | 0.000501921 | 0.0365664 | 0.00449304 | 0.445562 | NA | 7.1254e-06 | 2.57936e-05 | 1.78628 | 5.61639 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0309692 | 0.00173788 | 0.000351702 | 0.00647024 | 0.00183926 | 0.0724819 | 0.0185815 | 0.59 | NA | 0.00026097 | 0.00121025 | 35.418 | 32.1807 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.409385 | 0.05 | 0.0148753 | 0.130175 | 0.0334221 | 2.52731 | 0.245631 | 0.228664 | NA | 0.0207588 | 0.0624935 | 12603.5 | 1.07339 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.148495 | 0.00642379 | 0.00164789 | 0.0499101 | 0.00947713 | 0.971064 | 0.0890973 | 0.386107 | 0.57 | 0.00159132 | 0.00345097 | 400.5 | 0.418807 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.117693 | 0.00162314 | 0.000475871 | 0.0141224 | 0.00304378 | 0.655379 | 0.0706156 | 0.378979 | 0.417554 | -0.000802825 | 0.00104633 | 30.7175 | 1.44596 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.14008 | 0.0056993 | 0.0019173 | 208.531 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.094052 | 0.00570841 | 0.000630917 | 707.738 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.106485 | 0.00584873 | 0.00171662 | 313.773 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.503371 | 0.002686 | 0.000905699 | 24.0123 | 0.545766 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.503908 | 0.00251792 | 0.000849877 | 24.199 | 30.9718 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.501434 | 0.00275636 | 0.000950056 | 45.0372 | 2.37634 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.135365 | 0.01 | 0.00326498 | 86.2506 | 4.44743 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.231915 | 0.00642355 | 0.00171688 | 54.9128 | 3.84281 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00923351 | 0.000376173 | 0.000126616 | 0.908388 | 8.75956 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00620501 | 0.000376774 | 4.1641e-05 | 3.08315 | 4.60707 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00702668 | 0.00038608 | 0.000113326 | 1.36729 | 7.54499 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0442947 | 0.0022541 | 0.000584846 | 40.8173 | 56.6931 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0137088 | 0.000950689 | 8.10172e-05 | 25.2938 | 14.5436 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.034904 | 0.00200883 | 0.000389242 | 40.1429 | 47.6389 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.523684 | 0.05 | 0.0174013 | 5809.05 | 1.1674 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.259517 | 0.05 | 0.0108936 | 20038.3 | 1.12165 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.444954 | 0.05 | 0.0163309 | 11963 | 0.966011 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.184738 | 0.00640589 | 0.00200395 | 160.017 | 0.483115 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.111407 | 0.00649272 | 0.00101458 | 727.944 | 0.446567 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.149342 | 0.00637275 | 0.00192513 | 313.538 | 0.345477 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.119995 | 0.00165192 | 0.000481696 | 22.7119 | 0.516209 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.122114 | 0.001439 | 0.00039932 | 26.058 | 33.3511 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.110969 | 0.00177849 | 0.000546597 | 43.3825 | 2.28903 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.114256 | 0.00560601 | 0.00138015 | 0.0377667 | 0.00745855 | 0.573934 | 0.0685538 | 0.441214 | NA | 0.00164604 | 0.00465801 | 387.422 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.546445 | 0.00276333 | 0.000926 | 0.0219493 | 0.00628677 | 1.37584 | 0.327867 | 0.354667 | 0.468056 | -0.000881755 | 0.00135028 | 35.0332 | 1.64911 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.141686 | 0.01 | 0.00318572 | 0.0367521 | 0.00994952 | 0.647828 | 0.0850119 | 0.0153906 | 0.59 | 0.000142137 | 0.000707418 | 77.9193 | 4.01783 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.24243 | 0.00629142 | 0.00157602 | 0.0499525 | 0.0120979 | 1.15948 | 0.145458 | 0.155859 | 0.452442 | 8.94572e-06 | 4.99092e-05 | 48.5022 | 3.39419 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00753374 | 0.000370025 | 9.11227e-05 | 0.0024926 | 0.000492163 | 0.0378766 | 0.00452024 | 0.441273 | NA | 6.62127e-06 | 2.38891e-05 | 1.68775 | 5.3066 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.031031 | 0.00169333 | 0.000343799 | 0.00631705 | 0.00179261 | 0.0731911 | 0.0186186 | 0.59 | NA | 0.000241247 | 0.0011765 | 33.5312 | 30.4664 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.42885 | 0.05 | 0.0147371 | 0.132747 | 0.0337324 | 2.69119 | 0.25731 | 0.228682 | NA | 0.0203342 | 0.0613798 | 12481.8 | 1.06303 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.150898 | 0.0062781 | 0.00159346 | 0.0499101 | 0.00934535 | 1.02867 | 0.0905389 | 0.381469 | 0.585 | 0.00148567 | 0.00321488 | 378.628 | 0.395935 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0657305 | 0.00130219 | 0.000399445 | 0.0102065 | 0.00247979 | 0.302294 | 0.0394383 | 0.324104 | 0.431917 | -0.000120356 | 0.000377033 | 7.86142 | 0.370058 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.142717 | 0.00555344 | 0.0018482 | 191.949 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.0943759 | 0.00560244 | 0.000625099 | 678.651 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.105676 | 0.00566217 | 0.00166716 | 291.665 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.547515 | 0.00283434 | 0.000933089 | 27.7081 | 0.629766 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.549736 | 0.00260936 | 0.000869396 | 29.6624 | 37.9642 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.542084 | 0.00284628 | 0.000975514 | 47.7291 | 2.51837 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.141686 | 0.01 | 0.00318572 | 77.9193 | 4.01783 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.24243 | 0.00629142 | 0.00157602 | 48.5022 | 3.39419 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00940776 | 0.000366545 | 0.000122048 | 0.836021 | 8.06173 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00622613 | 0.000369778 | 4.1257e-05 | 2.95641 | 4.41769 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00696733 | 0.000373751 | 0.000110063 | 1.27082 | 7.01262 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0441797 | 0.0021836 | 0.000569613 | 38.2006 | 53.0587 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0140137 | 0.000938865 | 8.0449e-05 | 24.5028 | 14.0888 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0348995 | 0.00195752 | 0.000381336 | 37.8902 | 44.9655 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.539905 | 0.05 | 0.0171866 | 5569.17 | 1.1192 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.278841 | 0.05 | 0.0108631 | 20173 | 1.12919 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.467805 | 0.05 | 0.0161616 | 11703.1 | 0.945025 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.187237 | 0.0062663 | 0.00192382 | 145.358 | 0.438856 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.110203 | 0.00634409 | 0.00100132 | 698.689 | 0.42862 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.155254 | 0.0062239 | 0.00185524 | 291.835 | 0.321564 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.127726 | 0.00177257 | 0.000512272 | 26.2286 | 0.596139 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0347329 | 0.00106701 | 0.000343032 | -1.32218 | -1.69223 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0347329 | 0.00106701 | 0.000343032 | -1.32218 | -0.0697635 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.117797 | 0.00525302 | 0.00129235 | 0.0372741 | 0.00705768 | 0.640669 | 0.0706783 | 0.432742 | NA | 0.00137322 | 0.00334261 | 335.068 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.636638 | 0.00275044 | 0.000921628 | 0.0227859 | 0.0063982 | 1.57904 | 0.381983 | 0.352719 | 0.4722 | -0.000895936 | 0.00130348 | 37.2243 | 1.75225 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.152628 | 0.01 | 0.00306911 | 0.0386241 | 0.00999661 | 0.748552 | 0.0915766 | 0.0163594 | 0.572514 | 0.000120186 | 0.000540408 | 66.3416 | 3.42084 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.270423 | 0.00593145 | 0.0012983 | 0.0499525 | 0.0118475 | 1.4034 | 0.162254 | 0.143023 | 0.349445 | -3.97531e-06 | 6.13594e-05 | 42.354 | 2.96394 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00776719 | 0.000346714 | 8.5324e-05 | 0.00246009 | 0.000465695 | 0.0422858 | 0.00466032 | 0.432771 | NA | 5.50354e-06 | 1.63708e-05 | 1.4596 | 4.58925 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0316508 | 0.0015811 | 0.000325156 | 0.00594966 | 0.00167473 | 0.0764349 | 0.0189905 | 0.59 | NA | 0.000204367 | 0.00105515 | 28.9521 | 26.3058 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.466004 | 0.05 | 0.0146226 | 0.135407 | 0.0336955 | 3.03074 | 0.279603 | 0.228714 | NA | 0.0201781 | 0.0595022 | 12274.7 | 1.0454 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.160019 | 0.00592557 | 0.00148531 | 0.0499101 | 0.00894512 | 1.19709 | 0.0960116 | 0.367948 | NA | 0.00123251 | 0.0030806 | 329.475 | 0.344536 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.07012 | 0.00127715 | 0.000391271 | 0.0104865 | 0.0024351 | 0.345895 | 0.042072 | 0.312938 | 0.412578 | -0.000162185 | 0.000370324 | 8.77673 | 0.413145 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.14616 | 0.00522976 | 0.0017206 | 156.815 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.097094 | 0.00529885 | 0.00060926 | 603.832 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.110137 | 0.00523044 | 0.00154718 | 244.557 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.636007 | 0.00285749 | 0.00094774 | 30.6818 | 0.697354 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.642159 | 0.00259481 | 0.000862313 | 38.6175 | 49.4256 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.631749 | 0.00279903 | 0.00095483 | 42.3737 | 2.2358 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.152628 | 0.01 | 0.00306911 | 66.3416 | 3.42084 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.270423 | 0.00593145 | 0.0012983 | 42.354 | 2.96394 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.00963628 | 0.000345176 | 0.000113616 | 0.683115 | 6.58726 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00640462 | 0.000349733 | 4.02112e-05 | 2.63033 | 3.93043 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00726068 | 0.000345233 | 0.000102145 | 1.06536 | 5.87884 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0444957 | 0.00202095 | 0.000535786 | 32.4952 | 45.1341 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0150561 | 0.00090598 | 7.89257e-05 | 22.3828 | 12.8698 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0354006 | 0.00181637 | 0.000360757 | 31.9783 | 37.9497 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.557837 | 0.05 | 0.0170507 | 5229.45 | 1.05092 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.335 | 0.05 | 0.0107976 | 20565.9 | 1.15118 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.505176 | 0.05 | 0.0160196 | 11028.9 | 0.890581 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.192439 | 0.00595033 | 0.0017829 | 115.13 | 0.347595 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.118325 | 0.00594027 | 0.000963465 | 632.293 | 0.387889 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.169294 | 0.00588611 | 0.00170957 | 241.002 | 0.265552 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.138051 | 0.00182379 | 0.000532959 | 28.8095 | 0.654799 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0361544 | 0.00100383 | 0.000320426 | -1.23964 | -1.58659 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0361544 | 0.00100383 | 0.000320426 | -1.23964 | -0.0654082 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.125865 | 0.00527982 | 0.00129666 | 0.0373377 | 0.00710599 | 0.638535 | 0.0755192 | 0.430867 | NA | 0.00127726 | 0.00335246 | 334.87 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.644437 | 0.0026225 | 0.00090671 | 0.020999 | 0.00595815 | 1.74177 | 0.386662 | 0.33826 | 0.462573 | -0.000819139 | 0.000658639 | 17.4487 | 0.821359 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.139033 | 0.01 | 0.00321322 | 0.0362218 | 0.00975086 | 0.659199 | 0.0834196 | 0.0154219 | NA | 0.000188211 | 0.000809597 | 76.3163 | 3.93517 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.270403 | 0.00595657 | 0.00131774 | 0.0499525 | 0.0117574 | 1.30537 | 0.162242 | 0.146031 | 0.356945 | -4.20426e-06 | 8.17696e-05 | 42.232 | 2.95541 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00829382 | 0.00034848 | 8.56288e-05 | 0.00246429 | 0.000468778 | 0.042134 | 0.00497629 | 0.430703 | NA | 5.21512e-06 | 1.82162e-05 | 1.45864 | 4.58623 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0325223 | 0.00156144 | 0.000324667 | 0.00600726 | 0.00165437 | 0.0818641 | 0.0195134 | 0.59 | NA | 0.000189554 | 0.0010496 | 28.1464 | 25.5738 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.405237 | 0.05 | 0.0150093 | 0.126989 | 0.0319627 | 2.5892 | 0.243142 | 0.228911 | NA | 0.0219185 | 0.0592401 | 12168 | 1.03631 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.171029 | 0.00599335 | 0.00149477 | 0.0499101 | 0.00903961 | 1.12842 | 0.102618 | 0.369234 | 0.5485 | 0.00117299 | 0.00423195 | 330.686 | 0.345802 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.125031 | 0.00153195 | 0.000459651 | 0.0141066 | 0.0029211 | 0.704944 | 0.0750185 | 0.35925 | 0.388871 | -0.000789382 | 0.000884156 | 20.6699 | 0.97299 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.157947 | 0.00532881 | 0.00174301 | 154.671 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.103378 | 0.00526665 | 0.000607982 | 600.034 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.116271 | 0.00524401 | 0.00153899 | 249.905 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.64525 | 0.00274424 | 0.000944395 | 18.8702 | 0.428892 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.642725 | 0.00246118 | 0.000850722 | 8.87961 | 11.3648 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.645336 | 0.00266207 | 0.000925012 | 24.5964 | 1.2978 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.139033 | 0.01 | 0.00321322 | 76.3163 | 3.93517 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.270403 | 0.00595657 | 0.00131774 | 42.232 | 2.95541 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0104029 | 0.000351709 | 0.000115149 | 0.67339 | 6.49349 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00681839 | 0.000347624 | 4.01274e-05 | 2.61397 | 3.90598 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.00766014 | 0.000346107 | 0.000101611 | 1.08856 | 6.00687 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0455142 | 0.00200298 | 0.000536763 | 31.9178 | 44.3322 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.015662 | 0.00090324 | 7.88004e-05 | 22.2434 | 12.7897 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0363906 | 0.00177809 | 0.000358437 | 30.278 | 35.9319 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.539927 | 0.05 | 0.0175176 | 5388.95 | 1.08298 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.239414 | 0.05 | 0.0108692 | 19792.1 | 1.10786 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.436371 | 0.05 | 0.016641 | 11323 | 0.914334 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.210863 | 0.00606227 | 0.00179231 | 109.743 | 0.331329 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.12362 | 0.00593218 | 0.000958858 | 638.457 | 0.39167 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.178605 | 0.0059856 | 0.00173316 | 243.859 | 0.2687 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.124929 | 0.00169922 | 0.000521715 | 21.1564 | 0.480854 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.124161 | 0.00131618 | 0.000371308 | 14.2099 | 18.1869 | inflated_ratio |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.126003 | 0.00158045 | 0.00048593 | 26.6435 | 1.40582 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.141812 | 0.00497107 | 0.00123525 | 0.0370972 | 0.00668787 | 0.746728 | 0.0850873 | 0.421065 | NA | 0.00100389 | 0.00212759 | 282.273 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.634969 | 0.00272011 | 0.000937473 | 0.021709 | 0.0060087 | 1.74999 | 0.380982 | 0.335146 | 0.440254 | -0.000904255 | 0.00041294 | 10.1925 | 0.47979 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.155089 | 0.01 | 0.00314457 | 0.0385282 | 0.00969452 | 0.666587 | 0.0930532 | 0.0151484 | NA | 0.000196581 | 0.00045382 | 66.0553 | 3.40607 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.318594 | 0.0057641 | 0.00126925 | 0.050302 | 0.0122427 | 1.49053 | 0.191157 | 0.138938 | 0.309844 | 7.60963e-06 | 0.000108327 | 49.972 | 3.49705 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.00934981 | 0.000328076 | 8.1581e-05 | 0.00244841 | 0.000441125 | 0.0492844 | 0.00560989 | 0.42107 | NA | 4.204e-06 | 1.15738e-05 | 1.22928 | 3.8651 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0333212 | 0.0014663 | 0.000309735 | 0.00575572 | 0.00155337 | 0.0913236 | 0.0199927 | 0.59 | NA | 0.000149873 | 0.000968368 | 24.443 | 22.2089 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.420295 | 0.05 | 0.0153115 | 0.118561 | 0.0296917 | 2.35154 | 0.252177 | 0.228948 | NA | 0.0243185 | 0.0420712 | 11699.9 | 0.996442 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.206864 | 0.00577845 | 0.00141915 | 0.0499101 | 0.00882873 | 1.30528 | 0.124118 | 0.358266 | 0.552143 | 0.000896875 | 0.00441216 | 304.981 | 0.318922 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.071781 | 0.00128089 | 0.000403862 | 0.010461 | 0.00235447 | 0.307759 | 0.0430686 | 0.303385 | 0.368681 | -0.000315273 | 0.000227441 | 4.24573 | 0.199858 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.17499 | 0.00515198 | 0.00166839 | 129.01 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.111882 | 0.00479317 | 0.000584543 | 512.754 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.138564 | 0.00496807 | 0.0014528 | 205.053 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.636842 | 0.002928 | 0.00101169 | 13.7775 | 0.313143 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.633565 | 0.00252418 | 0.000858851 | 4.8312 | 6.18334 | none |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.6345 | 0.00270814 | 0.000941874 | 11.9689 | 0.631524 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.155089 | 0.01 | 0.00314457 | 66.0553 | 3.40607 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.318594 | 0.0057641 | 0.00126925 | 49.972 | 3.49705 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0115231 | 0.000340023 | 0.000110244 | 0.561442 | 5.41397 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0073878 | 0.000316317 | 3.85776e-05 | 2.23346 | 3.33741 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0091385 | 0.000327888 | 9.59211e-05 | 0.892947 | 4.92745 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0455422 | 0.0018933 | 0.000512 | 28.4641 | 39.5351 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0178868 | 0.000849352 | 7.65289e-05 | 19.1291 | 10.9989 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0365345 | 0.00165626 | 0.000340675 | 25.7359 | 30.5416 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.597203 | 0.05 | 0.0177911 | 5202.86 | 1.04558 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.215533 | 0.05 | 0.0109855 | 18794.2 | 1.05201 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.448149 | 0.05 | 0.0171578 | 11102.6 | 0.896535 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.244399 | 0.00595075 | 0.00169646 | 95.2381 | 0.287537 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.1641 | 0.00556165 | 0.000908002 | 612.443 | 0.375711 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.212093 | 0.00582295 | 0.00165298 | 207.262 | 0.228375 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.126804 | 0.00190058 | 0.00060804 | 13.7197 | 0.31183 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0442694 | 0.000971039 | 0.000301773 | -0.491261 | -0.628755 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0442694 | 0.000971039 | 0.000301773 | -0.491261 | -0.0259209 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

### `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64`

- Evaluated: 74
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.181611 | 0.00440856 | 0.00113551 | 0.0364749 | 0.00602275 | 1.02363 | 0.108966 | 0.401279 | NA | 0.0010103 | 0.00107591 | 196.473 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (12) |
| `delayed_observation/delayed_observation_offset` | 3 | evaluated=3 | 0.01 | 0.647775 | 0.00273607 | 0.00094981 | 0.0224048 | 0.0061312 | 1.98803 | 0.388665 | 0.328156 | 0.453394 | -0.000788336 | 0.000508679 | 21.777 | 1.0251 | none |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.189761 | 0.01 | 0.00297097 | 0.0420968 | 0.00981883 | 0.728419 | 0.113857 | 0.0156563 | 0.59 | 0.000212231 | 0.000107238 | 53.1582 | 2.74105 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.372139 | 0.00547512 | 0.00121702 | 0.0501026 | 0.0122289 | 1.7691 | 0.223283 | 0.128312 | 0.273938 | 3.03683e-05 | 5.6214e-05 | 54.5464 | 3.81717 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0121288 | 0.000290269 | 7.48858e-05 | 0.00240734 | 0.000397567 | 0.0679496 | 0.00727727 | 0.399602 | NA | 4.11837e-06 | 6.01831e-06 | 0.853634 | 2.68398 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0329295 | 0.00131723 | 0.000288571 | 0.00530942 | 0.00140003 | 0.126912 | 0.0197577 | 0.59 | NA | 0.000166309 | 0.000792589 | 18.9445 | 17.2129 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 0.557422 | 0.05 | 0.0153901 | 0.116701 | 0.0281376 | 2.22656 | 0.334453 | 0.229237 | NA | 0.0273003 | 0.029189 | 10914.5 | 0.929548 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.28834 | 0.00547776 | 0.00130864 | 0.0499101 | 0.00869343 | 1.61032 | 0.173004 | 0.346844 | 0.495 | 0.000791908 | 0.00245324 | 289.516 | 0.30275 | none |
| `sensory_feedback/sensory_feedback_offset` | 3 | evaluated=3 | 0.01 | 0.0775993 | 0.00126894 | 0.000395988 | 0.0104621 | 0.00232676 | 0.284438 | 0.0465596 | 0.296948 | 0.348801 | -0.000344377 | 0.000123741 | 7.85121 | 0.369578 | none |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.215863 | 0.00496867 | 0.00156509 | 99.9887 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.142613 | 0.00376782 | 0.000516387 | 342.838 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.186356 | 0.0044892 | 0.00132504 | 146.591 | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; command_input rows require the graph/channel adapter lane to define a matching analytical command-port intervention (4) |
| `delayed_observation/delayed_observation_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.653622 | 0.00297614 | 0.00103482 | 29.6446 | 0.67378 | none |
| `delayed_observation/delayed_observation_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.639247 | 0.00251502 | 0.000860465 | 9.03297 | 11.5611 | inflated_ratio |
| `delayed_observation/delayed_observation_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.650455 | 0.00271704 | 0.000954142 | 26.6533 | 1.40633 | none |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.189761 | 0.01 | 0.00297097 | 53.1582 | 2.74105 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.372139 | 0.00547512 | 0.00121702 | 54.5464 | 3.81717 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0142253 | 0.00032792 | 0.000103407 | 0.43484 | 4.19315 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00980438 | 0.00024674 | 3.38809e-05 | 1.48883 | 2.22473 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0123567 | 0.000296147 | 8.73698e-05 | 0.637228 | 3.51634 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0449186 | 0.00173457 | 0.000478488 | 23.7478 | 32.9845 | inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0174087 | 0.000729036 | 6.99602e-05 | 12.9526 | 7.44755 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0364611 | 0.00148807 | 0.000317263 | 20.1331 | 23.8926 | inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 0.813791 | 0.05 | 0.0177512 | 4827.74 | 0.970195 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.145915 | 0.05 | 0.0111751 | 17722.6 | 0.992024 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 0.71256 | 0.05 | 0.0172441 | 10193 | 0.823087 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.276143 | 0.00590535 | 0.00162113 | 82.2464 | 0.248313 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.301689 | 0.00501084 | 0.000798557 | 624.189 | 0.382917 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.287188 | 0.00551709 | 0.00150623 | 162.113 | 0.178627 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 1 | evaluated=1 | 0.01 | 0.131022 | 0.00192009 | 0.000614164 | 25.0131 | 0.568511 | none |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 1 | evaluated=1 | 0.01 | 0.0508878 | 0.000943359 | 0.0002869 | -0.729724 | -0.933958 | none |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 1 | evaluated=1 | 0.01 | 0.0508878 | 0.000943359 | 0.0002869 | -0.729724 | -0.0385031 | none |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, process_epsilon, sensory_feedback, and delayed_observation. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Command-input rows still require a separate analytical command-port intervention, and target-stream is deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
