# GRU perturbation-response bank

Issue: `e4800d6`. Source experiment: `e4800d6`.

The bank is controller-independent: it perturbs external task, command-port, process, sensory, observation, or target interfaces and does not mutate GRU internals.

v2 splits the former plant_force rows into command_input_pulse (post-controller command-port perturbations) and process_epsilon_pulse (mechanics.epsilon / B_w process perturbations). Process-epsilon rows span the canonical current physical block [px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]. v3 timing-aware rows evaluate plant-side command/process pulses at early/mid/late bins and controller-visible sensory/pre-noise delayed-measurement offsets at early_visible/mid_visible/late_visible bins.

## Bank

| Channel | Count |
|---|---:|
| `command_input` | 18 |
| `delayed_observation` | 36 |
| `initial_state` | 8 |
| `process_epsilon` | 48 |
| `sensory_feedback` | 36 |
| `target_stream` | 1 |

| Family | Count |
|---|---:|
| `command_input_pulse` | 12 |
| `delayed_observation_offset` | 36 |
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

### `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64`

- Evaluated: 146
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.223902 | 0.00287346 | 0.000516373 | 0.0307572 | 0.00500559 | 1.21794 | 0.134341 | 0.358115 | 0.424954 | 0.000588913 | 0.000121498 | 95.1562 | 1.30327 | 1.25785 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.231923 | 0.00280767 | 0.000493402 | 0.0306131 | 0.00504846 | 1.26174 | 0.139154 | 0.354484 | 0.410191 | 0.000581288 | 0.000128518 | 95.5254 | 1.30832 | 1.26273 | none |
| `delayed_observation/delayed_observation_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.215666 | 0.00175983 | 0.000304812 | 0.0203929 | 0.00286042 | 0.999234 | 0.1294 | 0.39797 | 0.456728 | 0.000349976 | 6.63207e-05 | 76.3436 | 0.102751 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.201381 | 0.01 | 0.00282319 | 0.0441601 | 0.010723 | 0.570454 | 0.120829 | 0.0163281 | 0.41793 | 0.000119747 | 7.28331e-05 | 59.3329 | 1.59121 | 1.33557 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.330884 | 0.00635876 | 0.00146637 | 0.0500256 | 0.0131546 | 1.07159 | 0.19853 | 0.15468 | 0.347836 | 0.000134601 | 7.63543e-05 | 58.3566 | 1.59346 | 1.25879 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.014678 | 0.000189884 | 3.41071e-05 | 0.00202913 | 0.000330001 | 0.0806533 | 0.00880679 | 0.359203 | 0.404406 | 3.22262e-06 | 3.83834e-07 | 0.413754 | 1.30092 | 1.25558 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0356954 | 0.00132427 | 0.00025875 | 0.00579658 | 0.00146931 | 0.123724 | 0.0214172 | 0.59 | NA | 0.000515057 | 0.00200501 | 24.1524 | 21.9448 | 22.0199 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 1.11978 | 0.05 | 0.0126331 | 0.208433 | 0.0377599 | 4.19225 | 0.67187 | 0.228729 | 0.50626 | 0.0183511 | 0.0164514 | 10783 | 0.91835 | 1.21673 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.322737 | 0.00575576 | 0.00110475 | 0.0513231 | 0.0100868 | 1.48932 | 0.193642 | 0.356109 | 0.423209 | 0.00165533 | 0.00066454 | 325.308 | 0.340178 | 1.44101 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.215666 | 0.00175983 | 0.000304812 | 0.0203929 | 0.00286042 | 0.999234 | 0.1294 | 0.39797 | 0.456728 | 0.000349976 | 6.63207e-05 | 76.3436 | 0.102751 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.269807 | 0.00311343 | 0.000603604 | 44.0047 | 1.84841 | 1.63449 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.141411 | 0.00268016 | 0.000401946 | 175.095 | 1.1397 | 1.14177 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.260489 | 0.0028268 | 0.000543569 | 66.3689 | 1.59533 | 1.422 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.278804 | 0.00299797 | 0.000594636 | 44.8271 | 1.88295 | 1.66503 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.142755 | 0.00266668 | 0.000400666 | 173.341 | 1.12829 | 1.13033 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.274211 | 0.00275836 | 0.000484904 | 68.4079 | 1.64434 | 1.46569 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.206543 | 0.00183547 | 0.000339847 | 30.4635 | 0.019976 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.171527 | 0.00137241 | 0.000153762 | 124.369 | 3.52412 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.268928 | 0.00207159 | 0.000420827 | 74.1986 | 0.110961 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.201381 | 0.01 | 0.00282319 | 59.3329 | 1.59121 | 1.33557 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.330884 | 0.00635876 | 0.00146637 | 58.3566 | 1.59346 | 1.25879 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.0178268 | 0.000205425 | 3.97673e-05 | 0.191862 | 1.85012 | 1.636 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.0090417 | 0.000177791 | 2.66715e-05 | 0.762202 | 1.13894 | 1.141 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0171654 | 0.000186437 | 3.58823e-05 | 0.287199 | 1.58482 | 1.41263 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.054147 | 0.00153047 | 0.000397721 | 24.8224 | 34.477 | 34.2819 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0150056 | 0.000891253 | 7.81406e-05 | 21.4942 | 12.3588 | 12.4367 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0379337 | 0.00155109 | 0.000300387 | 26.1405 | 31.0218 | 31.1873 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 1.31557 | 0.05 | 0.0135188 | 3769.35 | 0.757498 | 1.57789 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.493115 | 0.05 | 0.0108977 | 18923.8 | 1.05926 | 1.11512 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 1.55067 | 0.05 | 0.0134827 | 9655.79 | 0.779703 | 1.33593 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.389787 | 0.00599393 | 0.00129155 | 136.614 | 0.412458 | 2.1444 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.158267 | 0.00568398 | 0.000935567 | 592.42 | 0.363428 | 1.23325 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.420156 | 0.00558936 | 0.00108714 | 246.89 | 0.272039 | 1.85399 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.206543 | 0.00183547 | 0.000339847 | 30.4635 | 0.019976 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.171527 | 0.00137241 | 0.000153762 | 124.369 | 3.52412 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.268928 | 0.00207159 | 0.000420827 | 74.1986 | 0.110961 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

### `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64`

- Evaluated: 146
- Blocked: 0
- Not implemented: 0
- Not applicable: 1
- Rollout trials per replicate: 64
- Robust summaries: available

#### Class-Binned Summary

| Class | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Max delta state | AUC delta state | Max delta u | AUC delta u | Peak time | Recovery time | Mean endpoint delta | Mean terminal-speed delta | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 12 | evaluated=12 | 1 | 0.244053 | 0.00265292 | 0.000429435 | 0.0301497 | 0.00452086 | 1.40807 | 0.146432 | 0.349133 | 0.430246 | 0.000414196 | 3.76049e-05 | 80.6752 | 1.10493 | 1.06643 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 6 | evaluated=6 | 1 | 0.246774 | 0.00262656 | 0.00041575 | 0.0300343 | 0.0044932 | 1.41192 | 0.148065 | 0.348812 | 0.411202 | 0.000419535 | 4.37359e-05 | 80.5065 | 1.10262 | 1.0642 | none |
| `delayed_observation/delayed_observation_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.292025 | 0.00180109 | 0.000265663 | 0.0240286 | 0.00302756 | 1.46946 | 0.175215 | 0.382097 | 0.467656 | 0.000243736 | 0.000161136 | 93.4697 | 0.125801 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (36) |
| `initial_state/initial_position_offset` | 4 | evaluated=4 | 0.01 | 0.258812 | 0.01 | 0.00245725 | 0.0525507 | 0.011097 | 0.762293 | 0.155287 | 0.0151719 | 0.339602 | 1.6012e-06 | 2.63197e-06 | 48.2598 | 1.29425 | 1.08632 | none |
| `initial_state/initial_velocity_offset` | 4 | evaluated=4 | 0.05 | 0.333357 | 0.00621492 | 0.0013165 | 0.0499525 | 0.0122844 | 1.10307 | 0.200014 | 0.15093 | 0.342547 | -8.72668e-07 | -7.66597e-07 | 44.9618 | 1.22771 | 0.969856 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 12 | evaluated=12 | 0.01 | 0.0160396 | 0.000175116 | 2.83087e-05 | 0.00198974 | 0.000298154 | 0.0932497 | 0.00962373 | 0.349813 | 0.429088 | 2.19759e-06 | 2.09015e-07 | 0.351174 | 1.10416 | 1.06568 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 12 | evaluated=12 | 0.01 | 0.0336257 | 0.00127973 | 0.000241502 | 0.00582844 | 0.00140886 | 0.130693 | 0.0201754 | 0.59 | NA | 0.000431269 | 0.002597 | 23.792 | 21.6173 | 21.6913 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 12 | evaluated=12 | 0.01 | 1.52827 | 0.05 | 0.0113528 | 0.252486 | 0.0393125 | 6.20414 | 0.916962 | 0.228956 | 0.441658 | 0.0157831 | 0.00477753 | 10013.8 | 0.852842 | 1.12994 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 12 | evaluated=12 | 0.01 | 0.34957 | 0.00549588 | 0.000979007 | 0.0503292 | 0.00923585 | 1.76469 | 0.209742 | 0.349245 | 0.434219 | 0.00135421 | 0.000121005 | 274.051 | 0.286578 | 1.21396 | none |
| `sensory_feedback/sensory_feedback_offset` | 36 | evaluated=36 | 0.01, 0.05, 0.1 | 0.292025 | 0.00180109 | 0.000265663 | 0.0240286 | 0.00302756 | 1.46946 | 0.175215 | 0.382097 | 0.467656 | 0.000243736 | 0.000161136 | 93.4697 | 0.125801 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (36) |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

#### Timing-Cell Summary

| Cell | Rows | Status | Amplitudes | Mean delta action | Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | Warnings / not applicable |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 4 | evaluated=4 | 1 | 0.285873 | 0.00296779 | 0.000494871 | 32.4957 | 1.36497 | 1.207 | none |
| `command_input/command_input_pulse/late` | 4 | evaluated=4 | 1 | 0.151548 | 0.00247632 | 0.000378795 | 153.661 | 1.00019 | 1.002 | none |
| `command_input/command_input_pulse/mid` | 4 | evaluated=4 | 1 | 0.294737 | 0.00251465 | 0.000414639 | 55.8688 | 1.34293 | 1.19703 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 2 | evaluated=2 | 1 | 0.293061 | 0.00290907 | 0.000467297 | 32.3283 | 1.35794 | 1.20078 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 2 | evaluated=2 | 1 | 0.148287 | 0.00248362 | 0.000379901 | 153.17 | 0.99699 | 0.998796 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 2 | evaluated=2 | 1 | 0.298975 | 0.00248699 | 0.000400052 | 56.0216 | 1.3466 | 1.2003 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.24862 | 0.00176683 | 0.00029996 | 26.6703 | 0.0174887 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.225897 | 0.00154773 | 0.000180331 | 158.677 | 4.49627 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `delayed_observation/delayed_observation_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.40156 | 0.0020887 | 0.000316698 | 95.062 | 0.142162 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose clean delayed-measurement offset ports; extLQG carries this row today (12) |
| `initial_state/initial_position_offset/initial_condition` | 4 | evaluated=4 | 0.01 | 0.258812 | 0.01 | 0.00245725 | 48.2598 | 1.29425 | 1.08632 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 4 | evaluated=4 | 0.05 | 0.333357 | 0.00621492 | 0.0013165 | 44.9618 | 1.22771 | 0.969856 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 4 | evaluated=4 | 0.01 | 0.018837 | 0.000195812 | 3.26573e-05 | 0.141962 | 1.36894 | 1.21051 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 4 | evaluated=4 | 0.01 | 0.00986757 | 0.000163752 | 2.50646e-05 | 0.668838 | 0.999427 | 1.00124 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0194141 | 0.000165785 | 2.72043e-05 | 0.242722 | 1.33939 | 1.19387 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 4 | evaluated=4 | 0.01 | 0.0491913 | 0.00150168 | 0.000373484 | 25.3917 | 35.2679 | 35.0682 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 4 | evaluated=4 | 0.01 | 0.0141997 | 0.000880601 | 7.73666e-05 | 20.8636 | 11.9963 | 12.0718 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 4 | evaluated=4 | 0.01 | 0.0374861 | 0.00145692 | 0.000273655 | 25.1205 | 29.8113 | 29.9703 | inflated_ratio; inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 4 | evaluated=4 | 0.01 | 1.82018 | 0.05 | 0.0116832 | 3247.33 | 0.652593 | 1.35937 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 4 | evaluated=4 | 0.01 | 0.577789 | 0.05 | 0.0109042 | 17777.1 | 0.995074 | 1.04755 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 4 | evaluated=4 | 0.01 | 2.18684 | 0.05 | 0.0114709 | 9017.01 | 0.728122 | 1.24755 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 4 | evaluated=4 | 0.01 | 0.419556 | 0.00582357 | 0.00109033 | 91.8627 | 0.277346 | 1.44194 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 4 | evaluated=4 | 0.01 | 0.172927 | 0.00543142 | 0.000909837 | 545.354 | 0.334555 | 1.13527 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 4 | evaluated=4 | 0.01 | 0.456228 | 0.00523264 | 0.000936854 | 184.937 | 0.203776 | 1.38877 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.24862 | 0.00176683 | 0.00029996 | 26.6703 | 0.0174887 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.225897 | 0.00154773 | 0.000180331 | 158.677 | 4.49627 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 12 | evaluated=12 | 0.01, 0.05, 0.1 | 0.40156 | 0.0020887 | 0.000316698 | 95.062 | 0.142162 | NA | no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; robust output-feedback released-forward replay does not yet expose post-noise measurement-offset ports; extLQG carries this row today (12) |
| `target_stream/target_stream_jump/not_applicable` | 1 | not_applicable=1 | 0.01 | NA | NA | NA | NA | NA | NA | no meaningful extLQG full-Q/R/Q_f denominator for this channel/family; no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family; target_stream is deferred: current fixed-target C&S GRU validation checkpoints do not consume a target-position input stream (1); target_stream rows are deferred for current fixed-target checkpoints without a controller-visible target input stream (1); fixed-target checkpoints do not expose a target stream, and the robust analytical comparator has no target-stream intervention (1) |

## Residuals

- ExtLQG comparator: available_for_initial_state_command_input_process_epsilon_sensory_feedback_and_delayed_observation - Deterministic extLQG response rows are evaluated for perturbations with clean analytical interfaces: initial_state, command_input, process_epsilon, sensory_feedback, and delayed_observation. Command-input rows add an external pulse after the controller command and before the plant input. Sensory-feedback rows offset the post-noise measurement delivered to the estimator; delayed-observation rows offset the clean delayed measurement before sensory noise. Target-stream remains deferred for current fixed-target checkpoints.
- Full-Q/R/Q_f perturbation cost: available - Costs are rescored post hoc from states.mechanics.vector and states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. They are audit-only perturbation diagnostics and are not used for checkpoint selection.
