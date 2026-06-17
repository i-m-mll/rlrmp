# SISU Perturbation-Class Robustification Comparison

Issue: `e4800d6`. Source experiment: `e4800d6`.

This report compares SISU=1 against SISU=0 within each trained targetfix model on the calibrated 020a65b-style perturbation bank. It is discovery-trained robustness evidence, not teacher/distillation behavior, not trial-history adaptation, and not a formal H-infinity equivalence claim.

## Interpretation

- For response magnitudes and full-Q/R/Qf delta cost, lower is better; a SISU1/SISU0 ratio below 1 is an improvement.
- Endpoint and terminal-speed deltas are signed diagnostics; the SISU1-SISU0 difference shows direction, and values closer to zero are usually preferable.
- Existing targetfix perturbation summaries were sufficient for the SISU=1 side only, but this materialization reran both SISU=0 and SISU=1 locally through the same evaluator for a paired comparison. No remote training and no raw rollout arrays were written.

## Bank

- Bank id: `cs_calibrated_perturbation_response_v3`
- Bank mode: `calibrated`
- Perturbation rows: 439
- Rollout trials per replicate: 64

## raw strong gamma-1.05 targetfix

Run: `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64`

### Metric Glossary

- Ratios are `SISU=1 / SISU=0`; values below 1 mean the high-SISU condition had the smaller perturbation response.
- `Mean delta action ratio`: mean command-change norm under perturbation.
- `Max delta x ratio`: peak hand-position response magnitude in meters.
- `AUC delta x ratio`: time-integrated hand-position response magnitude.
- `Cost SISU=0`, `Cost SISU=1`, `Cost ratio`, and `Cost diff`: post-hoc full-Q/R/Q_f perturbation delta cost, with `diff = SISU1 - SISU0`.
- Signed diagnostics are separated because endpoint and terminal-speed deltas are directional sidecars, not simple lower-is-better ratios.

### Headline

| Metric | Class groups with ratio < 1 | ratio = 1 | ratio > 1 | unavailable |
|---|---:|---:|---:|---:|
| Full-Q/R/Qf delta cost | 6 | 0 | 4 | 1 |
| Max delta x | 6 | 2 | 2 | 1 |
| Mean delta action | 0 | 0 | 10 | 1 |

### Class-Binned Summary

| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | Cost diff | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 1.23466 | 0.944966 | 0.831612 | 702.14 | 682.582 | 0.972146 | -19.5576 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 1.26441 | 0.938873 | 0.844292 | 702.074 | 686.989 | 0.978514 | -15.0845 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 1.4675 | 1.1127 | 0.896614 | 475.156 | 737.926 | 1.55302 | 262.77 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 1.14407 | 1 | 0.919937 | 353.957 | 331.525 | 0.936624 | -22.4322 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 1.14464 | 0.991899 | 0.962353 | 32.9283 | 38.6112 | 1.17258 | 5.6829 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 1.23447 | 0.945085 | 0.831796 | 702.399 | 682.961 | 0.972326 | -19.4384 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 1.08426 | 0.912778 | 0.882378 | 3537.64 | 3361.8 | 0.950292 | -175.847 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 1.51686 | 1 | 0.894223 | 2280.5 | 2332.12 | 1.02263 | 51.6119 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 1.32048 | 0.965705 | 0.856598 | 1139.53 | 1138.93 | 0.999473 | -0.60104 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 1.4675 | 1.1127 | 0.896614 | 475.156 | 737.926 | 1.55302 | 262.77 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | NA | NA | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

#### Signed Diagnostics

| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | terminal diff |
|---|---:|---:|---:|---:|---:|---:|
| `command_input/command_input_pulse` | 0.00143757 | 0.00166697 | 0.000229398 | 0.0044814 | 0.00279902 | -0.00168239 |
| `command_input/target_aligned_lateral_command_load_pulse` | 0.00155605 | 0.00162122 | 6.51679e-05 | 0.00462398 | 0.00311609 | -0.00150789 |
| `delayed_observation/delayed_observation_offset` | 0.00110429 | 0.00145837 | 0.000354076 | 0.00231431 | 0.00419249 | 0.00187818 |
| `initial_state/initial_position_offset` | 0.00100101 | 0.000582097 | -0.000418914 | 9.45585e-05 | 0.000308754 | 0.000214196 |
| `initial_state/initial_velocity_offset` | 2.50926e-05 | 8.59744e-05 | 6.08818e-05 | 2.14064e-05 | 5.02957e-05 | 2.88893e-05 |
| `process_epsilon/process_epsilon_force_state_xy` | 0.0014538 | 0.0016653 | 0.000211503 | 0.00446537 | 0.00280862 | -0.00165675 |
| `process_epsilon/process_epsilon_integrator_xy` | 0.00741358 | 0.00799564 | 0.000582065 | 0.0391035 | 0.0402998 | 0.00119625 |
| `process_epsilon/process_epsilon_position_xy` | 0.00766817 | 0.00690001 | -0.000768161 | 0.00410788 | 0.00277592 | -0.00133197 |
| `process_epsilon/process_epsilon_velocity_xy` | 0.00271759 | 0.00291624 | 0.000198648 | 0.00562834 | 0.00425143 | -0.0013769 |
| `sensory_feedback/sensory_feedback_offset` | 0.00110429 | 0.00145837 | 0.000354076 | 0.00231431 | 0.00419249 | 0.00187818 |
| `target_stream/target_stream_jump` | NA | NA | NA | NA | NA | NA |

### Timing-Cell Summary

| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | Full-Q/R/Qf cost ratio | cost diff | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | 1.30586 | 0.963745 | 0.803026 | 1.35815 | 13.2216 | none |
| `command_input/command_input_pulse/late` | 12 | 1.1243 | 0.936981 | 0.944875 | 0.955214 | -87.8464 | none |
| `command_input/command_input_pulse/mid` | 12 | 1.38119 | 0.946002 | 0.716764 | 1.14765 | 15.9522 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | 1.31344 | 0.963807 | 0.911164 | 1.47685 | 16.4412 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | 1.15028 | 0.92929 | 0.936341 | 0.956895 | -84.785 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | 1.42844 | 0.938988 | 0.678194 | 1.22033 | 23.0903 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | 1.36428 | 1.02104 | 0.747776 | 1.29127 | 50.5458 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | 1.43377 | 1.17664 | 1.19147 | 1.50601 | 480.371 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | 1.58681 | 1.15816 | 0.962196 | 1.85062 | 257.394 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset/initial_condition` | 12 | 1.14407 | 1 | 0.919937 | 0.936624 | -22.4322 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | 1.14464 | 0.991899 | 0.962353 | 1.17258 | 5.6829 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | 1.30695 | 0.963759 | 0.801175 | 1.36122 | 13.2956 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | 1.12343 | 0.937257 | 0.945202 | 0.95542 | -87.4956 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | 1.38144 | 0.945872 | 0.718525 | 1.14744 | 15.8847 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | 1.25871 | 0.776906 | 0.799274 | 0.725385 | -123.394 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | 0.97006 | 0.991529 | 0.992849 | 0.977595 | -205.608 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | 1.14108 | 0.864732 | 0.867815 | 0.798816 | -198.54 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | 1.3528 | 1 | 0.86169 | 1.03192 | 25.793 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | 1.64758 | 1 | 0.991032 | 1.00439 | 17.6446 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | 1.64411 | 1 | 0.856885 | 1.0552 | 111.398 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | 1.35262 | 0.982719 | 0.860852 | 1.46297 | 37.525 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | 1.12232 | 0.96145 | 0.969949 | 0.966411 | -103.488 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | 1.53416 | 0.959305 | 0.721112 | 1.25011 | 64.1594 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | 1.36428 | 1.02104 | 0.747776 | 1.29127 | 50.5458 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | 1.43377 | 1.17664 | 1.19147 | 1.50601 | 480.371 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | 1.58681 | 1.15816 | 0.962196 | 1.85062 | 257.394 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump/not_applicable` | 1 | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

## effective 020a65b PGD targetfix

Run: `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64`

### Metric Glossary

- Ratios are `SISU=1 / SISU=0`; values below 1 mean the high-SISU condition had the smaller perturbation response.
- `Mean delta action ratio`: mean command-change norm under perturbation.
- `Max delta x ratio`: peak hand-position response magnitude in meters.
- `AUC delta x ratio`: time-integrated hand-position response magnitude.
- `Cost SISU=0`, `Cost SISU=1`, `Cost ratio`, and `Cost diff`: post-hoc full-Q/R/Q_f perturbation delta cost, with `diff = SISU1 - SISU0`.
- Signed diagnostics are separated because endpoint and terminal-speed deltas are directional sidecars, not simple lower-is-better ratios.

### Headline

| Metric | Class groups with ratio < 1 | ratio = 1 | ratio > 1 | unavailable |
|---|---:|---:|---:|---:|
| Full-Q/R/Qf delta cost | 8 | 0 | 2 | 1 |
| Max delta x | 6 | 2 | 2 | 1 |
| Mean delta action | 0 | 0 | 10 | 1 |

### Class-Binned Summary

| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | Cost diff | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 1.30232 | 0.913207 | 0.683954 | 641.638 | 564.131 | 0.879204 | -77.507 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 1.30381 | 0.913857 | 0.691696 | 641.113 | 564.032 | 0.879769 | -77.0817 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 1.83253 | 1.15712 | 0.746099 | 437.282 | 777.037 | 1.77697 | 339.755 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 1.30072 | 1 | 0.845999 | 293.539 | 270.492 | 0.921485 | -23.0471 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 1.22697 | 0.990659 | 0.813881 | 30.0456 | 29.8008 | 0.991853 | -0.244781 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 1.30246 | 0.913237 | 0.683473 | 642.183 | 564.947 | 0.879728 | -77.2366 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 1.06966 | 0.861127 | 0.806675 | 3619.15 | 3296.61 | 0.91088 | -322.541 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 2.29807 | 1 | 0.777894 | 2247.66 | 2227.98 | 0.991242 | -19.6851 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 1.5093 | 0.933304 | 0.715412 | 1092.65 | 980.106 | 0.896996 | -112.548 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 1.83253 | 1.15712 | 0.746099 | 437.282 | 777.037 | 1.77697 | 339.755 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | NA | NA | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

#### Signed Diagnostics

| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | terminal diff |
|---|---:|---:|---:|---:|---:|---:|
| `command_input/command_input_pulse` | 0.00146103 | 0.00148365 | 2.2615e-05 | 0.00255341 | 0.000594924 | -0.00195848 |
| `command_input/target_aligned_lateral_command_load_pulse` | 0.0016032 | 0.00148686 | -0.000116338 | 0.00280245 | 0.000605037 | -0.00219741 |
| `delayed_observation/delayed_observation_offset` | 0.00121496 | 0.00103756 | -0.000177403 | 0.00265054 | 0.00384389 | 0.00119334 |
| `initial_state/initial_position_offset` | 0.00082143 | 1.36154e-05 | -0.000807814 | 0.000172195 | 1.48396e-05 | -0.000157355 |
| `initial_state/initial_velocity_offset` | 4.41611e-05 | -5.57161e-07 | -4.47182e-05 | 2.65992e-05 | -4.61557e-07 | -2.70608e-05 |
| `process_epsilon/process_epsilon_force_state_xy` | 0.0014844 | 0.00148066 | -3.7432e-06 | 0.00255612 | 0.000649278 | -0.00190684 |
| `process_epsilon/process_epsilon_integrator_xy` | 0.00769697 | 0.00761887 | -7.80984e-05 | 0.0411956 | 0.0431717 | 0.00197613 |
| `process_epsilon/process_epsilon_position_xy` | 0.0086948 | 0.00587428 | -0.00282052 | 0.00431029 | 0.000721094 | -0.0035892 |
| `process_epsilon/process_epsilon_velocity_xy` | 0.0031176 | 0.00273306 | -0.000384537 | 0.00315097 | 0.000795935 | -0.00235504 |
| `sensory_feedback/sensory_feedback_offset` | 0.00121496 | 0.00103756 | -0.000177403 | 0.00265054 | 0.00384389 | 0.00119334 |
| `target_stream/target_stream_jump` | NA | NA | NA | NA | NA | NA |

### Timing-Cell Summary

| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | Full-Q/R/Qf cost ratio | cost diff | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | 1.48783 | 0.949421 | 0.566157 | 0.995032 | -0.185241 | none |
| `command_input/command_input_pulse/late` | 12 | 1.04482 | 0.916814 | 0.937099 | 0.868641 | -234.537 | none |
| `command_input/command_input_pulse/mid` | 12 | 1.69572 | 0.874816 | 0.507205 | 1.02155 | 2.20157 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | 1.47567 | 0.954005 | 0.58087 | 1.11343 | 3.76428 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | 1.05545 | 0.914253 | 0.935113 | 0.864996 | -241.917 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | 1.65869 | 0.87916 | 0.50527 | 1.07032 | 6.90746 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | 1.60281 | 0.996012 | 0.568654 | 1.11844 | 20.4915 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | 1.70729 | 1.26854 | 1.30766 | 1.69825 | 610.878 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | 2.13742 | 1.24119 | 0.773012 | 2.46953 | 387.894 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset/initial_condition` | 12 | 1.30072 | 1 | 0.845999 | 0.921485 | -23.0471 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | 1.22697 | 0.990659 | 0.813881 | 0.991853 | -0.244781 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | 1.4846 | 0.949568 | 0.566477 | 0.999807 | -0.00718674 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | 1.04615 | 0.916696 | 0.936945 | 0.869081 | -234.002 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | 1.69684 | 0.875052 | 0.505752 | 1.02256 | 2.2998 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | 1.19908 | 0.689579 | 0.69841 | 0.615847 | -208.429 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | 0.954137 | 0.981478 | 0.984839 | 0.95395 | -423.809 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | 1.16182 | 0.778502 | 0.767719 | 0.69828 | -335.384 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | 1.80335 | 1 | 0.741195 | 0.957874 | -32.0631 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | 2.66644 | 1 | 0.980993 | 0.991389 | -34.0327 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | 2.82466 | 1 | 0.674385 | 1.00347 | 7.04057 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | 1.58886 | 0.972855 | 0.629951 | 1.02449 | 1.92387 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | 1.10293 | 0.930689 | 0.953613 | 0.886186 | -335.74 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | 2.01837 | 0.905057 | 0.540233 | 0.984654 | -3.82866 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | 1.60281 | 0.996012 | 0.568654 | 1.11844 | 20.4915 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | 1.70729 | 1.26854 | 1.30766 | 1.69825 | 610.878 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | 2.13742 | 1.24119 | 0.773012 | 2.46953 | 387.894 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump/not_applicable` | 1 | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
