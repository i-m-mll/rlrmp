# SISU Perturbation-Class Robustification Comparison

Issue: `7c1f7ed`. Source experiment: `7c1f7ed`.

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

## raw strong gamma-1.05 delayed SISU

Run: `delayed_sisu_spectrum__raw_strong_gamma_1p05_radius_lr1e-2_clip5_b64`

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
| Full-Q/R/Qf delta cost | 7 | 0 | 3 | 1 |
| Max delta x | 10 | 0 | 0 | 1 |
| Mean delta action | 0 | 0 | 10 | 1 |

### Class-Binned Summary

| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | Cost diff | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 1.18475 | 0.957107 | 0.807698 | 542.315 | 547.16 | 1.00893 | 4.84488 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 1.16314 | 0.96273 | 0.833725 | 560.368 | 570.617 | 1.01829 | 10.2493 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 1.07774 | 0.968941 | 0.772976 | 275.339 | 130.027 | 0.472243 | -145.312 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 1.14195 | 0.998568 | 0.876478 | 820.044 | 450.293 | 0.549109 | -369.751 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 1.17615 | 0.989909 | 0.892781 | 44.0264 | 43.2216 | 0.98172 | -0.804786 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 1.18596 | 0.954853 | 0.805356 | 541.526 | 545.113 | 1.00662 | 3.58631 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 1.05161 | 0.91872 | 0.882613 | 2914.88 | 2712.95 | 0.930724 | -201.931 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 1.44637 | 0.999997 | 0.876123 | 2342.22 | 2176.92 | 0.929427 | -165.297 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 1.24422 | 0.963246 | 0.824472 | 1036.15 | 991.383 | 0.956794 | -44.7683 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 1.07774 | 0.968941 | 0.772976 | 275.339 | 130.027 | 0.472243 | -145.312 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | NA | NA | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

#### Signed Diagnostics

| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | terminal diff |
|---|---:|---:|---:|---:|---:|---:|
| `command_input/command_input_pulse` | 0.000770258 | 0.00112319 | 0.000352934 | 0.00378517 | 0.00374102 | -4.41536e-05 |
| `command_input/target_aligned_lateral_command_load_pulse` | 0.000976646 | 0.00117814 | 0.000201497 | 0.00477277 | 0.00439531 | -0.000377455 |
| `delayed_observation/delayed_observation_offset` | 0.00104962 | 0.000462207 | -0.000587416 | 0.000547891 | 4.87466e-05 | -0.000499145 |
| `initial_state/initial_position_offset` | 0.00131877 | 0.000542429 | -0.000776338 | 0.00291405 | 0.000564041 | -0.00235 |
| `initial_state/initial_velocity_offset` | 7.79705e-05 | 8.17796e-05 | 3.80917e-06 | 0.00021637 | 5.40173e-05 | -0.000162353 |
| `process_epsilon/process_epsilon_force_state_xy` | 0.000790965 | 0.00113192 | 0.000340955 | 0.00383103 | 0.0036245 | -0.000206534 |
| `process_epsilon/process_epsilon_integrator_xy` | 0.00568308 | 0.00677022 | 0.00108714 | 0.0289953 | 0.0289637 | -3.15444e-05 |
| `process_epsilon/process_epsilon_position_xy` | 0.00767961 | 0.00723571 | -0.000443904 | 0.00459608 | 0.00158464 | -0.00301144 |
| `process_epsilon/process_epsilon_velocity_xy` | 0.00212384 | 0.00245908 | 0.000335241 | 0.00521372 | 0.0043867 | -0.000827024 |
| `sensory_feedback/sensory_feedback_offset` | 0.00104962 | 0.000462207 | -0.000587416 | 0.000547891 | 4.87466e-05 | -0.000499145 |
| `target_stream/target_stream_jump` | NA | NA | NA | NA | NA | NA |

### Timing-Cell Summary

| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | Full-Q/R/Qf cost ratio | cost diff | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | 1.18757 | 0.985292 | 0.789021 | 1.28689 | 9.81196 | none |
| `command_input/command_input_pulse/late` | 12 | 1.11629 | 0.966345 | 0.96043 | 1.01497 | 22.3141 | none |
| `command_input/command_input_pulse/mid` | 12 | 1.33813 | 0.91585 | 0.666331 | 0.828214 | -17.5915 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | 1.13812 | 0.9814 | 0.8446 | 1.18676 | 6.0681 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | 1.12706 | 0.970147 | 0.963423 | 1.02337 | 36.4676 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | 1.26869 | 0.932226 | 0.680885 | 0.866373 | -11.7878 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | 1.00253 | 0.945732 | 0.757752 | 0.143764 | -208.819 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | 1.30039 | 1.03418 | 0.749448 | 0.853685 | -23.6594 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | 1.01753 | 0.938255 | 0.802455 | 0.516077 | -203.458 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset/initial_condition` | 12 | 1.14195 | 0.998568 | 0.876478 | 0.549109 | -369.751 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | 1.17615 | 0.989909 | 0.892781 | 0.98172 | -0.804786 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | 1.18752 | 0.984962 | 0.785527 | 1.29579 | 10.1718 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | 1.11686 | 0.962066 | 0.957753 | 1.01248 | 18.578 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | 1.34201 | 0.915797 | 0.665296 | 0.823812 | -17.9909 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | 1.12368 | 0.796055 | 0.797185 | 0.691224 | -97.0025 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | 0.996918 | 0.982557 | 0.990232 | 0.952319 | -370.873 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | 1.0903 | 0.876644 | 0.866587 | 0.788581 | -137.917 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | 1.29104 | 1 | 0.843011 | 0.919934 | -67.3807 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | 1.71484 | 1 | 0.990217 | 0.988238 | -46.1858 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | 1.59261 | 0.999992 | 0.827919 | 0.830714 | -382.323 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | 1.24684 | 0.988264 | 0.791341 | 1.19447 | 13.7898 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | 1.04909 | 0.967833 | 0.977844 | 0.961899 | -106.023 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | 1.53973 | 0.934302 | 0.689534 | 0.834924 | -42.0722 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | 1.00253 | 0.945732 | 0.757752 | 0.143764 | -208.819 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | 1.30039 | 1.03418 | 0.749448 | 0.853685 | -23.6594 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | 1.01753 | 0.938255 | 0.802455 | 0.516077 | -203.458 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump/not_applicable` | 1 | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

## effective 020a65b PGD delayed SISU

Run: `delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64`

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
| Full-Q/R/Qf delta cost | 10 | 0 | 0 | 1 |
| Max delta x | 8 | 2 | 0 | 1 |
| Mean delta action | 0 | 0 | 10 | 1 |

### Class-Binned Summary

| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | Cost diff | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 1.30942 | 0.91958 | 0.6965 | 489.155 | 458.791 | 0.937925 | -30.3641 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 1.28668 | 0.927956 | 0.71234 | 482.572 | 462.345 | 0.958084 | -20.2275 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 1.18682 | 0.950564 | 0.690887 | 319.87 | 112.9 | 0.352954 | -206.971 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 1.44276 | 1 | 0.836364 | 458.685 | 366.246 | 0.798468 | -92.4396 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 1.28209 | 0.995792 | 0.840739 | 35.6781 | 33.4464 | 0.937451 | -2.23164 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 1.31147 | 0.91925 | 0.695104 | 487.907 | 457.975 | 0.938651 | -29.9326 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 1.1356 | 0.87245 | 0.810458 | 2731.89 | 2497.08 | 0.914047 | -234.814 | sisu_0:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 2.09018 | 1 | 0.809731 | 2299.2 | 2058.2 | 0.895182 | -240.999 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 1.51476 | 0.943415 | 0.727192 | 949.366 | 903.543 | 0.951733 | -45.8226 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 1.18682 | 0.950564 | 0.690887 | 319.87 | 112.9 | 0.352954 | -206.971 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | NA | NA | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

#### Signed Diagnostics

| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | terminal diff |
|---|---:|---:|---:|---:|---:|---:|
| `command_input/command_input_pulse` | 0.000719726 | 0.000877344 | 0.000157619 | 0.00228884 | 0.00155194 | -0.000736896 |
| `command_input/target_aligned_lateral_command_load_pulse` | 0.000870055 | 0.000909436 | 3.93806e-05 | 0.00198104 | 0.0014789 | -0.000502139 |
| `delayed_observation/delayed_observation_offset` | 0.001022 | 0.00020211 | -0.000819889 | 0.000560303 | -2.58938e-05 | -0.000586197 |
| `initial_state/initial_position_offset` | 0.00100199 | 3.47807e-05 | -0.000967214 | 0.000479348 | -0.000152496 | -0.000631845 |
| `initial_state/initial_velocity_offset` | 7.01052e-05 | 8.48203e-06 | -6.16232e-05 | -4.8754e-05 | -1.59843e-05 | 3.27696e-05 |
| `process_epsilon/process_epsilon_force_state_xy` | 0.000735897 | 0.000890145 | 0.000154248 | 0.0022865 | 0.00165623 | -0.000630273 |
| `process_epsilon/process_epsilon_integrator_xy` | 0.00612476 | 0.00649529 | 0.00037053 | 0.0277083 | 0.0286256 | 0.000917307 |
| `process_epsilon/process_epsilon_position_xy` | 0.00798797 | 0.00626843 | -0.00171954 | 0.00336723 | 0.000680622 | -0.00268661 |
| `process_epsilon/process_epsilon_velocity_xy` | 0.00210183 | 0.00205154 | -5.02959e-05 | 0.00331861 | 0.00171555 | -0.00160306 |
| `sensory_feedback/sensory_feedback_offset` | 0.001022 | 0.00020211 | -0.000819889 | 0.000560303 | -2.58938e-05 | -0.000586197 |
| `target_stream/target_stream_jump` | NA | NA | NA | NA | NA | NA |

### Timing-Cell Summary

| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | Full-Q/R/Qf cost ratio | cost diff | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | 1.45502 | 0.937867 | 0.642592 | 1.09822 | 3.18701 | none |
| `command_input/command_input_pulse/late` | 12 | 1.0871 | 0.95049 | 0.951185 | 0.944556 | -74.0441 | none |
| `command_input/command_input_pulse/mid` | 12 | 1.75286 | 0.850674 | 0.495463 | 0.796729 | -20.2352 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | 1.37987 | 0.948902 | 0.683651 | 1.17928 | 5.61443 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | 1.10374 | 0.946969 | 0.94225 | 0.960949 | -51.84 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | 1.65901 | 0.875657 | 0.499546 | 0.83741 | -14.4571 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | 1.06384 | 0.943005 | 0.746103 | 0.199064 | -268.986 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | 1.52756 | 1.02207 | 0.638416 | 0.866213 | -24.5244 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | 1.08391 | 0.898912 | 0.679086 | 0.256684 | -327.402 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset/initial_condition` | 12 | 1.44276 | 1 | 0.836364 | 0.798468 | -92.4396 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | 1.28209 | 0.995792 | 0.840739 | 0.937451 | -2.23164 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | 1.45105 | 0.937773 | 0.641731 | 1.11449 | 3.74625 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | 1.09311 | 0.949278 | 0.94937 | 0.94499 | -73.2266 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | 1.7507 | 0.851756 | 0.49479 | 0.796522 | -20.3173 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | 1.28482 | 0.70155 | 0.69744 | 0.589141 | -143.564 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | 0.999947 | 0.982877 | 0.987937 | 0.952856 | -336.136 | none |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | 1.25703 | 0.786309 | 0.764275 | 0.686253 | -224.741 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | 1.59401 | 1 | 0.772766 | 0.827298 | -138.802 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | 3.39384 | 1 | 0.974748 | 0.973628 | -103.49 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | 2.52897 | 1 | 0.729683 | 0.778442 | -480.704 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | 1.50019 | 0.965217 | 0.65648 | 1.01213 | 0.826368 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | 1.20056 | 0.960596 | 0.961968 | 0.968346 | -79.8262 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | 2.11897 | 0.896276 | 0.551547 | 0.773507 | -58.468 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | 1.06384 | 0.943005 | 0.746103 | 0.199064 | -268.986 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | 1.52756 | 1.02207 | 0.638416 | 0.866213 | -24.5244 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | 1.08391 | 0.898912 | 0.679086 | 0.256684 | -327.402 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump/not_applicable` | 1 | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
