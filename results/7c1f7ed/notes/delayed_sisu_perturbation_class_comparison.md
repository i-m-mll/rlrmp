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
| Full-Q/R/Qf delta cost | 10 | 0 | 0 | 1 |
| Max delta x | 10 | 0 | 0 | 1 |
| Mean delta action | 1 | 0 | 9 | 1 |

### Class-Binned Summary

| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | Cost diff | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 1.08899 | 0.983704 | 0.807136 | 168.41 | 143.789 | 0.853808 | -24.6202 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 1.08315 | 0.978577 | 0.812839 | 189.635 | 123.259 | 0.64998 | -66.3762 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 1.07774 | 0.968941 | 0.772976 | 275.339 | 130.027 | 0.472243 | -145.312 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 1.10961 | 0.998956 | 0.938226 | 326.655 | 204.893 | 0.627247 | -121.762 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 0.995633 | 0.935752 | 0.871976 | 94.5129 | 48.4343 | 0.512462 | -46.0787 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 1.09088 | 0.982417 | 0.804986 | 158.341 | 143.939 | 0.909041 | -14.4026 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 1.0366 | 0.828203 | 0.841095 | 4511.1 | 3236.43 | 0.717436 | -1274.67 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 1.23655 | 1 | 0.887979 | 532.128 | 418.054 | 0.785628 | -114.073 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 1.08642 | 0.949756 | 0.802844 | 329.433 | 234.414 | 0.711566 | -95.0197 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 1.07774 | 0.968941 | 0.772976 | 275.339 | 130.027 | 0.472243 | -145.312 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | NA | NA | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

#### Signed Diagnostics

| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | terminal diff |
|---|---:|---:|---:|---:|---:|---:|
| `command_input/command_input_pulse` | 0.000230575 | 0.000123536 | -0.000107039 | 0.000577467 | 0.000187989 | -0.000389478 |
| `command_input/target_aligned_lateral_command_load_pulse` | 0.000343415 | 0.000181843 | -0.000161571 | 0.00102984 | 0.000326264 | -0.000703577 |
| `delayed_observation/delayed_observation_offset` | 0.00104962 | 0.000462207 | -0.000587416 | 0.000547891 | 4.87466e-05 | -0.000499145 |
| `initial_state/initial_position_offset` | 0.00136953 | 0.000565871 | -0.000803658 | 0.00244169 | 0.00183597 | -0.000605721 |
| `initial_state/initial_velocity_offset` | 0.000235267 | 4.79488e-05 | -0.000187318 | -4.04919e-05 | -1.42244e-05 | 2.62675e-05 |
| `process_epsilon/process_epsilon_force_state_xy` | 0.000244819 | 0.000132083 | -0.000112736 | 0.0004841 | 0.000196116 | -0.000287985 |
| `process_epsilon/process_epsilon_integrator_xy` | 0.0117301 | 0.010733 | -0.000997115 | 0.0198475 | 0.0234135 | 0.00356605 |
| `process_epsilon/process_epsilon_position_xy` | 0.00167317 | 0.000738166 | -0.000935004 | 0.00223995 | 0.000740069 | -0.00149989 |
| `process_epsilon/process_epsilon_velocity_xy` | 0.000665475 | 0.000243282 | -0.000422193 | 0.000783935 | 0.000443115 | -0.00034082 |
| `sensory_feedback/sensory_feedback_offset` | 0.00104962 | 0.000462207 | -0.000587416 | 0.000547891 | 4.87466e-05 | -0.000499145 |
| `target_stream/target_stream_jump` | NA | NA | NA | NA | NA | NA |

### Timing-Cell Summary

| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | Full-Q/R/Qf cost ratio | cost diff | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | 1.04272 | 0.999725 | 0.928126 | 0.701013 | -4.19678 | none |
| `command_input/command_input_pulse/late` | 12 | 1.16892 | 0.978785 | 0.751895 | 0.865489 | -60.1253 | none |
| `command_input/command_input_pulse/mid` | 12 | 0.99182 | 0.984761 | 0.816805 | 0.7842 | -9.5384 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | 1.01995 | 1.00332 | 0.974368 | 1.29969 | 1.40175 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | 1.11732 | 0.974523 | 0.753727 | 0.633588 | -189.624 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | 1.078 | 0.970375 | 0.797553 | 0.766517 | -10.9067 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | 1.00253 | 0.945732 | 0.757752 | 0.143764 | -208.819 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | 1.30039 | 1.03418 | 0.749448 | 0.853685 | -23.6594 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | 1.01753 | 0.938255 | 0.802455 | 0.516077 | -203.458 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset/initial_condition` | 12 | 1.10961 | 0.998956 | 0.938226 | 0.627247 | -121.762 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | 0.995633 | 0.935752 | 0.871976 | 0.512462 | -46.0787 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | 1.04114 | 0.990998 | 0.923149 | 0.589029 | -5.62173 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | 1.17613 | 0.978948 | 0.750198 | 0.934994 | -27.1039 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | 0.988176 | 0.985347 | 0.816386 | 0.763929 | -10.482 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | 1.02368 | 0.872055 | 0.854426 | 0.66311 | -115.476 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | 1.03344 | 0.82665 | 0.840255 | 0.720656 | -3465.17 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | 1.055 | 0.805057 | 0.833007 | 0.690312 | -243.38 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | 1.15293 | 0.999988 | 0.915666 | 0.581029 | -136.348 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | 1.31082 | 1 | 0.838747 | 0.922747 | -71.6723 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | 1.2408 | 1.00001 | 0.893469 | 0.60896 | -134.2 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | 1.07816 | 0.950467 | 0.885862 | 0.478721 | -52.3492 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | 1.23025 | 0.985907 | 0.787777 | 1.00177 | 1.04242 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | 0.914713 | 0.892163 | 0.743528 | 0.213888 | -233.752 | none |
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
| Max delta x | 10 | 0 | 0 | 1 |
| Mean delta action | 0 | 0 | 10 | 1 |

### Class-Binned Summary

| Class | Rows | Status | Mean delta action ratio | Max delta x ratio | AUC delta x ratio | Cost SISU=0 | Cost SISU=1 | Cost ratio | Cost diff | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse` | 36 | evaluated=36 | 1.23713 | 0.95766 | 0.708892 | 110.12 | 96.1177 | 0.872845 | -14.0023 | none |
| `command_input/target_aligned_lateral_command_load_pulse` | 18 | evaluated=18 | 1.22508 | 0.966477 | 0.723312 | 101.367 | 99.6216 | 0.982778 | -1.74575 | none |
| `delayed_observation/delayed_observation_offset` | 108 | evaluated=108 | 1.18682 | 0.950564 | 0.690887 | 319.87 | 112.9 | 0.352954 | -206.971 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset` | 12 | evaluated=12 | 1.27855 | 0.999913 | 0.920579 | 260.544 | 140.865 | 0.540658 | -119.679 | none |
| `initial_state/initial_velocity_offset` | 12 | evaluated=12 | 1.06936 | 0.989073 | 0.916662 | 14.1872 | -0.0955358 | -0.00673395 | -14.2827 | none |
| `process_epsilon/process_epsilon_force_state_xy` | 36 | evaluated=36 | 1.23109 | 0.957437 | 0.708672 | 110.629 | 97.5101 | 0.881418 | -13.1185 | none |
| `process_epsilon/process_epsilon_integrator_xy` | 36 | evaluated=36 | 1.20571 | 0.7577 | 0.76891 | 4792.42 | 3260.15 | 0.680273 | -1532.27 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy` | 36 | evaluated=36 | 1.46852 | 0.999996 | 0.844216 | 475.245 | 346.043 | 0.728137 | -129.202 | none |
| `process_epsilon/process_epsilon_velocity_xy` | 36 | evaluated=36 | 1.23204 | 0.980116 | 0.762461 | 166.589 | 133.441 | 0.801021 | -33.1477 | none |
| `sensory_feedback/sensory_feedback_offset` | 108 | evaluated=108 | 1.18682 | 0.950564 | 0.690887 | 319.87 | 112.9 | 0.352954 | -206.971 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump` | 1 | not_applicable=1 | NA | NA | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |

#### Signed Diagnostics

| Class | Endpoint delta SISU=0 | Endpoint delta SISU=1 | endpoint diff | Terminal-speed delta SISU=0 | Terminal-speed delta SISU=1 | terminal diff |
|---|---:|---:|---:|---:|---:|---:|
| `command_input/command_input_pulse` | 0.000199387 | -3.32085e-06 | -0.000202708 | 0.000136424 | -4.3628e-05 | -0.000180052 |
| `command_input/target_aligned_lateral_command_load_pulse` | 0.0002456 | -6.40878e-06 | -0.000252009 | 6.35058e-05 | -2.60818e-05 | -8.95876e-05 |
| `delayed_observation/delayed_observation_offset` | 0.001022 | 0.00020211 | -0.000819889 | 0.000560303 | -2.58938e-05 | -0.000586197 |
| `initial_state/initial_position_offset` | 0.00113654 | 9.21204e-05 | -0.00104442 | -2.79981e-05 | -0.000184844 | -0.000156846 |
| `initial_state/initial_velocity_offset` | 4.91956e-05 | 5.04369e-07 | -4.86912e-05 | -4.40749e-05 | -6.30727e-06 | 3.77676e-05 |
| `process_epsilon/process_epsilon_force_state_xy` | 0.000206805 | -1.42271e-06 | -0.000208228 | 0.000120572 | -4.4936e-05 | -0.000165508 |
| `process_epsilon/process_epsilon_integrator_xy` | 0.0129745 | 0.0104889 | -0.00248551 | 0.020409 | 0.0253665 | 0.00495744 |
| `process_epsilon/process_epsilon_position_xy` | 0.00147627 | 0.00014569 | -0.00133058 | 0.000544538 | -0.000170964 | -0.000715502 |
| `process_epsilon/process_epsilon_velocity_xy` | 0.000458276 | 1.49652e-05 | -0.000443311 | 9.47613e-05 | -7.90516e-05 | -0.000173813 |
| `sensory_feedback/sensory_feedback_offset` | 0.001022 | 0.00020211 | -0.000819889 | 0.000560303 | -2.58938e-05 | -0.000586197 |
| `target_stream/target_stream_jump` | NA | NA | NA | NA | NA | NA |

### Timing-Cell Summary

| Cell | Rows | Mean delta action ratio | Max dx ratio | AUC dx ratio | Full-Q/R/Qf cost ratio | cost diff | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input/command_input_pulse/early` | 12 | 1.04196 | 0.987137 | 0.861709 | 8.06761 | -7.6204 | none |
| `command_input/command_input_pulse/late` | 12 | 1.41358 | 0.934915 | 0.611026 | 0.922504 | -23.6175 | none |
| `command_input/command_input_pulse/mid` | 12 | 1.08544 | 0.999107 | 0.830595 | 0.596373 | -10.7689 | none |
| `command_input/target_aligned_lateral_command_load_pulse/early` | 6 | 1.05936 | 1.00223 | 0.861651 | -1.66766 | 4.88601 | none |
| `command_input/target_aligned_lateral_command_load_pulse/late` | 6 | 1.35934 | 0.94605 | 0.621003 | 1.00596 | 1.68434 | none |
| `command_input/target_aligned_lateral_command_load_pulse/mid` | 6 | 1.1154 | 0.992284 | 0.829992 | 0.489012 | -11.8076 | none |
| `delayed_observation/delayed_observation_offset/early_visible` | 36 | 1.06384 | 0.943005 | 0.746103 | 0.199064 | -268.986 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/late_visible` | 36 | 1.52756 | 1.02207 | 0.638416 | 0.866213 | -24.5244 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `delayed_observation/delayed_observation_offset/mid_visible` | 36 | 1.08391 | 0.898912 | 0.679086 | 0.256684 | -327.402 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `initial_state/initial_position_offset/initial_condition` | 12 | 1.27855 | 0.999913 | 0.920579 | 0.540658 | -119.679 | none |
| `initial_state/initial_velocity_offset/initial_condition` | 12 | 1.06936 | 0.989073 | 0.916662 | -0.00673395 | -14.2827 | none |
| `process_epsilon/process_epsilon_force_state_xy/early` | 12 | 1.0384 | 0.984642 | 0.859577 | 19.0572 | -8.52934 | none |
| `process_epsilon/process_epsilon_force_state_xy/late` | 12 | 1.40616 | 0.934872 | 0.611034 | 0.933334 | -20.4025 | none |
| `process_epsilon/process_epsilon_force_state_xy/mid` | 12 | 1.08031 | 1.00015 | 0.831635 | 0.603955 | -10.4238 | none |
| `process_epsilon/process_epsilon_integrator_xy/early` | 12 | 1.17377 | 0.744634 | 0.782242 | 0.570974 | -159.432 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/late` | 12 | 1.20863 | 0.778765 | 0.77229 | 0.690244 | -4061.44 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_integrator_xy/mid` | 12 | 1.22475 | 0.689401 | 0.751037 | 0.579447 | -375.925 | sisu_0:inflated_ratio; sisu_1:inflated_ratio |
| `process_epsilon/process_epsilon_position_xy/early` | 12 | 1.33243 | 0.999988 | 0.891087 | 0.626695 | -98.8093 | none |
| `process_epsilon/process_epsilon_position_xy/late` | 12 | 1.62766 | 1 | 0.765867 | 0.826039 | -153.893 | none |
| `process_epsilon/process_epsilon_position_xy/mid` | 12 | 1.42285 | 1 | 0.850194 | 0.511944 | -134.903 | none |
| `process_epsilon/process_epsilon_velocity_xy/early` | 12 | 1.04571 | 0.998726 | 0.887422 | -0.0644251 | -33.9367 | none |
| `process_epsilon/process_epsilon_velocity_xy/late` | 12 | 1.45578 | 0.962845 | 0.662028 | 0.916989 | -35.022 | none |
| `process_epsilon/process_epsilon_velocity_xy/mid` | 12 | 1.06152 | 0.998074 | 0.824372 | 0.337144 | -30.4845 | none |
| `sensory_feedback/sensory_feedback_offset/early_visible` | 36 | 1.06384 | 0.943005 | 0.746103 | 0.199064 | -268.986 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/late_visible` | 36 | 1.52756 | 1.02207 | 0.638416 | 0.866213 | -24.5244 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `sensory_feedback/sensory_feedback_offset/mid_visible` | 36 | 1.08391 | 0.898912 | 0.679086 | 0.256684 | -327.402 | sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
| `target_stream/target_stream_jump/not_applicable` | 1 | NA | NA | NA | NA | NA | sisu_0:extlqg_not_applicable_reasons=1; sisu_0:not_applicable_reasons=1; sisu_0:robust_analytical_not_applicable_reasons=1; sisu_1:extlqg_not_applicable_reasons=1; sisu_1:not_applicable_reasons=1; sisu_1:robust_analytical_not_applicable_reasons=1 |
