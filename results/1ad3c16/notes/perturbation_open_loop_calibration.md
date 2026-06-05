# Perturbation Open-Loop Calibration

- Issue: `1ad3c16`
- Scope: `extlqg_nominal_command_open_loop_physical_effect_calibration`
- Open-loop reference: extLQG nominal command replay.
- Closed-loop extLQG is reported at the same amplitudes where supported.
- Calibration mode: reach-relative peak `delta x`, with target peak `delta x = fraction * reach_length`.
- Unit sensitivities are calibrated by perturbation family and timing bin; reach/level rows are deterministic scalings from those sensitivities, not independent calibrations.
- Replay geometry: canonical 0.15 m +x extLQG nominal command replay; the reach-relative targets vary the requested physical effect size, not the nominal replay task.
- GRU baseline for later closed-loop calibration: `5f70333` / `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` (single-target nominal-only; documented for later closed-loop comparison, not retrained here).
- Bulk row manifest: `_artifacts/1ad3c16/perturbation_open_loop_calibration/perturbation_open_loop_calibration.json`

## Deterministic Config

Reach lengths:

| Label | Split | Reach length | Role |
|---|---|---:|---|
| `seen_train_0p10` | seen/train | 100.000 mm | multi_target_training_reach_length |
| `seen_train_anchor_0p15` | seen/train | 150.000 mm | multi_target_training_reach_length_and_original_anchor |
| `heldout_eval_0p12` | held-out/eval | 120.000 mm | multi_target_held_out_evaluation_reach_length |
| `heldout_eval_0p18` | held-out/eval | 180.000 mm | multi_target_held_out_evaluation_reach_length |

Levels:

| Level | Fraction of reach | Role |
|---|---:|---|
| `small` | 5.0% | small_probe |
| `moderate` | 10.0% | moderate_probe |
| `stress` | 25.0% | stress_probe |

Plant-side timing bins:

| Bin | Start step | Duration | Role |
|---|---:|---:|---|
| `early` | 5 | 5 | plant_side_open_loop_calibration |
| `mid` | 15 | 5 | plant_side_open_loop_calibration |
| `late` | 35 | 5 | plant_side_open_loop_calibration |

Controller-visible/native conventions:

| Family | Channel | Native rule | Timing rule | Report metric |
|---|---|---|---|---|
| `sensory_feedback_offset` | `sensory_feedback` | position offsets are fractions of reach length; velocity offsets are fractions of nominal peak speed when available | controller-visible starts 10/20/40 with 5-step duration | closed-loop induced discrepancy against paired nominal rollout |
| `delayed_observation_offset` | `delayed_observation` | pre-noise delayed-measurement position offsets are fractions of reach length; velocity offsets use nominal peak speed placeholder when the actual peak speed is unavailable | controller-visible starts 10/20/40 with 5-step duration | closed-loop induced discrepancy against paired nominal rollout |
| `target_stream_jump` | `target_stream` | target offsets are fractions of reach length | controller-visible starts 10/20/40 with 5-step duration | closed-loop induced discrepancy once target-stream rows exist |
| `true_extra_delay_steps` | `feedback_delay` | integer extra delay steps, not a reach-relative amplitude | applies to the feedback path delay schedule rather than pulse timing | induced discrepancy from added delay, to be reported in future rows |

## Selected Reach-Relative Amplitudes

| Reach | Level | Family | Timing bin | Amplitude | Target peak dx | Achieved peak dx | Achieved % reach | AUC dx | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `heldout_eval_0p12` | `small` | `command_input_pulse` | `early` | 0.270336 | 6.000 mm | 6.000 mm | 5.000% | 0.0014281 | none |
| `heldout_eval_0p12` | `small` | `command_input_pulse` | `mid` | 0.344968 | 6.000 mm | 6.000 mm | 5.000% | 0.00113113 | none |
| `heldout_eval_0p12` | `small` | `command_input_pulse` | `late` | 0.776493 | 6.000 mm | 6.000 mm | 5.000% | 5.5870e-04 | none |
| `heldout_eval_0p12` | `small` | `initial_position_offset` | `initial_condition` | 0.006 | 6.000 mm | 6.000 mm | 5.000% | 0.0036 | none |
| `heldout_eval_0p12` | `small` | `initial_velocity_offset` | `initial_condition` | 0.010298 | 6.000 mm | 6.000 mm | 5.000% | 0.001848 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_force_state_xy` | `early` | 0.04096 | 6.000 mm | 6.000 mm | 5.000% | 0.0014281 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_force_state_xy` | `mid` | 0.0522679 | 6.000 mm | 6.000 mm | 5.000% | 0.00113113 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_force_state_xy` | `late` | 0.11765 | 6.000 mm | 6.000 mm | 5.000% | 5.5870e-04 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_integrator_xy` | `early` | 0.00919457 | 6.000 mm | 6.000 mm | 5.000% | 0.00106596 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_integrator_xy` | `mid` | 0.0141078 | 6.000 mm | 6.000 mm | 5.000% | 8.6479e-04 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_integrator_xy` | `late` | 0.0520728 | 6.000 mm | 6.000 mm | 5.000% | 4.6448e-04 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_position_xy` | `early` | 0.0012 | 6.000 mm | 6.000 mm | 5.000% | 0.00318 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_position_xy` | `mid` | 0.0012 | 6.000 mm | 6.000 mm | 5.000% | 0.00258 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_position_xy` | `late` | 0.0012 | 6.000 mm | 6.000 mm | 5.000% | 0.00138 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_velocity_xy` | `early` | 0.0023671 | 6.000 mm | 6.000 mm | 5.000% | 0.00160467 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_velocity_xy` | `mid` | 0.0029162 | 6.000 mm | 6.000 mm | 5.000% | 0.00130025 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_velocity_xy` | `late` | 0.00551229 | 6.000 mm | 6.000 mm | 5.000% | 6.9514e-04 | none |
| `heldout_eval_0p12` | `moderate` | `command_input_pulse` | `early` | 0.540672 | 12.000 mm | 12.000 mm | 10.000% | 0.00285621 | none |
| `heldout_eval_0p12` | `moderate` | `command_input_pulse` | `mid` | 0.689936 | 12.000 mm | 12.000 mm | 10.000% | 0.00226227 | none |
| `heldout_eval_0p12` | `moderate` | `command_input_pulse` | `late` | 1.55299 | 12.000 mm | 12.000 mm | 10.000% | 0.0011174 | none |
| `heldout_eval_0p12` | `moderate` | `initial_position_offset` | `initial_condition` | 0.012 | 12.000 mm | 12.000 mm | 10.000% | 0.0072 | none |
| `heldout_eval_0p12` | `moderate` | `initial_velocity_offset` | `initial_condition` | 0.020596 | 12.000 mm | 12.000 mm | 10.000% | 0.00369601 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_force_state_xy` | `early` | 0.0819201 | 12.000 mm | 12.000 mm | 10.000% | 0.00285621 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_force_state_xy` | `mid` | 0.104536 | 12.000 mm | 12.000 mm | 10.000% | 0.00226227 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_force_state_xy` | `late` | 0.235301 | 12.000 mm | 12.000 mm | 10.000% | 0.0011174 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_integrator_xy` | `early` | 0.0183891 | 12.000 mm | 12.000 mm | 10.000% | 0.00213192 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_integrator_xy` | `mid` | 0.0282157 | 12.000 mm | 12.000 mm | 10.000% | 0.00172958 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_integrator_xy` | `late` | 0.104146 | 12.000 mm | 12.000 mm | 10.000% | 9.2895e-04 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_position_xy` | `early` | 0.0024 | 12.000 mm | 12.000 mm | 10.000% | 0.00636 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_position_xy` | `mid` | 0.0024 | 12.000 mm | 12.000 mm | 10.000% | 0.00516 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_position_xy` | `late` | 0.0024 | 12.000 mm | 12.000 mm | 10.000% | 0.00276 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_velocity_xy` | `early` | 0.00473421 | 12.000 mm | 12.000 mm | 10.000% | 0.00320935 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_velocity_xy` | `mid` | 0.0058324 | 12.000 mm | 12.000 mm | 10.000% | 0.0026005 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_velocity_xy` | `late` | 0.0110246 | 12.000 mm | 12.000 mm | 10.000% | 0.00139029 | none |
| `heldout_eval_0p12` | `stress` | `command_input_pulse` | `early` | 1.35168 | 30.000 mm | 30.000 mm | 25.000% | 0.00714052 | none |
| `heldout_eval_0p12` | `stress` | `command_input_pulse` | `mid` | 1.72484 | 30.000 mm | 30.000 mm | 25.000% | 0.00565567 | none |
| `heldout_eval_0p12` | `stress` | `command_input_pulse` | `late` | 3.88247 | 30.000 mm | 30.000 mm | 25.000% | 0.0027935 | none |
| `heldout_eval_0p12` | `stress` | `initial_position_offset` | `initial_condition` | 0.03 | 30.000 mm | 30.000 mm | 25.000% | 0.018 | none |
| `heldout_eval_0p12` | `stress` | `initial_velocity_offset` | `initial_condition` | 0.05149 | 30.000 mm | 30.000 mm | 25.000% | 0.00924001 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_force_state_xy` | `early` | 0.2048 | 30.000 mm | 30.000 mm | 25.000% | 0.00714052 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_force_state_xy` | `mid` | 0.261339 | 30.000 mm | 30.000 mm | 25.000% | 0.00565567 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_force_state_xy` | `late` | 0.588252 | 30.000 mm | 30.000 mm | 25.000% | 0.0027935 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_integrator_xy` | `early` | 0.0459728 | 30.000 mm | 30.000 mm | 25.000% | 0.0053298 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_integrator_xy` | `mid` | 0.0705392 | 30.000 mm | 30.000 mm | 25.000% | 0.00432395 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_integrator_xy` | `late` | 0.260364 | 30.000 mm | 30.000 mm | 25.000% | 0.00232238 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_position_xy` | `early` | 0.006 | 30.000 mm | 30.000 mm | 25.000% | 0.0159 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_position_xy` | `mid` | 0.006 | 30.000 mm | 30.000 mm | 25.000% | 0.0129 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_position_xy` | `late` | 0.006 | 30.000 mm | 30.000 mm | 25.000% | 0.0069 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_velocity_xy` | `early` | 0.0118355 | 30.000 mm | 30.000 mm | 25.000% | 0.00802337 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_velocity_xy` | `mid` | 0.014581 | 30.000 mm | 30.000 mm | 25.000% | 0.00650124 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_velocity_xy` | `late` | 0.0275614 | 30.000 mm | 30.000 mm | 25.000% | 0.00347572 | none |
| `heldout_eval_0p18` | `small` | `command_input_pulse` | `early` | 0.405504 | 9.000 mm | 9.000 mm | 5.000% | 0.00214216 | none |
| `heldout_eval_0p18` | `small` | `command_input_pulse` | `mid` | 0.517452 | 9.000 mm | 9.000 mm | 5.000% | 0.0016967 | none |
| `heldout_eval_0p18` | `small` | `command_input_pulse` | `late` | 1.16474 | 9.000 mm | 9.000 mm | 5.000% | 8.3805e-04 | none |
| `heldout_eval_0p18` | `small` | `initial_position_offset` | `initial_condition` | 0.009 | 9.000 mm | 9.000 mm | 5.000% | 0.0054 | none |
| `heldout_eval_0p18` | `small` | `initial_velocity_offset` | `initial_condition` | 0.015447 | 9.000 mm | 9.000 mm | 5.000% | 0.002772 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_force_state_xy` | `early` | 0.06144 | 9.000 mm | 9.000 mm | 5.000% | 0.00214216 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_force_state_xy` | `mid` | 0.0784018 | 9.000 mm | 9.000 mm | 5.000% | 0.0016967 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_force_state_xy` | `late` | 0.176476 | 9.000 mm | 9.000 mm | 5.000% | 8.3805e-04 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_integrator_xy` | `early` | 0.0137919 | 9.000 mm | 9.000 mm | 5.000% | 0.00159894 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_integrator_xy` | `mid` | 0.0211618 | 9.000 mm | 9.000 mm | 5.000% | 0.00129718 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_integrator_xy` | `late` | 0.0781092 | 9.000 mm | 9.000 mm | 5.000% | 6.9671e-04 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_position_xy` | `early` | 0.0018 | 9.000 mm | 9.000 mm | 5.000% | 0.00477 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_position_xy` | `mid` | 0.0018 | 9.000 mm | 9.000 mm | 5.000% | 0.00387 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_position_xy` | `late` | 0.0018 | 9.000 mm | 9.000 mm | 5.000% | 0.00207 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_velocity_xy` | `early` | 0.00355065 | 9.000 mm | 9.000 mm | 5.000% | 0.00240701 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_velocity_xy` | `mid` | 0.0043743 | 9.000 mm | 9.000 mm | 5.000% | 0.00195037 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_velocity_xy` | `late` | 0.00826843 | 9.000 mm | 9.000 mm | 5.000% | 0.00104272 | none |
| `heldout_eval_0p18` | `moderate` | `command_input_pulse` | `early` | 0.811009 | 18.000 mm | 18.000 mm | 10.000% | 0.00428431 | none |
| `heldout_eval_0p18` | `moderate` | `command_input_pulse` | `mid` | 1.0349 | 18.000 mm | 18.000 mm | 10.000% | 0.0033934 | none |
| `heldout_eval_0p18` | `moderate` | `command_input_pulse` | `late` | 2.32948 | 18.000 mm | 18.000 mm | 10.000% | 0.0016761 | none |
| `heldout_eval_0p18` | `moderate` | `initial_position_offset` | `initial_condition` | 0.018 | 18.000 mm | 18.000 mm | 10.000% | 0.0108 | none |
| `heldout_eval_0p18` | `moderate` | `initial_velocity_offset` | `initial_condition` | 0.030894 | 18.000 mm | 18.000 mm | 10.000% | 0.00554401 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_force_state_xy` | `early` | 0.12288 | 18.000 mm | 18.000 mm | 10.000% | 0.00428431 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_force_state_xy` | `mid` | 0.156804 | 18.000 mm | 18.000 mm | 10.000% | 0.0033934 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_force_state_xy` | `late` | 0.352951 | 18.000 mm | 18.000 mm | 10.000% | 0.0016761 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_integrator_xy` | `early` | 0.0275837 | 18.000 mm | 18.000 mm | 10.000% | 0.00319788 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_integrator_xy` | `mid` | 0.0423235 | 18.000 mm | 18.000 mm | 10.000% | 0.00259437 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_integrator_xy` | `late` | 0.156218 | 18.000 mm | 18.000 mm | 10.000% | 0.00139343 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_position_xy` | `early` | 0.0036 | 18.000 mm | 18.000 mm | 10.000% | 0.00954 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_position_xy` | `mid` | 0.0036 | 18.000 mm | 18.000 mm | 10.000% | 0.00774 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_position_xy` | `late` | 0.0036 | 18.000 mm | 18.000 mm | 10.000% | 0.00414 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_velocity_xy` | `early` | 0.00710131 | 18.000 mm | 18.000 mm | 10.000% | 0.00481402 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_velocity_xy` | `mid` | 0.00874861 | 18.000 mm | 18.000 mm | 10.000% | 0.00390074 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_velocity_xy` | `late` | 0.0165369 | 18.000 mm | 18.000 mm | 10.000% | 0.00208543 | none |
| `heldout_eval_0p18` | `stress` | `command_input_pulse` | `early` | 2.02752 | 45.000 mm | 45.000 mm | 25.000% | 0.0107108 | none |
| `heldout_eval_0p18` | `stress` | `command_input_pulse` | `mid` | 2.58726 | 45.000 mm | 45.000 mm | 25.000% | 0.0084835 | none |
| `heldout_eval_0p18` | `stress` | `command_input_pulse` | `late` | 5.8237 | 45.000 mm | 45.000 mm | 25.000% | 0.00419026 | none |
| `heldout_eval_0p18` | `stress` | `initial_position_offset` | `initial_condition` | 0.045 | 45.000 mm | 45.000 mm | 25.000% | 0.027 | none |
| `heldout_eval_0p18` | `stress` | `initial_velocity_offset` | `initial_condition` | 0.077235 | 45.000 mm | 45.000 mm | 25.000% | 0.01386 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_force_state_xy` | `early` | 0.3072 | 45.000 mm | 45.000 mm | 25.000% | 0.0107108 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_force_state_xy` | `mid` | 0.392009 | 45.000 mm | 45.000 mm | 25.000% | 0.0084835 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_force_state_xy` | `late` | 0.882379 | 45.000 mm | 45.000 mm | 25.000% | 0.00419026 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_integrator_xy` | `early` | 0.0689593 | 45.000 mm | 45.000 mm | 25.000% | 0.00799469 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_integrator_xy` | `mid` | 0.105809 | 45.000 mm | 45.000 mm | 25.000% | 0.00648592 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_integrator_xy` | `late` | 0.390546 | 45.000 mm | 45.000 mm | 25.000% | 0.00348357 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_position_xy` | `early` | 0.009 | 45.000 mm | 45.000 mm | 25.000% | 0.02385 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_position_xy` | `mid` | 0.009 | 45.000 mm | 45.000 mm | 25.000% | 0.01935 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_position_xy` | `late` | 0.009 | 45.000 mm | 45.000 mm | 25.000% | 0.01035 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_velocity_xy` | `early` | 0.0177533 | 45.000 mm | 45.000 mm | 25.000% | 0.0120351 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_velocity_xy` | `mid` | 0.0218715 | 45.000 mm | 45.000 mm | 25.000% | 0.00975186 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_velocity_xy` | `late` | 0.0413421 | 45.000 mm | 45.000 mm | 25.000% | 0.00521358 | none |
| `seen_train_0p10` | `small` | `command_input_pulse` | `early` | 0.22528 | 5.000 mm | 5.000 mm | 5.000% | 0.00119009 | none |
| `seen_train_0p10` | `small` | `command_input_pulse` | `mid` | 0.287473 | 5.000 mm | 5.000 mm | 5.000% | 9.4261e-04 | none |
| `seen_train_0p10` | `small` | `command_input_pulse` | `late` | 0.647078 | 5.000 mm | 5.000 mm | 5.000% | 4.6558e-04 | none |
| `seen_train_0p10` | `small` | `initial_position_offset` | `initial_condition` | 0.005 | 5.000 mm | 5.000 mm | 5.000% | 0.003 | none |
| `seen_train_0p10` | `small` | `initial_velocity_offset` | `initial_condition` | 0.00858167 | 5.000 mm | 5.000 mm | 5.000% | 0.00154 | none |
| `seen_train_0p10` | `small` | `process_epsilon_force_state_xy` | `early` | 0.0341334 | 5.000 mm | 5.000 mm | 5.000% | 0.00119009 | none |
| `seen_train_0p10` | `small` | `process_epsilon_force_state_xy` | `mid` | 0.0435566 | 5.000 mm | 5.000 mm | 5.000% | 9.4261e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_force_state_xy` | `late` | 0.0980421 | 5.000 mm | 5.000 mm | 5.000% | 4.6558e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_integrator_xy` | `early` | 0.00766214 | 5.000 mm | 5.000 mm | 5.000% | 8.8830e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_integrator_xy` | `mid` | 0.0117565 | 5.000 mm | 5.000 mm | 5.000% | 7.2066e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_integrator_xy` | `late` | 0.043394 | 5.000 mm | 5.000 mm | 5.000% | 3.8706e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_position_xy` | `early` | 0.001 | 5.000 mm | 5.000 mm | 5.000% | 0.00265 | none |
| `seen_train_0p10` | `small` | `process_epsilon_position_xy` | `mid` | 0.001 | 5.000 mm | 5.000 mm | 5.000% | 0.00215 | none |
| `seen_train_0p10` | `small` | `process_epsilon_position_xy` | `late` | 0.001 | 5.000 mm | 5.000 mm | 5.000% | 0.00115 | none |
| `seen_train_0p10` | `small` | `process_epsilon_velocity_xy` | `early` | 0.00197259 | 5.000 mm | 5.000 mm | 5.000% | 0.00133723 | none |
| `seen_train_0p10` | `small` | `process_epsilon_velocity_xy` | `mid` | 0.00243017 | 5.000 mm | 5.000 mm | 5.000% | 0.00108354 | none |
| `seen_train_0p10` | `small` | `process_epsilon_velocity_xy` | `late` | 0.00459357 | 5.000 mm | 5.000 mm | 5.000% | 5.7929e-04 | none |
| `seen_train_0p10` | `moderate` | `command_input_pulse` | `early` | 0.45056 | 10.000 mm | 10.000 mm | 10.000% | 0.00238017 | none |
| `seen_train_0p10` | `moderate` | `command_input_pulse` | `mid` | 0.574947 | 10.000 mm | 10.000 mm | 10.000% | 0.00188522 | none |
| `seen_train_0p10` | `moderate` | `command_input_pulse` | `late` | 1.29416 | 10.000 mm | 10.000 mm | 10.000% | 9.3117e-04 | none |
| `seen_train_0p10` | `moderate` | `initial_position_offset` | `initial_condition` | 0.01 | 10.000 mm | 10.000 mm | 10.000% | 0.006 | none |
| `seen_train_0p10` | `moderate` | `initial_velocity_offset` | `initial_condition` | 0.0171633 | 10.000 mm | 10.000 mm | 10.000% | 0.00308 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_force_state_xy` | `early` | 0.0682667 | 10.000 mm | 10.000 mm | 10.000% | 0.00238017 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_force_state_xy` | `mid` | 0.0871132 | 10.000 mm | 10.000 mm | 10.000% | 0.00188522 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_force_state_xy` | `late` | 0.196084 | 10.000 mm | 10.000 mm | 10.000% | 9.3117e-04 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_integrator_xy` | `early` | 0.0153243 | 10.000 mm | 10.000 mm | 10.000% | 0.0017766 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_integrator_xy` | `mid` | 0.0235131 | 10.000 mm | 10.000 mm | 10.000% | 0.00144132 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_integrator_xy` | `late` | 0.086788 | 10.000 mm | 10.000 mm | 10.000% | 7.7413e-04 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_position_xy` | `early` | 0.002 | 10.000 mm | 10.000 mm | 10.000% | 0.0053 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_position_xy` | `mid` | 0.002 | 10.000 mm | 10.000 mm | 10.000% | 0.0043 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_position_xy` | `late` | 0.002 | 10.000 mm | 10.000 mm | 10.000% | 0.0023 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_velocity_xy` | `early` | 0.00394517 | 10.000 mm | 10.000 mm | 10.000% | 0.00267446 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_velocity_xy` | `mid` | 0.00486034 | 10.000 mm | 10.000 mm | 10.000% | 0.00216708 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_velocity_xy` | `late` | 0.00918714 | 10.000 mm | 10.000 mm | 10.000% | 0.00115857 | none |
| `seen_train_0p10` | `stress` | `command_input_pulse` | `early` | 1.1264 | 25.000 mm | 25.000 mm | 25.000% | 0.00595043 | none |
| `seen_train_0p10` | `stress` | `command_input_pulse` | `mid` | 1.43737 | 25.000 mm | 25.000 mm | 25.000% | 0.00471306 | none |
| `seen_train_0p10` | `stress` | `command_input_pulse` | `late` | 3.23539 | 25.000 mm | 25.000 mm | 25.000% | 0.00232792 | none |
| `seen_train_0p10` | `stress` | `initial_position_offset` | `initial_condition` | 0.025 | 25.000 mm | 25.000 mm | 25.000% | 0.015 | none |
| `seen_train_0p10` | `stress` | `initial_velocity_offset` | `initial_condition` | 0.0429083 | 25.000 mm | 25.000 mm | 25.000% | 0.00770001 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_force_state_xy` | `early` | 0.170667 | 25.000 mm | 25.000 mm | 25.000% | 0.00595043 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_force_state_xy` | `mid` | 0.217783 | 25.000 mm | 25.000 mm | 25.000% | 0.00471306 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_force_state_xy` | `late` | 0.49021 | 25.000 mm | 25.000 mm | 25.000% | 0.00232792 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_integrator_xy` | `early` | 0.0383107 | 25.000 mm | 25.000 mm | 25.000% | 0.0044415 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_integrator_xy` | `mid` | 0.0587827 | 25.000 mm | 25.000 mm | 25.000% | 0.00360329 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_integrator_xy` | `late` | 0.21697 | 25.000 mm | 25.000 mm | 25.000% | 0.00193532 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_position_xy` | `early` | 0.005 | 25.000 mm | 25.000 mm | 25.000% | 0.01325 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_position_xy` | `mid` | 0.005 | 25.000 mm | 25.000 mm | 25.000% | 0.01075 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_position_xy` | `late` | 0.005 | 25.000 mm | 25.000 mm | 25.000% | 0.00575 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_velocity_xy` | `early` | 0.00986293 | 25.000 mm | 25.000 mm | 25.000% | 0.00668614 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_velocity_xy` | `mid` | 0.0121508 | 25.000 mm | 25.000 mm | 25.000% | 0.0054177 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_velocity_xy` | `late` | 0.0229679 | 25.000 mm | 25.000 mm | 25.000% | 0.00289643 | none |
| `seen_train_anchor_0p15` | `small` | `command_input_pulse` | `early` | 0.33792 | 7.500 mm | 7.500 mm | 5.000% | 0.00178513 | none |
| `seen_train_anchor_0p15` | `small` | `command_input_pulse` | `mid` | 0.43121 | 7.500 mm | 7.500 mm | 5.000% | 0.00141392 | none |
| `seen_train_anchor_0p15` | `small` | `command_input_pulse` | `late` | 0.970616 | 7.500 mm | 7.500 mm | 5.000% | 6.9838e-04 | none |
| `seen_train_anchor_0p15` | `small` | `initial_position_offset` | `initial_condition` | 0.0075 | 7.500 mm | 7.500 mm | 5.000% | 0.0045 | none |
| `seen_train_anchor_0p15` | `small` | `initial_velocity_offset` | `initial_condition` | 0.0128725 | 7.500 mm | 7.500 mm | 5.000% | 0.00231 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_force_state_xy` | `early` | 0.0512 | 7.500 mm | 7.500 mm | 5.000% | 0.00178513 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_force_state_xy` | `mid` | 0.0653349 | 7.500 mm | 7.500 mm | 5.000% | 0.00141392 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_force_state_xy` | `late` | 0.147063 | 7.500 mm | 7.500 mm | 5.000% | 6.9838e-04 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_integrator_xy` | `early` | 0.0114932 | 7.500 mm | 7.500 mm | 5.000% | 0.00133245 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_integrator_xy` | `mid` | 0.0176348 | 7.500 mm | 7.500 mm | 5.000% | 0.00108099 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_integrator_xy` | `late` | 0.065091 | 7.500 mm | 7.500 mm | 5.000% | 5.8060e-04 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_position_xy` | `early` | 0.0015 | 7.500 mm | 7.500 mm | 5.000% | 0.003975 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_position_xy` | `mid` | 0.0015 | 7.500 mm | 7.500 mm | 5.000% | 0.003225 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_position_xy` | `late` | 0.0015 | 7.500 mm | 7.500 mm | 5.000% | 0.001725 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_velocity_xy` | `early` | 0.00295888 | 7.500 mm | 7.500 mm | 5.000% | 0.00200584 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_velocity_xy` | `mid` | 0.00364525 | 7.500 mm | 7.500 mm | 5.000% | 0.00162531 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_velocity_xy` | `late` | 0.00689036 | 7.500 mm | 7.500 mm | 5.000% | 8.6893e-04 | none |
| `seen_train_anchor_0p15` | `moderate` | `command_input_pulse` | `early` | 0.675841 | 15.000 mm | 15.000 mm | 10.000% | 0.00357026 | none |
| `seen_train_anchor_0p15` | `moderate` | `command_input_pulse` | `mid` | 0.86242 | 15.000 mm | 15.000 mm | 10.000% | 0.00282783 | none |
| `seen_train_anchor_0p15` | `moderate` | `command_input_pulse` | `late` | 1.94123 | 15.000 mm | 15.000 mm | 10.000% | 0.00139675 | none |
| `seen_train_anchor_0p15` | `moderate` | `initial_position_offset` | `initial_condition` | 0.015 | 15.000 mm | 15.000 mm | 10.000% | 0.009 | none |
| `seen_train_anchor_0p15` | `moderate` | `initial_velocity_offset` | `initial_condition` | 0.025745 | 15.000 mm | 15.000 mm | 10.000% | 0.00462001 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_force_state_xy` | `early` | 0.1024 | 15.000 mm | 15.000 mm | 10.000% | 0.00357026 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_force_state_xy` | `mid` | 0.13067 | 15.000 mm | 15.000 mm | 10.000% | 0.00282783 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_force_state_xy` | `late` | 0.294126 | 15.000 mm | 15.000 mm | 10.000% | 0.00139675 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_integrator_xy` | `early` | 0.0229864 | 15.000 mm | 15.000 mm | 10.000% | 0.0026649 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_integrator_xy` | `mid` | 0.0352696 | 15.000 mm | 15.000 mm | 10.000% | 0.00216197 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_integrator_xy` | `late` | 0.130182 | 15.000 mm | 15.000 mm | 10.000% | 0.00116119 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_position_xy` | `early` | 0.003 | 15.000 mm | 15.000 mm | 10.000% | 0.00795 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_position_xy` | `mid` | 0.003 | 15.000 mm | 15.000 mm | 10.000% | 0.00645 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_position_xy` | `late` | 0.003 | 15.000 mm | 15.000 mm | 10.000% | 0.00345 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_velocity_xy` | `early` | 0.00591776 | 15.000 mm | 15.000 mm | 10.000% | 0.00401169 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_velocity_xy` | `mid` | 0.00729051 | 15.000 mm | 15.000 mm | 10.000% | 0.00325062 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_velocity_xy` | `late` | 0.0137807 | 15.000 mm | 15.000 mm | 10.000% | 0.00173786 | none |
| `seen_train_anchor_0p15` | `stress` | `command_input_pulse` | `early` | 1.6896 | 37.500 mm | 37.500 mm | 25.000% | 0.00892565 | none |
| `seen_train_anchor_0p15` | `stress` | `command_input_pulse` | `mid` | 2.15605 | 37.500 mm | 37.500 mm | 25.000% | 0.00706959 | none |
| `seen_train_anchor_0p15` | `stress` | `command_input_pulse` | `late` | 4.85308 | 37.500 mm | 37.500 mm | 25.000% | 0.00349188 | none |
| `seen_train_anchor_0p15` | `stress` | `initial_position_offset` | `initial_condition` | 0.0375 | 37.500 mm | 37.500 mm | 25.000% | 0.0225 | none |
| `seen_train_anchor_0p15` | `stress` | `initial_velocity_offset` | `initial_condition` | 0.0643625 | 37.500 mm | 37.500 mm | 25.000% | 0.01155 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_force_state_xy` | `early` | 0.256 | 37.500 mm | 37.500 mm | 25.000% | 0.00892565 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_force_state_xy` | `mid` | 0.326674 | 37.500 mm | 37.500 mm | 25.000% | 0.00706959 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_force_state_xy` | `late` | 0.735316 | 37.500 mm | 37.500 mm | 25.000% | 0.00349188 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_integrator_xy` | `early` | 0.0574661 | 37.500 mm | 37.500 mm | 25.000% | 0.00666225 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_integrator_xy` | `mid` | 0.088174 | 37.500 mm | 37.500 mm | 25.000% | 0.00540493 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_integrator_xy` | `late` | 0.325455 | 37.500 mm | 37.500 mm | 25.000% | 0.00290298 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_position_xy` | `early` | 0.0075 | 37.500 mm | 37.500 mm | 25.000% | 0.019875 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_position_xy` | `mid` | 0.0075 | 37.500 mm | 37.500 mm | 25.000% | 0.016125 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_position_xy` | `late` | 0.0075 | 37.500 mm | 37.500 mm | 25.000% | 0.008625 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_velocity_xy` | `early` | 0.0147944 | 37.500 mm | 37.500 mm | 25.000% | 0.0100292 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_velocity_xy` | `mid` | 0.0182263 | 37.500 mm | 37.500 mm | 25.000% | 0.00812655 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_velocity_xy` | `late` | 0.0344518 | 37.500 mm | 37.500 mm | 25.000% | 0.00434465 | none |

## Rerun Command

```bash
uv run python scripts/materialize_perturbation_open_loop_calibration.py
```

Bulk per-row data is written under `_artifacts/1ad3c16/...`; keep it out of `results/`.
