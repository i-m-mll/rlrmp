# Perturbation Open-Loop Calibration

- Issue: `1ad3c16`
- Scope: `extlqg_nominal_command_open_loop_physical_effect_calibration`
- Open-loop reference: extLQG nominal command replay.
- Closed-loop extLQG is reported at the same amplitudes where supported.
- Calibration mode: reach-relative peak `delta x`, with target peak `delta x = fraction * reach_length`.
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

## Selected Reach-Relative Amplitudes

| Reach | Level | Family | Amplitude | Target peak dx | Achieved peak dx | Achieved % reach | AUC dx | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| `heldout_eval_0p12` | `small` | `command_input_pulse` | 0.400615 | 6.000 mm | 6.000 mm | 5.000% | 9.8455e-04 | none |
| `heldout_eval_0p12` | `small` | `initial_position_offset` | 0.006 | 6.000 mm | 6.000 mm | 5.000% | 0.0036 | none |
| `heldout_eval_0p12` | `small` | `initial_velocity_offset` | 0.010298 | 6.000 mm | 6.000 mm | 5.000% | 0.001848 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_force_state_xy` | 0.0606992 | 6.000 mm | 6.000 mm | 5.000% | 9.8455e-04 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_integrator_xy` | 0.0182022 | 6.000 mm | 6.000 mm | 5.000% | 7.6440e-04 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_position_xy` | 0.0012 | 6.000 mm | 6.000 mm | 5.000% | 0.00228 | none |
| `heldout_eval_0p12` | `small` | `process_epsilon_velocity_xy` | 0.00330208 | 6.000 mm | 6.000 mm | 5.000% | 0.00114846 | none |
| `heldout_eval_0p12` | `moderate` | `command_input_pulse` | 0.80123 | 12.000 mm | 12.000 mm | 10.000% | 0.00196909 | none |
| `heldout_eval_0p12` | `moderate` | `initial_position_offset` | 0.012 | 12.000 mm | 12.000 mm | 10.000% | 0.0072 | none |
| `heldout_eval_0p12` | `moderate` | `initial_velocity_offset` | 0.020596 | 12.000 mm | 12.000 mm | 10.000% | 0.00369601 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_force_state_xy` | 0.121398 | 12.000 mm | 12.000 mm | 10.000% | 0.00196909 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_integrator_xy` | 0.0364044 | 12.000 mm | 12.000 mm | 10.000% | 0.00152881 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_position_xy` | 0.0024 | 12.000 mm | 12.000 mm | 10.000% | 0.00456 | none |
| `heldout_eval_0p12` | `moderate` | `process_epsilon_velocity_xy` | 0.00660416 | 12.000 mm | 12.000 mm | 10.000% | 0.00229693 | none |
| `heldout_eval_0p12` | `stress` | `command_input_pulse` | 2.00307 | 30.000 mm | 30.000 mm | 25.000% | 0.00492274 | none |
| `heldout_eval_0p12` | `stress` | `initial_position_offset` | 0.03 | 30.000 mm | 30.000 mm | 25.000% | 0.018 | none |
| `heldout_eval_0p12` | `stress` | `initial_velocity_offset` | 0.05149 | 30.000 mm | 30.000 mm | 25.000% | 0.00924001 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_force_state_xy` | 0.303496 | 30.000 mm | 30.000 mm | 25.000% | 0.00492274 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_integrator_xy` | 0.0910111 | 30.000 mm | 30.000 mm | 25.000% | 0.00382202 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_position_xy` | 0.006 | 30.000 mm | 30.000 mm | 25.000% | 0.0114 | none |
| `heldout_eval_0p12` | `stress` | `process_epsilon_velocity_xy` | 0.0165104 | 30.000 mm | 30.000 mm | 25.000% | 0.00574232 | none |
| `heldout_eval_0p18` | `small` | `command_input_pulse` | 0.600922 | 9.000 mm | 9.000 mm | 5.000% | 0.00147682 | none |
| `heldout_eval_0p18` | `small` | `initial_position_offset` | 0.009 | 9.000 mm | 9.000 mm | 5.000% | 0.0054 | none |
| `heldout_eval_0p18` | `small` | `initial_velocity_offset` | 0.015447 | 9.000 mm | 9.000 mm | 5.000% | 0.002772 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_force_state_xy` | 0.0910488 | 9.000 mm | 9.000 mm | 5.000% | 0.00147682 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_integrator_xy` | 0.0273033 | 9.000 mm | 9.000 mm | 5.000% | 0.00114661 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_position_xy` | 0.0018 | 9.000 mm | 9.000 mm | 5.000% | 0.00342 | none |
| `heldout_eval_0p18` | `small` | `process_epsilon_velocity_xy` | 0.00495312 | 9.000 mm | 9.000 mm | 5.000% | 0.0017227 | none |
| `heldout_eval_0p18` | `moderate` | `command_input_pulse` | 1.20184 | 18.000 mm | 18.000 mm | 10.000% | 0.00295364 | none |
| `heldout_eval_0p18` | `moderate` | `initial_position_offset` | 0.018 | 18.000 mm | 18.000 mm | 10.000% | 0.0108 | none |
| `heldout_eval_0p18` | `moderate` | `initial_velocity_offset` | 0.030894 | 18.000 mm | 18.000 mm | 10.000% | 0.00554401 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_force_state_xy` | 0.182098 | 18.000 mm | 18.000 mm | 10.000% | 0.00295364 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_integrator_xy` | 0.0546067 | 18.000 mm | 18.000 mm | 10.000% | 0.00229321 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_position_xy` | 0.0036 | 18.000 mm | 18.000 mm | 10.000% | 0.00684 | none |
| `heldout_eval_0p18` | `moderate` | `process_epsilon_velocity_xy` | 0.00990624 | 18.000 mm | 18.000 mm | 10.000% | 0.00344539 | none |
| `heldout_eval_0p18` | `stress` | `command_input_pulse` | 3.00461 | 45.000 mm | 45.000 mm | 25.000% | 0.0073841 | none |
| `heldout_eval_0p18` | `stress` | `initial_position_offset` | 0.045 | 45.000 mm | 45.000 mm | 25.000% | 0.027 | none |
| `heldout_eval_0p18` | `stress` | `initial_velocity_offset` | 0.077235 | 45.000 mm | 45.000 mm | 25.000% | 0.01386 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_force_state_xy` | 0.455244 | 45.000 mm | 45.000 mm | 25.000% | 0.0073841 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_integrator_xy` | 0.136517 | 45.000 mm | 45.000 mm | 25.000% | 0.00573303 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_position_xy` | 0.009 | 45.000 mm | 45.000 mm | 25.000% | 0.0171 | none |
| `heldout_eval_0p18` | `stress` | `process_epsilon_velocity_xy` | 0.0247656 | 45.000 mm | 45.000 mm | 25.000% | 0.00861349 | none |
| `seen_train_0p10` | `small` | `command_input_pulse` | 0.333846 | 5.000 mm | 5.000 mm | 5.000% | 8.2046e-04 | none |
| `seen_train_0p10` | `small` | `initial_position_offset` | 0.005 | 5.000 mm | 5.000 mm | 5.000% | 0.003 | none |
| `seen_train_0p10` | `small` | `initial_velocity_offset` | 0.00858167 | 5.000 mm | 5.000 mm | 5.000% | 0.00154 | none |
| `seen_train_0p10` | `small` | `process_epsilon_force_state_xy` | 0.0505827 | 5.000 mm | 5.000 mm | 5.000% | 8.2046e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_integrator_xy` | 0.0151685 | 5.000 mm | 5.000 mm | 5.000% | 6.3700e-04 | none |
| `seen_train_0p10` | `small` | `process_epsilon_position_xy` | 0.001 | 5.000 mm | 5.000 mm | 5.000% | 0.0019 | none |
| `seen_train_0p10` | `small` | `process_epsilon_velocity_xy` | 0.00275173 | 5.000 mm | 5.000 mm | 5.000% | 9.5705e-04 | none |
| `seen_train_0p10` | `moderate` | `command_input_pulse` | 0.667691 | 10.000 mm | 10.000 mm | 10.000% | 0.00164091 | none |
| `seen_train_0p10` | `moderate` | `initial_position_offset` | 0.01 | 10.000 mm | 10.000 mm | 10.000% | 0.006 | none |
| `seen_train_0p10` | `moderate` | `initial_velocity_offset` | 0.0171633 | 10.000 mm | 10.000 mm | 10.000% | 0.00308 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_force_state_xy` | 0.101165 | 10.000 mm | 10.000 mm | 10.000% | 0.00164091 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_integrator_xy` | 0.030337 | 10.000 mm | 10.000 mm | 10.000% | 0.00127401 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_position_xy` | 0.002 | 10.000 mm | 10.000 mm | 10.000% | 0.0038 | none |
| `seen_train_0p10` | `moderate` | `process_epsilon_velocity_xy` | 0.00550347 | 10.000 mm | 10.000 mm | 10.000% | 0.00191411 | none |
| `seen_train_0p10` | `stress` | `command_input_pulse` | 1.66923 | 25.000 mm | 25.000 mm | 25.000% | 0.00410228 | none |
| `seen_train_0p10` | `stress` | `initial_position_offset` | 0.025 | 25.000 mm | 25.000 mm | 25.000% | 0.015 | none |
| `seen_train_0p10` | `stress` | `initial_velocity_offset` | 0.0429083 | 25.000 mm | 25.000 mm | 25.000% | 0.00770001 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_force_state_xy` | 0.252913 | 25.000 mm | 25.000 mm | 25.000% | 0.00410228 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_integrator_xy` | 0.0758426 | 25.000 mm | 25.000 mm | 25.000% | 0.00318502 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_position_xy` | 0.005 | 25.000 mm | 25.000 mm | 25.000% | 0.0095 | none |
| `seen_train_0p10` | `stress` | `process_epsilon_velocity_xy` | 0.0137587 | 25.000 mm | 25.000 mm | 25.000% | 0.00478527 | none |
| `seen_train_anchor_0p15` | `small` | `command_input_pulse` | 0.500768 | 7.500 mm | 7.500 mm | 5.000% | 0.00123068 | none |
| `seen_train_anchor_0p15` | `small` | `initial_position_offset` | 0.0075 | 7.500 mm | 7.500 mm | 5.000% | 0.0045 | none |
| `seen_train_anchor_0p15` | `small` | `initial_velocity_offset` | 0.0128725 | 7.500 mm | 7.500 mm | 5.000% | 0.00231 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_force_state_xy` | 0.075874 | 7.500 mm | 7.500 mm | 5.000% | 0.00123068 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_integrator_xy` | 0.0227528 | 7.500 mm | 7.500 mm | 5.000% | 9.5551e-04 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_position_xy` | 0.0015 | 7.500 mm | 7.500 mm | 5.000% | 0.00285 | none |
| `seen_train_anchor_0p15` | `small` | `process_epsilon_velocity_xy` | 0.0041276 | 7.500 mm | 7.500 mm | 5.000% | 0.00143558 | none |
| `seen_train_anchor_0p15` | `moderate` | `command_input_pulse` | 1.00154 | 15.000 mm | 15.000 mm | 10.000% | 0.00246137 | none |
| `seen_train_anchor_0p15` | `moderate` | `initial_position_offset` | 0.015 | 15.000 mm | 15.000 mm | 10.000% | 0.009 | none |
| `seen_train_anchor_0p15` | `moderate` | `initial_velocity_offset` | 0.025745 | 15.000 mm | 15.000 mm | 10.000% | 0.00462001 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_force_state_xy` | 0.151748 | 15.000 mm | 15.000 mm | 10.000% | 0.00246137 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_integrator_xy` | 0.0455056 | 15.000 mm | 15.000 mm | 10.000% | 0.00191101 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_position_xy` | 0.003 | 15.000 mm | 15.000 mm | 10.000% | 0.0057 | none |
| `seen_train_anchor_0p15` | `moderate` | `process_epsilon_velocity_xy` | 0.0082552 | 15.000 mm | 15.000 mm | 10.000% | 0.00287116 | none |
| `seen_train_anchor_0p15` | `stress` | `command_input_pulse` | 2.50384 | 37.500 mm | 37.500 mm | 25.000% | 0.00615342 | none |
| `seen_train_anchor_0p15` | `stress` | `initial_position_offset` | 0.0375 | 37.500 mm | 37.500 mm | 25.000% | 0.0225 | none |
| `seen_train_anchor_0p15` | `stress` | `initial_velocity_offset` | 0.0643625 | 37.500 mm | 37.500 mm | 25.000% | 0.01155 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_force_state_xy` | 0.37937 | 37.500 mm | 37.500 mm | 25.000% | 0.00615342 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_integrator_xy` | 0.113764 | 37.500 mm | 37.500 mm | 25.000% | 0.00477753 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_position_xy` | 0.0075 | 37.500 mm | 37.500 mm | 25.000% | 0.01425 | none |
| `seen_train_anchor_0p15` | `stress` | `process_epsilon_velocity_xy` | 0.020638 | 37.500 mm | 37.500 mm | 25.000% | 0.0071779 | none |

## Rerun Command

```bash
uv run python scripts/materialize_perturbation_open_loop_calibration.py
```

Bulk per-row data is written under `_artifacts/1ad3c16/...`; keep it out of `results/`.
