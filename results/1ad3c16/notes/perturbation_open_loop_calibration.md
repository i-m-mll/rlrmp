# Perturbation Open-Loop Calibration

- Issue: `1ad3c16`
- Scope: `extlqg_nominal_command_open_loop_physical_effect_calibration`
- Open-loop reference: extLQG nominal command replay.
- Closed-loop extLQG is reported at the same amplitudes where supported.
- GRU baseline for later closed-loop calibration: `5f70333` / `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` (single-target nominal-only).
- Bulk row manifest: `_artifacts/1ad3c16/perturbation_open_loop_calibration/perturbation_open_loop_calibration.json`

## Selected Amplitude Candidates

| Family | Small amp | Small peak dx | Moderate amp | Moderate peak dx | Strong amp | Strong peak dx | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `command_input_pulse` | 0.2 | 0.0029954 | 1 | 0.014977 | 2 | 0.029954 | none |
| `initial_position_offset` | 0.002 | 0.002 | 0.01 | 0.01 | 0.02 | 0.02 | none |
| `initial_velocity_offset` | 0.005 | 0.00291319 | 0.025 | 0.0145659 | 0.05 | 0.0291319 | none |
| `process_epsilon_force_state_xy` | 0.1 | 0.00354741 | 2 | 0.0167719 | 1 | 0.0354741 | none |
| `process_epsilon_integrator_xy` | 0.05 | 0.00340766 | 0.2 | 0.0136306 | 0.5 | 0.0340766 | none |
| `process_epsilon_position_xy` | 5.0000e-04 | 0.0025 | 0.002 | 0.01 | 0.005 | 0.025 | none |
| `process_epsilon_velocity_xy` | 0.01 | 0.00348902 | 0.02 | 0.0168637 | 0.1 | 0.0348902 | none |
