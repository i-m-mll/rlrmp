# No-PGD Calibrated Perturbation Matrix

Issue `c92ebd8` prepares a nine-row no-PGD training matrix for the final
no-hold C&S GRU set. The basis is the `const_band16` H0 row from `33b0dcb`:
`band16` means 16 held-out target directions from a 72-direction support, so
training uses 56 target directions. Rows use the 6D no-integrator process
contract, 12000 training batches per row, no PGD/adversary, and random 2D
command-input pulse directions with calibrated vector-norm amplitudes.

This directory is a pre-run lock area only. No training has been launched from
these artifacts and no pod has been acquired. Current status: this directory is
the row-planning source of truth. Open-loop calibration still has source-backed
constants plus an existing `materialize_perturbation_open_loop_calibration`
producer. Closed-loop 6D
calibration values are materialized in
`notes/closed_loop_calibration_table.json`. Rows 1-9 now have materialized
pre-run commands that select the intended generic calibration regime; every row
still requires explicit user confirmation before any billable launch.

Tracked artifacts:

- `RUN_PLAN.md`: concise pre-run matrix, analysis plan, and blockers.
- `notes/no_pgd_calibrated_perturb_matrix_regeneration_spec.json`:
  machine-readable row/spec status, including calibration-source and launch
  gates.
- `notes/closed_loop_calibration_table.json`: closed-loop 6D extLQG
  unit-sensitivity and physical-level calibration values for sensory feedback,
  command/random force pulses, and target-aligned lateral command loads.
- `scripts/materialize_closed_loop_calibration.py`: experiment-local
  materializer for the closed-loop calibration table.
