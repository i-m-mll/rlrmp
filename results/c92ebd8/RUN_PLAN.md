# No-PGD Calibrated Perturbation Matrix Run Plan

## Scope

Prepare, but do not launch, the no-PGD calibrated perturbation matrix for
`c92ebd8`. The matrix asks which calibrated perturbation-training substrate best
separates weak feedback use from robustness-like behavior in final no-hold H0
GRU rows.

## Fixed decisions

- Rows: 9 total, crossing 3 calibration regimes with 3 physical levels.
- Basis: `33b0dcb` `const_band16` H0 no-PGD row at 0.15 m reach support.
- Target support: 72 total directions; 56 training targets and 16 held-out
  directions. `band16` names the held-out directions, not the training count.
- Process contract: 6D no-integrator GRU/controller process state; every row
  command includes `--no-integrator-state`.
- Training: no PGD, no policy adversary, 12000 batches per row, no 1000-batch
  launch gate.
- Command-input training pulses: sample uniform random 2D directions and apply
  calibrated amplitude as a 2D vector norm.
- Deterministic evaluation/diagnostic perturbation banks may stay cardinal or
  target-aligned for interpretability.
- Calibration values/specs should live under `results/c92ebd8/` when they are
  data-backed; they should not be new hard-coded source constants.

## Matrix

| Row | Calibration regime | Physical level | Status | Command |
| --- | --- | --- | --- | --- |
| 1 | all families open-loop | small | `launchable_after_user_confirmation` | materialized selector command |
| 2 | all families open-loop | moderate | `launchable_after_user_confirmation` | materialized selector command |
| 3 | all families open-loop | stress | `launchable_after_user_confirmation` | materialized selector command |
| 4 | sensory feedback closed-loop; other families open-loop | small | `launchable_after_user_confirmation` | materialized selector command |
| 5 | sensory feedback closed-loop; other families open-loop | moderate | `launchable_after_user_confirmation` | materialized selector command |
| 6 | sensory feedback closed-loop; other families open-loop | stress | `launchable_after_user_confirmation` | materialized selector command |
| 7 | sensory feedback, command/random force pulses, and target-aligned lateral load closed-loop; remaining families open-loop | small | `launchable_after_user_confirmation` | materialized selector command |
| 8 | sensory feedback, command/random force pulses, and target-aligned lateral load closed-loop; remaining families open-loop | moderate | `launchable_after_user_confirmation` | materialized selector command |
| 9 | sensory feedback, command/random force pulses, and target-aligned lateral load closed-loop; remaining families open-loop | stress | `launchable_after_user_confirmation` | materialized selector command |

## Calibration status

- Open-loop rows use the extLQG nominal-command replay calibration concept.
  Existing constants are still source-backed, and the current producer
  `materialize_perturbation_open_loop_calibration` can materialize the standard
  open-loop calibration artifact. Rows 1-3 are launchable only after explicit
  user confirmation.
- Closed-loop calibration is now materialized as
  `results/c92ebd8/notes/closed_loop_calibration_table.json`, produced by
  `results/c92ebd8/scripts/materialize_closed_loop_calibration.py`. The table
  uses 6D extLQG released-forward rollouts with perturbations introduced during
  closed-loop execution and covers sensory feedback, command/random force
  pulses, and target-aligned lateral command loads at `small`, `moderate`, and
  `stress`.
- Rows 4-9 now use the generic run/spec-consumable
  `perturbation_calibration_regime` selector plus
  `closed_loop_calibration_table_path` to mix open-loop and closed-loop
  calibration sources by family. Rows 7-9 additionally enable the
  target-aligned lateral command-load family in the randomized training sampler.
- Open-loop and closed-loop calibration should share machinery where practical:
  common row metadata, physical level naming, target/reach contract, perturbation
  family definitions, and artifact schemas.
- This issue should not claim full calibration data-product migration while the
  known data-product hard block remains.

## Analysis after training

Post-training analysis should use `moderate` perturbation profiles for every
trained row, matching the useful `3244f1a` view where practical. The current
`3244f1a` materializer is reusable but hard-coded; c92 analysis needs row
parameterization/generalization before it can cover all nine rows cleanly.

Required analysis families:

- Moderate perturbation response profiles per trained row.
- Feedback-quality diagnostics.
- Robustness phenotype diagnostics.

## Launch gate

Before any RunPod acquisition or training launch, rows 1-9 still need explicit
user confirmation in the current conversation. This pre-run artifact does not
authorize pod acquisition, training launch, push, protected-branch auth, or
issue closure.
