# Soft Lambda Scale Sanity and No-Launch Spec Lock

Issue: `27dece3`. Parent umbrella: `54389a4`.

> No training launch is approved by this artifact. This is a durable no-launch
> spec lock for parent/user review. It proposes training-facing rows and
> optimizer settings, but a billable run still requires an explicit later
> user approval.

## Scale Contract

- Source frozen runs: `c92ebd8` c92 open-loop no-PGD rows.
- Controller weights were frozen; these are audits, not training runs.
- Epsilon is a 6D process input with metadata `B_w[:epsilon_dim, :] = I` and the remaining state rows zero.
- Safety cap radius: `0.0045455` coordinate-L2 units from `ofb_6d_no_integrator_gamma_1p4_rollout_radius`.
- This cap is `3.03%` of a 15 cm reference length if read as a coordinate displacement scale.
- The code path does not provide a Newton or muscle-force conversion for this
  epsilon. Treat the values below as process-coordinate scale evidence, not an
  exact physical force calibration.
- Objective: `mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]`, where energy
  is squared L2 over the epsilon sequence.

## Practical Lambda Sanity

Lowest valid means the smallest tested multiplier that was finite, useful,
and interior (`cap_bound_fraction = 0` and `max_norm_over_cap <= 0.99`).

| row | mechanism | optimizer/status | selected lambda multiplier | gain | norm/cap behavior | scale interpretation |
|---|---|---|---:|---:|---|---|
| `c92 small` | direct epsilon | `pgd_projected_epsilon` / bounded_optimizer_materialized_valid_point | 2.83x | 700 | max 0.746x cap; cap-bound 0.0% | coordinate perturbation becomes useful and interior near this multiplier; this is the direct frozen-audit scale reference |
| `c92 small` | closed-loop linear no-bias | `lbfgsb` / bounded_optimizer_materialized_valid_point | 2x | 734 | max 0.911x cap; cap-bound 0.0% | full closed-loop no-bias policy has useful interior points in the same lambda region; f3c5db9 found matching zero-start Adam settings |
| `c92 small` | closed-loop affine | `lbfgsb` / bounded_optimizer_materialized_valid_point | 2.18x | 608 | max 0.752x cap; cap-bound 0.0% | affine has useful interior frozen-audit points; f3c5db9 found matching zero-start Adam settings |
| `c92 moderate` | direct epsilon | `pgd_projected_epsilon` / bounded_optimizer_materialized_valid_point | 3.08x | 384 | max 0.492x cap; cap-bound 0.0% | coordinate perturbation becomes useful and interior near this multiplier; this is the direct frozen-audit scale reference |
| `c92 moderate` | closed-loop linear no-bias | `lbfgsb` / bounded_optimizer_materialized_valid_point | 2x | 539 | max 0.893x cap; cap-bound 0.0% | full closed-loop no-bias policy has useful interior points in the same lambda region; f3c5db9 found matching zero-start Adam settings |
| `c92 moderate` | closed-loop affine | `lbfgsb` / bounded_optimizer_materialized_valid_point | 2.18x | 583 | max 0.865x cap; cap-bound 0.0% | affine has useful interior frozen-audit points; f3c5db9 found matching zero-start Adam settings |
| `c92 stress` | direct epsilon | `pgd_projected_epsilon` / bounded_optimizer_materialized_valid_point | 2.18x | 490 | max 0.745x cap; cap-bound 0.0% | coordinate perturbation becomes useful and interior near this multiplier; this is the direct frozen-audit scale reference |
| `c92 stress` | closed-loop linear no-bias | `lbfgsb` / bounded_optimizer_materialized_valid_point | 1.41x | 542 | max 0.831x cap; cap-bound 0.0% | full closed-loop no-bias policy has useful interior points in the same lambda region; f3c5db9 found matching zero-start Adam settings |
| `c92 stress` | closed-loop affine | `lbfgsb` / bounded_optimizer_materialized_valid_point | 1.68x | 525 | max 0.853x cap; cap-bound 0.0% | affine has useful interior frozen-audit points; f3c5db9 found matching zero-start Adam settings |

## Current Adam Status

`f3c5db9` resolves the previous optimizer gate: Stage 1 zero-start Adam
matches all direct, linear, and affine reference regions. The conservative
common setting for training-facing smoke tests is `steps=12`, `lr=1e-5`.

| row | mechanism | lambda range | Adam setting | finite/useful/interior | norm/cap | gain | recommendation |
|---|---|---|---|---|---:|---:|---|
| `c92 small` | direct epsilon | 2.59368x-2.82843x; reference=2.82843x | `steps=64; lr=1.0e-05; init=zero` | match | 0.833x cap; cap-bound 0.0% | 775 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 small` | closed-loop linear no-bias | 1.83401x-2x; reference=2x | `steps=12; lr=1.0e-05; init=zero` | match | 0.558x cap; cap-bound 0.0% | 585 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 small` | closed-loop affine | 2x-2.18102x; reference=2.18102x | `steps=12; lr=1.0e-05; init=zero` | match | 0.613x cap; cap-bound 0.0% | 635 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 moderate` | direct epsilon | 2.82843x-3.08442x; reference=3.08442x | `steps=128; lr=3.0e-05; init=zero` | match | 0.713x cap; cap-bound 0.0% | 449 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 moderate` | closed-loop linear no-bias | 1.83401x-2x; reference=2x | `steps=12; lr=1.0e-05; init=zero` | match | 0.53x cap; cap-bound 0.0% | 458 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 moderate` | closed-loop affine | 2x-2.18102x; reference=2.18102x | `steps=12; lr=1.0e-05; init=zero` | match | 0.568x cap; cap-bound 0.0% | 511 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 stress` | direct epsilon | 2x-2.18102x; reference=2.18102x | `steps=128; lr=1.0e-04; init=zero` | match | 0.875x cap; cap-bound 0.0% | 567 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 stress` | closed-loop linear no-bias | 1.29684x-1.41421x; reference=1.41421x | `steps=12; lr=1.0e-05; init=zero` | match | 0.521x cap; cap-bound 0.0% | 476 | Adam stage 1 is reliable enough for this frozen-audit row. |
| `c92 stress` | closed-loop affine | 1.54221x-1.68179x; reference=1.68179x | `steps=12; lr=1.0e-05; init=zero` | match | 0.593x cap; cap-bound 0.0% | 506 | Adam stage 1 is reliable enough for this frozen-audit row. |

## Known-Direction Checks

Known-direction rows are expressivity sanity checks from `3b850d6`; they show
that useful finite closed-loop directions exist, but they are not a full
inner-optimizer choice for training.

| row | mechanism | optimizer/status | selected lambda multiplier | gain | norm/cap behavior | scale interpretation |
|---|---|---|---:|---:|---|---|
| `c92 small` | closed-loop linear no-bias known direction | `line_search_known_direction` / reference_direction_only | 2x | 1.02e+03 | max 0.685x cap; cap-bound 0.0% | known-direction expressivity check only; useful for sanity, not a training inner-optimizer recommendation |
| `c92 small` | closed-loop affine known direction | `line_search_known_direction` / reference_direction_only | 2x | 1.3e+03 | max 1.1x cap; cap-bound 100.0% | known-direction expressivity check only; useful for sanity, not a training inner-optimizer recommendation |
| `c92 moderate` | closed-loop linear no-bias known direction | `line_search_known_direction` / reference_direction_only | 2x | 769 | max 0.7x cap; cap-bound 0.0% | known-direction expressivity check only; useful for sanity, not a training inner-optimizer recommendation |
| `c92 moderate` | closed-loop affine known direction | `line_search_known_direction` / reference_direction_only | 2x | 1.18e+03 | max 1.23x cap; cap-bound 100.0% | known-direction expressivity check only; useful for sanity, not a training inner-optimizer recommendation |
| `c92 stress` | closed-loop linear no-bias known direction | `line_search_known_direction` / reference_direction_only | 2x | 349 | max 0.517x cap; cap-bound 0.0% | known-direction expressivity check only; useful for sanity, not a training inner-optimizer recommendation |
| `c92 stress` | closed-loop affine known direction | `line_search_known_direction` / reference_direction_only | 2x | 212 | max 0.402x cap; cap-bound 0.0% | known-direction expressivity check only; useful for sanity, not a training inner-optimizer recommendation |

## Proposed Training-Facing Rows

These rows are proposed for the approval discussion only. They must not be
launched without an explicit user-approved run spec.

| proposed row | mechanism | training inner optimizer | multipliers by c92 row | Adam setting | status |
|---|---|---|---|---|---|
| `direct_epsilon_calibrated` | direct epsilon | `zero_start_adam` | c92 small=2.83x, c92 moderate=3.08x, c92 stress=2.18x | `steps=12; lr=1.0e-05` | candidate_no_launch_adam_smoke |
| `linear_no_bias_calibrated` | closed-loop linear no-bias | `zero_start_adam` | c92 small=2x, c92 moderate=2x, c92 stress=1.41x | `steps=12; lr=1.0e-05` | candidate_no_launch_adam_smoke |
| `affine_calibrated` | closed-loop affine | `zero_start_adam` | c92 small=2.18x, c92 moderate=2.18x, c92 stress=1.68x | `steps=12; lr=1.0e-05` | candidate_no_launch_adam_smoke |
| `fallback_benchmark_if_training_unreliable` | closed-loop affine | `adam_vs_optax_lbfgs_benchmark` | c92 small=2.18x, c92 moderate=2.18x, c92 stress=1.68x | `steps=12; lr=1.0e-05` | future_path_only |

If Adam fails during training or the affine row remains unreliable after
the frozen-audit match, future benchmark issue `2e60620` is the planned place for the
Adam-vs-Optax-L-BFGS comparison. This artifact only references that issue;
it does not add a comment there.

## Dependency Status

- `results/f3c5db9/` is present: `True`.
- The sibling Adam closeout has been consumed through the tracked
  `results/f3c5db9/frozen_adam_audit_tuning.json` artifact.
- This spec is finalized as a no-launch approval packet; it is not launch
  authorization.
