<!-- AUTO-GENERATED: materialization_summary -->
# Delayed movement-bank PGD clip5 materialization

This note materializes the completed PGD delayed-reach row against the
non-PGD delayed movement-bank baseline on the same final-checkpoint,
calibrated perturbation-response lens.

## Run contract

| field | PGD row | baseline |
| --- | --- | --- |
| run | delayed_movement_bank_pgd_clip5 | delayed_movement_bank |
| issue | 4d79e07 | 6c36536 |
| PGD enabled | True | False |
| PGD support | movement_epoch_only | not used |
| budget scale | 3.6882404 | 1.0 |
| effective 15 cm L2 radius | 0.0045455001 | 0.00123243 |
| inner steps | 10 | not used |
| step fraction | 0.25 | not used |
| gradient clip | 5 | none |
| LR schedule | warmup_cosine | delayed_cosine |
| warmup / alpha | 500 / 0.1 | 0 / 1.0 |
| initial H0 encoder | disabled | disabled |

## Training diagnostics

| metric | value |
| --- | --- |
| completed batches | 12000 / 12000 |
| diagnostics ok | True |
| final train loss mean | 2866.79 |
| final validation loss mean | 4806.05 |
| PGD inner objective best mean | 3682.84 |
| PGD inner improvement mean | 1491.95 |
| PGD final endpoint gap mean | 0 |
| PGD boundary fraction mean | 0.571875 |
| PGD radius-ratio mean | 0.571875 |

## Delayed kinematics and hold checks

| bank | PGD peak velocity m/s | baseline peak velocity m/s | PGD change | PGD peak time s | baseline peak time s |
| --- | --- | --- | --- | --- | --- |
| no_catch | 0.7694 | 0.68859 | +11.7% | 0.17 | 0.16 |
| catch | 0 | 0 | NA | -0.1 | -0.1 |

Behavior sidecars on the PGD row: endpoint error mean 0.001838 m, overshoot mean 0.00053314 m, first-five-step command norm mean 0.034249.

## Standard certificate

| row | status | state-weighted action mismatch | classification | transition/value/Bellman |
| --- | --- | --- | --- | --- |
| PGD | partial_standard_certificate_blocked | 1.93683 | external_rollout_mismatch | not_applicable / not_applicable / not_applicable |
| baseline | partial_standard_certificate_blocked | 1.75776 | external_rollout_mismatch | not_applicable / not_applicable / not_applicable |

The shared blocker remains the 6D GRU feedback contract versus the
current 8D analytical output-feedback response-map contract.

## Perturbation response comparison

| class | PGD cost | baseline cost | PGD/base cost | PGD max dx | base max dx | PGD/extLQG | base/extLQG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| command_input/command_input_pulse | 251.35 | 1739.8 | 0.1445 | 0.0087225 | 0.012556 | 0.48588 | 3.3633 |
| command_input/target_aligned_lateral_command_load_pulse | 222.36 | 1485.1 | 0.1497 | 0.0085544 | 0.01165 | 0.42984 | 2.8708 |
| delayed_observation/delayed_observation_offset | 91.258 | 171.08 | 0.5334 | 0.0024917 | 0.0029018 | 0.00093981 | 0.0021111 |
| initial_state/initial_position_offset | 183.06 | 1319.5 | 0.1387 | 0.020005 | 0.020077 | 0.87278 | 6.291 |
| initial_state/initial_velocity_offset | 479.47 | 2014.6 | 0.238 | 0.01318 | 0.01299 | 19.753 | 82.996 |
| process_epsilon/process_epsilon_force_state_xy | 255.4 | 1725.7 | 0.148 | 0.0087241 | 0.012558 | 0.49372 | 3.336 |
| process_epsilon/process_epsilon_integrator_xy | 3168.5 | 11248 | 0.2817 | 0.013538 | 0.027186 | 12.304 | 43.678 |
| process_epsilon/process_epsilon_position_xy | 462.09 | 2110.9 | 0.2189 | 0.02 | 0.020028 | 0.17491 | 0.79902 |
| process_epsilon/process_epsilon_velocity_xy | 316.04 | 2319.2 | 0.1363 | 0.010489 | 0.014139 | 0.10266 | 0.75338 |
| sensory_feedback/sensory_feedback_offset | 91.258 | 171.08 | 0.5334 | 0.0024917 | 0.0029018 | 0.00093981 | 0.0021111 |
| target_stream/target_stream_jump | NA | NA | NA | NA | NA | NA | NA |

Full class comparison CSV: `_artifacts/4d79e07/comparisons/delayed_movement_bank_pgd_clip5_vs_delayed_movement_bank/perturbation_class_comparison.csv`.

## Interpretation

- Movement-only PGD made the fixed-bank no-catch reach faster than the
  non-PGD delayed baseline, while catch/hold trials stayed flat at zero
  forward velocity on the fixed bank.
- PGD reduced full-Q/R/Q_f perturbation delta cost across every
  comparable perturbation class in this bank, while action-response
  norms increased on several command, observation, and sensory classes.
- The standard empirical/nonlinear certificate remains a partial blocked
  certificate, and the PGD row has a modestly larger clean-rollout action
  mismatch than the non-PGD delayed baseline.

## Artifact size cleanup

| scope | before | current |
| --- | --- | --- |
| all 4d79e07 artifacts | 3.39 GiB | 450.50 MiB |
| PGD perturbation bulk | 2.98 GiB | 27.68 MiB |
| raw perturbation NPZ caches | 438 files / 2.95 GiB | 0 files / 0 B |

## Residual blockers

- Objective comparator was skipped for this final-checkpoint lens; the existing comparator expects validation-selected checkpoints.
- Map decomposition was skipped by the upstream materializer because the delayed-bank rollout arrays broadcast as (1, 60, 2, 240) versus requested (80, 90, 2, 240).
- Feedback ablation was skipped because calibrated force/filter feedback rows require a controller-feedback scale manifest wiring path.
- Standard response-map components remain blocked by the 6D GRU feedback versus 8D analytical output-feedback contract.
<!-- /AUTO-GENERATED -->
