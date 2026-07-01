# Output-feedback analytical damage recursion comparison at beta 1.05

## Contract

- Reference: 6D no-integrator output-feedback H-infinity teacher.
- Gamma: 9625.17284975 = 1.05 * gamma_star (9166.83128547).
- Joint state: `z = [x, xhat]`; clean condition uses `F = 0`; adversarial condition uses `epsilon_t = F_t z_t`.
- Reported damage is task cost only: Q/R/Qf state/control/terminal cost, with no `gamma^2 ||epsilon||^2` subtraction.
- Deterministic recursion is exact quadratic cost-to-go. The noisy single-seed recursion uses the same seed and implements the known time-varying signal-dependent multiplicative term plus affine sensory, motor, and process noise offsets.

## Headline Comparison

| Mode | Quantity | Rollout | Recursion | Abs diff | Relative diff |
|---|---|---:|---:|---:|---:|
| deterministic_noise_off | clean_cost | 4570.2937876 | 4570.2937876 | 3.30146576744e-10 | 7.22374954625e-14 |
| deterministic_noise_off | adversarial_cost | 5018.18418996 | 5018.18418996 | 6.17546902504e-10 | 1.23061824582e-13 |
| deterministic_noise_off | paired_damage | 447.890402367 | 447.890402368 | 9.47693479247e-10 | 2.11590486029e-12 |
| nominal_noise_paired_single_seed | clean_cost | 6456.47644404 | 6456.47644404 | 5.00222085975e-11 | 7.74760181209e-15 |
| nominal_noise_paired_single_seed | adversarial_cost | 8368.36954118 | 8368.36954118 | 4.92946128361e-10 | 5.89058747866e-14 |
| nominal_noise_paired_single_seed | paired_damage | 1911.89309715 | 1911.89309715 | 5.42968336958e-10 | 2.83995134335e-13 |

## Disturbance Energy

| Mode | Rollout | Recursion | Abs diff | Relative diff |
|---|---:|---:|---:|---:|
| deterministic_noise_off | 3.06716551679e-06 | 3.06716551679e-06 | 4.57397791517e-20 | 1.49127195456e-14 |
| nominal_noise_paired_single_seed | 2.04621073082e-05 | 2.04621073082e-05 | 2.30392961653e-19 | 1.12594933739e-14 |

## Assessment

- Match status: `matches_rollout_within_float_tolerance`.
- Second iteration recommended: `False`.
- Reason: The deterministic and paired single-seed noisy affine recursions reproduce the rollout costs, damage, and disturbance energies to floating-point precision.

## Verification Notes

- Teacher package check: `checked_beta_invariant_geometry_only`.
- Maximum absolute mismatch observed across headline values: 9.47693479247e-10.
