# Output-feedback analytical damage recursion comparison

## Contract

- Reference: 6D no-integrator output-feedback H-infinity teacher.
- Gamma: 12833.5637997 = 1.4 * gamma_star (9166.83128547).
- Joint state: `z = [x, xhat]`; clean condition uses `F = 0`; adversarial condition uses `epsilon_t = F_t z_t`.
- Reported damage is task cost only: Q/R/Qf state/control/terminal cost, with no `gamma^2 ||epsilon||^2` subtraction.
- Deterministic recursion is exact quadratic cost-to-go. The noisy single-seed recursion uses the same seed and implements the known time-varying signal-dependent multiplicative term plus affine sensory, motor, and process noise offsets.

## Headline Comparison

| Mode | Quantity | Rollout | Recursion | Abs diff | Relative diff |
|---|---|---:|---:|---:|---:|
| deterministic_noise_off | clean_cost | 4441.49067466 | 4441.49067466 | 1.27329258248e-11 | 2.86681358974e-15 |
| deterministic_noise_off | adversarial_cost | 8146.45394348 | 8146.45394348 | 1.64709490491e-09 | 2.02185505048e-13 |
| deterministic_noise_off | paired_damage | 3704.96326882 | 3704.96326882 | 1.65982783074e-09 | 4.48001156909e-13 |
| nominal_noise_paired_single_seed | clean_cost | 4755.43667425 | 4755.43667425 | 1.4370016288e-10 | 3.02180793739e-14 |
| nominal_noise_paired_single_seed | adversarial_cost | 10887.1273507 | 10887.1273507 | 2.04636307899e-09 | 1.87961710474e-13 |
| nominal_noise_paired_single_seed | paired_damage | 6131.6906765 | 6131.6906765 | 2.19006324187e-09 | 3.57171187755e-13 |

## Disturbance Energy

| Mode | Rollout | Recursion | Abs diff | Relative diff |
|---|---:|---:|---:|---:|
| deterministic_noise_off | 2.06571286822e-05 | 2.06571286822e-05 | 3.38813178902e-21 | 1.64017557384e-16 |
| nominal_noise_paired_single_seed | 3.42666348969e-05 | 3.42666348969e-05 | 1.95834017405e-18 | 5.7150058065e-14 |

## Assessment

- Match status: `matches_rollout_within_float_tolerance`.
- Second iteration recommended: `False`.
- Reason: The deterministic and paired single-seed noisy affine recursions reproduce the rollout costs, damage, and disturbance energies to floating-point precision.

## Verification Notes

- Teacher package check: `checked`.
- Maximum absolute mismatch observed across headline values: 2.19006324187e-09.
