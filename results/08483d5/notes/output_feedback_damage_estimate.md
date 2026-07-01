# 6D output-feedback H-infinity rollout damage estimate

## Source

- Teacher package: `_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz`
- Teacher manifest: `_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers_manifest.json`
- Live functions: `build_no_integrator_game`, `find_gamma_star`, `solve_hinf_riccati`, `robust_estimator_covariances`, `robust_output_feedback_gains`, `robust_estimator_fixed_adversary_policy`, `rollout_with_robust_estimator`, `rollout_with_robust_estimator_policy`.

## Contract

- State basis: 36D delay-augmented state = 6 delayed physical blocks of [x, y, vx, vy, force_x, force_y]; no disturbance-integrator coordinates; observation is the oldest delayed 6D physical block.
- Horizon: 60 steps at 0.01 s.
- Target convention: goal-centered C&S state with INIT_POS=[0,0], TARGET_POS=[0.15,0]; x0 repeats [-0.15, 0, 0, 0, 0, 0] across all six delay blocks.
- Q/R/Qf: C&S Eq. 15 no-integrator 6D schedule: Q shape (60, 36, 36), R shape (60, 2, 2), Qf shape (36, 36).
- Gamma ratio: 1.4 (gamma_star=9166.83128547, gamma=12833.5637997).
- F application: Matched optimal fixed policy F_t from robust_estimator_fixed_adversary_policy; applied each step as epsilon_t = F_t @ concat([x_t, xhat_t]) through plant.Bw. The clean condition uses the same controller/estimator with epsilon_t=0.

## Results

| Mode | Clean cost | Adversarial cost | Paired damage | Disturbance energy | H-inf objective delta |
|---|---:|---:|---:|---:|---:|
| deterministic_noise_off | 4441.49067466 | 8146.45394348 | 3704.96326882 | 2.06571286822e-05 | 302.726742426 |
| nominal_noise_paired | 4755.43667425 | 10887.1273507 | 6131.6906765 | 3.42666348969e-05 | 487.963579838 |

## Cost Components

| Mode | Condition | State stage | Control stage | Terminal | Peak forward velocity | Terminal error |
|---|---|---:|---:|---:|---:|---:|
| deterministic_noise_off | clean | 2566.55281458 | 1874.51594052 | 0.421919557294 | 0.760769041963 | 0.00072968061219 |
| deterministic_noise_off | adversarial | 5940.35914632 | 1828.6854867 | 377.409310464 | 0.764970739461 | 0.0220461661732 |
| nominal_noise_paired | clean | 2661.70235245 | 2090.99753494 | 2.73678685281 | 0.761394521554 | 0.00158680510925 |
| nominal_noise_paired | adversarial | 8208.08532085 | 2044.97036747 | 634.07166243 | 0.765553465379 | 0.028493110379 |

## Noise

- Deterministic rollout: noise off.
- Stochastic rollout: nominal C&S released-code noise terms available and paired with seed `376023`; clean and adversarial conditions use the same sensory, motor, process, and signal-dependent standard draws.

## Verification

- Teacher package verification status: `checked`.
- Recomputed live arrays match the stored teacher package at rtol/atol 1e-9.
