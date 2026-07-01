# 6D output-feedback H-infinity rollout damage estimate at beta 1.05

## Source

- Teacher package: `_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz`
- Teacher manifest: `_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers_manifest.json`
- Live functions: `build_no_integrator_game`, `find_gamma_star`, `solve_hinf_riccati`, `robust_estimator_covariances`, `robust_output_feedback_gains`, `robust_estimator_fixed_adversary_policy`, `rollout_with_robust_estimator`, `rollout_with_robust_estimator_policy`.

## Contract

- State basis: 36D delay-augmented state = 6 delayed physical blocks of [x, y, vx, vy, force_x, force_y]; no disturbance-integrator coordinates; observation is the oldest delayed 6D physical block.
- Horizon: 60 steps at 0.01 s.
- Target convention: goal-centered C&S state with INIT_POS=[0,0], TARGET_POS=[0.15,0]; x0 repeats [-0.15, 0, 0, 0, 0, 0] across all six delay blocks.
- Q/R/Qf: C&S Eq. 15 no-integrator 6D schedule: Q shape (60, 36, 36), R shape (60, 2, 2), Qf shape (36, 36).
- Gamma ratio: 1.05 (gamma_star=9166.83128547, gamma=9625.17284975).
- F application: Matched optimal fixed policy F_t from robust_estimator_fixed_adversary_policy; applied each step as epsilon_t = F_t @ concat([x_t, xhat_t]) through plant.Bw. The clean condition uses the same controller/estimator with epsilon_t=0.

## Results

| Mode | Clean cost | Adversarial cost | Paired damage | Disturbance energy | H-inf objective delta |
|---|---:|---:|---:|---:|---:|
| deterministic_noise_off | 4570.2937876 | 5018.18418996 | 447.890402367 | 3.06716551679e-06 | 163.736066265 |
| nominal_noise_paired | 6456.47644404 | 8368.36954118 | 1911.89309715 | 2.04621073082e-05 | 16.2026019335 |

## Cost Components

| Mode | Condition | State stage | Control stage | Terminal | Peak forward velocity | Terminal error |
|---|---|---:|---:|---:|---:|---:|
| deterministic_noise_off | clean | 2503.74919378 | 2066.54451961 | 7.4203974635e-05 | 0.785204359391 | 9.63112251742e-06 |
| deterministic_noise_off | adversarial | 3020.62737975 | 1996.7394118 | 0.817398415784 | 0.793002368671 | 0.00103386688668 |
| nominal_noise_paired | clean | 2771.12758405 | 3682.04195456 | 3.30690543068 | 0.785836111378 | 0.00185727497709 |
| nominal_noise_paired | adversarial | 4575.78385687 | 3589.03370994 | 203.551974377 | 0.793554261189 | 0.0162820258605 |

## Noise

- Deterministic rollout: noise off.
- Stochastic rollout: nominal C&S released-code noise terms available and paired with seed `376023`; clean and adversarial conditions use the same sensory, motor, process, and signal-dependent standard draws.

## Verification

- Teacher package verification status: `checked_beta_invariant_geometry_only`.
- Recomputed live geometry arrays match the stored beta-1.4 teacher package at rtol/atol 1e-9; beta-specific H-infinity arrays were recomputed live.
