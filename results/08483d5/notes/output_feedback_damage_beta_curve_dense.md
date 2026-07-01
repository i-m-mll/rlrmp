# 6D output-feedback H-infinity damage beta curves

## Method

- Source convention: same recursion/rollout convention as the beta 1.05 and beta 1.4 scripts on this issue branch.
- Controller: 6D no-integrator output-feedback H-infinity analytical teacher.
- State: 36D delay-augmented state, six 6D physical blocks `[x, y, vx, vy, force_x, force_y]`; no disturbance integrators.
- Damage: task cost increase only, `Q/R/Qf` cost under the H-infinity controller's fixed optimal adversary policy `F_t`, minus the same controller/estimator with `F_t = 0`; no `gamma^2 ||epsilon||^2` subtraction in the reported damage.
- Deterministic convention: no forward-simulation noise.
- Noisy convention: paired single-seed nominal C&S noise with seed `376023`; clean and adversarial conditions share sensory, motor, process, and signal-dependent standard draws.
- Beta-specific gamma, Riccati solution, estimator covariance, controller gains, and adversary policy were recomputed live for every beta.

## Benchmark

- Beta grid: 128 points. It includes beta `1.001`, every `0.01` from `1.01` through `2.00`, and every `0.001` from `1.330` through `1.360`.
- Benchmark betas: [1.05, 1.4, 2.0].
- Max rollout-vs-recursion damage absolute difference: 2.19006324187e-09.
- Mean rollout time per beta: 0.404314 s.
- Mean recursion time per beta: 0.0721077 s.
- Method selected for full grid: `affine_value_recursion`.

## Sanity checks

- beta=1.05 deterministic: 447.890402368 (prior 447.890402367).
- beta=1.05 noisy: 1911.89309715 (prior 1911.89309715).
- beta=1.40 deterministic: 3704.96326882 (prior about 3704.963).
- beta=1.40 noisy: 6131.6906765 (prior about 6131.691).

## Headline values

| beta | deterministic damage | noisy damage |
|---:|---:|---:|
| 1.001 | 651.008898962 | 16574.8653505 |
| 1.010 | 587.92136027 | 140291.089592 |
| 1.050 | 447.890402368 | 1911.89309715 |
| 1.400 | 3704.96326882 | 6131.6906765 |
| 2.000 | 247.303150034 | 357.829669168 |

## Dense Spike Region

- The maximum deterministic damage on this grid occurs at beta `1.342`:
  `166706128570.546`.
- The point lies inside a very narrow spike. Neighboring deterministic values are
  `7924912.142` at beta `1.341` and `8268701.845` at beta `1.343`.
- Treat this as a stress-test/conditioning candidate until the separate
  component and conditioning diagnostic determines whether it is a real
  closed-loop transient amplification or a numerical/implementation pathology.

## Uncertainties

- The noisy curve is one paired nominal-noise draw, not a Monte Carlo expectation over many seeds. It is faithful to the prior beta scripts' convention, but does not quantify sampling variability.
- I did not use historical cap/radius/trust-region constants; no training or PGD safety cap enters this analytical recursion.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --no-sync python results/08483d5/scripts/compute_output_feedback_damage_beta_curve_dense.py
```
