<!-- AUTO-GENERATED: frozen_policy_adam_lambda_gate -->
# Frozen Policy Adam Lambda Gate

- Source run: `ae9f30f/linear_no_bias_b1p05`
- Checkpoints: `[500, 12000]`
- Lambda input: `2.81033e+08` from `results/06a4dc8/canonical_soft_lambda_hvp.json` beta `1.05`
- Ready for no-launch spec: `True`
- Caveat: Finite production rows now use Adam over finite parameters, but current training integration materializes static epsilon from a clean rollout pre-step. Do not describe the next spec as a true live-perturbed closed-loop finite-policy run until a Feedbax live rollout hook exists.

## Lambda Floors

| mechanism | lambda input | max grad pressure | recommended floor |
|---|---:|---:|---:|
| direct_epsilon | 2.81e+08 | 4.13e+08 | 4.13e+08 |
| linear_no_bias | 2.81e+08 | 4.13e+08 | 4.13e+08 |
| affine | 2.81e+08 | 4.13e+08 | 4.13e+08 |

## Frozen Rows

| checkpoint | mechanism | grad pressure | directional lambda* | selected energy | objective gain | boundary frac | nonfinite |
|---|---:|---:|---:|---:|---:|---:|---|
| 500 | direct_epsilon | 4.13e+08 | 4.468e+13 | 1.519e-06 | 1041 | 1 | nan=False, inf=False |
| 500 | linear_no_bias | 4.13e+08 | 2.369e+15 | 1.519e-06 | 270.5 | 1 | nan=False, inf=False |
| 500 | affine | 4.13e+08 | 2.277e+15 | 1.519e-06 | 303.1 | 1 | nan=False, inf=False |
| 12000 | direct_epsilon | 1.793e+08 | 3.346e+13 | 1.519e-06 | 382.1 | 1 | nan=False, inf=False |
| 12000 | linear_no_bias | 1.793e+08 | 4.832e+15 | 0 | 0 | 0 | nan=False, inf=False |
| 12000 | affine | 1.793e+08 | 4.865e+15 | 0 | 0 | 0 | nan=False, inf=False |
<!-- /AUTO-GENERATED -->
