<!-- AUTO-GENERATED: frozen_policy_adam_lambda_gate -->
# Frozen Policy Adam Lambda Gate

This regenerated v2 artifact supersedes the cap-conditioned `7ea17b8` closeout. The old `recommended_lambda_floor` values are invalid for launch planning because the previous launch-facing floors promoted cap/trust-radius gradient pressure into lambda recommendations. This v2 artifact keeps those quantities diagnostic-only and uses cap-independent HVP bases.

- Source run: `ae9f30f/linear_no_bias_b1p05`
- Checkpoints: `[500, 12000]`
- Direct-epsilon launch anchor: `2.81033e+08` from `results/06a4dc8/canonical_soft_lambda_hvp.json` beta `1.05` (pooled)
- Direct source note: `06a4dc8` also carries per-substrate beta mappings; use those row-specific values in a substrate-indexed run spec.
- Launch/readiness cap use: `False`
- Ready for no-launch spec: `True`
- Caveat: Finite production rows now use Adam over finite parameters, but current training integration materializes static epsilon from a clean rollout pre-step. Do not describe the next spec as a true live-perturbed closed-loop finite-policy run until a Feedbax live rollout hook exists.

## Cap-Independent Lambda Candidates

| mechanism | candidate basis | lambda input | lambda* / source | candidate lambda | grad pressure diagnostic |
|---|---|---:|---:|---:|---:|
| direct_epsilon | fixed_hvp_p90 | 2.81e+08 | 2.549e+08 | 2.81e+08 | 4.13e+08 |
| linear_no_bias | hvp_generalized_eigen | 2.81e+08 | 3.597e+08 | 3.597e+08 | 4.13e+08 |
| affine | hvp_generalized_eigen | 2.81e+08 | 3.598e+08 | 3.598e+08 | 4.13e+08 |

Gradient pressure, trust-radius, and cap-boundary quantities are diagnostic-only; they are not lambda or readiness criteria.

## Frozen Rows

| checkpoint | mechanism | grad pressure diagnostic | curvature lambda* | curvature status | selected energy | objective gain | boundary frac | nonfinite |
|---|---|---:|---:|---|---:|---:|---:|---|
| 500 | direct_epsilon | 4.13e+08 | 2.715e+08 | directional_approximation | 1.519e-06 | 1041 | 1 | nan=False, inf=False |
| 500 | linear_no_bias | 4.13e+08 | 3.597e+08 | finite | 1.519e-06 | 270.5 | 1 | nan=False, inf=False |
| 500 | affine | 4.13e+08 | 3.598e+08 | finite | 1.519e-06 | 303.1 | 1 | nan=False, inf=False |
| 12000 | direct_epsilon | 1.793e+08 | 2.033e+08 | directional_approximation | 1.519e-06 | 382.1 | 1 | nan=False, inf=False |
| 12000 | linear_no_bias | 1.793e+08 | 2.368e+08 | finite | 0 | 0 | 0 | nan=False, inf=False |
| 12000 | affine | 1.793e+08 | 2.368e+08 | finite | 0 | 0 | 0 | nan=False, inf=False |
<!-- /AUTO-GENERATED -->
