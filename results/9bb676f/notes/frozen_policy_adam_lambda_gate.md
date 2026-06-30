<!-- AUTO-GENERATED: frozen_policy_adam_lambda_gate -->
# Frozen Policy Adam Lambda Gate

This regenerated v2 artifact supersedes the cap-conditioned `7ea17b8` closeout. The old `recommended_lambda_floor` values are invalid for launch planning because the previous launch-facing floors promoted cap/trust-radius gradient pressure into lambda recommendations. This v2 artifact keeps those quantities diagnostic-only and uses cap-independent HVP bases.

- Source run: `ae9f30f/linear_no_bias_b1p05`
- Checkpoints: `[500, 12000]`
- Direct-epsilon launch anchor: `2.81033e+08` from `results/06a4dc8/canonical_soft_lambda_hvp.json` beta `1.05` (pooled)
- Direct source note: `06a4dc8` also carries per-substrate beta mappings; use those row-specific values in a substrate-indexed run spec.
- Launch/readiness cap use: `False`
- Ready for no-launch spec: `True`
- Implementation note: Launch-facing finite rows should use broad_epsilon_pgd_training with mechanism linear_no_bias or affine and inner_maximizer.method=adam. That route optimizes finite-policy parameters and evaluates them through the live graph-component rollout. The older policy_adversary_training finite path remains a legacy/static-clean-rollout lane and should not be used for the live finite-policy rows.

## Cap-Independent Lambda Candidates

| mechanism | candidate basis | lambda input | lambda* / source | candidate lambda | grad pressure diagnostic |
|---|---|---:|---:|---:|---:|
| direct_epsilon | fixed_hvp_p90 | 2.81e+08 | 2.549e+08 | 2.81e+08 | 3.926e+08 |
| linear_no_bias | hvp_generalized_eigen | 2.81e+08 | 3.596e+08 | 3.596e+08 | 3.926e+08 |
| affine | hvp_generalized_eigen | 2.81e+08 | 3.596e+08 | 3.596e+08 | 3.926e+08 |

Gradient pressure, trust-radius, and cap-boundary quantities are diagnostic-only; they are not lambda or readiness criteria.

## Frozen Rows

| checkpoint | mechanism | grad pressure diagnostic | curvature lambda* | curvature status | selected energy | objective gain | boundary frac | nonfinite |
|---|---|---:|---:|---|---:|---:|---:|---|
| 500 | direct_epsilon | 3.926e+08 | 2.587e+08 | directional_approximation | 1.519e-06 | 990.2 | 1 | nan=False, inf=False |
| 500 | linear_no_bias | 3.926e+08 | 3.596e+08 | finite | 1.519e-06 | 164.1 | 1 | nan=False, inf=False |
| 500 | affine | 3.926e+08 | 3.596e+08 | finite | 1.519e-06 | 206.7 | 1 | nan=False, inf=False |
| 12000 | direct_epsilon | 1.757e+08 | 2.063e+08 | directional_approximation | 1.519e-06 | 361.4 | 1 | nan=False, inf=False |
| 12000 | linear_no_bias | 1.757e+08 | 2.376e+08 | finite | 0 | 0 | 0 | nan=False, inf=False |
| 12000 | affine | 1.757e+08 | 2.376e+08 | finite | 0 | 0 | 0 | nan=False, inf=False |
<!-- /AUTO-GENERATED -->
