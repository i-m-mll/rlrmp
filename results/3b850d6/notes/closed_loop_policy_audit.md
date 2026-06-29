<!-- AUTO-GENERATED: closed_loop_policy_audit -->
# Closed-loop policy audit

Issue: `3b850d6`. Source frozen no-PGD runs: `c92ebd8`.

No controller weights were updated and no training was launched. The repaired audit scores raw finite-policy output with `J(raw_epsilon) - lambda * E(raw_epsilon)`. The per-trial L2 cap is then applied only to produce selected/clipped diagnostics.

## Lambda choices

Lambda values use the `093d949` per-trial p90 sweep center and compact multipliers around the observed cap-to-interior transition.

| row | multiplier | lambda | direct cap-bound | direct gain |
|---|---:|---:|---:|---:|
| `open_loop_small` | 2 | 2.22793e+08 | 100.000% | 2607.36 |
| `open_loop_small` | 4 | 4.45586e+08 | 0.000% | 231.314 |
| `open_loop_moderate` | 2 | 2.05863e+08 | 100.000% | 2240.4 |
| `open_loop_moderate` | 4 | 4.11726e+08 | 0.000% | 177.819 |
| `open_loop_stress` | 2 | 2.71611e+08 | 0.000% | 732.953 |
| `open_loop_stress` | 4 | 5.43222e+08 | 0.000% | 22.2769 |

## Best known directions

The table shows the best scalar-amplitude row for each known direction. Full per-amplitude raw norms, selected/clipped norms, raw-to-selected ratios, cap-violation fractions, raw energy penalties, selected/clipped energy penalties, objective gains, and finite/nonfinite statuses are in the tracked JSON and CSV.

| row | direction | lambda mult | amplitude | raw gain | selected gain | raw energy penalty | clipped energy penalty | raw/cap max | cap violations | finite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `open_loop_small` | `affine_mean_direct` | 2 | 4 | 1504.43 | 1300.11 | 5546.54 | 4603.25 | 1.09769 | 100.000% | finite |
| `open_loop_small` | `linear_ridge_direct` | 2 | 0.5 | 1018.19 | 1018.19 | 1606.55 | 1606.55 | 0.685043 | 0.000% | finite |
| `open_loop_moderate` | `affine_mean_direct` | 2 | 4 | 1573.57 | 1180.36 | 6408.82 | 4253.46 | 1.22749 | 100.000% | finite |
| `open_loop_moderate` | `linear_ridge_direct` | 2 | 0.5 | 768.987 | 768.987 | 1684.9 | 1684.9 | 0.700088 | 0.000% | finite |
| `open_loop_stress` | `affine_mean_direct` | 2 | 1 | 212.48 | 212.48 | 908.985 | 908.985 | 0.40246 | 0.000% | finite |
| `open_loop_stress` | `linear_ridge_direct` | 2 | 0.5 | 349.066 | 349.066 | 1013.36 | 1013.36 | 0.516627 | 0.000% | finite |

## Interpretation

Known affine mean-direct directions improve the raw objective on frozen c92 rows, so the old zero closed-loop policy result should not be read as evidence that finite closed-loop policies lack useful directions. It is best interpreted as an optimizer/projection artifact; linear no-bias expressivity remains basis/scaling dependent.

The affine mean-direct direction is an expressivity check, not a scientific success criterion. A positive known-direction line-search result means the previous zero closed-loop policy result cannot by itself support a no-useful-policy claim. The linear no-bias ridge direction is still sensitive to the live feature basis and feature scaling; weak fits should be read as unresolved basis/scaling evidence, not as proof that no-bias policies are dead.

## Runtime caveats

- The line search uses frozen deterministic c92 train batches and existing final checkpoints.
- The linear ridge fit is per-time from zero-policy live features to direct epsilon; it does not optimize policy weights.
- Heavy arrays were not persisted; tracked JSON/CSV carry scalar summaries only.
<!-- /AUTO-GENERATED -->
