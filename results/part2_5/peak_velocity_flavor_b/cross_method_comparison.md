# Empirical peak-velocity Δv on trained flavor-B controllers

Cross-method comparison of empirical peak-forward-velocity inflation Δv
measured on the **trained** `LinearDynamicsAdversary` (flavor-B) controllers
relative to the no-perturbation `baseline_standard_12k` GRU controller.
Companion to the analytical Δv predictions tabulated in
`results/flavor_a_vs_b/synthesis.md` §5.3 / §7.2.

## Δv definition

`Δv % = (pv_target − pv_baseline) / pv_baseline × 100`, where
`pv = peak forward velocity` is the signed projection of the velocity
trajectory onto the unit reach axis `unit(target − init)`. Matches
`rlrmp.analysis.hinf_riccati.compute_velocity_inflation` (issue `f90bf74`).
Evaluation: single canonical reach (15 cm forward, init=(0,0), target=(0.15,0)),
SISU=0.5, `pert_scale=0.0` (no test-time perturbation), eval_seed=42, 5 internal
replicates per training run.

Baseline (`baseline_standard_12k`, GRU, trained without perturbations,
n=5 replicates):

- mean peak forward velocity = **1.643 m/s** (SD 0.689)

## Per-η_max aggregates (n=15 = 3 seeds × 5 replicates)

| η_max | mean peak forward velocity (m/s) | mean Δv % | SD Δv % |
|---|---|---|---|
| 0.03 | 1.302 | **−20.75** | 40.80 |
| 0.10 | 1.209 | **−26.40** | 36.55 |
| 0.30 | 1.221 | **−25.68** | 40.88 |

**Headline.** Trained flavor-B controllers are **slower** than the
no-perturbation baseline at every η_max. Mean Δv ≈ −20% to −27%, with
small differences (~6%) between η_max conditions that are dwarfed by the
per-condition SD (~40%). The η_max stratification is **weak** at this
operating point.

## Per-replicate sign distribution (bimodality)

The replicate-level Δv distribution is **bimodal**: within each η_max
group, most replicates land between −45% and −60% (markedly slower than
baseline) while a minority land between +40% and +55% (markedly faster).
Concretely, fraction of replicates with positive Δv:

| η_max | positive-Δv fraction (of 15) | range of positive Δv | range of negative Δv |
|---|---|---|---|
| 0.03 | 4/15 (≈27%) | +4.4 to +56.8 % | −11.6 to −72.8 % |
| 0.10 | 3/15 (≈20%) | +27.5 to +50.1 % | −26.4 to −62.2 % |
| 0.30 | 3/15 (≈20%) | +48.7 to +51.6 % | −17.5 to −57.5 % |

The negative *mean* therefore reflects a majority-negative replicate
distribution, not a small uniform shift. Aggregating means hides the
structural bimodality; downstream analysis should consider per-replicate
behaviour or partition by the trained controller's converged solution.

## Comparison to analytical predictions

Analytical Δv on the rlrmp regime (synthesis §5.3 / §4.2-revised, with the
corrected full-state $B_w$): **+10.8% at $1.5\gamma_*^{(a)}$**, growing to
~+27% near $\gamma_*$. With the corrected `cs_faithful_pointmass()`
(synthesis §4.2-revised) the analytical sign is **positive** on both rlrmp
and C&S regimes.

Empirical sign here is **negative** (mean), and bimodal at the replicate
level — opposite-sign to the analytical prediction at the group mean,
and in only a minority of replicates does the empirical Δv even land on
the right sign. The gap is the central new question of this bundle (cross-
ref synthesis §8.0 / §8.0.1 revised).

## Cross-references

- Tracking issue: `c723082` (LinearDynamicsAdversary).
- Analytical Δv (corrected): `97c227a` and synthesis doc §4.2 / §5.4 / §7.2
  (May 2026 revision via `feature/cs-faithful-riccati-investigation`).
- Phase context: synthesis doc §7.1.1 (this empirical Δv result),
  §8.0 (joint reading).
- Run script: `scripts/run_peak_velocity_flavor_b.py`.
- Run spec: `results/part2_5/runs/peak_velocity_flavor_b/run.json`.
- Bulk results: `_artifacts/part2_5/runs/peak_velocity_flavor_b/summary.json`.
