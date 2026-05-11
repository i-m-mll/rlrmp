# Part 2.5 — Flavor-(b) LinearDynamicsAdversary sweep

## Scope

First end-to-end training sweep of the flavor-(b) `LinearDynamicsAdversary` /
`DynamicsMatrixPerturb` pair (issue `c723082`). Phase 1 standard warm-start
(2 000 batches), Phase 2 minimax adversarial training (5 000 batches) with
PGD-projected ΔA inside the plant `LTISystem.vector_field`.

- **Conditions**: `eta_max ∈ {0.03, 0.10, 0.30}` × `seed ∈ {0, 1, 2}` = 9 runs
  × 5 internal `vmap` replicates = 45 trained models.
- **Hardware**: RunPod RTX 5090 (`82.221.170.242:31593`).
- **Timing**: ~2 h per run (warmup ~9 min, adversarial ~2 h fused).

## Layout

- `runs/flavor_b_eta{X}__seed_{Y}/run.json` — per-run spec (hyperparameters,
  artifact pointers, loss summary, timing).
- Bulk artifacts (model `.eqx`, checkpoints, `.npz` loss arrays, training logs,
  and a derived `flavor_b_summary.json` with per-run loss trajectories at
  every 500 batches and per-eta aggregates) are mirrored under
  `_artifacts/part2_5/runpod/flavor_b/`.

## High-level findings

See the comment thread on `c723082` for the substantive analysis. Headline:

- All 9 runs converge cleanly. No NaN/Inf. No >20% loss spikes between
  500-batch checkpoints across the entire sweep.
- Warmup-end ctrl loss ~14.2 (plateau); adversarial-start (batch 0) ~9.4 once
  the ΔA adversary is wired in; adversarial-end (batch 5000) ~4.7-5.1.
- Final loss is **non-monotonic in `eta_max`**: 0.03 → 4.75 ± 0.57,
  0.10 → 5.12 ± 0.34, 0.30 → 4.70 ± 0.18. The mid-budget condition is
  marginally worse on the controller objective, and across-seed variance
  collapses with the largest budget.

## Cross-references

- Substantive issue: `c723082` (LinearDynamicsAdversary)
- Analyses coord: `4d38c15` (cross-ref of pending induced-gain analysis on
  these models)
- Phase: methodology-fix umbrella `b557d4e`
- Pre-registered downstream metric: `γ_sd × qr_cost` (induced-gain analyser)
