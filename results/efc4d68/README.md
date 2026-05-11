# 2×2×2 Baseline Matrix: Variance Analysis Results

*Run date: 2026-05-08 / 2026-05-09 UTC*
*Tracking issue: `efc4d68`*

## Experiment Design

2×2×2 matrix: **architecture** (GRU vs vRNN) × **smoothness** (nn_hidden_derivative 0.001 vs off) × **jerk** (nn_output_jerk 1e5 vs off).
- 5 internal replicates per cell
- 12k warmup batches, batch size 250, no adversary
- Metric: pairwise RMSE on post-go (t ∈ [0, 1.0 s]) mean forward-velocity profile, direction-averaged across 8 validation targets
- Within-cell: C(5,2)=10 pairs; NN across-cell: 5×5=25 pairs

Run-specs: `results/part2_5/runs/baseline_jerk_vrnn_matrix__<label>/run.json` (8 files).

## 8-Cell Variance Table

| Cell label | Arch | Smooth | Jerk | Within RMSE | NN-Across RMSE | Ratio |
|---|---|---|---|---|---|---|
| GRU / none | GRU | off | off | 0.391 | 0.271 | 1.443 |
| GRU / smooth | GRU | on | off | 0.264 | 0.203 | 1.296 |
| **GRU / jerk** | **GRU** | **off** | **on** | **0.112** | **0.147** | **0.758** |
| GRU / smooth+jerk | GRU | on | on | 0.194 | 0.147 | 1.315 |
| vRNN / none | vRNN | off | off | 0.034 | 0.026 | 1.338 |
| vRNN / smooth | vRNN | on | off | 0.030 | 0.029 | 1.021 |
| vRNN / jerk | vRNN | off | on | 0.025 | 0.021 | 1.194 |
| vRNN / smooth+jerk | vRNN | on | on | 0.023 | 0.021 | 1.105 |

Convergence (final training / validation losses):

| Cell | Train loss | Val loss |
|---|---|---|
| GRU / none | 3.76 | 6.80 |
| GRU / smooth | 3.84 | 7.16 |
| GRU / jerk | 3.62 | 6.53 |
| GRU / smooth+jerk | 4.03 | 7.27 |
| vRNN / none | 10.1 | 14.8 |
| vRNN / smooth | 10.1 | 14.5 |
| vRNN / jerk | 10.3 | 15.1 |
| vRNN / smooth+jerk | 10.4 | 15.0 |

## Winner

**GRU / jerk** — only cell with ratio < 1.0 (0.758). Within-cell RMSE 0.112 m/s; 71% reduction vs GRU/none baseline (0.391).

## Key Findings

1. **Jerk is the primary driver for GRU.** Jerk alone: 71% within-RMSE reduction. Smoothness alone: 33% reduction.
2. **Combining smooth+jerk is worse than jerk alone.** The two terms interfere — GRU/smooth+jerk ratio = 1.315 vs GRU/jerk ratio = 0.758.
3. **vRNN cells under-converged at 12k batches.** Final train loss ~10+ vs GRU ~3.7. All vRNN cells produce nearly-identical velocity profiles (low absolute within-cell RMSE 0.02–0.03 but also low cross-cell RMSE). Tau mismatch (`tau=0.1`) is the leading hypothesis.
4. **vRNN ratio never drops below 1.0.** Best vRNN ratio is 1.021 (vRNN/smooth). Low absolute RMSE but ratio remains > 1 throughout.

## Replicate Clustering (User Observation)

In GRU/jerk and GRU/smooth+jerk: 3/5 replicates cluster tightly; 2/5 deviate (slower, less bell-shaped velocity profile). The 0.758 ratio reflects this mixture.

## Status

Not adopting GRU+jerk as default. 0.758 ratio is encouraging but replicate clustering (3 tight + 2 deviants) needs resolution first.

## Follow-Up Issues

- `32efb6f` — vRNN tau-sweep: test `tau ∈ {0.02, 0.05, 0.1, 0.2, 0.5}` to find time-constant matching task scale
- `3f111e9` — Analytical Riccati hyperparameter sweep: vary α and horizon n_steps to map Δv landscape
- `6ec6b19` — Empirical cost-schedule sweep (existing, ongoing)
- `8d6b88f` — Post-training-run protocol: CLAUDE.md addition to codify this workflow

## Artifacts

- Velocity plots (HTML, not committed): `_artifacts/efc4d68/` (forward + lateral velocity profiles)
- Model checkpoints: `_artifacts/efc4d68/<label>/`
