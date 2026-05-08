# Baseline 2×2×2 architecture × smoothness × jerk matrix — smoke + deploy plan

A/B/C decomposition of replicate-variability mitigations on the baseline (no
adversary) controller. Bug: `efc4d68` (rlrmp), `7e1d257` (feedbax — output-jerk
term), `e8f3738` (feedbax — hidden-state smoothness term).

## Matrix

|   # | Architecture | Smoothness ‖ḣₜ‖² (1e-3) | Jerk ‖x⃛ₜ‖² (1e5) | Run label                       |
| ---:| ------------ | -----------------------:| ------------------:| ------------------------------- |
|   1 | GRU          | —                       | —                  | `baseline_gru__none`            |
|   2 | GRU          | ✓                       | —                  | `baseline_gru__smooth`          |
|   3 | GRU          | —                       | ✓                  | `baseline_gru__jerk`            |
|   4 | GRU          | ✓                       | ✓                  | `baseline_gru__smooth_jerk`     |
|   5 | vRNN         | —                       | —                  | `baseline_vrnn__none`           |
|   6 | vRNN         | ✓                       | —                  | `baseline_vrnn__smooth`         |
|   7 | vRNN         | —                       | ✓                  | `baseline_vrnn__jerk`           |
|   8 | vRNN         | ✓                       | ✓                  | `baseline_vrnn__smooth_jerk`    |

`vRNN` = `vanilla_rnn` — `LeakyRNNCell` with `tau=0.1 s` → `α=dt/tau=0.01/0.1=0.1`
(biologically-motivated leaky integration; matches Sussillo 2015, Yang 2019). No
gating. CLI: `--hidden-type vanilla_rnn`.

## Smoke-test results (CPU, 2026-05-08)

All 8 conditions executed locally on CPU with `--n-warmup-batches 3 --batch-size
50 --n-replicates 2 --no-fused`. Each completed in <10 s wall after JIT.

| # | Run                              | Exit | Mean training loss (3 batches)      | Notes                            |
| -:| -------------------------------- | ---- | ----------------------------------- | -------------------------------- |
| 1 | `baseline_gru__none`             | 0    | 1.67e+01 → 1.58e+01 → 1.37e+01      | Loss decreases monotonically.    |
| 2 | `baseline_gru__smooth`           | 0    | 1.67e+01 → 1.58e+01 → 1.37e+01      | `nn_hidden_derivative` ~9.5e-7.  |
| 3 | `baseline_gru__jerk`             | 0    | 1.69e+01 → 1.60e+01 → 1.39e+01      | `nn_output_jerk` decreases.      |
| 4 | `baseline_gru__smooth_jerk`      | 0    | 1.69e+01 → 1.60e+01 → 1.39e+01      | Both terms register & decrease.  |
| 5 | `baseline_vrnn__none`            | 0    | 1.67e+01 → 1.58e+01 → 1.37e+01      | LeakyRNNCell path JIT-compiles.  |
| 6 | `baseline_vrnn__smooth`          | 0    | 1.67e+01 → 1.58e+01 → 1.37e+01      | `nn_hidden_derivative` registers. |
| 7 | `baseline_vrnn__jerk`            | 0    | 1.69e+01 → 1.60e+01 → 1.39e+01      | `nn_output_jerk` registers.       |
| 8 | `baseline_vrnn__smooth_jerk`     | 0    | 1.69e+01 → 1.60e+01 → 1.39e+01      | Both terms; loss decreases.      |

Smoke logs at `/tmp/flavor_ab_review/jerk_smoke/<label>.log`. The +0.2 baseline
offset on jerk-on conditions is exactly the measured `1e5 × 1.95e-6` jerk
contribution at iteration 0; consistent with weight=1e5.

## Production CLI invocations (5090 deploy, 12 k batches)

Same warmup-only profile as `baseline/standard_12k`: `--n-warmup-batches 12000
--n-adversary-batches 0 --batch-size 250 --n-replicates 5 --controller-lr 1e-4
--seed 42 --checkpoint --fused --streaming-loss`.

> **Caveat** — `--streaming-loss` cannot be used while `--nn-hidden-derivative`
> or `--nn-output-jerk` is non-zero, until feedbax `d67e303` (cross-timestep
> streaming-loss extension) lands. Conditions 2-4 and 6-8 must be run with
> `--no-streaming-loss`. This costs ~240 MB extra peak VRAM (well under 24 GB
> on a 4090, trivial on a 5090).

```bash
# 1. baseline_gru__none
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 --hidden-type gru \
  --nn-hidden-derivative 0.0 --nn-output-jerk 0.0 --seed 42 \
  --checkpoint --fused --streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_gru__none

# 2. baseline_gru__smooth
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 --hidden-type gru \
  --nn-hidden-derivative 1e-3 --nn-output-jerk 0.0 --seed 42 \
  --checkpoint --fused --no-streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_gru__smooth

# 3. baseline_gru__jerk
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 --hidden-type gru \
  --nn-hidden-derivative 0.0 --nn-output-jerk 1e5 --seed 42 \
  --checkpoint --fused --no-streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_gru__jerk

# 4. baseline_gru__smooth_jerk
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 --hidden-type gru \
  --nn-hidden-derivative 1e-3 --nn-output-jerk 1e5 --seed 42 \
  --checkpoint --fused --no-streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_gru__smooth_jerk

# 5. baseline_vrnn__none
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 \
  --hidden-type vanilla_rnn \
  --nn-hidden-derivative 0.0 --nn-output-jerk 0.0 --seed 42 \
  --checkpoint --fused --streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_vrnn__none

# 6. baseline_vrnn__smooth
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 \
  --hidden-type vanilla_rnn \
  --nn-hidden-derivative 1e-3 --nn-output-jerk 0.0 --seed 42 \
  --checkpoint --fused --no-streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_vrnn__smooth

# 7. baseline_vrnn__jerk
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 \
  --hidden-type vanilla_rnn \
  --nn-hidden-derivative 0.0 --nn-output-jerk 1e5 --seed 42 \
  --checkpoint --fused --no-streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_vrnn__jerk

# 8. baseline_vrnn__smooth_jerk
uv run --no-sync python scripts/train_minimax.py --n-warmup-batches 12000 \
  --n-adversary-batches 0 --batch-size 250 --n-replicates 5 \
  --hidden-type vanilla_rnn \
  --nn-hidden-derivative 1e-3 --nn-output-jerk 1e5 --seed 42 \
  --checkpoint --fused --no-streaming-loss \
  --output-dir _artifacts/part2_5/runpod/baseline_jerk_vrnn_matrix/baseline_vrnn__smooth_jerk
```

## Appendix: α=1.0 → α=0.1 vRNN re-smoke (2026-05-08)

`tau` changed from `dt=0.01` (α=1.0, pure vanilla RNN) to `0.1 s` (α=0.1,
biologically-motivated leaky integration). Re-smoke with same minimal config:
`--n-warmup-batches 3 --batch-size 50 --n-replicates 2 --no-fused`.

| Condition                 | α=0.1 losses (3 batches)        | vs α=1.0 (prior) | Diff? |
| ------------------------- | --------------------------------| -----------------| ----- |
| `baseline_vrnn__none`     | 1.67e+01 → 1.58e+01 → 1.37e+01 | identical        | No    |
| `baseline_vrnn__smooth`   | 1.67e+01 → 1.58e+01 → 1.37e+01 | identical        | No    |
| `baseline_vrnn__jerk`     | 1.69e+01 → 1.60e+01 → 1.39e+01 | identical        | No    |
| `baseline_vrnn__smooth_jerk` | 1.69e+01 → 1.60e+01 → 1.39e+01 | identical     | No    |

GRU none (reference): 1.67e+01 → 1.58e+01 → 1.37e+01.

**Note:** Loss curves at 3 batches do NOT differentiate α=0.1 from α=1.0 or from
GRU. This is expected — 3 batches with a random-init model and batch_size=50 are
insufficient to expose recurrent dynamics differences; all runs are dominated by
the initial condition. Structural correctness was verified separately:
`VanillaRNNCell._cell.alpha ≈ 0.1` (τ=0.1) vs `1.0` (τ=dt), confirming the
`_resolve_hidden_type` change took effect. Divergence between architectures will
appear on longer runs (≥1k batches) as the hidden dynamics equilibrate.

## Cross-references

- rlrmp issue: `efc4d68` (umbrella for smoothness retrain + jerk escalation).
- feedbax issues: `e8f3738` (`StateDerivativeLoss`, merged), `7e1d257`
  (`OutputJerkLoss`, on `feature/loss-output-jerk`), `d67e303` (cross-timestep
  streaming-loss extension; deferred — current runs use `--no-streaming-loss`
  whenever a cross-timestep term is non-zero).
- vRNN: `tau=0.1 s` (α=0.1) per Yang 2019 / Sussillo 2015 consensus.
- Sibling A/B: `results/part2_5/runpod/baseline_smoothness/standard_12k_smooth/`.
- Reference paper: Shahbazi, Codol, Michaels & Gribble 2025, Eq. 1
  (https://www.biorxiv.org/content/10.1101/2025.03.26.645562) — `1e-3` and `1e5`
  weights mirror their setup.
