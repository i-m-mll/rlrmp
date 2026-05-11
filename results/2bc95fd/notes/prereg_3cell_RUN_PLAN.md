# 3-Cell Anti-Anticipation Matrix — Production Run Plan

Pre-registration issue: 2bc95fd
Anti-anticipation strategy menu: 5acdaae
Smoothness retrain umbrella: efc4d68
Phase umbrella: f695729

## Task config change

`epoch_len_ranges = [[0, 1], [10, 30]]` (pure-hold 0 steps; target-on 100-300 ms at dt=0.01 s).
Matches Shahbazi 2025 §4.2. `n_steps` bumped to 140 (0 + 30 + 80 + 30 buffer).

## Cell matrix

| # | Run label            | Added flag(s)                    | Anti-anticipation lever             |
|---|----------------------|----------------------------------|-------------------------------------|
| 1 | `gru__jerk`          | `--nn-output-jerk 1e5`           | none (control, new task only)       |
| 2 | `gru__jerk_motor_pre`| `--nn-output-jerk 1e5 --nn-output-pre-go 1e-2` | strategy 1: pre-go motor-output mask |
| 3 | `gru__jerk_smooth_high` | `--nn-output-jerk 1e5 --nn-hidden-derivative 1e2` | strategy 3: full-trajectory hidden smoothness (Shahbazi-aligned) |

## Shared parameters

- Architecture: GRU (`--hidden-type gru`)
- Warmup: 12000 batches (`--n-warmup-batches 12000`)
- Batch size: 250 (`--batch-size 250`)
- Adversary batches: 0 (warmup-only; use `--n-adversary-batches 0` or omit adversarial phase)
- Replicates: 5 (`--n-replicates 5`)
- No adversary: baseline training only (standard gusts)
- Seed: 42 (default)
- Checkpoint every 1000 batches

## Production CLI invocations

Run from `/workspace/rlrmp` on the pod (rsync worktree to pod, not main):

### Cell 1 — gru__jerk (control)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --batch-size 250 \
  --n-replicates 5 \
  --n-adversary-batches 0 \
  --nn-output-jerk 1e5 \
  --output-dir _artifacts/part2_5/runs/gru__jerk \
  --checkpoint \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk.log 2>&1 &
```

### Cell 2 — gru__jerk_motor_pre (strategy 1: pre-go motor mask)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --batch-size 250 \
  --n-replicates 5 \
  --n-adversary-batches 0 \
  --nn-output-jerk 1e5 \
  --nn-output-pre-go 1e-2 \
  --output-dir _artifacts/part2_5/runs/gru__jerk_motor_pre \
  --checkpoint \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_motor_pre.log 2>&1 &
```

### Cell 3 — gru__jerk_smooth_high (strategy 3: full-trajectory smoothness 1e2)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --batch-size 250 \
  --n-replicates 5 \
  --n-adversary-batches 0 \
  --nn-output-jerk 1e5 \
  --nn-hidden-derivative 1e2 \
  --output-dir _artifacts/part2_5/runs/gru__jerk_smooth_high \
  --checkpoint \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_smooth_high.log 2>&1 &
```

## Monitoring

Use the runbook cadence from CLAUDE.md §7: check 1 min after start (JIT visible?),
every 5 min through early loss decline, every 30 min once steadily descending.
Watch for `ptxas` warnings, OOM, and Traceback patterns.

## Post-run analysis

After all 3 cells complete, measure and report on 2bc95fd:

(A) Within-cell pairwise RMSE on post-go forward velocity (10 pairs per cell).
(B) Pre-go drift: mm of forward motion in [-200 ms, 0 ms] window.
(C) Variance ratio: within-cell RMSE / min(across-cell RMSE).

Pass criteria — see pre-registration 2bc95fd for decision logic.
