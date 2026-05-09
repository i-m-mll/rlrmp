# 6-Cell Anti-Anticipation + Loss-Shape Matrix — Production Run Plan

Pre-registration issue: 2bc95fd (expanded from 3-cell to 6-cell matrix)
Anti-anticipation strategy menu: 5acdaae
Smoothness retrain umbrella: efc4d68
Phase umbrella: f695729
Loss-history review (previous subagent): /tmp/flavor_ab_review/findings/loss_history.md

## Background: matrix expansion

The original 3-cell pre-reg tested anti-anticipation mechanisms (motor-pre-go mask and
full-trajectory hidden smoothness). The previous loss-history subagent identified two
cost-structure changes that plausibly drive inter-replicate variance in the current loss:

1. **Terminal-step velocity penalty was dropped.** Historical `simple_reach_loss`
   (feedbax commit e985e0e) fired `effector_final_velocity` only at `t=T`. The current
   `effector_vel_late` spreads the penalty across `[go+80, T]`, which is more permissive
   and admits diverse stopping strategies — different replicates can satisfy it via
   different velocity profiles.

2. **Position discount changed shape.** Historical loss used a `(t/T)^6` discount (98%
   of weight in the last 30% of the trial). Current `running 1.0 + late 0.5×cosine(1→2)`
   is roughly flat from the go cue, creating a broader basin with more local minima.

Cells 5 and 6 test restoring these terms via the new CLI flags added in this branch.

## Task config (unchanged from 3-cell pre-reg)

`epoch_len_ranges = [[0, 1], [10, 30]]` — pure-hold 0 steps; target-on 100-300 ms.
`n_steps = 140`. `dt = 0.01 s`. Matches Shahbazi 2025 §4.2. Bug: 2bc95fd.

## Cell matrix

All cells share: `--hidden-type gru --n-warmup-batches 12000 --n-adversary-batches 0
--batch-size 250 --n-replicates 5 --nn-output-jerk 1e5 --seed 42 --checkpoint
--fused --no-streaming-loss`.

| # | Run label | Anti-anticipation lever | Loss-shape variant |
|---|---|---|---|
| 1 | `gru__jerk` | none (control) | current default |
| 2 | `gru__jerk_motor_pre` | `--nn-output-pre-go 1e-2` | current default |
| 3 | `gru__jerk_smooth_high` | `--nn-hidden-derivative 1e2` | current default |
| 4 | `gru__jerk_motor_smooth_combo` | `--nn-output-pre-go 1e-2 --nn-hidden-derivative 1e2` | current default |
| 5 | `gru__jerk_loss_v_terminal` | none | Variant A: terminal-step velocity replaces window velocity |
| 6 | `gru__jerk_loss_historical` | none | Variant B: historical shape (full cosine ramp + terminal velocity) |

**Variant A flags:** `--effector-final-vel 1.0 --effector-vel-late 0.0`
**Variant B flags:** `--effector-final-vel 1.0 --effector-vel-late 0.0
  --effector-pos-running 0.0 --effector-pos-late-weight 1.0
  --effector-pos-late-final-scale 6.0 --effector-pos-late-start-step 0`

## Production CLI invocations

Run from `/workspace/rlrmp` on the pod. Create log dir first: `mkdir -p /workspace/logs`.

### Cell 1 — gru__jerk (control)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --nn-output-jerk 1e5 \
  --seed 42 \
  --checkpoint \
  --fused \
  --no-streaming-loss \
  --output-dir _artifacts/part2_5/runs/gru__jerk \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk.log 2>&1 &
```

### Cell 2 — gru__jerk_motor_pre (strategy 1: pre-go motor mask)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --nn-output-jerk 1e5 \
  --nn-output-pre-go 1e-2 \
  --seed 42 \
  --checkpoint \
  --fused \
  --no-streaming-loss \
  --output-dir _artifacts/part2_5/runs/gru__jerk_motor_pre \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_motor_pre.log 2>&1 &
```

### Cell 3 — gru__jerk_smooth_high (strategy 3: full-trajectory smoothness 1e2)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --nn-output-jerk 1e5 \
  --nn-hidden-derivative 1e2 \
  --seed 42 \
  --checkpoint \
  --fused \
  --no-streaming-loss \
  --output-dir _artifacts/part2_5/runs/gru__jerk_smooth_high \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_smooth_high.log 2>&1 &
```

### Cell 4 — gru__jerk_motor_smooth_combo (strategies 1+3 combined)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --nn-output-jerk 1e5 \
  --nn-output-pre-go 1e-2 \
  --nn-hidden-derivative 1e2 \
  --seed 42 \
  --checkpoint \
  --fused \
  --no-streaming-loss \
  --output-dir _artifacts/part2_5/runs/gru__jerk_motor_smooth_combo \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_motor_smooth_combo.log 2>&1 &
```

### Cell 5 — gru__jerk_loss_v_terminal (Variant A: terminal velocity only)

Restores the historical terminal-step velocity penalty at the final timestep; drops the
window velocity penalty. Position terms unchanged from default.

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --nn-output-jerk 1e5 \
  --effector-final-vel 1.0 \
  --effector-vel-late 0.0 \
  --seed 42 \
  --checkpoint \
  --fused \
  --no-streaming-loss \
  --output-dir _artifacts/part2_5/runs/gru__jerk_loss_v_terminal \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_loss_v_terminal.log 2>&1 &
```

### Cell 6 — gru__jerk_loss_historical (Variant B: full historical shape)

Approximates the historical `(t/T)^6`-discounted position + terminal velocity shape.
Drops the flat running position term; cosine ramp starts at go cue (step 0) with a steep
final scale of 6.0; terminal velocity replaces window velocity.

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --nn-output-jerk 1e5 \
  --effector-final-vel 1.0 \
  --effector-vel-late 0.0 \
  --effector-pos-running 0.0 \
  --effector-pos-late-weight 1.0 \
  --effector-pos-late-final-scale 6.0 \
  --effector-pos-late-start-step 0 \
  --seed 42 \
  --checkpoint \
  --fused \
  --no-streaming-loss \
  --output-dir _artifacts/part2_5/runs/gru__jerk_loss_historical \
  --checkpoint-every 1000 \
  > /workspace/logs/gru_jerk_loss_historical.log 2>&1 &
```

## Wall-clock and cost estimates

- 12000 warmup batches × 250 trials × 5 replicates on RTX 5090: ~30-40 min per cell.
- 6 cells run sequentially: ~3-4 hours total; in parallel (6 pods): ~35-45 min.
- Cost at ~$0.39/hr (5090 SecureCloud EUR-IS-2): ~$0.40-0.60 per cell; ~$2.50-3.50 total
  sequential. Single-pod sequential is cheapest; 6-pod parallel reduces wall time at ~6x cost.
- Recommended: run all 6 sequentially on one pod to minimize cost.

## Monitoring

Use the runbook cadence from CLAUDE.md §7: check 1 min after start (JIT visible?),
every 5 min through early loss decline, every 30 min once steadily descending.
Watch for `ptxas` warnings, OOM, and Traceback patterns.

## Post-run analysis

After all 6 cells complete, measure and report on 2bc95fd:

(A) Within-cell pairwise RMSE on post-go forward velocity (10 pairs per cell).
(B) Pre-go drift: mm of forward motion in [-200 ms, 0 ms] window.
(C) Variance ratio: within-cell RMSE / min(across-cell RMSE).

Pass criteria — see pre-registration 2bc95fd for base decision logic.
Additional criterion for Variants A and B: if either beats the best of cells (1)-(4) on
within-cell RMSE, recommend adopting the historical loss shape as the new training default
and close out the basin-selection component of the smoothness umbrella (efc4d68).
