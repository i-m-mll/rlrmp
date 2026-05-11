# Lit-Replication 6-Cell Matrix — Production Run Plan

Tracking issue: f47abb1 (Lit-replication 6-cell matrix)
Power-law schedule implementation: 2e1a6ad
Training-methods coord: c99ad9d
Phase umbrella: b33e8da (methodology-fix, b557d4e)

## Background

This 6-cell matrix tests whether a faithful replication of the Chaisanguanthum & Shenoy 2019
(C&S) loss schedule produces models with systematically better velocity-RMSE ratios and lower
inter-replicate variance than the current production loss.

Two design dimensions are crossed:
1. **Jerk regulariser** (`nn_output_jerk`): on (1e5, cells 1–3) vs off (0, cells 4–6).
   Shahbazi et al. 2025 Eq. 1 used jerk; C&S 2019 did not.  This axis tests whether the
   jerk term interacts with the position schedule.
2. **Position schedule**: flat (cells 1, 4), post-go `(t/T)^6` (cells 2, 5), full-trial
   `(t/T)^6` on both hold and running (cells 3, 6).  The power-law schedule concentrates
   ~98 % of position-error weight in the last 30 % of the trial, matching C&S Eq. 15.

Prerequisites: commit `2e1a6ad` (power-law schedule support) must be merged before the pod
is set up.  The relevant CLI flags are `--effector-pos-running-schedule {flat,powerlaw}`,
`--effector-hold-pos-schedule {flat,powerlaw}`, and `--position-powerlaw-power 6.0`.

## Task config (shared with prior 6-cell matrix)

`epoch_len_ranges = [[0, 1], [10, 30]]` — pure-hold 0 steps; target-on 100–300 ms.
`n_steps = 140`. `dt = 0.01 s`. Matches Shahbazi 2025 §4.2. Bug: 2bc95fd.

## Cell matrix

| # | Cell name | `nn_output_jerk` | Position schedule (hold) | Position schedule (running) |
|---|---|---|---|---|
| 1 | `lit__flat_jerk` | 1e5 | flat | flat |
| 2 | `lit__post_jerk` | 1e5 | flat | `(t/T)^6` |
| 3 | `lit__full_jerk` | 1e5 | `(t/T)^6` | `(t/T)^6` |
| 4 | `lit__flat_nojerk` | 0 | flat | flat |
| 5 | `lit__post_nojerk` | 0 | flat | `(t/T)^6` |
| 6 | `lit__full_nojerk` | 0 | `(t/T)^6` | `(t/T)^6` |

Notes:
- `effector_hold_vel` is set to 0.0 for all cells — the C&S setup penalises only position
  during hold, not velocity.
- `effector_pos_late`, `effector_vel_late`, and `effector_final_vel` are all 0.0: the
  `(t/T)^6` schedule subsumes the function of the late-window terms.
- `effector_pos_running` weight 10.0 (same as `effector_hold_pos`) for all cells —
  the schedule shape does the heavy lifting; the outer weights balance hold vs reach.
- `p_catch_trial 0.5` maintains the Shahbazi 2025 §4.2 catch-trial protocol.
- `--adversary-type linear_dynamics` is kept for consistency with the current production
  pipeline; `--n-adversary-batches 0` means no actual adversarial phase runs.

## Shared flags

These are common across all six cells:

```
--hidden-type gru
--n-warmup-batches 12000
--n-adversary-batches 0
--batch-size 250
--n-replicates 5
--seed 42
--effector-hold-pos 10.0
--effector-hold-vel 0.0
--effector-pos-running 10.0
--effector-pos-late-weight 0.0
--effector-vel-late 0.0
--effector-final-vel 0.0
--nn-output-pre-go 0.0
--nn-hidden-derivative-pre-go 0.0
--no-loss-update-enabled
--position-powerlaw-power 6.0
--adversary-type linear_dynamics
--checkpoint --fused --no-streaming-loss
--checkpoint-every 1000
```

Note: `p_catch_trial = 0.5`, `nn_output = 1e-5`, and `nn_hidden = 1e-5` are hardcoded
defaults in `build_hps` and do not have separate CLI flags.

## Production CLI invocations

Run from `/workspace/rlrmp` on the pod. Create log dir first: `mkdir -p /workspace/logs`.

### Cell 1 — lit__flat_jerk (jerk on, flat schedule)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-pos 10.0 \
  --effector-hold-vel 0.0 \
  --effector-pos-running 10.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --nn-output-jerk 1e5 \
  --nn-output-pre-go 0.0 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --effector-pos-running-schedule flat \
  --effector-hold-pos-schedule flat \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --output-dir _artifacts/f47abb1/runs/lit__flat_jerk \
  > /workspace/logs/lit__flat_jerk.log 2>&1 &
```

### Cell 2 — lit__post_jerk (jerk on, post-go powerlaw)

Running position term uses `(t/T)^6`; hold position term remains flat.

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-pos 10.0 \
  --effector-hold-vel 0.0 \
  --effector-pos-running 10.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --nn-output-jerk 1e5 \
  --nn-output-pre-go 0.0 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --effector-pos-running-schedule powerlaw \
  --effector-hold-pos-schedule flat \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --output-dir _artifacts/f47abb1/runs/lit__post_jerk \
  > /workspace/logs/lit__post_jerk.log 2>&1 &
```

### Cell 3 — lit__full_jerk (jerk on, full-trial powerlaw on both terms)

Both hold position and running position terms use `(t/T)^6` (full-trial normalisation).

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-pos 10.0 \
  --effector-hold-vel 0.0 \
  --effector-pos-running 10.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --nn-output-jerk 1e5 \
  --nn-output-pre-go 0.0 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --effector-pos-running-schedule powerlaw \
  --effector-hold-pos-schedule powerlaw \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --output-dir _artifacts/f47abb1/runs/lit__full_jerk \
  > /workspace/logs/lit__full_jerk.log 2>&1 &
```

### Cell 4 — lit__flat_nojerk (jerk off, flat schedule)

Control cell: no jerk regulariser, flat position schedule.

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-pos 10.0 \
  --effector-hold-vel 0.0 \
  --effector-pos-running 10.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --nn-output-jerk 0.0 \
  --nn-output-pre-go 0.0 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --effector-pos-running-schedule flat \
  --effector-hold-pos-schedule flat \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --output-dir _artifacts/f47abb1/runs/lit__flat_nojerk \
  > /workspace/logs/lit__flat_nojerk.log 2>&1 &
```

### Cell 5 — lit__post_nojerk (jerk off, post-go powerlaw)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-pos 10.0 \
  --effector-hold-vel 0.0 \
  --effector-pos-running 10.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --nn-output-jerk 0.0 \
  --nn-output-pre-go 0.0 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --effector-pos-running-schedule powerlaw \
  --effector-hold-pos-schedule flat \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --output-dir _artifacts/f47abb1/runs/lit__post_nojerk \
  > /workspace/logs/lit__post_nojerk.log 2>&1 &
```

### Cell 6 — lit__full_nojerk (jerk off, full-trial powerlaw on both terms)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-pos 10.0 \
  --effector-hold-vel 0.0 \
  --effector-pos-running 10.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --nn-output-jerk 0.0 \
  --nn-output-pre-go 0.0 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --effector-pos-running-schedule powerlaw \
  --effector-hold-pos-schedule powerlaw \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --output-dir _artifacts/f47abb1/runs/lit__full_nojerk \
  > /workspace/logs/lit__full_nojerk.log 2>&1 &
```

## Wall-clock and cost estimates

- 12000 warmup batches × 250 trials × 5 replicates on RTX 4090: ~40–50 min per cell.
- 6 cells run sequentially on one pod: ~4–5 hr total; ~$0.45–$0.55 per cell, ~$0.50 total.
- RTX 5090 (EUR-IS-2, ~$0.62/hr): ~30–40 min per cell; ~$0.35–$0.45 per cell, ~$2.20–$2.70 total
  sequential.
- **Recommended**: run all 6 sequentially on one 4090 pod to minimise cost.

## Monitoring

Use the runbook cadence from CLAUDE.md §7: check 1 min after start (JIT compilation
visible?), every 5 min through early loss decline, every 30 min once steadily descending.
Watch for `ptxas` warnings, OOM, and Traceback patterns alongside loss-progress signal.

## Decision criteria

**Primary metric**: vel-RMSE ratio = max(pairwise_RMSE) / median(velocity_peak).
**Threshold for "good convergence"**: vel-RMSE ratio < 0.5 (replicates converge to
similar velocity profiles).

**Auxiliary metrics**:
- CV (coefficient of variation across 5 replicates): lower is better.
- Hold drift: mm of forward motion in [−200 ms, 0 ms] pre-go window (lower is better).
- Peak velocity: m/s at peak of the reach (sanity check; should be 0.3–0.8 m/s).
- Time-to-peak: steps after go cue (should be 40–80 steps ≈ 400–800 ms).

**Winning condition**: the cell with the lowest vel-RMSE ratio that also passes the hold-drift
threshold (< 0.5 mm pre-go drift) is the candidate new production loss schedule.

**Comparison to prior matrix**: cells 1 and 4 (flat schedule) should be directly
comparable to the `gru__jerk` and `gru__jerk_loss_historical` cells from the 2bc95fd matrix.
If cell 1 (jerk on, flat) significantly outperforms all prior cells, that was the dominant
factor and the schedule doesn't matter; if cells 3 or 6 (full powerlaw) win, the C&S
schedule provides an independent benefit.

## Conditional follow-up

If jerk-disabled cells (4–6) show significant anticipation (pre-go hold drift > 1 mm
or visible pre-go velocity ramp in the state-space plots), reintroduce `--nn-output-pre-go`
as a follow-up matrix lever.  The suggested starting weight is 1e-2 (per efc4d68 matrix),
adjusted to avoid disrupting convergence.
