# Pre-go Motor Mask Follow-up Matrix — Production Run Plan

Tracking issue: 3702f54 (Pre-go motor mask follow-up matrix)
Trigger: f47abb1 (Lit-replication 6-cell matrix) — see issue body and corrective comments
Strategy source: 5acdaae (strategy 1: re-introduce `--nn-output-pre-go` lever)
Training-methods coord: c99ad9d

## Background context

The f47abb1 lit-replication 6-cell matrix identified `lit__post_nojerk` and
`lit__full_nojerk` as the leading no-jerk powerlaw configurations on the primary
metrics (vel-RMSE ratio, CV across replicates, peak-velocity sanity). Both, however,
retained a residual ~2–3 mm of pre-go anticipation (forward drift in the [−200 ms,
0 ms] hold window) — visible in the per-cell state-space plots and called out in the
corrective comments on the f47abb1 tracking issue. The original RUN_PLAN's
"conditional follow-up" section anticipated this outcome and pre-registered the
`--nn-output-pre-go` lever (initially suggested at 1e-2) as the next step. This matrix
formalises and broadens that follow-up.

## Background

Two design dimensions are crossed, with two structurally different suppression
mechanisms:

1. **Position-error rescale** (`__pos10` cells, 2 cells): hold and running position
   weights are scaled 10× from the f47abb1 baseline of 1.0 → 10.0. This tests whether
   the residual drift is simply under-penalised position error. In particular,
   `post_go_pl__pos10` retains a *flat* hold-pos schedule so the 10× rescale is
   actually applied throughout hold; `full_trial_pl__pos10` keeps the powerlaw hold
   schedule (where the hold-period multiplier ends at ≈ 7×10⁻⁴) and therefore tests
   the 10× rescale primarily as an effect on the running term.
2. **Pre-go output regulariser** (`--nn-output-pre-go`, 6 cells): direct penalty on
   network output during the pre-go window. Three weights `{1e-3, 5e-2, 1.0}` span
   ~3 orders of magnitude crossed against two position-rescale levels `{pos×1, pos×10}`,
   all on the `full_trial_pl` schedule. This dimension is on `full_trial_pl` (not
   `post_go_pl`) because `full_trial_pl` has effectively-zero hold-period position
   penalty after the powerlaw — the only effective pre-go pressure comes from catch
   trials (p=0.5), so the `--nn-output-pre-go` lever has a clean independent effect
   to measure.

All 8 cells are no-jerk (the residual-anticipation finding only concerns the no-jerk
cells; the jerk-on cells already pass the hold-drift threshold in f47abb1). The
`__nojerk` suffix is dropped from cell names since the whole matrix is no-jerk.

Two baseline anchors from f47abb1 are included in all downstream comparison plots but
are **not retrained**:

- `lit__post_nojerk` — f47abb1 cell 5; post-go powerlaw, no `--nn-output-pre-go`.
- `lit__full_nojerk` — f47abb1 cell 6; full-trial powerlaw, no `--nn-output-pre-go`.

## Task config (shared with f47abb1)

`epoch_len_ranges = [[0, 1], [10, 30]]` — pure-hold 0 steps; target-on 100–300 ms.
`n_steps = 140`. `dt = 0.01 s`. Same as f47abb1; no task-side changes.

## Cell matrix

All 8 cells share: GRU, 12000 warmup batches, 0 adversary batches, batch size 250,
5 replicates, seed 42, jerk OFF (`--nn-output-jerk 0.0`).

| # | Cell name | `hold_pos` | `pos_running` | hold-pos sched | pos-running sched | `nn_output_pre_go` |
|---|---|---|---|---|---|---|
| 1 | `post_go_pl__pos10` | 10.0 | 10.0 | flat | powerlaw | 0.0 |
| 2 | `full_trial_pl__pos10` | 10.0 | 10.0 | powerlaw | powerlaw | 0.0 |
| 3 | `full_trial_pl__prego_1e-3` | 1.0 | 1.0 | powerlaw | powerlaw | 1e-3 |
| 4 | `full_trial_pl__prego_5e-2` | 1.0 | 1.0 | powerlaw | powerlaw | 5e-2 |
| 5 | `full_trial_pl__prego_1` | 1.0 | 1.0 | powerlaw | powerlaw | 1.0 |
| 6 | `full_trial_pl__pos10_prego_1e-3` | 10.0 | 10.0 | powerlaw | powerlaw | 1e-3 |
| 7 | `full_trial_pl__pos10_prego_5e-2` | 10.0 | 10.0 | powerlaw | powerlaw | 5e-2 |
| 8 | `full_trial_pl__pos10_prego_1` | 10.0 | 10.0 | powerlaw | powerlaw | 1.0 |

## Design rationale

- **Cells 1–2 (`__pos10` baselines, no pre-go output term).** Test whether scaling
  position weight 10× makes the network reach faster (in f47abb1 the no-jerk powerlaw
  cells arrived near the end of the trial) and whether `post_go_pl__pos10` specifically
  — with its *flat* hold schedule — suppresses pre-go anticipation via the now-meaningful
  hold-period position penalty. `full_trial_pl__pos10` is the matched control with the
  powerlaw hold schedule, which renders the 10× hold rescale numerically near-inert.
- **Cells 3–5 (`full_trial_pl__prego_*`, 3 pre-go weights at pos×1).** Test whether
  `--nn-output-pre-go` suppresses residual anticipation in the `full_trial_pl`
  configuration. This is the cleanest test bed for an independent pre-go output
  effect, because `full_trial_pl` has effectively-zero hold-period position penalty
  (powerlaw schedule applied to `effector_hold_pos` makes the multiplier ≈ 7×10⁻⁴ at
  the end of hold) — so any anticipation suppression observed is attributable to the
  pre-go output term, not to coupling with the position term.
- **Cells 6–8 (`full_trial_pl__pos10_prego_*`, 3 pre-go weights at pos×10).** Same
  sweep at the higher position-weight level, in case the pre-go output lever needs to
  combine with a stronger position-error signal to converge cleanly.
- **Weight choices `{1e-3, 5e-2, 1.0}`** span ~3 orders of magnitude. 5e-2 is near
  the efc4d68 starting point of 1e-2 / the f47abb1 follow-up suggestion of 1e-2; 1e-3
  probes whether a weak prior is sufficient given the powerlaw schedule already in
  place; 1.0 probes whether a strong prior breaks convergence or simply suppresses
  more drift.

## Shared flags

Common across all 8 cells:

```
--hidden-type gru
--n-warmup-batches 12000
--n-adversary-batches 0
--batch-size 250
--n-replicates 5
--seed 42
--effector-hold-vel 0.0
--effector-pos-late-weight 0.0
--effector-vel-late 0.0
--effector-final-vel 0.0
--p-catch-trial 0.5
--nn-output 1e-5
--nn-hidden 1e-5
--nn-output-jerk 0.0
--nn-hidden-derivative 1e-3
--nn-hidden-derivative-pre-go 0.0
--no-loss-update-enabled
--position-powerlaw-power 6.0
--adversary-type linear_dynamics
--checkpoint --fused --no-streaming-loss
--checkpoint-every 1000
```

Per-cell flags that vary: `--effector-hold-pos`, `--effector-pos-running`,
`--effector-hold-pos-schedule`, `--effector-pos-running-schedule`,
`--nn-output-pre-go`, `--output-dir`.

## Production CLI invocations

Run from `/workspace/rlrmp` on the pod. Create log dir first: `mkdir -p /workspace/logs`.

### Cell 1 — post_go_pl__pos10 (flat hold schedule, powerlaw running, pos×10, no pre-go output)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 10.0 \
  --effector-pos-running 10.0 \
  --effector-hold-pos-schedule flat \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 0.0 \
  --output-dir _artifacts/3702f54/runs/post_go_pl__pos10 \
  > /workspace/logs/post_go_pl__pos10.log 2>&1 &
```

### Cell 2 — full_trial_pl__pos10 (powerlaw on both, pos×10, no pre-go output)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 10.0 \
  --effector-pos-running 10.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 0.0 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__pos10 \
  > /workspace/logs/full_trial_pl__pos10.log 2>&1 &
```

### Cell 3 — full_trial_pl__prego_1e-3 (powerlaw on both, pos×1, pre-go output 1e-3)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 1.0 \
  --effector-pos-running 1.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 1e-3 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__prego_1e-3 \
  > /workspace/logs/full_trial_pl__prego_1e-3.log 2>&1 &
```

### Cell 4 — full_trial_pl__prego_5e-2 (powerlaw on both, pos×1, pre-go output 5e-2)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 1.0 \
  --effector-pos-running 1.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 5e-2 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__prego_5e-2 \
  > /workspace/logs/full_trial_pl__prego_5e-2.log 2>&1 &
```

### Cell 5 — full_trial_pl__prego_1 (powerlaw on both, pos×1, pre-go output 1.0)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 1.0 \
  --effector-pos-running 1.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 1.0 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__prego_1 \
  > /workspace/logs/full_trial_pl__prego_1.log 2>&1 &
```

### Cell 6 — full_trial_pl__pos10_prego_1e-3 (powerlaw on both, pos×10, pre-go output 1e-3)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 10.0 \
  --effector-pos-running 10.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 1e-3 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__pos10_prego_1e-3 \
  > /workspace/logs/full_trial_pl__pos10_prego_1e-3.log 2>&1 &
```

### Cell 7 — full_trial_pl__pos10_prego_5e-2 (powerlaw on both, pos×10, pre-go output 5e-2)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 10.0 \
  --effector-pos-running 10.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 5e-2 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__pos10_prego_5e-2 \
  > /workspace/logs/full_trial_pl__pos10_prego_5e-2.log 2>&1 &
```

### Cell 8 — full_trial_pl__pos10_prego_1 (powerlaw on both, pos×10, pre-go output 1.0)

```bash
nohup uv run python scripts/train_minimax.py \
  --hidden-type gru \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --batch-size 250 \
  --n-replicates 5 \
  --seed 42 \
  --effector-hold-vel 0.0 \
  --effector-pos-late-weight 0.0 \
  --effector-vel-late 0.0 \
  --effector-final-vel 0.0 \
  --p-catch-trial 0.5 \
  --nn-output 1e-5 \
  --nn-hidden 1e-5 \
  --nn-output-jerk 0.0 \
  --nn-hidden-derivative 1e-3 \
  --nn-hidden-derivative-pre-go 0.0 \
  --no-loss-update-enabled \
  --position-powerlaw-power 6.0 \
  --adversary-type linear_dynamics \
  --checkpoint --fused --no-streaming-loss \
  --checkpoint-every 1000 \
  --effector-hold-pos 10.0 \
  --effector-pos-running 10.0 \
  --effector-hold-pos-schedule powerlaw \
  --effector-pos-running-schedule powerlaw \
  --nn-output-pre-go 1.0 \
  --output-dir _artifacts/3702f54/runs/full_trial_pl__pos10_prego_1 \
  > /workspace/logs/full_trial_pl__pos10_prego_1.log 2>&1 &
```

## Wall-clock and cost estimates

Each f47abb1 cell took ~32 min on RTX 4090. 8 cells × 32 min ≈ 4.3 hr sequential on
one 4090 (~$0.45–$0.55 per cell, ~$3.6–$4.4 total). Could parallelise across multiple
pods/GPUs if budget allows; otherwise sequential on one 4090 is the cheapest path.
RTX 5090 (EUR-IS-2) would shave per-cell time at higher hourly rate — roughly cost-
parity at this batch count; not recommended without explicit go-ahead.

## Monitoring

Use the runbook cadence from CLAUDE.md §7: check 1 min after start (JIT compilation
visible?), every 5 min through early loss decline, every 30 min once steadily
descending. Watch for `ptxas` warnings, OOM, and Traceback patterns alongside
loss-progress signal.

## Decision criteria

**Primary metric**: hold-drift (mm of forward motion in [−200 ms, 0 ms] pre-go window).
**Threshold**: < 0.5 mm pre-go drift across replicates.

**Auxiliary metrics** (must not regress from f47abb1 baselines):
- vel-RMSE ratio = max(pairwise_RMSE) / median(velocity_peak) — should stay < 0.5.
- CV across 5 replicates — should not exceed the f47abb1 baseline cells by > 20 %.
- Peak velocity (m/s) — sanity-check 0.3–0.8 m/s.
- Time-to-peak (steps after go cue) — sanity-check 40–80 steps ≈ 400–800 ms.

**Winning condition**: the cell with the lowest hold-drift that also (i) keeps
vel-RMSE ratio < 0.5 and (ii) does not regress materially on CV is the candidate new
production loss configuration.

## Analysis note

Include `lit__post_nojerk` and `lit__full_nojerk` from f47abb1 in all comparison plots
as baseline anchors — these are not retrained.
