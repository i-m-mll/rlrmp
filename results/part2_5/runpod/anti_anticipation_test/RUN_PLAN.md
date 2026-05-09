# Anti-anticipation 5090 follow-up matrix

**Tracking:** `efc4d68` (smoothness retrain umbrella).
**feedbax dependency:** `50507a9` (`EpochMaskedLoss`), feature branch
`feature/loss-epoch-masked` until merged.
**Status:** SPEC ONLY — do not deploy without explicit user approval.
**Surfaced:** 2026-05-09, after the previous 8-cell baseline matrix
identified `gru + jerk` as the best architecture/regulariser combination
but left unexplained pre-go anticipatory motor output as a residual quality
issue.

## Motivating question

Does penalising the controller force during the pre-go (hold + target_on)
window suppress anticipatory motor output without regressing post-go reach
quality?

The prior 8-cell run showed `gru__jerk` produced clean post-go reach
trajectories but visible pre-go controller activity — the controller
"warming up" before the go cue. Output-jerk alone does not address this:
jerk is a smoothness penalty on the *time-derivative*, which permits a
nonzero baseline force as long as it is steady. The new
`EpochMaskedLoss(TargetStateLoss, epoch_indices=(0,1))` directly penalises
the magnitude of the controller force during the pre-go window only.

## Matrix (4 cells × 5 replicates = 20 trained models)

| # | Architecture | `--nn-output-jerk` | `--nn-output-pre-go` | `--nn-hidden-derivative-pre-go` | Run label |
|---|--------------|--------------------|----------------------|----------------------------------|-----------|
| 1 | GRU          | `1e5`              | `0.0`                | `0.0`                            | `gru__jerk` (control, replicates the previous winner) |
| 2 | GRU          | `1e5`              | `1e-2`               | `0.0`                            | `gru__jerk_motor_pre` (primary intervention) |
| 3 | GRU          | `1e5`              | `0.0`                | `1e-2`                           | `gru__jerk_hidden_pre` (variant — also suppresses preparation) |
| 4 | GRU          | `1e5`              | `1e-2`               | `1e-2`                           | `gru__jerk_both_pre` (both pre-go terms; control comparator) |

## Production CLI invocations

Output dirs follow the rlrmp `<group>__<variant>` convention; the group is
`anti_anticipation_test`, variants are the labels above.

```bash
# Cell 1: jerk only (control, replicates previous winner)
uv run python scripts/train_minimax.py \
  --hidden-type gru --n-replicates 5 \
  --n-warmup-batches 12000 --n-adversary-batches 0 --batch-size 5000 \
  --streaming-loss \
  --nn-output-jerk 1e5 \
  --output-dir _artifacts/part2_5/runpod/anti_anticipation_test/runs/gru__jerk \
  --checkpoint --fused

# Cell 2: jerk + pre-go motor penalty (primary intervention)
uv run python scripts/train_minimax.py \
  --hidden-type gru --n-replicates 5 \
  --n-warmup-batches 12000 --n-adversary-batches 0 --batch-size 5000 \
  --streaming-loss \
  --nn-output-jerk 1e5 --nn-output-pre-go 1e-2 \
  --output-dir _artifacts/part2_5/runpod/anti_anticipation_test/runs/gru__jerk_motor_pre \
  --checkpoint --fused

# Cell 3: jerk + pre-go hidden-derivative penalty (variant)
uv run python scripts/train_minimax.py \
  --hidden-type gru --n-replicates 5 \
  --n-warmup-batches 12000 --n-adversary-batches 0 --batch-size 5000 \
  --streaming-loss \
  --nn-output-jerk 1e5 --nn-hidden-derivative-pre-go 1e-2 \
  --output-dir _artifacts/part2_5/runpod/anti_anticipation_test/runs/gru__jerk_hidden_pre \
  --checkpoint --fused

# Cell 4: jerk + BOTH pre-go penalties (comparator)
uv run python scripts/train_minimax.py \
  --hidden-type gru --n-replicates 5 \
  --n-warmup-batches 12000 --n-adversary-batches 0 --batch-size 5000 \
  --streaming-loss \
  --nn-output-jerk 1e5 --nn-output-pre-go 1e-2 --nn-hidden-derivative-pre-go 1e-2 \
  --output-dir _artifacts/part2_5/runpod/anti_anticipation_test/runs/gru__jerk_both_pre \
  --checkpoint --fused
```

## Resource estimate

- 4 cells × 5 replicates = 20 trained models.
- Replicates run vmapped (single training process per cell).
- Wall-clock estimate: ~30-40 min on 5090, smaller than the 8-cell run.
- One pod, sequential cells. Matches the existing 5090 run pattern from the
  previous baseline matrix.

## Suggested initial weights — rationale

`--nn-output-pre-go 1e-2`:
- Existing post-aggregated `nn_output` weight is `1e-5` over the full
  trial.
- Pre-go window is roughly 1/3 of the trial; effective comparable weight
  on the pre-go region alone would be `~3e-5`.
- 1e-2 is ~330x the effective comparable weight. Strong enough to force
  the controller to suppress pre-go force, weak enough that it should not
  destabilise post-go reach dynamics.
- This is a guess; an explicit weight sweep `[1e-3, 1e-2, 1e-1]` is the
  natural follow-up if the primary intervention shows no effect or
  destabilises.

`--nn-hidden-derivative-pre-go 1e-2`:
- Mirrors the `--nn-hidden-derivative` Shahbazi et al. (1e-3) but applied
  only pre-go. The pre-go window is shorter than the full trial, so a
  modestly larger weight is justified to produce comparable per-step
  pressure.
- Note: this term suppresses *both* anticipatory motor output AND
  null-space preparatory hidden activity, which may be undesirable. The
  user has flagged this as a risk; this cell is included as a comparator,
  not the primary intervention.

## Smoke validation

Local CPU smoke runs (5 batches × 50 trials × 2 replicates) on
2026-05-09 verified that:
- All four configurations complete cleanly with no JIT or runtime errors.
- The new terms are registered in the loss tree and report finite values.
- Setting both new weights to 0 (`baseline_gru__none_repro`) produces a
  loss curve identical to the prior `baseline_gru__none` run, confirming
  the new terms are no-ops at zero weight.

Smoke artifacts under `_artifacts/anti_anticipation_smoke/` (gitignored).

## Success criteria for the 5090 run

1. Cell 2 (`gru__jerk_motor_pre`) shows reduced pre-go controller-force
   magnitude vs Cell 1, measured as `mean(||u_t||²)` for `t < go_idx`
   averaged over validation trials.
2. Cell 2 does NOT show worse post-go reach quality vs Cell 1, measured as
   the existing `effector_pos_late` validation loss.
3. Cell 3 (`gru__jerk_hidden_pre`) is informative as a secondary
   intervention — expected to suppress pre-go hidden activity (including
   null-space preparation), which the user has flagged as a risk.
4. Cell 4 confirms that combining both pre-go terms doesn't introduce
   pathological interactions.
