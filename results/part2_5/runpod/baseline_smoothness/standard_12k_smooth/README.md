# baseline_smoothness/standard_12k_smooth

A/B retrain of the existing Part 2.5 baseline (`_artifacts/part2_5/runpod/baseline/standard_12k/`) with the compositional `||h_t - h_{t-1}||²` hidden-state smoothness penalty enabled at weight `1e-3`. This is the rlrmp wiring for issue `efc4d68` and feedbax issue `e8f3738` (StateDerivativeLoss).

## Single delta vs. baseline (intent)

Identical to `baseline/standard_12k` in every respect except:

```diff
   loss.weights:
     ...
     nn_output: 1e-5
     nn_hidden: 1e-5
+    nn_hidden_derivative: 1e-3   # was implicitly 0 (term not present)
```

All other hyperparameters — replicates (5), warmup batches (12000), adversary batches (0), batch size (250), hidden size (180), feedback delay (5), motor noise (0.01), damping (10.0), tau_rise (0.05), GRU recurrent cell, additive SISU gating, the running + late + hold loss schedule — are unchanged. This makes any difference in trajectory-level replicate variability attributable to the smoothness term alone.

## Forced auxiliary delta: streaming loss disabled

`feedbax.streaming_loss.make_streaming_loss_fn` only supports `TargetStateLoss`
and `ModelLoss` (per-step or state-independent). `StateDerivativeLoss` is
fundamentally cross-timestep (needs `h_t` and `h_{t-1}`), so we set
`streaming_loss = false` in this run. This restores the full-trajectory
evaluation path, identical to what feedbax used before streaming was added.
Memory impact is modest (~120 MB rollout buffer per JIT compile, see
DEPLOY_PLAN.md §VRAM); 24 GB 4090 is sufficient.

A future feedbax extension could add streaming support for
`StateDerivativeLoss` by carrying `h_{t-1}` as a scan-state field; tracked
implicitly via feedbax issue `e8f3738`.

## Training command

Run from the repo root inside the rlrmp `feature/baseline-retrain-with-smoothness` worktree (or `/workspace/rlrmp` on the RunPod pod after the `sed` patches):

```bash
uv run python scripts/train_minimax.py \
  --n-warmup-batches 12000 \
  --n-adversary-batches 0 \
  --controller-lr 1e-4 \
  --hidden-type gru \
  --sisu-gating additive \
  --nn-hidden-derivative 1e-3 \
  --loss-update-enabled --loss-update-ratio 0.3 \
  --no-streaming-loss \
  --fused \
  --checkpoint --checkpoint-every 500 \
  --seed 0 \
  --output-dir /workspace/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth
```

The `--spec-dir` is omitted; the script derives it via the mirror invariant
(`_artifacts/<exp>/runs/<run>/` ↔ `results/<exp>/runs/<run>/`), so `run.json` lands at
`results/part2_5/runpod/baseline_smoothness/standard_12k_smooth/run.json`.

## Intended args namespace (for reference)

The training script writes the actual `run.json` (under the same path as this
README) at launch time, capturing `vars(args)` plus git/GPU metadata. The
intent for this run is:

```json
{
  "n_warmup_batches": 12000,
  "n_adversary_batches": 0,
  "n_adversary_steps": 5,
  "adversary_lr": 0.0003,
  "controller_lr": 0.0001,
  "n_bumps": 3,
  "force_max": 1.0,
  "n_adversaries": 1,
  "adv_batch_size": null,
  "warmup_model": null,
  "output_dir": "/workspace/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth",
  "seed": 0,
  "checkpoint": true,
  "checkpoint_every": 500,
  "resume": false,
  "loss_update_enabled": true,
  "loss_update_ratio": 0.3,
  "fused": true,
  "streaming_loss": false,
  "hidden_type": "gru",
  "sisu_gating": "additive",
  "nn_hidden_derivative": 0.001
}
```

The single delta vs. the existing baseline `_artifacts/part2_5/runpod/baseline/standard_12k/config.json` is `nn_hidden_derivative: 0.001` (was implicitly 0) and the forced `streaming_loss: false` documented above.

## Outputs

- Spec recipe (this directory): `config.json` (intent), `run.json` (actual args + git/gpu metadata, written by the script).
- Bulk artifacts (gitignored): `_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth/`
  - `warmup_model.eqx` — trained controller (5 replicates).
  - `checkpoints_warmup/` — per-500-batch checkpoints.
  - `loss_history.npz` — per-batch + per-replicate scalar losses (incl. `nn_hidden_derivative` if present in `streaming_loss`).

## A/B comparison plan

After this run completes, compare against the existing baseline at
`_artifacts/part2_5/runpod/baseline/standard_12k/`:

- **Primary metric**: within-condition pairwise RMSE on post-go forward velocity profiles
  (currently 0.39–0.45 m/s on baseline; target reduction toward across-condition RMSE so inter-condition comparisons become valid).
- **Sanity checks**: training loss converges to a similar level; effector position error not catastrophically worse (smoothness should regularise the hidden state, not break the task).

If the term alone is insufficient, follow-up steps from `efc4d68`:
1. Add jerk penalty on output (1e5).
2. Bump existing `nn_hidden`/`nn_output` from 1e-5 to 1e-3.
3. Vanilla-RNN ablation.
4. Low-rank recurrent constraint.

## Provenance

- feedbax feature branch: `feature/loss-hidden-derivative` (from `develop`), commit pending merge.
- rlrmp feature branch: `feature/baseline-retrain-with-smoothness` (from `main`).
- pyproject `[tool.uv.sources.feedbax].path` temporarily points at the feedbax feature worktree; will be reverted to `develop` once that branch lands.
- Bug: efc4d68
