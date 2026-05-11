# RunPod deploy plan — baseline_smoothness/standard_12k_smooth

A/B retrain of `baseline/standard_12k` with the new compositional `||h_t - h_{t-1}||²` hidden-state smoothness term enabled at `1e-3`. Bug: `efc4d68` (rlrmp), `e8f3738` (feedbax).

> This is a **plan only**. Do not execute.

## 1. GPU + cloud choice

- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM).
- **Cloud type**: `COMMUNITY` (cheapest validated option; CLAUDE.md "GPU choice").
- **Datacenter**: omit `--data-center-ids` for COMMUNITY 4090.
- **Disks**: `--container-disk-in-gb 30 --volume-in-gb 30` (matches the existing baseline run; ample headroom for one model's worth of checkpoints + logs).

### VRAM verdict

**4090 (24 GB) is sufficient.** Reasoning:

- Existing baseline `_artifacts/e81f491/runs/baseline__standard_12k/` ran on this exact config (modulo the smoothness term) on a GPU pod and produced a 4.0 MiB model in ~93 min wall-clock — peak VRAM was well below 24 GB.
- The smoothness term itself adds negligible memory: `jnp.diff(h, axis=1)` reuses the rollout's hidden buffer; the `(trial=250, time=130, hidden=180)` float32 trajectory is **~23 MB per replicate**, **~117 MB for the 5-replicate ensemble**.
- We do, however, force `--no-streaming-loss` because `StateDerivativeLoss` is cross-timestep (see config-comment in this run's README, "Forced auxiliary delta"). That changes peak VRAM from streaming (no trajectory stored) → full-trajectory (~117 MB extra during JIT'd forward + matching gradient buffer ~117 MB). Total ≈ 240 MB extra peak, still <2% of 24 GB.
- Optimizer (AdamW: `m`, `v` per param) on a 4 MiB model is ~12 MiB. Batched activations during `eval_trials` are dominated by the 117 MB rollout. JIT compilation cache: <1 GB. Headroom is generous.

No 5090 needed. Single-process, internal ensemble vmap (the existing baseline's pattern). 5 replicates fit comfortably; no parallel-process partitioning required.

## 2. Pre-flight checks

- [x] Docker image `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204` verified on Docker Hub (manifest pulled 2026-05-08; last pushed 2025-12-17). Active and current.
- [ ] At deploy time: `runpodctl datacenter list` to confirm 4090 availability.
- [ ] Verify `~/.runpod/config.toml` has API key configured.
- [ ] Confirm SSH key at `~/.runpod/ssh/RunPod-Key-Go`.

## 3. Pod creation

```bash
runpodctl pod create \
  --image "runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204" \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --cloud-type COMMUNITY \
  --container-disk-in-gb 30 \
  --volume-in-gb 30
```

Poll until SSH ready (~1–2 min). Bail and recreate if `uptimeSeconds == 0` past 2 min (CLAUDE.md cost discipline).

## 4. rsync 3 path-deps to `/workspace`

Run from the local machine. **Note** — the rlrmp source must come from the rlrmp **feature worktree** (`feature__baseline-retrain-with-smoothness`), and the feedbax source must come from the **feedbax feature worktree** (`feature__loss-hidden-derivative`), so the `StateDerivativeLoss` term is available.

```bash
# rlrmp feature worktree (carries the new --nn-hidden-derivative CLI flag and
# the get_reach_loss wiring)
rsync -av --exclude='_artifacts' --exclude='worktrees' --exclude='.venv' \
  "/Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/feature__baseline-retrain-with-smoothness/" \
  root@<pod-ip>:/workspace/rlrmp/

# feedbax feature worktree (carries StateDerivativeLoss + tests). Note the
# path is the FEATURE worktree, NOT develop, until the feedbax PR merges.
rsync -av --exclude='_artifacts' --exclude='worktrees' --exclude='.venv' \
  "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/feature__loss-hidden-derivative/" \
  root@<pod-ip>:/workspace/feedbax/

# jax-cookbook is unchanged
rsync -av --exclude='worktrees' \
  "/Users/mll/Main/05 Utils/jax-cookbook/" \
  root@<pod-ip>:/workspace/jax-cookbook/
```

## 5. Path patches (sed) on the pod

The rlrmp `pyproject.toml` in this feature worktree points at the feedbax FEATURE worktree on the local host. The standard sed pattern still works because it normalises any `feedbax[^"]*` path to `/workspace/feedbax`:

```bash
sed -i 's|.*/feedbax[^"]*|/workspace/feedbax|g' /workspace/rlrmp/pyproject.toml
sed -i 's|.*/feedbax[^"]*|/workspace/feedbax|g' /workspace/rlrmp/uv.lock
sed -i 's|.*/jax.cookbook[^"]*|/workspace/jax-cookbook|g' /workspace/rlrmp/uv.lock
sed -i 's|.*/jax.cookbook[^"]*|/workspace/jax-cookbook|g' /workspace/feedbax/pyproject.toml
sed -i 's|.*/jax.cookbook[^"]*|/workspace/jax-cookbook|g' /workspace/feedbax/uv.lock
```

## 6. Install (nohup + sentinel)

```bash
cd /workspace/rlrmp && \
  nohup uv sync > /workspace/uv_sync.log 2>&1 &

# After uv sync finishes — verify with `tail -1 /workspace/uv_sync.log` showing
# the install summary line — then install the CUDA jaxlib (uv sync resolves a
# CPU-only jaxlib; the post-sync upgrade swaps it for CUDA12 build):
nohup bash -c \
  'cd /workspace/rlrmp && uv pip install -U "jax[cuda12]" && touch /workspace/install_done' \
  > /workspace/jax_install.log 2>&1 &

# Poll: ls /workspace/install_done  (~3–5 min total)
```

CRITICAL (CLAUDE.md §A1): do **not** re-run `uv sync` after `jax[cuda12]` install — it will overwrite the CUDA jaxlib with a CPU-only one. If you need to verify imports, use `uv run --no-sync python ...`.

GPU smoke check before training:
```bash
uv run --no-sync python -c "
import jax, jax.numpy as jnp
print('devices:', jax.devices())
x = jnp.ones((4096, 4096))
print('matmul ok, sum=', float((x @ x).sum()))
from jax import lax
y = jnp.ones((1, 8, 8, 1)); k = jnp.ones((3, 3, 1, 1))
print('cudnn conv ok, out_shape=', lax.conv_general_dilated(y, k, (1,1), 'SAME').shape)
"
# Expect: devices: [CudaDevice(id=0)]; matmul ok; cudnn conv ok.
```

## 7. Smoke test (3-batch warmup, 0 adv batches)

Validates that the new term + wiring works end-to-end on the pod hardware. ~30 s.

```bash
cd /workspace/rlrmp
nohup uv run --no-sync python scripts/train_minimax.py \
  --n-warmup-batches 3 --n-adversary-batches 0 \
  --controller-lr 1e-4 \
  --hidden-type gru --sisu-gating additive \
  --nn-hidden-derivative 1e-3 \
  --loss-update-enabled --loss-update-ratio 0.3 \
  --no-streaming-loss \
  --fused \
  --no-checkpoint \
  --seed 0 \
  --output-dir /workspace/smoke_test_smooth \
  > /workspace/smoke.log 2>&1 &

# Watch for: nn_hidden_derivative term appearing in the per-iteration loss
# breakdown ("nn_hidden_derivative: X.XXe-XX"); no NotImplementedError from
# streaming_loss; no OOM.
tail -f /workspace/smoke.log
```

If the loss breakdown does NOT contain `nn_hidden_derivative`, the wiring failed
— STOP and diagnose before launching the full run.

## 8. Full training run

```bash
cd /workspace/rlrmp
mkdir -p /workspace/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth

nohup uv run --no-sync python scripts/train_minimax.py \
  --n-warmup-batches 12000 --n-adversary-batches 0 \
  --controller-lr 1e-4 \
  --hidden-type gru --sisu-gating additive \
  --nn-hidden-derivative 1e-3 \
  --loss-update-enabled --loss-update-ratio 0.3 \
  --no-streaming-loss \
  --fused \
  --checkpoint --checkpoint-every 500 \
  --seed 0 \
  --output-dir /workspace/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth \
  > /workspace/train_smooth.log 2>&1 && \
  touch /workspace/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth/done.sentinel &
```

### Expected wall-clock

- Existing baseline (streaming-loss enabled) on a GPU pod: **~93 min** for 12k warmup batches.
- This run disables streaming → expect a 10–25% slowdown from the extra rollout-buffer materialisation: **~100–115 min**.
- Smoothness term itself adds one `jnp.diff` + one squared-norm + one `mean` per forward pass; negligible (<1%) compared to the rollout.

### Monitoring cadence (per CLAUDE.md §7)

- 1 min after launch: confirm JIT compile finished, first batch loss printed, `nn_hidden_derivative` shows up in the loss breakdown.
- Every 5 min during early training (first ~30 min): watch for OOM, ptxas warnings, NaN losses.
- Every 30 min after loss stabilises: just confirm progress.
- On completion: `ls _artifacts/.../done.sentinel`, copy results back to local.

### Result rsync (after `done.sentinel`)

```bash
rsync -av root@<pod-ip>:/workspace/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth/ \
  "/Users/mll/Main/10 Projects/10 PhD/rlrmp/_artifacts/part2_5/runpod/baseline_smoothness/standard_12k_smooth/"
```

The `run.json` will land at `results/efc4d68/notes/run.json` automatically via the script's mirror-invariant logic.

## 9. Cost estimate

- 4090 community: ~$0.34 / hr.
- Setup: rsync + uv sync + jax[cuda12] + smoke test = ~10 min.
- Training: ~100–115 min.
- Result rsync + teardown: ~5 min.
- **Total: ~120–130 min, ≈ $0.70 of compute.** Comparable to the existing baseline run (~$0.55 at the same rate).

If 4090 community pricing has shifted, check `runpodctl datacenter list`. Stay on COMMUNITY 4090 unless told otherwise (CLAUDE.md cost discipline §8: never unilaterally upgrade).

## 10. Teardown

```bash
runpodctl remove pod <POD_ID>
```

Confirm the pod is fully removed (not stopped) — the smoothness retrain is a one-shot run; no need to retain storage.

## 11. Open questions / risks

- **Streaming-loss support for `StateDerivativeLoss`** is a future feedbax extension (would require carrying `h_{t-1}` in scan state). Not in scope here. Tracked implicitly via feedbax issue `e8f3738`.
- **`loss_update_enabled`**: the existing baseline uses adaptive `nn_output` weighting (target ratio 0.3). The smoothness term is NOT included in that update — it remains at the configured weight throughout. If the smoothness term turns out to need a similar adaptive schedule, that's a follow-up (not in scope for the A/B).
- **Seed**: this plan uses seed=0 (matches existing baseline). For a proper variance estimate of the smoothness effect, a multi-seed sweep would be the natural next step — but not part of this single A/B. Issue `efc4d68` triggers escalation steps if a single-seed comparison is inconclusive.
