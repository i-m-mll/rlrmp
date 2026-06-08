# RunPod RTX 4090 Secure Packing Benchmark

Date: 2026-06-08

Tracking issue: `4d79e07`

Pod: `1m2tnpffvpf3us`, secure-cloud RTX 4090, image
`runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204`.

Environment:

- JAX CUDA wheels installed after `uv sync` with `uv pip install -U "jax[cuda12]"`.
- Verified `jax 0.10.1`, `CudaDevice(id=0)`, and a CUDA matmul before benchmarking.
- Commands used `uv run --no-sync` after CUDA JAX installation to avoid reverting to CPU wheels.
- XLA preallocation disabled with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Benchmark protocol:

- One GPU, parallel process counts `n = 1, 2, 4, 8`.
- Each point used 4 warmup batches, 60 seconds burn-in, 90 seconds measured window,
  4-batch measured chunks, and 5-second GPU sampling.
- Rows use the production-size C&S nominal GRU contract: batch size 64, 5 replicates,
  hidden size 180, C&S LSS backend, `cs2019-rollout`, full analytical Q/R/Qf loss,
  calibrated small proprioceptive perturbation training, LR `3e-3`, warmup+cosine
  schedule, gradient clip 5.
- PGD rows use `broad_epsilon_pgd_training=true`, moderate level, budget scale
  `3.688240371719434`, `broad_epsilon_pgd_steps=10`, and step-size fraction `0.25`.
- Delayed rows use target-present-from-trial-start delayed reach with go cue in steps
  10 to 30 and the current delayed-reach loss/task contract.

Important correction:

The first RunPod PGD benchmark attempt was invalid: the provider-neutral packing
benchmark built PGD hps metadata but did not pass the normal C&S PGD `pre_step_fn`
into `train_pair`. This made PGD throughput nearly identical to no-PGD. The harness
now constructs `make_broad_epsilon_pgd_pre_step(hps.broad_epsilon_pgd_training)` and
passes it to `train_pair`, matching the normal training script. A corrected PGD
sanity run at n=1 measured about `3.57` batches/sec, versus about `11.3` batches/sec
for no-PGD, confirming the fix changed the measured workload.

Raw local artifacts:

- Valid immediate no-PGD: `_artifacts/4d79e07/packing_benchmark/runpod_4090_secure_20260608/v2_valid_immediate_no_pgd/`
- Fixed PGD and delayed rows: `_artifacts/4d79e07/packing_benchmark/runpod_4090_secure_20260608/v4_trimmed/`
- Fixed PGD sanity check: `_artifacts/4d79e07/packing_benchmark/runpod_4090_secure_20260608/sanity_pgd_fixed/`

The pod was stopped after syncing artifacts; final `runpodctl pod get` reported
`desiredStatus: EXITED`.

## Results

Throughput is aggregate measured batches/sec. VRAM is peak sampled GPU memory.

| Row | n=1 b/s | n=2 b/s | n=4 b/s | n=8 b/s | Best n | Best b/s | Peak VRAM at best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate no-PGD | 11.07 | 18.87 | 26.28 | 27.81 | 8 | 27.81 | 8835 MiB |
| Immediate PGD | 3.52 | 3.98 | 4.09 | 4.08 | 4 | 4.09 | 4428 MiB |
| Delayed no-PGD | 9.39 | 15.61 | 19.12 | 19.13 | 8 | 19.13 | 8835 MiB |
| Delayed PGD | 2.50 | 2.69 | 2.67 | 2.72 | 8 | 2.72 | 8867 MiB |

## Interpretation

- Immediate no-PGD scales strongly through 4 workers and only modestly from 4 to 8.
- Immediate PGD is compute-bound and saturates by 2 to 4 workers; 8 workers does not
  improve throughput.
- Delayed no-PGD is slower than immediate no-PGD and effectively saturates at 4 workers.
- Delayed PGD is the slowest row and has little useful packing gain beyond 2 workers.
- VRAM is not the limiting factor for these rows on a 24 GB 4090; compute contention is.

