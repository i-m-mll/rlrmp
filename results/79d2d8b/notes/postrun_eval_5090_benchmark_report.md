# RTX 5090 post-run eval materialization benchmark

## Correction: phase attribution

This report predates the benchmark phase-timing schema added after review. The
tracked timing JSON for `opt11_rollout_product_5090_small` reports only
per-bundle call, post-call ready-block, summary, and total times. Because the
bundle calls return host summaries, XLA compilation, first execution, internal
synchronization/host transfer, Python materialization work, and materializer-
owned writes are all hidden inside the old `call_elapsed_s` field. Treat the
5090 table below as end-to-end per-bundle wall-clock evidence, not as a clean
compile/calculation/transfer decomposition.

Future benchmark reports use schema
`rlrmp.postrun_eval_materialization_benchmark.v2`, which records setup timing,
cold-call timing, optional warm replay, explicit JAX readiness blocks,
output-write timing/classification, report serialization timing, and an
explicit `not_measured` compile/execution split when that split is not available
from the harness boundary.

Benchmark date: 2026-06-22 UTC.

Source implementation: `74177a2c710da3f5e0c22b03e319d227f2e20acc`
(`opt11_rollout_product`) from the local
`feature/79d2d8b-eval-bank-speed` worktree. The local worktree contents were
rsynced to the pod; no remote branch assumption was used.

Pod: RunPod secure cloud RTX 5090, id `t1cfou2i2dcux1`, name
`rlrmp-79d2d8b-gpu-bench-20260621-215214-r71687-1782093134-EU-RO-1-1-7168717820931361`.
The first secure 5090 datacenter candidate (`EU-CZ-1`) had no capacity; the pod
was created in `EU-RO-1`. The pod was deleted after artifact copy-back and
`runpodctl pod list` returned `[]`.

Environment:

- Image: `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204`
- GPU: NVIDIA GeForce RTX 5090
- Driver/CUDA from `nvidia-smi`: driver `580.126.20`, CUDA `13.0`
- Python: `3.12.12`
- JAX: `0.10.2`
- JAX backend/devices: `gpu`, `cuda:0`

Command:

```bash
JAX_PLATFORM_NAME=gpu uv run --no-sync python -m rlrmp.benchmarks.postrun_eval_materialization \
  --step-label opt11_rollout_product_5090_small \
  --perturbation-rows 7 \
  --feedback-bins 5 \
  --worst-case-steps 1 \
  --worst-case-restarts 1 \
  --worst-case-optimizer-backend serial \
  --no-write-bulk-arrays
```

The run initially failed before timing because the deploy sync did not include
the required source-run bulk artifact directory. After syncing only
`_artifacts/020a65b/runs/target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64/`,
the same benchmark command completed.

## Timing

| Bundle | 5090 small (s) | CPU opt4 small (s) | CPU opt11 heavy (s) | 5090 vs CPU opt4 small |
|---|---:|---:|---:|---:|
| standard certificate | 185.51 | 10.78 | 12.23 | 17.2x slower |
| evaluation diagnostics | 56.74 | 1.71 | 3.60 | 33.2x slower |
| pilot figures | 29.59 | 3.09 | 3.93 | 9.6x slower |
| objective comparator | 114.20 | 24.20 | 27.23 | 4.7x slower |
| map decomposition | 7.92 | 3.97 | 4.13 | 2.0x slower |
| perturbation response | 35.56 | 3.05 | 8.08 | 11.7x slower |
| feedback ablation | 60.94 | 12.30 | 13.76 | 5.0x slower |
| worst-case epsilon | 26.35 | 2.37 | 2.81 | 11.1x slower |
| total | 516.81 | 61.46 | 75.78 | 8.4x slower |

The heavier comparable subset was skipped because the small 5090 run took
516.81 s, much longer than both the CPU small subset and the CPU opt11 heavy
subset.

## Interpretation

This benchmark is negative evidence for using a fresh 5090 pod for the current
small all-bundle post-run eval materialization path. JAX allocated about 24.7 GB
on the GPU and intermittent utilization samples reached 98%, 58%, and 5%, but
many samples were 0% and the end-to-end runtime was dominated by startup,
materialization, compilation, and host-heavy bundle work.

The timing JSON is tracked at
`results/79d2d8b/notes/postrun_eval_timing_opt11_rollout_product_5090_small.json`.
The remote benchmark log and environment capture were copied locally under
`_artifacts/79d2d8b/runpod_5090_benchmark/`; those logs are ignored and not
durable Git artifacts.

No completed prior GPU timing result was found in this issue's notes or
comments. The issue history contains a prior planned 4090 GPU benchmark
continuation at `6706b59`, but no completed GPU timing table was present.
