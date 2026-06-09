# Local CPU Packing Benchmark, 2026-06-09
Benchmarking issue: `cdfc132`. Row-selection context: `4d79e07`.
Protocol: local CPU forced with `JAX_PLATFORM_NAME=cpu` and `JAX_PLATFORMS=cpu`; XLA preallocation disabled; one parent process launched 1, 2, 4, or 8 workers; each worker compiled/warmed with 4 batches, waited at a start barrier, ran 60 s burn-in, then a 90 s measured window with `chunk_batches=1`. Scenario used the production C&S GRU contract from the corrected RunPod packing benchmark: batch 64, 5 replicates, hidden 180, `cs_lss`, `cs2019-rollout`, `full_analytical_qrf`, target-relative multitarget, force/filter feedback, calibrated small perturbation training, LR 3e-3, gradient clip 5. PGD rows used `broad_epsilon_pgd_steps=10`, moderate level, budget scale `3.688240371719434`, step-size fraction 0.25. Delayed rows used go cue 10-30, `p_catch_trial=0.5`, and `nn_output_pre_go=1.0`.
Raw summaries live under `_artifacts/4d79e07/packing_benchmark/local_cpu_20260609/full/`.
## Results
| Row | n=1 b/s | n=2 b/s | n=4 b/s | n=8 b/s | Best n | Best b/s | Peak RSS at best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate no-PGD | 3.377 | 5.281 | 7.354 | 8.381 | 8 | 8.381 | 12.81 GiB |
| Immediate PGD n_steps=10 | 0.344 | 0.589 | 0.818 | 0.995 | 8 | 0.995 | 16.55 GiB |
| Delayed no-PGD | 2.470 | 4.009 | 5.497 | 6.784 | 8 | 6.784 | 16.17 GiB |
| Delayed PGD n_steps=10 | 0.239 | 0.403 | 0.550 | 0.666 | 8 | 0.666 | 20.85 GiB |

## Per-Worker Throughput and RSS
| Row | n | mean worker b/s | peak total RSS | peak worker RSS | elapsed s |
|---|---:|---:|---:|---:|---:|
| Immediate no-PGD | 1 | 3.377 | 1.62 GiB | 1.62 GiB | 166.4 |
| Immediate no-PGD | 2 | 2.641 | 3.26 GiB | 1.64 GiB | 170.7 |
| Immediate no-PGD | 4 | 1.839 | 6.66 GiB | 1.72 GiB | 175.5 |
| Immediate no-PGD | 8 | 1.048 | 12.81 GiB | 1.64 GiB | 187.8 |
| Immediate PGD n_steps=10 | 1 | 0.344 | 1.96 GiB | 1.96 GiB | 175.7 |
| Immediate PGD n_steps=10 | 2 | 0.294 | 4.04 GiB | 2.02 GiB | 176.9 |
| Immediate PGD n_steps=10 | 4 | 0.204 | 8.12 GiB | 2.06 GiB | 191.6 |
| Immediate PGD n_steps=10 | 8 | 0.124 | 16.55 GiB | 2.18 GiB | 212.0 |
| Delayed no-PGD | 1 | 2.470 | 1.83 GiB | 1.83 GiB | 167.6 |
| Delayed no-PGD | 2 | 2.004 | 3.80 GiB | 1.96 GiB | 169.8 |
| Delayed no-PGD | 4 | 1.374 | 7.55 GiB | 1.98 GiB | 175.5 |
| Delayed no-PGD | 8 | 0.848 | 16.17 GiB | 2.19 GiB | 183.9 |
| Delayed PGD n_steps=10 | 1 | 0.239 | 2.36 GiB | 2.36 GiB | 177.8 |
| Delayed PGD n_steps=10 | 2 | 0.201 | 4.81 GiB | 2.44 GiB | 188.0 |
| Delayed PGD n_steps=10 | 4 | 0.138 | 9.67 GiB | 2.43 GiB | 197.8 |
| Delayed PGD n_steps=10 | 8 | 0.083 | 20.85 GiB | 2.75 GiB | 231.5 |

## Interpretation
- Local CPU packing keeps improving aggregate throughput through 8 workers for all four rows, but per-worker throughput drops strongly.
- PGD is roughly an order of magnitude slower than no-PGD at n=1: immediate 0.344 vs 3.377 b/s; delayed 0.239 vs 2.470 b/s.
- Delayed rows are slower than immediate rows under the same PGD/no-PGD setting.
- RSS is approximately linear in worker count and reaches about 20.85 GiB for delayed PGD n=8.
