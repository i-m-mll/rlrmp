# Titan Xp Packing Benchmark

Benchmark date: 2026-06-07

Target contract: representative `b8aa38e` proprio row
`target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64`.
The benchmark uses `cs_lss`, full analytical Q/R/Q_f loss, target-relative
multi-target training, delayed force/filter feedback, calibrated perturbation
training at stress level, `hidden_size=180`, `n_replicates=5`, LR `1e-3`,
warmup-cosine with 500 warmup batches, cosine alpha `0.01`, and global-norm clip
`5`. It sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Each run used 2 compile/warmup batches, 60 s burn-in, 90 s measured window,
4-batch chunks, and `schedule_total_batches=1000`. The full 12k training run was
not executed. Worker launches were staggered by 3 s.

| row | workers | aggregate batches/s | mean worker batches/s | max VRAM MiB | mean compile+warmup s | max compile+warmup s |
|---|---:|---:|---:|---:|---:|---:|
| b64_n1 | 1 | 2.8218 | 2.8218 | 2918 | 26.52 | 26.52 |
| b64_n2 | 2 | 2.8334 | 1.4167 | 3776 | 34.08 | 34.49 |
| b64_n3 | 3 | 2.6248 | 0.8749 | 4634 | 44.25 | 45.32 |
| b64_n4 | 4 | 2.1461 | 0.5365 | 5494 | 53.35 | 54.85 |

Interpretation: this Titan Xp does not benefit from packing for this row. Two
workers roughly tie one worker in aggregate throughput, while three and four
workers are slower. Peak memory remains well below 12 GiB even with four
workers, so the limiting factor is throughput/contention rather than capacity.

Local ignored summaries:
`_artifacts/b8aa38e/titan_xp_packing_benchmark/{b64_n1,b64_n2,b64_n3,b64_n4}/summary.json`.

Remote source summaries:
`/media/babbo/rlrmp-titan/benchmarks/b8aa38e_titan/{b64_n1,b64_n2,b64_n3,b64_n4}/summary.json`.
