# Bench: throughput benchmark runs (reconstructed)

**Reconstructed.** This experiment dir's `runs/*.json` specs were reconstructed during the f485c26 results/ reorganisation. The original work was deployed to RunPod without going through the post-training-run protocol (CLAUDE.md §9), so no `run.json` was committed at the time. Reconstruction sources: `_artifacts/<orphan>/config.json` (CLI flags + training hyperparameters), filesystem layout (checkpoints/log presence), and the RunPod deploy runbook conventions. Confidence: **high** — the reconstruction is straightforward path-normalisation of a config.json that already carried all training hyperparameters.

## Scope

Reconstructed run specs for the parallel-process GPU throughput benchmarks (`bench__bs5000_proc{1..4}`) used to validate XLA fusion + scaling on RunPod.

## Per-run inputs

Run specs live in `runs/<variant>.json` (flat). The corresponding bulk artifacts (model checkpoints, `.eqx`, training logs, `.npz`) live under `_artifacts/cdfc132/runs/<variant>/`.

## Cross-refs

- Reorg/spec-archaeology context: rlrmp issue `f485c26`.
