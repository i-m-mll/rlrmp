# Part 2.5 tier-1 training-objective comparison runs (reconstructed)

**Reconstructed.** This experiment dir's `runs/*.json` specs were reconstructed during the f485c26 results/ reorganisation. The original work was deployed to RunPod without going through the post-training-run protocol (CLAUDE.md §9), so no `run.json` was committed at the time. Reconstruction sources: `_artifacts/<orphan>/config.json` (CLI flags + training hyperparameters), filesystem layout (checkpoints/log presence), and the RunPod deploy runbook conventions. Confidence: **high** — the reconstruction is straightforward path-normalisation of a config.json that already carried all training hyperparameters.

## Scope

Reconstructed run specs for the Part 2.5 tier-1 sweep training runs deployed on RunPod (`baseline__standard_12k`, `minimax_single__seed_{0..4}`, `mult__{pop5,single}`, `ratio03__{pop5,single}`, `ratio_sweep__r{02,05}_{gru,vrnn}_{add,mult}_{pop5,single}`, `tier1_redo__r03_*`, `vanilla__{pop5,single}`).

## Per-run inputs

Run specs live in `runs/<variant>.json` (flat). The corresponding bulk artifacts (model checkpoints, `.eqx`, training logs, `.npz`) live under `_artifacts/e81f491/runs/<variant>/`.

## Cross-refs

- Reorg/spec-archaeology context: rlrmp issue `f485c26`.
