# `_artifacts/` — bulk reproducible outputs

This directory mirrors `results/` and holds **bulk** artifacts that are
reproducible from the specs in `results/`. It is gitignored in its entirety
(see the role-based block in the repo `.gitignore`).

## What lives here

- Model checkpoints (`.eqx`)
- Training history pytrees, optimizer state
- Large evaluation arrays (`.npz`, `.pkl`)
- Training logs (`.log`)
- Full-DPI figure renders (large `.html`, big `.png`, animations)
- Cloud-runner output trees (e.g., what was previously `results/<exp>/runpod/...`)

## What does NOT live here

- Run specs (`run.json`) — those go under `results/<exp>/runs/<run>/`
- Figure specs (`spec.json`) — those go under `results/<exp>/figures/<fig>/`
- Narratives (`*.md`) — those go under `results/<exp>/`
- Small Plotly figure JSON / thumbnail PNG (committed under `results/` when
  under threshold; see CLAUDE.md → `Experiment Artifacts: Tracked vs Ignored`)

## Layout

The path under `_artifacts/<exp>/...` mirrors the equivalent path under
`results/<exp>/...`. Code resolves one from the other via a single helper
(see `src/rlrmp/paths.py` once it lands). Examples:

```
results/part2_5/runs/baseline__standard_12k/run.json     <- spec (tracked)
_artifacts/part2_5/runs/baseline__standard_12k/          <- bulk (ignored)
    ├── adversarial_model.eqx
    ├── checkpoints_warmup/
    ├── train.log
    └── adversarial_losses.npz

results/part2_5/figures/peak_velocity_by_sisu/spec.json  <- spec (tracked)
_artifacts/part2_5/figures/peak_velocity_by_sisu/        <- bulk renders (ignored)
    └── figure_full_dpi.html
```

## Why a sibling tree (and not co-located heavy + light)?

A separate top-level `_artifacts/` lets `du -sh _artifacts/` and
`du -sh results/` answer different questions cleanly: "how much disk does
the heavy data take" vs "how big is the committable spec/narrative
registry". It also makes `_artifacts/` trivially relocatable to a
separate filesystem mount, SSHFS share, or rsync target without touching
repo paths. See the project structure proposal §2/§3 for the full
discussion.
