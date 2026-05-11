# `_artifacts/` — bulk reproducible outputs

This directory mirrors `results/` and holds **bulk** artifacts that are
reproducible from the specs in `results/`. It is gitignored in its entirety
(see the role-based block in the repo `.gitignore`).

## What lives here

- Model checkpoints (`.eqx`)
- Training history pytrees, optimizer state
- Large evaluation arrays (`.npz`, `.pkl`)
- Training logs (`.log`)
- Full-DPI figure renders (`.html`, large `.png`, animations)
- Cloud-runner output trees (paths that previously embedded `runpod/` etc.;
  see Bug `f485c26` — cloud-provider subdir leakage has been removed)

## What does NOT live here

- Run specs (`run.json`) — those go under `results/<hash>/runs/<variant>.json`
- Figure specs (`spec.json`) — those go under `results/<hash>/figures/<topic>/`
- Narratives (`*.md`) — those go under `results/<hash>/notes/`
- Small Plotly figure JSON / thumbnail PNG (committed under `results/` when
  under threshold; see CLAUDE.md → `Experiment Artifacts: Tracked vs Ignored`)

## Layout (post-f485c26 flat-by-hash)

Each top-level entry under `_artifacts/` is a directory named by its 7-char
tracking-issue prefix. The path under `_artifacts/<hash>/...` mirrors the
equivalent path under `results/<hash>/...`. Examples:

```
results/efc4d68/runs/baseline_gru__none.json    <- spec (tracked, flat)
_artifacts/efc4d68/runs/baseline_gru__none/     <- bulk (ignored)
    ├── adversarial_model.eqx
    ├── checkpoints_warmup/
    ├── train.log
    └── adversarial_losses.npz

results/2bc95fd/figures/peak_velocity_distributions/spec.json  <- spec (tracked)
_artifacts/2bc95fd/figures/peak_velocity_distributions/        <- bulk renders (ignored)
    └── figure.html
```

The relative-symlink from the spec dir to the render file (created
automatically by `feedbax.plot.save_figure` per rlrmp's registered
`figure_routing` config) makes navigation feel single-tree locally while
preserving the role-based git-track / git-ignore split.

## Why a sibling tree (and not co-located heavy + light)?

A separate top-level `_artifacts/` lets `du -sh _artifacts/` and
`du -sh results/` answer different questions cleanly: "how much disk does
the heavy data take" vs "how big is the committable spec/narrative
registry". It also makes `_artifacts/` trivially relocatable to a
separate filesystem mount, SSHFS share, or rsync target without touching
repo paths.

## Run-ID naming: `<group>__<variant>` (Phase 2 completion, Bug `0077b42`)

Run identifiers use the canonical `<group>__<variant>` double-underscore
form (see CLAUDE.md → *Experiment Artifacts*). Single underscore is
ambiguous: `running_cost_standard` could be parsed as group `running` +
variant `cost_standard`, or `running_cost` + `standard`. Double underscore
splits unambiguously and matches the branch-naming convention used
elsewhere in the project.

## Pre-f485c26 legacy paths

The previous layout nested experiments under `results/part2_5/` and used
cloud-provider subdirs like `runpod/`. After the f485c26 reorg, every
experiment lives at `results/<hash>/...` and `_artifacts/<hash>/...`. The
legacy `models/` block and pre-migration centerout dirs are archived
under `results/2ef67ca/models/` and `_artifacts/2ef67ca/models/`
respectively. Future runs do **not** add phase-named or provider-named
dirs.
