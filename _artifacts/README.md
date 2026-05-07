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

## Run-ID naming: old → new (Phase 2 completion, Bug `0077b42`)

Phase 2 renamed run identifiers from the legacy single-underscore form to
the canonical `<group>__<variant>` double-underscore form (see CLAUDE.md →
*Experiment Artifacts*: "Run identifier convention: `<group>__<variant>`").
Existing run directories under `_artifacts/<exp>/...` and
`results/<exp>/...` keep their old names until opportunistically migrated.
New runs use the double-underscore form by default.

### Why double underscore?

Single underscore is ambiguous: `running_cost_standard` could be parsed as
group `running` + variant `cost_standard`, or `running_cost` + `standard`,
or any other split. Double underscore is unambiguous: `running_cost__standard`
splits exactly once at `__` and matches the branch-naming convention used
elsewhere in the project.

### Renames applied to default run identifiers

The following table lists the canonical renames for `part2_5` runs whose
defaults appear in `scripts/train_part2_5.py` and the eval scripts. **None
of these have been applied to existing on-disk directories yet** — see the
opt-in migration helper below.

| Old (single underscore) | New (double underscore) |
|---|---|
| `running_cost_standard` | `running_cost__standard` |
| `softmin_standard` | `softmin__standard` |
| `default_standard` | `default__standard` |
| `combined_standard` | `combined__standard` |
| `running_cost_cvar` | `running_cost__cvar` |
| `running_cost_nn1e4` | `running_cost__nn1e4` |
| `running_cost_nn1e6` | `running_cost__nn1e6` |
| `baseline_standard_12k` | `baseline__standard_12k` |
| `baseline_apt` | `baseline__apt` |
| `baseline_cvar` | `baseline__cvar` |
| `baseline_no_pert` | `baseline__no_pert` |
| `baseline_nn1e6` | `baseline__nn1e6` |
| `apt_lr001` | `apt__lr001` |
| `apt_pert2` | `apt__pert2` |
| `tier1_redo` | `tier1__redo` |
| `ratio_sweep` | `ratio__sweep` |
| `mult_pop5` | `mult__pop5` |
| `mult_single` | `mult__single` |
| `vanilla_pop5` | `vanilla__pop5` |
| `vanilla_single` | `vanilla__single` |
| `minimax_single` | `minimax__single` |
| `ratio03_pop5` | `ratio03__pop5` |
| `ratio03_single` | `ratio03__single` |

> The split point is the **last** logical boundary between method/group and
> variant qualifier. When in doubt, see the helper script below — it prints
> proposed renames in dry-run mode for review before any disk writes.

### Migration helper

An opt-in migration helper is provided at
`scripts/migrate_run_ids.sh`. By default it runs in **dry-run** mode and
prints proposed renames; pass `--apply` to actually `mv` directories.

```bash
# Dry run (default): print proposed renames, no disk writes
./scripts/migrate_run_ids.sh

# Apply renames in-place
./scripts/migrate_run_ids.sh --apply

# Restrict to a single experiment subdirectory
./scripts/migrate_run_ids.sh --exp part2_5
./scripts/migrate_run_ids.sh --exp part2_5 --apply
```

The helper walks `_artifacts/<exp>/` and proposes renames for any directory
whose name matches an entry in the rename table above. Out-of-table
directories (and any directory whose name already contains `__`) are left
untouched. Renames are *opportunistic*: pre-migration dirs that nobody
references continue to work as-is; only directories you actually want to
re-canonicalize need the helper.

Note that this only renames directories under `_artifacts/`. If you also
have parallel spec directories under `results/<exp>/runs/`, run the helper
a second time with `--root results` (the script accepts that flag).
