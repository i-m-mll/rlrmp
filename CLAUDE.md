# rlrmp

## Python/JAX Coding Conventions

### Coding Style & Naming
- Follow PEP 8: 4-space indentation, 100-char soft line limit, type hints required for public APIs.
- Always place imports at the top of files, except in the rare case that they should be in a conditional for performance or typing reasons.
- Naming: modules/packages `lower_snake_case`; functions/variables `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Docstrings: Google style; include shapes/dtypes for JAX arrays when relevant.

### Environment Management
- Use `uv` for all package management. Do not run `pip install` directly.

### Equinox Modules
- Subclass `equinox.Module` for dataclasses-that-are-PyTrees; do not add `@dataclass` again. `Module` subclasses are already dataclasses and PyTree nodes.
- Treat `Module` instances as immutable. Use `equinox.tree_at` (or `eqx.tree_at`) for out-of-place updates; avoid direct attribute assignment.
- Use `eqx.field` for defaults/converters. Only implement custom flattening when necessary; otherwise rely on `Module`'s default behavior.

### JAX Tree API
- Import once as `import jax.tree as jt` and use the `jt.*` namespace throughout (e.g., `jt.map`, `jt.leaves`, `jt.structure`, `jt.flatten`, `jt.unflatten`).
- Do not use deprecated `jax.tree_*` helpers (e.g., `jax.tree_map`, `jax.tree_leaves`). Prefer `jax.tree.*` consistently.

### jax_cookbook Helpers
- `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX (e.g., `jtree.unzip`, `jtree.get_ensemble`). Use `jtree.*` for these helpers.
- `from jax_cookbook import is_type, is_module, is_none` for convenient shorthands. For example, `jt.map(..., is_leaf=is_type(tuple))`, `jt.map(..., is_leaf=is_module)` (for `equinox.Module` instances).

### Feedbax Integration
- rlrmp depends on feedbax. When fixing bugs or adding features that require changes to feedbax, make those changes directly in the feedbax repo (on the appropriate feature branch) rather than working around them in rlrmp. Maintain near-integration with feedbax at all times — do not deviate from feedbax's APIs or bypass its abstractions.
- The feedbax repo is at `~/Main/10 Projects/10 PhD/20 Feedbax/feedbax/`. Use worktrees for feature work, following the same conventions as this repo.

## Feedbax Studio
Feedbax Studio (web app) runs from the Feedbax repo. See Feedbax CLAUDE.md for server startup instructions.

## Experiment Artifacts: Tracked vs Ignored

The repo separates artifacts by ROLE, not by directory name:

- **`results/`** is tracked. It holds *specs* (recipes) and *narratives* (prose).
- **`_artifacts/`** is gitignored. It mirrors `results/` and holds *bulk* outputs.
- **Cloud-provider directory names (`runpod/`, `modal/`, etc.) are NOT meaningful** — they all go under `_artifacts/`. Never patch `.gitignore` with a new provider name.

### If you produce X, put it at Y

| You produce | Path |
|---|---|
| Model checkpoint, `.eqx`, training log, large `.npz` | `_artifacts/<exp>/runs/<run>/` |
| Hyperparameters that produced a run | `results/<exp>/runs/<run>/run.json` |
| Per-run commentary (optional) | `results/<exp>/runs/<run>/notes.md` |
| Long-form analysis or post-mortem | `results/<exp>/<topic>_review.md` |
| Figure spec (always) | `results/<exp>/figures/<fig>/spec.json` |
| Figure JSON (only if ≤ 2 MB) | `results/<exp>/figures/<fig>/figure.json` |
| Figure thumbnail (only if ≤ 300 KB at 100 DPI) | `results/<exp>/figures/<fig>/figure.png` |
| Heavy figure render (HTML, full-DPI PNG, MP4) | `_artifacts/<exp>/figures/<fig>/` |
| Final-cut paper figure | `manuscript/figures/<fig>/` (same rules) |

Run identifier convention: `<group>__<variant>` (double underscore separator, matching the branch-naming convention). Examples: `baseline__standard_12k`, `minimax_single__seed_0`.

### Run-spec vs figure-spec

A `run.json` captures hyperparameters that produced model weights — stable, one per run.
A `spec.json` captures plotting parameters and the data-transform pipeline — volatile, many per run, **references** input runs by path. Never inline run hyperparameters into a figure spec.

### Figure-saving helper (when available)

Use `feedbax.plot.io.save_figure_with_spec(fig, spec, dst_dir)` when it exists (tracking: feedbax issue `0eebc71`). Until then, hand-write `spec.json` next to the figure with at minimum: `figure_kind`, `input_artifacts` (paths + sha256), `plot_kwargs`, and `feedbax`/`jax`/`rlrmp` versions.

### Adding a new experiment

1. Create `results/<exp>/README.md` with one paragraph of context.
2. Each run script writes `results/<exp>/runs/<run>/run.json` (spec) and all heavy outputs under `_artifacts/<exp>/runs/<run>/` (mirror).
3. Never write `.eqx`, large `.npz`, full-DPI images, or training logs anywhere under `results/`.

### What NOT to gitignore

Do not add directory-name patterns (`runpod/`, `modal/`, `coreweave/`, `tpu/`, `gpu_box/`) to `.gitignore`. The role-based whitelist already excludes these by construction. If you find yourself wanting to add a name-based ignore, the artifact is in the wrong tree — move it under `_artifacts/`.

### Legacy paths

Pre-migration directories under `results/` (e.g., `centerout_apt_pert1/config.json`, `results/part2_5/models/<name>/config.json`) keep their existing layout for now. The new `<exp>/runs/<group>__<variant>/run.json` convention applies to new experiments; legacy directories migrate opportunistically when their experiments are revisited.

## Issue Coordination

### Project Analyses Coordination

Issue `4d38c15` is the **project analyses coordination issue** (label: `coordination`) — central place for cross-cutting decisions about analyses. **Always check it when:**

- **Discovering a new analysis worth doing.** File a normal issue for the analysis itself, AND comment on the umbrella with: new issue ID, initial tier guess (essential / desirable / auxiliary), one-line rationale.
- **Shifting tier priority** of an existing analysis. Comment on the umbrella, NOT on the individual analysis issue. Include: issue ID(s), old tier → new tier, reason.
- **Cross-cutting findings** affecting multiple analyses (e.g. "X turned out subsumed by Y", "Z became less informative after we changed training method").

Individual analysis issues focus on the analysis itself; how analyses fit into the bigger scheme is dealt with on the umbrella.

**Do NOT reference coordination issues in commit `Bug:` trailers** — that auto-closes them on merge (pending dotfiles issue `49e81d9` to add a label-based skip). Reference only the relevant child issues.
