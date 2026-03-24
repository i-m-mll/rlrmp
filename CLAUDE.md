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
- **Module instances are frozen (immutable).** You cannot assign to `self.field` after `__init__`. This is enforced by `dataclasses.FrozenInstanceError`. Never use `self.x = ...` outside `__init__`; never use `dataclasses.replace` on Modules with computed fields (it calls `__init__` with all field values as kwargs, which fails if `__init__` computes some fields internally).
- Use `eqx.tree_at` for out-of-place updates — it returns a new Module with the specified leaves replaced. Prefer `eqx.tree_at` over `dataclasses.replace` in all cases.
- Use `eqx.field` for defaults/converters. Only implement custom flattening when necessary; otherwise rely on `Module`'s default behavior.
- When a Module has a custom `__init__` that computes fields (e.g., `self.state_index = StateIndex(self._initial_state)`), reconstructing the Module requires calling `__init__` with the constructor arguments, not `dataclasses.replace` (which passes ALL fields including computed ones).

### JAX Tree API
- Import once as `import jax.tree as jt` and use the `jt.*` namespace throughout (e.g., `jt.map`, `jt.leaves`, `jt.structure`, `jt.flatten`, `jt.unflatten`).
- Do not use deprecated `jax.tree_*` helpers (e.g., `jax.tree_map`, `jax.tree_leaves`). Prefer `jax.tree.*` consistently.

### jax_cookbook Helpers
- `import jax_cookbook.tree as jtree` for PyTree utilities not in core JAX (e.g., `jtree.unzip`, `jtree.get_ensemble`). Use `jtree.*` for these helpers.
- `from jax_cookbook import is_type, is_module, is_none` for convenient shorthands. For example, `jt.map(..., is_leaf=is_type(tuple))`, `jt.map(..., is_leaf=is_module)` (for `equinox.Module` instances).

### TPU Multi-Process (Independent Jobs on Separate Chips)
To run N independent training jobs on a multi-chip TPU (e.g., v4-8 with 4 chips), use these env vars per process — NO `jax.distributed.initialize()`:
```bash
TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=<chip_id> python train.py
```
Each process sees exactly 1 device via `jax.devices()`. They share no state. Clear `/tmp/libtpu_lockfile` before launching. Source: [Skye's canonical gist](https://gist.github.com/skye/f82ba45d2445bb19d53545538754f9a3).

### Cloud/Remote Training Practices
- **Always verify the latest code is deployed** before running on cloud instances. Stale code on TPU/GPU is a recurring source of wasted time.
- **Never kill processes on TPU VMs via SSH.** Any `kill`, `pkill`, or signal sent during an SSH command can disrupt the SSH session itself, causing the connection to drop. If a process has crashed, just clear `/tmp/libtpu_lockfile` and launch a new one — the crashed process is already dead.
- **Use `uv` for all package management**, including on cloud instances. Do not use `pip install` directly (this is also in the global CLAUDE.md but bears repeating for remote contexts where habits may slip).

### Experiment Results [TEMPORARY]
When adding new training results to `results/part2_5/`, update `results/part2_5/README.md` with a row in the appropriate table and a brief description. This keeps the results navigable for the user.
