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

This project uses a small set of long-lived **coordination issues** (label: `coordination`) as decision-tracking surfaces. They are distinct from `umbrella` issues (which bundle a specific phase of work) and from ordinary `feature` / `error` issues (which carry the substantive work). Future agents working in rlrmp must know which coordination issue to comment on when, what to file as a new issue vs. a comment, and how the project keeps these surfaces from becoming a dumping ground.

For git-bug / `mandible issue` command syntax, see the global `~/.claude/CLAUDE.md` **Issue Tracking Commands** convention. This section covers only project-specific coordination protocol.

### The four coordination issues

Each coordination issue has the `coordination` label and is project-lifetime (no auto-close, no phase scope).

| ID | Name | Scope |
|---|---|---|
| `4d38c15` | Project analyses coordination | Cross-cutting decisions about *analyses applied to trained models*. |
| `c99ad9d` | Project training-methods coordination | Cross-cutting decisions about *how models are trained*. |
| `b33e8da` | Project phases coordination | Index of phase umbrellas, phase boundaries, pivots, outcomes. |
| `1d9ae6f` | Project meta coordination | Cross-project / workflow concerns surfaced from rlrmp work (Mandible, feedbax, dotfiles, general workflow). |

#### `4d38c15` — analyses

**Owns:** new analyses worth doing; tier shifts (essential / desirable / auxiliary / deprecated); cross-cutting findings across multiple analyses; deprecation/archival of analyses.

**Does NOT own:** the design/math/implementation/results of any single analysis (those live on the analysis's own issue).

**Triggers:** discovered a new analysis to add; want to re-rank tiers; one analysis subsumes another; an analysis became less informative after a method change.

#### `c99ad9d` — training-methods

**Owns:** training method menu (standard backprop, CVaR, APT, minimax, LEQG, PAI-ASF, BCS, DAI); adversary classes (parametric force fields, GaussianBumpAdversary, structural ΔA); flavor-of-`max` choices (input-instance / model-class / LEQG-via-Whittle); SISU wirings; plant-regime parameters when they couple to training (damping, motor noise, reach geometry, loss schedule); method deprecations and promotions.

**Does NOT own:** specific analyses (→ `4d38c15`); phase markers (→ `b33e8da`); model-structure decisions independent of training method (may eventually move to a separate model-structure coord).

**Triggers:** introducing a new training method or adversary class; redesigning an existing adversary; flavor-of-`max` decisions; cross-method tier shifts; training-relevant plant-regime changes.

#### `b33e8da` — phases

**Owns:** index of phase umbrellas (current and past) with one-line motivating-question + verdict; phase boundaries and pivots; cross-references to phase artifacts (READMEs, synthesis docs).

**Does NOT own:** the work content of any phase (lives on the phase umbrella) or in-phase analyses (→ `4d38c15`).

**Triggers:** starting a new phase / work-bundle (file a `umbrella`-labeled issue for the phase, then comment here); a phase ends, pivots, or is abandoned (follow-up comment with outcome).

#### `1d9ae6f` — meta cross-project

**Owns:** workflow / tooling concerns surfaced during rlrmp work that need a fix in **another** repo — Mandible, feedbax, dotfiles, or general workflow. The body is an index; the actual fixes live in the destination repos.

**Does NOT own:** rlrmp-internal concerns (those go to one of the three coords above or to a normal issue).

**Triggers:** noticing a Mandible bug while working in rlrmp; needing a feedbax API change to support rlrmp work; spotting a global CLAUDE.md gap; identifying a tooling improvement.

### Umbrella vs coordination — which label?

Both labels mark issues that don't carry direct work, but they behave differently:

- **`umbrella`** — phase-tied or work-bundle-tied. **May auto-close** when associated commits merge (this is by design — when the bundle is done, the umbrella closes). Example: `b557d4e` (methodology-fix phase umbrella) auto-closed when its synthesis-review commit merged. The phase work continues on its children; the umbrella's job was just to mark the bundle.
- **`coordination`** — project-spanning decision-tracking surface. **Never auto-closes** — Mandible's commit hook explicitly skips auto-close for `coordination`-labeled issues (implemented in mandible commit `34e6e9c`). These persist for the project's lifetime.

**Decision rule:** "Should this issue close when the work it tracks merges?" — Yes → `umbrella`. No → `coordination`.

### Body content directive (umbrella-verbosity, `1ba096f`)

> Higher-level coordination/umbrella issue **bodies** must be **minimal** — cross-references to children/related issues, plus only material that does not already live in a finer-grained issue. Long-form discussion belongs in the relevant analysis/feature issue. This avoids duplication and reduces maintenance burden as child issues evolve. Comments on the coordination/umbrella are timestamped + threaded; use them for cross-cutting decisions, tier shifts, and similar.

In practice:
- A coordination-issue body should read like a table of contents: scope, what's owned, what isn't, cross-refs to siblings.
- Tier ordering, phase inventory, and similar cross-cutting state belong in **comments**, not the body, because they are timestamped and revisable.
- Substantive findings (results, plots, math) live on the relevant child issue, not on the coordination issue. The coord may carry a one-line cross-ref pointing at the child.

### Decision flow — when to file vs. comment

| Discovery / Decision | File new issue? | Comment on |
|---|---|---|
| New analysis worth doing | yes (`feature`) | `4d38c15` (with tier guess + rationale) |
| Tier shift on existing analysis | no | `4d38c15` (issue ID, old → new tier, reason) |
| Cross-cutting finding across analyses (subsumes / deprecates / reframes) | no | `4d38c15` |
| Deprecating/archiving an analysis | no | `4d38c15` |
| New training method or adversary class to try | yes (`feature`) | `c99ad9d` |
| Adversary class redesign (e.g. structural ΔA lift) | yes (`feature`) | `c99ad9d` |
| Flavor-of-`max` decision (input-instance / model-class / LEQG) | maybe (if requires implementation) | `c99ad9d` |
| Plant-regime parameter change *that couples to training* | maybe | `c99ad9d` |
| Plant-regime parameter change *independent of training* | yes (normal issue) | (no coord — until model-structure coord exists) |
| Starting a new phase / work-bundle | yes (`umbrella`) | `b33e8da` (umbrella ID + one-line motivating question) |
| Phase outcome (merged / abandoned / pivoted) | no | `b33e8da` |
| Mandible / feedbax / dotfiles concern surfaced from rlrmp | yes, **in destination repo** | `1d9ae6f` (destination repo + issue ID + one-liner) |
| General-workflow / tooling improvement | yes, in appropriate repo | `1d9ae6f` |
| Bug or feature *entirely within rlrmp* | yes (`error` / `feature`) | (no coord — direct work) |

When the table doesn't cover your case: ask "is this a project-lifetime decision (→ coord comment) or a unit of work (→ new issue)?". If both, do both — file the issue, then comment on the coord with the issue ID + cross-cutting framing.

### Cross-referencing protocol

- Use **7-character issue-ID prefixes** in bodies and comments (e.g. `4d38c15`, not the full 40-char hash). Matches the style used throughout existing issues and `Bug:` trailers.
- When filing in a destination repo (per `1d9ae6f`), include a one-line "surfaced from rlrmp" note in the destination issue's body, plus the rlrmp issue ID or branch that surfaced it.
- When the destination-repo issue resolves, comment back on `1d9ae6f` with the resolution (merge SHA / closing comment link).
- Coordination issue bodies index children — they do not duplicate child content. If you find yourself pasting a child's results into a coord body, move them to the child and replace with a cross-ref.

### Commit `Bug:` trailers — never reference coordination issues

Even though Mandible's auto-close hook now skips `coordination`-labeled issues (`34e6e9c`), **the convention remains: do not reference coordination issues in commit `Bug:` trailers.** Trailers are for the relevant child / feature / bug issue — the unit of work the commit completes. Coordination issues are decision-tracking surfaces, not commit destinations. The auto-close skip is a safety net, not a green light.

This means `agent-commit --issue <id>` should always take a child / feature / bug issue ID, never `4d38c15` / `c99ad9d` / `b33e8da` / `1d9ae6f`.

### Phase umbrella protocol

When starting a new phase or work-bundle:

1. **Create a phase umbrella** — a new issue labeled `umbrella` (and `feature` if appropriate). Body: minimal — motivating question, scope, links to phase artifacts (e.g. a `results/<exp>/README.md`).
2. **Comment on `b33e8da`** with the phase umbrella ID + one-line motivating question. This is what makes `b33e8da` a discovery surface for "what umbrellas are active right now."
3. **Children of the phase umbrella** reference the umbrella in **their bodies** (e.g. "Part of phase `b557d4e`."), not in their commit `Bug:` trailers. Their trailers reference themselves or their sub-features. The umbrella may auto-close when one of its children merges with a `Bug:` trailer pointing at the umbrella — that is fine and expected; the comment thread on `b33e8da` carries the live phase state regardless.
4. **On phase end / pivot / abandonment**, comment on `b33e8da` with the outcome (one line: "merged via X", "pivoted to Y", "abandoned because Z").

Past phases for orientation (see `b33e8da` comment thread for the live inventory): Part 1 (`297260c`), Part 2 (`0af472c`), Part 2.5 (no umbrella; `results/part2_5/README.md` is the artifact), Methodology-fix (`b557d4e`, currently active).

### Worked example: cross-cutting Riccati flavor-(a) finding

While running `tests/test_hinf_riccati.py::test_cs_faithful_qr_velocity_inflation`, the test xfailed with a substantive diagnosis: faithful C&S Eq. 15 Q,R schedule on the C&S regime gave Δv = −0.04% at 1.5γ\*, identifying that production Riccati's `B_w` is most likely flavor-(a) additive force, not flavor-(b) `ΔA·x` model-class. This is a **cross-cutting** finding because it ties a training-method concern (flavor-of-`max`) to an existing analysis-side issue (`c723082`, LinearDynamicsAdversary).

The protocol followed:

1. **Discovery**: the xfail occurred during a normal test run.
2. **Substantive finding lives in code**: the test's xfail reason string carries the full diagnosis (table, mechanism, implication). This is the implementation-level documentation.
3. **Coordination comment on `c99ad9d`** (training-methods): noted the cross-cutting concern, table of measured Δv, implication for production Riccati `B_w`, cross-ref to `c723082` (the analogous adversary-side lift). This made the finding discoverable from the training-methods coord without inflating that coord's body.
4. **Future follow-up issue** (e.g. "Riccati `B_w` generalization to `ΔA`"): when actually planned, file as a normal `feature` issue, then add a comment on `c99ad9d` cross-referencing it. Do **not** file the follow-up just to have a placeholder.

### What NOT to do

- **Don't comment tier opinions on individual analysis issues.** Tier shifts are cross-cutting; they go on `4d38c15`. Polluting the analysis issue's thread with tier debate makes the analysis issue harder to use as a working ledger.
- **Don't put long-form discussion in coordination issue bodies.** Bodies are tables of contents. Discussion goes in comments (timestamped, threaded) or on the relevant child issue.
- **Don't reference `coordination`-labeled issues in commit `Bug:` trailers.** Use the child / feature / bug issue. The auto-close skip is a safety net, not a substitute for convention.
- **Don't create a new coordination issue when an existing one's scope covers the concern.** Comment on the existing one. New coordination issues are project-lifetime commitments — adding one is a structural change that should be discussed first.
- **Don't paste subagent output or raw analysis into a coord body.** Move it to the child issue (or to a `results/<exp>/` doc) and replace with a one-line cross-ref.
- **Don't index every commit on the coord.** Only commits that change cross-cutting state (a tier, a method choice, a phase boundary) merit a coord comment. Ordinary work-on-a-child does not.
