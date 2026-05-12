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
- Feedbax's protected branch is `develop`, not `main`. All canonical feedbax behaviour, APIs, and architectural patterns reside on `develop`. Feedbax-side feature branches must derive from `develop` (`wt feature/<name> develop`), not from `main`. When reading feedbax source code to understand current behaviour, check out develop or read files at the develop branch reference (`git show develop:path/to/file.py`). The feedbax `main` branch may lag develop substantially and should not be treated as authoritative.

## RunPod Deploy Runbook for rlrmp Experiments

### 1. Prerequisites
- `runpodctl` binary from GitHub releases (the `install.sh` requires sudo; download the binary manually if sudo is unavailable).
- SSH key at `~/.runpod/ssh/RunPod-Key-Go`.
- API key configured: `runpodctl config --apiKey <key>` → writes `~/.runpod/config.toml`.

### 2. Pre-flight checks
(Cross-ref dotfiles `3602840` cost policy + verify-resources comment.)
- Verify the Docker image tag exists on Docker Hub **before** `pod create` — stale tags cause silent deploy failures.
- Check GPU availability per DC: `runpodctl datacenter list`.

### 3. GPU choice
(Cross-ref subagent GPU analysis, Part 2.5 session.)
- **RTX 4090** (community cloud): cheapest validated option.
- **RTX 5090** (Blackwell, EUR-IS-2 or similar): faster, but some templates carry stale image references.
  - Use image `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204` or newer (`cu1290`/`cu1300`) for Blackwell support.
  - Skip the deprecated `runpod/pytorch:2.8.0-...` template.

### 4. Deploy steps
```bash
# 4a. Create pod
runpodctl pod create \
  --image "runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204" \
  --gpu-id "NVIDIA GeForce RTX 5090" \
  --data-center-ids EUR-IS-2 \
  --cloud-type SECURE \
  --container-disk-in-gb 30 \
  --volume-in-gb 30
# (For 4090: use --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY, omit --data-center-ids)

# 4b. Poll until SSH ready (~1–2 min)
runpodctl pod get <POD_ID>   # wait for runtime.ports to populate; bail if >3 min

# 4c. rsync 3 path-deps to /workspace (run from local machine)
rsync -av --exclude='_artifacts' --exclude='worktrees' --exclude='.venv' \
  "/Users/mll/Main/10 Projects/10 PhD/rlrmp/" root@<pod-ip>:/workspace/rlrmp/
rsync -av --exclude='_artifacts' --exclude='worktrees' --exclude='.venv' \
  "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/develop/" root@<pod-ip>:/workspace/feedbax/
rsync -av --exclude='worktrees' \
  "/Users/mll/Main/05 Utils/jax-cookbook/" root@<pod-ip>:/workspace/jax-cookbook/
# Note: jax-cookbook is ~480 MB; omit worktrees/ to avoid transferring ~480 MB of duplicates.

# 4d. Patch embedded local paths on the pod (SSH in, then run):
sed -i 's|.*/feedbax[^"]*|/workspace/feedbax|g' /workspace/rlrmp/pyproject.toml
sed -i 's|.*/feedbax[^"]*|/workspace/feedbax|g' /workspace/rlrmp/uv.lock
sed -i 's|.*/jax.cookbook[^"]*|/workspace/jax-cookbook|g' /workspace/rlrmp/uv.lock
sed -i 's|.*/jax.cookbook[^"]*|/workspace/jax-cookbook|g' /workspace/feedbax/pyproject.toml  # easy to forget
sed -i 's|.*/jax.cookbook[^"]*|/workspace/jax-cookbook|g' /workspace/feedbax/uv.lock

# 4e. Install (must survive SSH disconnect — use nohup; see §5)
cd /workspace/rlrmp && nohup uv sync > /workspace/uv_sync.log 2>&1 &
# After uv sync completes:
nohup uv pip install -U "jax[cuda12]" > /workspace/jax_install.log 2>&1 && touch /workspace/install_done &
# Poll: tail /workspace/jax_install.log; wait for /workspace/install_done (~3–5 min)
```

### 5. nohup pattern (mandatory for installs)
SSH session killed mid-install → SIGHUP kills the process → wasted bandwidth and a broken env.
Always run long setup commands as `nohup <cmd> > <logfile> 2>&1 &` and touch a sentinel file on completion. Poll the sentinel rather than the process.

### 6. Smoke test
```bash
cd /workspace/rlrmp
uv run python scripts/train_minimax.py \
  --adversary-type linear_dynamics \
  --n-warmup-batches 3 --n-adversary-batches 20 --adv-batch-size 250 \
  --n-replicates 5 --hidden-type gru \
  --output-dir /workspace/smoke_test --checkpoint --fused
```
Adjust flags to match the current script's CLI if it has changed.

### 7. Monitoring cadence
- **Smoke test**: check 1 min after start, then every 5 min.
- **Full run**: check at 1 min (confirm JIT compilation visible), every 5 min during early loss decline, drop to every 30 min once loss is steadily descending.
- Watch for `ptxas` warnings, `OOM`, and `Traceback` patterns alongside loss-progress signal.

### 8. Cost discipline
(Cross-ref dotfiles `3602840`.)
- Pod billing starts on **creation**, not on container start.
- Verify `uptimeSeconds > 0` within 2 min of creation; terminate and recreate if stuck.
- Do not unilaterally upgrade cloud tier or GPU class — ask the user first.

### 9. Post-training-run protocol

After every remote training run completes, do all five steps before closing out the session:

1. **Commit run-specs on a feature branch**: Move `run.json` spec files from `_artifacts/<exp>/<label>/` to `results/<exp>/runs/<group>__<variant>/run.json` and commit via `agent-commit --issue <tracking-issue>`.
2. **Submit auth request**: `mandible auth request feature/<name> --issue <tracking-issue> --no-watch`.
3. **Comment on tracking issue**: Key metrics table + winning condition + key findings.
4. **Comment on `c99ad9d`** (training-methods coord) if the run reflects a training-method decision (new method, new loss term, new adversary class).
5. **Comment on `4d38c15`** (analyses coord) if new analyses or tier shifts are motivated.

(Relates to `efc4d68`. Codified after the 2026-05-08 baseline matrix session, where step 1 was deferred until a separate follow-up task.)

## Feedbax Studio
Feedbax Studio (web app) runs from the Feedbax repo. See Feedbax CLAUDE.md for server startup instructions.

## Experiment Artifacts: Tracked vs Ignored

The repo separates artifacts by ROLE, not by directory name:

- **`results/`** is tracked. It holds *specs* (recipes) and *narratives* (prose).
- **`_artifacts/`** is gitignored. It mirrors `results/` and holds *bulk* outputs.
- **Cloud-provider directory names (`runpod/`, `modal/`, etc.) are NOT meaningful** — they all go under `_artifacts/`. Never patch `.gitignore` with a new provider name.

### Flat-by-hash layout (Bug: `f485c26`)

Each top-level entry under `results/` and `_artifacts/` is a **directory named by its 7-character tracking-issue prefix** (e.g. `results/2bc95fd/`, `_artifacts/efc4d68/`). The issue is the atomic unit; no phase-level parent dirs in the directory tree (phase membership lives on the issue body / `b33e8da` coord).

Inside each `<hash>/`:

```
<hash>/
├── README.md                      one paragraph orienting context
├── RUN_PLAN.md                    if the work involves training runs
├── notes/<topic>.md               narrative + analysis writeups
├── figures/<topic>/spec.json      figure specs (one dir per topic)
└── runs/<variant>.json            per-run hyperparameters (flat, not nested)
```

The mirror `_artifacts/<hash>/...` follows the same structure (`runs/<variant>/` is a dir here, holding the bulk training outputs; `runs/<variant>.json` in `results/` is the corresponding spec file).

### Run-folder convention

- **`runs/<variant>.json`** flat by default — one JSON file per run, not a directory with a single file in it.
- For complex sweeps where variant names balloon (~50+ chars), use `runs/<hash>.json` + `runs_index.json` mapping hash → human label + params.
- Promote a run from file to directory (`runs/<variant>/` with `run.json` inside) only when additional per-run files accrue (notes, debug artifacts). Lazy promotion.

### If you produce X, put it at Y

| You produce | Path |
|---|---|
| Model checkpoint, `.eqx`, training log, large `.npz` | `_artifacts/<hash>/runs/<variant>/` |
| Hyperparameters that produced a run | `results/<hash>/runs/<variant>.json` (flat default) |
| Per-run commentary (optional) | `results/<hash>/notes/<variant>.md` |
| Long-form analysis or post-mortem | `results/<hash>/notes/<topic>.md` |
| Figure spec (always) | `results/<hash>/figures/<topic>/spec.json` |
| Figure render (HTML) — written automatically | `_artifacts/<hash>/figures/<topic>/figure.html` |
| Symlink from spec dir to render — automatic | `results/<hash>/figures/<topic>/figure.html` -> `_artifacts/...` |
| Final-cut paper figure | `manuscript/figures/<fig>/` (same rules) |

Run identifier convention: `<group>__<variant>` (double underscore separator, matching the branch-naming convention). Examples: `baseline__standard_12k`, `minimax_single__seed_0`.

### Run-spec vs figure-spec

A `run.json` captures hyperparameters that produced model weights — stable, one per run.
A `spec.json` captures plotting parameters and the data-transform pipeline — volatile, many per run, **references** input runs by path. Never inline run hyperparameters into a figure spec.

### Figure saving: use `feedbax.plot.save_figure`

```python
from feedbax.plot import save_figure

save_figure(
    fig=fig, spec=spec,
    package="rlrmp",
    experiment="<7-char-hash>",   # e.g. "2bc95fd"
    topic="<figure-topic>",        # e.g. "training_loss"
    extra_packages=["rlrmp"],
)
```

This reads rlrmp's registered `figure_routing` config (`src/rlrmp/__init__.py`) and writes:

- `results/<experiment>/figures/<topic>/spec.json` (tracked spec)
- `_artifacts/<experiment>/figures/<topic>/figure.html` (gitignored heavy render)
- A relative symlink `results/<experiment>/figures/<topic>/figure.html` → the render (for one-tree local navigation)

Bug: `f485c26`, feedbax `67bf476`. The dual-tree write + symlink is automatic per the routing config; do not hand-write per-figure dual paths.

**When to use `save_figure_with_spec` instead.** Use the lower-level `feedbax.plot.io.save_figure_with_spec(fig, spec, dst_dir)` only when the destination is a dynamic per-run dir (e.g. `eval_diagnostics.py` writing to `<results_dir>/adversary_force_profiles/`), not a stable experiment topic.

### Adding a new experiment

1. File or pick a tracking issue. Use its 7-char prefix as `<hash>`.
2. Create `results/<hash>/README.md` with one paragraph of context.
3. Run scripts write `results/<hash>/runs/<variant>.json` (spec, flat) and all heavy outputs under `_artifacts/<hash>/runs/<variant>/` (mirror).
4. Figure scripts use `save_figure(package="rlrmp", experiment="<hash>", topic=...)`.
5. Never write `.eqx`, large `.npz`, full-DPI images, or training logs anywhere under `results/`.

### Reconstructed runs (orphan archaeology)

When a run is committed without going through the post-training-run protocol (CLAUDE.md §9) and only the bulk `_artifacts/<orphan>/config.json` survives, reconstruct the `run.json` spec with a `reconstructed: true` marker:

```json
{
  "reconstructed": true,
  "reconstruction_sources": ["_artifacts/<orphan>/config.json", "..."],
  "reconstruction_confidence": "high|medium|low",
  "reconstruction_notes": "...",
  ...
}
```

The README for a reconstructed-run hash dir should begin with a "Reconstructed" preamble that lists the sources and confidence.

### What NOT to gitignore

Do not add directory-name patterns (`runpod/`, `modal/`, `coreweave/`, `tpu/`, `gpu_box/`) to `.gitignore`. The role-based whitelist already excludes these by construction. If you find yourself wanting to add a name-based ignore, the artifact is in the wrong tree — move it under `_artifacts/`.

### Legacy paths

Pre-migration directories under `results/` (e.g. the legacy `results/2ef67ca/models/<name>/config.json` block and the four top-level `centerout_*_pert1/` dirs prior to the f485c26 reorg) keep their existing internal structure (legacy `config.json`, no `run.json`). They are housed under the legacy-archive hash dir `2ef67ca`. Future agents touching these dirs may opportunistically migrate them.

Out-of-scope for f485c26 (tracked separately on `e75ddd7`): the `1_general.assets/` and `2_general.assets/` PNG dumps at the top of `results/`, and the `1_general.md` / `2_general.md` / `2_training-methods.md` notebook narratives that embed them. These stay at `results/` top level until the asset-strip lands.

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

Past phases for orientation (see `b33e8da` comment thread for the live inventory): Part 1 (`297260c`), Part 2 (`0af472c`), Part 2.5 (`844ef95`), Methodology-fix (`b557d4e`, currently active).

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


