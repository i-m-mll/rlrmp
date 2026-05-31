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

## Standard Certificate Presentation

When presenting Phase 3 bridge standard-certificate results, show a table rather
than only a prose summary. Include row identity, status, training distribution,
evaluation lens, objective/cost ratio, state-weighted action mismatch, closed-loop
transition mismatch, value gap, Bellman-Hessian residual, exact-L2/gamma
sidecars, and raw gain mismatch as a diagnostic sidecar.

Use superscript annotations below the table when state-weighted action mismatch
and Bellman-Hessian residual are both shown. The standard annotation is:
`<sup>1</sup> State-weighted action mismatch and Bellman-Hessian residual can
match exactly when the Bellman action Hessian is a scalar multiple of the action
cost geometry on that row. In that case they are the same evidence expressed
through two certificate views; they diverge when downstream value geometry
weights action directions differently.`

Also annotate raw gain mismatch as diagnostic-only:
`<sup>2</sup> Gain mismatch is a diagnostic sidecar, not the bridge gate. The
gate is disturbance-relevant same-game behavior under the standard certificate
components.`

When a standard-certificate row fails, present the failure decomposition as the
standard companion diagnostic rather than as a replacement certificate. The
table should include row identity, classification, learned/reference objective,
learned/reference gradient or projected-gradient where defined, learned-to-
reference interpolation, and visited/weakly visited gain-error decomposition
from the same state distribution used by the row. Keep annotations explicit that
failure decomposition explains the failure but does not change the bridge gate.

Do not leave standard components partial when they can be recomputed. If a
compact manifest lacks fitted gains, trajectories, sampled states, or
covariances, rerun the relevant deterministic or stochastic analysis to recover
the full certificate inputs. Only use `missing` or `not_applicable` when the
quantity is truly impossible or not meaningful for the architecture/evaluation
lens. Evaluation lenses such as nominal-clean, Riccati-epsilon, process-noise,
coverage-induced, and held-out validation are not training axes; keep them
separate from optimal-vs-robust and coverage-vs-no-coverage training factors.

Certificate mode is part of the row contract. Use `static_gain` only when the
controller has a time-local gain over the action state used by the row. Use
`augmented_linear` for linear recurrent rows only when the manifest supplies the
augmented state, action sensitivity, and closed-loop transition over that state
(for example `z_t = [x_t; h_t]`); then report action, transition, value, and
Bellman components in the augmented-state basis. If those augmented inputs are
absent, keep invalid static-gain transition/value/Bellman components explicit
as `not_applicable` rather than silently falling back to plant-state gains. Use
empirical/nonlinear reporting for GRU or other nonlinear recurrent rows unless a
separate local-linear certificate is deliberately defined.

When exact-L2/gamma sidecars improve but action/value/transition/reference-
equivalence metrics still fail, label the failure
`sidecar_improving_non_equivalent`. This label documents a useful sidecar trend;
it is not a bridge pass and must be presented alongside the standard certificate
components. Report aggregate action-energy mismatch (`R_u`) alongside the mean
timewise action mismatch ratio for recurrent and augmented-state rows.

## Plotting Conventions

### Profile-comparison subplots share y-axes (Bug: 06f7faf)

Multi-panel profile-comparison figures — one subplot per condition (cell, regime, perturbation, etc.) where each panel plots the same kind of quantity on the same x-axis semantics — MUST share y-axes across panels. Without shared y-axes each panel auto-scales independently, hiding the cross-condition magnitude differences the panel layout is meant to expose.

Affected figure families include `forward_velocity_profiles`, `hold_drift_profiles`, and any per-replicate variants of these.

**This is enforced at the plot-helper level, not per-script.** Analysis scripts must build profile-comparison grids via `rlrmp.viz.profile_comparison_grid` (the default is `shared_yaxes='all'`). Do not call `plotly.subplots.make_subplots` directly for profile-comparison figures, and do not pass `shared_yaxes` as a per-call override unless there is a documented reason to deviate (in which case file a follow-up issue capturing why).

```python
from rlrmp.viz import profile_comparison_grid

fig = profile_comparison_grid(
    n_panels=n_cells,
    subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
    vertical_spacing=0.025,
)
```

### Aligned-profile aggregators trim by default

The `pooled_trial_mean_with_band` and `replicate_mean_curves` helpers in `rlrmp.analysis.trial_alignment` trim aligned profiles to the strict full-support column window (`min_coverage=1.0`) before reducing. Callers receive the trim slice alongside the curves so companion time axes can be sliced consistently (`t = ((np.arange(n) - center) * dt)[sl]`). Pass `trim=False` only when downstream code needs identical step axes across multiple invocations (e.g. cross-cell pairwise RMSE) and the reducer is already NaN-tolerant.

### Auto-generated note sections (Bug: 06f7faf)

Analysis scripts that write Markdown narrative files under `results/<exp>/notes/` MUST use `rlrmp.io.update_marked_section` instead of overwriting the whole file. This preserves hand-edited preambles (e.g. a "Corrected after go-cue alignment fix" note) across re-runs.

Auto-generated content is wrapped in named HTML comment markers:

```markdown
<!-- AUTO-GENERATED: <marker_name> -->
... script-written content ...
<!-- /AUTO-GENERATED -->
```

On re-run, `update_marked_section` replaces only the content between the markers; everything outside is untouched. If the file does not exist, it is created. If the markers are absent in an existing file, the block is appended.

```python
from rlrmp.io import update_marked_section

update_marked_section(notes_path, "variance_analysis", "\n".join(lines) + "\n")
```

**`marker_name`** should be a short, stable, underscore-delimited identifier matching the logical content of the block (e.g. `"variance_analysis"`, `"results_table"`, `"delta_v_summary"`). New analysis scripts must follow this convention; do not open notes files with `open(..., "w")`.

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
- **RTX 4090 secure cloud**: acceptable when the user requests secure cloud; availability can be low, but do not infer failure from `uptimeSeconds: 0` (see §4b).
- **RTX 5090** (Blackwell, EUR-IS-2 or similar): faster, but some templates carry stale image references.
  - Template `runpod-torch-v280` / image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` has been observed to boot and expose SSH on a 5090 (driver 570.124.06). After `uv sync`, install `jax[cuda12]`; this upgrades to CUDA 12.9 wheels (`jax==0.10.0`) and exposes `CudaDevice(id=0)`.
  - Image `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204` or newer (`cu1290`/`cu1300`) should also support Blackwell — verify the Docker tag exists before creating the pod.
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
  --volume-in-gb 30 \
  --ports "22/tcp,8080/http"   # REQUIRED: default exposes no ports → direct TCP SSH unreachable
# (For 4090: use --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY, omit --data-center-ids)

# 4b. Poll until SSH ready (~1–3 min). Bug: b399efc.
runpodctl pod get <POD_ID>
# Correct readiness criterion:
#   1. `runpodctl pod get <POD_ID>` output contains an `.ssh` object with `ip`,
#      `port`, and `ssh_command`.
#   2. That SSH command succeeds with a functional probe:
#        ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <port> root@<ip> \
#            'nvidia-smi --query-gpu=name --format=csv,noheader'
# Do NOT use `.runtime`, `.runtime.ports`, or `uptimeSeconds` as the primary
# readiness signal. Observed counterexample: a healthy RTX 5090 pod had
# `uptimeSeconds: 0` and no `.runtime` object, but `.ssh.ssh_command` was valid
# and SSH worked. Following the old `runtime.ports`-populated heuristic
# terminated ~5 healthy pods. Keep polling within the normal boot window;
# stop/recreate only if `.ssh` never populates or the functional probe fails.

# 4c. rsync 3 path-deps to /workspace (run from local machine). Bug: b399efc.
# Notes:
# - Use inline excludes; do not stash flags in a shell variable (word-splitting
#   in zsh has caused oversized transfers).
# - --no-owner --no-group: RunPod volumes reject chown, so without these flags
#   rsync exits 23 even when data transferred successfully.
# - macOS ships old Apple-patched rsync that rejects --info=stats2,progress2.
#   Use --stats for a portable transfer summary.
# - Exclude tracked legacy *.assets image dumps (~100 MB) — not needed for
#   training.
rsync -az --stats --no-owner --no-group \
  --exclude='_artifacts' --exclude='worktrees' --exclude='.venv' --exclude='.git' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.assets' \
  --exclude='results/*.assets' --exclude='manuscript/results.assets' \
  --exclude='TODO.assets' \
  "/Users/mll/Main/10 Projects/10 PhD/rlrmp/" root@<pod-ip>:/workspace/rlrmp/

rsync -az --stats --no-owner --no-group \
  --exclude='_artifacts' --exclude='worktrees' --exclude='.venv' --exclude='.git' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='web' \
  "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/develop/" root@<pod-ip>:/workspace/feedbax/

rsync -az --stats --no-owner --no-group \
  --exclude='worktrees' --exclude='.venv' --exclude='.git' \
  --exclude='__pycache__' --exclude='.pytest_cache' \
  "/Users/mll/Main/10 Projects/05 Utils/jax-cookbook/" root@<pod-ip>:/workspace/jax-cookbook/
# jax-cookbook including worktrees/ is ~480 MB; the exclude keeps it <1 MB.

# 4d. Patch embedded local paths on the pod (SSH in, then run). Bug: b399efc.
# Use perl with \Q...\E (literal-string quoting) rather than broad sed globs.
# The old `sed -i 's|.*/feedbax[^"]*|...|g'` form matched too broadly and
# corrupted unrelated TOML fields and lockfile metadata.
perl -0pi -e \
  's|\Q/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/develop\E|/workspace/feedbax|g;
   s|\Q../../../20 Feedbax/feedbax/worktrees/develop\E|/workspace/feedbax|g' \
  /workspace/rlrmp/pyproject.toml /workspace/rlrmp/uv.lock

perl -0pi -e \
  's|\Q/Users/mll/Main/10 Projects/05 Utils/jax-cookbook\E|/workspace/jax-cookbook|g;
   s|\Q../../../../05 Utils/jax-cookbook\E|/workspace/jax-cookbook|g;
   s|\Q../../../../../05 Utils/jax-cookbook\E|/workspace/jax-cookbook|g' \
  /workspace/rlrmp/uv.lock /workspace/feedbax/pyproject.toml /workspace/feedbax/uv.lock

# Verify no local editable paths remain before proceeding:
grep -RIn '/Users\|10 Projects\|\.\./\.\./.*jax-cookbook\|\.\./\.\./.*feedbax' \
  /workspace/rlrmp/pyproject.toml /workspace/rlrmp/uv.lock \
  /workspace/feedbax/pyproject.toml /workspace/feedbax/uv.lock || true

# 4e. Install (must survive SSH disconnect — use nohup; see §5)
cd /workspace/rlrmp && nohup uv sync > /workspace/uv_sync.log 2>&1 &
# After uv sync completes:
nohup uv pip install -U "jax[cuda12]" > /workspace/jax_install.log 2>&1 \
  && touch /workspace/install_done &
# Poll: tail /workspace/jax_install.log; wait for /workspace/install_done (~3–5 min)
# Do NOT run `uv sync` again after the CUDA JAX install — it can revert to the
# lockfile's CPU JAX wheels. Use `uv run --no-sync` for all subsequent commands.
# Verify before training:
#   XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python -c \
#     'import jax; print(jax.__version__); print(jax.devices())'
```

### 5. nohup pattern (mandatory for installs)
SSH session killed mid-install → SIGHUP kills the process → wasted bandwidth and a broken env.
Always run long setup commands as `nohup <cmd> > <logfile> 2>&1 &` and touch a sentinel file on completion. Poll the sentinel rather than the process.

### 6. Smoke test
```bash
cd /workspace/rlrmp
uv run --no-sync python scripts/train_minimax.py \
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
- For 1k warmup-only smoke tests on a 5090, first compilation takes ~30 s and the whole run completes in a few minutes. Pause after the smoke test and do not launch the main matrix until the user confirms.

### 8. Cost discipline
(Cross-ref dotfiles `3602840`.)
- Pod billing starts on **creation**, not on container start.
- Verify the pod is reachable via the `.ssh.ssh_command` from `runpodctl pod get`, confirmed by a functional SSH probe (`ssh ... 'true'` or `ssh ... 'nvidia-smi'`). Do NOT rely on `uptimeSeconds > 0`, `.runtime`, or `.runtime.ports` as the primary liveness signal — a working pod can show `uptimeSeconds: 0` and no `.runtime` object while `.ssh` is valid (Bug: b399efc).
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

### Script placement: experiment-specific vs reusable (Bug: 8404108)

The top-level `scripts/` directory is for cross-cutting tooling — scripts that operate generically across experiments (e.g. `train_minimax.py`, `train_part2_5.py`, `eval_minimax.py`, `eval_diagnostics.py`, infrastructure shell scripts). It is NOT a dumping ground for experiment-specific analysis code.

**Why this matters.** Pre-`8404108`, `scripts/` had accumulated 36 files mixing CLI entry-points, shared library code, experiment-specific drivers, and one-off analyses — with 25 of those scripts pulling each other in via `sys.path.insert(...)`-style injection. That made the directory unsearchable (no role separation), made the code untestable (sibling imports only resolve at execute time), and made every analysis script silently depend on the order in which `scripts/` happened to be on `PYTHONPATH`. The rules below prevent that recurrence.

**Hard rules:**

1. **Capability-named library modules.** Modules under `src/rlrmp/` MUST be named by capability — `eval`, `train`, `plot`/`viz`, `analysis`, `lme`, etc. They MUST NOT be named by experiment, phase, or paper (no `part2_5`, no `methodology_fix`, no `shahbazi`, no `tier1`). When you find yourself wanting to call a new module `<phase>_helpers.py` or `<paper>_metrics.py`, that's a signal to identify the underlying capability and use that name instead. Within a capability module, training-method-specific sub-modules ARE allowed (e.g. `rlrmp.train.minimax`, `rlrmp.eval.minimax_io`) because training methods are stable concepts that span experiments. Sub-modules named by experiment are still forbidden.

2. **Experiment-specific scripts** (analysis pipelines, plotting code, one-off diagnostics tied to a single tracking issue) live with the experiment: `results/<hash>/scripts/<name>.py`. Commit them alongside the experiment's `runs/`, `notes/`, and `figures/` content under the same `Bug: <hash>` trailer.

3. **Reusable components** (utility functions, plotting primitives, analysis routines that several experiments will call) MUST be refactored into the capability-named library module BEFORE the experiment script lands. If you're tempted to put a helper inside an experiment-specific script "for now," extract it now — it will outlast the script. Submit the library change via an auth request to `src/rlrmp/` (or `feedbax/` if the abstraction is plant- or task-general).

4. **Mixed scripts** (experiment-specific driver that uses generic helpers) split: the driver under `results/<hash>/scripts/`, the helpers in `src/rlrmp/`. Both can land in the same auth request — the driver carries the `Bug: <hash>` trailer; the library change carries its own feature issue if it's substantial.

5. **Cross-cutting CLI entry-points** (training/eval launchers that operate generically across experiments) stay in `scripts/`. Examples: `scripts/train_minimax.py`, `scripts/train_part2_5.py`, `scripts/eval_minimax.py`, `scripts/eval_diagnostics.py`. These scripts MUST import their reusable helpers from `src/rlrmp/` (capability-named modules), not from each other.

6. **No `sys.path.insert(...)` anywhere.** Use absolute imports (`from rlrmp.eval import ...`, `from rlrmp.train.minimax import build_hps`). If you catch yourself reaching for `sys.path.insert`, stop and extract the dependency into `src/rlrmp/` instead. Sibling-script imports between two files under `scripts/` (e.g. `eval_diagnostics.py` pulling from `eval_minimax.py`) are also forbidden — extract the shared piece to `src/rlrmp/`. Within a single `results/<hash>/scripts/` directory, sibling imports DO work natively (Python auto-adds the executing script's directory to `sys.path`) and are fine for tightly-coupled experiment code that doesn't generalise.

**Concrete examples (from the 8404108 refactor):**

| Right placement | Why |
|---|---|
| `src/rlrmp/eval/{ensemble,kinematics,sisu,pert,minimax_io}.py` | Generic eval primitives — used across 14+ scripts. Capability-named modules under `rlrmp.eval`. |
| `src/rlrmp/train/{minimax,standard}.py` | Hyperparameter constructors for two training methods. Module names are training methods, not phases. |
| `results/2bc95fd/scripts/analyse_anti_anticipation_6cell_variance.py` | Experiment-specific analysis tied to issue `2bc95fd`. Lives with its experiment. |
| `scripts/train_minimax.py` | Generic minimax-training CLI. Stays in `scripts/` as a cross-cutting entry-point. Imports `build_hps` from `rlrmp.train.minimax`. |

| Wrong placement | Why it's wrong |
|---|---|
| `src/rlrmp/part2_5_eval.py` | Module name encodes a phase. Use `src/rlrmp/eval/` instead. |
| `src/rlrmp/methodology_fix_helpers.py` | Module name encodes a phase. Use the underlying capability name. |
| `scripts/analyse_pregomatrix.py` | Experiment-specific analysis. Belongs under `results/3702f54/scripts/`. |
| `scripts/eval_part2_5_figures.py` exporting `eval_ensemble_on_trials` | Sibling-script import of a generic primitive. Extract to `src/rlrmp/eval/`. |

**Promotion case (script-specific → reusable).** When a function or pattern starts being reused by multiple experiments — e.g. a kinematics helper written for experiment A is being copy-pasted into experiment B's script — promote it to the relevant capability-named library module in `src/rlrmp/` in a dedicated feature issue. Update both call sites in the same auth request. Don't let the same helper drift in two `results/<hash>/scripts/` directories.

**Re-use audit cadence.** When opening a new experiment's analysis script that imports anything from `src/rlrmp/eval/` or `src/rlrmp/train/`, take a moment to scan the imports for "things that should be in the library but aren't yet" — usually surface as `from rlrmp.eval import ...` followed by a long block of private helper functions in the script. Those private helpers are the next promotion candidates.

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

### Worktree symlink convention for `_artifacts/` (Bug: `0887e3e`)

rlrmp's `.worktree.yaml` adds `_artifacts` to its `shared:` list, so `~/.dotfiles/bin/wt` symlinks the directory from the main worktree's repo root into every feature worktree at creation time. All writes to `_artifacts/` from a worktree therefore go to the main repo's `_artifacts/`, regardless of which worktree the script runs from. This prevents gitignored bulk outputs (figure HTML renders, training checkpoints, etc.) from being silently deleted when `dwt` removes the worktree post-merge — the original failure mode that motivated this convention (see issue `0887e3e` for the design discussion).

- **New worktrees** inherit the symlink automatically via `wt` (it processes `.worktree.yaml` at creation time).
- **Pre-existing worktrees** can pick up the symlink by running `~/.dotfiles/bin/wt-sync` from inside the worktree.
- **`.worktree.yaml` also declares `setup: mkdir -p _artifacts`** so the directory exists on fresh clones (otherwise `wt` would warn "not found in repo root, skipping" and no symlink would be created).

Constraint: nothing under `_artifacts/` may be tracked in git. A tracked file would be materialized by `git checkout` when the worktree is created, then conflict with the symlink replacement that `wt` performs. The `.gitignore` pattern is therefore `_artifacts` (no trailing slash, no re-include exceptions) — see Bug: `0887e3e` for why the prior `!_artifacts/README.md` exception had to be removed.

Caveat: parallel worktrees share one `_artifacts/`. Concurrent writes to the same `_artifacts/<hash>/...` subtree from two worktrees could collide; in practice, hash-keyed experiment directories make collisions unlikely.

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

For Mandible issue command syntax, see the global `~/.claude/CLAUDE.md` issue-tracking convention. This section covers only project-specific coordination protocol.

### The four coordination issues

Each coordination issue has the `coordination` label and is project-lifetime (no closure-on-merge intent, no phase scope).

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

- **`umbrella`** — phase-tied or work-bundle-tied. **May close deliberately** when the bundle is done, via an auth request `--closes-issue` field, explicit `Closes:` / `Resolves:` trailer, or user action. Example: `b557d4e` (methodology-fix phase umbrella) closed when its synthesis-review work merged. The phase work continues on its children; the umbrella's job was just to mark the bundle.
- **`coordination`** — project-spanning decision-tracking surface. **Should not close on merge** and should not be referenced as the completed work unit in `Bug:` trailers. These persist for the project's lifetime.

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

Even though `Bug:` trailers are reference links rather than closure signals, **the convention remains: do not reference coordination issues in commit `Bug:` trailers.** Trailers are for the relevant child / feature / bug issue — the unit of work the commit completes. Coordination issues are decision-tracking surfaces, not commit destinations.

This means `agent-commit --issue <id>` should always take a child / feature / bug issue ID, never `4d38c15` / `c99ad9d` / `b33e8da` / `1d9ae6f`.

### Phase umbrella protocol

When starting a new phase or work-bundle:

1. **Create a phase umbrella** — a new issue labeled `umbrella` (and `feature` if appropriate). Body: minimal — motivating question, scope, links to phase artifacts (e.g. a `results/<exp>/README.md`).
2. **Comment on `b33e8da`** with the phase umbrella ID + one-line motivating question. This is what makes `b33e8da` a discovery surface for "what umbrellas are active right now."
3. **Children of the phase umbrella** reference the umbrella in **their bodies** (e.g. "Part of phase `b557d4e`."), not in their commit `Bug:` trailers. Their trailers reference themselves or their sub-features. Close the umbrella only deliberately when the phase is done; the comment thread on `b33e8da` carries the live phase state regardless.
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
- **Don't reference `coordination`-labeled issues in commit `Bug:` trailers.** Use the child / feature / bug issue. `Bug:` is a reference link, and the convention keeps coordination issues out of ordinary commit destinations.
- **Don't create a new coordination issue when an existing one's scope covers the concern.** Comment on the existing one. New coordination issues are project-lifetime commitments — adding one is a structural change that should be discussed first.
- **Don't paste subagent output or raw analysis into a coord body.** Move it to the child issue (or to a `results/<exp>/` doc) and replace with a one-line cross-ref.
- **Don't index every commit on the coord.** Only commits that change cross-cutting state (a tier, a method choice, a phase boundary) merit a coord comment. Ordinary work-on-a-child does not.
