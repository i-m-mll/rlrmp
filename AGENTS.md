# rlrmp

<!-- CLAUDE.md is a symlink to this file; edit here. -->

## Python/JAX Coding Conventions

### Coding Style & Naming
- Follow PEP 8: 4-space indentation, 100-char soft line limit, type hints required for public APIs.
- Always place imports at the top of files, except in the rare case that they should be in a conditional for performance or typing reasons.
- Naming: modules/packages `lower_snake_case`; functions/variables `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Docstrings: Google style; include shapes/dtypes for JAX arrays when relevant.

### Environment Management
- Use `uv` for all package management. Do not run `pip install` directly.
- **Never invoke `python`, `python3`, `pytest`, or `pip` bare.** Always go through `uv run` (or `uv run --no-sync` after a CUDA-JAX install on a pod — see runbook §4e). Lint/format with `uv run ruff check` / `uv run ruff format` (`ruff` is a dev dependency).

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
- The feedbax repo is at `~/Main/10 Projects/10 PhD/20 Feedbax/feedbax/` (rlrmp's `pyproject.toml` points its editable source at this repo root). The Python package lives at `feedbax/feedbax/` inside the repo — there is **no `src/` layout**. Import via `uv run --no-sync python`, never bare `python`.
- Feedbax's protected branch is `develop`, not `main`. All canonical feedbax behaviour, APIs, and architectural patterns reside on `develop`, which the repo root now carries; the `main` branch may lag substantially and is not authoritative. Use worktrees for feature work (`wt feature/<name> develop`), following this repo's conventions. To read current behaviour, read files at the repo root or use `git show develop:path/to/file.py`.

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

### Current training-method orientation (May 2026)

- Recent calibration work (`f47abb1`, then `3702f54`) shifted the project from
  ratio-based loss interpretation toward absolute within-cell behaviour: pre-go
  drift, forward-velocity RMSE, peak velocity, time-to-peak, and endpoint
  quality matter more than cross-cell RMSE ratios.
- The best warmup-only cell from the pre-go motor-mask matrix was
  `full_trial_pl__prego_1`: full-trial power-law position schedule, no output
  jerk, `nn_output_pre_go=1.0`, position weight 1x, `n_adversary_batches=0`.
  It suppressed pre-go RMS drift to about `0.02 mm`, with velocity RMSE about
  `0.0176 m/s`, peak velocity about `1.087 m/s`, and time-to-peak about
  `36.8` steps. These were warmup-only results and still need adversarial
  revalidation before being treated as final.
- Increasing position weight to 10x made training worse despite reaching more
  aggressively: higher replicate variance and poorer overall behaviour. Do not
  assume "more position weight" is a clean fix for lazy reaches.
- Saturating `nn_output_pre_go` may not be monotonic. A 1k smoke test with
  `nn_output_pre_go=100` stayed finite but plateaued around validation loss
  `6.3`, worse than the prior `prego=1` early trajectory. Treat very large
  pre-go penalties as potentially destabilizing until smoke-tested.
- For movement-ramp experiments, target-on/pre-movement position and velocity
  costs must be explicitly zero (`effector_hold_pos=0`,
  `effector_hold_vel=0`). The only nonzero position-error term should be
  `effector_pos_running` with a movement-epoch-locked ramp. This ramp should
  start at the actual movement epoch, have fixed duration across go-cue timing
  conditions, and remain at max weight after the ramp completes.

### Spec lock before launch

Before any **billable** training launch (`runpodctl pod create` for a run, or a non-smoke `modal run`), present a one-table run spec — task structure, loss terms, dims, `n_batches`, `lr`, seeds, GPU/cloud — and obtain **explicit user confirmation**. Mid-session plan discussion is NOT launch authorization.

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

Prefer the Feedbax deploy automation for RunPod setup instead of reproducing the
manual sequence here:

- `~/Main/10 Projects/10 PhD/20 Feedbax/feedbax/scripts/deploy/runpod_deploy.sh`
- `~/Main/10 Projects/10 PhD/20 Feedbax/feedbax/scripts/deploy/poll_run.sh`

Those scripts own the normal pod-create, SSH readiness polling, path-dependency
sync, local-path patching, and install flow for the rlrmp/feedbax/jax-cookbook
stack. Use their `--help` / dry-run style surfaces first, and improve the
Feedbax scripts when the deployment procedure changes rather than copying a new
manual recipe into this file.

Manual fallback/debugging invariants:

- Expose `22/tcp` at pod creation; the default RunPod port set can make direct
  TCP SSH unreachable.
- Poll readiness from the `.ssh` object and a functional SSH probe such as
  `nvidia-smi`, not from `.runtime`, `.runtime.ports`, or `uptimeSeconds`
  (Bug: b399efc).
- Use `rsync -az --stats --no-owner --no-group` with inline excludes for
  rlrmp, feedbax, and jax-cookbook. Do not stash rsync flags in a shell
  variable.
- Patch embedded local editable paths with literal-string replacements; avoid
  broad `sed` globs that can corrupt TOML or lockfile metadata.
- After installing CUDA JAX on the pod, do not run `uv sync` again. Use
  `uv run --no-sync` for subsequent commands.

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
- **Modal:** destructive CLI ops (`modal app stop`, `modal volume rm`) require `--yes`/`-y`; non-interactive shells abort without it.

### 9. Post-training-run protocol

After every remote training run completes, use `scripts/post_run.sh` wherever
possible as the deterministic handoff step. It owns the mechanical protocol:
artifact sync from local, Modal, or pod sources; tracked run-spec creation under
`results/<issue>/runs/<run>.json`; bulk artifact placement under
`_artifacts/<issue>/runs/<run>/`; metrics-table rendering from
`training_summary.json`; `git add`; `agent-commit`; and Mandible auth request
submission.

Run it from the feature worktree that should own the post-run commit:

```bash
scripts/post_run.sh --issue <tracking-issue> --run <group>__<variant> \
  --artifacts-src <local:/path|/path|modal[:volume]|pod:user@host:/path> \
  --dry-run

scripts/post_run.sh --issue <tracking-issue> --run <group>__<variant> \
  --artifacts-src <local:/path|/path|modal[:volume]|pod:user@host:/path>
```

Use the dry run first when source paths, volume names, or branch state are not
obvious. If the script cannot cover the source shape, preserve its layout and
auth/commit conventions when doing the fallback manually, and report the script
gap on the tracking issue or a workflow issue.

The residual agent-owned judgment after `scripts/post_run.sh` is:

1. **Interpret the run**: Use the emitted issue-comment template, fill in the
   outcome, key metric movement, winning condition, and residual blockers, then
   comment on the tracking issue.
2. **Decide coordination updates**: Comment on `c99ad9d` (training-methods
   coord) only when the run reflects a training-method decision, such as a new
   method, loss term, or adversary class.
3. **Decide analysis updates**: Comment on `4d38c15` (analyses coord) only
   when new analyses, analysis-tier shifts, or analysis deprecations are
   motivated.

(Relates to `efc4d68`. Codified after the 2026-05-08 baseline matrix session, where step 1 was deferred until a separate follow-up task.)

Stable post-run provenance contract:

- `scripts/post_run.sh` pins Feedbax's manifest root for this repo to
  `_artifacts/feedbax_runs/` by exporting/checking `FEEDBAX_RUNS_DIR`. If the
  environment points anywhere else, the wrapper must fail before committing.
- Dry runs must print the post-run provenance stamp preview: rlrmp SHA, Feedbax
  SHA, Feedbax manifest/provider schema versions, the post-run provenance
  schema version, the pinned manifest root, and GraphSpec hash/version when the
  graph sidecar is present.
- Non-dry runs stamp the tracked run spec with `post_run_provenance`, then run
  Feedbax `TrainingRunManifest` parity before `git add` / `agent-commit`. A
  mismatched tracked run spec and matching manifest blocks the commit.
- Existing legacy runs without a matching Feedbax `TrainingRunManifest` may
  continue through the wrapper, but the output must report that parity was
  `not_found` rather than silently implying a checked manifest.

## Feedbax Studio
Feedbax Studio (web app) runs from the Feedbax repo. See feedbax repo instructions for server startup.

## Review packets & session handoffs

- **Review packets**: external review packets are built with the `make-review-packet` skill, live under `/tmp/rlrmp_<hash>_<topic>/`, are standalone, and are **NEVER** committed to the repo.
- **Session handoffs**: fresh-session handoff notes live at `results/<hash>/notes/` (a stable path). Never reference a handoff by a `worktrees/...` path — worktrees are deleted after merge.

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

### Script placement: experiment-specific vs reusable (Bug: 8404108)

The top-level `scripts/` directory is for cross-cutting tooling — scripts that operate generically across experiments (e.g. `train_minimax.py`, `train_part2_5.py`, `eval_minimax.py`, `eval_diagnostics.py`, infrastructure shell scripts). It is NOT a dumping ground for experiment-specific analysis code.

**Hard rules:**

1. **Capability-named library modules.** Modules under `src/rlrmp/` MUST be named by capability — `eval`, `train`, `plot`/`viz`, `analysis`, `lme`, etc. — never by experiment, phase, or paper (no `part2_5`, `methodology_fix`, `shahbazi`, `tier1`). If you want to call a module `<phase>_helpers.py`, identify the underlying capability and use that name instead. Within a capability module, training-method-specific sub-modules ARE allowed (`rlrmp.train.minimax`, `rlrmp.eval.minimax_io`) because training methods are stable concepts spanning experiments. Experiment-named sub-modules are still forbidden.
2. **Experiment-specific scripts** (analysis pipelines, plotting, one-off diagnostics tied to a single tracking issue) live with the experiment: `results/<hash>/scripts/<name>.py`. Commit them alongside the experiment's `runs/`, `notes/`, `figures/` with `agent-commit --issue <hash>` so the commit carries the matching `Mandible-Issue: <hash>` trailer.
3. **Reusable components** (utilities, plotting primitives, analysis routines several experiments call) MUST be refactored into the capability-named library module BEFORE the experiment script lands. Extract now, not "for now" — the helper will outlast the script. Submit the library change via an auth request to `src/rlrmp/` (or `feedbax/` if plant- or task-general).
4. **Mixed scripts** split: the driver under `results/<hash>/scripts/`, the helpers in `src/rlrmp/`. Both can land in the same auth request - the driver commit carries the experiment's `Mandible-Issue: <hash>` trailer; a substantial library change carries its own feature issue.
5. **Cross-cutting CLI entry-points** (training/eval launchers operating generically across experiments) stay in `scripts/` (`scripts/train_minimax.py`, `scripts/eval_minimax.py`, etc.). They MUST import reusable helpers from `src/rlrmp/`, not from each other.
6. **No `sys.path.insert(...)` anywhere.** Use absolute imports (`from rlrmp.eval import ...`, `from rlrmp.train.minimax import build_hps`). Sibling-script imports between two files under `scripts/` are forbidden — extract the shared piece to `src/rlrmp/`. Within a single `results/<hash>/scripts/` directory, sibling imports DO work natively and are fine for tightly-coupled experiment code that doesn't generalise.

**Concrete examples (from the 8404108 refactor):**

| Right placement | Why |
|---|---|
| `src/rlrmp/eval/{ensemble,kinematics,sisu,pert,minimax_io}.py` | Generic eval primitives used across 14+ scripts. Capability-named under `rlrmp.eval`. |
| `src/rlrmp/train/{minimax,standard}.py` | Hyperparameter constructors for two training methods. Names are methods, not phases. |
| `results/2bc95fd/scripts/analyse_anti_anticipation_6cell_variance.py` | Experiment-specific analysis tied to `2bc95fd`. Lives with its experiment. |
| `scripts/train_minimax.py` | Generic minimax-training CLI. Stays in `scripts/`; imports `build_hps` from `rlrmp.train.minimax`. |

| Wrong placement | Why it's wrong |
|---|---|
| `src/rlrmp/part2_5_eval.py` | Module name encodes a phase. Use `src/rlrmp/eval/`. |
| `src/rlrmp/methodology_fix_helpers.py` | Module name encodes a phase. Use the underlying capability name. |
| `scripts/analyse_pregomatrix.py` | Experiment-specific analysis. Belongs under `results/3702f54/scripts/`. |
| `scripts/eval_part2_5_figures.py` exporting `eval_ensemble_on_trials` | Sibling-script import of a generic primitive. Extract to `src/rlrmp/eval/`. |

**Promotion.** When a helper starts being reused across experiments, promote it to the relevant capability module in `src/rlrmp/` in a dedicated feature issue and update both call sites in the same auth request. When opening a new analysis script that imports from `src/rlrmp/eval/` or `src/rlrmp/train/`, scan for private helpers that should already be in the library — those are the next promotion candidates.

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

### Worktree symlink convention for `_artifacts/` (Bug: `0887e3e`)

rlrmp's `.worktree.yaml` adds `_artifacts` to its `shared:` list, so `~/.dotfiles/bin/wt` symlinks the directory from the main worktree's repo root into every feature worktree at creation time. All writes to `_artifacts/` from a worktree therefore go to the main repo's `_artifacts/`, regardless of which worktree the script runs from. This prevents gitignored bulk outputs (figure HTML renders, training checkpoints, etc.) from being silently deleted when `dwt` removes the worktree post-merge — the original failure mode that motivated this convention (see issue `0887e3e` for the design discussion).

- **New worktrees** inherit the symlink automatically via `wt` (it processes `.worktree.yaml` at creation time).
- **Pre-existing worktrees** can pick up the symlink by running `~/.dotfiles/bin/wt-sync` from inside the worktree.
- **`.worktree.yaml` also declares `setup: mkdir -p _artifacts`** so the directory exists on fresh clones (otherwise `wt` would warn "not found in repo root, skipping" and no symlink would be created).

Constraint: nothing under `_artifacts/` may be tracked in git. A tracked file would be materialized by `git checkout` when the worktree is created, then conflict with the symlink replacement that `wt` performs. The `.gitignore` pattern is therefore `_artifacts` (no trailing slash, no re-include exceptions) — see Bug: `0887e3e` for why the prior `!_artifacts/README.md` exception had to be removed.

Caveat: parallel worktrees share one `_artifacts/`. Concurrent writes to the same `_artifacts/<hash>/...` subtree from two worktrees could collide; in practice, hash-keyed experiment directories make collisions unlikely.

### Reconstructed runs (orphan archaeology)

When a run is committed without going through the post-training-run protocol (§9) and only the bulk `_artifacts/<orphan>/config.json` survives, reconstruct the `run.json` spec with a `reconstructed: true` marker:

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

This project uses a small set of long-lived **coordination issues** (label: `coordination`) as decision-tracking surfaces. They are distinct from `umbrella` issues (which bundle a specific phase of work) and from ordinary `feature` / `error` issues (which carry the substantive work). Future agents must know which coordination issue to comment on when, what to file as a new issue vs. a comment, and how to keep these surfaces from becoming a dumping ground.

For Mandible issue command syntax, see the global `~/.codex/AGENTS.md` issue-tracking convention. This section covers only project-specific coordination protocol.

### The four coordination issues

Each has the `coordination` label and is project-lifetime (no closure-on-merge intent, no phase scope).

| ID | Name | Scope |
|---|---|---|
| `4d38c15` | Project analyses coordination | Cross-cutting decisions about *analyses applied to trained models*. |
| `c99ad9d` | Project training-methods coordination | Cross-cutting decisions about *how models are trained*. |
| `b33e8da` | Project phases coordination | Index of phase umbrellas, phase boundaries, pivots, outcomes. |
| `1d9ae6f` | Project meta coordination | Cross-project / workflow concerns surfaced from rlrmp work (Mandible, feedbax, dotfiles, general workflow). |

#### `4d38c15` — analyses

**Owns:** new analyses worth doing; tier shifts (essential / desirable / auxiliary / deprecated); cross-cutting findings across multiple analyses; deprecation/archival of analyses.
**Does NOT own:** the design/math/implementation/results of any single analysis (those live on the analysis's own issue).
**Triggers:** discovered a new analysis; want to re-rank tiers; one analysis subsumes another; an analysis became less informative after a method change.

#### `c99ad9d` — training-methods

**Owns:** training method menu (standard backprop, CVaR, APT, minimax, LEQG, PAI-ASF, BCS, DAI); adversary classes (parametric force fields, GaussianBumpAdversary, structural ΔA); flavor-of-`max` choices (input-instance / model-class / LEQG-via-Whittle); SISU wirings; plant-regime parameters when they couple to training (damping, motor noise, reach geometry, loss schedule); method deprecations and promotions.
**Does NOT own:** specific analyses (→ `4d38c15`); phase markers (→ `b33e8da`); model-structure decisions independent of training method (may eventually move to a separate model-structure coord).
**Triggers:** introducing a new training method or adversary class; redesigning an existing adversary; flavor-of-`max` decisions; cross-method tier shifts; training-relevant plant-regime changes.

#### `b33e8da` — phases

**Owns:** index of phase umbrellas (current and past) with one-line motivating-question + verdict; phase boundaries and pivots; cross-references to phase artifacts (READMEs, synthesis docs).
**Does NOT own:** the work content of any phase (lives on the phase umbrella) or in-phase analyses (→ `4d38c15`).
**Triggers:** starting a new phase / work-bundle (file an `umbrella`-labeled issue, then comment here); a phase ends, pivots, or is abandoned (follow-up comment with outcome).

#### `1d9ae6f` — meta cross-project

**Owns:** workflow / tooling concerns surfaced during rlrmp work that need a fix in **another** repo — Mandible, feedbax, dotfiles, or general workflow. The body is an index; the actual fixes live in the destination repos.
**Does NOT own:** rlrmp-internal concerns (those go to one of the three coords above or to a normal issue).
**Triggers:** noticing a Mandible bug while in rlrmp; needing a feedbax API change to support rlrmp work; spotting a global AGENTS.md gap; identifying a tooling improvement.

### Umbrella vs coordination — which label?

- **`umbrella`** — phase-tied or work-bundle-tied. **May close deliberately** when the bundle is done, via an auth request `--closes-issue` field, explicit `Closes-Mandible-Issue:` trailer, or user action. Example: `b557d4e` (methodology-fix phase umbrella) closed when its synthesis-review work merged. The phase work continues on its children; the umbrella just marked the bundle.
- **`coordination`** — project-spanning decision-tracking surface. **Should not close on merge** and should not be referenced as the completed work unit in `Mandible-Issue:` trailers. These persist for the project's lifetime.

**Decision rule:** "Should this issue close when the work it tracks merges?" — Yes → `umbrella`. No → `coordination`.

### Body content directive (umbrella-verbosity, `1ba096f`)

> Higher-level coordination/umbrella issue **bodies** must be **minimal** — cross-references to children/related issues, plus only material that does not already live in a finer-grained issue. Long-form discussion belongs in the relevant analysis/feature issue. Comments on the coordination/umbrella are timestamped + threaded; use them for cross-cutting decisions, tier shifts, and similar.

In practice: a coord body reads like a table of contents (scope, what's owned, what isn't, cross-refs). Tier ordering, phase inventory, and similar cross-cutting state belong in **comments** (timestamped, revisable). Substantive findings (results, plots, math) live on the relevant child issue; the coord may carry a one-line cross-ref.

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

When the table doesn't cover your case: ask "is this a project-lifetime decision (→ coord comment) or a unit of work (→ new issue)?". If both, do both.

### Cross-referencing protocol

- Use **7-character issue-ID prefixes** in bodies and comments (e.g. `4d38c15`).
- When filing in a destination repo (per `1d9ae6f`), include a one-line "surfaced from rlrmp" note in the destination issue's body, plus the rlrmp issue ID or branch that surfaced it.
- When the destination-repo issue resolves, comment back on `1d9ae6f` with the resolution (merge SHA / closing comment link).
- Coordination issue bodies index children — they do not duplicate child content.

### Commit `Mandible-Issue:` trailers — never reference coordination issues

`Mandible-Issue:` trailers are for the relevant child / feature / bug issue - the unit of work the commit completes. Coordination issues are decision-tracking surfaces, not commit destinations. So `agent-commit --issue <id>` should always take a child / feature / bug issue ID, never `4d38c15` / `c99ad9d` / `b33e8da` / `1d9ae6f`. Use `Closes-Mandible-Issue: <id>` only when the commit should deliberately close that issue through the auth merge.

### Phase umbrella protocol

1. **Create a phase umbrella** — a new issue labeled `umbrella` (and `feature` if appropriate). Body: minimal — motivating question, scope, links to phase artifacts (e.g. a `results/<exp>/README.md`).
2. **Comment on `b33e8da`** with the umbrella ID + one-line motivating question (this makes `b33e8da` the "what umbrellas are active" discovery surface).
3. **Children** reference the umbrella in **their bodies** ("Part of phase `b557d4e`."), not in commit `Mandible-Issue:` trailers. Close the umbrella only deliberately when the phase is done.
4. **On phase end / pivot / abandonment**, comment on `b33e8da` with the one-line outcome ("merged via X", "pivoted to Y", "abandoned because Z").

Past phases for orientation (see `b33e8da` for the live inventory): Part 1 (`297260c`), Part 2 (`0af472c`), Part 2.5 (`844ef95`), Methodology-fix (`b557d4e`, currently active).

### Worked example: cross-cutting Riccati flavor-(a) finding

`tests/test_hinf_riccati.py::test_cs_faithful_qr_velocity_inflation` xfailed with a substantive diagnosis: faithful C&S Eq. 15 Q,R on the C&S regime gave Δv = −0.04% at 1.5γ\*, implying production Riccati's `B_w` is most likely flavor-(a) additive force, not flavor-(b) `ΔA·x`. This ties a training-method concern (flavor-of-`max`) to an analysis-side issue (`c723082`, LinearDynamicsAdversary). Protocol: the full diagnosis lives in the test's xfail reason string (implementation-level doc); a coordination comment on `c99ad9d` noted the cross-cutting concern + cross-ref to `c723082`; the planned follow-up ("Riccati `B_w` generalization to `ΔA`") gets a normal `feature` issue only when actually scheduled, then a cross-ref comment on `c99ad9d`.

### What NOT to do

- **Don't comment tier opinions on individual analysis issues.** Tier shifts are cross-cutting → `4d38c15`.
- **Don't put long-form discussion in coordination issue bodies.** Bodies are tables of contents; discussion goes in comments or on the child issue.
- **Don't reference `coordination`-labeled issues in commit `Mandible-Issue:` trailers.** Use the child / feature / bug issue.
- **Don't create a new coordination issue when an existing one's scope covers the concern.** New coords are project-lifetime commitments — discuss first.
- **Don't paste subagent output or raw analysis into a coord body.** Move it to the child issue (or a `results/<exp>/` doc) and replace with a one-line cross-ref.
- **Don't index every commit on the coord.** Only commits that change cross-cutting state (a tier, a method choice, a phase boundary) merit a coord comment.
