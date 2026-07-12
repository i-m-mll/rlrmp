# rlrmp

<!-- CLAUDE.md is a symlink to this file; edit here. -->

## What this project is about

rlrmp is a neuroscience and behavior project about robust motor control. Its core question is how robustness can be induced in neural network controllers, whether the resulting behavior matches robust-control or H-infinity-like signatures observed in human reaching experiments such as Crevecoeur, Cluff, and Scott 2019, and what those trained controllers can then tell us about neural computation.

The neural networks are not the endpoint as an ML benchmark. They are model systems: if a recurrent controller acquires human-like robust behavior under controlled training pressures, we can ask what internal mechanisms support that behavior, whether those mechanisms resemble analytical robust-control formalisms, and what predictions they suggest for brain activity, motor behavior, perturbation responses, or electrophysiology.

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

## Repo Implementation Policy

### Non-negotiable implementation principles

- **Data stays separate from code.** Generated, empirical, or adopted data lives
  in tracked specs or governed data products with schema versions, roles, and
  hashes. Source code holds schemas, loaders, and builders, not baked-in data
  tables.
- **Contracts are obeyed, never bypassed.** If feedbax cannot express an rlrmp
  need, file and fix the feedbax gap. Do not add rlrmp-side shims,
  monkeypatches, private writes, or temporary wrappers that bypass the owning
  contract.
- **Everything official is registered.** Trainers, analyses, diagnostics,
  adversaries, perturbations, reports, and graph pieces must be registered
  feedbax primitives when general, or rlrmp components that fully participate in
  feedbax registration, schema, and migration contracts when project-specific.
- **Graphs are parsimonious and modular.** Factor work into small reusable
  components. General leaves belong in feedbax; only genuinely rlrmp-specific
  scientific components stay here.
- **Runs, evaluations, analyses, and reports are spec-first.** Specs are
  schema-compliant, migratable, and serialized as the source of truth. Execution
  materializes specs and manifests; do not re-derive canonical payloads after
  the fact.
- **A fix without a guard is half a fix.** When a residual class is fixed, add
  or extend a structural CI gate so the class cannot quietly return.
- **Residuals are fixed in-wave.** Bugs and contract gaps discovered during a
  wave are filed, adopted into the relevant coordination surface, and resolved
  in that wave unless blocked by an explicit user decision or external
  infrastructure.

### Integration verification bar

- Before any commit lands on an integration or auth path, run the full test
  suite through the memoized wrapper: `scripts/full_suite.sh`. The wrapper
  keeps the integration bar as the whole `tests/` tree, runs it with
  `pytest-xdist` (`-n auto`) when the memo does not apply, and records a green
  result only for a clean tree whose fingerprint includes the rlrmp tree,
  feedbax editable checkout HEAD, `uv.lock` hash, and Python/JAX versions. If
  any component is dirty or unresolved, the memo fails closed and the full suite
  runs. The legacy underlying command is
  `PYTHONPATH="$PWD/src" uv run --no-sync python -m pytest tests/ -q`.
  `feedbax_contract`-marked tests cannot be skipped by construction.
- While iterating on a fix, run the narrowest relevant tests first: explicit
  node IDs or paths, `-k`, `pytest --lf`, or the repo's selective runner when
  one exists. In this repo, use `scripts/dev_tests.sh` for that inner loop:
  it accepts pytest node IDs, paths, `-k`, and `--lf` passthrough, supports
  `--testmon`, and records branch/HEAD/merge-base metadata so stale
  `.testmondata` is deleted on branch switches or rebases before selection is
  trusted. The first instrumented testmon run can be slower, and JAX tracing
  can make coverage coarse.
- Run the repo's full integration bar only at lane closeout before work lands
  on an integration or auth path, and at most once or twice per lane when a
  rerun is justified. Do not use the full bar to check whether a single fix
  worked. Repo instructions define the integration bar; this norm governs how
  often to pay it. Targeted `-k`/subset runs, `--lf`, and testmon are
  development aids, never auth/integration gates.
- During integration and debugging, targeted selection
  (`scripts/dev_tests.sh`, explicit node IDs, `-k`, `--lf`) is the default
  tool, and the full suite is a scarce, serialized resource to be spent
  deliberately, not a routine check.
- **Full-suite invocations must NEVER run in parallel.** Never run two
  `scripts/full_suite.sh` (or raw `pytest tests/`) invocations at the same
  time — not two in one checkout, not one per worktree across this repo's
  worktrees, and not via concurrently-dispatched subagents each kicking off
  their own run. The full suite claims shared GPU/JAX/compilation resources
  and the memo's fingerprinting assumes a single writer; concurrent runs
  produce spurious failures and corrupt the memoized result. Any session that
  delegates work to subagents must pass this constraint down explicitly to
  every subagent it spawns, not assume it is inherited.
- New tests must be safe under `pytest-xdist`: write only to `tmp_path` or a
  unique per-test directory, do not write to shared `_artifacts/` locations
  unless the path includes a test-unique segment, and restore any process-global
  JAX, registry, environment, or cwd changes before the test exits. Tests must
  not depend on collection or execution order.
- The suite includes a feedbax pin-drift check
  (`tests/test_feedbax_ref_pin.py`): it fails when `ci/feedbax-ref.toml` no
  longer matches the feedbax checkout the editable install runs against. The
  fix is to bump the pin in the same wave as the feedbax change and rerun the
  full suite — never to skip or delete the check. (Precedent escape: `7766182`
  sat undetected on `main` because local feedbax had advanced past the CI
  pin.)
- Auth-request specs for changes that produce durable outputs must name the
  registered surfaces (recipes, manifests, custody routes) those outputs route
  through; an output that bypasses them must be called out and justified in
  the spec.

### Analysis pipeline policy

The current analysis pipeline is feedbax-native and manifest-canonical:

1. Registered evaluation recipes under `rlrmp.eval.*` produce
   `EvaluationRunManifest` records and cached evaluation states. Evaluation
   identity comes from the canonical `EvaluationRunSpec`; analyses must consume
   these manifests and must not rerun rollouts internally.
2. Analyses are registered recipes that build `AbstractAnalysis` nodes. Their
   inputs are `ParentRef`s to upstream manifests, especially evaluation
   manifests when the analysis consumes rollout/evaluation states.
3. Reports are registered `rlrmp.report.*` recipes. Report stages render
   feedbax-custody `report_render` artifacts. Tracked Markdown notes are
   downstream exports of those report artifacts; `rlrmp.io.update_marked_section`
   still governs any tracked note copy.
4. For paired or comparative conditions, such as "SISU-on vs SISU-off", the
   canonical shape is one staged `AnalysisBundleSpec`: one evaluation stage per
   condition using the same evaluation recipe with different params, then
   analysis stages whose `depends_on` lists those evaluation stages. Use
   ungrouped/per-run stages for independent per-condition analyses. Use grouped
   stages for one analysis consuming all conditions together, and set
   `include_bundle_inputs` when that grouped analysis also needs the original
   training context. Feedbax caches and reuses evals by manifest identity.

### Feedbax contract CI gates

The marked gate is `feedbax_contract` in `ci/feedbax-contract-suite.toml`. The
manifest is part of the contract: skips are failures, negative canaries should
remain meaningful, and allowlists are shrink-only unless a new issue documents
why an exception is still required.

| Family | Protects |
|---|---|
| `analysis_recipe_contract` | Analysis recipes stay registered, schema-bearing, and executable through feedbax. |
| `analysis_write_custody` | Analysis outputs use feedbax custody instead of direct durable writes. |
| `analysis_eval_dependency` | Registered analyses that need rollouts depend on evaluation manifests. |
| `product_identity_hash` | Governed data products fail closed when identities or hashes drift. |
| `generated_data_constant_scan` | Generated/adopted data does not re-enter `src/` as high-precision constants. |
| `retired_id_scan` | Retired component IDs remain confined to archive/compatibility readers. |
| `write_surface` | Durable writer surfaces are explicit, reviewed, and custody-routed. |
| `import_boundary` | rlrmp imports feedbax through public, canonical APIs only. |
| `graph_spec_contract` and `artifact_manifest_normalization` | Graph specs and manifests remain schema-compatible with feedbax. |
| `reaccretion_ratchet`, `lane_b_terminal_gate`, `lane_c_terminal_gate` | Deleted or remediated residual classes cannot quietly return. |

### Query-language adoption is authoring-time, not retrofit

Feedbax expression-grammar declarations (`Coalesce`, `Filter`, defaults,
optional sources) belong on governed spec surfaces — documents, registrations,
manifests — declared when the surface is authored. Do not retrofit in-function
Python into expression ASTs: the per-target adjudication in
`results/96ac0e5/notes/adjudication.md` (resolved in
`results/86e1dd1/notes/post_grammar_adjudication.md`) rejected every such
migration. Migrate only when the expression lands on a governed surface or
kills cross-file duplication of one selection/gating shape. Semantic-drift
caveat: `Coalesce`'s absence class is path-missing/zero-match only — it is not
Python `or`-falsiness, and it treats an explicit null as a hit, unlike
`is None`-advance idioms.

### LEGACY-banner convention

A `LEGACY (frozen ..., issue ...)` banner means the file or function is retained
for provenance or deferred porting. It is not contract-native, may not run, and
is never a pattern to copy. New work must either port the behavior into the
feedbax-native recipe/bundle/manifest pipeline or delete the obsolete surface
when its provenance value is gone. The current inventory is
`results/3cf909c/notes/legacy_materializers.md`.

## Standard Certificate Presentation

The prescriptions below are the presentation contract for Phase-3 bridge
standard-certificate results. They are being migrated into the
`rlrmp.report.bridge_certificate_notes` recipe so the table and annotations are
rendered by construction (rlrmp `9c342ba`; a spec-governed template layer,
feedbax `4d1558c`, is the longer-term home). Until that lands this section is the
authority, and the rules here that no render can enforce — showing a table
rather than prose when presenting results in conversation, recomputing rather
than leaving components partial, and keeping evaluation lenses distinct from
training axes — remain agent-facing regardless.

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

The `pooled_trial_mean_with_band` and `replicate_mean_curves` helpers in `rlrmp.analysis.math.trial_alignment` trim aligned profiles to the strict full-support column window (`min_coverage=1.0`) before reducing. Callers receive the trim slice alongside the curves so companion time axes can be sliced consistently (`t = ((np.arange(n) - center) * dt)[sl]`). Pass `trim=False` only when downstream code needs identical step axes across multiple invocations (e.g. cross-cell pairwise RMSE) and the reducer is already NaN-tolerant.

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

## Experiment Protocol

Experiment-running issues use the `experiment` label. They stay open until a
verdict comment records whether the experiment was `answered`, `superseded`, or
`bracketed`; intermediate runs and partial analyses do not close the issue by
themselves.

Use the dotfiles `run-experiment` skill as the detailed workflow owner for
experiment setup, run management, post-run interpretation, review packets, and
closure semantics. rlrmp sessions should preserve the standard topology: an
umbrella or parent thread owns coordination and reads ledger status, an
implementation thread owns code/spec changes, and a run-management thread owns
remote/local run execution and monitoring.

## RunPod Deploy Runbook for rlrmp Experiments

### Spec lock before launch (hard gate)

Before any **billable** training launch (`runpodctl pod create` for a run, or a
non-smoke `modal run`), present a one-table run spec — task structure, loss
terms, dims, `n_batches`, `lr`, seeds, GPU/cloud — and obtain **explicit user
confirmation**. Mid-session plan discussion is NOT launch authorization.

Do not unilaterally upgrade cloud tier or GPU class — ask the user first.
Secure cloud is the default for billable RunPod launches; use community cloud
only when the user explicitly accepts the lower isolation tier for that run.

### Prerequisites and GPU choice

- `runpodctl` binary from GitHub releases (its `install.sh` needs sudo; download
  the binary directly if sudo is unavailable). SSH key at
  `~/.runpod/ssh/RunPod-Key-Go`; API key via `runpodctl config --apiKey <key>`.
- Check GPU availability per datacenter with `runpodctl datacenter list`.
- **RTX 4090 secure cloud** is the validated baseline. **RTX 5090** (Blackwell)
  is faster but image-sensitive: use a `cu1281`-or-newer PyTorch image whose
  Docker tag you have verified exists, then install `jax[cuda12]` after
  `uv sync`; skip the deprecated `runpod/pytorch:2.8.0-...` template.

### Deploy and monitor via the feedbax scripts

Use the feedbax deploy automation as the deploy/bootstrap/monitor surface; do
not reproduce a manual recipe here.

- `~/Main/10 Projects/10 PhD/20 Feedbax/feedbax/scripts/deploy/runpod_deploy.sh`
- `~/Main/10 Projects/10 PhD/20 Feedbax/feedbax/scripts/deploy/poll_run.sh`

They own pod-create, SSH endpoint discovery, dead-state fail-fast, path-dependency
sync and local-path patching, the nohup+sentinel install pattern, the
poisoned-venv probe-and-rebuild on pod reuse, and cadence-based polling with
per-row/per-batch progress. Read their `--help` / dry-run surfaces first. **When
they fail or fall short, harden the scripts and file the gap** (feedbax issue,
surfaced-from note per the meta coordination protocol) rather than reverting to
a manual recipe in this file.

Residual invariants the scripts do not fully own (agent judgment still required):

- Expose `22/tcp` at pod creation; the default RunPod port set can leave direct
  TCP SSH unreachable.
- Scan the nohup log for `ptxas` warnings, `OOM`, and `Traceback` alongside the
  loss-progress signal — `poll_run.sh` reports structured status, not error
  patterns.
- Verify the Docker image tag exists on Docker Hub before `pod create`; stale
  tags cause silent deploy failures. (The `uptimeSeconds`-is-not-liveness rule,
  Bug `b399efc`, is implemented in the scripts — do not reintroduce manual
  liveness heuristics around them.)
- After a CUDA-JAX install on a pod, use `uv run --no-sync`; do not re-run
  `uv sync`.
- In manual fallback only: patch embedded local editable paths with
  literal-string replacements, never broad `sed` globs — those can corrupt TOML
  and lockfile metadata.
- Pod billing starts on **creation**, not container start (dotfiles `3602840`);
  a live-but-slow pod bills while it boots.
- **Modal:** destructive CLI ops (`modal app stop`, `modal volume rm`) require
  `--yes`/`-y`; non-interactive shells abort without it.

### Smoke test

For multi-row launches forked from one source checkpoint, use the tracked
pre-launch gate instead of pod-local wrapper scripts. Author the launch rows as
a tracked `TrainingRunMatrixSpec`; render the spec-lock table from that matrix
with Feedbax's run-matrix tooling and carry the rendered table in `RUN_PLAN.md`
inside the normal marked section.

Emit new matrices through the RLRMP storage entry point; do not write a
materialized base into the tracked matrix. The source document must use a
content-pinned `authored_intent` or `resolved_output` base. The command writes
the canonical tracked intent and places the resolved snapshot and execution
capsule in content-addressed Feedbax custody:

```bash
PYTHONPATH=src uv run --no-sync python scripts/emit_training_run_matrix.py \
  results/<issue>/runs/matrix.intent.json \
  --output results/<issue>/runs/matrix.json
```

Run the gate from the owning feature worktree and point all checkpoint roots at
tmp or launch-owned locations, not shared `_artifacts` test scratch:

```bash
PYTHONPATH=src uv run --no-sync python scripts/fork_checkpoint_gate.py \
  --matrix results/<issue>/runs/matrix.json \
  --source-checkpoint-root /workspace/source/checkpoints_adversarial \
  --parity-output results/<issue>/notes/fork_parity.json \
  --target row_a=/workspace/row_a/checkpoints_adversarial \
  --target row_b=/workspace/row_b/checkpoints_adversarial
```

The gate registers RLRMP training methods, extracts nested
`feedbax_training_run_spec` payloads before wrapper validation, performs the
one-source fork, reads the target fork manifests, writes a row-by-slot digest
table, and fails nonzero with `row=<id> slot=<slot>` if a target digest differs
from the source. It also emits `LR_CONTINUATION step=<n> lr=<x>` from the first
target row's declared continuation mode, so the launch log records whether LR
semantics restart or continue.

```bash
cd /workspace/rlrmp
uv run --no-sync python scripts/train_minimax.py \
  --adversary-type linear_dynamics \
  --n-warmup-batches 3 --n-adversary-batches 20 --adv-batch-size 250 \
  --n-replicates 5 --hidden-type gru \
  --output-dir /workspace/smoke_test --checkpoint --fused
```

Adjust flags to the current CLI. Pause after the smoke test and do not launch
the main matrix until the user confirms. rlrmp training scripts emit
grep-friendly `BATCH phase=<phase> batch=<i>/<n> [loss=<x>] [elapsed=<s>s]`
progress lines (helper `rlrmp.train.progress`) that `poll_run.sh` consumes;
these are log-only, never Mandible checkpoints (see the run-status convention
below).

### Post-training-run protocol

After every remote training run completes, use `scripts/post_run.sh` wherever
possible as the deterministic handoff step. It owns the mechanical protocol:
artifact sync from local, Modal, or pod sources; tracked run-spec creation under
`results/<issue>/runs/<run>.json`; bulk artifact placement under
`_artifacts/<issue>/runs/<run>/`; metrics-table rendering from
`training_summary.json`; `git add`; `agent-commit` through the wrapper; Mandible
auth request submission; and the terminal run-status checkpoint.

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

Run-management sessions that are not authorized to commit or submit auth should
use sync-only mode first:

```bash
scripts/post_run.sh --issue <tracking-issue> --run <group>__<variant> \
  --artifacts-src <local:/path|/path|modal[:volume]|pod:user@host:/path> \
  --sync-only --dry-run

scripts/post_run.sh --issue <tracking-issue> --run <group>__<variant> \
  --artifacts-src <local:/path|/path|modal[:volume]|pod:user@host:/path> \
  --sync-only
```

`--sync-only` performs the artifact sync, verifies the synced artifact payload,
renders the metrics table from `training_summary.json`, emits the one terminal
`run-status` checkpoint, and writes
`_artifacts/<issue>/runs/<run>/.post_run_synced.json`. It deliberately skips
run-spec creation, `git add`, commit, and auth request submission. A later
authorized full invocation re-verifies the synced artifacts, writes/commits the
tracked run spec, and skips both re-transfer and duplicate run-status emission.
Pass `--force-sync` only when the source should be transferred again.

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

Provenance and manifest enforcement live in `scripts/post_run.sh` itself, not
here: it pins the feedbax manifest root (`FEEDBAX_RUNS_DIR`), verifies the
emitted `TrainingRunManifest` and tracked recipe identity before committing
(without reconstructing or hash-reconciling the manifest), requires the run
recipe at the flat canonical path `results/<hash>/runs/<run>.json`, blocks a
dirty `uv.lock` (`POST_RUN_ALLOW_DIRTY_UV_LOCK=1` is a documented emergency
override only), and reports `not_found` / `archive-only` parity for legacy runs
rather than implying a checked manifest. Run a dry run first — it previews the
full provenance stamp. If the script cannot cover a source shape, preserve its
layout and auth/commit conventions in the manual fallback and file the script
gap.

Run-status checkpoint convention (`e8b5b3b`):

- The Mandible ledger gets **one terminal run-status checkpoint per run**, not
  a stream of transient phase checkpoints. `scripts/post_run.sh` emits exactly
  one `kind=run-status` checkpoint (`phase=completed`) on a non-dry run via
  `mandible issue checkpoint add <issue> --kind run-status --payload-file -`.
  Use `phase=failed` if recording a failed run by hand. An optional `launched`
  checkpoint at launch is justified only when a *different* session will resume
  monitoring.
- Schema (`schema_version=1`): `run_id`, `phase`
  (`launched`|`completed`|`failed`), `timestamp`, `artifact_dir` (repo-relative),
  `run_spec_path` (repo-relative **reference** to the tracked `run.json` — never
  inline hyperparameters), and `metrics_summary` (the scalar metrics from
  `training_summary.json`).
- A dry run **previews** the payload (prints the intended
  `mandible issue checkpoint add` command and the JSON) without writing it.
- **Resumability test** — emit a ledger checkpoint only for state a fresh
  session would need to resume or close out the run: "would a fresh session
  need this to tell a finished run from an in-flight one?" If yes (terminal
  status, artifact location, headline metrics) → the one run-status checkpoint.
  If no (per-poll progress, transient retries, batch counters) → chat or the
  nohup log, never a checkpoint. The per-batch `BATCH` progress lines (§7) are
  log-only for exactly this reason.

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

### Generated/adopted empirical data lives in governed products, not source constants (Bug: `ea6ccb4`)

Generated, empirical, or adopted-analytical datasets (calibration tables, budget anchors, and similar) must NOT be baked into `src/` as module-level Python constants. Source code keeps schemas, loaders, and builders; the data lives in a tracked, schema-versioned data product — persisted under `results/<hash>/data_products/` on the Feedbax `AnalysisDataProduct` envelope with an rlrmp `product_schema_id` — and is loaded at runtime by typed product identity with fail-closed validation (schema, role, `product_identity_hash`, and artifact hash), never trusted as a source literal. Consumers use the loaders in `rlrmp.data_products` (`load_open_loop_calibration`, `load_broad_epsilon_anchors`); emitted run specs snapshot each consumed identity via `add_consumed_data_identity`. The AST data-lint (`rlrmp.data_products.lint`, family `generated_data_constant_scan` in `ci/feedbax-contract-suite.toml`) enforces this: it flags multi-entry high-precision float container literals under `src/` unless they are loader-fed or allowlisted-with-rationale. Small conventional constants, dimensions, solver tolerances, and enum-like labels are out of scope; a single adopted scalar may be allowlisted-with-rationale instead of migrated.

### Flat-by-hash layout (Bug: `f485c26`)

Each top-level entry under `results/` and `_artifacts/` is a **directory named by its 7-character tracking-issue prefix** (e.g. `results/2bc95fd/`, `_artifacts/efc4d68/`). The issue is the atomic unit; no umbrella-level parent dirs in the directory tree (umbrella membership lives on the issue body / `b33e8da` index).

Inside each `<hash>/`:

```
<hash>/
├── README.md                      one paragraph orienting context
├── RUN_PLAN.md                    if the work involves training runs
├── notes/<topic>.md               narrative + analysis writeups
├── figures/<topic>/spec.json      figure specs (one dir per topic)
└── runs/<variant>.json            canonical per-run recipe (flat, not nested)
```

The mirror `_artifacts/<hash>/...` follows the same structure by role:
`_artifacts/<hash>/runs/<variant>/` is a directory holding the bulk training
outputs; `results/<hash>/runs/<variant>.json` is the corresponding canonical
recipe. If a run needs additional tracked sidecars, use
`results/<hash>/runs/<variant>/` for those sidecars only.

### Run-folder convention

- **`runs/<variant>.json`** flat by default — one JSON file per run, not a directory with a single file in it.
- **`runs/<variant>/`** is optional tracked sidecar space for lightweight
  per-run notes, debug metadata, or historical `run.json` compatibility. Do not
  put new canonical recipes there.
- **Run row labels are compact local identifiers, not full parameter
  summaries.** They must be path-safe and unique within the experiment issue's
  run set. Include only fields that distinguish sibling rows or prevent
  ambiguity with existing rows in the same issue. Parameters that are constant
  across the whole run batch belong in the spec lock table, `RUN_PLAN.md`, and
  the run JSON, not repeated in every row label. For example, if every row uses
  `gradient_clip_norm=5`, batch size 64, the same objective, and the same
  schedule, do not add `clip5`, `b64`, the objective, or the schedule to every
  label merely for completeness; use labels such as `lr1e-3` / `lr3e-3` or
  `no_pgd` / `pgd_ofb` when those are the actual varying axes.
- For complex sweeps where variant names balloon (~50+ chars), use `runs/<hash>.json` + `runs_index.json` mapping hash → human label + params.
- Keep bulk arrays/checkpoints/logs under `_artifacts/<hash>/runs/<variant>/`;
  do not promote heavy run outputs into `results/<hash>/runs/<variant>/`.

### Script placement: experiment-specific vs reusable (Bug: 8404108)

The top-level `scripts/` directory is for cross-cutting tooling — scripts that operate generically across experiments (e.g. `train_minimax.py`, `eval_minimax.py`, `eval_diagnostics.py`, infrastructure shell scripts). It is NOT a dumping ground for experiment-specific analysis code.

**Hard rules:**

1. **Capability-named library modules.** Modules under `src/rlrmp/` MUST be named by capability — `eval`, `train`, `plot`/`viz`, `analysis`, `lme`, etc. — never by experiment, phase, or paper (no `part2_5`, `methodology_fix`, `shahbazi`, `tier1`). If you want to call a module `<phase>_helpers.py`, identify the underlying capability and use that name instead. Within a capability module, training-method-specific sub-modules ARE allowed (`rlrmp.train.minimax`, `rlrmp.eval.minimax_io`) because training methods are stable concepts spanning experiments. Experiment-named sub-modules are still forbidden.
2. **Experiment-specific scripts** (analysis pipelines, plotting, one-off diagnostics tied to a single tracking issue) live with the experiment: `results/<hash>/scripts/<name>.py`. Commit them alongside the experiment's `runs/`, `notes/`, `figures/` with `mandible commit linked --issue <hash>` so the commit carries the matching `Mandible-Issue: <hash>` trailer.
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
| A top-level experiment script exporting `eval_ensemble_on_trials` | Sibling-script import of a generic primitive. Extract to `src/rlrmp/eval/`. |

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
- **Pre-existing worktrees** can pick up the symlink by running `~/.dotfiles/bin/wt sync` from inside the worktree.
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

This project uses a small set of long-lived **coordination issues** (label:
`coordination`) as themed decision logs. A coordination issue is a dated
timeline of decisions and notes on one theme — the body is a stable orientation
index (scope, what's owned, cross-refs), and the comments are the log, which
legitimately includes long-form theme notes that do not deserve their own issue.
They are distinct from `umbrella` issues (which bundle a specific body of work
and close when it lands) and from ordinary `feature` / `error` issues (which
carry the substantive work). The generic rules here (closure test, body/comment
split, trailer hygiene, cross-repo surfacing) are being promoted to the global
instructions (dotfiles `86aa0b5`); once that lands, the global wording is
authoritative and this section keeps only rlrmp's concrete surfaces and routing.

For Mandible issue command syntax, see the global `~/.codex/AGENTS.md`
issue-tracking convention.

### The four coordination issues

Each has the `coordination` label and is project-lifetime (no closure-on-merge
intent).

| ID | Name | Scope |
|---|---|---|
| `4d38c15` | Project analyses coordination | Cross-cutting decisions about *analyses applied to trained models*. |
| `c99ad9d` | Project training-methods coordination | Cross-cutting decisions about *how models are trained*. |
| `b33e8da` | Project umbrella index | Timeline of experimental/work umbrellas: creation, boundaries, pivots, outcomes. |
| `1d9ae6f` | Project meta coordination | Cross-project / workflow concerns surfaced from rlrmp work (Mandible, feedbax, dotfiles, general workflow). |

#### `4d38c15` — analyses

**Owns:** new analyses worth doing; tier shifts (essential / desirable / auxiliary / deprecated); cross-cutting findings across multiple analyses; deprecation/archival of analyses.
**Does NOT own:** the design/math/implementation/results of any single analysis (those live on the analysis's own issue).
**Triggers:** discovered a new analysis; want to re-rank tiers; one analysis subsumes another; an analysis became less informative after a method change.

#### `c99ad9d` — training-methods

**Owns:** training method menu (standard backprop, CVaR, APT, minimax, LEQG, PAI-ASF, BCS, DAI); adversary classes (parametric force fields, GaussianBumpAdversary, structural ΔA); flavor-of-`max` choices (input-instance / model-class / LEQG-via-Whittle); SISU wirings; plant-regime parameters when they couple to training (damping, motor noise, reach geometry, loss schedule); method deprecations and promotions.
**Does NOT own:** specific analyses (→ `4d38c15`); phase markers (→ `b33e8da`); model-structure decisions independent of training method (may eventually move to a separate model-structure coord).
**Triggers:** introducing a new training method or adversary class; redesigning an existing adversary; flavor-of-`max` decisions; cross-method tier shifts; training-relevant plant-regime changes.

#### `b33e8da` — umbrella index

**Owns:** the timeline of work umbrellas (current and past) with one-line
motivating-question + outcome; umbrella boundaries, pivots, and successions;
cross-references to umbrella artifacts (READMEs, synthesis docs).
**Does NOT own:** the work content of any umbrella (lives on the umbrella and its
children) or in-umbrella analyses (→ `4d38c15`).
**Triggers:** creating a new umbrella (comment here with its ID + motivating
question); an umbrella ends, pivots, or is abandoned (follow-up comment with the
outcome). The `coordinate-umbrella` skill posts these entries as a step, so the
index stays current rather than depending on memory.
**Note:** "phase" was prose convention layered on ordinary umbrellas with no
schema or CLI backing; this surface is now simply the umbrella index. Historical
comments that say "phase" mean "umbrella".

#### `1d9ae6f` — meta cross-project

**Owns:** workflow / tooling concerns surfaced during rlrmp work that need a fix in **another** repo — Mandible, feedbax, dotfiles, or general workflow. The body is an index; the actual fixes live in the destination repos.
**Does NOT own:** rlrmp-internal concerns (those go to one of the three coords above or to a normal issue).
**Triggers:** noticing a Mandible bug while in rlrmp; needing a feedbax API change to support rlrmp work; spotting a global AGENTS.md gap; identifying a tooling improvement.

### Umbrella vs coordination — which label?

Apply the global closure test: *should this issue close when the work it tracks
merges?* Yes → `umbrella` (work-bundle-tied; may close deliberately via an auth
`--closes-issue`, a `Closes-Mandible-Issue:` trailer, or user action — e.g.
`b557d4e` closed when its synthesis-review work merged). No → `coordination`
(project-lifetime; persists).

### Body vs comments on a coordination issue (`1ba096f`)

- The **body** is a stable index: scope, what's owned and what isn't, and
  cross-references. Keep it small; it rarely changes.
- The **comments** are the dated log: decisions, tier shifts, corrections, and
  long-form theme notes that do not warrant their own issue. Long-form notes
  here are legitimate — this is the project's research timeline for the theme,
  and later comments routinely cite earlier ones by ordinal.
- **Substantive results tied to a specific unit of work** (a run's numbers, an
  analysis's plots) live on that work's own issue; the coordination comment
  carries the decision plus a one-line cross-ref, not the full write-up.
- **Hard rule (enforcement-worthy):** never attach a work unit, branch, or
  `Mandible-Issue:` trailer to a coordination issue. It is a log, not a
  work destination. Anything that becomes a unit of work graduates to its own
  issue.

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
| Starting a new work-bundle umbrella | yes (`umbrella`) | `b33e8da` (umbrella ID + one-line motivating question) |
| Umbrella outcome (merged / abandoned / pivoted) | no | `b33e8da` |
| Mandible / feedbax / dotfiles concern surfaced from rlrmp | yes, **in destination repo** | `1d9ae6f` (destination repo + issue ID + one-liner) |
| General-workflow / tooling improvement | yes, in appropriate repo | `1d9ae6f` |
| Bug or feature *entirely within rlrmp* | yes (`error` / `feature`) | (no coord — direct work) |

When the table doesn't cover your case: ask "is this a project-lifetime decision (→ coord comment) or a unit of work (→ new issue)?". If both, do both.

### Cross-referencing protocol

- Use **7-character issue-ID prefixes** in bodies and comments (e.g. `4d38c15`).
- When filing in a destination repo (per `1d9ae6f`), include a one-line
  "surfaced from rlrmp" note in the destination issue's body with the rlrmp
  issue ID or branch that surfaced it, **and create a structured cross-repo
  link** (`mandible issue link`) at filing time so the unresolved dependency is
  queryable — a prose promise to "report back later" is not a mechanism and has
  empirically not held.
- When you notice a destination-repo issue has resolved, comment the resolution
  (merge SHA / closing pointer) back on `1d9ae6f` — good practice, but the
  structured link is the requirement.
- Coordination issue bodies index children — they do not duplicate child content.

### Commit `Mandible-Issue:` trailers — never reference coordination issues

`Mandible-Issue:` trailers are for the relevant child / feature / bug issue - the unit of work the commit completes. Coordination issues are decision-tracking surfaces, not commit destinations. So `mandible commit linked --issue <id>` should always take a child / feature / bug issue ID, never `4d38c15` / `c99ad9d` / `b33e8da` / `1d9ae6f`. Use `Closes-Mandible-Issue: <id>` only when the commit should deliberately close that issue through the auth merge.

### Umbrella protocol

1. **Create the umbrella** — a new issue labeled `umbrella` (and `feature` if
   appropriate). Body: minimal — motivating question, scope, links to artifacts
   (e.g. a `results/<hash>/README.md`).
2. **Comment on `b33e8da`** with the umbrella ID + one-line motivating question,
   so the index stays the "what umbrellas exist" discovery surface. The
   `coordinate-umbrella` skill does this as a step.
3. **Children** reference the umbrella in **their bodies** ("Part of
   `<umbrella>`."), not in commit trailers. Close the umbrella only deliberately
   when its work is done.
4. **On end / pivot / abandonment**, comment on `b33e8da` with the one-line
   outcome ("merged via X", "pivoted to Y", "abandoned because Z").

Past umbrellas for orientation (see `b33e8da` for the live inventory): Part 1
(`297260c`), Part 2 (`0af472c`), Part 2.5 (`844ef95`), Methodology-fix
(`b557d4e`), feedbax-native alignment (`64a04e0`). Check the ledger for the
current umbrella before treating any historical one as active.

### Worked example: routing a cross-cutting finding

A test xfail (`tests/test_hinf_riccati.py::test_cs_faithful_qr_velocity_inflation`)
produced a diagnosis that touches both a training-method concern (Riccati
flavor-of-`max`) and an analysis-side issue (`c723082`). Routing: the full
diagnosis stays in the test's xfail reason string (implementation-level doc); a
`c99ad9d` comment records the cross-cutting concern + cross-ref; the follow-up
gets its own `feature` issue only when scheduled. The pattern — deep detail on
the work surface, a decision-plus-cross-ref on the coordination surface — is the
point.

### What NOT to do

- **Don't comment tier opinions on individual analysis issues.** Tier shifts are
  cross-cutting → `4d38c15`.
- **Don't attach work units, branches, or `Mandible-Issue:` trailers to a
  coordination issue.** Use the child / feature / bug issue.
- **Don't create a new coordination issue when an existing scope covers it.**
  New coordination surfaces are project-lifetime commitments — discuss first.
- **Don't log every commit.** Only commits that change cross-cutting state (a
  tier, a method choice, an umbrella boundary) merit a coordination comment.
