# Feedbax backlog: code-archaeology audit (issue `05883e7`)

Everything in this note belongs in the **feedbax repo**, not rlrmp. It is a
filing backlog, not filed work — per the project's cross-repo surfacing
convention, each entry graduates to its own feedbax issue at remediation
time, with a "surfaced from rlrmp `05883e7`" note and a structured cross-repo
link back to this audit. Nothing here is authorized work; this document only
records what the audit found and where the evidence lives.

## 1. Real bug: broken `AbstractIntervenor` import

`feedbax/analysis/support.py:122` does
`from feedbax.intervene import AbstractIntervenor`, but `AbstractIntervenor`
is not actually exported from the `feedbax.intervene` package facade
(`feedbax/intervene/__init__.py`, 31 LOC — a thin package facade that does
not currently re-export the name). The Phase 3 dangling-reference sweep
flagged this as `dangling_import_name` at `medium` confidence and its own
spot-check confirmed the line reads exactly as flagged. This is a real,
actionable bug, not a sweep false-positive: the confidence is `medium` rather
than `high` only because the sweep's own limitation (a name reachable via a
dynamic `__getattr__` mechanism can still register as a false positive) has
not been separately ruled out for this specific case — worth a quick manual
check of whether `feedbax.intervene`'s `__init__.py` defines `__getattr__`
before filing, but nothing in the catalog or classification evidence
suggests it does.

**Evidence:** `_artifacts/05883e7/audit/sweeps/dangling/dangling.jsonl`
(`{"confidence": "medium", "kind": "dangling_import_name", "line": 122,
"module": "feedbax/analysis/support.py", "target":
"feedbax.intervene.AbstractIntervenor"}`); spot-checked in
`sweeps/dangling/summary.md`'s manual-spot-check section.

**Suggested destination issue title:** "Fix broken `from feedbax.intervene
import AbstractIntervenor` in `analysis/support.py`"

## 2. Seven deprecate/delete legacy modules (zero-consumer analysis layer + sqlite chain)

The Phase 3 reverse audit individually investigated all 34 modules Phase 1's
catalog flagged non-`framework_general`, checking each against real
import/consumer evidence rather than trusting the catalog's first-pass
impression. Seven came back `deprecate_delete`, and — this is the point worth
making explicit — **they form one coherent orphaned subsystem, not seven
unrelated findings**: a Feedbax-Studio-era analysis and persistence layer
built around a legacy SQLAlchemy sqlite database that current rlrmp (spec-
first, manifest-canonical) no longer touches at all.

- **`feedbax/analysis/setup.py`** (454 LOC) — its only live call chain is
  `query_and_load_model`, called from `analysis/execution.py`'s legacy
  `db_session` model-loading path. No `.db` files and no
  `persistence.database` imports exist anywhere in current rlrmp source.
- **`feedbax/bin/db_merge.py`** (355 LOC) — exists solely to merge sqlite
  databases of the kind `analysis/setup.py`'s legacy path reads; no consumer
  anywhere, feedbax-internal or rlrmp.
- **`feedbax/analysis/effector.py`** (149 LOC) — zero importers in either
  repo, no analysis-recipe registration. Reach-task-specific 2D
  effector-trajectory scatter plotting; if wanted again, recreate via the
  current `ScatterPlots`/`AlignedVars` recipe path rather than reviving this.
- **`feedbax/analysis/profiles.py`** (156 LOC) — zero resolved consumers,
  verified via both xref and a direct grep of both trees.
- **`feedbax/config/defaults.py`** (48 LOC) — `POS_ENDPOINTS_ALIGNED` has
  zero inbound references anywhere and should be deleted outright regardless
  of the rest of the module's fate; the remaining constants
  (`REPLICATE_CRITERION`, `TASK_EVAL_PARAMS`,
  `get_iterations_to_save_model_parameters`) are consumed only by the same
  legacy `feedbax/analysis/*` + `config/hyperparams.py` chain — retire
  together if that chain retires.
- **`feedbax/plot/mpl.py`** (1,302 LOC) — zero consumers across feedbax,
  tests, examples, and rlrmp; superseded wholesale by the plotly-based
  `feedbax.plot` package. **Before deleting**, check the merge-spec
  rationale at `docs/design/feedbax_merge_spec.md:212` — this module was a
  *deliberate* retention at merge time, so confirm no downstream consumer
  (Feedbax Studio, notebooks outside the audited tree) still relies on it.
  (Separately, 29 of its objects, 1,194 LOC, are independently
  verification-confirmed dead at the object level — see remediation
  portfolio class (a) — consistent with, not contradicting, the whole-module
  finding here.)
- **`feedbax/web/ws/simulation.py`** (16 LOC) — zero test coverage, zero
  frontend consumer, a hardcoded no-op payload. Either finish wiring it to a
  real simulation-state stream or remove the router mount; a feedbax issue
  should decide which rather than silently deleting a half-built feature.

**File these as one feedbax cleanup wave** (a single umbrella issue with the
seven as either sub-items or one combined PR), not seven independent issues —
the shared root cause (an abandoned Studio-era db-centric analysis path) means
review and testing effort is largely shared across all seven.

**Evidence:** `_artifacts/05883e7/audit/sweeps/reverse_audit/summary.md`
("Deprecate / delete" table); LOC figures cross-checked against
`_artifacts/05883e7/audit/synthesis/portfolio.json`'s `b_legacy_tree_retirements`
array (`b_reverse_audit/*` entries).

**Suggested destination issue title:** "Retire the orphaned Studio-era sqlite
analysis/persistence layer (`analysis/{setup,effector,profiles}.py`,
`bin/db_merge.py`, `config/defaults.py`, `plot/mpl.py`, `web/ws/simulation.py`)"

## 3. Five generalize-in-place items (rlrmp-shaped defaults baked into live core)

Distinct from the deprecate/delete cluster above: these five modules have
**real, multi-consumer feedbax-internal call sites** and should stay in
feedbax — but each carries rlrmp-shaped naming, defaults, or content baked
directly into otherwise-general framework code, which the reverse audit
recommends lifting into caller-supplied config rather than moving the module.

- **`feedbax/analysis/execution.py`** — `load_trained_models_and_aux_objects`
  and `setup_eval_for_module` hardcode the `'train__pert__std'` LDict
  grouping key and an explicit `sisu` special case (SISU is an rlrmp
  training-method concept — see rlrmp's `c99ad9d` training-methods
  coordination) directly as literal strings. **Note:** this is
  `feedbax/analysis/execution.py` (996 LOC), a different module from the
  separate, much larger `feedbax/studio/execution.py` (3,003 LOC) — worth
  being precise about which `execution.py` when filing, since both exist in
  the tree. Fix: lift the grouping key and the SISU case into
  caller-supplied config/registration.
- **`feedbax/component_registry/cde_templates.py`** — ships four hardcoded,
  versioned architecture presets ("Anti-NF", "CDE Hybrid v9b", with a comment
  literally reading "Production architecture") as core registry-default
  content. Fix: move these four specific presets out of core builtins into
  an examples/demo template pack, or a project-registerable template-pack
  extension point, so `component_registry/builtins.py` stops shipping one
  research program's iteration history as framework default content.
- **`feedbax/persistence/database.py`** — `EvaluationRecord`/`ModelRecord`
  declare ORM columns (`perturbation_config`, `sisu_params`, `pert__type`,
  `pert__std`) tied to one project's reach/perturbation study; the module's
  own docstring calls itself "this legacy SQLAlchemy database." Fix: rename
  to a schema-neutral extensible-metadata pattern (e.g. a generic JSON
  `condition_metadata` column plus documented conventional keys). Stays in
  feedbax — all real callers (Studio web API, `bin/check_db.py`,
  `analysis/execution.py`) are feedbax-internal, and rlrmp has zero live
  imports of this module.
- **`feedbax/plot/color_setup.py`** — the mechanism
  (`ColorscaleSpec`/`setup_colors`/`get_colors_dicts_from_discrete`/
  `is_discrete_colorscale`) is general and its only real call-sites
  (`analysis/execution.py`, `analysis/effector.py`) are feedbax-internal, but
  the actual *content* — the `COLORSCALES`/`COMMON_COLOR_SPECS` default
  table — is populated exclusively with rlrmp's hyperparameter naming scheme
  (`pert__amp`, `train__pert__std`, `sisu`, `reach_condition`, `eval_n`); no
  other feedbax consumer supplies a matching shape, and the declared
  `analysis_module.COLOR_FNS` extension point meant to let other projects
  supply their own color specs is currently unused/dead in both repos. Fix:
  keep the mechanism in feedbax, relocate the rlrmp-specific default table to
  rlrmp (registered as this project's own color-fn contribution), and change
  `feedbax/analysis/execution.py:366` to source base color specs from a
  caller-supplied/registered value instead of importing
  `COMMON_COLOR_SPECS` directly — this also fixes the dead `COLOR_FNS`
  extension point by making injection the only path.
- **`feedbax/plot/experiments.py`** — content-wise this module is fine and
  not rlrmp-shaped (rlrmp never even imports it), but the filename
  "experiments.py" misleadingly suggests project-specific code, creating a
  naming-driven false-positive smell (this is exactly why Phase 1's catalog
  flagged it `unclear`). Fix: rename to something like
  `feedbax/plot/analysis_helpers.py` (or fold into the existing
  `feedbax.plot` namespace). No consumer migration needed.

**Evidence:** `_artifacts/05883e7/audit/sweeps/reverse_audit/summary.md`
("Generalize in place" table); `_artifacts/05883e7/audit/feedbax_catalog/catalog_index.md`
("project_smell"/"unclear" sections) for the original flagging.

**Suggested destination issue titles:** one issue per module — "Generalize
`analysis/execution.py`'s rlrmp-shaped grouping key and SISU special-case",
"Move `cde_templates.py`'s four bespoke presets out of core builtins",
"Generalize `persistence/database.py`'s rlrmp-shaped ORM columns",
"Relocate `plot/color_setup.py`'s rlrmp-shaped default color table to rlrmp",
"Rename `plot/experiments.py` off its misleading project-specific-sounding name".

## 4. Confirmed-dead feedbax objects from verification

These are feedbax-side objects from the audit's exhaustively-verified
confirmed-dead-deletion population (remediation portfolio class (a) — every
one of these was independently re-derived by Phase 4, not merely classified).

- **The `ALL_MEASURES`/`MEASURE_LABELS` catalog cluster**
  (`feedbax/analysis/aligned.py`, 27 objects, 208 LOC, confidence `high` on
  all 27) — every helper in this cluster exists solely to build entries of a
  measure catalog that a repo-wide grep confirmed has zero consumers; the
  whole cluster is one coherent deletion, not 27 independent ones.
- **`ApplyFunctional`** (`feedbax/analysis/func.py:68-91`, 24 LOC) — a
  registered `AbstractAnalysis` subclass with `usage_status=dead` but
  disposition `needs_decision`, **not** in the confirmed-delete class:
  it is `CallerPorts`'s other concrete consumer alongside `Jacobians`/
  `Hessians` (both live), so before deleting, confirm no downstream recipe
  actually constructs it — the classification record itself flags this
  distinction (medium confidence, "confirm downstream recipe usage before
  deleting").
- **Orphaned module-level loggers** — a recurring one-line pattern: a
  `logger = logging.getLogger(__name__)` defined but never called anywhere
  in its module. Found (each independently, each -1 LOC unless noted)
  across `feedbax/mechanics/{skeleton/arm,skeleton/arm_dae,skeleton/pointmass,
  skeleton/pointmass_dae,skeleton/skeleton,geometry,hill_muscles,
  templates/arm_6muscle}.py`, `feedbax/acausal/system.py`,
  `feedbax/xabdeef/losses.py` (part of a 2-object, 46 LOC delete),
  `feedbax/runtime/noise.py` (part of a 23 LOC delete), and
  `feedbax/plot/mpl.py` (one of its 29 confirmed-dead objects). Not urgent
  individually, but worth a single sweep-and-delete pass across the
  `mechanics/skeleton/` family given how many independent instances exist.
- **`feedbax/config/tree.py`**'s YAML-diffing helper family (16 objects, 228
  LOC) — an array-aware equality helper and a "extract varying leaves across
  pytrees for YAML-diffing" utility with a docstring and example but zero
  call sites; a review doc independently flagged part of this cluster as
  untested.
- **`feedbax/bin/_orchestrate.py`** (whole module, 5 objects, 55 LOC) — a
  speculative single/batch orchestration helper written to unify
  `feedbax.bin.analysis`/`train` loops but never adopted by either.
- **`feedbax/mechanics/muscle.py`**'s legacy "Virtual Muscle" model cluster
  (14 objects, 142 LOC) — an entirely separate, unused model from the live
  `hill_muscles.py` module despite the similar "muscle"/"Hill" naming; worth
  flagging explicitly so a future reader doesn't confuse the two.
- **`feedbax/training/rl/tasks.py:target_at_t`'s deprecated predecessor** (2
  objects, 128 LOC) — self-documented deprecated, its replacement
  (`sample_task_params_jax` + `target_at_t`) is already the production path.

**Evidence:** `_artifacts/05883e7/audit/synthesis/portfolio.json`'s
`a_confirmed_dead_deletions` array (feedbax-repo entries); `ApplyFunctional`
specifically in `_artifacts/05883e7/audit/classification/chunk_0010.jsonl`.

**Suggested destination issue titles:** "Delete the dead `ALL_MEASURES`
catalog cluster in `analysis/aligned.py`", "Confirm/delete `ApplyFunctional`
in `analysis/func.py`", "Sweep dead module-level loggers across
`mechanics/skeleton/`", "Delete dead YAML-diff helpers in `config/tree.py`",
"Delete unadopted `bin/_orchestrate.py`", "Delete the dead legacy Virtual
Muscle model in `mechanics/muscle.py`".

## 5. Notable catalog anomalies (structural, not dead-code)

Cross-cutting anomalies Phase 1's catalog fan-out flagged that are not
usage-status findings but are worth feedbax's own attention:

- **Private cross-module imports in `acausal/`.** `feedbax/acausal/assembly.py`
  imports private helpers (`_resolve`, `_topo_sort_through_eqs`) directly
  from `feedbax/acausal/analysis.py` — leading-underscore names used as a de
  facto internal API across a module boundary rather than through a public
  surface. `acausal/analysis.py` itself defines `_resolve`, `_build_networks`,
  `_through_equation_cycle`, `_topo_sort_through_eqs` specifically for this
  cross-module use. Not a bug, but a maintainability smell worth a small
  follow-up (either promote these to a documented internal API or fold
  `assembly.py`'s consumption into `analysis.py` directly).
- **Never-stored constructor arguments in `acausal/multibody.py`.**
  `RevoluteJoint`'s `parent_frame`/`child_frame` arguments, `Anchor`'s
  `frame`/`world_frame` arguments, and `PointMarker`'s `frame`/`offset`/
  `marker_name` arguments are all accepted by their constructors but never
  assigned to any attribute or param, and none of these classes define any
  equations. Per the module's own docstring, these are placeholder
  constructors awaiting a not-yet-built planar-multibody graph compiler —
  not obviously bugs, but incomplete wiring worth tracking so it isn't
  mistaken for finished functionality by a future reader.
- **The Studio-DB-to-manifest migration seam in `analysis/context.py`.**
  `record_figure` currently dual-writes: it writes to both the legacy
  SQLAlchemy `db_session`/`eval_info`/`model_info` path (see item 2 above)
  and the newer manifest-based artifact system. This reads as an
  in-progress, never-finished migration rather than a single coherent
  design — likely the right fix is to land it in the same wave as item 2's
  sqlite-layer retirement, once that decision is made, so `record_figure`
  writes only to the manifest system. Separately, `_route_figure_projection`
  does a local (function-body) import of `feedbax.plot.io.save_figure` with
  a `# noqa: PLC0415` suppression — likely avoiding a circular import; worth
  checking whether `plot.io` could import context-safe types at module load
  instead, to remove the suppression.

**Evidence:** `_artifacts/05883e7/audit/feedbax_catalog/catalog_merged.jsonl`
(per-module `anomalies` field for `acausal/analysis.py`, `acausal/assembly.py`,
`acausal/multibody.py`, `analysis/context.py`).

**Suggested destination issue titles:** "Clean up private cross-module
imports between `acausal/analysis.py` and `acausal/assembly.py`", "Document
or wire the never-stored constructor args in `acausal/multibody.py`'s
placeholder elements", "Finish the Studio DB-to-manifest migration in
`analysis/context.py`'s `record_figure`".

## 6. Small move-to-feedbax / general-belongs-in-feedbax object set

The classification fan-out separately flagged a handful of small,
rlrmp-side objects whose *content* is fully generic — no rlrmp science, no
rlrmp-specific coupling — and which read as reasonable feedbax-primitive
candidates rather than rlrmp-specific code. One record actually carries the
formal `move_to_feedbax` disposition (the only one in the whole corpus, per
`synthesis.md`'s caveat that this single record "was not verified at all" by
Phase 4 — treat it as classification-confidence only); the rest are flagged
`generality: general_belongs_in_feedbax` with dispositions ranging from
`keep` (too trivial to be worth promoting) to `needs_decision` (a genuine
promotion candidate):

- **`rlrmp:results/2ef67ca/scripts/eval_robustness.py:add_mean_std_band`**
  (66 LOC, disposition `move_to_feedbax`) — a generic Plotly mean±std-band
  trace helper with no rlrmp-specific content.
- **`rlrmp:src/rlrmp/train/executor/equivalence.py`**'s whole
  fixed-seed-comparison harness (`LeafDiff`, `EquivalenceTolerance`,
  `EquivalenceReport`, `compare_pytrees`, `run_paired_equivalence` — 5
  objects, ~70 LOC combined) — a generic pytree-equivalence testing utility
  with heavy test adoption (9 call sites) and no rlrmp science content; the
  strongest promotion candidate in this set precisely because it already has
  multiple internal consumers, meaning the promotion is a pure relocation
  with an established API, not a design exercise.
- Smaller, lower-priority candidates: `rlrmp:src/rlrmp/eval/recipes.py:_canonical_json_bytes`
  (generic canonical-JSON serialization, likely duplicates something feedbax
  already has), `rlrmp:src/rlrmp/model/factory.py:_get_or_default` and
  `rlrmp:src/rlrmp/paths.py:mkdir_p` (both `keep`-dispositioned as "too
  trivial to be worth the promotion churn"), `rlrmp:src/rlrmp/runtime/checkpoint_custody.py:{serialize,deserialize}_pytree_slot`
  (generic Equinox pytree round-trip helpers, currently `keep` pending
  confirmation of no existing feedbax counterpart).

**Evidence:** grep `"generality": "general_belongs_in_feedbax"` across
`_artifacts/05883e7/audit/classification/chunk_*.jsonl`; the single
`move_to_feedbax` disposition record is in `chunk_0167.jsonl`.

**Suggested destination issue titles:** "Evaluate promoting rlrmp's
pytree-equivalence testing harness (`train/executor/equivalence.py`) to
feedbax", "Evaluate promoting `add_mean_std_band` (currently
`results/2ef67ca/scripts/eval_robustness.py`) to `feedbax.plot`" — the
remaining small candidates are lower priority and can be batched into
whichever of these two issues lands first, or left as a `deferred` issue if
neither is picked up soon.
