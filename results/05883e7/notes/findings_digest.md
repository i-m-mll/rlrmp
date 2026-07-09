# Findings digest: code-archaeology audit (issue `05883e7`)

This is the plain-language review surface for the whole audit. It is meant to
let a reader walk through every lane — what was measured, what the headline
numbers mean, and the most consequential concrete findings — without needing
the conversation that produced them. Each section names the actual modules
and objects involved and points into the underlying corpus for drill-down.
Companion documents: `results/05883e7/notes/remediation_portfolio.md` (the
ranked cleanup plan, out of scope for this audit itself),
`results/05883e7/notes/ci_gate_proposals.md` (proposed structural guards), and
`results/05883e7/notes/feedbax_backlog.md` (the subset of findings that belong
in the feedbax repo).

A note on vocabulary before the numbers: this audit repeatedly needs to say
whether a piece of code is actually *used*. "Live" means something in the
current codebase calls or imports it. "Dead" means nothing does, as far as the
audit's tools could tell — which, as several sections below show, is not
always the same as nothing *actually* does. "Registered" means the object is
wired into one of the project's plugin-style registries (a way of looking
something up by name at runtime, e.g. `component_registry` or an analysis
recipe registry) rather than being called directly by name in Python.

## 0. Census and cross-reference skeleton

The audit starts from an automated pass (an "AST scan", meaning it reads
Python source structurally rather than running it) over both repositories'
source trees. For every module and every top-level function, class, or
constant, it records identity, line count, imports, and a best-effort map of
who calls or imports what (the "cross-reference index", or "xref").

**Scale.** The full corpus spans 393,139 lines of code across 883 modules and
14,479 objects (14,474 unique — the census's own chunk plan has 5 duplicate
entries, a pre-existing quirk documented and worked around, not fixed, in the
classification fix-up notes):

| repo | tree | modules | LOC | objects |
|---|---|---:|---:|---:|
| rlrmp | `src/` | 148 | 115,185 | 4,328 |
| feedbax | `package/` | 302 | 112,166 | 3,859 |
| rlrmp | `results/*/scripts/` | 98 | 58,172 | 2,284 |
| rlrmp | `tests/` | 139 | 47,956 | 1,784 |
| feedbax | `tests/` | 132 | 45,888 | 1,726 |
| rlrmp | `scripts/` | 57 | 11,592 | 422 |
| feedbax | `scripts/` | 6 | 1,943 | 66 |
| feedbax | `examples/` | 1 | 237 | 10 |

Object kinds: 9,589 functions, 3,057 module-level constants, 1,737 classes,
96 async functions. The largest modules by line count are
`src/rlrmp/train/cs_nominal_gru.py` (8,902 LOC) and its test file
`tests/test_cs_nominal_gru.py` (7,618 LOC), followed by
`src/rlrmp/train/cs_perturbation_training.py` (7,116 LOC) — `cs_nominal_gru.py`
is exactly the ~8,900-line module the audit was filed to investigate (see the
umbrella issue body).

**What the xref can and can't tell you.** The cross-reference pass resolves
ordinary imports (`import x`, `from x import y`, relative imports) but *not*
star imports, dynamic `importlib`/`getattr` access, or a name re-exported a
second time through another module. Two derived signals are used heavily
downstream and both are noisy by design:

- `unresolved_name_hits` — a bare-name match with no resolved import route.
  Cheap to compute, but a function called `loss` or `weight` will pick up
  unrelated same-named locals; it is not proof of use.
- `string_hits` — how many times an object's bare name appears inside string
  literals corpus-wide. This is the signal that catches registry keys (an
  object referenced by name rather than imported), but it is equally
  collision-prone for short names. Treat `string_hits > 0` as "worth checking
  for registry wiring," not as confirmation.

Given those limits, **2,947 of 13,328 objects (22.1%) have zero inbound
references of any kind and are not registered or `__all__`-listed** — this is
a *candidate*-dead signal, not a verdict; Section 4 below shows concretely why
that distinction matters. Separately, the census found 37 modules carrying a
`LEGACY (frozen ..., issue ...)` banner (see `CLAUDE.md`'s LEGACY-banner
convention) and 1,316 objects with at least one hyperparameter-like signal
(a dict key or kwarg name that looks like a hyperparameter, or a numeric
literal container with 4+ entries) — the raw material for the
hyperparameter-in-code findings in Sections 3 and the remediation portfolio.

Drill-down: `_artifacts/05883e7/audit/census/summary.md`,
`census/modules.jsonl`, `census/objects.jsonl`, `census/xref.jsonl`.

## 1. Feedbax capability catalog

Before anything in rlrmp's tree can be judged "this duplicates feedbax" or
"this is general and belongs in feedbax," someone has to know what feedbax
actually offers. Phase 1 produced a structured catalog of all 302 modules
under `feedbax/feedbax/`, tagging each with a one-line summary, its public
objects, and a judgment of whether its content reads as general framework
infrastructure (`framework_general`) or as something shaped by one specific
downstream project's needs (`project_smell` or `unclear`).

**Coverage was checked against the filesystem, not trusted from the fan-out.**
The catalog states its own coverage audit up front: a plain filesystem walk of
`feedbax/feedbax` found 302 modules; the merged catalog covers all 302 with no
missing and no extra relpaths. This is the standing lesson worth generalizing:
an agent transcribing a large tree into a structured catalog can silently drop
entries, and the only way to know it didn't is to check the transcription
against an independent, deterministic ground truth (here, a directory walk)
rather than trust the fan-out's own bookkeeping. The mechanism visibly still
works in this run — the catalog's chunk-file sequence jumps from
`chunk_58.jsonl` to `chunk_60.jsonl` (`chunk_59.jsonl` is simply absent from
disk), and the coverage check is precisely what confirms that gap didn't cost
any module content: all 302 modules are present across the 61 files that do
exist, so nothing was lost, but the anomaly is flagged as a residual
generation-process question worth checking if the catalog is regenerated.

**34 of 302 modules (11%) are not `framework_general`** — 13 flagged
`project_smell`, 21 `unclear`. These are concrete, named findings, not a
vague "some code is coupled" impression:

- `feedbax.analysis.aligned` and `feedbax.analysis.state_utils` hardcode the
  `'train__pert__std'` LDict grouping key (rlrmp's training-sweep naming
  convention) and reach-specific geometry assumptions directly in what reads
  as general framework analysis code.
- `feedbax.analysis.execution` hardcodes the same `'train__pert__std'`
  grouping key and an explicit `sisu` special case (SISU is an rlrmp
  training-method concept, tracked on rlrmp's `c99ad9d` training-methods
  coordination issue) inside `setup_eval_for_module`.
- `feedbax.component_registry.cde_templates` ships four hardcoded,
  versioned architecture presets ("Anti-NF", "CDE Hybrid v9b") as core
  framework-default registry content rather than example/demo material.
- `feedbax.config.defaults` bakes reach-task-specific evaluation presets
  (`TASK_EVAL_PARAMS`, `POS_ENDPOINTS_ALIGNED`, `EVAL_REACH_LENGTH`) into a
  generically named `feedbax.config` module.
- `feedbax.models.networks` carries an `sisu_gating`/`sisu_alpha` feature
  (rlrmp's SISU input-gain modulation) baked into an otherwise-general
  architecture module.
- `feedbax.persistence.database` declares ORM columns
  (`perturbation_config`, `sisu_params`, `pert__type`, `pert__std`) tied to
  one project's reach/perturbation study, and its own docstring calls itself
  "this legacy SQLAlchemy database."
- `feedbax.plot.color_setup`'s `COLORSCALES`/`COMMON_COLOR_SPECS` tables are
  populated exclusively with rlrmp's hyperparameter naming scheme
  (`pert__amp`, `train__pert__std`, `sisu`, `reach_condition`).
- `feedbax.training.rl.rewards` and `feedbax.training.rl.tasks` hand-tune
  reward/task constants for one reach/hold/track/swing experiment paradigm
  (citing "Klar et al." in comments).

189 of 302 modules (63%) carry at least one recorded anomaly of some kind
(dead code, fragile control flow, naming/indirection traps); Section 4's
feedbax backlog picks out the cross-cutting ones worth filing.

Drill-down: `_artifacts/05883e7/audit/feedbax_catalog/catalog_index.md`,
`catalog_merged.jsonl`.

## 2. Classification fan-out

Phase 2 assigned every one of the 14,474 unique object ids (plus 2
legitimately-emitted extra module records, i.e. 14,476 total) to one of 375
chunks, each independently classified by a Sonnet agent against a fixed
record schema (purpose, generality, usage status, disposition — never a
free-form label). Two things happened here worth surfacing on their own:

### The corpus needed a repair pass

Four defects were found and fixed after the fan-out, recorded durably in
`_artifacts/05883e7/audit/classification/corpus_notes.md`:

1. **`chunk_0228.jsonl` (36 objects) did not exist on disk at all.** A prior
   agent had written its working data to three misnamed sibling files instead
   of the canonical chunk file, and the two draft passes disagreed on every
   single record. The chunk (all objects in
   `src/rlrmp/analysis/math/hinf_riccati.py` and
   `src/rlrmp/analysis/math/induced_gain.py`) was rebuilt from scratch against
   the authoritative census/xref data and the actual source, and the stray
   files were quarantined rather than trusted.
2. Two chunk files had duplicated id lines from a prior run (`chunk_0024.jsonl`,
   `chunk_0025.jsonl`); the duplicates were dropped, keeping first occurrence.
3. Five stray scratch/working files were quarantined.
4. Two module-level records beyond `chunk_plan.json`'s listed scope were
   confirmed legitimate and kept.

### The Sonnet-vs-spark model-comparison episode

A separate methodology check re-ran 5 already-classified chunks (200 objects,
all feedbax by chance of the draw) through `gpt-5.3-codex-spark` via `codex
exec`, using the same task prompt, to see whether a cheaper/faster model could
reproduce Sonnet's classification judgments. The comparison is documented in
full at `_artifacts/05883e7/audit/spark_comparison/report.md`; the headline
numbers:

| field | exact agreement | Cohen's kappa |
|---|---:|---:|
| purpose | 77.5% | 0.729 |
| generality | 95.5% | — |
| usage_status | 47.0% | 0.119 |
| disposition | 90.0% | 0.041 |

("Cohen's kappa" here is agreement corrected for how often two raters would
match by chance alone — it is the number to trust over raw percent agreement
when one label dominates the data, which is exactly what happened on
`disposition`: 90% raw agreement looks good, but the kappa of 0.04 reveals
that's almost entirely two models saying "keep" by default, not shared
judgment.)

The consequential finding: on the **104 deletion-enabling rows** (either model
said `usage_status=dead` or `disposition` in `{delete, merge_dedupe}`),
**full agreement between the two models was 0 out of 104.** Spark produced
**zero** `delete` or `merge_dedupe` dispositions across all 200 objects it
classified, versus Sonnet's 10. A qualitative spot-check of 10 disagreements
against the actual source and call sites found spark wrong on all 10, with 8
of the 10 sharing one systematic failure mode: spark applied the audit's
"dead" rule (zero *routed inbound references from outside the object's own
module*) literally to private helper functions that are visibly called
*within* their own module — labeling them `dead` even while spark's own
disposition notes described their call sites, and even while pairing that
`dead` label with `keep` (an internally incoherent combination Sonnet never
produced). This exposed a literal-reading hazard in the dead-rule wording
itself, not just a spark-specific weakness — the prompt permitted a reading
that a careful-but-literal reader could follow into wrong answers. The
practical guardrail this motivated: fix the dead-rule wording to require
counting intra-module references, and route every `dead`/`delete`/
`merge_dedupe` verdict (including a model's *absence* of any) through
independent verification rather than trusting a single pass. As a result of
this finding, the classification prompt was corrected mid-run: chunks 1-173
(referred to as "V1" throughout this audit) used the original wording, chunks
174-375 ("V2") used the corrected wording. Section 4 shows the corrected
wording did not eliminate the underlying gap, but the QA calibration below
does show the two populations converge once independently re-classified.

## 3. Cross-cutting sweeps

Three sweeps ran once every object had an id and a classification.

### Duplication

Near-duplicate clustering over 11,572 candidate objects (functions, classes,
and constants of at least 3 lines) found **978 clusters totalling an
estimated 37,189 redundant lines of code** (the sum, per cluster, of total
member LOC minus the largest member — a proxy for "LOC removable by
deduplicating down to one canonical implementation"). Two hot zones stand out:

- **`src/rlrmp/analysis/pipelines/`**: 115 clusters, 5,017 redundant LOC.
- **`results/*/scripts/`**: 262 clusters, 14,174 redundant LOC — the largest
  single contributor, and expected: this tree is 97+ experiment-specific
  scripts across many tracking issues, exactly the kind of tree where
  boilerplate that should have been promoted to `src/rlrmp/` accumulates
  instead.

Concrete named examples: `write_outputs` (a driver function) is copy-pasted
structurally across roughly 10 `src/rlrmp/analysis/math/*.py` modules even
though the original is individually LEGACY-banned as "not to copy"
(`adversary_equivalence.py`, frozen 2026-07-03, issue `64d5f13`);
`cs_perturbation_training.py` duplicates most of its own PGD (projected
gradient descent) inner-maximizer logic between
`run_broad_epsilon_pgd_inner_maximizer` and
`_run_finite_broad_epsilon_pgd_inner_maximizer` in the same file; the legacy
`results/2ef67ca/scripts/eval_robustness.py:eval_fixed_pert` and
`scripts/eval_diagnostics.py:_eval_at_pert` both hand-roll logic that
`rlrmp.eval.pert.eval_at_pert_scale` already provides as the canonical
current helper. Not every cluster is a defect: feedbax's family of Equinox
normalization-layer wrappers (`BatchNorm`, `LayerNorm`, `GroupNorm`, ...) and
its `ConvTranspose{1,2,3}d` family cluster by design — expected
framework-family shape, not duplication to fix.

### Dangling references

A sweep for paths, imports, and doc references that no longer resolve to
anything on disk found 1,494 findings across 883 modules: 1,322 stale path
literals, 68 template-like path literals (excluded from the stale count
because they contain wildcards), 55 dangling import names, 29 stale doc
references, 20 unresolved third-party imports. Most of the volume is
low-confidence by the sweep's own documented limitations (comments are noisy;
`_artifacts/` references are often just "hasn't run yet," not stale).

**One finding is a real, confirmed bug in feedbax**: `feedbax/analysis/support.py:122`
does `from feedbax.intervene import AbstractIntervenor`, but `AbstractIntervenor`
is not actually exported from the `feedbax.intervene` package facade (flagged
`dangling_import_name`, confidence `medium` — the sweep's own spot-check
confirmed the line reads exactly as flagged). See
`results/05883e7/notes/feedbax_backlog.md` for the filing detail.

Separately, **36 references still point at the pre-refactor flat
`scripts/<name>.py` layout** that CLAUDE.md's script-placement policy (Bug
`8404108`) moved to `results/<hash>/scripts/<name>.py`. These are mostly
stale module docstrings in the very files that moved (e.g.
`results/2ef67ca/scripts/eval_robustness.py`'s own docstring still says
`"""scripts/eval_robustness.py`), plus test literals in
`tests/test_reaccretion_ratchet.py` referencing a `scripts/train_existing.py`
that no longer exists at that path. Low-risk, high-volume cleanup: harmless
stale prose, not broken functionality, but worth batching into the docstring
housekeeping pass.

### Feedbax reverse audit

The mirror image of "does rlrmp duplicate something general in feedbax?" is
"does feedbax host something only rlrmp actually uses?" All 34 modules
Phase 1 flagged non-`framework_general` were individually checked against
real import/consumer evidence:

| Recommendation | Count | Examples |
|---|---:|---|
| move_to_rlrmp | 1 | `feedbax.analysis.network` (zero current consumers anywhere) |
| generalize_in_place | 5 | `analysis.execution`, `component_registry.cde_templates`, `persistence.database`, `plot.color_setup`, `plot.experiments` |
| keep_as_is | 21 | catalog flags were false positives once real consumer evidence was checked |
| deprecate_delete | 7 | `analysis.effector`, `analysis.profiles`, `analysis.setup`, `bin.db_merge`, `config.defaults`, `plot.mpl`, `web.ws.simulation` |

The **deprecate_delete cluster is one coherent finding, not seven unrelated
ones**: `analysis.setup`'s only live call chain is a legacy SQLAlchemy
`db_session` model-loading path that rlrmp no longer touches (no `.db`
artifacts, no `persistence.database` imports anywhere in current rlrmp
source); `bin.db_merge` exists solely to merge those same sqlite databases;
`analysis.effector` and `analysis.profiles` are zero-consumer plotting
helpers from the same era. This is a single "orphaned Studio-era analysis and
persistence layer" that should retire together, not piecemeal. `plot.mpl` is
a separate finding (superseded wholesale by the plotly-based `feedbax.plot`
package) but is folded into the same remediation class because it shares the
"zero consumers, safe as a batch" shape. See
`results/05883e7/notes/feedbax_backlog.md` for the full filing detail on all
seven, and `results/05883e7/notes/remediation_portfolio.md` class (b) for the
LOC accounting.

## 4. Verification and QA

Phase 4 independently re-derived the usage/disposition calls for the
highest-stakes population — every record whose original disposition was
`delete`, `move_to_feedbax`, or `replace_with_declarative_surface` — plus a
calibration sample of everything else, **without seeing the original
record's reasoning**. 888 records were checked this way.

| Verdict | Count | Share |
|---|---:|---:|
| confirmed | 866 | 97.5% |
| refuted | 21 | 2.4% |
| uncertain | 1 | 0.1% |

Both scan populations (V1 = chunks ≤173, the original dead-rule wording; V2 =
chunks >173, the corrected wording) landed at essentially the same refutation
rate (V1 2.4%, V2 2.3%) — the mid-run prompt fix did not obviously change the
error rate on this particular sample, though the QA calibration below tells a
more specific story about *where* the two populations differ.

### A cross-language finding worth generalizing

One of the 21 refutations is a standing lesson on its own.
`feedbax.analysis.aligned.get_trivial_reach_directions` was originally called
`dead`/`needs_decision` on the grounds that it was "never passed as
`directions_fn` or otherwise referenced anywhere." That claim was factually
wrong: Feedbax Studio's TypeScript frontend data
(`feedbax/web/src/data/rlrmp-part2.ts:1104`) references the exact function
name as a `directions_fn` parameter value, and a TypeScript test
(`rlrmp-part2.test.ts:527`) asserts on it directly. But — and this is why the
corrected call is `dynamic_reference_unconfirmed_wiring`, not `live` — no
Python-side mechanism that resolves that TypeScript string back to the actual
Python function was found either. **The standing lesson: rlrmp/feedbax's
Python-side static xref cannot see Studio's TypeScript frontend at all, so a
Python-only "zero references" claim about anything that might be reachable
from Studio's graph-spec data needs an explicit TypeScript-side check before
it can be trusted as dead — and even after finding the TS-side reference, the
Python-side resolution question stays genuinely open, which is why this
stays `needs_decision` rather than flipping straight to `live`.**

A related but distinct dynamic-access miss: `feedbax.plot.colors.adjust_color_brightness`
was called `dead`/`delete` but is actually invoked as
`fbp.adjust_color_brightness(...)` through a re-exported package namespace —
an attribute lookup the static import-resolution pass does not chase. Between
these two and `feedbax.tasks.task.safe_state_set` (refuted from
`registry_or_string_referenced`/`delete` to `live_public_api`/`keep` on
public-export-and-test evidence), three of the 21 refutations trace to the
same underlying gap: **the census's import resolution is real but bounded,
and anything reached through a namespace re-export, a registry lookup, or a
non-Python caller will look "dead" to it even when it demonstrably isn't.**

### Confirmed-deletion LOC is the most trustworthy number in this audit

Among the sampled records whose original disposition was `delete`, 286 were
confirmed correct, covering 5,385 LOC — and per the synthesis's full
(non-sampled) accounting, **all 287 `delete`-disposition records in the
entire corpus were checked, not just this sample** (2 were refuted into
`keep`, 1 `needs_decision` record was promoted into `delete`), landing the
final exhaustively-verified confirmed-dead-deletion total at 5,401 LOC across
287 objects / 107 modules. This is the one LOC figure in the whole audit that
is fully verification-gated end to end, not classification-confidence-only.

### QA calibration: is the classification scheme itself reliable?

A second independent reviewer ("QA") re-classified an 800-record sample from
scratch, blind to the original calls, and was scored against those original
calls as an answer key:

| Field | Agreement | Cohen's kappa |
|---|---:|---:|
| purpose | 74.1% | 0.706 |
| generality | 98.6% | — |
| usage_status | 83.2% | 0.075 |
| disposition | 97.0% | 0.340 |
| confidence | 88.0% | — |

`usage_status` — the field this whole audit leans on most heavily for
dead/legacy/live counts — has the **lowest** Cohen's kappa of any field
(0.075), meaning its category *boundaries* are the least reliably applied
part of the scheme even though 83.2% raw agreement looks respectable. The
directional finding (rlrmp's `src/` is mostly live code, dead code is a small
minority) is robust to this noise; the exact dead/legacy/live *counts* should
be read with that caveat attached.

The QA pass also settled a specific worry: the original census found roughly
a 4.1% dead-rate in V1 records versus 1.2% in V2 — was that a real difference
in the two code populations, or an artifact of the prompt fix made partway
through? Running one uniform classification procedure across both populations
via QA **collapsed the gap to 0.2%** (V1 0.5% dead-rate, V2 0.2%, versus an
answer-key dead-rate of 0.0% in the sampled subset for both). **The original
V1/V2 dead-rate gap was mostly a scan-procedure artifact, not a real
difference between the two halves of the corpus** — dead code did not
actually concentrate in the earlier-scanned half of the codebase, the earlier
scan procedure was just more prone to calling things dead. Combined with the
spark episode's finding of ~0.25–0.5% systematic dead-under-reporting in both
prompt populations, the practical read is: the classification corpus's raw
`dead` counts likely understate true liveness by a small amount uniformly
across the whole scan, not in one lopsided half of it.

Drill-down: `_artifacts/05883e7/audit/verification/summary.md`,
`calibration.md`, `verdicts_merged.jsonl`.

## 5. Synthesis headlines

The deterministic Phase 5A crunch (`results/05883e7/scripts/synthesize.py`,
stdlib-only, no LLM calls, rerunnable) over all 14,474 classified objects
after Phase 4 correction answers the question the audit was filed to answer.

**Purpose composition.** 103,215 of `src/rlrmp/`'s 115,185 lines (90%) sit
inside a classifiable object. Spec/manifest construction (19.9%), analysis
transforms (17.3%), and core math/algorithms (16.0%) together account for
53% of that — exactly the shape the project's own implementation policy says
`src/` should have. **The audit was filed because of specific symptoms (one
~8,900-line training module, issue-hash-named run planners baked into
`src/`), not because the tree's overall composition looked wrong — and the
composition table confirms that framing: the tree is broadly healthy, with
identifiable pockets of misplacement rather than a systemic problem.**

**Usage status, corpus-wide, post-correction:**

| usage_status | LOC | Share |
|---|---:|---:|
| live | 265,847 | 78.2% |
| test_only | 48,954 | 14.4% |
| legacy_only | 10,979 | 3.2% |
| dead | 5,954 | 1.8% |
| registry_or_string_referenced | 4,809 | 1.4% |
| ambiguous | 3,223 | 0.9% |

Within rlrmp `src/` specifically, `dead` is 1.4% (1,405 of 103,215 LOC) — a
small absolute number, reassuring on its own terms, and (per Section 4) if
anything a slight overcount relative to true liveness.

**The remediation portfolio in one table** (full detail and per-item
breakdown in `results/05883e7/notes/remediation_portfolio.md`):

| Class | Items | LOC | Confidence basis |
|---|---|---:|---|
| (a) Confirmed-dead deletions | 107 modules / 287 objects | -5,401 | Exhaustively verification-gated |
| (b) Legacy-tree retirements | 40 items | -13,459 | `legacy_only` clusters + feedbax reverse-audit deprecations |
| (c) Dedupe/promotion (top clusters) | 79 of 978 clusters | -15,478 | Sweep-heuristic; top slice of 37,189 total redundant LOC |
| (d) Move to `results/*/scripts/` | 10 modules / 79 objects | 1,691 relocated | Not independently verified |
| (e) Contract-violation remediation | 110 modules / 362 objects | 11,889 relocated/migrated | Not independently verified |
| (f) Feedbax generalize-in-place | 5 feedbax modules | n/a (refactor) | Reverse-audit judgment |
| (g) Dangling-reference fixes | 89 modules / 278 findings | n/a (correctness) | High/medium-confidence findings only, of 1,494 total |
| needs_decision (unresolved) | 602 objects | 18,467 | Deliberately left open |

**Module structure.** `src/rlrmp/train/cs_nominal_gru.py` tops the SPLIT
ranking (which scores modules by LOC × purpose-entropy × distinct contract
flags) at a score of 121,088 — nearly double the runner-up
(`cs_perturbation_training.py`, 63,462) — carrying 5 distinct contract flags
including 5 `data_in_code` hits (793 LOC) and 5 `experiment_named_in_src`
hits (358 LOC). This is exactly the module named in the audit's motivating
symptoms. Full ranking: `results/05883e7/notes/module_structure.md`.

## Standing lessons

Four patterns recurred across independent parts of this audit strongly enough
to generalize as working rules for future archaeology/classification work in
this codebase:

1. **Validate agent-produced transcriptions against deterministic ground
   truth.** The feedbax capability catalog's own coverage audit (a plain
   filesystem walk compared against the merged catalog) is what confirms
   302/302 module coverage and what caught the `chunk_59.jsonl` naming
   anomaly before it could silently drop content. Any large agent-fan-out
   transcription task over a fixed corpus should ship this kind of
   independent completeness check, not just trust the fan-out's own count.

2. **Exclude nested checkouts and virtualenvs from repo-wide greps.**
   Verification evidence throughout `verdicts_merged.jsonl` explicitly scopes
   greps to exclude `worktrees/` and `.venv/` (e.g. "`rg ... across feedbax
   (excl worktrees/.venv)`"); a grep that doesn't exclude these will find
   false "live" hits inside old worktree copies or installed packages that
   have nothing to do with the current tree's actual call graph.

3. **Python-side static analysis cannot see cross-language references.**
   Feedbax Studio's TypeScript frontend (`feedbax/web/src/`) references
   Python object names as data (e.g. `directions_fn: 'get_trivial_reach_directions'`).
   A Python-only xref pass will call such an object dead with no way to know
   otherwise; any "confirmed dead" claim about analysis code reachable from
   Studio's graph-spec surface needs an explicit TypeScript-side check.

4. **Registry-wired and dynamically-accessed liveness needs signals beyond
   routed inbound references.** `component_registry`, recipe registries, and
   namespace re-exports (`import feedbax.plot as fbp; fbp.some_fn(...)`) all
   produce objects that are genuinely live but show zero *routed* inbound
   references in a static xref pass. The corpus's own `registered` and
   `string_hits` fields exist precisely to catch this, and three of the 21
   Phase 4 refutations trace directly to a verifier or classifier ignoring
   one of those signals in favor of the routed-inbound-only reading — most
   sharply illustrated by the spark comparison (Section 2), where a model
   applying the literal "zero routed inbound references" rule to
   intra-module private helpers produced systematically wrong `dead` labels
   at high stated confidence, on code whose own call sites it could see.
