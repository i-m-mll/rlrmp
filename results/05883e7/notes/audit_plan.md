# Audit plan: code archaeology of rlrmp and feedbax (issue `05883e7`)

This note is the phase plan for a future reader picking up any phase without
the conversation that produced it. It complements the umbrella issue body
(motivating question, scope, execution model) and the record schema in
`record_schema.md`.

## Why this audit exists

rlrmp's `src/` (~115k lines, 148 modules) is close in size to the feedbax
framework it depends on (~112k lines, 301 modules). The project's own
implementation policy says general code belongs in feedbax and only
genuinely rlrmp-specific scientific components should stay here — but nobody
has ever checked the tree against that policy exhaustively. The existing
`feedbax_contract` CI gate family (see repo `CLAUDE.md`) checks that code
behaves correctly at boundaries it already crosses (registration, custody,
import routes) — it does not, and structurally cannot, ask whether a given
object should exist, in its current location, in its current form. Concrete
symptoms that triggered filing this (see issue `05883e7` body) include
issue-hash-named run-row planners and default-hyperparameter bundles baked
into `src/rlrmp/train/cs_nominal_gru.py` (a single ~8,900-line module), and
objects such as `LaunchContinuation` that read as feedbax-general capabilities
sitting in the rlrmp tree instead.

## Phase sequence

### Phase 0 — deterministic census + cross-reference skeleton (this phase)

An AST + git census over both repos producing, for every module and every
top-level object (function, async function, class, module-level constant):
identity, LOC, imports, best-effort outbound references, literal statistics
(numeric-literal density, largest numeric container, hyperparameter-like key
names, path-like string literals), registration signals, and git provenance
(introducing commit, last-touch commit, all `Mandible-Issue:` trailers seen in
the file's history). Alongside this, a whole-corpus cross-reference index
(`xref.jsonl`) resolves imports where possible and counts inbound references
per (repo, tree), separating routed (import-resolved) hits from bare
unresolved name matches and from string-literal hits (registry keys / recipe
IDs that reference an object by name without importing it). This makes
`usage_status` in Phase 2 a computed fact: an object with zero routed inbound
references, zero string hits, no `__all__` membership, and no registration
decorator is a strong `dead` candidate; one with `registered=true` or nonzero
`string_hits` cannot be waved off as dead even if no import route resolves to
it.

Deliverables: `scripts/census.py` (this note's sibling), and its output corpus
under `_artifacts/05883e7/audit/census/` — `modules.jsonl`, `objects.jsonl`,
`xref.jsonl`, `chunk_plan.json`, `summary.md`. The corpus is bulk/regenerable;
only the script and this plan are tracked.

**Known xref-resolution limitations for later phases to account for** (see the
final Phase-0 report for the concrete numbers from the run that produced the
corpus in this repo):

- Import resolution handles `import x`, `import x as y`, `from x import y`,
  `from x import y as z`, and relative imports (`from . import y`,
  `from .x import y`), resolved against the census's own module-dotted-name
  table. It does **not** resolve star imports (`from x import *`), dynamic
  `importlib`/`getattr`-based access, or names re-exported through an
  `__init__.py` under a different binding than their defining module (the
  re-export itself is a resolvable import, but a *second* re-export hop is not
  chased beyond one level in this pass).
- `unresolved_name_hits` (bare-name matches with no resolved import route) are
  a noisy, over-inclusive signal for common short identifiers (e.g. a
  function called `loss` or `weight` will pick up unrelated same-named
  locals). Phase 2/4 agents must sanity-check any dead/duplicate verdict that
  leans on `unresolved_name_hits` rather than routed inbound refs,
  `registered`, or `string_hits`.
- `string_hits` is a corpus-wide substring count of the object's bare name
  inside string literals; it is name-collision-prone for short/generic names
  and does not distinguish "this string is a registry key naming this exact
  object" from "this string happens to contain the same characters."
  Treat `string_hits > 0` as "investigate for registry wiring," not proof.
- Per-object git provenance (`last-touch`) is computed by aggregating a
  single `git blame --porcelain` pass per file over the object's line span;
  it reflects the most recent line-level edit in that span, which can be a
  drive-by formatting change rather than a substantive rewrite.
- The census does not execute or import any code — it is a pure static
  (AST + git) pass. Anything reachable only through metaprogramming,
  monkeypatching, dynamic attribute construction, or plugin-loader string
  dispatch (feedbax's `component_registry`, rlrmp's recipe registries) will
  show up as `registered`/`string_hits` signals rather than resolved import
  routes — Phase 2 classification must read those signals, not just routed
  inbound counts.

### Phase 1 — feedbax capability catalog

Sonnet fan-out over feedbax's `feedbax/` package tree (grouped by the same
kind of chunk plan this phase produces, scoped to feedbax) producing a
structured catalog of public capabilities: what each subpackage
(`training/`, `intervene/`, `objectives/`, `component_registry/`, `plot/`,
`analysis/`, ...) exposes, its registration/spec/extension surfaces (what a
downstream project is supposed to subclass, register, or call vs. what is
feedbax-internal plumbing), and its public API boundary (what `import_boundary`
in `ci/feedbax-contract-suite.toml` currently permits rlrmp to import). This
catalog is the ground truth Phase 2 needs to call something
`duplicates_feedbax` or `general_belongs_in_feedbax` instead of guessing from
a name resemblance.

### Phase 2 — per-object classification fan-out

Sonnet fan-out over `chunk_plan.json` from Phase 0. Each chunk agent emits one
classification record per object id, following `record_schema.md` exactly
(enums verbatim, no free-form substitutes). Purpose and generality draw on the
Phase 1 capability catalog for `generality`/`feedbax_counterpart` judgments.
`usage_status` must cite the actual xref evidence for that object id rather
than re-deriving it. Chunk agents must not invent new enum values; anything
that doesn't fit an enum cleanly is `other` (purpose) or `needs_decision`
(disposition) with a note, never a made-up label.

### Phase 3 — cross-cutting sweeps

Sweeps that only make sense once every object has an id and a home:

- **Near-duplicate clustering**, with two named hot spots to check first:
  `src/rlrmp/analysis/pipelines/` (a family of similarly-shaped pipeline
  modules) and the `results/*/scripts/` tree (97 files, 57k lines of
  experiment-specific scripts across many issues — likely to contain repeated
  boilerplate that should have been promoted to `src/rlrmp/` per the
  script-placement policy in `CLAUDE.md`, and never was).
- **Dangling-reference sweep**: paths, issue IDs, or manifest keys referenced
  in code that no longer resolve to anything on disk or in the ledger.
- **Hyperparameter-literal deep scan**: every object flagged with
  `hp_or_data_constants` purpose or nonzero `hp_like_keys`/dense numeric
  containers, cross-checked against the `generated_data_constant_scan`
  contract gate family, to find hp-in-code that the existing AST lint
  allowlists or doesn't yet cover.
- **Test-to-object pinning map**: which tests actually exercise which src
  objects (via the same import-resolution machinery as Phase 0's xref, scoped
  to `tests/` as consumer), to estimate refactor/deletion cost per object.
- **Feedbax reverse audit**: objects flagged `project_specific_should_leave_feedbax`
  in Phase 1/2 — i.e., single-rlrmp-consumer feedbax capabilities that should
  probably be demoted out of feedbax into rlrmp, the mirror image of
  `general_belongs_in_feedbax`.

### Phase 4 — adversarial verification

Independent refutation pass over every record whose `disposition` is
`delete`, `move_to_feedbax`, or `replace_with_declarative_surface` (the
deletion-enabling claims) — a second agent, without sight of the original
record's reasoning, re-derives usage status and generality from the same
census evidence and either confirms or contests. Contested records go back to
`needs_decision`. In addition, sample ~5-10% of all other records (not just
deletion-enabling ones) for calibration QA, to catch systematic
over/under-confidence in the fan-out rather than only checking the
highest-stakes claims.

Declarative-replaceability claims (`replace_with_declarative_surface`) are
constrained by existing project precedent: `results/96ac0e5/notes/adjudication.md`,
resolved in `results/86e1dd1/notes/post_grammar_adjudication.md`, rejected
every per-target retrofit of in-function Python into feedbax's expression-AST
grammar (`Coalesce`, `Filter`, etc.). Phase 4 verifiers must reject any Phase 2
`replace_with_declarative_surface` disposition that merely proposes rewriting
existing imperative code into an expression AST in place — that migration
path is already adjudicated closed. The only live question here is whether an
object *constructs a governed spec surface* (a manifest, a recipe registration,
a run spec) imperatively in Python where that surface should instead be
*authored* declaratively at recipe/spec-definition time. A disposition that
conflates "uses if/else instead of `Coalesce`" with "hand-builds a spec dict
that should be a declared schema" must be split by the verifier into a
rejected claim and (if applicable) a legitimate one.

### Phase 5 — synthesis

Deterministic, non-fan-out synthesis over the Phase 2 records (post
Phase-4 correction): LOC broken down by purpose × generality × usage_status;
a ranked remediation portfolio (each item: what to do, estimated LOC delta,
risk, which coordination issue or new issue it should graduate to per the
project's issue-coordination rules — remediation work itself is explicitly
out of scope for this audit and graduates to its own issues); a feedbax
feature backlog (objects that should move there); and one proposed CI gate
family per newly-found violation class, following the existing
`feedbax_contract` table shape in `CLAUDE.md` so any adopted gate slots in
next to `analysis_recipe_contract`, `write_surface`, etc. without inventing a
parallel gate system.

## Cross-cutting constraint carried through every phase

Per `CLAUDE.md`'s query-language-adoption section: feedbax expression-grammar
declarations are authoring-time, not retrofit. Every phase that touches a
`replace_with_declarative_surface` judgment (Phase 2 classification, Phase 3
sweeps, Phase 4 verification, Phase 5 synthesis) must respect the
`results/96ac0e5` + `results/86e1dd1` precedent described above rather than
re-litigating it per target.

## Scope boundary

This audit produces a census, a classification corpus, and a synthesis. It
does not itself delete, move, or rewrite any code. Remediation items surfaced
by Phase 5 are handed off as new issues (or comments on the relevant
coordination issue — `4d38c15` for analysis-side findings, `c99ad9d` for
training-method-side findings, `1d9ae6f` for anything that turns out to be a
feedbax/Mandible/dotfiles gap) rather than implemented inline as part of this
umbrella.
