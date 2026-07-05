# Post-grammar re-adjudication of the deferred query-language migrations

**Date:** 2026-07-05 · **Issue:** 86e1dd1 · **Prerequisite:** feedbax expression-grammar
extensions (feedbax issue 51c26e3, merged to develop at `17710f1c`) ·
**Governing principle:** `results/96ac0e5/notes/adjudication.md`

## Outcome

All four deferred migration targets are **rejected**, not migrated. The 96ac0e5
audit classified these sites as "blocked only by missing grammar." With the
grammar now real (`ValueQuery.default`, `Coalesce`, `Filter`,
`startswith`/`endswith` compare ops, optional sources with declared
placeholders, residual passthrough) and the four target files re-read line by
line, none of them satisfies either prong of the governing adjudication
principle:

1. the expression becomes part of a governed spec surface (a document,
   registration, or manifest carries the declaration), or
2. it eliminates cross-file duplication of one selection/gating shape.

What the audit read as "grammar-blocked declarative extraction" is, at the code
level, in-function normalization with computed fallbacks, dynamic keys, and
single call sites. Migrating those sites would wrap local control flow in
expression ASTs — indirection, which the principle rejects by design
("expressible ≠ worth migrating"). This is the anticipated resolution path the
deferral itself named: re-adjudicate once the grammar exists, and let verdicts
flip to rejection where the governed-surface prong fails.

## Per-target verdicts

| Target | Audit classification | Verdict | Decisive grounds |
|---|---|---|---|
| `analysis/matrix/standard_matrix.py` | best whole-file candidate (near-A extraction + fail-closed gates) | **reject** | computed fallbacks, dynamic keys, single call sites; gates belong to the params model |
| `analysis/rollout_cleanup.py` | 4 near-identical resolver chains → one Coalesce shape | **reject** | 2 of 4 chains never read the payload; the other 2 read one field each with computed fallbacks |
| `analysis/pipelines/hinf_phenotype_sidecar.py` | ~10-function extraction cluster needing optional sources + residual | **reject here**; conversion routed to its own issue (`6ef623e`) | whole-file conversion impossible; partial conversion splits one product across two sources of truth |
| `eval/recipes.py` | 2–3 level parameter coalesces + filter-to-many | **reject** | alias-key resolution owned by the typed params models; filters already driven by declarative spec params |

### `analysis/matrix/standard_matrix.py`

The "multi-source coalescing" is `_merge_cell_metadata`: precedence merging of
`run_id`/`label`/`display_name`/`color` across the cell payload, the params
`cell_metadata` table, and manifest metadata. Its fallbacks are *computed*
values — `f"cell_{fallback_index}"`, the previously-computed `label`, a
function-argument `fallback_run_id` — which `Coalesce` deliberately cannot
express (defaults are declared literals). The chains use Python `or`
(falsiness), so a `Coalesce` migration silently changes behavior for
falsy-but-present values (empty-string labels, explicit-null colors).
`_metric_value` needs per-metric paths built at runtime from a dynamic metric
list — a runtime-constructed AST, not a declaration. The fail-closed
legacy-payload gates are params validation whose contract-native home is the
registered pydantic params model (`StandardMatrixEvalParams`), i.e. the
typed-params program, not expression ASTs. Everything is behind single call
sites in one module; no document or registration would carry the expressions.

### `analysis/rollout_cleanup.py`

The audit headline — four near-identical override→manifest-field→default
resolver chains — is wrong about payload involvement. `_resolve_summary_paths`
and `_resolve_manifest_out_path` never read the manifest payload at all (pure
filename and mode computation). `_resolve_bulk_dir` and
`_resolve_regeneration_spec_path` each read exactly one payload field, and
their fallbacks are computed: `.parent` of a resolved path,
`os.path.commonpath` over a recursive reference walk, a stem-derived
conventional path. The shared shape is "explicit override → payload field →
computed convention," and three of its four legs are outside the grammar's
domain by design. A migration would replace one `.get` chain per function with
an AST evaluation while the surrounding Python stays. No governed surface, no
cross-file duplication.

### `analysis/pipelines/hinf_phenotype_sidecar.py`

The one genuinely extraction-shaped file: tracked JSON manifests in, tracked
JSON product out, and `load_hinf_phenotype_sources` is a near-exact match for
the new optional-source/`missing_payload` semantics. It is still rejected as a
grammar retrofit because whole-file conversion is impossible and partial
conversion is worse than none:

- The row set is *discovered* by unioning run ids across all sources; the
  grammar has no parameterized or templated per-row evaluation.
- The row core is out of the language by design: `_count_by` aggregation,
  `_contains_key` recursive any-depth key search, token-based baseline/robust
  pair inference, and `_strip_row_suffix` (a string transform, deliberately
  excluded from the grammar).
- A partial conversion (a spec document owning the components table, Python
  owning rows/summary/claims) gives one product two sources of truth and
  forces re-emitting a diagnostic tied to closed issue `abe33da`.

The genuine long-term fix is architecture, not grammar: convert the module to
a registered analysis/report recipe under the manifest-canonical pipeline
policy. That is filed as deferred issue `6ef623e`; extraction-shaped parts can
become declared mappings there at authoring time.

### `eval/recipes.py`

This module already sits on strict registered pydantic params models. Every
"coalesce" in the audit is alias-key resolution — `perturbation`|`pert_axis`,
`bank_mode`|`mode`, `source_experiment`|`experiment`, four family-key aliases —
which is a schema concern whose contract-native fix is alias consolidation in
the params models / spec migrations, not expression ASTs at read sites. The
family/id subset selection ("filter-to-many") is driven by already-declarative
spec params; a `Filter` would have its `where` built at runtime from those same
params, adding a second declarative layer under an existing one, while the
missing-entry diagnostics (set differences, available-value error messages)
stay Python regardless. `_row_calibration_provenance` filters mapping *keys*
(with `startswith`), a shape `Filter` (list-entry) does not cover and should
not grow to cover. `_requested_perturbation_families` advances past explicit
nulls (`value is None: continue`); `Coalesce` treats an explicit null as a hit,
so a mechanical migration would silently change eval behavior.

## Cross-cutting findings

1. **Semantic drift is structural, not incidental.** Python `or`-coalescing
   and `is None`-advance idioms both differ from `Coalesce`'s absence class
   (path-missing / zero-match only). No in-function chain migrates
   behavior-neutrally for falsy or explicit-null payload values, so every
   "mechanical" migration would need its own census for zero structural
   payoff. This alone disqualifies bulk retrofits of live Python.
2. **The grammar's adoption path is authoring-time, not retrofit.** The
   extensions earn their keep when new governed surfaces — extraction product
   documents, `run_condition`s, registration-carried gates — declare them from
   birth. rlrmp's live consumers remain the broad-epsilon extraction document
   (frozen at v1, never re-emitted), the feedback-quality registration gating
   exprs, and `analysis/manifest_queries.py`.
3. **The 51c26e3 extensions were still correctly filed.** The audit corpus was
   evidence that a selection/gating grammar without absence handling, subsets,
   and optional sources could not express real spec surfaces at authoring
   time. It was an expressiveness case, not a migration work-list; rejecting
   the retrofits does not undercut the extensions.

## Disposition

- Issue `86e1dd1` closes as **adjudicated: no migration** when this record
  merges (closure via the auth request for this branch).
- `ci/feedbax-ref.toml` bumps `6e46427b…` → `17710f1c…` (verified pushed to
  `origin/develop`), keeping the clean-install CI pin aligned with the feedbax
  develop the editable install and contract suite actually run against.
- Deferred issue `6ef623e` owns the hinf-phenotype-sidecar pipeline
  conversion. `f852a72` (results-script extraction conversions) is unaffected
  and remains open.

Authored by Claude (Fable 5).
