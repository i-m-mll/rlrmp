# 96ac0e5 adjudication: query-language adoption and wrapper deprecation

Adjudication over the seven-file read-only audit (scratchpad `audit/`; to be
preserved under `results/96ac0e5/notes/audit/`). User mandate: no backward
compatibility — sole user, prefer deletion/migration over preserved names.

## Adjudication principle (governs every decision below)

Migrate a site to the feedbax query language (`Expr`/`Select`/`ValueQuery`/
`ExtractionProductSpec`) only when at least one of these holds:

1. **The expression becomes part of a governed spec surface** — an extraction
   product, a bundle `run_condition`, or a registration-carried gating
   declaration. Declarative-in-a-document is the point of the grammar.
2. **It eliminates cross-file duplication of the same selection/gating shape**
   (copy-paste drift already observed or structurally likely).

Ordinary in-function Python whose shape merely *matches* the grammar (one-off
in-memory selects, dispatch that chooses which Python code runs, value
formatting) stays Python. Wrapping local control flow in expression ASTs is
indirection, not parsimony. This principle rejects a large fraction of the
audit's raw "candidates" deliberately.

## Corrections to audit headlines

- **The ~150 `params.get(key, default)` sites in `model/cs_lss_gru.py`,
  `model/feedbax_graph.py`, `train/cs_nominal_gru.py` are NOT query-language
  targets.** The audit called `ValueQuery.default` the top gap on their
  strength; the correct owner for those sites is the cc3a61f typed-params
  program (schema-owned defaults via pydantic models + the defaults ratchet),
  which is already in flight. A coalesce/default grammar extension is still
  justified (see feedbax filing), but on the strength of the genuine
  spec-surface sites (rollout_cleanup ×4 chains, eval/recipes coalesces,
  standard_matrix multi-source merges), not these.
- **The `_component_summary` "4 reimplementations" shrink to ~2 live.** The two
  `scripts/materialize_output_feedback_*` copies are inside LEGACY-frozen files
  (64d5f13 skip list) and are port-or-delete, not migration targets. Live
  copies: `gru_map_error_decomposition.py` (+ `cs_gru_standard_materialization.py`'s
  `_summary`). Duplication pressure is therefore lower than headline; folded
  into M1 rather than treated as the top target.

## Category C verdicts (wrapper deprecation — Lane W, dispatching now)

| Name | Verdict | Rationale |
|---|---|---|
| `calibration.build_open_loop_calibration_product` | **delete** | Zero callers anywhere; real materializer (`scripts/materialize_perturbation_open_loop_calibration.py` → `gru_perturbation_calibration`) bypasses it. |
| `calibration.write_open_loop_calibration_product` | **delete** | Same. |
| `calibration.build_perturbation_calibration_defaults_payload` | **delete + rewire test** | Only caller is the round-trip identity test. The defaults product's source constants were deleted in 7cfe941 — the tracked document IS the source of truth; regeneration-from-code is impossible by design, so the builder is vestigial by construction. The identity test verifies the tracked payload bytes against the pinned sha256 instead. |
| `calibration.write_perturbation_calibration_defaults_payload` | **delete** | Zero callers. |
| `calibration.build_perturbation_calibration_defaults_product` | **delete** | Zero callers. |
| `calibration.write_perturbation_calibration_defaults_product` | **delete** | Zero callers. |
| `calibration.open_loop_peak_delta_x_per_unit` | **delete** | Dead accessor; all real usage reads the `OpenLoopCalibration` attribute. |
| `calibration.controller_visible_velocity_scale_m_s` (function) | **delete** | Same. |
| `broad_epsilon.build_broad_epsilon_budget_anchors_product` | **delete** | Pass-through around `materialize_extraction_product`; the materialize script calls the feedbax engine directly with the tracked spec path. This answers the user's original question: the name dies. |
| `broad_epsilon.write_broad_epsilon_budget_anchors_product` | **delete** | Pass-through write; inlined into the script. |
| `broad_epsilon.verify_broad_epsilon_budget_anchors_product` | **delete** | Callers (script + `load_broad_epsilon_anchors`) call `verify_extraction_product` directly. |
| `broad_epsilon.load_broad_epsilon_anchors` | **keep** | Real logic: fail-closed verify + contract-key typed projection consumed by training. Issue body pre-adjudicated this. |
| `calibration.load_open_loop_calibration`, `load_perturbation_calibration_defaults`, `envelope.validate_data_product`, `registry.register_data_product_identity` | **keep** | Real fail-closed logic, load-bearing. |
| `envelope.read_data_product` / `load_data_product` | **keep** | Internal engine composition used by both loaders; not user-facing wrapper names. |
| `consumed_*_identity` ×3 | **keep for now; consolidate post-round-2** | Consolidating into one generic `envelope` helper touches `train/cs_perturbation_training.py`, owned by the running cc3a61f Lane D. Queued as an integration-time follow-up, not a Lane W edit. |
| `broad_epsilon._contract()` post-load projection | **keep as Python** | Loader produces typed Python objects; a second extraction-spec layer here is indirection with no governed-document payoff (principle 1 fails, no duplication). Audit top-5 item 4 rejected. |
| `lint.violations` zero-caller status | **keep** | Exercised by the CI gate tests; that is its job. |

**Guard (fix-with-guard policy):** Lane W adds an explicit public-surface
inventory test for `rlrmp.data_products` (pinned name set; any addition or
removal is a deliberate test edit). Prevents silent reaccretion of parallel
builder paths — the residual class here was "bespoke build/write path bypassed
by the real materializer."

Also in Lane W (module hygiene): consolidate the 6 independent `_read_json`
one-liners onto a shared `rlrmp.io.read_json`; preserve the one
divergent-semantics variant (`training_diagnostics`' None-on-missing) as a
call-through. Preserve the audit corpus + this adjudication under
`results/96ac0e5/notes/`.

## Migration lane M1 (current grammar suffices — new issue, next dispatch)

1. **Feedback-quality component gating** (`declarative_materialization.py`
   1407-1470, 2619-2676): each `FeedbackQualityComponentRegistration` already
   carries a `gating_spec` *string* naming its condition while the gating runs
   procedurally, duplicated across the materialize and output paths. Replace
   the strings with declared `Expr` values evaluated through the feedbax
   evaluator at both call sites. Passes principle 1 (registration-carried
   declaration) and 2 (duplicated cascade). Strongest single target in the
   audit; behavior-identical census required.
2. **`artifact_migration._manifest_parent`** → `Select(where=Compare(kind eq))`
   wiring proof — production code that is character-for-character the grammar's
   own example.
3. **`_is_cs_lss_run_spec`** → declared `AnyOf` predicate (optional, small).
4. **Live `_component_summary`/`_summary`/`_find_standard_row`**
   (`gru_map_error_decomposition.py`, `cs_gru_standard_materialization.py`) →
   shared Select/ValueQuery-backed helper (kills the live duplication).

## Feedbax filing F (grammar gaps; implementation lane after triage there)

Scope (tight, selection/gating only — the grammar must not grow into a
programming language):

1. `ValueQuery.default` + ordered `Coalesce` over queries — evidence:
   rollout_cleanup ×4 fallback chains, `eval/recipes.py` coalesces,
   `standard_matrix` multi-source merge, `_uses_open_loop_calibration`.
2. Filter-to-many (Select is exactly-one) — evidence: `reports.py` role filter,
   `_perturbation_bank_from_params`, `load_completed_training_manifests`,
   feedback-quality multi-component selection.
3. Residual-passthrough field mapping — evidence: `PerturbationSpec.from_mapping`
   `extra` dict.
4. `startswith`/`endswith` CompareOps — evidence: `_row_calibration_provenance`.
5. Design question (not a demand): opt-in optional `SourceBinding` with a
   declared missing-status, for the `hinf_phenotype_sidecar` degrade-to-missing
   pattern, without weakening the fail-closed default.
6. Documentation: keep payload `Compare(exists)` conceptually distinct from
   filesystem existence (repeated conflation risk in the audit).

**Deliberately excluded:** Switch/if-then-else value production (the many
"Partial" B sites are in-function control flow that stays Python); string
transforms beyond prefix/suffix ops; argmin/argmax (live-site value too thin —
observed sites are LEGACY, results-scripts, or genuinely computational).

## Deferred (filed as `deferred` issues, not umbrella children)

- **D1 — grammar-blocked migrations** (blocked by F): `standard_matrix` near-A
  extraction + fail-closed gates, rollout_cleanup resolver chains,
  `hinf_phenotype_sidecar` extraction cluster, `eval/recipes` coalesces.
- **D2 — results-script extraction-product conversions** (expressible today,
  low priority): `results/27dece3/.../materialize_scale_spec_lock.py`,
  `results/4d79e07/.../materialize_delayed_pgd_comparison.py` (pure-extraction
  portions become `ExtractionProductSpec` documents; derived-ratio fields stay
  code).

## Explicitly not migrating (recorded rationale)

In-memory one-off selects and dispatch across `analysis/math/*` (the 9×
`gamma_factor` `next(...)` idiom, select-by-label, compound row picks) — the
payloads never round-trip through manifests at those points; the prerequisite
is manifest routing, which is the report-stage era's concern, not a bolt-on.
`Namespace(**config)` rehydration idiom in legacy eval scripts — legacy CLI
surface, no governed-document payoff. `perturbation_rows` channel rule table —
already a declarative Python table doing per-row schema validation.
