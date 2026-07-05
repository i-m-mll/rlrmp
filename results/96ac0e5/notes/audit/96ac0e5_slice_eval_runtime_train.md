# Issue 96ac0e5 audit slice — `eval/`, `runtime/`, `train/`, `model/`, `cloud/`, `data_products/`, top-level modules

Worktree: `/Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/integration__d1f90ac-defaults-cascade/`
Reference contracts read first: `feedbax/feedbax/contracts/expressions.py` (Compare/AllOf/AnyOf/Not/Select/Coerce/ValueQuery), `feedbax/feedbax/contracts/extraction.py` (ExtractionProductSpec: SourceBinding + FieldMapping, tracked-JSON-only, extraction-not-computation).

Grammar recap load-bearing for the findings below:
- `ValueQuery` has **no default-value fallback** — a missing path raises `ExpressionPathMissing` unless wrapped in a `Compare(op="exists")`. There is no `.get(key, default)` equivalent.
- `Select` requires **exactly one** match (`ExpressionSelectAmbiguous` otherwise) — no "filter to many" / "filter to zero-or-more" mode.
- `Compare` has no string-transform ops (`startswith`, `endswith`, strip/replace) and no cross-item value coalesce (`ValueQuery` reads exactly one bound item).
- Named predicates must be pure (no I/O, clock, randomness) — filesystem `.exists()`/`.is_file()` checks cannot be named predicates as-is.

---

## LEGACY-banner skip list

Searched the whole slice (`eval/`, `runtime/`, `train/`, `model/`, `cloud/`, `data_products/`, and all top-level `*.py` + `viz/`) for `LEGACY (frozen`. **Zero matches.** Every module-level `LEGACY (frozen ...)` banner in the repo lives under `analysis/` or `controllers/`, both out of scope for this slice. No files or functions were skipped for this reason. (The slice does contain many `LEGACY_*`-named constants — e.g. `runtime/run_specs.py:LEGACY_POINT_MASS_GRAPH_TYPES`, `runtime/spec_migrations.py:LEGACY_TRAINING_CONFIG_KIND` — these are retired-ID/compat markers, not the frozen-banner convention, and were read normally.)

---

## Category A — extraction / ValueQuery candidates

| Site | file:lines | What it does | Sources read | Computation? | Blocker/rationale |
|---|---|---|---|---|---|
| `load_open_loop_calibration` | `data_products/calibration.py:527-550` | Loads tracked product JSON, hand-selects/coerces `peak_delta_x_per_unit` (nested dict comprehension), 3 scalar fields into `OpenLoopCalibration` dataclass | tracked JSON via `load_data_product` | Only `float()`/`str()` coercion | Clean ValueQuery/Coerce candidate; nested-dict comprehension (family→bin→value) is a double `Select`-shaped iteration the current `Select` (single-list, exact-one) doesn't cover — needs a "map over all entries" mode |
| `load_perturbation_calibration_defaults` | `data_products/calibration.py:553-629` | Loads product, resolves a second artifact (payload JSON) by role+hash, then hand-builds 6 typed tuples of dataclasses from payload lists | tracked JSON (product) + separate tracked JSON (payload artifact) | Only coercion (`float/int/str`) | Two-source extraction (product + payload) is expressible as two `SourceBinding`s; per-list dataclass construction is repeated `Select`-like iteration over lists — same "map all entries" gap as above |
| `_manifest_parent` | `artifact_migration.py:264-270` | `matches = [p for p in manifest.provenance.parents if p.kind == kind]; if len(matches) != 1: raise` | in-memory `ModelArtifactManifest` (itself loaded from tracked JSON just above) | None | **Positive match** — this is *exactly* `Select(where=Compare(item="entry", path="kind", op="eq", value=kind))`; a clean instance of the grammar already covering the shape in production code, just not wired through it |
| `_subset` | `eval/recipes.py:898-899` | `{key: params[key] for key in keys if key in params}` | in-memory validated params dict | None | Field-subset selection = N `FieldMapping`s; blocked only by **in-memory-only source** (params isn't a tracked JSON file at this call site — it originates upstream from an `EvaluationRunSpec`) |
| `_pick` | `loss.py:164-165` | `{k: d[k] for k in keys if k in d}` — selects `DEFAULT_TOP_WEIGHTS` entries restricted to the dynamically-computed `terms.keys()` | module constant dict, keyed by a **runtime-computed** key set | None | Same subset shape as `_subset`, but the key set (`terms.keys()`) is not statically known at spec-authoring time — blocked by dynamic/data-driven field enumeration, which `FieldMapping` (a static list) cannot express |
| Graph/run-spec node builders (class of sites) | `model/cs_lss_gru.py` (~20 call sites, e.g. lines 1103-1290), `model/feedbax_graph.py` (37 call sites, e.g. lines 197-246), `train/cs_nominal_gru.py` (93 call sites, e.g. lines 915-932) | Read one field at a time out of a tracked GraphSpec-node `params` dict or run-spec dict via `params.get(key, default)`, coerce with `int()/str()/float()/jnp.dtype()` | tracked JSON (`graph_spec_path` node params; `run.json` top-level fields) | Only type coercion | **This is the single largest, most systemic Category A class in the slice** (see Category D #1 below — it is blocked by exactly one missing grammar feature: `ValueQuery` has no default-value fallback) |

## Category B — predicate / gating candidates

| Site | file:lines | What it does | Blocker/rationale |
|---|---|---|---|
| `validate_nominal_gru_run_spec` | `runtime/run_specs.py:145-247` | AllOf-shaped chain: missing-top-level-keys check, `controller_kind == "gru"`, training-mode split+AnyOf-membership, cross-field equality (`loss_summary.objective_profile == loss_objective`), missing-provenance-keys, missing-graph-pointer-keys, then per-pointer conditional sidecar-file-exists check | Mostly `Compare`/`AllOf`/`AnyOf`-expressible over the run-spec payload; blocked where it does real filesystem `.is_file()` checks (I/O, not a pure predicate) and where it needs cross-field compare (`Compare` compares one item's path against a **static** value, not two paths on the same item) |
| `_is_cs_lss_run_spec` | `runtime/run_specs.py:267-277` | `AnyOf` over 4 candidate nested paths, each tolerant of missing intermediate keys (returns `None`, never raises) | Structurally a clean `AnyOf([AllOf([Compare(exists), Compare(eq)]) for path in paths])` — fully expressible, just not wired through the contract |
| `_validate_cs_lss_graph_spec_sidecar` | `runtime/run_specs.py:289-325` | Reads sidecar JSON, gates on `nodes.mechanics.type == required` and `nodes.feedback.type in allowed-set`; also computes a diagnostic-only `legacy_types` set for the error message | Core boolean gate is `Compare`-expressible (dict-key path resolution is exactly `_get_path`'s dict-then-getattr walk); blocked only by the diagnostic aggregation (set intersection + sorted) needed for the error text, which Expr doesn't produce |
| `require_run_seed` / `require_run_dt` | `runtime/run_spec_access.py:19-57` | Fail-closed accessor: `exists` + coerce, else raise with a domain-specific message; `require_run_dt` additionally falls back from `game_card.dt` to a **different item** (`hps.dt`, a live object, not JSON) | `require_run_seed` maps to `Compare(exists)` + `Coerce`; `require_run_dt` is blocked by cross-item coalesce (`ValueQuery` binds one item) and a non-JSON fallback source |
| `_require_one_of` | `eval/recipes.py:902-905` | `if not any(key in params for key in keys): raise` | Clean `AnyOf([Compare(exists) for key in keys])` — fully expressible |
| `_uses_open_loop_calibration` | `eval/recipes.py:511-514` | `Compare(eq True)` OR `Compare(eq "calibrated")` on a value that is itself a 2-level coalesce (`bank_mode` falling back to `mode` falling back to `"raw"`) | Boolean shape is `AnyOf`-expressible; blocked by the inner coalesce (no default in `ValueQuery`) |
| `load_completed_training_manifests` | `runtime/studio_records.py:74-108` | Loads each candidate manifest, filters by `AllOf(isinstance-check, status=="completed", optional run_set_id match, optional id/job_id-in-wanted-set)`, raises if the result list is empty | Per-item predicate is mostly `Compare`/`AllOf`-expressible (modulo the Python-`isinstance` type check, which isn't the same as `Compare(has_type)`, which checks the bound `ContextItem.kind` tag, not a runtime Python class); the **collect all passing items** result shape needs "filter to many," which `Select` (exact-one) does not support |
| `validate_synced_modal_run` / `_validate_optional_graph_spec` | `cloud/modal_artifact_sync.py:168-221` | Same shape as `run_specs.py`'s sidecar validation: missing-file list comprehension + conditional pointer/sentinel check | Duplicate instance of the same two blockers (I/O-based existence checks; "collect all missing" instead of a single boolean) |
| `_uses_legacy_array_store_schema` / execution-backend gate | `artifact_migration.py:139-144, 191-196` | `isinstance(dict) and schema_version == LEGACY_VERSION`; `backend != LEGACY_EXECUTION_BACKEND: raise` | Equality gates are clean `Compare` matches; `isinstance` on a nested field's runtime type (as opposed to the bound item's own `kind` tag) has no direct `CompareOp` |
| Per-term weight gates (dozens of sites) | `loss.py:1404-1720` (repeated `if getattr(user_outer_weights, name, 0.0) != 0.0: terms[name] = TargetStateLoss(...)`) | Each is a clean `Compare(ne, 0.0)` gate on the same item (`user_outer_weights`) with a different path | The **gate** is trivially expressible; the **gated action** constructs a live Python loss-term object (lambda-bearing `TargetStateLoss`), not a JSON value — blocked because Expr only ever returns bool/scalar, never a typed object |
| `resolve_run_spec_execution_args` | `train/cs_nominal_gru.py:753-785` | Membership-gated allowlist (`key in RUN_SPEC_RUNTIME_OVERRIDE_KEYS`), cross-item coalesce (`spec_values.get(key, getattr(defaults, key, None))`), custom equality (`_cli_values_match`), collect-all-mismatches then raise | Allowlist gate is `Compare(in)`-expressible; blocked by cross-item coalesce, a non-`eq` custom comparator, and "collect all failing," same pattern as `_missing_keys` |
| `assert_supported_graph_spec_version` | `runtime/feedbax_contract_versions.py:20-27` | `version not in SUPPORTED_GRAPH_SPEC_VERSIONS: raise` | Trivial `Compare(op="in")`; only one call site, low value to migrate on its own |

## Category C — thin-wrapper deprecation inventory (`data_products/` full sweep)

Every public `build_*`/`write_*`/`verify_*`/`load_*`/`*_requirement` function under `data_products/`, classified, with the complete non-test caller inventory (`src/`, `scripts/`, `results/*/scripts/`) plus a test-caller count.

### `data_products/calibration.py`

| Function | Class | Non-test callers | Test callers |
|---|---|---|---|
| `build_open_loop_calibration_product` | (ii) thin envelope assembly | **none** (only its own `__all__`/producer-string self-reference) | 0 |
| `write_open_loop_calibration_product` | (i) pass-through JSON write | **none** | 0 |
| `build_perturbation_calibration_defaults_payload` | (ii) thin envelope assembly | **none** | 1 (`tests/test_product_identity_hash.py:41,151`, round-trip identity check) |
| `write_perturbation_calibration_defaults_payload` | (i) pass-through JSON write | **none** | 0 |
| `build_perturbation_calibration_defaults_product` | (ii) thin envelope assembly | **none** (only self-reference in its own producer string) | 0 |
| `write_perturbation_calibration_defaults_product` | (i) pass-through JSON write | **none** | 0 |
| `calibration_data_product_requirement` | (ii) thin requirement factory | `data_products/calibration.py` (self, registration + loader) | 5 (`tests/test_product_identity_hash.py:42,62,97,239,248`) |
| `calibration_defaults_data_product_requirement` | (ii) thin requirement factory | `data_products/calibration.py` (self) | 4 (`tests/test_product_identity_hash.py:43,81,84,122`) |
| `load_open_loop_calibration` | (iii) real logic (typed accessor, fail-closed, `lru_cache`) | `analysis/pipelines/gru_perturbation_bank.py:523`; `data_products/calibration.py` (self, 2 call sites); `data_products/__init__.py`; `train/cs_perturbation_training.py` (7 call sites: 2115, 4687, 4690, 4708, 4767, 4952, 4973, 5521) | 3 |
| `load_perturbation_calibration_defaults` | (iii) real logic | `analysis/pipelines/gru_perturbation_bank.py` (2 sites); `analysis/pipelines/gru_perturbation_calibration.py` (6 sites); `data_products/calibration.py` (self); `train/cs_perturbation_training.py` (6 sites) | 1 |
| `open_loop_peak_delta_x_per_unit` | (i) pass-through accessor | **none found outside its own module/`__all__`** and one comment reference in `cs_perturbation_training.py` (not a call) | 0 |
| `controller_visible_velocity_scale_m_s` | (i) pass-through accessor | `analysis/pipelines/gru_perturbation_bank.py` (3 sites, as a local variable name shadowing, not calling the function itself — actual function calls are 0 outside `calibration.py`) | 0 |
| `consumed_calibration_identity` | (ii) thin envelope assembly | `train/cs_perturbation_training.py:2188`; `eval/recipes.py:518-520` | 0 |
| `consumed_perturbation_calibration_defaults_identity` | (ii) thin envelope assembly | `analysis/pipelines/gru_perturbation_calibration.py:946` | 0 |

**Confirmed dead code (0 non-test AND 0 test callers):** `build_open_loop_calibration_product`, `write_open_loop_calibration_product`, `write_perturbation_calibration_defaults_payload`, `build_perturbation_calibration_defaults_product`, `write_perturbation_calibration_defaults_product`. Verified this isn't a stale-grep artifact: `scripts/materialize_perturbation_open_loop_calibration.py` exists and is the actual materializer entry point, but it imports from `rlrmp.analysis.pipelines.gru_perturbation_calibration` and writes JSON directly — it never calls any of `calibration.py`'s `build_*`/`write_*` functions. These five functions were written as the intended production path but the real materializer bypasses them entirely.

`build_perturbation_calibration_defaults_payload` has exactly one caller, a round-trip test — candidate for demotion to test-only/inlining if no other consumer appears in a full-repo check.

### `data_products/broad_epsilon.py`

| Function | Class | Non-test callers | Test callers |
|---|---|---|---|
| `build_broad_epsilon_budget_anchors_product` | (ii) thin, delegates to feedbax `materialize_extraction_product` | `scripts/materialize_broad_epsilon_budget_anchors.py:48` | 0 |
| `verify_broad_epsilon_budget_anchors_product` | (iii) real logic (re-verifies against extraction spec) | `scripts/materialize_broad_epsilon_budget_anchors.py:53`; `data_products/broad_epsilon.py` (self, from `load_broad_epsilon_anchors`) | 0 |
| `write_broad_epsilon_budget_anchors_product` | (i) pass-through JSON write | `scripts/materialize_broad_epsilon_budget_anchors.py:51` | 0 |
| `broad_epsilon_data_product_requirement` | (ii) thin requirement factory | self (registration + `verify_...`) | 1 |
| `load_broad_epsilon_anchors` | (iii) real logic | `analysis/pipelines/gru_worst_case_epsilon_audit.py:190`; `data_products/broad_epsilon.py` (self); `train/cs_perturbation_training.py` (4 sites: 293, 310, 413, 500) | 1 |
| `consumed_broad_epsilon_identity` | (ii) thin envelope assembly | `train/cs_perturbation_training.py:2190` | 0 |

`broad_epsilon.py` is the **exemplar** of the correct pattern for this class of module (it actually routes through `ExtractionProductSpec`/`materialize_extraction_product`/`verify_extraction_product`); `calibration.py` predates that pattern and never migrated — that's *why* its `build_*`/`write_*` functions are orphaned. This is the strongest concrete signal in the slice for what "done right" looks like vs. what should be deleted.

### `data_products/envelope.py`

`load_data_product`, `read_data_product`, `validate_data_product` — all class (iii), carry the actual fail-closed identity/schema/role/hash validation logic referenced from CLAUDE.md. Heavily used (both calibration and broad_epsilon load through `load_data_product`); not deprecation candidates.

### `data_products/registry.py`

`register_data_product_identity`, `registered_data_product_identities` — registry-pattern functions, not `build_/write_/verify_/load_/*_requirement`-named, out of the requested verb set; not evaluated for deletion.

### Lighter touch outside `data_products/` (same verb family, not exhaustively caller-counted)

- `eval/minimax_io.py:load_config` — class (i), pure pass-through JSON read (returns the raw dict, no field mapping); used by CLI eval scripts. Not a governed data product; not a deletion candidate, just noted for completeness.
- `runtime/spec_migrations.py:load_rlrmp_spec_payload` — class (ii)/(iii) hybrid: JSON read + one `if kind == LEGACY_TRAINING_CONFIG_KIND: raise ArchiveOnlySpecError` dispatch, then delegates to `accept_rlrmp_spec_payload`. Not evaluated for full caller count (out of `data_products/`); worth a follow-up sweep if the caller wants the full-repo verb inventory.
- Dozens more `build_*`/`write_*`/`load_*` functions exist in `runtime/training_run_specs.py`, `runtime/checkpoint_custody.py`, `train/cs_nominal_gru.py`, `train/minimax.py`, `train/closed_loop_distillation.py`, `train/guided_distillation.py`, `model/feedbax_graph.py`, `model/cs_lss_gru.py`, `cloud/modal_runner.py` — these are almost all training-run/checkpoint/graph-spec builders with real callers from CLI scripts and are load-bearing, not orphaned thin wrappers. A full caller-count pass on all of them was out of scope for this effort level; flagging as a residual sweep if a future audit wants full C coverage beyond `data_products/`.

## Category D — borderline (blocking grammar gap named)

| Site | file:lines | Pattern | Blocking gap |
|---|---|---|---|
| **Systemic `.get(key, default)` graph/run-spec field reads** | `model/cs_lss_gru.py` (~20 sites), `model/feedbax_graph.py` (37 sites), `train/cs_nominal_gru.py` (93 sites) | Every GraphSpec-node/run-spec field read follows `params.get(key, DEFAULT)` then coerces | **`ValueQuery` has no default-value fallback.** This is the single highest-leverage grammar gap in the whole slice — one field (`default: Any | None`) on `ValueQuery`, applied only when `Compare(exists)` is false, would make the dominant idiom in this codebase spec-expressible. Directly matches the branch's "defaults-cascade" theme. |
| `_key_from_params` | `model/cs_lss_gru.py:1231-1235` | `key` present → convert; else derive `PRNGKey(require_run_seed(params))` | Fallback branch is a **computed** value (PRNG derivation), not a literal default |
| `_params_with_parent_key` | `model/cs_lss_gru.py:1238-1247` | Inherit parent's `key` into a child params dict only if child lacks `key`/`seed` and parent has one | Conditional inherit-from-parent-into-child merge; no such combinator exists (nearest concept, `AllOf` gating a value copy, doesn't produce a merged payload) |
| `_hidden_type_name` | `model/cs_lss_gru.py:1220-1228` | Type-dispatch + attribute-chain coalesce over a live callable/class object | Source is not JSON — a Python type/callable, not a bound tracked-JSON payload |
| `_run_spec_source` | `runtime/run_spec_access.py:60-71` | Coalesce over 4 candidate keys, then string-format from two other fields, else `"<unknown>"` | Coalesce-across-static-keys + string formatting, neither expressible |
| `_run_id` / `_run_id_from_spec` | `runtime/checkpoint_custody.py:322-336` | Coalesce over 3-4 candidate keys with a `value not in (None, "")` truthiness rule, f-string with prefix, else hash-based fallback id | Coalesce + string formatting + a `_canonical_sha256` computation for the final fallback |
| `spec_digests` | `runtime/checkpoint_custody.py:225-243` | `model_dump()` three nested specs, then `hashlib.sha256`-style canonical hash each | In-memory pydantic source (not tracked JSON) + genuine hashing computation |
| `_requested_perturbation_families` / `_requested_perturbation_ids` / `_source_experiment` / `_repo_root_for_eval` / `_perturbation_eval_bulk_dir` | `eval/recipes.py:565-771` | Coalesce over 2-4 candidate keys, some falling back to computed defaults (`REPO_ROOT`, a joined `Path`) | Coalesce chain + computed (non-literal) fallback + type dispatch (str vs Sequence) |
| `_perturbation_bank_from_params` | `eval/recipes.py:523-562` | Filters `bank["perturbations"]` rows to a requested family/id set, returning the matching **subset** (many rows), plus computes a "missing" diagnostic via set difference | `Select` requires exactly one match; this needs "filter to many" |
| `_row_calibration_provenance` | `eval/recipes.py:659-686` | `key.startswith("calibration_") or key in known_keys` filter building a new dict | No string-transform op (`startswith`) in `CompareOp`; also dict-filter (not list-`Select`) |
| `_consumed_data_identities` | `eval/recipes.py:889-895` | `isinstance` type-dispatch (Mapping vs Sequence) normalizing to a list | Type-based branching on a runtime value's Python type, not on a bound item's declared `kind` |
| `_missing_keys` | `runtime/run_specs.py:411-412` | `[key for key in required if key not in mapping]` — collects **all** absent keys for a readable error | `AllOf`/`AnyOf` only return a single boolean; no "collect all failing predicates" result shape |
| `run_spec_path` / `resolve_run_artifact_path` | `paths.py:101-132, 151-171` | Try flat path, then legacy path, then default; try 3 candidate on-disk layouts in order | Predicate is `Path.exists()` — real filesystem I/O, disallowed for named predicates by the "no I/O" purity rule |
| `minimax_args_from_run_spec` | `artifact_migration.py:102-112` | Multi-stage cascade: frozen defaults dict → overlay normalized CLI-flag keys present in defaults → overlay direct run-spec keys present in defaults (excluding 2 keys) | Dynamic-key merge-with-priority-order across an unbounded key set; no "merge/override cascade" primitive, and the key set isn't statically declared |
| `_normalized_cli_flags` | `artifact_migration.py:273-282` | `lstrip("-")`, `replace("-", "_")`, then a `no_`-prefix-strip-and-negate transform | String-transform ops (`lstrip`, `replace`, prefix-strip) entirely absent from `Coerce`/`CompareOp` |
| `_nsget` | `loss.py:144-154` | Generic dot-path getter tolerant of dict-or-object hybrids, returns `default` on any missing segment | Structurally a re-implementation of `expressions._get_path`, but default-tolerant instead of raising — same missing-default-fallback gap as the top row |
| `_filter_nonzero` | `loss.py:168-169` | `{k: v for k, v in d.items() if float(v) != 0.0}` | Dict-value filter; `Select` only filters lists via the `entry` alias |
| `_graph_from_training_manifest` / `_copy_training_scenario_payload` / `_graph_manifest_refs` | `runtime/studio_records.py:351-418` | `isinstance` dispatch across 3 payload shapes (`SpecPayload`/`ParentRef`/neither), each guarding either a differently-shaped object construction or an in-place mutation | Grammar has no mutation/apply-if primitive and no typed-object-construction target; also coalesce (`graph_spec.sha256 or f"..."`) |
| `_default_workspace_label` / `_default_job_id` | `runtime/studio_records.py:460-471` | `manifest.run_set_id or manifest.job_id or manifest.id` coalesce chains | Same missing-default-fallback / coalesce gap |

---

## Counts

| Category | Count of distinct sites/classes flagged |
|---|---|
| A — extraction/ValueQuery candidates | 8 (2 in `data_products/`, 1 in `artifact_migration.py` [positive `Select` match], 2 in `eval/recipes.py`/`loss.py`, 3 systemic multi-site classes in `model/cs_lss_gru.py`, `model/feedbax_graph.py`, `train/cs_nominal_gru.py` totalling ~150 individual `.get(key, default)` call sites) |
| B — predicate/gating candidates | 13 |
| C — thin-wrapper inventory (`data_products/` full sweep) | 20 functions classified; **5 confirmed dead code**, 1 test-only, 14 live/load-bearing |
| D — borderline (grammar gap named) | 18 |
| LEGACY-skipped files | 0 (none in this slice's directories) |

## Top-3 targets

1. **`ValueQuery` default-value gap (Category A/D, systemic).** Add an optional `default: Any` to `ValueQuery` (returned when the path/item is absent, instead of raising `ExpressionPathMissing`). This one change would make the dominant idiom in the slice — `params.get(key, DEFAULT)` field reads across `model/cs_lss_gru.py`, `model/feedbax_graph.py`, and `train/cs_nominal_gru.py` (roughly 150 call sites combined) — expressible as declarative extraction specs instead of hand-written Python. Directly on-theme for this "defaults-cascade" branch.

2. **Delete 5 confirmed-dead `data_products/calibration.py` functions.** `build_open_loop_calibration_product`, `write_open_loop_calibration_product`, `write_perturbation_calibration_defaults_payload`, `build_perturbation_calibration_defaults_product`, `write_perturbation_calibration_defaults_product` have zero callers anywhere in `src/`, `scripts/`, `results/*/scripts/`, or `tests/`. The real materializer (`scripts/materialize_perturbation_open_loop_calibration.py`) bypasses them entirely via `analysis/pipelines/gru_perturbation_calibration`. `data_products/broad_epsilon.py` (which *does* route through `ExtractionProductSpec`/`materialize_extraction_product`) is the template these should either be migrated to match, or deleted outright since they don't earn their existence.

3. **`_manifest_parent` (`artifact_migration.py:264-270`) as the wiring template.** It's already an exact, unmodified match for `Select(where=Compare(item="entry", path="kind", op="eq", value=kind))` — the cleanest positive proof-of-concept in the slice that the existing `Select` primitive covers real production code today. Worth using as the first concrete migration/wiring exercise before tackling the harder coalesce/default-value gaps above.

## File path

`/private/tmp/claude-501/-Users-mll-Main-10-Projects-10-PhD-rlrmp/08a431d4-ec60-4f13-ab0e-b9622d4fe1d8/scratchpad/audit/96ac0e5_slice_eval_runtime_train.md`
