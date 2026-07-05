# Query-language migration sweep — issue 96ac0e5

Audited tree: `worktrees/integration__d1f90ac-defaults-cascade` (integration worktree with latest merged state). Read-only; no repo files were modified.

**Methodology note.** `src/rlrmp/analysis/pipelines/`, `src/rlrmp/data_products/`, and `scripts/` were read in full or near-full depth (every `json.load`/`.read_text(` call site plus surrounding function bodies). `src/rlrmp/analysis/` core + `math/` + `matrix/`, `eval/`, `train/`, `runtime/`, `benchmarks/`, `cloud/`, and misc top-level modules were swept via targeted grep for JSON-read/selection/gating signal patterns (`json.load`, `.read_text(`, `next(`, `.get(` chains, `for x in y: if ...: return`) followed by full reads of every hit, rather than line-by-line reads of files with no such signal — those directories are training/model/numeric code with little or no manifest consumption, confirmed empty by the sweep. `results/*/scripts/` (81 files, ~35 experiment dirs) got the lighter pass the issue asked for: full file list, grep for the same signal patterns, and targeted reads of the highest-signal hits (aggregation-named scripts, repeated helper names found elsewhere). This is not a claim that every one of the ~200 files was read top to bottom; it is a claim that every JSON-reading / manifest-consuming code path was located and read.

An in-session fork was also dispatched to independently cover `analysis/` core + `math/` + `matrix/`; a parallel-forked audit does not nest further, so the other 3 planned forks executed inline here. If that fork's separate report later reconciles with this one, treat this document as primary — its findings for that slice were produced by direct grep+read.

---

## Category A: extraction/ValueQuery candidates

| Site | file:lines | What it does | Sources read | Classification | Blocker/rationale |
|---|---|---|---|---|---|
| `_component_summary` (×2) | `src/rlrmp/analysis/pipelines/gru_map_error_decomposition.py:509-516`; `src/rlrmp/analysis/pipelines/failure_decomposition.py` (LEGACY, same helper name — see skipped list) | Loops `row["certificate_components"]`, returns `{status, summary, reason}` for the first component whose `name` matches | in-memory manifest row (loaded elsewhere via `_read_json`) | **Full candidate** | Exact `Select(where=Compare(item=entry, path=name, op=eq, value=name))` + 3-field `FieldMapping`. No computation. |
| `_component_summary` (×2, near-duplicate) | `scripts/materialize_output_feedback_failure_decomposition.py:837-842`; `scripts/materialize_output_feedback_sweep_certificates.py:1285-1290` | Same Select-by-name pattern; the sweep-certificates variant adds a second predicate (`status == "available"`) | tracked JSON manifests read via local `_read_json` | **Full candidate** | `AllOf(Compare(name==name), Compare(status==available))` + `ValueQuery(path="summary.<key>")`. **This exact helper is reimplemented independently in 4 files** (2 in `analysis/pipelines/`, 2 in `scripts/`) — see top-5 list. |
| `_summary` | `src/rlrmp/analysis/pipelines/cs_gru_standard_materialization.py:1373-1374` | `components.get(name, {}).get("summary", {}).get(key)` | in-memory components dict | **Full candidate** | Literally the `ValueQuery` path-walk (`name.summary.key`), except it silently returns `None` on a missing segment instead of raising `ExpressionPathMissing` — same semantic gap noted for `_nested_get` below. |
| `_nested_get` | `src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py:995-1001` | Generic dotted-path walk over a `Mapping`, returns `None` on any missing/non-mapping segment | in-memory payload | **Full candidate (helper itself)** | This is a hand-rolled reimplementation of `expressions._get_path`, used by ~6 call sites in the same file (`_flatten_behavior_metrics`, `_compact_class_summary`, `_compact_sisu_perturbation_comparison`). Its "return None on missing" semantics differ from `ValueQuery`'s raise-unless-tolerant default — worth flagging as a semantic decision for any migration, not just a mechanical swap. |
| `_copy_present` | `src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py:983-984` | `{key: source[key] for key in keys if key in source and source[key] is not None}` | in-memory payload | **Full candidate** | Exactly a list of `FieldMapping`s with implicit `exists`-gating per key. |
| `_find_row_by_run_id` / `_run_record` | `src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py:754-763`, `745-751` | Linear-search a list of rows for the one matching `run_id` (or dict-key lookup), returns a copy | in-memory payload | **Full candidate** | `Select(where=Compare(item=entry, path=run_id, op=eq, value=run_id))`. |
| `_certificate_component` | `src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py:788-797` | Tries 3 alternate locations (`certificate_components[name]`, `row[name]`, `metrics[name]`) for a named component dict, returns first hit | in-memory payload | **Partial** | The "try location A, then B, then C" shape is `AnyOf`-adjacent for *existence*, but each branch returns a *different value*, which no node in `expressions.py` composes (see gap note below). |
| `_compact_class_summary`, `_compact_sisu_perturbation_comparison`, `_perturbation_marker_summary` | `src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py:866-948` | Rebuild a compact dict from a source payload: iterate a group map, pull ~5 named fields per group (some via `_nested_get`), drop empty values | in-memory payload | **Full candidate** | Per-group `FieldMapping` list with renames (e.g. `group.get("rows_sisu_0")` → `n_rows_sisu_0`); the "drop `None`/`{}`/`[]`" step is a uniform post-filter, not per-field computation. |
| `_sisu_perturbation_marker_summary` | `src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py:969-981` | Pulls 3 named fields from `run_record["headline"]` | in-memory payload | **Full candidate** | Trivial 3-field extraction; the accompanying `"ratio_meaning"` string is a static literal, not derived. |
| `_public_cost_summary`, `_without_values`, `_cost_source_key` | `src/rlrmp/analysis/pipelines/objective_comparator.py:1669-1694` | Renames known cost-term keys via a lookup dict, strips a `"values"` key from each nested summary | in-memory payload | **Full candidate** | Rename table (`_cost_source_key`) is exactly a `FieldMapping.output_path`↔source-key mapping; `_without_values` is a key-drop projection. |
| `_load_checkpoint_selection` | `src/rlrmp/analysis/pipelines/objective_comparator.py:1827-1829` | `manifest.get("checkpoint_selection", manifest)` — nested-or-self fallback | tracked JSON (loaded at `objective_comparator.py:1856`) | **Partial** | `exists`-style fallback but the "default" is the container itself, not a value — not directly `ValueQuery`-expressible without an explicit `default=self` extension. |
| `_selection_row_from_feedbax_group` / legacy-view assembly | `src/rlrmp/analysis/pipelines/gru_checkpoint_selection.py:1200-1223` | Builds a legacy dict via `.setdefault()` cascades reading `manifest.metadata.get(...)` fields, then reshapes `manifest.selections` into a `runs` map | Feedbax `CheckpointSelectionManifest` (typed object, not raw JSON) | **Partial** | Field renaming is pure extraction; the `sorted(rows, key=lambda row: int(row.get("replicate", 0)))` step is real computation (sort) so the function as a whole is not eligible, only its field-pull portion. |
| `_graph_metadata` fallback chain | `src/rlrmp/runtime/training_run_specs.py:1392-1408` | `payload.get("schema_version") or payload.get("version") or payload.get("$schema")` after reading a tracked graph-spec JSON | tracked JSON (`graph_spec_path`) | **Full candidate (mechanical shape only)** | Per the issue's instruction, this file's fail-closed semantics are deliberate policy — flagged only for the mechanical 3-way fallback-get shape, not as a critique of the policy. |
| `_component_summary` variant | `results/08483d5/scripts/*` (none found — see results-scripts section) | — | — | — | — |
| `minimax_args_from_run_spec` | `src/rlrmp/artifact_migration.py:100-110` | Selects known CLI-flag keys out of a historical run-spec dict into an argparse `Namespace`, with defaults | tracked/legacy JSON run spec | **Partial, low priority** | Pure key-selection-with-defaults (`FieldMapping`-shaped), but it's one-off migration/compat code for a frozen legacy artifact format (`b41c940`), not an ongoing materializer — lower migration value. |

## Category B: predicate/gating candidates

| Site | file:lines | What it does | Sources read | Classification | Blocker/rationale |
|---|---|---|---|---|---|
| `_is_cs_lss_run_spec` | `src/rlrmp/runtime/run_specs.py:264-274` | Checks 4 alternate dotted paths for `plant_backend == CS_LSS_PLANT_BACKEND`, OR one more path `is True` | run-spec dict (in-memory, loaded via `validate_nominal_gru_run_spec_file`) | **Strong candidate** | Textbook `AnyOf([Compare(path=p, op=eq, value=CS_LSS_PLANT_BACKEND) for p in 4 paths] + [Compare(path=..., op=eq, value=True)])`. |
| `_checkpoint_policy_for_manifest`-style dispatch | `src/rlrmp/analysis/pipelines/gru_feedback_ablation.py:990-1001` | `if manifest is not None: return "fixed_bank..." else: return "validation_selected..."` | result of `load_materialized_fixed_bank_manifest` | **Strong candidate** | `Compare(item=manifest, op=exists)`-gated value selection; two-way, not just skip/run, so needs the value-production extension noted in the gap section, but the *condition itself* is directly expressible. |
| `load_checkpoint_selection_legacy_payload` dispatch | `src/rlrmp/analysis/pipelines/gru_checkpoint_selection.py:1226-1236` | `if payload.get("kind") == "CheckpointSelectionManifest": ... else: return dict(payload)` | tracked JSON | **Strong candidate** | `Compare(item=payload, path=kind, op=eq, value="CheckpointSelectionManifest")` — classic kind-dispatch gate. |
| `materialize_feedback_selected_checkpoint_manifest` precondition | `src/rlrmp/analysis/pipelines/gru_feedback_ablation.py:1014-1017` | `if audit.get("status") != "materialized": raise ValueError(...)` | tracked JSON (`feedback_ablation_manifest_path`) | **Candidate (gate only)** | `Compare(op=ne, value="materialized")` is the gate; the surrounding function does real work (rollout/objective computation) so only the precondition check migrates, not the whole function. |
| `fixed_bank_rescore_manifest_status` | `src/rlrmp/analysis/pipelines/gru_postrun_materialization.py:1118-1136` | `path.exists()` early-return, then ternary `"fixed_bank_rescore" if materialization_status == "materialized" else "sparse_history_fallback"` | tracked JSON | **Partial** | The existence gate and the `==` comparison are both direct `Compare` nodes; but the ternary *produces* one of two string values, which again needs an if/then/else value node the current `expressions.py` doesn't have (only `evaluate_expr` → bool). |
| `_add_metric_alerts` threshold checks | `src/rlrmp/analysis/training_diagnostics.py:414-424` | `if ratio_max is not None and ratio_max > 1.0001: ... if gap_min is not None and gap_min < -1e-6: ...` | in-memory JSON-derived dict (from `_read_json`) | **Strong candidate for the gate; not eligible for the alert text** | `AllOf(Compare(exists), Compare(op=gt, value=1.0001))` and the second threshold likewise — clean `Compare` targets. The `alerts.append(f"...:{value:.6g}")` string formatting is real computation and stays outside the query language. |
| `_resolve_bulk_dir` fallback cascade | `src/rlrmp/analysis/rollout_cleanup.py:186-200` | `if bulk_dir is not None: return... elif detail.get("path"): return... elif refs: ...` | tracked cleanup manifest | **Partial** | Same "each branch returns a different value" limitation as above — the *existence checks* are `Compare`/`exists`-shaped, the value production per branch is not. |
| `_normalized_legacy_manifest_payload` conditional patch | `src/rlrmp/artifact_migration.py:166-172` | `if metadata.get("rlrmp_graph_schema_version") == _LEGACY_...VERSION and graph_spec.get("schema_version") is None: graph_spec["schema_version"] = ...` | legacy manifest JSON | **Borderline** | `AllOf(Compare(eq), Compare(op=exists is False))`-shaped gate, but this is a one-off migration/compat shim for a frozen legacy format, not an ongoing analysis gate — lower priority. |
| `load_rlrmp_spec_payload` kind dispatch | `src/rlrmp/runtime/spec_migrations.py:268-273` | `if kind == LEGACY_TRAINING_CONFIG_KIND: raise ArchiveOnlySpecError(...)` | tracked JSON spec | **Borderline** | `Compare(eq)`-shaped, but this is deliberate fail-closed schema-family policy analogous to `training_run_specs.py`'s carve-out — flagged for shape only, not as a migration push. |

**Cross-cutting note on the query-language gap these findings share:** the majority of Category B "hand-rolled gating" sites are not pure skip/run predicates (which `run_condition: Expr` already covers cleanly) — they are **fallback/dispatch expressions that select among several *values*** (a manifest field, a string label, a whole sub-dict). `feedbax.contracts.expressions` currently has no if/then/else or coalesce node — `evaluate_expr` returns only `bool`, and `ValueQuery` has no `default=` or `AnyOf`-over-queries construct. Every "Partial" classification above is blocked by this same missing primitive, not by anything rlrmp-specific. This is worth surfacing as a feedbax-side gap (a `Coalesce`/`Switch` value-expression) rather than reworking each site individually.

## Category C: thin-wrapper deprecation targets — `src/rlrmp/data_products/`

### `envelope.py`

| Function | Classification | Rationale | Non-test callers | Test-caller files |
|---|---|---|---|---|
| `read_data_product` | (ii) thin envelope assembly | Reads JSON, wraps `AnalysisDataProduct.model_validate`, translates pydantic errors to `DataProductError` | none outside module (only called by `load_data_product`) | 0 |
| `validate_data_product` | (iii) carries real logic | 7-branch fail-closed comparison against `AnalysisDataProductRequirement` (schema id/version, role, basis hash, identity hash, artifact hash) — this *is* the contract enforcement | none outside module | 1 |
| `load_data_product` | (i) pass-through | `read_data_product` + `validate_data_product`, returns the product | none outside `data_products/` (all real callers go through per-product `load_*` wrappers in `calibration.py`/`broad_epsilon.py`) | 2 |

Identity-pin constants: none in this module (it's generic; identities live in `calibration.py`/`broad_epsilon.py`).

### `registry.py`

| Function | Classification | Rationale | Non-test callers | Test-caller files |
|---|---|---|---|---|
| `register_data_product_identity` | (iii) carries real logic | Collision detection across 3 keys (`role`, `product_schema_id`, `logical_name`) with `setdefault` idempotency | called at import time by `calibration.py` (×2) and `broad_epsilon.py` (×1) — module-level side effects, not runtime call sites | 1 |
| `registered_data_product_identities` | (ii) thin envelope assembly | Returns a `MappingProxyType` copy of the module-level registry dict | none found outside module/tests | 1 |

Identity-pin constants: none (this module *holds* the registry, doesn't define product identities).

### `calibration.py`

| Function | Classification | Rationale | Non-test callers | Test-caller files |
|---|---|---|---|---|
| `build_open_loop_calibration_product` | (ii) thin envelope assembly | Wraps 4 scalar/dict args into an `AnalysisDataProduct` with fixed schema/role/metadata | none found (no script currently calls it — see note below) | 0 |
| `write_open_loop_calibration_product` | (i) pass-through | `model_dump_json` + `write_text` | none found | 0 |
| `build_perturbation_calibration_defaults_payload` | (ii) thin envelope assembly | Assembles the adopted-defaults JSON document from 6 dataclass sequences | none found | 1 |
| `write_perturbation_calibration_defaults_payload` | (i) pass-through | `json.dumps` + `write_text` | none found | 0 |
| `build_perturbation_calibration_defaults_product` | (ii) thin envelope assembly | Wraps payload sha256 + adoption records into an `AnalysisDataProduct` with an `ArtifactRef` | referenced only inside its own `producer_manifest_id` string in the persisted JSON (`results/ea6ccb4/data_products/perturbation_calibration_defaults.json`), i.e. **zero live callers** | 0 |
| `write_perturbation_calibration_defaults_product` | (i) pass-through | `model_dump_json` + `write_text` | none found | 0 |
| `calibration_data_product_requirement` | (ii) thin envelope assembly | Builds a fixed `AnalysisDataProductRequirement` from module constants | called only internally (`load_open_loop_calibration`) | 1 |
| `calibration_defaults_data_product_requirement` | (ii) thin envelope assembly | Same, for the defaults product | called only internally (`load_perturbation_calibration_defaults`) | 1 |
| `load_open_loop_calibration` | (iii) carries real logic | Fail-closed load + typed-dataclass projection (`OpenLoopCalibration`), `@lru_cache` | **6 call sites**: `analysis/pipelines/gru_perturbation_bank.py:519,523`; `train/cs_perturbation_training.py:33,2115,4687,4690,4708,4767,4952,4973,5521` (many uses within one file) | 3 |
| `load_perturbation_calibration_defaults` | (iii) carries real logic | Fail-closed load + byte-hash re-verification of a sidecar payload artifact + typed-dataclass projection (6 nested dataclasses), `@lru_cache` | **3 call sites**: `analysis/pipelines/gru_perturbation_calibration.py` (7 uses), `analysis/pipelines/gru_perturbation_bank.py:135,137,520,524`, `train/cs_perturbation_training.py` (6 uses) | 1 |
| `open_loop_peak_delta_x_per_unit` | (i) pass-through accessor | `load_open_loop_calibration().peak_delta_x_per_unit` | `train/cs_perturbation_training.py:2139,2152` (string label + payload assembly, not a real "call" of the accessor itself — actually calls `.peak_delta_x_per_unit` via the dataclass directly at 2152) | 0 |
| `controller_visible_velocity_scale_m_s` | (i) pass-through accessor | `load_open_loop_calibration().controller_visible_velocity_scale_m_s` | none call the *function*; callers instead go through `calibration.controller_visible_velocity_scale_m_s` (the dataclass attribute) directly in `gru_perturbation_bank.py:525,857,902` and `cs_perturbation_training.py:2149-2150,5521` | 0 |
| `consumed_calibration_identity` | (ii) thin envelope assembly | `{role, schema, hash}` triple from a loaded product | `train/cs_perturbation_training.py:2188`, `eval/recipes.py:518,520` | 0 |
| `consumed_perturbation_calibration_defaults_identity` | (ii) thin envelope assembly | Same shape for the defaults product | `analysis/pipelines/gru_perturbation_calibration.py:946` | 0 |

Identity-pin constants: `CALIBRATION_PRODUCT_IDENTITY_HASH`, `CALIBRATION_DEFAULTS_PRODUCT_IDENTITY_HASH`, `CALIBRATION_DEFAULTS_PAYLOAD_SHA256`. All three are referenced **only inside this module** (the two `*_data_product_requirement()` functions and `_calibration_defaults_payload_artifact`) — no caller reaches around the loaders to touch the hashes directly, which is the intended fail-closed shape.

**Note:** `build_open_loop_calibration_product`/`write_open_loop_calibration_product` and `build_perturbation_calibration_defaults_product`/`write_perturbation_calibration_defaults_product` have **zero non-test callers** in the current tree — no `scripts/materialize_*` counterpart was found for the open-loop calibration product (unlike `broad_epsilon`, which has `scripts/materialize_broad_epsilon_budget_anchors.py`). These builder/writer pairs appear to be either (a) exercised only via tests/one-off regeneration, or (b) dead code the CI gate doesn't catch because it targets baked-constant literals, not unused loader-side builders. Worth a follow-up check, out of scope for this sweep.

### `broad_epsilon.py`

| Function | Classification | Rationale | Non-test callers | Test-caller files |
|---|---|---|---|---|
| `build_broad_epsilon_budget_anchors_product` | (i) pass-through | Single call to `materialize_extraction_product(spec, REPO_ROOT)`, wraps the identity-mismatch exception | `scripts/materialize_broad_epsilon_budget_anchors.py:18,48` | 0 |
| `verify_broad_epsilon_budget_anchors_product` | (i) pass-through | `load_data_product` + `verify_extraction_product`, wraps `DataProductDrift` | `scripts/materialize_broad_epsilon_budget_anchors.py:19,53` (also called internally by `load_broad_epsilon_anchors`) | 0 |
| `write_broad_epsilon_budget_anchors_product` | (i) pass-through | `model_dump_json` + `write_text` | `scripts/materialize_broad_epsilon_budget_anchors.py:20,51` | 0 |
| `broad_epsilon_data_product_requirement` | (ii) thin envelope assembly | Fixed `AnalysisDataProductRequirement` from module constants | called only internally | 1 |
| `load_broad_epsilon_anchors` | (iii) carries real logic | Verify (re-run extraction + drift check) + per-level contract-key projection into `BroadEpsilonAnchors`, `@lru_cache` | **`analysis/pipelines/gru_worst_case_epsilon_audit.py:190`**; **`train/cs_perturbation_training.py:293,310,413,500`** | 1 |
| `consumed_broad_epsilon_identity` | (ii) thin envelope assembly | `{role, schema, hash}` triple | `train/cs_perturbation_training.py:2190` | 0 |
| `_contract` (private) | n/a (not public) | `{key: anchor[key] for key in _CONTRACT_KEYS}` — a plain `FieldMapping` projection over an already-loaded in-memory dict | called by `load_broad_epsilon_anchors` only | — |

Identity-pin constant: `BROAD_EPSILON_PRODUCT_IDENTITY_HASH`. Referenced only inside `broad_epsilon_data_product_requirement()`. This module is **already migrated to the declarative extraction engine** (`ExtractionProductSpec`/`materialize_extraction_product`/`verify_extraction_product`) — it is the strongest existing precedent in the repo for what a fully-migrated Category-A site looks like, and its own `_contract()` post-load projection (Category A, listed above) is the one remaining hand-rolled bit inside an otherwise-migrated module.

### `lint.py`

Not in scope for build/write/verify/load classification (its public surface is `scan_source`/`scan_tree`/`violations`, an AST-based CI lint, not a data-product loader). Flagged `borderline`/out-of-scope: it is itself a hand-rolled pattern-matcher (AST walk + name-hint heuristics), but over Python source, not tracked JSON — the query language doesn't apply to this class of problem.

### `__init__.py`

Pure re-export surface (`i` pass-through by construction); no findings.

### Category C summary

- **(i) pass-through:** 10 functions — `read_data_product`, `load_data_product`, `write_open_loop_calibration_product`, `write_perturbation_calibration_defaults_payload`, `write_perturbation_calibration_defaults_product`, `open_loop_peak_delta_x_per_unit`, `controller_visible_velocity_scale_m_s`, `build_broad_epsilon_budget_anchors_product`, `verify_broad_epsilon_budget_anchors_product`, `write_broad_epsilon_budget_anchors_product`.
- **(ii) thin envelope assembly:** 11 functions.
- **(iii) carries real logic:** 5 functions — `validate_data_product`, `register_data_product_identity`, `load_open_loop_calibration`, `load_perturbation_calibration_defaults`, `load_broad_epsilon_anchors`.

**Strongest thin-wrapper deprecation candidates:** `open_loop_peak_delta_x_per_unit` and `controller_visible_velocity_scale_m_s` — both are one-line pass-through accessors over `load_open_loop_calibration()` with effectively zero direct callers (real callers already reach through the dataclass attribute directly), making them the cleanest "delete and inline" candidates. `write_open_loop_calibration_product` / `build_perturbation_calibration_defaults_product` are candidates for a different reason: zero live callers at all, so before deprecating anything, confirm whether they're still needed for regeneration.

**Must stay (real logic, load-bearing):** `load_open_loop_calibration`, `load_perturbation_calibration_defaults`, `load_broad_epsilon_anchors` — each is the sole fail-closed entry point for governed data consumed by `train/cs_perturbation_training.py`'s adversary-budget logic; collapsing them into inline extraction would defeat the whole point of the `ea6ccb4` policy (identity pinning, byte-hash re-verification, cache).

## Category D: borderline / other

| Site | file:lines | What it does | Sources read | Rationale |
|---|---|---|---|---|
| `_section_for_artifact` | `src/rlrmp/analysis/reports.py:245-274` | Dispatches on `artifact.media_type`/`path.suffix` to decide whether to parse JSON, read as text, or mark unsupported | `ArtifactRef` (typed, in-memory) | `borderline` — media-type dispatch produces structurally different output shapes per branch (not just skip/run), so it's dispatch-shaped like Category B but the branches aren't eligibility gates, they're type-directed readers. |
| `_stage_params_payload` | `src/rlrmp/analysis/reports.py:235-241` | `if isinstance(params.get("stage_params"), Mapping): return dict(stage_params); return params` | report spec params | `borderline` — single `exists`+`has_type`-shaped gate producing one of two whole-dict values; same value-production gap as Category B findings. |
| `_read_json` (multiple modules) | `hinf_phenotype_sidecar.py:1046`, `gru_map_error_decomposition.py:520-522`, `cs_gru_standard_materialization.py:1387-1388`, `rollout_cleanup.py`, `training_diagnostics.py:430-431`, `objective_comparator.py` (via `args.manifest.read_text`) | Each module independently defines its own trivial `_read_json(path) -> dict` helper (read + `json.loads`) | tracked JSON | `borderline` — not itself extraction/gating, but **6 independent reimplementations of the same one-line helper** across the audited files is a real duplication smell adjacent to this sweep's purpose (a shared `rlrmp`-level `read_json` utility, or routing all of these through `SourceBinding`/`ExtractionProductSpec` sources, would remove the duplication at the root). |
| `_graph_metadata` | `src/rlrmp/runtime/training_run_specs.py:1392-1408` | Reads a graph-spec sidecar JSON opportunistically (try/except around the read) purely to extract a version string for provenance metadata | tracked JSON | `borderline` — flagged per the issue's instruction to note only the mechanical path-walking shape of this file, not critique its fail-closed policy. |
| `load_rlrmp_spec_payload` | `src/rlrmp/runtime/spec_migrations.py:258-273` | Loads a JSON spec, dispatches by `kind`, routes through `accept_rlrmp_spec_payload` → `migrate_structured_spec_payload` | tracked JSON | `borderline` — schema-family migration policy (deliberate), not a materializer; the dispatch shape is `Compare`-expressible but this is infrastructure code with different stakes than an analysis materializer. |
| `_validate_cs_lss_graph_spec_sidecar` | `src/rlrmp/runtime/run_specs.py:288-300` | Reads a graph-spec sidecar JSON, validates `nodes` is a dict, then further structural checks (truncated in this sweep) | tracked JSON | `borderline` — validation-shaped code; likely has both `Compare`-expressible existence/type gates and real structural-invariant checks past what was read here. Flagging as a follow-up read target rather than a full classification. |
| `benchmarks/local_parallel.py`, `benchmarks/packing.py` status-file reads | `local_parallel.py:396,467`; `packing.py:673,773` | Poll worker status/summary JSON files during a local-parallel run sweep | tracked-at-runtime status JSON (not `results/`-tracked manifests) | `borderline` — these are ephemeral run-orchestration status files, not governed analysis manifests; the query language's target domain (tracked `results/` JSON) doesn't obviously apply, but the shape (read + `.get()` chain) is structurally similar. Out of scope by domain, included for completeness. |
| `_add_regeneration_pointer` / `_write_index` | `scripts/backfill_gru_regeneration_specs.py:460-486` | One-off backfill script: reads a manifest, patches in a `regeneration_spec` pointer, writes an index | tracked JSON | `borderline` — trivial extraction/patch shape, but this is a one-shot historical backfill tied to a specific past issue, not an ongoing materializer; low migration value. |
| `render()` cost-summary rendering | `results/c723082/scripts/build_cross_method_comparison.py:89-` | Reads a tracked summary JSON, renders a comparison table | tracked JSON | `borderline` — the file's headline function (`_aggregate_by_eta`) is real statistical aggregation (median/MAD, correctly excluded from Category A above); the render-side field access wasn't fully read in this lighter pass and may contain smaller extraction-shaped sub-patterns. |

---

## Skipped files (LEGACY-bannered)

Full top-of-file `"""LEGACY (frozen ..., issue ...)."""` banners — entire file skipped per instructions:

- `src/rlrmp/analysis/pipelines/bridge_aggregation.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/delayed_diagnostic_bundle.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/cs_stochastic_phase1.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/cs_stochastic_phase3.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/failure_decomposition.py` (frozen 2026-07-03, issue 64d5f13) — **note:** this file contains its own copy of the `_component_summary` Select-pattern helper (see Category A), i.e. the duplicate predates at least one freeze.
- `src/rlrmp/analysis/pipelines/output_feedback_interpolated_starts.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/output_feedback_affine_tracker.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/sisu_perturbation_comparison.py` (frozen 2026-07-03, issue 64d5f13)
- `src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py` (frozen 2026-07-03, issue 64d5f13)

Partial-legacy (module has `LEGACY_*`-prefixed constants/schema versions but is NOT a frozen file — read normally, legacy identifiers just noted): `src/rlrmp/analysis/pipelines/gru_checkpoint_selection.py` (`LEGACY_SPARSE_HISTORY_SCHEMA_VERSION`, `LEGACY_FIXED_BANK_SCHEMA_VERSION`, `LEGACY_DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION`) and `src/rlrmp/analysis/pipelines/gru_feedback_ablation.py` (`LEGACY_FIXED_BANK_SCHEMA_VERSION`). (`output_feedback_phase_modulated_recurrent.py`'s `LEGACY_AUDIT_RANKS`/`LEGACY_SUPERVISED_ROW_FAMILIES` constants are inside a fully-frozen file already listed above, not a separate partial case.)

`tests/` was out of scope except caller counts, which are reported inline in the Category C tables above.

---

## Summary counts

| Classification | Count |
|---|---|
| Category A — full extraction/ValueQuery candidates | 9 |
| Category A — partial (blocked by computation or missing value-production primitive) | 5 |
| Category B — strong predicate/gating candidates | 4 |
| Category B — partial/gate-only or borderline-policy | 5 |
| Category C — pass-through (i) | 10 |
| Category C — thin envelope (ii) | 11 |
| Category C — real logic (iii), must stay | 5 |
| Category D — borderline/other | 9 |
| Files fully skipped (LEGACY banner) | 11 |

## Top five sites to migrate first

1. **The `_component_summary` Select-by-name helper, reimplemented independently in 4 (arguably 5, counting the LEGACY copy) locations** — `analysis/pipelines/gru_map_error_decomposition.py:509`, `analysis/pipelines/failure_decomposition.py` (frozen), `scripts/materialize_output_feedback_failure_decomposition.py:837`, `scripts/materialize_output_feedback_sweep_certificates.py:1285`. Same `Select(where=Compare(name==name)) + ValueQuery(path="summary.<key>")` shape every time, with copy-paste drift already visible (one variant added a second predicate on `status`). Highest value because it's not one site, it's a pattern that has already diverged across copies — a single declarative `Select`/`ValueQuery` would prevent the next divergence.
2. **`hinf_phenotype_sidecar.py`'s extraction cluster** (`_nested_get`, `_copy_present`, `_find_row_by_run_id`, `_compact_class_summary`, `_compact_sisu_perturbation_comparison`, `_perturbation_marker_summary`, `_sisu_perturbation_marker_summary`) — one file, ~10 functions, all pure path-walk/select/rename over sidecar-source payloads, no computation. This is the single richest concentration of Category-A material in the repo and the most self-contained migration unit.
3. **`_is_cs_lss_run_spec`** (`runtime/run_specs.py:264`) — a clean 5-way `AnyOf` predicate over run-spec fields, already gating real validation control flow (whether to run `_validate_cs_lss_graph_spec_sidecar`). Small, isolated, and a good "first declarative predicate" proof point outside the analysis-materializer world.
4. **`broad_epsilon.py`'s `_contract()` post-load projection** — the one remaining hand-rolled piece in a module that is *already* the reference implementation of `ExtractionProductSpec` end-to-end. Migrating this closes the loop and gives a template for migrating `calibration.py`'s loaders (`load_open_loop_calibration`/`load_perturbation_calibration_defaults`) the same way, since those still do their post-load projection by hand rather than via a second extraction spec layer.
5. **`gru_checkpoint_selection.py`'s `load_checkpoint_selection_legacy_payload` kind-dispatch + `gru_feedback_ablation.py`'s manifest-presence dispatch** — both are `Compare`-shaped gates sitting directly in front of real materializer logic (not buried in one-off scripts), so migrating them buys the most leverage per site: one declarative gate replaces a branch that currently silently diverges between "legacy bytes" and "Feedbax manifest" code paths.

Output file: `/private/tmp/claude-501/-Users-mll-Main-10-Projects-10-PhD-rlrmp/08a431d4-ec60-4f13-ab0e-b9622d4fe1d8/scratchpad/audit/96ac0e5_query_language_sweep.md`
