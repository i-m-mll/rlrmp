# Category C partial (data_products/) — from audit agent

## Category C: data_products/ wrapper classification

### __init__.py
Pure re-export module (39 lines). Imports and re-exports names from `broad_epsilon.py`, `calibration.py`, `envelope.py` via `__all__`; contains no logic of its own.

### broad_epsilon.py

| function | file:line | classification | non-test callers | test callers | identity-pin constants used |
|---|---|---|---|---|---|
| `build_broad_epsilon_budget_anchors_product()` | broad_epsilon.py:139 | (i) pass-through — calls `materialize_extraction_product(spec, REPO_ROOT)`, wraps `ExtractionProductIdentityMismatch` into `DataProductError` | scripts/materialize_broad_epsilon_budget_anchors.py:48 | 0 | none directly (uses cached spec) |
| `verify_broad_epsilon_budget_anchors_product(*, path=None)` | broad_epsilon.py:152 | (i) pass-through — loads via `load_data_product`, then calls `verify_extraction_product`, wraps `DataProductDrift` | scripts/materialize_broad_epsilon_budget_anchors.py:53 | 0 | `BROAD_EPSILON_PRODUCT_PATH` (default arg) |
| `write_broad_epsilon_budget_anchors_product(product, *, path=None)` | broad_epsilon.py:170 | (ii) thin envelope assembly — `mkdir_p` + `model_dump_json` write | scripts/materialize_broad_epsilon_budget_anchors.py:51 | 0 | `BROAD_EPSILON_PRODUCT_PATH` |
| `broad_epsilon_data_product_requirement()` | broad_epsilon.py:186 | (ii) thin envelope assembly — builds `AnalysisDataProductRequirement` from module constants | none outside module (used internally by `load_broad_epsilon_anchors`→`load_data_product` at broad_epsilon.py:159, and as `requirement_factory` in `register_data_product_identity` at broad_epsilon.py:203) | 0 | `BROAD_EPSILON_PRODUCT_ROLE`, `BROAD_EPSILON_PRODUCT_SCHEMA_ID`, `BROAD_EPSILON_PRODUCT_SCHEMA_VERSION`, `BROAD_EPSILON_PRODUCT_LOGICAL_NAME`, `BROAD_EPSILON_PRODUCT_IDENTITY_HASH` (all directly, in its own body) |
| `load_broad_epsilon_anchors()` | broad_epsilon.py:209 | (iii) carries real logic — fail-closed verify, then reshapes `product.parameters["levels"]` into the legacy `BroadEpsilonAnchors` typed contract, checking `_EXPECTED_LEVELS` membership and re-keying via `_contract()`/`_CONTRACT_KEYS` (a hand-rolled field-select) | src/rlrmp/analysis/pipelines/gru_worst_case_epsilon_audit.py:190; src/rlrmp/train/cs_perturbation_training.py:293,310,413,500 | 1 file (`tests/test_lane_c_terminal_gate.py`, 3 occurrences — import/incidental-reference count rather than confirmed direct test invocation) | `BROAD_EPSILON_PRODUCT_IDENTITY_HASH` not touched by callers — callers only go through the loader/typed accessor |
| `consumed_broad_epsilon_identity()` | broad_epsilon.py:236 | (ii) thin envelope assembly — assembles `{role, schema, hash}` dict from a loaded `BroadEpsilonAnchors` | src/rlrmp/train/cs_perturbation_training.py:2190 | 0 | none directly (delegates to `load_broad_epsilon_anchors()` for the hash) |

### calibration.py

| function | file:line | classification | non-test callers | test callers | identity-pin constants used |
|---|---|---|---|---|---|
| `build_open_loop_calibration_product(*, peak_delta_x_per_unit, controller_visible_velocity_scale_m_s, controller_visible_force_filter_scale_n=1.0, reference_reach_m=0.15)` | calibration.py:313 | (ii) thin envelope assembly — constructs `AnalysisDataProduct` from module schema/role/producer constants + `_calibration_parameters()` | **0** (no caller anywhere in src/scripts/results — the actual production materializer path is `rlrmp.analysis.pipelines.gru_perturbation_calibration.materialize_perturbation_open_loop_calibration`, invoked by `scripts/materialize_perturbation_open_loop_calibration.py`, which does not use this function) | 0 | n/a |
| `write_open_loop_calibration_product(product, *, path=None)` | calibration.py:351 | (ii) thin envelope assembly — `mkdir_p` + JSON write | 0 | 0 | `CALIBRATION_PRODUCT_PATH` |
| `build_perturbation_calibration_defaults_payload(*, amplitude_factors, reach_calibration_points, reach_relative_levels, plant_timing_bins, controller_visible_timing_bins, native_conventions)` | calibration.py:367 | (ii) thin envelope assembly — dict assembly, delegates per-item shaping to each dataclass's `.to_json()` | 0 | `tests/test_product_identity_hash.py` (1 file) | none |
| `write_perturbation_calibration_defaults_payload(payload, *, path=None)` | calibration.py:397 | (ii) thin envelope assembly — JSON write | 0 | 0 | `CALIBRATION_DEFAULTS_PAYLOAD_PATH` |
| `build_perturbation_calibration_defaults_product(*, payload_sha256, payload_relpath=CALIBRATION_DEFAULTS_PAYLOAD_RELPATH)` | calibration.py:413 | (ii) thin envelope assembly — `AnalysisDataProduct` + `ArtifactRef` construction from constants + `CALIBRATION_DEFAULTS_ADOPTION_RECORDS` | 0 | 0 | `CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_ID/VERSION/ROLE/LOGICAL_NAME`, `CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_ID/VERSION` |
| `write_perturbation_calibration_defaults_product(product, *, path=None)` | calibration.py:467 | (ii) thin envelope assembly — JSON write | 0 | 0 | `CALIBRATION_DEFAULTS_PRODUCT_PATH` |
| `calibration_data_product_requirement()` | calibration.py:483 | (ii) thin envelope assembly — builds `AnalysisDataProductRequirement` from constants | internally used only (via `load_open_loop_calibration`→`load_data_product` at calibration.py:531-534, and as `requirement_factory` at calibration.py:513) | `tests/test_product_identity_hash.py` (1 file, 5 occurrences) | `CALIBRATION_PRODUCT_ROLE/SCHEMA_ID/SCHEMA_VERSION/LOGICAL_NAME/IDENTITY_HASH` (directly, in its own body) |
| `calibration_defaults_data_product_requirement()` | calibration.py:495 | (ii) thin envelope assembly | internally used only (calibration.py:522, calibration.py:557-560) | `tests/test_product_identity_hash.py` (1 file, 4 occurrences) | `CALIBRATION_DEFAULTS_PRODUCT_*` + `CALIBRATION_DEFAULTS_PAYLOAD_SHA256` (directly) |
| `load_open_loop_calibration()` | calibration.py:528 | (iii) carries real logic — fail-closed load + reshapes `product.parameters` dict into typed `OpenLoopCalibration`, coercing nested table values to float | src/rlrmp/analysis/pipelines/gru_perturbation_bank.py:519,523; src/rlrmp/train/cs_perturbation_training.py:33,2115,4687,4690,4708,4767,4952,4973,5521 (heavily used — indexes result via `calibration[family][timing]`, and `.controller_visible_velocity_scale_m_s` attribute) | 3 files, 8 occurrences | callers never touch `CALIBRATION_PRODUCT_IDENTITY_HASH` directly — always go through the loader |
| `load_perturbation_calibration_defaults()` | calibration.py:553 | (iii) carries real logic — fail-closed load, artifact-uri/hash cross-check (`_calibration_defaults_payload_artifact`, `_sha256_file`), payload read+schema validation, then manual field-by-field reconstruction of 6 typed dataclass tuples from the JSON payload | src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py:95,260,266,272,278,284,290; src/rlrmp/analysis/pipelines/gru_perturbation_bank.py:135,137,520,524; src/rlrmp/train/cs_perturbation_training.py:34,201,216,220,2006,5640,5647 | 1 file, 6 occurrences | callers never touch `CALIBRATION_DEFAULTS_PRODUCT_IDENTITY_HASH`/`CALIBRATION_DEFAULTS_PAYLOAD_SHA256` directly — always via the loader |
| `open_loop_peak_delta_x_per_unit()` | calibration.py:632 | (i) pass-through (module-level convenience wrapper around `load_open_loop_calibration().peak_delta_x_per_unit`) | **0** — dead: all real call sites access `calibration.peak_delta_x_per_unit` as an attribute on a directly-loaded `OpenLoopCalibration` instance (e.g. src/rlrmp/train/cs_perturbation_training.py:2152) | 0 | n/a |
| `controller_visible_velocity_scale_m_s()` | calibration.py:638 | (i) pass-through (same shape as above, wraps `.controller_visible_velocity_scale_m_s`) | **0** — dead for the same reason; all real usage (src/rlrmp/analysis/pipelines/gru_perturbation_bank.py:525,857,902; src/rlrmp/train/cs_perturbation_training.py:2150,5521) is via the `OpenLoopCalibration` attribute directly | 0 | n/a |
| `consumed_calibration_identity()` | calibration.py:644 | (ii) thin envelope assembly — `{role, schema, hash}` dict | src/rlrmp/train/cs_perturbation_training.py:2188; src/rlrmp/eval/recipes.py:518,520 | 0 | none directly |
| `consumed_perturbation_calibration_defaults_identity()` | calibration.py:655 | (ii) thin envelope assembly | src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py:946 | 0 | none directly |

### envelope.py

| function | file:line | classification | non-test callers | test callers | identity-pin constants used |
|---|---|---|---|---|---|
| `read_data_product(path)` | envelope.py:56 | (i) pass-through — JSON read + `AnalysisDataProduct.model_validate`, wraps I/O and Pydantic errors into `DataProductError` | internally only, via `load_data_product` (envelope.py:187) | 0 | n/a (generic, no schema pin belongs here) |
| `validate_data_product(product, requirement, *, source="<product>")` | envelope.py:90 | (iii) carries real logic — the actual fail-closed contract check: recomputes `analysis_data_product_identity_hash`, then checks `logical_name`, `product_schema_id`, schema-version compatibility, `role`, `descriptor_basis_hash`, pinned `product_identity_hash`, pinned `artifact_sha256` — core domain logic, not replaceable by a generic engine call | internally only, via `load_data_product` (envelope.py:188) | `tests/test_product_identity_hash.py` (2 occurrences) | operates generically on whatever `AnalysisDataProductRequirement` it's given — no module-level identity constants of its own |
| `load_data_product(path, requirement)` | envelope.py:181 | (i) pass-through — composes `read_data_product` + `validate_data_product` | src/rlrmp/data_products/calibration.py:531,557; src/rlrmp/data_products/broad_epsilon.py:159 (all intra-package) | 2 files, 7 occurrences | n/a (takes requirement as a parameter) |

Note: `envelope.py` has no non-test callers *outside* `data_products/` itself for any of its three functions — every external consumer goes through the typed loaders in `broad_epsilon.py`/`calibration.py`, which in turn call `load_data_product`. Expected/by-design (envelope.py is the generic engine the sibling modules wrap).

### lint.py

lint.py has no loader-shaped public functions in the sense this audit cares about — it is the AST-based CI linter for the `generated_data_constant_scan` family (`ci/feedbax-contract-suite.toml:156`), gated through `tests/test_data_lint_generated_constants.py` and `tests/test_lane_c_terminal_gate.py`. Its four public functions:

| function | file:line | classification | non-test callers | test callers | identity-pin constants used |
|---|---|---|---|---|---|
| `significant_figures(value)` | lint.py:107 | (iii) carries real logic — sig-fig counting from `repr()` mantissa | 0 | 4 occurrences (1 file) | n/a |
| `scan_source(text, relpath)` | lint.py:143 | (iii) carries real logic — AST-parses one module's top-level assignments, applies the high-precision/cardinality/science-data-path heuristics | 0 | 4 occurrences (1 file) | `HIGH_PRECISION_SIGFIG_THRESHOLD`, `MIN_HIGH_PRECISION_CARDINALITY` (internal) |
| `scan_tree(src_root, *, repo_root)` | lint.py:192 | (iii) carries real logic — walks `*.py` under a root, excludes `__pycache__`, calls `scan_source` per file | internally only (via `violations`, lint.py:220) | 1 occurrence | n/a |
| `violations(src_root, *, repo_root)` | lint.py:211 | (ii) thin filtering wrapper — `scan_tree` result minus `ALLOWLIST` keys | **0** — no non-test caller anywhere; only exercised by the CI gate's own test files | 2 files | `ALLOWLIST` (internal, keyed `"<relpath>::<name>"`) |

### registry.py

| function | file:line | classification | non-test callers | test callers | identity-pin constants used |
|---|---|---|---|---|---|
| `register_data_product_identity(*, role, product_schema_id, product_schema_version, logical_name, requirement_factory, document_relpath)` | registry.py:33 | (iii) carries real logic — registry side effect (`_DATA_PRODUCT_IDENTITIES.setdefault`) plus collision detection across `role`/`product_schema_id`/`logical_name` keys | src/rlrmp/data_products/broad_epsilon.py:198; src/rlrmp/data_products/calibration.py:508,517 (all intra-package, module-level at import time) | 2 occurrences (1 file) | n/a (constants come from each caller) |
| `registered_data_product_identities()` | registry.py:63 | (ii) thin envelope assembly — returns a `MappingProxyType` snapshot of the registry dict | **0** — no non-test caller anywhere | 1 occurrence | n/a |

## Secondary: extraction-shaped logic inside data_products/

| site | file:lines | what it does | note |
|---|---|---|---|
| `_contract()` | broad_epsilon.py:113-114 | Hand-rolled field select/rename: `{key: anchor[key] for key in _CONTRACT_KEYS}` picks 6 named fields out of a persisted-product dict per level | Small, but exactly the select-fields-by-name shape `ValueQuery`/`FieldMapping` covers |
| `load_perturbation_calibration_defaults()` body | calibration.py:579-629 | Six parallel hand-rolled loops, each iterating a JSON list from the payload and constructing a typed dataclass instance field-by-field with manual `str()`/`float()`/`int()` coercion per field | Largest extraction-shaped candidate in this directory — a per-row select+coerce+rename loop repeated 6 times, structurally identical to what `ExtractionProductSpec`/`FieldMapping`/`ValueQuery` with a list-of-records source could express |
| `_read_calibration_defaults_payload()` / `_validate_calibration_defaults_payload()` | calibration.py:695-746 | Hand-rolled JSON read + required-keys/list-non-empty schema check | Smaller; mostly presence/shape validation, but the payload-schema-id/version/role checks (lines 720-732) are a hand-rolled `Compare(..., op=eq)` triple |
