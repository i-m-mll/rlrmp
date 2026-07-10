# Inventory replay and closure evidence

Replay source:
`_artifacts/b6136cc/second_order/data_in_code_inventory.jsonl`

- SHA-256: `93bbdb93de3c7c9a6e93c911026cc5246cdbb65b8c85c9d12ceaba236063baf8`
- Total inventory rows: 1,440
- RLRMP inventory rows: 1,437
- Matching rule: same surviving module and exact inventory qualname, or a detector finding
  that owns that qualname as an enclosing/contained object.
- Source existence and AST-object existence are reported separately because sibling work
  removed historical objects after the inventory was created.

## Replay results

| Inventory payload | Inventory rows | Source survives | Exact object survives | Raw caught | Raw rate |
|---|---:|---:|---:|---:|---:|
| `planner_rows` | 29 | 15 | 4 | 0 | 0.00% |
| `default_param_bundle` | 25 | 25 | 25 | 19 | 76.00% |
| `run_hyperparameters` | 346 | 316 | 305 | 244 | 77.22% |

The raw rate intentionally includes every surviving file in the inventory class. It is not
the static eligibility rate.

### D1 planner rows

The historical planner functions were migrated/deleted by sibling issue `6add6b2`, so
inventory replay is not a meaningful recall test for their former syntax. Four inventory
qualnames still resolve, but none retains the D1 argv-row shape. The live contract is
stronger than a historical percentage: D1 in `src` is enforced, and the one live smoke
command finding is a rationale-bearing allowlist entry. No D1 `src` finding can be hidden
in the ratchet baseline.

### D3 default bundles

Static eligibility requires a bundle object to own a recoverable literal dictionary,
literal `Namespace` expansion, incremental literal dictionary update, or static config/
parameter defaults. Result: **19/19 caught (100.00%)**, above the 95% target.

The six raw-denominator exclusions are:

| Object | Exclusion reason |
|---|---|
| `results/08483d5/scripts/run_cap_free_direct_epsilon_adaptive_replay.py::OptimizerConfig` | Typed fields have no defaults; values come from parsed arguments. |
| `results/b413bb0/scripts/materialize_beta1p4_stabilization_diagnostics.py::StabilizationRowSource` | Descriptive row-source model with string fields and no numeric parameter bundle. |
| `src/rlrmp/analysis/pipelines/bridge_certificates.py::CertificateNumerics` | Numerical tolerances/floors, intentionally excluded by the criterion. |
| `src/rlrmp/analysis/pipelines/failure_decomposition.py::FailureDecompositionNumerics` | Numerical tolerances/diagnostic thresholds, not a run-parameter bundle. |
| `src/rlrmp/eval/recipes.py::CenterOutEnsembleEvalParams` | Governed schema model with `None` and empty-container defaults, not literal run values. |
| `src/rlrmp/train/standard.py::build_hps` | Dynamic merge/override coordinator; its literal source bundles are independently caught. |

### D4 named literal constants

Eligibility requires a surviving exact constant assignment with a numeric literal tree and
no dimension, tolerance, schema, path, label, or other named exemption. Result:
**241/241 caught (100.00%)**, above the 95% target, with no eligible misses.

The broader raw `run_hyperparameters` rate is 244/316 (77.22%). The 72 uncaught rows include
deleted objects, function-owned/dynamic values, neutral names, and other shapes outside D4.
Per the committed gate specification, that raw percentage is report-only because bare
scalar intent and indirect D2 flow cannot be decided reliably from the declaration syntax.
D2 has one live direct-spec finding in
`src/rlrmp/train/minimax.py::build_minimax_training_run_spec`; its positive and negative
canaries pass.

## Final live-tree shape

The strengthened scanner reports 318 findings:

| Detector | `src` | `scripts` | `results_scripts` | Total |
|---|---:|---:|---:|---:|
| `argv_rows` | 1 | 0 | 1 | 2 |
| `spec_flow` | 1 | 0 | 0 | 1 |
| `default_bundle` | 42 | 0 | 16 | 58 |
| `hp_constant` | 89 | 2 | 164 | 255 |
| `empirical_table` | 1 | 0 | 1 | 2 |

Policy shape: 2 enforced, 136 ratchet, and 180 advisory findings. The two enforced findings
are both curated exceptions: the fixed smoke-command bounds and the inherited D5 empirical
table exception. There are zero policy violations.

The generated ratchet baseline now contains 136 keys, down from 222. Its shape is:

| Detector/tier | Baseline keys |
|---|---:|
| D1 `argv_rows` / `results_scripts` | 1 |
| D2 `spec_flow` / `src` | 1 |
| D3 `default_bundle` / `src` | 42 |
| D4 `hp_constant` / `src` | 89 |
| D4 `hp_constant` / `scripts` | 2 |
| D5 `empirical_table` / `results_scripts` | 1 |

The shrink from 222 to 136 is chiefly the corrected baseline semantics: advisory findings
are no longer serialized, and enforced findings cannot be serialized. Detector
strengthening added already-live ratchet findings; it did not add an exception for them.
Draining the remaining live ratchet keys belongs to the migration issue that owns each
surface, including sibling `6a298a1`; this detector lane does not migrate or baseline-drain
those sibling-owned values.

## Targeted verification

`scripts/dev_tests.sh tests/test_data_in_code_scan.py -q` completed with **28 passed**.
The suite includes D1 enforcement/baseline separation, D3 nested/assigned/incremental bundle
canaries, D4 plural/compound-name canaries, and false-positive controls for summaries,
dimensions, tolerances, schema/path constants, labels, and string-only values.
It also proves that an unparseable scanned file fails closed instead of being skipped.

No full-suite test was run in this lane.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
