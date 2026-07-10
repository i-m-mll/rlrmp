# Data-in-code baseline drain closure

The lane started from the 136-key baseline at integration commit
`470ffe0928712fcdbc7cbaf8f3042b5e919f8008` (baseline SHA-256
`233b8cd399f3e6a0a91595ac902ad85e739553366b71fd58d0a202239a726900`).
`closing_manifest.json` resolves every key exactly once.

## Final scanner state

The final scan reports 249 findings: 181 advisory, 66 ratchet, and 2 enforced. All 68
enforced/ratchet findings are curated rationale-bearing exceptions. The generated ratchet
baseline contains zero keys and `violations()` returns zero.

The curated set consists of 44 config-tier keys explicitly owned by `cd137d8`, three
objects under the user hold on `e04bd36`, 16 typed models already owning their schema
defaults, three legitimate diagnostic/operational constants, and the two pre-existing
enforced exceptions. No `e04bd36` object was modified; the worker verified the held ASTs
were identical before and after migration.

## Preservation manifest and load proof

The combined manifest has 136/136 resolutions and 31 passed load proofs. Resolution routes:

- 41 registered analysis parameter presets;
- 7 registered runtime parameter presets;
- 7 registered typed loss presets;
- 6 registered model parameter presets and one versioned graph-migration preset;
- 3 governed `AnalysisDataProduct` migrations;
- one tracked historical evaluation spec;
- 16 owning-schema defaults, 3 legitimate constants, and 47 curated ownership/hold
  exceptions;
- 4 stale keys removed only after integrated sibling work removed their live findings.

All migrated preset loaders fail closed on exact schema identity/version and content hash.
The c723 empirical table and the upgraded broad-epsilon/calibration products additionally
pin product identity and payload hashes.

## Inventory replay

Replay against `_artifacts/b6136cc/second_order/data_in_code_inventory.jsonl` after the
drain produced:

| Payload class | Inventory | Surviving files | Surviving objects | Raw caught | Raw rate |
|---|---:|---:|---:|---:|---:|
| `planner_rows` | 29 | 15 | 4 | 0 | 0.00% |
| `default_param_bundle` | 25 | 25 | 25 | 19 | 76.00% |
| `run_hyperparameters` | 346 | 304 | 303 | 184 | 60.53% |

D3 catches 19/19 live statically eligible bundles (100%). D4 catches 182/182 live eligible
literal constants (100%). The lower raw rates are expected: migrated values no longer have
literal-backed source objects, while held/curated values remain visible to the detector.

## Verification

- Final closure set: 64 passed
  (`test_data_in_code_scan.py`, closing manifest, analysis destinations, runtime
  destinations).
- Analysis destination tests: 11 passed; broader affected set: 116 passed. Three existing
  `gru_pilot_figures` FigureSpec binding failures were reproduced and are unrelated to the
  migrated rollout-trial preset.
- Runtime destination tests: 21 passed; existing product tests: 27 passed; model/modal/eval
  targeted tests: 73 passed; cs-nominal/eval targeted tests: 3 passed.
- Ruff, compile checks, JSON validation, and `git diff --check` passed.
- No full-suite run was performed, per lane instruction.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
