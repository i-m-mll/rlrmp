# 3cd018b tracked-materialization inventory

The eight flat `runs/<row>.json` documents were historical post-run envelopes,
not authored intent: each combined evidence and launch metadata with a complete
`feedbax.spec.training_run.v2` payload, an expanded inline graph, and an
intentionally frozen
`rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload.v1` method
payload. The v1 method payload has no automatic migration to v2, so this pass
does not reinterpret it.

Each flat document is now a compact, one-row
`feedbax.spec.training_run_matrix.v3` authored intent. Its `resolved_output`
base points to a content-addressed snapshot of the **complete original outer
envelope** and selects `feedbax_training_run_spec` with `payload_path`. This
preserves both historical evidence and the exact frozen executable payload.
Feedbax's storage emitter also wrote the resolved row set and execution capsule
to content-addressed custody.

Emitter-generated authored `.artifact.json` sidecars are deliberately not
tracked in this conversion. The current authored-artifact sidecar writer records
an absolute worktree URI, so checking those sidecars in would make the compact
intent non-portable and reintroduce an absolute-path gate finding. Linked
follow-up [issue:e093cd9] owns that emitter correction. The compact matrices and
their content-addressed envelope snapshots, resolved-row snapshots, and
execution capsules are the canonical outputs of this conversion.

## Document roles

| Tracked documents | Count | Before | After |
|---|---:|---|---|
| `runs/<row>.json` | 8 | Expanded post-run/execution envelopes | Compact authored one-row matrices with custody-pinned resolved bases |
| `runs/<row>/model.graph.manifest.json` | 8 | Graph evidence sidecars | Unchanged graph evidence sidecars |
| `runs/ramp3500_to1000/matrix.json` | 1 | Earlier compact authored-intent precursor | Unchanged; its resolved snapshot remains custody-resolvable |

`README.md`, `RUN_PLAN.md`, and the reconciliation notes remain narrative or
evidence surfaces and were not rewritten. No analysis or training-authoring
source was involved.

## Lossless snapshot evidence

For every row, the original raw file hash was captured before replacement. The
complete parsed envelope was encoded with
`build_resolved_semantics_snapshot`, stored with
`store_canonical_json_artifact`, decoded with `decode_resolved_snapshot`, and
compared for exact JSON-tree equality. Re-encoding the decoded tree with
`training_spec_canonical_bytes` also matched the original tree's canonical
bytes exactly.

| Row | Original bytes | Original raw SHA-256 | Envelope snapshot artifact SHA-256 | Snapshot root hash | Compact bytes | Exact decode / canonical re-encode |
|---|---:|---|---|---|---:|---|
| `const1000` | 728,795 | `77ad271d723ad627e8a7c1c827962b1f174362fcbdc8962ce87a870abcc4ec38` | `19c618ef94adb43f236bce2c8956a22cf2ebb06b961dbb291091108e8f4e6fe6` | `5e9088b8d93a4e0400dbe6d0c513c4acdcde2cb689c13ddd1d50a78fe4fec110` | 1,432 | yes / yes |
| `const1750` | 728,795 | `d282c4af0613f42d83939bf9f8a29306aef68feef80842314ea9396737d2605a` | `7f67fd7094f904180b4902b1c74ee4ca2016951c2ed792d194f0e2d2adf51d8b` | `2053066ee4cefedb8e117198293e7b64c518870131f0c45e22623a07ac614372` | 1,432 | yes / yes |
| `const250` | 728,752 | `64f0a0f9d67eb7e852d2f0a21091d2eceaaaf724cafaef3045b73887ae68f3c1` | `a0848ecf7e4c02f71dcaf8324263eb3cd9729d0944088ae39ddc64d6de49c35a` | `7eaa04ab62b5d3ac1cb68b2017a0c4315f9cff2fd1ee79d8ca3a44d3227bd338` | 1,425 | yes / yes |
| `hold1750_to1000` | 728,981 | `a133aaa46c2e28ca75fc7c1d9d49b09d1076639eef333b0a0865be6c395ed6fb` | `acdef191de1a743ddff610ca6d4ea8204c78830bc37638b98e26a4aa8dcc9264` | `47e7c4b42e7a278ec7a47680a71629d9e688805fd213949592cacc99bf821c4f` | 1,474 | yes / yes |
| `hold1750_to250` | 728,946 | `d557a4e73ffe6c360d7b05ec276cc0db39654b0ad6c995250f5f94d75cd75f66` | `29c0c662f0fe80abb645933384c2758bf7e420dbc07528b08c887bb0c2ff0506` | `53425de25b75ebf6d36b202e308ed65539ead29a5209e03e95654f87385fa8d3` | 1,467 | yes / yes |
| `hold3500_to1000` | 728,981 | `73e379a28de533796f484dfa21fdb31a6b32d067f8962783ab493182dadfa5e4` | `41171a2df80ee3242452df6c99bea0537bdf1741aa1ff1fb58a1bea7a5e42bb5` | `7c681f2dba4c2ad7879ad01735d329a3a30b42cdada1791ec711e12e3bf28135` | 1,474 | yes / yes |
| `hold3500_to250` | 728,946 | `5d9d1025514536326db916279e5c4c6f487ecc10450c416a2603b70efa6b5c0c` | `eb26b6f8c9fc7a7fe43ac7a145a8aaa6c9ca1ed4d5a7802ad544e10aaf434e5b` | `df903ae5096b411e7c476ed68513084e3ec38dc2fc018d1f988517473b1b0f3b` | 1,467 | yes / yes |
| `ramp3500_to1000` | 728,969 | `5598eea97e3f0be8d499e062986c5548162db8e7384642f26e28471e0e24ceff` | `108cbdfcf74f5a231888dfa63e0ec52e9fd07539314b2fbbba3240627e8a2c88` | `b72252c388841972d1774134984151dc87c519ef0a7088381d7b86a8e15a38eb` | 1,474 | yes / yes |

The eight originals totaled 5,831,165 bytes; the compact replacements total
11,645 bytes. All referenced base snapshots, resolved row snapshots, and
execution capsules were re-opened from their declared custody paths during
verification, and their artifact hashes matched their bytes.
