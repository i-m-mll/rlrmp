# Destination plan

The starting point is the 136-key baseline at integration commit
`470ffe0928712fcdbc7cbaf8f3042b5e919f8008` (SHA-256
`233b8cd399f3e6a0a91595ac902ad85e739553366b71fd58d0a202239a726900`). The live
strengthened scan has 132 of those keys; four are stale after integrated sibling work.

## Resolution partition

| Family | Count | Resolution |
|---|---:|---|
| Config tier owned by `cd137d8` | 44 | Curated, rationale-bearing allowlist; no source changes in this lane. |
| User-held `e04bd36` objects | 3 | Curated, rationale-bearing allowlist; no deletion or source modification. |
| Values already on owning typed schema models | 16 | Curated schema-default rationale; keep the field defaults colocated with their model. |
| Legitimate diagnostic/operational constants | 3 | Curated purpose rationale; not scientific run parameters. |
| Integrated sibling removals | 4 | Remove stale baseline key only after confirming the live finding is absent. |
| Analysis/spec-product migration | 43 | Move literals to experiment specs, registered recipe params/presets, or governed products. |
| Model/runtime/product migration | 23 | Move literals to registered model/runtime presets, owning products, or schema defaults. |

The first five rows are represented in `curated_resolution_manifest.json`. The two implementation
partitions produce disjoint resolution fragments. The closing-manifest materializer refuses
missing, duplicate, or unexpected keys, so the combined output must resolve all 136 entries
exactly once.

## Binding rules

- The old inventory's `destination_surface` is the default assignment. Every deviation is
  stated in the key's rationale.
- A historical run/evaluation value moves to a tracked spec before its Python literal is
  removed. A reusable value may instead move to a registered preset/template.
- Calibration and empirical tables use schema/version/hash-governed data products loaded
  through `rlrmp.data_products`.
- Source deletion never substitutes for value preservation.
- The final load proof covers at least ten of the largest migrated entries, and the manifest
  records the exact probe for each.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
