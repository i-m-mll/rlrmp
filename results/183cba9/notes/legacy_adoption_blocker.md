# Legacy Baseline Adoption Resolution

Issue 183cba9 initially blocked while reconstructing the legacy producer
template for the 08483d5 baseline. That blocker is now resolved locally by the
feedbax legacy-adoption tool on
`feature/48b8b91-legacy-checkpoint-adoption` and the issue-local legacy import
shims in `results/183cba9/scripts/legacy_checkpoint_builders.py`.

## Adopted Baseline

| item | value |
| --- | --- |
| legacy checkpoint | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_0012000` |
| producing commit used for manifest | `899278bb30006153c28a0c92aee9dffedc6c4633` |
| issue-recorded commit | `9f919c65e52b0042181d615d4a40e1cc6fab5d0b` |
| LeafManifest | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_0012000/leaf_manifest.json` |
| adopted checkpoint root | `_artifacts/3cd018b/runs/ramp3500_to1000/checkpoints` |
| latest transaction | `tx-80f84748ea934dca8d18be93a5cb7308` |

The adopted checkpoint round-tripped through feedbax checkpoint custody before
`latest.json` was published. The continuation facts are recorded in
`results/3cd018b/notes/legacy_baseline_adoption.json`: completed batch 12000,
stop target 12500, and a 500-batch continuation.

## Mapping Notes

The current graph refactor moved old GRU leaves from `/nodes/net/net/...` into
`/nodes/net/nodes/...`. The explicit model and optimizer path rules are recorded
in `results/183cba9/notes/legacy_path_mapping.json`. Old serialized structure
arrays with no current trainable target are verified against the stream and then
explicitly dropped by rule; the new current cell state output is kept from the
current template by explicit allowlist.
