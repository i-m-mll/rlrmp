Issue 3cd018b prepares an eight-row adaptive-epsilon target-schedule matrix under
the current feedbax-native RLRMP training system. The matrix varies target-damage
start, final, and ramp behavior while keeping the epsilon-scaled outer training
exposure fixed, and includes one current-system replication of the earlier
`0 -> 3500 -> 1000` row as a sanity check against prior results.

## Recipe Reconciliation

The tracked recipes were reconciled on 2026-07-08 against the synced pod
`float32_rebound_specs` used by the 2026-07-07 RunPod launch. The committed row
JSON now records the launched method payloads, float32 runtime graph constants,
and repo-relative tracked spec paths such as `results/3cd018b/runs/<row>`.

Future resumes or forks of these checkpoints must not try to byte-match the old
checkpoint bindings by restoring absolute worktree paths or old code projections.
Use the sanctioned new-lineage fork path: call `fork_checkpoint_transaction` with
`allow_new_lineage_override`, record the override reason in fork metadata, and
make the new tracked recipe the source of truth for the resumed lineage.
