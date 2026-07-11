# Harmonized nominal baseline

This directory holds the spec-locked Stage-1 C&S no-integrator baseline that
will supply the Stage-2 checkpoint fork. It is one `vmap` ensemble row with
five internal replicas, not five independent rows. See [RUN_PLAN.md](RUN_PLAN.md)
for the frozen task, perturbation bank, seam probe, R-star derivation, and
explicit launch boundary. A future Stage-2 matrix references this tracked
recipe through `matrix.metadata.rlrmp_source_run_spec_ref`; its
`rlrmp_task_identity` values are canonical JSON hash labels derived by the
prelaunch gate, not copied task snapshots.
