# Force-state observability by robust training

This experiment asks whether a GRU needs direct access to the delayed force/filter
state to develop the engineering signatures expected from broad-epsilon PGD
training, or whether delayed target-relative position and velocity are sufficient.
The owner selected a callback-free 2 x 2 family: force/filter feedback visible or
hidden, crossed with nominal or broad-epsilon PGD training. All planned runs are a
one-seed, 100-batch, local-only engineering smoke; they cannot answer the scientific
question.

Static authoring is currently blocked before a governed matrix can be written. The
matrix compiler patches an already-lowered `TrainingRunSpec`, while both selected
axes change multiple derived graph, descriptor, training-mode, loss/fidelity, and
worker-contract fields. Manually patching those compiled fields would be the escape
this KPI lane is meant to detect. See [RUN_PLAN.md](RUN_PLAN.md) for the frozen smoke
protocol and [notes/static_authoring_gap.md](notes/static_authoring_gap.md) for the
path-level evidence. [issue:5816bf0] now owns governed per-row re-lowering and
structurally blocks this experiment. The canonical-tool-shaped KPI input draft is
also ignored by `.gitignore` line 240; [issue:fddd87a] blocks tracking it normally,
and this lane must not force-add it.
