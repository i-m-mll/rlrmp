# Force-state observability by robust training

This experiment asks whether a GRU needs direct access to the delayed force/filter
state to develop the engineering signatures expected from broad-epsilon PGD
training, or whether delayed target-relative position and velocity are sufficient.
The owner selected a callback-free 2 x 2 family: force/filter feedback visible or
hidden, crossed with nominal or broad-epsilon PGD training. All planned runs are a
one-seed, 100-batch, local-only engineering smoke; they cannot answer the scientific
question.

The governed per-row re-lowering and fresh-matrix routes are now integrated. The
tracked compact base and four-row matrix intent lower, emit, and assemble with
distinct planned-run, authored-payload, and execution-payload identities. Local
execution nevertheless stops before batch 1 because schedule preflight cannot find
an optimizer at a supported typed path in the inline run spec. No optimizer shim,
preflight bypass, or alternate executor was used. See [RUN_PLAN.md](RUN_PLAN.md) for the
frozen smoke protocol, [notes/engineering_smoke_evidence.md](notes/engineering_smoke_evidence.md)
for current evidence, and [notes/static_authoring_gap.md](notes/static_authoring_gap.md)
for the superseded pre-integration diagnosis. [issue:ebd5d02] owns the current
typed-optimizer preflight gap and structurally blocks this experiment.
