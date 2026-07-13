# Force-state observability by robust training

This experiment asks whether a GRU needs direct access to the delayed force/filter
state to develop the engineering signatures expected from broad-epsilon PGD
training, or whether delayed target-relative position and velocity are sufficient.
The owner selected a callback-free 2 x 2 family: force/filter feedback visible or
hidden, crossed with nominal or broad-epsilon PGD training. All planned runs are a
one-seed, 100-batch, local-only engineering smoke; they cannot answer the scientific
question.

The governed per-row re-lowering, fresh-matrix, typed-optimizer, and local
environment-fingerprint routes now execute on reviewed local Feedbax staging. The
tracked compact base and four-row matrix intent lower, emit, and assemble with
distinct planned-run, authored-payload, and execution-payload identities. The first
row trained, but the requested stop at batch 50 overran to 100 and registration
failed on missing LR samples plus a canonical seed-path mismatch. No shared
environment mutation, certificate bypass, or alternate executor was used. See [RUN_PLAN.md](RUN_PLAN.md) for the
frozen smoke protocol, [notes/engineering_smoke_evidence.md](notes/engineering_smoke_evidence.md)
for current evidence, and [notes/static_authoring_gap.md](notes/static_authoring_gap.md)
for the superseded pre-integration diagnosis. [issue:c37df92], [issue:0a97038], and
[issue:feedbax/b9ddd04] own the current blockers.
