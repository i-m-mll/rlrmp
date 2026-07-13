# Canonical heterogeneous C&S training bases

This result surface defines the governed training inputs for comparing three
controller architectures on the same Crevecoeur-and-Scott task:

- GRU (`gru`)
- memoryless static linear feedback (`time_constrained_free_gain`)
- zero-bias identity recurrent control (`linear_recurrence`)

Each architecture has nominal and broad-epsilon-PGD training intent. Every row
must lower through the registered `rlrmp/cs_supervised/v1` method, start without
an inherited checkpoint or continuation, and use a row-local artifact,
checkpoint, manifest, and tracked-spec route.

The recurrent contract uses a zero hidden-state initializer, an identity
VanillaRNN cell with no bias, and a bias-free action readout. Its certificate
basis is the controller-visible target-relative coupled state concatenated with
the recurrent hidden state; it must not be interpreted as a plant-state static
gain.

## Current dependency

No launch-ready matrix is tracked yet. The first draft expanded complete
`TrainingRunSpec` objects into top-level row replacements; that representation
was removed because it was neither compact nor owned by the authored-row
lowering contract.

Emission is blocked on [issue:5816bf0]. Its public authored-row interface must
add a typed `config.controller_architecture` field with values `gru`,
`time_constrained_free_gain`, and `linear_recurrence`, then dispatch those
values through registered architecture providers while retaining the generic
lowerer's provenance identities. The public module must declare two capability
constants so consumers can fail closed on partial support:

- `RLRMP_TRAINING_ARCHITECTURE_CONTRACT = "rlrmp.heterogeneous_cs_architecture.v1"`
- `RLRMP_TRAINING_ARCHITECTURES = ("gru", "time_constrained_free_gain", "linear_recurrence")`

The authored schema must continue rejecting compiled `graph`, `task`,
`method_payload`, and `worker_execution` input.

Once that interface lands, this lane's emitter will author six rows using only
the architecture selector, broad-epsilon-PGD boolean, and row-local output/spec
routes. It will pass the compact matrix to
`rlrmp.runtime.spec_storage.emit_rlrmp_training_run_spec_storage`, which owns
the tracked intent plus resolved snapshot and execution-capsule custody. Direct
writes of expanded per-row training specifications are not an allowed fallback.
