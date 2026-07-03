# f5d9695 — Training-manifest lineage audit (terminal)

Terminal child of the `64a04e0` feedbax-native umbrella. After every
production-path child (native manifest emission, checkpoint custody adoption,
run-record collapse, spec-driven launchers, data-product identity snapshots)
landed on `integration/64a04e0-feedbax-native`, this child turns the six
end-to-end provenance-lineage invariants and the deny-by-default durable-output
custody rules into executable, gate-registered checks. The new machinery is a
static write-surface custody guard (an AST scan over the training
run-production/emission path that pins every raw durable-write site and
generates a conditional-emitter branch matrix from the emitter sites); the six
invariants are enforced as marked checks that assert-integrate the existing
(mostly unmarked) production anchors into the required CI standing gate. The
audit narrative, per-invariant verdicts, and the documented standing findings
are in `notes/lineage_audit.md`.
