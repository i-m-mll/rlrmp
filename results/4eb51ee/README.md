# Cross-architecture certificate agreement across evaluation lenses

Current execution state: governed authoring, six-row lowering, storage, and validation
pass. Local execution is blocked before batch 1 by the shared fresh-matrix orchestration
precondition; see `runs/execution_status.json` and `notes/execution_smoke.md`. No
scientific evidence was produced.

This experiment family asks whether nominal-versus-robust conclusions agree across a
static-gain linear controller, an augmented-linear recurrent controller, and a nonlinear
GRU when the plant, task, budget, seed, and evaluation disturbances are matched. It is a
local engineering-smoke test only: it cannot answer the scientific question. The frozen
cohort has six training rows and four post-training evaluation lenses. The
low-level `static_gain`, `augmented_linear`, and `empirical_nonlinear` certificate
components, explicit `not_applicable` semantics, and custody-routed certificate report
renderer are present. The integrated typed architecture lowerer generated all six rows,
and the grouped registered analysis type is `rlrmp.certificate.standard`. Execution is
now blocked only at the shared fresh-matrix orchestration entry point. The packet refuses
invented source checkpoints, direct native-executor calls, legacy payloads, hand joins,
and GRU/static-gain coercion. Exact identities and the current blocker are recorded in
`runs/execution_status.json`; the original selection record remains in
`runs/cohort.intent.json`.
