# e9fc384 — Graph sidecar fidelity audit

Retroactive structural fidelity audit of all 38 tracked archived Feedbax
graph sidecars (`model.graph.json`) under `results/`, done as a hard
prerequisite of [issue ae15851]'s conversion of these sidecars into clean
new-format `GraphSpec` loadability regression fixtures. Each sidecar is
classified `clean` or `known_wrong` by inspected node structure (not by
production date): 36 are clean point-mass/filter/channel family sidecars (32
`gru`, 4 `vanilla_rnn`), and the 2 `results/30f2313/runs/cs_stochastic_gru__*`
sidecars are known-wrong — they encode the point-mass shape but should encode
the analytical CS-LSS (`LinearStateSpace` + delayed-feedback) plant per their
own `run.json` provenance. See `notes/graph_sidecar_audit.md` for the full
narrative and `notes/graph_sidecar_audit_manifest.json` for the committed
machine-readable manifest that `tests/test_graph_sidecar_audit_manifest.py`
guards against drift.

The historical nested recipes were retired under issue `ef8e1df`; recover them from git tag `legacy/ef8e1df-nested-run-json-retired` (the bytes are also in Mandible custody).
