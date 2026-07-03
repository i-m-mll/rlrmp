# 64a04e0 — Feedbax-native training and graph consumption cleanup

This directory holds coordination artifacts for the `64a04e0` umbrella, which
aligned rlrmp so that it consumes Feedbax public contracts (generic training-run
specs, execution, checkpoint/resume custody, GraphSpec primitives,
controller/reference selectors, and cloud execution contracts) instead of
carrying its own parallel machinery. rlrmp keeps ownership of the domain-specific
scientific specs, bridge/adversary definitions, and analysis/report logic.

The umbrella's work was organized into lanes. Each lane closes with a *terminal
acceptance gate*: an executable, CI-enrolled check that proves the lane's outcome
holds by construction rather than by spot-check.

- **Lane A** — component-ID confinement and export parity (terminal gate:
  `7811e47`).
- **Lane B** — spec-first training migration, including cloud plan rendering
  (terminal gate declared under issue `08bb6d4`).
- **Lane C** — data-products and source hygiene (terminal gate declared under
  issue `08bb6d4`).
- **`f5d9695`** — training-manifest lineage audit (terminal write-surface
  custody gate).

See `notes/terminal_gate_readiness.md` for the readiness checklist recording,
per lane, the terminal gate, the enforcing gate families and their test counts,
and status.
