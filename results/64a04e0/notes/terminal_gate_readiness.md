# Terminal-gate readiness checklist — 64a04e0 Feedbax-native umbrella

This note records the final acceptance status of the `64a04e0` umbrella, which
aligned rlrmp so it consumes Feedbax's public contracts for training-run specs,
execution, checkpoint/resume custody, GraphSpec primitives, and cloud execution.
The umbrella's work was organized into lanes; each lane closes with a *terminal
acceptance gate* — an executable, CI-enrolled check that proves the lane's
outcome holds by construction rather than by manual spot-check.

All gates below carry the `feedbax_contract` pytest marker and are enrolled as
`live` families in `ci/feedbax-contract-suite.toml`. The marked suite is run as
one gate:

```
PYTHONPATH=src uv run --no-sync python -m pytest -m feedbax_contract --strict-markers
```

Under this marker, `tests/conftest.py` collects only files enrolled as `live`
families, and `tests/test_feedbax_contract_meta.py` forbids skipped or
non-strict-xfail tests and enforces a per-family minimum test count. As of this
checklist the marked suite collects **151 tests**, all passing.

## Status by lane

| Lane | Terminal acceptance gate | Enforcing gate families (test file — collected count) | Status |
|---|---|---|---|
| **Lane A** — component-ID confinement & export parity (`7811e47`) | Retired-ID confinement scan + descriptor conformance canary + model export parity | `retired_id_scan` (`test_retired_component_id_scan.py` — 7); `descriptor_conformance_canary` (`test_descriptor_conformance_canary.py` — 18); `model_export_parity` (`test_model_export_parity.py` — 11) | Live, green |
| **Lane B** — spec-first training migration (`08bb6d4`) | End-to-end spec-first smoke: C&S `TrainingRunSpec` executes a few batches and its native `TrainingRunManifest` resolves; minimax dry-run spec + fingerprint guard; local/modal/runpod `ExecutionPlan`s render with no provider contact | `lane_b_terminal_gate` (`test_lane_b_terminal_gate.py` — 4) | Live, green |
| **Lane C** — data-products & source hygiene (`08bb6d4`) | Executable assertions that the three Lane C outcomes hold by construction, referencing their enforcing families | `lane_c_terminal_gate` (`test_lane_c_terminal_gate.py` — 4); enforced by `generated_data_constant_scan` (`test_data_lint_generated_constants.py` — 6), `descriptor_basis_hash` + `feedback_descriptor_scan` (`test_controller_feedback_descriptors.py` — 5), and `write_surface` (see f5d9695) | Live, green |
| **`f5d9695`** — training-manifest lineage audit | Deny-by-default durable-output custody guard + six provenance-lineage invariants | `write_surface` (`test_write_surface_custody.py` — 16) | Live, green (issue open) |

## Lane B terminal gate detail (`tests/test_lane_b_terminal_gate.py`)

The migration children each have their own unit coverage
(`a1b6118` → `7aeda5a` → `95a3865` → `0efc92d` → `5b571ae`, plus `b047981`,
`54b0c2e`, `799fcb9`, `5cc6c90`, `d6b7018`). The terminal gate proves the lane's
*outcome* end-to-end:

- **C&S spec-first path.** A smoke-size composed C&S run spec is built through
  `rlrmp.runtime.training_run_specs` (the composed payload carries the Feedbax
  `TrainingRunSpec` under `FEEDBAX_TRAINING_RUN_SPEC_KEY`), validates as a
  Feedbax `TrainingRunSpec`, executes two real batches through the spec-first
  `run_full_training` adapter, and the natively-emitted `TrainingRunManifest`
  resolves back through `resolve_run_record`. The manifest is emitted under a
  per-test temporary repo root, so the gate never writes into the tracked
  `_artifacts` tree.
- **Minimax dry-run.** A validated minimax `TrainingRunSpec` is built through its
  spec-first construction path; its effective-phase fingerprint is validated and
  a tampered fingerprint is rejected. No full training is run.
- **Cloud plans.** Local + Modal + RunPod `ExecutionPlan`s are rendered from the
  same C&S spec via `ExecutionSpec.training_run_spec` with zero provider contact,
  each deriving the Feedbax generic-executor command
  (`python -m feedbax execute-training-run-spec`).

## Lane C terminal gate detail (`tests/test_lane_c_terminal_gate.py`)

Lane C's substance is enforced by pre-existing families; this gate asserts the
three outcomes hold and that their enforcing families are live, without
re-implementing them:

1. **No generated tables remain in source.** `rlrmp.data_products.lint.violations`
   reports zero un-allowlisted generated-constant tables under `src/`
   (family `generated_data_constant_scan`).
2. **Consumed data-product identities are fail-closed and non-null.** Run specs
   snapshot consumed identities via `add_consumed_data_identity`, whose guard
   refuses an empty role/schema/hash; the loader side
   (`rlrmp.data_products.envelope`) re-derives and pins the identity hash. The
   identity-shape invariant is already enrolled under the `write_surface` family.
3. **Stopgap surfaces are descriptor-backed.** `controller_feedback_scales`
   resolve from a serialized descriptor payload carrying a `sha256:` basis hash
   rather than hard-coded slices/order (families `descriptor_basis_hash` and
   `feedback_descriptor_scan`).

## Notes on open issues

`f5d9695` remains open as an issue, but its substance — the write-surface custody
guard and the six lineage invariants — is fully enforced by the live
`write_surface` family. A small number of non-production or follow-on children of
the umbrella (`00f97d5`, `103db99`, `63cec06`, `020a65b`, `743213d`, `b20e0ea`,
`0f67665`) remain out of the terminal blocker set and are tracked separately.
