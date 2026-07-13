# Matrix-family spot audit: `c6c5997` and `158b580`

Date: 2026-07-13
Tracking issue: `ee7a6f4`

## Outcome

Both families have small, valid `TrainingRunMatrixSpec` documents whose shared
`resolved_output` base resolves from content-addressed custody. The custody
artifact's byte hash, declared semantic root, public decoder round trip, and
matrix materialization all validate.

The audit found two residual defect classes outside the `3cd018b` conversion:

- `c6c5997` still tracks three 745 KB historical outer run documents containing
  full expanded `feedbax_training_run_spec` envelopes. This is the same
  minified-materialization shape, although the family matrix itself is compact.
- `158b580` has no expanded run envelope, but its 422-byte authored-matrix
  artifact sidecar records a stale absolute worktree URI. The sidecar's byte
  hash (`85591f5e60c6d555cd2b899201bc6d81a9d7d7da60cd13370fb612437d4a0ae0`)
  is correct; only the URI is non-portable and currently unresolved.

These findings do not invalidate either matrix or its custody base. The first
needs a separate compact/custody migration; the second should be corrected at
the emitter/sidecar policy boundary and the sidecar regenerated, rather than
patched as an isolated historical string. The current RLRMP emitter constructs
this URI with `authored_path.resolve()`, so a one-file edit would recur on the
next emission.

## Inventory and classification

### `c6c5997`

| Tracked surface | Bytes | Shape and role | Audit result |
|---|---:|---|---|
| `runs/matrix.json` | 6,788 | Compact authored `feedbax.spec.training_run_matrix.v3`; three rows; custody-pinned `resolved_output` base | Valid and materializes all three rows |
| `runs/{flat_3e-5,rewarm_3e-4,rewarm_3e-3}-epsilon-ramp.json` | 745,594 / 745,742 / 745,660 | Historical outer run recipes that also embed full expanded `feedbax.spec.training_run.v2`, expanded `hps`, and inline graph content | Residual minified materializations; all are one line and exceed the 256 KiB gate threshold |
| `runs/*/model.graph.manifest.json` | 154,176 each | Historical model/graph evidence manifests | Below the byte gate and not detected as expanded run envelopes |
| `deploy/*.json`, `deploy/*.md`, `RUN_PLAN.md` | 443-4,816 each | Launch evidence and operator narrative | Compact; the two row-source/manifest JSON records contain historical `/workspace/rlrmp` operational roots, not matrix custody refs |

The live recursive gate reports all three outer run documents as
`expanded_run_envelope`, `expanded_inline_envelope`,
`oversized_json_payload`, and `absolute_filesystem_path_in_spec`. The strings
responsible for the final reason are the four JSON Pointer rule paths
`/phase_program/phases/0`, `/phase_program/phases/1`,
`/phase_program/optimizer_bindings/0`, and
`/phase_program/transitions/0/guard`; they are not filesystem locations. Gate
detection semantics are frozen for this work, so this audit records that
distinction without changing the gate.

### `158b580`

| Tracked surface | Bytes | Shape and role | Audit result |
|---|---:|---|---|
| `runs/matrix.source.json` | 12,247 | Pretty-printed compact matrix source | JSON-semantically identical to `matrix.json`; no inline base or expanded envelope |
| `runs/matrix.json` | 8,439 | Minified compact authored `feedbax.spec.training_run_matrix.v3`; two rows; custody-pinned `resolved_output` base | Valid and materializes both rows; minification is not hiding expanded content |
| `runs/matrix.json.artifact.json` | 422 | Authored-document identity sidecar | Declared SHA-256 matches `matrix.json`, but `uri` is an absolute path in the removed `feature__158b580-orchestrate-cutover` worktree and does not resolve |

The matrix's two `metadata.tracked_run_spec` paths do not currently exist.
They are output-target metadata, not custody inputs: matrix loading and row
materialization succeed without them. The source run recipe and source
checkpoint root named by the matrix both exist in this checkout. If a consumer
starts treating `metadata.tracked_run_spec` as a required input rather than an
output destination, that contract will need a separate correction.

## Custody and losslessness evidence

Both matrices name the same base:

```text
_artifacts/spec-storage/sha256/c0/c0affc9e5b1df8555331a10761f99e390a4debd26e96baea36e90375a81538dc.json
```

The file exists. Its SHA-256 is
`c0affc9e5b1df8555331a10761f99e390a4debd26e96baea36e90375a81538dc`,
matching the content-addressed filename. Its declared root is
`2414a75c2e7971e8c19cc0e8d0ac0b01bea3b9a9bb2b806cd28054ed738d850b`,
matching both matrix `base.resolved_root_hash` values.

Using the public Feedbax helpers
`decode_resolved_snapshot` and `build_resolved_semantics_snapshot`:

- decoding the custody snapshot exactly reconstructs the
  `feedbax_training_run_spec` embedded in
  `c6c5997/runs/flat_3e-5-epsilon-ramp.json`;
- rebuilding a snapshot from those decoded semantics reproduces the declared
  root hash; and
- `materialize_run_matrix` produces the expected rows
  `flat_3e-5-epsilon-ramp`, `rewarm_3e-4-epsilon-ramp`, and
  `rewarm_3e-3-epsilon-ramp` for `c6c5997`, and
  `flat_3e-5-epsilon-ramp` and `rewarm_3e-4-epsilon-ramp` for `158b580`.

The paved-road validator also accepts both authored documents:

```text
valid TrainingRunMatrixSpec: results/c6c5997/runs/matrix.json
valid TrainingRunMatrixSpec: results/158b580/runs/matrix.json
```

## Reproduction commands

The audit used only read-only inventory, gate, hash, decoder, materializer, and
validation operations:

```bash
git ls-files results/c6c5997 results/158b580
find results/c6c5997 results/158b580 -type f -print0 | xargs -0 wc -c
jq -c 'keys' <tracked-json>
jq -r '.. | strings | select(startswith("/"))' <tracked-json>
sha256sum _artifacts/spec-storage/sha256/c0/c0affc9e5b1df8555331a10761f99e390a4debd26e96baea36e90375a81538dc.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" uv run --no-sync python <decoder-and-materializer-audit>
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" uv run --no-sync python scripts/launch_training.py validate results/c6c5997/runs/matrix.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" uv run --no-sync python scripts/launch_training.py validate results/158b580/runs/matrix.json
```
