# Training-manifest lineage audit (f5d9695)

Terminal audit for the `64a04e0` feedbax-native umbrella. Everything below is
enforced by `tests/test_write_surface_custody.py` (family `write_surface`, now
`live` in `ci/feedbax-contract-suite.toml`) plus the existing contract gate.
The marked `feedbax_contract` suite grows from **126 → 142** collected tests
(16 new checks); skips count as failures in this gate
(`tests/conftest.py` + the meta-test's no-SKIP / strict-xfail rule).

Run corpus context: the audit is enforced against the source tree of
`integration/64a04e0-feedbax-native` (the single native emitter, the
run-record resolver, the post-run provenance stamp, the consumed-data-identity
API, and the feedbax migration substrate), not against a frozen snapshot of one
run's artifacts. That makes each invariant a standing gate rather than a
one-time spot-check: any regression on a future run is caught at CI time.

## Per-invariant verdict

| # | Invariant | Verdict | Enforcing check(s) |
|---|-----------|---------|--------------------|
| 1 | No new-format run reconstructs parity via a `comparable_training_spec()`-style derivation; the native `TrainingRunManifest` is the sole source of truth. | **enforced (by absence) + new marked check** | `test_invariant1_no_comparable_spec_parity_reconstruction` asserts no such derivation function exists anywhere in `src/`/`scripts/` and that `run_specs.resolve_run_record` reads the manifest. Assert-integrates the unmarked `test_run_record_resolver.py` (resolver returns the manifest payload; same-id-different-content fails). |
| 2 | Every new-format manifest carries a full `post_run_provenance` stamp (rlrmp SHA, feedbax SHA, schema versions, GraphSpec hash/version, manifest root, `absolute_path_sha256`). | **new marked check** | `test_invariant2_post_run_provenance_field_set_complete` calls `attach_post_run_provenance` and asserts the complete field set incl. a 64-hex `absolute_path_sha256` and `feedbax_graph.graph_spec_version`. Assert-integrates `test_post_run_sh.py`. |
| 3 | Every new-format manifest carries consumed-data identities (role, schema, hash) for its data-product inputs. | **new marked check** | `test_invariant3_consumed_data_identity_shape` calls `add_consumed_data_identity`, asserts the exact `{role, schema, hash}` entry shape and empty-field rejection. Assert-integrates the `108b4d3` product-identity lane and `test_run_record_resolver.py::test_add_consumed_data_identity_appends_stable_identity`. |
| 4 | Legacy (pre-native-emission) runs use only explicit `not_found` / `archive-only` discriminators — never silently missing, never misclassified as native. | **new marked check** | `test_invariant4_legacy_discriminators_are_explicit` asserts the resolver raises with both `not_found` and `archive-only`, and `ArchiveOnlySpecError` exists. Assert-integrates `test_run_record_resolver.py::test_resolve_run_record_missing_manifest_reports_archive_only` and `test_post_run_sh.py` (`archive-only` parity, legacy-nested explicit error). |
| 5 | No parallel manifest-emission path exists alongside `0efc92d`'s native emitter. | **new marked check** | `test_invariant5_single_native_manifest_emitter` AST-scans all of `src/`/`scripts/` and asserts exactly one `TrainingRunManifest(...)` construction site (`training_run_specs.py`). Assert-integrates the `reaccretion_ratchet` writer-site guard. |
| 6 | Single-substrate guard: no parallel provenance-versioning schema outside `feedbax/contracts/migrations.py`. | **new marked check** | `test_invariant6_single_substrate_no_parallel_versioning_registry` asserts rlrmp defines no `SpecSchemaRegistry` / `SpecSchemaFamily` / `SchemaMigration` class and imports them from `feedbax.contracts.migrations`. Assert-integrates the version-pin meta-test. |

No invariant failed. Invariants 1, 5, 6 hold structurally (absence of a
derivation path; a single emitter; a single migration substrate); 2, 3, 4 hold
by exercising the production API/anchors. Where an invariant was already covered
by an existing (mostly *unmarked*) unit test, the new marked check **references
and asserts** the same fact so the guarantee is pulled into the required
`feedbax_contract` standing gate rather than duplicated.

## Write-surface custody guard (the new machinery)

`tests/test_write_surface_custody.py` statically scans the training
run-production/emission domain — `scripts/train_*.py` + `src/rlrmp/train/*.py`,
the same training-entry-point domain the re-accretion ratchet pins — for every
*raw* durable-output write call: `fbx_save`, `eqx.tree_serialise_leaves`,
`np.save*`, `open(..., "w")`, `Path.write_text`, `Path.write_bytes`.
(Analysis-pipeline and eval-script outputs are governed separately by the
DataProduct / ReportManifest substrate and are out of scope for the
*training-manifest* lineage guarded here.)

**Classification by target root.** Each raw write is classified from its target
expression's root variable:

- **ephemeral / atomic-staging** — the raw write targets a `tmp`-rooted path
  (staged, then `os.replace`-d / renamed into place). The durable
  materialization is the atomic rename, not a raw serialization to a declared
  durable root, so these are safe and intentionally not allowlisted. 14
  ephemeral sites found (checkpoint staging in `train_minimax`, `cs_nominal_gru`,
  `guided_distillation`); `test_ephemeral_writes_are_tmp_staged` proves the
  classification is sound (every ephemeral site is tmp-rooted).
- **durable** — everything else (writes rooted at `output_dir` / `spec_path` /
  `paths[...]` / a parameter). 20 durable sites found.

**Deny-by-default.** Every durable raw-write must be named in
`ci/write-surface-allowlist.toml` with an owning issue and a role
(`test_durable_write_sites_match_allowlist`). A new, unlisted durable raw-write
fails the gate; landing it requires a deliberate allowlist edit or routing the
write through the sanctioned custody writer / a run-spec emitter. Stale entries
must be removed (`test_allowlist_has_no_dead_entries`); keys are structural
(path / function / kind / target-label) so ordinary line moves don't churn the
list. `test_single_custody_pytree_writer` asserts the one feedbax-owned durable
transaction writer (`write_checkpoint_transaction`) is reached only through the
rlrmp custody adapter, never called directly elsewhere in `src/rlrmp`.

**Conditional-emitter branch matrix — generated from emitter sites.** The guard
walks the AST of each emitter for enclosing `if`/`else` guards and for ternary
dispatch over emitter functions, and derives per-site conditionality plus the
set of mutually-exclusive emitter groups. `conditional` in the allowlist must
equal the generated value (`test_conditional_flag_matches_generated_branch_matrix`),
so the matrix can never drift into a hand-curated list. The matrix is
non-vacuous (`test_conditional_emitter_branch_matrix_is_non_vacuous`): on the
minimax path it captures two mutually-exclusive emitter groups a single toy run
can never jointly exercise —

1. **single- vs multi-adversary artifact layout** — an in-function `if/else`
   writing `output_dir/trained_adversary.eqx` (one adversary) *or*
   `adversaries/adversary_{i}.eqx` (population); and
2. **force-profile vs ΔA adversary-log dispatch** — a module-level ternary
   `log_fn = _log_linear_dynamics_adversary if use_linear_dynamics else
   _log_adversary_force_profiles`, whose two branches write
   `adversary_delta_A*.npz` *or* `adversary_force_profiles*.npz`.

This is exactly why a one-toy-run spot-check is insufficient and a static
branch-matrix guard is required. Two negative canaries
(`test_write_surface_negative_canary_flags_new_durable_write` /
`..._ignores_tmp_staged_write`) prove the scan flags a fresh `output_dir` write
and ignores a `tmp`-staged one.

### Standing findings (documented, not silently relied against)

The minimax final-output emitters — `warmup_model.eqx`, `adversarial_model.eqx`,
`trained_adversary.eqx` / `adversaries/*`, `adversarial_losses.npz`, and the
adversary `_log_*` npz — are raw `fbx_save` / `np.savez` writes to `output_dir`,
**not** routed through the single feedbax custody writer and **not**
manifest-addressed at write time. This is the anticipated standing state on the
minimax path; the issue names "a raw `warmup_model.eqx` write on the minimax
path" as exactly the emitter this regression must keep visible. The native
`TrainingRunManifest` emitter addresses these post-hoc. **This audit does not
weaken the invariant**: it pins each such site in the allowlist (owner, role,
conditionality) so the surface is deny-by-default and cannot grow silently.

Recommended coordinator follow-up (production work, not this terminal child):
collapse the minimax final-output emitters onto the single feedbax-owned
artifact writer so they are custody-routed and manifest-addressed at write time,
retiring the `role = "model_artifact" / "history_artifact" / "adversary_artifact"`
findings owned by `f5d9695` in the allowlist. Until then they remain pinned and
gated.

### Escape modes the static guard does not by itself close

Documented in the allowlist header and here so they are not mistaken for
in-scope guarantees of the static surface guard: subprocess / native-library
writes (a child process or C extension writing to a durable root is invisible to
the AST scan); remote object-store writes (Modal-volume / S3 handoff);
symlink / hardlink traversal (a write to a symlinked durable path). These are
covered by other governance (post-run provenance, custody hashing, manifest
parity).

## Reports

Reports remain governed and custody-decoupled via the existing feedbax
`ReportManifest` / `ReportSpec` contract with `feedbax-local` or `mandible`
storage backends (feedbax stays the schema / lineage / hash authority; bytes may
be curated by Mandible). Report emission is not part of the training
run-production domain scanned by the write-surface guard and is not duplicated
here.

## Suite accounting

- Marked `feedbax_contract` collected tests: **126 → 142** (16 new).
- New `write_surface` family: `live`, `minimum_non_skipped = 14`, negative
  canary `test_write_surface_negative_canary_flags_new_durable_write`.
- Full marked suite: 142 passed, 0 failed, 0 skipped; contract meta-test passes.
