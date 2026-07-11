# 7ae2916 — legacy checkpoint migration: attempted, abandoned by decision

Final outcome record, 2026-07-11.

## Outcome

Issue 7ae2916 originally asked for a one-time migration of the June-2026
020a65b/e901a20 checkpoint family into the current native checkpoint format,
followed by retirement of the compat reader
`src/rlrmp/eval/legacy_checkpoints.py`. The migration was investigated, found
to be blocked on an architecture-level weight port (see below), and
**deliberately abandoned by user decision on 2026-07-11**. The issue's
delivered outcome is instead:

- The checkpoint bytes under `_artifacts/020a65b/` and `_artifacts/e901a20/`
  stay in their historical format permanently. They are readable only via
  historical code revisions (checkouts predating the feedbax-native Graph
  decomposition, umbrella `64a04e0`).
- `src/rlrmp/eval/legacy_checkpoints.py` and
  `tests/eval/test_legacy_checkpoints.py` were deleted. The module was
  non-functional against the current codebase: every one of its three readers
  failed for every one of the 11 target run directories (verified
  empirically, see "The blocker" below), so it documented a capability that
  did not exist.
- The two consumer scripts
  (`results/e901a20/scripts/materialize_nominal_velocity_profile_comparison.py`
  and `results/e901a20/scripts/materialize_nonh0_no_pgd_extlqg_velocity.py`)
  are retained for provenance with LEGACY banners; their checkpoint-loading
  call sites now raise a loud `RuntimeError` pointing at the legacy inventory
  instead of pretending to load.
- The inventory entry in `results/3cf909c/notes/legacy_materializers.md`
  ("e901a20 legacy Equinox checkpoints") was updated in place to record the
  deletion and the decision.

Revisit only if a concrete re-materialization need for these runs arises; in
that case the port described under "The blocker" is the prerequisite.

## Scope that was established

- **Target files (231)**: 11 `trained_model.eqx` + 220 `checkpoints/*/model.eqx`
  across 11 run directories (9 under `_artifacts/020a65b/runs/`, 2 under
  `_artifacts/e901a20/runs/`: `h0_policy_adversary__plain`,
  `h0_policy_adversary__energy`).
- **Excluded**: `trained_policy_adversary.eqx` (2) and
  `checkpoints/*/adversary_policy.eqx` (48) under e901a20. A whole-repo grep
  found no consumer of these files (only the training executor writes them
  and one training test asserts they exist after a run); they were never in
  reconstruction scope.
- **hps sourcing** (as the materializer scripts do): flat tracked run spec
  `results/<exp>/runs/<run_id>.json` when present (3 of 11 runs); otherwise
  the identical run-spec dict embedded as the JSON first line of the run's
  own `trained_model.eqx`. Then
  `dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)`
  and `seed = int(run_spec.get("seed", 42))`.
- **Current native format** (from the live writers in
  `src/rlrmp/train/executor/checkpoints.py` / `cs_supervised.py`):
  `trained_model.eqx` is `jax_cookbook.save` with a JSON run-spec header;
  `checkpoints/<name>/model.eqx` is plain `eqx.tree_serialise_leaves`.
  Migration precedent:
  `results/9727d79/scripts/migrate_distillation_artifacts.py`.

## The blocker (empirical)

`rlrmp.eval.legacy_checkpoints` assumed `model.nodes["net"]` is a monolithic
`SimpleStagedNetwork` (wrapped in `InitialHiddenStagedNetwork` for H0 runs)
with `.net.readout` / `.net.dtype` / population-structure fields — the
June-2026 on-disk byte layout. The current `setup_task_model_pair` (post
feedbax-native alignment, umbrella `64a04e0`; e.g. commits `5976d44d`,
`e6dbb880`, `61ed4604`) builds `model.nodes["net"]` as a decomposed
Feedbax-native `Graph` with sub-nodes `cell` (GRU), `input_mux` (Mux),
`readout` (Linear), plus `h0_encoder` (Linear) and `hidden_source` (Gain) for
H0 runs.

Confirmed by directly invoking the production function
`materialize_nominal_velocity_profile_comparison.evaluate_profile(NO_PGD_REF)`
in-process: it raised `AttributeError: 'Graph' object has no attribute 'net'`
inside `legacy_checkpoints.force_legacy_masked_readout`. A probe across all
11 run directories showed all three readers
(`load_trained_model_compatible`, `load_checkpoint_model_compatible`,
`load_nonh0_checkpoint_model_compatible`) failing identically
(`AttributeError: 'Graph' object has no attribute 'net'` or `TreePathError`
at `nodes.net.nodes.hidden_source.gain` / `nodes.sensory.delay`), for H0 and
non-H0 runs alike.

A correct migration would therefore require a monolithic-`SimpleStagedNetwork`
→ decomposed-`Graph` weight port (old/new leaf correspondence, `input_mux`
port ordering, `hidden_source` gain semantics, H0 vs non-H0 topology), not
just the MaskedLinear/float64/replicated-scalar reconstruction the deleted
module implemented. The requested bitwise reload-vs-reference verification
could not catch a wrong port, because both sides would flow through the same
porting code; only behavioral comparison (e.g. rollouts) against a
historical-revision load would. The user decided this port is not worth
doing.

## Not landed: training_configs.py legacy-provenance fallback

During investigation, a second (independent) blocker was found and
provisionally fixed, then **reverted per the same decision** — it would have
reintroduced exactly the compat-fallback class that issue `ef8e1df` retired,
and with the migration abandoned it has no consumer. Recorded here so a
future port effort can re-derive it: the June-2026 hps blobs spell PGD budget
provenance as `budget_contract.source_issue` / `budget_contract.source_note`,
while the current normalizer requires `budget_contract.budget_source.key`;
without a fallback, `setup_task_model_pair` on any of the 11 runs' hps fails
in `PgdFullStateEpsilonTrainingConfig` validation with "fixed PGD L2 radius
requires explicit provenance" before checkpoint bytes are even touched.

The reverted hunk (was verified against
`tests/test_training_config_flat_legacy_keys.py` and
`tests/test_cs_nominal_gru.py -k "pgd or broad_epsilon"`, all passing):

```diff
--- a/src/rlrmp/train/training_configs.py
+++ b/src/rlrmp/train/training_configs.py
@@ -3030,7 +3030,18 @@ def _normalize_broad_epsilon_pgd_payload(config: Any) -> PgdFullStateEpsilonTrai
         None
         if budget_schedule != BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE
         else _optional_str(
-            _payload_get(budget_source, "key", None)
+            # Current-shape provenance is `budget_contract.budget_source.key`.
+            # June-2026 020a65b/e901a20 hps blobs (frozen, issue ef8e1df)
+            # predate `budget_source` and instead recorded provenance at
+            # `budget_contract.source_issue` / `.source_note`. Fall back to
+            # those so historical hps payloads keep validating; this does not
+            # change behavior for payloads that already carry `budget_source`.
+            _first_payload_value(
+                (budget_source, "key"),
+                (budget_contract, "source_issue"),
+                (budget_contract, "source_note"),
+                default=None,
+            )
         )
     )
     adam = _payload_get(inner, "adam", None)
```

## Cross-references

- Inventory entry: `results/3cf909c/notes/legacy_materializers.md`,
  "e901a20 legacy Equinox checkpoints".
- Freeze provenance: issue `ef8e1df` (Pass C), which centralized the inline
  shims into the (now deleted) archival reader and deferred migration.
- Recovery shape for the deleted module and for checkpoint-reading code:
  historical revisions in this repo's history (any tree at or before the
  7ae2916 deletion commit carries `src/rlrmp/eval/legacy_checkpoints.py`;
  actually *loading* the checkpoint bytes additionally requires a checkout
  predating the `64a04e0` Graph decomposition).
