# Analysis-system completeness audit (3cf909c wave 1)

This report is the durable output of Mandible issue `85dd226`, wave 1 of
phase umbrella `3cf909c`. It is a read-only audit: it checks whether every
part of rlrmp's analysis system participates in feedbax's recipe/bundle/
manifest contracts (the machinery that governs how analyses declare their
inputs, produce versioned outputs, and get tracked for reproducibility). No
code was changed to produce this report -- it is inventory and diagnosis
only. Section 10 records confirmed dispositions and next steps (confirmed
by the user, 2026-07-03).

## 1. Verdict

The analysis system's compliant core is genuinely feedbax-native -- the
registered recipes, the primitives libraries, and the data-products package
all do the right thing. But there is a systemic gap: nothing in CI watches
over the ad hoc writers living directly under `analysis/`, so new
non-compliant code can land silently. On top of that, one child issue of the
prior audit round (`588483d`), specifically `c4416c5`, was closed as done
despite having landed only about one sixth of what it promised.

Coverage of this audit is complete and high-confidence: 34 pipeline files
were read individually, `math/**`, the top-level `analysis/*.py` files, and
`eval/**` were read in full, the feedbax-side contracts were documented
directly from source, and all 15 children of `588483d` were reconciled
against the code that actually merged (not just against issue labels or
closing comments).

## 2. Method

Four read-only Sonnet lanes did the archaeology that this report formats:

- **(a)** Inventory of rlrmp's `analysis/**` and `eval/**` trees.
- **(b)** Inventory of `data_products`/sidecar/materialization code and the
  `ci/*-allowlist.toml` files.
- **(c)** A reference read of feedbax's analysis contracts, plus
  reconciliation of all 15 `588483d` children against merged code.
- **(d)** A follow-up lane that closed the remaining medium-confidence band
  in the pipeline-file survey and specifically re-verified the `6cfa892`
  diff.

## 3. The compliant core (positive confirmations)

- Three registered recipe modules, wired at `src/rlrmp/__init__.py:145-157`:
  - `analysis/matrix/standard_matrix.py:97` -> `rlrmp.standard_matrix`
  - `analysis/training_diagnostics.py:90` -> `rlrmp.training_diagnostics_summary`
  - `analysis/declarative_materialization.py:85-114` -> six registered types,
    including `rlrmp.certificate.gru_standard`, `rlrmp.robustness_phenotype`,
    and `rlrmp.output_feedback_bridge.rollout_recovery`
- Pure-math backer libraries directly imported by the registered recipes are
  compliant helpers, not recipes themselves: `hinf_riccati.py`,
  `output_feedback.py` (core), `cs_game_card.py` (math),
  `cs_released_simulation.py`, `rerun_metadata.py`, `trial_alignment.py`.
- `eval/**` (`ensemble`, `kinematics`, `pert`, `sisu`, `minimax_io`) is a
  clean primitives library -- correctly *not* a set of recipes, and it should
  stay that way.
- The `generated_data_constant_scan` data-lint is clean tree-wide (0
  violations, verified live). Empirical calibration/anchor data is loaded
  from governed `AnalysisDataProduct`s (`data_products/{calibration,
  broad_epsilon,envelope,lint}.py`) rather than baked in as constants. The
  `data_products/` package is textbook-compliant -- the `calibration` and
  `broad_epsilon` loaders both fail closed, and `broad_epsilon` additionally
  re-verifies its source manifests live rather than trusting a cached hash.
- `diagnostic_provenance.py` is genuinely compliant: it is built on real
  feedbax `RegenerationSpec`/`ArtifactRef`/`Provenance` primitives via
  `model_dump` (imports at `:12-22`, real construction at `:31-95`) -- though
  it is currently consumed under a `compatibility` role label rather than a
  first-class one.
- `feedbax_controllers.py` is covered by the real `import_boundary` CI gate.
  No `sys.path.insert` hacks and no deprecated TODO/FIXME markers turned up
  anywhere in scope.
- `gru_steady_state_perturbation_bank.py` and
  `sisu_spectrum_diagnostics.py` are the two best-practice pipeline
  examples in the codebase: correct tracked/bulk artifact split, and correct
  use of the `update_marked_section` notes convention.

## 4. The dominant structural gap

There is no CI custody over unregistered `analysis/` writers. The
`analysis_recipe_contract` check only validates the 8 registered analysis
types -- it has nothing to say about anything else in the tree. The
`write_surface` gate's scan domain is hard-limited to `scripts/train_*.py`
and `src/rlrmp/train/*.py` (per `ci/feedbax-contract-suite.toml`), so it is
structurally blind to every hand-rolled writer living under `analysis/`.

The `product_identity_hash` family is still `pending_enrollment` (owner
`108b4d3`). `tests/test_write_surface_custody.py:65-70` explicitly declares
`analysis/**` out of scope, deferring to an aspirational "DataProduct/
ReportManifest" substrate -- but `ReportManifest` has zero references
anywhere in the repo. It is aspirational and currently unenforced.

Net effect: 19 of the 34 pipeline files surveyed are script-only and
unregistered. They hand-roll their own note and manifest JSON with no
custody at all, and a brand-new writer of this shape would pass CI silently
today. There is no `7811e47`-style scan-plus-canary-plus-gate that asserts
every materializer under `analysis/` is either registered or explicitly
allowlisted as a library helper.

## 5. 588483d reconciliation

**Feedbax side.** All 8 children (`9970336`, `eda0433`, `854f411`,
`66d3866`, `a6af537`, `58e3bff`, `5f0a2c5`, `e8662b2`) landed real, tested
code on `develop` -- verified by reading the merged code and its focused
tests, not by trusting issue labels. One caveat: the `e8662b2` final audit
was completed by its own parent issue after both worker threads had already
terminated, so it is not an independent audit, and roughly half its claims
concern rlrmp-side coverage that cannot be verified from the feedbax repo
alone. A later package reorganization (`65ac7efe`) rehomed modules into
`feedbax/contracts/`, `feedbax/runtime/`, and `feedbax/analysis/`, so
pre-reorg path citations in the older children are stale even though the
surfaces themselves survived the move.

**rlrmp side.** Six of seven children genuinely adopted the contracts they
promised:

| Child | Merge | Note |
|---|---|---|
| `34d7ce4` | `241db3d` | -- |
| `7f65080` | `faf2138` | Module moved to `rlrmp.runtime.spec_migrations`, a superset of the original scope |
| `2805498` | `864a986` | -- |
| `af77a06` | `e24fdf7` | -- |
| `769aea6` | `c5c5110` | Minor unreachable dead branch: `_formal_hinf_claim` `game_card` hardcoded `False` |
| `0e3223d` | `b7f0316` | -- |

**`c4416c5` -- the one real gap.** This child promised six output-feedback
bridge diagnostics migrated to feedbax bundles; only one (rollout recovery)
actually landed as a recipe. The other five stayed bespoke scripts. The
issue was nonetheless closed done/merged on roughly one sixth of its stated
criteria, and the remaining five diagnostics are currently untracked. This
audit reopens `c4416c5`.

## 6. Per-surface gap table

| Surface | Path:anchor | Verdict | Reason |
|---|---|---|---|
| `gru_checkpoint_selection.py` | `:40-42,528,634` | Non-compliant (high-leverage, 9+ consumers) | Invents 3 local unversioned schema strings, none registered in `runtime/spec_migrations.py`; ignores feedbax `CheckpointSelectionManifest` (`feedbax/contracts/manifest.py:504`); raw `write_text` |
| `gru_pilot_figures.py` | `:301,732,840` | Non-compliant | Raw `fig.write_html` bypassing `feedbax.plot.save_figure` dual-tree routing; no schema version |
| `output_feedback_affine_tracker.py` | -- | Non-compliant | Unregistered script-only materializer; hand-rolled note+manifest JSON, no custody |
| `output_feedback_interpolated_starts.py` | -- | Non-compliant | Same pattern |
| `output_feedback_linear_recurrent.py` | -- | Non-compliant | Same pattern |
| `output_feedback_phase_modulated_recurrent.py` | -- | Non-compliant | Same pattern |
| `output_feedback_time_constrained.py` | -- | Non-compliant | Same pattern |
| `cs_stochastic_phase1.py` | -- | Non-compliant | Same pattern |
| `cs_stochastic_phase3.py` | -- | Non-compliant | Same pattern |
| `sisu_perturbation_comparison.py` | -- | Non-compliant | Same pattern |
| `bridge_aggregation.py` | -- | Non-compliant | Same pattern |
| `failure_decomposition.py` (driver) | -- | Non-compliant | Same pattern |
| `gru_feedback_ablation.py` | via `gru_postrun_materialization.py` | Partial | Schema family registered transitively, but hand-rolled writes + notes-convention violation |
| `gru_perturbation_bank.py` | via `gru_postrun_materialization.py` | Partial | Same |
| `gru_map_error_decomposition.py` | via `gru_postrun_materialization.py` | Partial | Same |
| `objective_comparator.py` | via `gru_postrun_materialization.py`, `:5` | Partial | Same, plus an `import argparse` CLI smell |
| `hinf_phenotype_sidecar.py` | `:142-143`; schema at `runtime/spec_migrations.py:289-297` | Partial | Recipe-backed and schema-governed (`rlrmp.hinf_phenotype_sidecar.v1`), but hand-rolled write and notes-convention violation |
| `bridge_contracts.py` | `:288-296` | Partial | The rlrmp-owned `BridgeRunManifest` envelope underpins the whole bridge/certificate stack (16+ referencers), but it reaches feedbax only via opaque wrapping |
| `math/cs_game_card.py` | `:644` (`write_outputs`) | Non-compliant | Raw writer |
| `math/adversary_equivalence.py` | `:581` (`write_outputs`) | Non-compliant | Raw writer |
| `math/linear_equivalence_certificate.py` | `:672` (`write_outputs`) | Non-compliant | Raw writer |
| `math/linear_round_trip.py` | `:930` (`write_outputs`) | Non-compliant | Raw writer |
| `math/robust_bellman.py` | `:2162` (`write_outputs`) | Non-compliant | Raw writer |
| `math/output_feedback.py` | `:2592,2690` (`write_outputs`) | Non-compliant | Raw writer |
| `frozen_policy_gate.py` | `validate_direct_hvp_lambda_source:259` | Ungated | Gate-shaped, fail-closed function with zero CI enforcement (unlike the `7811e47` pattern) |
| `lambda_recommendations.py` | `_require_launch_candidate:121` | Ungated | Same |
| `induced_gain.py` | -- | Partial | Feeds `hinf_phenotype_sidecar` via a stringly-typed JSON key handoff, not an import |
| `rollout_cleanup.py` | -- | Risk | Destructive `.npz` deletion checked only by file existence, no content-hash pre-delete verification, no CI gate |
| `delayed_diagnostic_bundle.py` | -- | Dormant | Registered schema family, zero production callers (tests only) |

## 7. Convention violations (Bug 06f7faf and others)

Only 2 of the 34 pipeline files surveyed use `update_marked_section`; the
rest overwrite `results/*/notes/*.md` with raw `write_text`, which destroys
any hand-edited preamble a human added -- for example,
`robust_bellman.py::write_outputs` overwrites
`results/583d764/notes/robust_bellman.md` wholesale on every run.

The `linear_*` files bake a bare `python scripts/materialize_*.py` command
into tracked `regeneration_command` JSON, which violates the "never invoke
python bare" convention -- it should be `uv run --no-sync python ...`.

Several manifests embed `*_out_of_scope: True` flags in place of filing a
proper follow-up issue, which loses the finding once the manifest is no
longer being actively read.

## 8. Confirmed: 6cfa892 double-serialization (unremediated)

`results/6cfa892/scripts/materialize_closed_loop_soft_lambda_redo.py:232-234`
appends the same `row_dict` to both `row_dicts` and `flat_rows`. At
`:303-304` the payload then includes both `rows` (nested) and `flat_rows`.
At `:173`, `write_compact_json` serializes the whole blob, doubled, into the
tracked `results/6cfa892/closed_loop_soft_lambda_redo.json`, with no split
to `_artifacts/`. Every row is serialized twice into a tracked file.

The remediation pattern already exists in the sibling script
`results/d469108/scripts/materialize_adam_soft_lambda_redo.py:181`: its
`split_payload` helper drops `rows`/`flat_rows` from the tracked slim
output and routes the bulk detail to `_artifacts/` with a sha256 pointer
instead. The `6cfa892` script has not been brought in line with this
pattern.

## 9. Deletion / de-shim candidates + allowlist inventory

- `ci/legacy-pattern-allowlist.toml`:
  - `branded_component_ids` -- 18 `RLRMP*` GraphSpec component-ID literals
  - `argparse_training_entry_points` -- 5 entries: `cs_nominal_gru`,
    `train_minimax`, `closed_loop_distillation`, `guided_distillation`,
    `train_part2_5`
  - `run_spec_writer_sites` -- 1 entry: `train_part2_5.py::run_training`
    raw `json.dump`
- `ci/retired-component-id-confinement.toml`: 8 python-scope entries, 4 test
  files, 3 glob entries (archived results JSON and conversion/audit code)
- `ci/write-surface-allowlist.toml`: training-domain raw-write sites across
  `train_minimax`, `train_part2_5`, `closed_loop_distillation`,
  `guided_distillation`, `cs_nominal_gru`
- Roughly 9 misplaced top-level `scripts/materialize_*.py` single-issue
  launchers that belong under `results/<hash>/scripts/` per Bug `8404108`
- `object.__setattr__` de-shim sites: `pipelines/_selected_eval_rollouts.py`,
  `pipelines/bridge_controllers.py`, `pipelines/gru_perturbation_bank.py`,
  `pipelines/gru_worst_case_epsilon_audit.py`, `eval/minimax_io.py`
- `delayed_diagnostic_bundle.py` (dormant, see Section 6), and the `legacy/*`
  `ExistingAnalysisArtifact` role prefixes in
  `declarative_materialization.py`

## 10. Confirmed dispositions (user, 2026-07-03)

The recommendations originally listed in this section have been reviewed
and confirmed by the user. This section records the confirmed dispositions
and the current status of the follow-up work they authorized.

**Materializer tail -- fix live, delete legacy.**
- Fix the live-but-noncompliant materializer surfaces (Section 6) in place;
  delete legacy ones outright (git history preserves them, so no
  deprecation shim is needed).
- The five `output_feedback_*` variants (Section 6 / Section 9) get
  per-diagnostic triage individually: port each one to a feedbax bundle if
  it still serves active science, otherwise delete it.

**`c4416c5` -- reopened.**
This audit reopened `c4416c5` (Section 5). Its five unported diagnostics
fold into the same per-diagnostic triage as the `output_feedback_*`
materializer tail above, rather than being tracked as a separate
re-migration effort.

**Wave 2 -- dispatched and completed 2026-07-03.**
- `c223bb8` -- analysis write-custody CI gate plus `product_identity_hash`
  enrollment; the gate now covers 160 marked tests.
- `5e01c2b` -- checkpoint-selection custody migrated onto feedbax's
  `CheckpointSelectionManifest`.
- `dcdba85` -- the `6cfa892` double-serialization fix (Section 8): tracked
  JSON dropped from 184KB to 19KB, with sha-pinned `_artifacts/` detail;
  plus notes-convention conversion of the 11 live materializers onto
  `update_marked_section`.

**Sequencing -- blocked on pipeline-alignment design (`e1ad278`).**
All remaining materializer-tail porting (fix-live / delete-legacy above)
and the `c4416c5` per-diagnostic triage block on the pipeline-alignment
design tracked at `e1ad278` (training -> eval -> analysis -> report,
eval-stage extraction). See the finding below for the rationale.

### Eval-stage finding (post-audit)

rlrmp currently runs evals inline inside analysis materializers -- only one
registered evaluation recipe exists, `rlrmp.standard_matrix_evaluation`,
and it is a cache shim -- while feedbax already provides the eval stage as
a first-class contract: `EvaluationRunSpec` at
`feedbax/contracts/manifest.py:392`, an evaluation-recipe registry at
`feedbax/analysis/evaluation.py:58`, and bundle stage kinds
`evaluation|analysis|materialization|report`. Porting the monolith
materializers onto compliant shape before that design lands would cement
the wrong shape -- hence the sequencing block above.

**Deferred (unchanged from the original recommendation):**
- Whether `bridge_contracts.py`'s `BridgeRunManifest` envelope should stay
  an opaque rlrmp-owned wrapper or be migrated onto feedbax's manifest
  types directly.

**Correction on carryover item `00f97d5`:** feedbax's `--batched` flag is
not marked deprecated in source. Its removal in `00f97d5` reflects
project-intent only, not an upstream deprecation -- this should not be
cited as feedbax-driven.
