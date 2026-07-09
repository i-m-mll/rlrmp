# Remediation portfolio: code-archaeology audit (issue `05883e7`)

**Remediation is out of scope for this audit.** Per the audit's own scope
boundary (`results/05883e7/notes/audit_plan.md`) and the project's issue
policy, this document is a ranked *plan*, not a to-do list this umbrella will
execute. Every item below graduates to its own issue — filed in rlrmp for
rlrmp-side items, filed in the feedbax repo (with a `surfaced from rlrmp`
cross-link) for feedbax-side items — only when someone actually picks it up.
Nothing here should be read as already-authorized work.

The underlying machine-readable data is
`_artifacts/05883e7/audit/synthesis/portfolio.json` (10,117 lines, one entry
per remediation item across classes (a)-(g)), produced deterministically by
`results/05883e7/scripts/synthesize.py` from the post-verification
classification corpus. This note curates the highest-impact items per class;
for the full list of every module/cluster, read the JSON directly (each entry
has the same `id` / `scope` / `est_loc_delta` / `risk_note` /
`suggested_issue_title` fields shown in the tables below).

**How items are ranked.** Within each class, items are sorted by the
magnitude of `est_loc_delta` (LOC impact). Confidence is not a separate
numeric field in the underlying data — instead it is a *class-level*
property (documented per class below and in
`results/05883e7/notes/synthesis.md`'s caveats section): classes (a) and the
`merge_dedupe`-flagged members within (c) are independently verification-gated;
classes (b)/(c)/(e)/(g) are classification- or heuristic-confidence only;
class (d) and the single `move_to_feedbax` record were not verified at all.
Ranking by LOC-impact within a class is therefore already ranking within a
roughly uniform confidence band; where an item's own risk note carries an
explicit per-object confidence breakdown (class (a)), that is preserved in
the risk-note text.

## Class (a): confirmed-dead deletions — 107 modules / 287 objects, -5,401 LOC

**Confidence: highest in this audit.** Every one of the 287 `delete`-disposition
records contributing to this class was independently re-derived by Phase 4
verification, not sampled — 2 were refuted back to `keep`, 1 `needs_decision`
record was promoted into `delete`. This is the only LOC figure in the whole
audit that is fully verification-gated end to end. All items are rlrmp- or
feedbax-side deletions (no relocation) — feedbax items graduate to feedbax
issues, rlrmp items to rlrmp issues.

| Scope | LOC delta | Suggested issue title | Risk note (truncated) |
|---|---:|---|---|
| feedbax:feedbax/plot/mpl.py | -1194 | Delete confirmed-dead code in feedbax/plot/mpl.py | 29 objects, confidence {high:25, medium:4}. 3D matplotlib trajectory plot, rotation animator, circular histogram, and other unreferenced plotting helpers superseded by the plotly-based `feedbax.plot` package. |
| rlrmp:src/rlrmp/train/cs_perturbation_training.py | -827 | Delete confirmed-dead code in src/rlrmp/train/cs_perturbation_training.py | 8 objects, confidence {high:8}. Includes a helper superseded by `_family_mask`, and a hardcoded smoke row plus two locked lr1e-3/lr3e-3 rows for tracking issue `ba82f3d` baked in as Python literals even though `results/ba82f3d/runs/` already has the canonical spec. |
| feedbax:feedbax/persistence/database.py | -491 | Delete confirmed-dead code in feedbax/persistence/database.py | 14 objects, confidence {medium:7, high:7}. Model-save/record entry points with no live callers and an explicit downstream stub marking them unavailable; part of the legacy-DB layer also flagged for retirement in class (b). |
| rlrmp:src/rlrmp/train/cs_nominal_gru.py | -403 | Delete confirmed-dead code in src/rlrmp/train/cs_nominal_gru.py | 5 objects, confidence {high:5}. Locked run rows for a completed experiment (`ef9c882`) baked into `src/` as Python literals and named after its tracking issue — the exact "issue-hash-named run planner" symptom that motivated this audit. |
| feedbax:feedbax/plot/experiments.py | -271 | Delete confirmed-dead code in feedbax/plot/experiments.py | 7 objects, confidence {medium:6, high:1}. Deprecated aliases and a zero-caller eigenvalue-scatter helper. |
| feedbax:feedbax/config/tree.py | -228 | Delete confirmed-dead code in feedbax/config/tree.py | 16 objects, confidence {high:9, medium:7}. A YAML-diffing helper family (`filter_varying_leaves` and its dependents) with zero call sites; one independently flagged untested by a review doc. |
| feedbax:feedbax/analysis/aligned.py | -208 | Delete confirmed-dead code in feedbax/analysis/aligned.py | 27 objects, confidence {high:27}. The `ALL_MEASURES`/`MEASURE_LABELS` catalog and every helper that exists solely to populate it — repo-wide grep confirmed zero consumers of the catalog itself, so the whole cluster dies together. See feedbax backlog "Measure catalog cluster." |
| rlrmp:src/rlrmp/train/guided_distillation.py | -171 | Delete confirmed-dead code in src/rlrmp/train/guided_distillation.py | 8 objects, confidence {high:8}. Save/load pair for a superseded local checkpoint format, orphaned by the switch to feedbax's native executor checkpointing. |
| rlrmp:src/rlrmp/analysis/pipelines/gru_feedback_ablation.py | -149 | Delete confirmed-dead code in src/rlrmp/analysis/pipelines/gru_feedback_ablation.py | 3 objects, confidence {high:3}. Transitively dead: only caller is itself a dead function. |
| feedbax:feedbax/mechanics/muscle.py | -142 | Delete confirmed-dead code in feedbax/mechanics/muscle.py | 14 objects, confidence {high:14}. The entire unused legacy "Virtual Muscle" model — distinct from the live `hill_muscles.py` module, despite similar naming. |
| feedbax:feedbax/training/rl/tasks.py | -128 | Delete confirmed-dead code in feedbax/training/rl/tasks.py | 2 objects, confidence {high:2}. Self-documented deprecated function whose replacement is already the production path in `env.py`/`ppo.py`. |
| feedbax:feedbax/plot/plotly.py | -116 | Delete confirmed-dead code in feedbax/plot/plotly.py | 1 object, confidence {medium:1}. Duplicate of `feedbax/analysis/activity.py:activity_sample_units` under the same name; this copy has no callers. |
| feedbax:feedbax/analysis/state_utils.py | -98 | Delete confirmed-dead code in feedbax/analysis/state_utils.py | 7 objects, confidence {medium:7}. Dead geometry/epoch-splitting helpers; the live subset of this file is load-bearing (see feedbax backlog), only this dead slice is in scope here. |
| feedbax:feedbax/training/support.py | -91 | Delete confirmed-dead code in feedbax/training/support.py | 5 objects, confidence {high:5}. Zero-caller timing/context utilities. |
| feedbax:feedbax/bin/_orchestrate.py | -55 | Delete confirmed-dead code in feedbax/bin/_orchestrate.py | 5 objects, confidence {high:5}. Whole module: a speculative orchestration helper never adopted by either of the loops it was meant to unify. |
| feedbax:feedbax/xabdeef/losses.py | -46 | Delete confirmed-dead code in feedbax/xabdeef/losses.py | 2 objects, confidence {high:1, medium:1}. Dead module-level logger plus a zero-test-coverage loss factory. |
| rlrmp:src/rlrmp/fb_response.py | -44 | Delete confirmed-dead code in src/rlrmp/fb_response.py | 2 objects, confidence {high:2}. Whole module: an unregistered, unimported `AbstractAnalysis` subclass. |
| feedbax:feedbax/analysis/setup.py | -41 | Delete confirmed-dead code in feedbax/analysis/setup.py | 2 objects, confidence {medium:2}. Zero-reference helpers, distinct from the legacy `query_and_load_model` chain flagged separately in class (b)/feedbax backlog. |
| feedbax:feedbax/plot/utils.py | -38 | Delete confirmed-dead code in feedbax/plot/utils.py | 2 objects, confidence {medium:1, high:1}. Superseded by the live top-level `savefig`. |
| rlrmp:src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py | -38 | Delete confirmed-dead code in src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py | 2 objects, confidence {high:2}. Zero-caller helpers superseded by inline handling in the live code path. |

87 more items (down to single-object, 1-2 LOC deletions) are in
`portfolio.json`'s `a_confirmed_dead_deletions` array. Not included: an empty
`a_confirmed_dead_deletions_footnote` structural placeholder for
unverified-delete candidates — empirically empty this run, since the delete
population was checked exhaustively rather than sampled.

## Class (b): legacy-tree retirements — 40 items, -13,459 LOC

**Confidence: mixed.** `legacy_only`-status clusters (32 rlrmp items,
`b_legacy/*` ids) are classification-confidence only — retirement is
explicitly gated on a human LEGACY-banner port-or-delete decision per
CLAUDE.md's convention, not a mechanical call. The 7 feedbax reverse-audit
items (`b_reverse_audit/*` ids) were each individually investigated against
real consumer evidence (see `sweeps/reverse_audit/summary.md`) and carry
higher confidence; these are the same 7 modules covered in
`results/05883e7/notes/feedbax_backlog.md`'s "deprecate/delete" entry — file
them together as one feedbax cleanup wave, not seven independent issues.

| Scope | LOC delta | Suggested issue title | Risk note (truncated) |
|---|---:|---|---|
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py | -2422 | Retire or port legacy_only code | 82 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_affine_tracker.py | -1989 | Retire or port legacy_only code | 49 legacy_only objects; gated on LEGACY-banner decision. |
| **feedbax**:feedbax/plot/mpl.py | -1302 | Deprecate/delete feedbax module (reverse audit) | Zero consumers repo-wide; superseded by plotly-based `feedbax.plot`. Check merge-spec rationale (`docs/design/feedbax_merge_spec.md:212`) before deleting — it was a deliberate retention at merge time. |
| rlrmp:src/rlrmp/analysis/math/output_feedback.py | -1291 | Retire or port legacy_only code | 14 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:src/rlrmp/analysis/pipelines/sisu_perturbation_comparison.py | -646 | Retire or port legacy_only code | 28 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:src/rlrmp/analysis/pipelines/cs_stochastic_phase1.py | -639 | Retire or port legacy_only code | 25 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:scripts/materialize_output_feedback_failure_decomposition.py | -631 | Retire or port legacy_only code | 37 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:src/rlrmp/analysis/math/robust_bellman.py | -613 | Retire or port legacy_only code | 3 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py | -591 | Retire or port legacy_only code | 24 legacy_only objects; gated on LEGACY-banner decision. |
| **feedbax**:feedbax/analysis/setup.py | -454 | Deprecate/delete feedbax module (reverse audit) | Only live call chain is a legacy sqlite `db_session` path rlrmp no longer touches; retire together with `bin/db_merge.py` in one wave. |
| rlrmp:scripts/materialize_output_feedback_sweep_certificates.py | -430 | Retire or port legacy_only code | 23 legacy_only objects; gated on LEGACY-banner decision. |
| **feedbax**:feedbax/bin/db_merge.py | -355 | Deprecate/delete feedbax module (reverse audit) | No consumer anywhere; retire alongside `analysis/setup.py`'s legacy sqlite chain. |
| rlrmp:scripts/materialize_output_feedback_observer_error_coverage.py | -341 | Retire or port legacy_only code | 24 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:scripts/materialize_output_feedback_optimizer_basin_diagnostic.py | -269 | Retire or port legacy_only code | 14 legacy_only objects; gated on LEGACY-banner decision. |
| rlrmp:src/rlrmp/analysis/math/cs_game_card.py | -232 | Retire or port legacy_only code | 4 legacy_only objects; gated on LEGACY-banner decision. |
| **feedbax**:feedbax/analysis/profiles.py | -156 | Deprecate/delete feedbax module (reverse audit) | Zero resolved consumers in either repo; candidate for the dead-code sweep rather than a move. |
| **feedbax**:feedbax/analysis/effector.py | -149 | Deprecate/delete feedbax module (reverse audit) | Zero importers and no analysis-recipe registration; recreate via `ScatterPlots`/`AlignedVars` if wanted later, don't revive this orphan. |
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py | -95 | Retire or port legacy_only code | 7 legacy_only objects; gated on LEGACY-banner decision. |
| **feedbax**:feedbax/config/defaults.py | -48 | Deprecate/delete feedbax module (reverse audit) | `POS_ENDPOINTS_ALIGNED` has zero references anywhere; the rest is consumed only by the legacy `feedbax/analysis/*` chain — retire together if that chain retires. |
| **feedbax**:feedbax/web/ws/simulation.py | -16 | Deprecate/delete feedbax module (reverse audit) | Zero test coverage, zero frontend consumer, hardcoded no-op payload. Either wire it to a real stream or remove the router mount. |

Remaining 20 items (mostly smaller `scripts/materialize_*` legacy sidecars, 6-93
LOC each, plus `feedbax/plot/mpl.py`'s 1 remaining `legacy_only`-tagged object
distinct from its class-(a)/reverse-audit entries above) are in
`portfolio.json`'s `b_legacy_tree_retirements` array.

## Class (c): dedupe/promotion (top clusters) — 79 of 978 clusters, -15,478 LOC

**Confidence: heuristic-based (LSH near-duplicate detection over token
shingles), not classification-confidence-gated.** Cluster membership does not
imply the classification corpus called these objects `merge_dedupe` — most
members are still individually `keep` per Phase 2 (the risk note's
"classification dispositions among members" breakdown shows this explicitly
per cluster). This is a *separate* structural signal: "these N implementations
are textually near-identical," independent of whether any one of them was
individually flagged for deletion. Only the ≥100-redundant-LOC subset (this
table) is broken into named portfolio items; the long tail (899 more clusters,
~21,700 more redundant LOC) is real but not individually worth a dedicated
issue — see `sweeps/duplication/summary.md` for the full list and the
heuristic's documented false-positive shapes (framework-family classes like
Equinox norm-layer wrappers cluster by design, not by defect).

| Scope | LOC delta | Suggested issue title | Risk note (truncated) |
|---|---:|---|---|
| dup_0001 | -873 | Dedupe class cluster (24 members) | Similarity 0.73. **Not a defect** — feedbax's Equinox norm-layer wrapper family (`BatchNorm`/`LayerNorm`/`GroupNorm`/...); expected framework shape. |
| dup_0002 | -521 | Dedupe function cluster (2 members) | Similarity 0.92. Two `results/*/scripts/` experiment-driver `main()` bodies (`analyse_lit_replication_6cell.py`, `analyse_anti_anticipation_6cell_variance.py`) — real promotion candidate. |
| dup_0003 | -487 | Dedupe function cluster (14 members) | Similarity 0.72. `add_band_trace`/`add_reference_trace` plotly helpers copy-pasted across ~7 `results/*/scripts/materialize_*velocity_profiles*.py` drivers — `rlrmp.viz` promotion candidate. |
| dup_0004 | -464 | Dedupe function cluster (8 members) | Similarity 0.77. Parametrized-style test bodies within one test module — pytest `parametrize` consolidation, not cross-file duplication. |
| dup_0005 | -461 | Dedupe function cluster (8 members) | Similarity 0.81. Figure-builder helpers duplicated between two `results/*/scripts/` analysis drivers. |
| dup_0006 | -447 | Dedupe function cluster (9 members) | Similarity 0.79. `add_profile_trace`-shaped plotly helper repeated across 3 more results-scripts figure drivers — same family as dup_0003. |
| dup_0007 | -405 | Dedupe function cluster (10 members) | Similarity 0.77. **Verified by reading source:** `write_outputs` is LEGACY-banned in `adversary_equivalence.py` (frozen, issue `64d5f13`) but the same shape is copy-pasted across ~10 `src/rlrmp/analysis/math/*.py` modules anyway. |
| dup_0008 | -372 | Dedupe function cluster (2 members) | Similarity 0.70. **Verified by reading source:** two large PGD inner-maximizer implementations duplicated within the same file, `cs_perturbation_training.py` — a real in-file extraction candidate. |
| dup_0009 | -368 | Dedupe class cluster (6 members) | Similarity 0.95. **Not a defect** — another feedbax framework-family cluster (`ConvTranspose{1,2,3}d`-style). |
| dup_0010 | -325 | Dedupe function cluster (13 members) | Similarity 0.67. CSV-writer helpers duplicated across the soft-lambda-sweep `results/*/scripts/` family. |
| dup_0011 | -272 | Dedupe function cluster (9 members) | Similarity 0.59. In-file duplication within `output_feedback_phase_modulated_recurrent.py` (part of the pipelines hot zone). |
| dup_0012 | -267 | Dedupe function cluster (3 members) | Similarity 0.82. Three near-identical output-writer variants in one script — looks like compat-shim accretion, not cross-file duplication. |
| dup_0013 | -263 | Dedupe function cluster (4 members) | Similarity 0.89. `materialize_figure` driver duplicated across the PGD/soft-PGD/beta sweep script family. |
| dup_0014 | -262 | Dedupe function cluster (4 members) | Similarity 0.88. Overlay-figure builders, same promotion family as dup_0003/0006/0013. |
| dup_0015 | -242 | Dedupe function cluster (5 members) | Similarity 0.83. Setup-helper duplicated across the PGD-1p05/beta1p4 script families. |
| dup_0016 | -238 | Dedupe function cluster (11 members) | Similarity 0.81. **Verified by reading source:** `rlrmp.eval.pert.eval_at_pert_scale` is the current canonical fixed-perturbation-scale helper; two legacy scripts hand-roll the same logic instead of calling it — a directly-verified delete/replace-with-import case. |
| dup_0017 | -225 | Dedupe function cluster (3 members) | Similarity 0.84. Per-row stabilization-diagnostic evaluator duplicated across 3 `results/*/scripts/` drivers. |
| dup_0018 | -222 | Dedupe function cluster (27 members, **cross-repo**) | Similarity 0.67. Feedbax's `component_registry` `*_output_prototype` shape-descriptor family (~24 near-identical functions) plus rlrmp's own hand-written counterparts. Legitimate registration convention, but the sheer count is worth flagging to feedbax as a declarative-shape-spec-table candidate (see gate proposals). |
| dup_0019 | -213 | Dedupe function cluster (7 members) | Similarity 0.62. Parametrized-style test bodies — pytest consolidation candidate. |
| dup_0020 | -208 | Dedupe class cluster (6 members) | Similarity 0.83. **Not a defect** — feedbax's small stateful discrete-component family (`ZeroOrderHold`, `RateLimiter`, `Derivative`). |

59 more clusters (down to ~100 redundant LOC each) are in `portfolio.json`'s
`c_dedupe_promotion` array; the full 978-cluster list (including sub-100-LOC
clusters) is `sweeps/duplication/duplication.jsonl`.

## Class (d): move to `results/*/scripts/` — 10 modules / 79 objects, 1,691 LOC relocated

**Confidence: not independently verified** (per `synthesis.md`'s caveats,
this population "was not verified at all" by Phase 4). Relocation only — no
deletion, per CLAUDE.md's script-placement policy. All rlrmp-side. Full list
(all 10 items — small enough to show in full):

| Scope | LOC | Suggested issue title |
|---|---:|---|
| rlrmp:src/rlrmp/analysis/pipelines/sisu_spectrum_diagnostics.py | 662 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/eval_part2_5_figures.py | 317 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/probe_round_trip_ratio.py | 193 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/diag_probe_anomalies.py | 158 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/eval_part2_5.py | 157 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:src/rlrmp/train/guided_distillation.py | 23 (13 objects) | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/diag_cs_bw_full_state_sweeps.py | 61 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:src/rlrmp/cloud/modal_runner.py | 50 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/diag_cs_bw_full_state.py | 45 | Move experiment-specific code into `results/<hash>/scripts/` |
| rlrmp:scripts/diag_cs_baseline.py | 25 | Move experiment-specific code into `results/<hash>/scripts/` |

Note the overlap with `module_structure.md`'s RELOCATE ranking: several of
these (`sisu_spectrum_diagnostics.py` at 85% flagged fraction,
`eval_part2_5_figures.py` and `eval_part2_5.py` at 100%) can be relocated
wholesale with no residual split needed.

## Class (e): contract-violation remediation — 110 modules / 362 objects, 11,889 LOC

**Confidence: not independently verified.** These are objects flagged
`data_in_code`, `experiment_named_in_src`, or `spec_first_violation` by the
classification corpus's contract-flag machinery — remediation is migration to
a governed data product or spec surface (per CLAUDE.md's "data stays separate
from code" and spec-first policies), not deletion. All rlrmp-side.

| Scope | LOC delta | Suggested issue title | Risk note (truncated) |
|---|---:|---|---|
| rlrmp:src/rlrmp/analysis/pipelines/sisu_spectrum_diagnostics.py | 0 (565 flagged) | Migrate off baked-in data/spec-first violations | 21 objects flagged `data_in_code`, `experiment_named_in_src`, `spec_first_violation`. |
| rlrmp:src/rlrmp/train/guided_distillation.py | 0 (399 flagged) | Migrate off baked-in data/spec-first violations | 20 objects, same three flags. |
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py | 0 (607 flagged) | Migrate off baked-in data/spec-first violations | 17 objects flagged `experiment_named_in_src`, `spec_first_violation`. |
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py | 0 (399 flagged) | Migrate off baked-in data/spec-first violations | 15 objects flagged `data_in_code`, `spec_first_violation`. |
| rlrmp:src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py | 0 (207 flagged) | Migrate off baked-in data/spec-first violations | 15 objects flagged `data_in_code`, `experiment_named_in_src`. |
| rlrmp:src/rlrmp/train/closed_loop_distillation.py | 0 (16 flagged) | Migrate off baked-in data/spec-first violations | 14 objects, same two flags. |
| rlrmp:src/rlrmp/train/cs_perturbation_training.py | 0 (874 flagged) | Migrate off baked-in data/spec-first violations | 12 objects flagged `data_in_code`, `experiment_named_in_src`, `spec_first_violation` — the largest flagged-LOC total in this class, on the module the audit was filed to investigate. |
| rlrmp:src/rlrmp/analysis/math/output_feedback.py | 0 (30 flagged) | Migrate off baked-in data/spec-first violations | 10 objects flagged `data_in_code`, `experiment_named_in_src`. |
| rlrmp:results/2ef67ca/scripts/eval_centerout_full.py | 0 (163 flagged) | Migrate off baked-in data/spec-first violations | 8 objects flagged `data_in_code`, `spec_first_violation`. |
| rlrmp:scripts/probe_round_trip_ratio.py | 0 (193 flagged) | Migrate off baked-in data/spec-first violations | 8 objects flagged `data_in_code`, `experiment_named_in_src`. |
| rlrmp:scripts/eval_part2_5_figures.py | 0 (276 flagged) | Migrate off baked-in data/spec-first violations | 7 objects, all three flags. |
| rlrmp:src/rlrmp/train/cs_nominal_gru.py | 0 (795 flagged) | Migrate off baked-in data/spec-first violations | 7 objects, all three flags — the second-largest flagged-LOC total, again on the audit's motivating module. |

`est_loc_delta` is 0 for every class-(e) item because remediation here is
migration/relocation, not deletion — the "LOC delta" column instead reports
flagged LOC (parenthetical) as the impact measure. 98 more items are in
`portfolio.json`'s `e_contract_violation_remediation` array, ranked there by
flagged LOC.

## Class (f): feedbax generalize-in-place — 5 feedbax modules, no size change

**Confidence: reverse-audit judgment** (each individually investigated
against real consumer evidence; not classification-confidence-only). All
feedbax-side. Full list (all 5 items):

| Module | Suggested issue title | What to do |
|---|---|---|
| feedbax/analysis/execution.py | Generalize `feedbax/analysis/execution.py` in place | Keep the module (real feedbax CLI/web/specs consumers); lift the `'train__pert__std'` grouping key and the `sisu` special-case out of literal strings in `setup_eval_for_module` into caller-supplied config/registration. |
| feedbax/component_registry/cde_templates.py | Generalize `feedbax/component_registry/cde_templates.py` in place | Keep template registration as a Studio capability; move the four hardcoded architecture presets ("Anti-NF", "CDE Hybrid v9b") out of core builtins into an examples/demo template pack. |
| feedbax/persistence/database.py | Generalize `feedbax/persistence/database.py` in place | Rename the hardcoded `pert__type`/`pert__std`/`sisu_params` columns to a schema-neutral extensible-metadata pattern (e.g. a generic `condition_metadata` JSON column). Keep the module in feedbax — all real callers are feedbax-internal. |
| feedbax/plot/experiments.py | Generalize `feedbax/plot/experiments.py` in place | Naming-only fix: rename off "experiments.py" (misleadingly suggests project-specific code) to something like `feedbax/plot/analysis_helpers.py`. No consumer migration needed. |
| feedbax/plot/color_setup.py | Generalize `feedbax/plot/color_setup.py` in place | Keep the mechanism (`ColorscaleSpec`/`setup_colors`/etc.) in feedbax; relocate the rlrmp-shaped `COLORSCALES`/`COMMON_COLOR_SPECS` default table to rlrmp as this project's own registered color-fn contribution, fixing the currently-dead `COLOR_FNS` extension point in the process. |

## Class (g): dangling-reference fixes — 89 modules / 278 findings, correctness fix

**Confidence: high/medium-confidence findings only** (278 of the sweep's 1,494
total findings; the rest are documented low-confidence noise excluded per the
sweep's own calibration — see `sweeps/dangling/summary.md`'s heuristic
limitations). No LOC impact — these are broken-reference corrections, not
size changes. Two rlrmp/feedbax items deserve individual issues rather than
batching:

- **`feedbax:feedbax/analysis/support.py`** — the confirmed broken
  `AbstractIntervenor` import (see
  `results/05883e7/notes/feedbax_backlog.md`). Feedbax-side, file separately
  from the batch below.
- **`rlrmp:results/c723082/scripts/run_induced_gain_flavor_b.py`** — 4
  findings mixing `dangling_import_name` (2) and `stale_path_literal` (2);
  worth a look since import-resolution findings are rarer and higher-signal
  than path-literal noise.

The rest cluster almost entirely as **stale test-literal housekeeping**
(top offenders by finding count):

| Scope | Findings | Kind breakdown |
|---|---:|---|
| rlrmp:tests/test_paths.py | 24 | all `stale_path_literal` |
| rlrmp:tests/analysis/pipelines/test_gru_postrun_materialization.py | 16 | all `stale_path_literal` |
| feedbax:tests/test_execution_contract.py | 13 | all `stale_path_literal` |
| feedbax:tests/test_analysis_context.py | 11 | all `stale_path_literal` |
| rlrmp:src/rlrmp/analysis/declarative_materialization.py | 11 | all `stale_path_literal` |
| rlrmp:tests/analysis/pipelines/test_hinf_phenotype_sidecar.py | 9 | all `stale_path_literal` |
| feedbax:tests/test_execution_local_embed.py | 7 | all `stale_path_literal` |
| rlrmp:tests/test_post_run_sh.py | 7 | all `stale_path_literal` |
| feedbax:feedbax/mechanics/skeleton/mjx_skeleton.py | 6 | all `unresolved_import_other` (likely an optional-dependency probe, not a genuine bug — check before filing) |

Plus the separately-tracked **36 stale references to the pre-refactor flat
`scripts/<name>.py` layout** (Bug `8404108`) — mostly stale module docstrings
in files that moved under `results/<hash>/scripts/`; see
`sweeps/dangling/summary.md`'s dedicated table for the full 36-row list. A
single batched issue per repo ("housekeeping: fix stale
`scripts/<name>.py`-layout doc references") is the right shape for this
subset — low individual risk, high mechanical volume.

75 more items are in `portfolio.json`'s `g_dangling_reference_fixes` array.

## `needs_decision` — 602 objects, 18,467 LOC (deliberately unresolved)

Not a remediation class — these are objects Phase 2/4 explicitly left open
because disposition requires human judgment the corpus's own evidence
couldn't settle mechanically (e.g. `induced_gain.py:logger`, a genuinely
unused module logger that is common defensive scaffolding rather than an
obvious delete). This population is the natural backlog for future targeted
review, not a remediation item to file as-is.
