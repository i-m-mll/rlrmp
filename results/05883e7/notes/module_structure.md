# Module structure candidates (issue `05883e7`)

This is the curated, human-readable version of
`_artifacts/05883e7/audit/synthesis/module_report.jsonl` /
`module_report.md`: the SPLIT, JOIN, and RELOCATE candidate rankings the
Phase 5A synthesis derived deterministically from the classification corpus
and the module import graph. All three rankings are mechanical scores, not
verdicts -- they surface where the evidence is strongest, not where a change
is guaranteed correct. Treat every entry as "worth a look," file the ones
worth acting on as their own issues, and cross-reference this note.

## How the scores work

- **SPLIT score** = module LOC x purpose-entropy (bits) x (1 + number of
  distinct contract flags present in the module). A module scores high when
  it is both large *and* mixes several different kinds of work (high
  entropy across the 17 `purpose` categories) *and* has multiple kinds of
  contract violation flagged inside it. Restricted to `src/`- and
  `package/`-tree modules with >=3 classified objects.
- **JOIN clusters** group sibling modules (same directory) that share a
  dominant purpose, are both individually small (<=250 LOC), and are
  tightly coupled: a direct import edge between them scores 2 points, plus
  the Jaccard similarity of their fan-in/fan-out neighbor sets. Pairs
  scoring above 0 are merged transitively (union-find), so a cluster can
  have more than 2 members if the coupling chains through several siblings.
- **RELOCATE candidates** are modules where a meaningful fraction of
  classified-object LOC is flagged `misplaced_should_be_results_scripts`,
  `misplaced_should_be_library`, or `experiment_named_in_src` -- i.e. the
  audit's own contract-flag machinery, not a separate heuristic.

## SPLIT candidates (top 15)

| Module | LOC | Objects | Entropy (bits) | Dominant purpose | Contract flags | Score |
|---|---|---|---|---|---|---|
| `src/rlrmp/train/cs_nominal_gru.py` | 8,902 | 254 | 2.27 | spec_manifest_construction | 5 | 121,088 |
| `src/rlrmp/train/cs_perturbation_training.py` | 7,116 | 246 | 1.78 | training_loop | 4 | 63,462 |
| `src/rlrmp/analysis/pipelines/gru_perturbation_bank.py` | 5,245 | 149 | 2.26 | eval_logic | 4 | 59,222 |
| `src/rlrmp/analysis/pipelines/gru_feedback_ablation.py` | 2,709 | 94 | 2.12 | eval_logic | 5 | 34,460 |
| `src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py` | 3,221 | 124 | 2.45 | core_math_algorithm | 3 | 31,554 |
| `src/rlrmp/analysis/math/output_feedback.py` | 2,825 | 64 | 2.30 | core_math_algorithm | 3 | 25,934 |
| `src/rlrmp/train/guided_distillation.py` | 1,604 | 82 | 2.79 | spec_manifest_construction | 4 | 22,399 |
| `src/rlrmp/analysis/declarative_materialization.py` | 3,626 | 137 | 2.08 | analysis_transform | 1 | 15,119 |
| `src/rlrmp/analysis/pipelines/output_feedback_affine_tracker.py` | 2,163 | 53 | 2.17 | core_math_algorithm | 2 | 14,113 |
| `src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py` | 1,334 | 50 | 2.09 | analysis_transform | 3 | 11,130 |
| `src/rlrmp/analysis/pipelines/output_feedback_rollout_recovery.py` | 1,977 | 46 | 1.86 | analysis_transform | 2 | 11,014 |
| `src/rlrmp/train/closed_loop_distillation.py` | 1,214 | 55 | 2.25 | spec_manifest_construction | 3 | 10,920 |
| `src/rlrmp/cloud/modal_runner.py` | 1,193 | 66 | 2.02 | spec_manifest_construction | 3 | 9,616 |
| `src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py` | 1,329 | 36 | 1.72 | core_math_algorithm | 3 | 9,120 |
| `src/rlrmp/analysis/pipelines/gru_pilot_figures.py` | 1,047 | 35 | 2.78 | viz | 2 | 8,732 |

**`cs_nominal_gru.py` tops the list by nearly 2x the runner-up**, exactly the
module the audit was filed to investigate. Its 254 objects span
spec-construction, training-loop, and hp-constant purposes (2.27 bits of
entropy across the corpus's 17-category scheme) and carry 5 distinct
contract flags: `data_in_code` (5 objects, 793 LOC), `experiment_named_in_src`
(5 objects, 358 LOC), `legacy_unbannered` (4 objects, 123 LOC), plus
`spec_first_violation` and `misplaced_should_be_results_scripts` hits. The
practical read: this module is not one thing. A first cut would separate (a)
the reusable training-loop/GRU-construction machinery that belongs in a
capability-named module, from (b) the experiment-named/hyperparameter-baked
functions that the per-object classification corpus already flags as
`delete`/`needs_decision`/`move_to_results_scripts` candidates (see
`notes/synthesis.md`'s remediation table) -- splitting out (b) first would
substantially shrink both the LOC and the entropy driving this score without
requiring a redesign of (a).

`cs_perturbation_training.py` and the four `analysis/pipelines/*` entries
follow a similar pattern: large modules mixing a dominant purpose with
hp/spec-construction and multiple misplacement-shaped contract flags. All
five are reasonable next candidates for the same treatment once
`cs_nominal_gru.py` sets the precedent.

## JOIN candidates (10 clusters)

Sibling modules under the same directory, same dominant purpose, tightly
coupled by the import graph. **Read these as "this directory's internal
factoring is worth reviewing," not as "merge these files."** Several of the
clusters below are *already* well-factored capability modules (the audit
surfacing them at all is a useful sanity check that the coupling-detection
heuristic works, not necessarily a remediation item).

| Cluster | Dominant purpose | Combined LOC | Coupling score |
|---|---|---|---|
| `feedbax/analysis/{dimred,effector,eig,fig_ops,fps,func,network,pca,transforms}.py` | analysis_transform | 1,242 | 10.75 |
| `src/rlrmp/eval/{ensemble,kinematics,pert,sisu}.py` | eval_logic | 312 | 8.59 |
| `feedbax/component_registry/{__init__,domains,meta,templates}.py` | registration_wiring | 436 | 6.78 |
| `feedbax/config/{hyperparams,mapping,namespace,selectors,utils}.py` | spec_manifest_construction | 897 | 6.54 |
| `feedbax/acausal/{base,mechanics,multibody}.py` | model_graph_definition | 407 | 5.60 |
| `feedbax/web/models/{__init__,inspection,statistics,trajectory}.py` | typing_protocols | 324 | 4.67 |
| `feedbax/xabdeef/{contexts,losses,models}.py` | model_graph_definition | 548 | 4.55 |
| `src/rlrmp/train/executor/{adapters,guards,initial_slots}.py` | training_loop | 299 | 3.67 |
| `feedbax/runtime/{channel,filters,graph_channel_adapters,noise,state_indices}.py` | model_graph_definition | 758 | 3.26 |
| `feedbax/runtime/{_graph,iteration,parameter_constraints,state,streaming}.py` | core_math_algorithm | 546 | 2.94 |

Two entries are worth a closer look, cross-referenced against other Phase 3
findings:

- **`src/rlrmp/eval/{ensemble,kinematics,pert,sisu}.py`** is the exact
  reusable-eval-primitive family CLAUDE.md's script-placement policy cites
  as the *correct* outcome of a prior refactor (issue `8404108`) -- these
  four small, tightly-coupled modules (a direct import edge in 3 of 6
  sibling pairs) are evidence the coupling-detection heuristic finds real
  signal, not that they should be merged. Any actual action here would be
  "keep the current factoring, maybe review whether `ensemble.py` should
  re-export the other three's most-used names," not a merge.
- **`feedbax/analysis/{dimred,effector,eig,fig_ops,fps,func,network,pca,
  transforms}.py`** is the largest cluster and overlaps directly with the
  Phase 3 reverse-audit's "orphaned analysis layer" finding
  (`notes/synthesis.md`'s remediation class (b)): `effector.py` and
  `profiles.py`'s dependency chain (`setup.py`) are recommended
  `deprecate_delete` there, while `network.py` is recommended
  `move_to_rlrmp`. A join within this directory should wait until that
  retirement/demotion decision lands -- joining modules that are about to be
  partially deleted would be wasted work.

## RELOCATE candidates (top 15)

| Module | LOC | Flagged LOC | Fraction | Flags | Suggested target |
|---|---|---|---|---|---|
| `src/rlrmp/train/cs_perturbation_training.py` | 7,116 | 813 | 12% | experiment_named_in_src | out of `src/` (experiment-named) |
| `scripts/materialize_output_feedback_sweep_certificates.py` | 1,315 | 636 | 55% | experiment_named_in_src, misplaced_should_be_library, misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `src/rlrmp/analysis/pipelines/sisu_spectrum_diagnostics.py` | 757 | 564 | 85% | experiment_named_in_src | out of `src/` (experiment-named) |
| `scripts/materialize_output_feedback_optimizer_basin_diagnostic.py` | 527 | 444 | 100% | misplaced_should_be_library, misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `scripts/eval_part2_5_figures.py` | 436 | 358 | 100% | experiment_named_in_src, misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `src/rlrmp/train/cs_nominal_gru.py` | 8,902 | 358 | 4% | experiment_named_in_src | out of `src/` (experiment-named) |
| `src/rlrmp/train/guided_distillation.py` | 1,604 | 349 | 24% | experiment_named_in_src | out of `src/` (experiment-named) |
| `scripts/materialize_output_feedback_observer_error_coverage.py` | 410 | 341 | 100% | misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `scripts/eval_part2_5.py` | 382 | 338 | 100% | misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `src/rlrmp/analysis/declarative_materialization.py` | 3,626 | 207 | 6% | experiment_named_in_src | out of `src/` (experiment-named) |
| `results/c92ebd8/scripts/materialize_pgd_ofb_budget_nominal_velocity_profiles.py` | 513 | 200 | 45% | misplaced_should_be_library | promote to `src/rlrmp/<capability>/` |
| `scripts/materialize_output_feedback_failure_decomposition.py` | 868 | 199 | 26% | misplaced_should_be_library, misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `scripts/probe_round_trip_ratio.py` | 290 | 193 | 100% | experiment_named_in_src, misplaced_should_be_results_scripts | `results/<hash>/scripts/` |
| `results/c723082/scripts/run_induced_gain_flavor_b.py` | 713 | 169 | 29% | misplaced_should_be_library | promote to `src/rlrmp/<capability>/` |
| `src/rlrmp/benchmarks/local_parallel.py` | 368 | 164 | 54% | misplaced_should_be_results_scripts | `results/<hash>/scripts/` |

Two directions of misplacement show up, both matching the script-placement
policy in CLAUDE.md:

- **`scripts/materialize_*` and `scripts/eval_part2_5*` modules flagged
  `misplaced_should_be_results_scripts`** (6 of the 15 rows, several at
  100% flagged LOC) are top-level `scripts/` entries that are actually tied
  to one experiment and should have lived under `results/<hash>/scripts/`
  from the start, per the "Script placement: experiment-specific vs
  reusable" rule.
- **Two `results/<hash>/scripts/*` modules flagged
  `misplaced_should_be_library`** run the opposite direction: reusable
  logic that never got promoted to `src/rlrmp/`, per the same policy's
  promotion rule ("when a helper starts being reused across experiments,
  promote it").

Effect if acted on: the four `100%`-flagged rows (basin-diagnostic,
part2.5-figures, observer-error-coverage, part2.5, round-trip-ratio --
1,676 LOC combined) can be relocated wholesale with no residual split
needed; the rest require separating the flagged portion from a larger
module that also holds legitimately-placed code (e.g. only 4-12% of
`cs_nominal_gru.py` and `cs_perturbation_training.py` is
`experiment_named_in_src`-flagged, consistent with the SPLIT-candidate
finding above that these modules mix a legitimate core with baked-in
experiment specifics rather than being wholesale misplaced).

## Drill-down

- `_artifacts/05883e7/audit/synthesis/module_report.jsonl` / `.md` -- full
  per-module stats (all 883 modules) and complete pairwise coupling
  evidence for every JOIN cluster.
- `_artifacts/05883e7/audit/synthesis/portfolio.json` -- remediation items
  (deletions, legacy retirements, dedupe clusters, relocations, contract
  fixes) this note's candidates connect to.
- `results/05883e7/notes/synthesis.md` -- the quantitative synthesis this
  note complements.
- Regenerate with:
  `PYTHONPATH=src uv run --no-sync python results/05883e7/scripts/synthesize.py`
