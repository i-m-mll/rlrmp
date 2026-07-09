# Synthesis: what is in rlrmp's 115k lines of `src/`? (issue `05883e7`)

This is the Phase 5A synthesis of the code-archaeology audit: a deterministic
crunch over the full Phase 0-4 corpus (14,474 classified objects across
rlrmp's `src/`, `tests/`, `scripts/`, and `results/*/scripts/` trees, and
feedbax's `package/`, `tests/`, `scripts/`, and `examples/` trees), after
applying the 21 corrections Phase 4 verification made to the classification.
It answers the question the audit was filed to answer, gives the headline
remediation arithmetic, and states the caveats a reader needs to weigh the
numbers correctly. The machine-readable version of everything here is under
`_artifacts/05883e7/audit/synthesis/` (`tables.json`, `module_report.jsonl`,
`portfolio.json`), produced by `results/05883e7/scripts/synthesize.py`
(stdlib-only, deterministic, no LLM calls -- rerun it to regenerate these
numbers from the corpus).

## What are the 115k lines?

`src/rlrmp/` is 115,185 lines across 148 modules. Of that, 103,215 lines
(90%) sit inside a classifiable top-level object (function, class, or
module-level constant); the remaining 10% is imports, blank lines, module
docstrings, and comments that don't belong to any single object. Classifying
those 103,215 lines by *purpose* -- what the code is actually for, not what
directory it happens to sit in -- gives the following breakdown:

| Purpose | LOC | % of classified src LOC | What it means |
|---|---|---|---|
| Spec/manifest construction | 20,538 | 19.9% | Building the declarative run/eval/analysis specs the project's spec-first policy requires |
| Analysis transforms | 17,829 | 17.3% | Post-hoc analysis of trained-model behaviour |
| Core math/algorithms | 16,469 | 16.0% | The actual robust-control / RL / linear-systems math |
| Training loops | 10,920 | 10.6% | Driving the optimizer over a model |
| I/O and custody | 7,879 | 7.6% | Reading/writing manifests, checkpoints, artifacts |
| Evaluation logic | 6,469 | 6.3% | Running trained models through rollouts/perturbations |
| CLI/orchestration | 4,720 | 4.6% | Command-line entry points, argument parsing, run drivers |
| Model/graph definitions | 4,211 | 4.1% | Network architectures and feedbax graph wiring |
| **Everything else (9 categories)** | **13,464** | **13.0%** | Hyperparameter/data constants, registration wiring, legacy-compat, docs, viz, launch/resume control, test support, typing, uncategorizable |

(Full 17-category table, plus the same breakdown split by repo x tree, is in
`tables.md`.)

The headline: **rlrmp's `src/` is dominated by spec/manifest construction,
analysis transforms, and core math** (53% of classified LOC combined) --
which is the shape the project's own implementation policy says `src/`
should have. The audit was filed because of specific symptoms (an ~8,900-line
training module, issue-hash-named run planners baked into `src/`), not
because the tree's overall composition looked wrong, and this table confirms
that read: the composition is broadly healthy, with specific, identifiable
pockets of misplacement rather than a systemic problem.

## Post-correction usage status

Applying Phase 4's 21 corrections (verification re-derived usage/disposition
from scratch on a sample and found the original census call wrong 21 times
out of 888 checked -- 2.4%), the usage-status picture across all trees and
both repos is:

| usage_status | LOC | Share of corpus |
|---|---|---|
| `live` | 265,847 | 78.2% |
| `test_only` | 48,954 | 14.4% |
| `legacy_only` | 10,979 | 3.2% |
| `dead` | 5,954 | 1.8% |
| `registry_or_string_referenced` | 4,809 | 1.4% |
| `ambiguous` | 3,223 | 0.9% |
| 6 verification-introduced non-enum labels (see caveat below) | 39 | <0.1% |

The `dead` share (1.8% corpus-wide, 1.4% within rlrmp `src/` specifically --
1,405 of 103,215 LOC) is small in absolute terms, which is reassuring: this
was not a tree full of abandoned code. The more interesting numbers are
`legacy_only` (LEGACY-bannered code awaiting a port-or-delete decision) and
the disposition-level portfolio below, because those identify concrete,
actionable remediation rather than just "some code somewhere is unused."

## Headline remediation arithmetic

These are the numbers a reader should walk away with. Full item-level detail
(one row per module/cluster, with risk notes and suggested issue titles) is
in `_artifacts/05883e7/audit/synthesis/portfolio.json`.

| Remediation class | Items | LOC | Confidence basis |
|---|---|---|---|
| **(a) Confirmed-dead deletions** | 107 modules, 287 objects | **-5,401** | Every `delete`-disposition record was verification-checked (287/287, exhaustive, not a sample); 2 originally-`delete` records were refuted into `keep`, 1 `needs_decision` record was refuted into `delete` |
| **(b) Legacy-tree retirements** | 40 items (33 module clusters + 7 feedbax modules) | **-13,459** | `legacy_only` usage-status clusters (post-correction) plus feedbax's reverse-audit "orphaned analysis/persistence layer" (`feedbax/analysis/{effector,profiles,setup}.py`, `feedbax/bin/db_merge.py`, parts of `feedbax/config/defaults.py`, `feedbax/plot/mpl.py`, `feedbax/web/ws/simulation.py` -- all zero-consumer sqlite-era infrastructure) |
| **(c) Dedupe/promotion (top clusters)** | 79 clusters (≥100 redundant LOC each) | **-15,478** | Out of 978 total near-duplicate clusters totalling ~37,200 redundant LOC corpus-wide; this row is the subset large enough to justify a dedicated remediation item |
| **(d) Move to `results/*/scripts/`** | 10 modules, 79 objects | 1,691 relocated (not deleted) | `move_to_results_scripts`-disposition records, grouped by source module |
| **(e) Contract-violation remediation** | 110 modules, 362 objects | 11,889 relocated/migrated (not deleted) | Objects flagged `data_in_code`, `experiment_named_in_src`, or `spec_first_violation` |
| **(f) Feedbax generalize-in-place** | 5 feedbax modules | n/a (refactor, not size change) | Reverse-audit modules where the fix is lifting rlrmp-shaped defaults into caller-supplied config, not moving the module |
| **(g) Dangling-reference fixes** | 89 modules, 278 findings | n/a (correctness fix) | High/medium-confidence findings only (278 of 1,494 total; the rest are low-confidence and excluded per the sweep's own calibration) |
| **needs_decision (unresolved)** | 602 objects | 18,467 | Deliberately left open by Phase 2/4 -- disposition requires a human call, not a mechanical one |

Two numbers worth calling out specifically:

- **Confirmed-dead deletions (5,401 LOC) is the only number in this table
  that is fully verification-gated** -- every one of the 287 records
  contributing to it was independently re-derived by Phase 4, not just
  sampled. It is the safest LOC figure in this report.
- **37,200 LOC of redundant code across 978 duplication clusters** is the
  largest single number the audit surfaced. Only the ≥100-LOC-redundant
  subset (79 clusters, 15,478 LOC) is broken into portfolio items here;
  the long tail of smaller clusters is real but individually not worth a
  dedicated remediation issue.

## Module structure: where to split, join, relocate

`_artifacts/05883e7/audit/synthesis/module_report.jsonl` (`.md` for the
human-readable ranked version) scores every `src/`- and `package`-tree module
for three remediation shapes. `results/05883e7/notes/module_structure.md` is
the curated version of that ranking; the headline: **`cs_nominal_gru.py`
(8,902 LOC) tops the SPLIT ranking by a wide margin** -- exactly the module
named in the audit's motivating symptoms -- with a purpose entropy of 2.27
bits (it mixes spec-construction, training-loop, and hp-constant code) and 5
distinct contract flags.

## Caveats

- **Classification is single-pass Sonnet, not consensus.** A second
  independent reviewer ("QA") re-classified an 800-record sample from
  scratch; field agreement against the original calls was purpose 74.1%,
  generality 98.6%, **usage_status 83.2%**, disposition 97.0%, confidence
  88.0%. `usage_status` is the field this report leans on most heavily
  (dead/legacy/live counts) and it is also the field with the lowest QA
  agreement and the lowest Cohen's kappa (0.075 overall) -- meaning the
  *category boundaries* for usage_status are the least reliable part of the
  classification scheme, even though the *directional* finding (rlrmp `src/`
  is mostly live, dead code is a small minority) is robust to that noise.
  Full detail: `_artifacts/05883e7/audit/verification/calibration.md`.
- **Verification covered deletion-enabling and select QA-sample populations,
  not everything.** All 287 `delete`-disposition and all 114
  `merge_dedupe`-disposition records were verified exhaustively; only
  235/607 `needs_decision`, 1/2 `replace_with_declarative_surface`, and a
  small QA sample of `keep` (249/13,384) were checked. `move_to_results_scripts`
  (79 records) and the single `move_to_feedbax` record were not verified at
  all. Treat every LOC figure in this report that is *not* gated on the
  `delete`/`merge_dedupe` populations as classification-confidence-only, not
  verification-confirmed.
- **Verification corrections can introduce non-enum labels.** 6 of the 21
  refutations produced `usage_status` values outside `record_schema.md`'s
  fixed enum (`registered_contract`, `dead_but_documented_public_api`,
  `live_public_api`, `dynamic_reference_unconfirmed_wiring`) because
  verification is a free-text override layer over the classification, not a
  second constrained pass. These appear as their own rows in the
  post-correction usage_status tables (39 LOC total, negligible).
- **Xref limitations propagate into every LOC total here.** Per
  `notes/audit_plan.md`: import resolution does not chase star imports,
  `importlib`/`getattr` dynamic access, or a second re-export hop; the
  census does not execute code, so anything reachable only through
  metaprogramming or registry string-dispatch shows up as a `registered`/
  `string_hits` signal rather than a routed inbound reference; and
  `unresolved_name_hits` is a noisy signal for short/generic identifiers.
  Several of the 21 verification refutations were exactly this failure mode
  (e.g. `feedbax/plot/colors.py:adjust_color_brightness`, called via
  `fbp.adjust_color_brightness(...)` through a re-exported package
  namespace, missed by static xref) -- so the true dead-code rate is very
  likely somewhat lower than the classification corpus's raw `dead` count,
  and the residual is concentrated in exactly this class of dynamic-access
  pattern.
- **Corpus quirks handled by `synthesize.py`, documented for transparency:**
  5 duplicate-id lines in `census/objects.jsonl` (module-level name
  rebindings picked up twice by the AST scan, ~9 LOC out of ~275k, first
  occurrence kept); 2 stray whole-module classification records
  (`feedbax/objectives/{service,spec}.py`) that duplicate real per-object
  records for the same files and are excluded from object-level tables;
  inconsistent leading-`feedbax/`-prefix formatting in
  `sweeps/reverse_audit/reverse_audit.jsonl` relpaths, resolved by trying
  both forms; and 31 records (86 LOC total, 0.025% of corpus LOC) where a
  classification record's hand-recorded `loc` differs by 1-2 lines from the
  census's `loc` for the same object (decorator/blank-line boundary
  disagreements, not a structural error).

## Drill-down paths

- `_artifacts/05883e7/audit/synthesis/tables.json` / `tables.md` -- full
  quantitative tables (this note curates a subset).
- `_artifacts/05883e7/audit/synthesis/module_report.jsonl` / `.md` -- per-module
  stats and the SPLIT/JOIN/RELOCATE rankings (`results/05883e7/notes/module_structure.md`
  is the curated version).
- `_artifacts/05883e7/audit/synthesis/portfolio.json` -- every remediation
  item, machine-readable, grouped by class (a)-(g).
- `_artifacts/05883e7/audit/classification/chunk_*.jsonl` -- the underlying
  per-object classification records.
- `_artifacts/05883e7/audit/verification/{verdicts_merged.jsonl,summary.md,calibration.md}` --
  the Phase 4 verification pass and QA calibration this synthesis corrects
  against.
- `_artifacts/05883e7/audit/sweeps/{duplication,dangling,reverse_audit}/` --
  the Phase 3 sweep outputs joined into the portfolio.
- Regenerate all of the above with:
  `PYTHONPATH=src uv run --no-sync python results/05883e7/scripts/synthesize.py`
