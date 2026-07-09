# Needs-decision backlog adjudication

Issue: [issue:e3eb3a2]  
Umbrella: [issue:31aaa31]

## Method

This note adjudicates the corrected `needs_decision` population from the
05883e7 code-archaeology audit. The population used here is the post-verification
set produced by `results/05883e7/scripts/synthesize.py`, which applies the
21 verification refutations in
`_artifacts/05883e7/audit/verification/verdicts_merged.jsonl` before counting
dispositions.

Corrected backlog size: 602 objects, 18,467 LOC.

The decision rule for this lane was:

1. Route a batch to an existing issue when an existing issue already owns the
   contract, migration, deletion, dedupe, relocation, or gate.
2. File a new follow-up issue only when no existing issue owns the decision.
3. Leave low-risk keep/defer items unimplemented here unless they reveal a
   concrete work unit.
4. Do not edit sibling-owned code from [issue:6add6b2], [issue:ebb419d],
   [issue:9977ff0], or [issue:e04bd36].

## Adjudication Table

| Batch | Objects | LOC | Decision | Target |
|---|---:|---:|---|---|
| Historical results-script cleanup | 57 | 2,104 | Existing issues own these. Re-verify at current HEAD before moving, deduping, or repairing references. | [issue:b632f57], [issue:f852a72], [issue:ddf86a7], [issue:8d69758] |
| Pipeline consolidation catch-all | 92 | 2,000 | Existing pipeline children own these. This lane does not implement them. | [issue:ebb419d] and children |
| Existing focused RLRMP contract issues / keep until consumer wave | 42 | 1,918 | Route to already-open contract or consumer-wave issues; otherwise keep until the owning consumer wave proves deadness. | [issue:4e95ae5], [issue:6ef623e], [issue:c4416c5], [issue:feedbax/704481b] where applicable |
| Frozen output-feedback bridge modules | 57 | 1,832 | Document-flag-delete or port per the bridge-retirement lane. | [issue:dd8523c], [issue:ebb419d] |
| Historical figure/result custody | 22 | 1,754 | Figure/custody adoption owns the active path; historical scripts convert opportunistically or remain provenance. | [issue:9977ff0], [issue:cf67730], [issue:f852a72] |
| Feedbax package residuals | 60 | 1,478 | Keep as feedbax backlog index items; graduate only concrete adopted items. | [issue:feedbax/1c98826] |
| Training/run-spec surfaces | 41 | 1,342 | Existing run-matrix, config-schema, trainer-retirement, and data-in-code lanes own these. | [issue:6add6b2], [issue:cd137d8], [issue:d58ff2f], [issue:caa3f1b] |
| Results-script relocation and frozen materializers | 70 | 1,263 | Existing relocation and legacy-retirement issues own these. | [issue:b632f57], [issue:e158a74], [issue:dd8523c] |
| Feedbax analysis/plot backlog | 47 | 1,131 | Keep in feedbax backlog index; figure-system, legacy-analysis, or linear-analysis issues take concrete slices. | [issue:feedbax/1c98826], [issue:feedbax/d42eeb3], [issue:feedbax/00f97d5], [issue:feedbax/704481b] |
| Small RLRMP diagnostic/helper fate decision | 21 | 964 | New follow-up. Decide keep/delete/banner for a small uncategorized helper set. | [issue:377bd51] |
| Low-risk keep/defer/no standalone issue | 37 | 649 | No new issue. These are small live/deferred/provenance items or already covered by broader policy gates. | None |
| Pipeline bundle ports | 19 | 598 | Existing C2 pipeline children own these. | [issue:d0189db], [issue:2adfe9e], [issue:56aad38] |
| Feedbax rlrmp-shaped defaults | 6 | 435 | Existing feedbax generalization issue owns these. | [issue:feedbax/1de0cb6] |
| Feedbax Studio/web residue | 15 | 369 | Keep in feedbax backlog index until a concrete Studio/API migration is selected. | [issue:feedbax/1c98826] |
| `scripts/train_minimax.py` legacy entrypoint | 12 | 369 | New follow-up. Decide whole-entrypoint fate after native executor/run-matrix migration. | [issue:10f1b8d] |
| Figure pipeline/custody | 4 | 261 | Existing figure adoption issue owns these. | [issue:9977ff0], [issue:cf67730] |

## New Follow-ups Filed

- [issue:377bd51]: re-verify and decide the fate of 21 objects / 964 LOC across
  `src/rlrmp/lme.py`, `src/rlrmp/viz/loss_viz.py`,
  `src/rlrmp/data_products/calibration.py`,
  `src/rlrmp/model/stochastic_runtime.py`, and
  `src/rlrmp/model/feedbax_graph.py`.
- [issue:10f1b8d]: decide whether `scripts/train_minimax.py` remains an active
  launcher, should be retired, or should be ported onto current run contracts.

Both were linked as children of [issue:31aaa31] and related follow-ups from
[issue:e3eb3a2].

## Existing Routes Linked

[issue:e3eb3a2] now has structured related links to the existing issue routes
receiving adopted batches: [issue:6add6b2], [issue:cd137d8], [issue:d58ff2f],
[issue:caa3f1b], [issue:ebb419d], [issue:dd8523c], [issue:9977ff0],
[issue:b632f57], [issue:e158a74], [issue:ddf86a7], [issue:8d69758],
[issue:feedbax/1c98826], and [issue:feedbax/1de0cb6].

## Non-actions

This lane made no code changes. It also did not implement [issue:e04bd36] or
reclassify verification-confirmed deletion records. Confirmed-dead deletion,
legacy retirement, pipeline porting, figure adoption, run-matrix adoption,
data-in-code enforcement, dangling-reference repair, and dedupe remain owned by
their existing issues.
