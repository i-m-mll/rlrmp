# Two-family marginal-cost implementation audit

This packet audits the owner-selected engineering-smoke families under
[issue:509368b]: [issue:2cb6a58] (M1, force-state observability by robust
training) and [issue:4eb51ee] (A1, mixed certificate modes across evaluation
lenses). It separates road conformance, early-training plausibility, and
scientific evidence. Every run described here was local and non-billable.

## Current verdict

| Claim | M1 | A1 |
|---|---|---|
| Frozen authored matrix | pass: four rows | pass: six rows |
| Portable authored-matrix sidecar | pass | pass |
| Local stop-50/resume-100 training | pass for all four rows | **blocked; not started** |
| Downstream evaluation/analysis/figure/report | blocked by partitioned road gaps | **blocked by preflight gaps** |
| Early-training plausibility | finite losses and movement only; incomplete downstream criteria | not observed |
| Scientific evidence | none | none |

No hypothesis is answered. M1 proves that the governed four-row training and
checkpoint lifecycle can execute locally; it does not establish convergence,
robustness, or an effect of force-state observability. A1 has not trained at all.
Its block is deliberate: training before the evaluation and figure producers are
executable would spend compute without producing the frozen acceptance packet.

## Revision and dependency snapshot

The 509 worktree was clean before this documentation-only update.

| Item | Identity |
|---|---|
| Experiment-authoring baseline | `bd5292565ea56148734384ef8ee3393dce73832b` |
| Evidence-correction base head | `5579d28d55f5dacf8b7bb45fadc9d3512e68868e` |
| A1 reproducibility commit / merge | `093fb567` / `45999cc8` |
| Portable-sidecar commit / merge | `46dbf51d` / `07282d2c` |
| Custody-restoration head before downstream integrations | `9dc3c5db` |
| Protected and pinned Feedbax `develop` | `060d65d285969ec11e4a284712913550c462ba18` |
| Exact Feedbax dependency used by accepted M1 runs | `a86f6b8685d5ce6a2761d26a814b65528b9dee1a` |
| Current clean signed Feedbax staging head | `2d657441ba7cdce3e18a7b135592f8bed9b8340c` |
| Implemented staging merges after accepted runs | `6e0352ab` ([issue:7e4cf6b]); `c2932138` ([issue:ca2f937]); `257573ea` ([issue:d81a868]) |
| `uv.lock` SHA-256 | `1c5e08022cd1eb54f32a84c01afb22638d63ee6dada161915a78fbd8b50b45e4` |
| Runtime | Python 3.13.5; local macOS arm64 |
| Execution policy | one seed (`42`), `n_batches=100`, local only, non-billable |

The protected pin remains the published dependency identity. Staging `a86f6b86`
is the exact dependency used for the accepted M1 runs. The live clean signed
staging head has since advanced through the staged-bundle CLI merge `6e0352ab`,
resolved-evaluation-inputs merge `c2932138`, and checkpoint resolver merge
`257573ea`. The current staging head is `2d657441ba7cdce3e18a7b135592f8bed9b8340c`.
Neither staging SHA is represented as the protected pin.

## Frozen authored identities and portable custody

| Family | Matrix SHA-256 and artifact ID | Sidecar URI |
|---|---|---|
| M1 | `547efe4d07e86f941c307a8a95ada987666935742310e2faa19a504cfeb9a1f5` | `repo://results/2cb6a58/runs/matrix.json` |
| A1 | `78108ca2286af701583e5c4eb87a92736820b5c9260129637722c61831a9e52f` | `repo://results/4eb51ee/runs/matrix.json` |

[issue:238eaea] made the tracked A1 authoring document reproduce the frozen
matrix byte-for-byte. [issue:e093cd9] made governed sidecars checkout-independent,
materializable, and hash-verifiable. Public cold emission reproduced both hashes
without hand-normalization. The A1 analysis intent remains
`7ada9db0fc412e9cd19b0e8a77308e7d295151c08cf05ee3fb0c54c02cbf62b6`.

## M1 training and custody evidence

All four local rows reached batch 100 with finite loss after an accepted batch-50
stop and strict resume. Every stop and resume conformance record reports
`overall=pass`. Training loss fell from batch 50 to batch 100 as follows:
`76456.8 -> 23244.4`, `72068.5 -> 22282.5`, `71876.6 -> 28446.3`, and
`73391.7 -> 32210.3`. Every batch-100 model blob hash differs from its exact
batch-50 parent.

| Row | Final run ID | Stop run set | Resume run set | Batch-100 loss | Latest checkpoint ref |
|---|---|---|---|---:|---|
| `force_visible__nominal_seed42_smoke100` | `feedbax-training-run:13ba53f325a05f24be910385774c1872` | `2026-07-13-6bae06ab` | `2026-07-13-b5e80253` | `23244.3765625` | `tx-3868327ebce5417aa8eeb169cb6d2cc8` |
| `force_hidden__nominal_seed42_smoke100` | `feedbax-training-run:97c76892178bd32eadcc8eefb834bfd6` | `2026-07-13-1a170b75` | `2026-07-13-1ac1bcee` | `22282.5359375` | `tx-44f070e42cce42afb488669e95465c84` |
| `force_visible__broad_pgd_seed42_smoke100` | `feedbax-training-run:99ef061bf8b05f8761db7483e75a2512` | `2026-07-13-7afcafb8` | `2026-07-13-bd90c6fc` | `28446.34140625` | `tx-41d81f97bb8447f097f906d0a9f094d9` |
| `force_hidden__broad_pgd_seed42_smoke100` | `feedbax-training-run:6ad196b423dec55afbf1816bc012c76d` | `2026-07-13-3d2417d7` | `2026-07-13-43c9cd35` | `32210.2875` | `tx-4f269ed689754c039cb4c2f4e44a095c` |

The standard custody roots are:

- `_artifacts/orchestration/<run-set>/assembly-request.json`, `bundle.json`,
  `conformance.json`, row events, manifests, registration, and sentinels;
- `_artifacts/2cb6a58/runs/<row>/latest.json`; and
- `_artifacts/2cb6a58/runs/<row>/transactions/<transaction>/manifest.json`
  with content-addressed model, optimizer, PRNG, and completed-batch slots.

For the accepted visible-nominal stop, the batch-50 transaction is
`tx-c29c9b098f364575a970f0f23ba889bf`; its checkpoint manifest SHA-256 is
`fe9eb894513e0d5c680c539a8875924bfcdff53beb3b8ea16faaa6a0294e0443`.

The completed visible-nominal `TrainingRunManifest` is the exact file
`_artifacts/orchestration/2026-07-13-b5e80253/collected/force_visible__nominal_seed42_smoke100/manifest.json`;
its SHA-256 is
`36412dcf4db037094151f506afa9c2c86d24e9fae91b9bcb6e7fb34cefd6ea5a`.

Stop-state wording is exact: the stop `TrainingRunManifest` has
`status=cancelled` and `completed_batches=50`; the registration record has
`status=stopped`, and conformance passes. The stop manifest does not carry a
`stopped=true` fact.

Historical M1 manifests are immutable exact-index inputs, not predicate-selectable
canonical RLRMP run records. Deleting an orchestration run set removes only that run
set; manifest/evidence providers and checkpoint authority remain independently
governed.

The finite final losses and changed model blobs establish engineering-road viability
only. The frozen endpoint,
action-energy, cached-evaluation, and report criteria remain unmeasured, so the M1
plausibility verdict is partial and the scientific verdict remains none. They do not
establish convergence or robustness.

## A1 preflight and exact block

A1 authoring and lowering pass for all six frozen identities:

| Rows | Architecture | Certificate mode | Training distributions |
|---|---|---|---|
| `sg_nominal_s42`, `sg_robust_s42` | static-gain linear | `static_gain` | nominal, broad-epsilon PGD |
| `alr_nominal_s42`, `alr_robust_s42` | linear recurrent | `augmented_linear` | nominal, broad-epsilon PGD |
| `gru_nominal_s42`, `gru_robust_s42` | GRU | `empirical_nonlinear` | nominal, broad-epsilon PGD |

Training remains **BLOCKED** because the acceptance graph is not executable:

1. `analysis/cross_lens/spec.json` is a project intent, not a Feedbax
   `AnalysisBundleSpec` or `FigureSpec`. Its grouped stage currently requests
   `include_bundle_inputs=true`, but `rlrmp.certificate.standard` accepts only
   `EvaluationRunManifest` parents.
2. None of the four named evaluation routes emits the canonical cached
   `standard_certificate_rows` payload. Existing grouped-adapter tests inject that
   payload and therefore do not prove production evaluation.
3. Only the augmented-linear component provider is registered. Static-gain and
   empirical-nonlinear producers are absent, and the augmented route still needs
   governed same-basis reference evidence and verified checkpoint lineage.
4. No executable certificate-agreement `FigureSpec`/template consumes the grouped
   analysis while preserving reason-coded `not_applicable` cells.
5. The mode-aware report renderer is present and custody-routed, but it cannot run
   until a real `AnalysisRunManifest` exists.

Therefore there are no A1 `TrainingRunManifest`, `EvaluationRunManifest`,
`AnalysisRunManifest`, `FigureManifest`, report render, checkpoint, loss, or
plausibility measurements. This absence is `blocked_not_generated`, not a
scientific negative result.

## Exact issue partitions

Completed authoring/custody prerequisites:

- [issue:238eaea] — reproduce the frozen A1 matrix from tracked authoring;
- [issue:e093cd9] — portable authored-matrix sidecars.

Implemented platform prerequisites on current Feedbax staging:

- [issue:7e4cf6b] — staged bundle execution through the analysis CLI, merged as
  `6e0352ab`;
- [issue:ca2f937] — resolved `ParentRef` inputs exposed to registered evaluation
  recipes, merged at `c2932138`.
- [issue:d81a868] — checkpoint-custody `ParentRef` decoding for evaluation,
  merged at `257573ea` (`done`).

A1 downstream partition:

- [issue:0be2b69] — production `standard_certificate_rows` from real evaluation
  manifests (`blocked`);
- [issue:6fa0431] — executable cross-lens bundle and certificate figure
  (`blocked`);
- [issue:0d6c2ae] — governed augmented-state reference evidence (`in_progress`).

M1 downstream partition:

- [issue:deadff5] — carry post-run selectors and RLRMP run identity into native
  training manifests (`in_progress`);
- [issue:37d13e1] — executable GRU post-run manifest graph (`in_progress`);
- [issue:639e30f] — native checkpoint transaction custody in model-driven
  evaluations (`in_progress`);
- [issue:8776106] — typed native training diagnostics in post-run analysis
  (`in_progress`);
- [issue:986a0bf] — exact orchestration-manifest indexing for bundle `ParentRef`s
  (open; worker status `none`);
- [issue:2412353] — preserve conformance and checkpoint custody in mapped run
  evidence.

These are bounded platform units. Their LOC, registrations, callbacks, and control
flow are not charged to the experiment-authoring KPI as if they were authored
inside M1 or A1.

## Manifest and stage inventory

| Stage or record | M1 | A1 |
|---|---|---|
| Governed matrix and portable sidecar | pass | pass |
| Batch-50 checkpoint and strict resume | pass, four rows | not run |
| Native training manifest | present in each accepted M1 run set; downstream selectors/indexing still partitioned | not generated |
| Evaluation manifests | blocked by M1 post-run/evaluation partitions | 24 required; not generated |
| Grouped analysis manifest | not generated | not generated |
| Figure manifest/render | not generated | not generated |
| Custody report render | not generated | not generated |

For A1, `not_applicable` remains valid only for structurally undefined certificate
components, such as a global linear transition/value/Bellman certificate for a GRU.
It must not conceal an absent producer, reference, manifest, or custody object.

## KPI c1-c5 and bypass inventory

The revision-pinned KPI records report:

| KPI | M1 | A1 |
|---|---:|---:|
| Authored production/spec LOC | 107 | 302 |
| Generated matrix LOC | 1 | 1 |
| c1 distinct authored keys | 60 | 127 |
| c2 new registry entries | 0 | 0 |
| c3 authored callbacks | 0 | 0 |
| c4 escape-hatch invocations | 0 | 0 |
| c5 non-boilerplate control flow | 0 | 0 |
| KPI revision | `9866740fb1fd21f12a05e8c7e0219c595b1facfd` | `7f3b503c3f02c0efeea957aa3265ae8c6d1886eb` |

The zero counts describe the experiment-authored surfaces, not the separately
attributed road repairs. Bypass inventory is zero: no inline/materialized base,
legacy payload mode, callback, compiler patch, compiled-field patch forest,
synthetic source checkpoint, alternate executor, parity skip, manual manifest join,
GRU-to-static coercion, plant-state fallback for augmented recurrence, direct durable
write, result-local plot, cloud launch, extra seed, tuning, or extra-batch research
run was accepted.

## Forbidden-diff proof

Baseline `bd529256` is the experiment-authoring comparison point. The experiment
paths are confined to `results/2cb6a58/**`, `results/4eb51ee/**`, and this
`results/509368b/**` audit. The broader branch necessarily contains `src/`,
`scripts/`, tests, pin, and policy changes, but those are merged, issue-linked
bounded repairs rather than hidden experiment-family edits. Review must therefore:

```text
git diff --name-only bd529256..HEAD -- results/2cb6a58 results/4eb51ee results/509368b
git log --first-parent --oneline bd529256..HEAD
git show --stat <owning-repair-commit>
```

The first command establishes experiment ownership; the latter two attribute every
non-result change to its repair issue. A global claim that the branch has no code
changes would be false. Neither family added an experiment-local callback, registry
entry, compiler edit, or private writer.

## Commands and reproducible evidence

The commands below define the exact issue staging checkout without publishing a
user-specific absolute path:

```text
FEEDBAX_STAGING="$HOME/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/integration__509368b-feedbax-staging"
```

M1 governed authoring used the generic matrix emitter:

```text
PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python scripts/emit_training_run_matrix.py \
  results/2cb6a58/runs/matrix.intent.json \
  --output results/2cb6a58/runs/matrix.json
```

A1 used its public heterogeneous authoring entry point, not the generic emitter:

```text
PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python \
  scripts/emit_heterogeneous_training_matrix.py \
  --base-intent results/4eb51ee/runs/base.intent.json \
  --matrix-authoring results/4eb51ee/runs/matrix.authoring.json \
  --issue 4eb51ee \
  --output results/4eb51ee/runs/matrix.json
```

Both families then used:

```text
PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python scripts/launch_training.py validate \
  results/<issue>/runs/matrix.json
PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python scripts/launch_training.py dry-run \
  results/<issue>/runs/matrix.json
```

Each accepted M1 lifecycle used the frozen row with local driver, stop control at
50, then strict resume to 100 against reviewed Feedbax staging `a86f6b86`. The
durable `assembly-request.json`, row `launch-packet.json`, events, conformance,
registration, and checkpoint manifests under the run-set IDs above are the exact
machine-readable command/environment evidence. No A1 execute command was run after
the preflight block. No RLRMP full suite, cloud, push, or protected auth ran in this
lane.

## Independent reproduction and falsification checklist

1. Verify the evidence-correction base head `5579d28d`, protected pin `060d65d`,
   accepted-run staging `a86f6b86`, and current staging `2d657441`; do not conflate
   these identities.
2. Recompute M1/A1 matrix SHA-256 values and resolve each `repo://` sidecar from a
   different checkout; require byte hashes to match.
3. Re-run cold public emission and require exact M1/A1 matrix bytes, not merely
   semantically equivalent JSON.
4. Validate the four M1 and six A1 rows, seed 42, 100 batches, and declared axes.
5. For each accepted M1 stop/resume run set, require `overall=pass`, one batch-50
   cancelled stop, strict continuation to 100, consistent run identity, and
   materializable checkpoint slots. For visible nominal require stop set
   `2026-07-13-6bae06ab`, transaction `tx-c29c9b...`, and manifest SHA
   `fe9eb894...`.
6. Recompute M1 final losses from typed diagnostics; do not infer convergence from
   four finite endpoints.
7. Confirm A1 has no training artifacts and reproduce each of the five downstream
   preflight failures above before authorizing training.
8. Require 24 A1 evaluation manifests with canonical `standard_certificate_rows`,
   then one grouped analysis, figure, and custody report. Reject injected rows.
9. Reconcile KPI inputs against their pinned revisions and independently inspect
   c2-c5 plus the zero-entry bypass inventory.
10. Diff experiment paths from `bd529256`, then attribute every non-result branch
    change to its issue-linked bounded repair.
11. Fail conformance on any fresh restart, manual join, private writer, missing
    reason for structural `not_applicable`, or untracked durable result.
12. Keep verdicts separate. Road conformance can pass while plausibility remains
    partial and scientific evidence remains none.
13. Run the focused packet-integrity test. Require every recorded full SHA-256 to be
    exactly 64 lowercase hexadecimal characters, match each byte-backed lifecycle
    reference against its file, preserve the cancelled-manifest versus stopped-
    registration distinction, and prove each batch-100 model blob differs from its
    exact batch-50 parent.
