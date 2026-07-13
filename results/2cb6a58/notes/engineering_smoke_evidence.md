# M1 local engineering-smoke evidence

Date: 2026-07-13

This packet is engineering evidence only. It does not answer the force-state
observability question and contains no convergence, multi-seed, statistical, or
scientific claim.

## Verdict

- **Paved-road authoring and emission:** pass. The compact content-pinned base,
  exact four-row matrix, registered science/architecture lowerers, resolved
  semantics, and execution capsule all validate and materialize.
- **Local execution:** blocked before batch 1. The fresh non-fork matrix is rejected
  because `build_orchestration_request` requires a common source checkpoint
  transaction unconditionally.
- **Initial-training plausibility:** not run. No loss, state, action, endpoint,
  checkpoint, or resume measurement exists.
- **Scientific evidence:** none.

No alternate executor, synthetic source checkpoint, fresh-start override, callback,
compiler edit, registry addition, direct durable writer, cloud launch, or extra seed
was used.

## Dependency and environment identity

| Item | Identity |
|---|---|
| RLRMP materializer commit | `bb21a426f46efa9b1867ec398072c82dbbb832c6` |
| Protected/pinned Feedbax develop | `060d65d285969ec11e4a284712913550c462ba18` |
| `uv.lock` SHA-256 | `1c5e08022cd1eb54f32a84c01afb22638d63ee6dada161915a78fbd8b50b45e4` |
| Python | `3.13.5` |
| execution policy | local-only, non-billable; one seed; exactly 100 batches |

## Authored and emitted identities

| Identity | Value |
|---|---|
| compact-base canonical content hash | `912f1fd2b727c04e1e9e5939fca894f73af828fc8f4920bcafdde01b4b4933c6` |
| compact-base file SHA-256 | `b873ead6b2c0bc8f8b7da94ee579b3412a5ad85fe2b1f34cb94b77deea5a77bb` |
| matrix-intent file SHA-256 | `68705858ffe5d0209a1618edea1912844767f898590a651e699995aa7b2eec45` |
| emitted matrix artifact ID | `authored-matrix:sha256:547efe4d07e86f941c307a8a95ada987666935742310e2faa19a504cfeb9a1f5` |
| emitted matrix SHA-256 | `547efe4d07e86f941c307a8a95ada987666935742310e2faa19a504cfeb9a1f5` |
| storage intent hash | `7f0029a9fbd41a646c854df71318324d60002b15bf9ad02908b881ee28066658` |
| authored-envelope hash | `f3eed09ab4ceb028d4f1f35ab0a13e9e51a9cd74178a8e7b5f9791983e4ebaa6` |
| composed-intent hash | `e45086a969b88a40f48b0aeaa1bda8ca55bdf5b8f5d33acc3f6ddef1362a48ab` |
| execution hash | `80ab230146c485851819e444e6ed13d6d3040c6aa87fd227fa13c0c5d1c67ffc` |
| resolved root hash | `82ee93f1644f5d05710a25fbc378b55e17d666e1151b09f1730c2014710754d8` |

Custody references:

- resolved semantics: `artifact://sha256/170fd74494ced87fc23b5e0ee99f43da07eb4d636847bdfb729446ee5597aa41`;
- execution capsule: `artifact://sha256/ee4f4909788fd952d9ece3d67a4789afa55bbb768cf185f385949be18bc3be23`.

Both custody blobs materialize under the issue-owned
`_artifacts/2cb6a58/spec-storage/` root and match their artifact hashes.

## Exact row identities

| Row | Planned run ID | Authored payload hash | Lowered execution hash | Feedback | Training |
|---|---|---|---|---:|---|
| `force_visible__nominal_seed42_smoke100` | `feedbax-training-run:5801214e0619a434ce07e1d238b0070a` | `11c7e36b1d22bc43a5be4464a1c3eb12b6cc3bfd620394f4063325bf02b3c245` | `ee015537b5c6939744538c2bc4d3bb58fa6813ace2f9241754b2a7904f600cb1` | 6D | nominal |
| `force_hidden__nominal_seed42_smoke100` | `feedbax-training-run:0355e55f9812b3a9ac1e2d93aa56d5e6` | `451e80039143c7a2370979bdbb0ddc1fa295aad1b2b46ef6f9429a3faa167bcf` | `34017b97a6234441947d5ebbafe07c3bc66adbf7e4930769a6c6d2fa97b1a586` | 4D | nominal |
| `force_visible__broad_pgd_seed42_smoke100` | `feedbax-training-run:3acfface78cfeb0990b93c2018babb37` | `364a47a1847676f3562064b987c1afa0a61d080fcbfcca69d3710bd12f7e1e4e` | `b026f4247697b92090752deb58a8943923966bbc8eda4b5c0d9e4ca38b7c0745` | 6D | broad-epsilon PGD |
| `force_hidden__broad_pgd_seed42_smoke100` | `feedbax-training-run:9659f397772f599cb1260085b0ac0712` | `88e1607b1ee21d784d2233d8f526bc5cf960895d1a579c2a76e6c73a458ea1bf` | `8628c39ce9b1084edd7a154719a4991c59b335146fc5bd558d93c1728ce98729` | 4D | broad-epsilon PGD |

Every row has seed 42, 100 batches, and checkpoint interval 50. Nominal rows use
`target_relative`, `objective.partial`, and the registered architecture lowerer;
PGD rows additionally use the registered `broad_epsilon_pgd` lowerer. All lowerer
versions are `v1`.

## Commands and stage results

All commands ran from the issue-linked `wt` worktree with
`PYTHONPATH="$PWD/src"` and a worktree-local `UV_CACHE_DIR="$PWD/.uv-cache"`.

| Stage | Command summary | Result |
|---|---|---|
| targeted conformance | `scripts/dev_tests.sh tests/test_training_matrix_row_relowering.py tests/test_training_matrix_native_provenance.py tests/test_feedbax_ref_pin.py` | pass: 5 tests |
| KPI/path guards | `scripts/dev_tests.sh tests/test_paths.py::test_experiment_marginal_cost_kpi_record_is_committable tests/test_experiment_kpi_gates.py::test_experiment_kpi_is_revision_pinned_and_deterministic` | pass: 3 tests |
| schema validation | `uv run --no-sync python scripts/launch_training.py validate results/2cb6a58/runs/matrix.intent.json` | pass |
| deterministic planning | `uv run --no-sync python scripts/launch_training.py dry-run results/2cb6a58/runs/matrix.intent.json` | pass: four exact row/run IDs |
| governed emission | `uv run --no-sync python scripts/emit_training_run_matrix.py results/2cb6a58/runs/matrix.intent.json --output results/2cb6a58/runs/matrix.json --custody-root _artifacts/2cb6a58/spec-storage` | pass |
| first half of first row | `uv run --no-sync python scripts/launch_training.py execute results/2cb6a58/runs/matrix.json --row force_visible__nominal_seed42_smoke100 --stop-after-batches 50 --driver local` | blocked before batch 1 |

The execution command emits the expected fresh-run evidence and then raises:

```text
ValueError: execute requires one common source checkpoint transaction
```

The exception originates at `src/rlrmp/train/launch.py:373`. The matrix has no
`fork` envelope, so inventing source checkpoint metadata would misrepresent a fresh
run as a continuation.

## Required products and raw plausibility measurements

| Product or measurement | Status |
|---|---|
| finite total/per-term loss | `blocked_not_run` |
| first/last/best ten-batch windows | `blocked_not_run` |
| endpoint-distance improvement fraction | `blocked_not_run` |
| action-energy min/median/max/nonzero fraction | `blocked_not_run` |
| batch-50 checkpoint transaction and digests | `blocked_not_generated` |
| strict resume coordinate and batch-100 completion | `blocked_not_run` |
| `TrainingRunManifest` | `blocked_not_generated` |
| `EvaluationRunManifest` | `blocked_not_generated` |
| `AnalysisRunManifest` | `blocked_not_generated` |
| `FigureManifest` and render | `blocked_not_generated` |
| custody-routed report renders | `blocked_not_generated` |

Downstream evaluation, analysis, figure, and report stages were not invoked because
there is no trained checkpoint or `TrainingRunManifest` to consume.

## KPI and bypass inventory

The normally trackable `marginal_cost_input.json` classifies the two authored JSON
documents and the emitted matrix separately. The revision-pinned KPI report remains
pending the final child commit. Current explicit counts are `c2=0`, `c3=0`, `c4=0`,
and `c5=0`.

| Bypass or pressure | Count |
|---|---:|
| compiler/materializer edit | 0 |
| new registry entry | 0 |
| callback | 0 |
| compiled-field patch forest | 0 |
| inline/materialized base | 0 |
| synthetic source checkpoint | 0 |
| alternate/direct training executor | 0 |
| fresh-start/parity override | 0 |
| direct durable write | 0 |
| cloud/pod/Modal launch | 0 |
| extra batch, seed, or tuning retry | 0 |

## Gap and reproduction checklist

The duplicate search found no existing issue covering fresh non-fork matrix
execution. [issue:52bacb3], **Allow fresh governed training matrices without source
checkpoint transactions**, now owns the correction and structurally blocks this
experiment and the sibling A1 smoke; it is also related to [issue:509368b].

An independent reviewer can reproduce or falsify this packet by:

1. checking RLRMP and Feedbax SHAs plus the pin file;
2. recomputing the base canonical hash and all file/custody SHA-256 values;
3. validating that the matrix contains exactly the four frozen row IDs and seed 42;
4. rerunning the three targeted test files;
5. rerunning `validate` and `dry-run` and comparing all planned IDs;
6. decoding the resolved snapshot and checking 4D/6D feedback, nominal/PGD mode,
   100 batches, interval 50, and the listed lowerers for every row;
7. rerunning the governed emission and comparing the storage identities;
8. rerunning the single local execute command and confirming failure before any
   batch/checkpoint/manifest write at the stated precondition;
9. confirming no child-owned training artifact directory contains batch output;
10. diffing against baseline `bd529256` and verifying no `src/`, `scripts/`,
    `tests/`, compiler, registry, dependency, or shared-environment file changed;
11. generating the KPI report only from the final committed revision; and
12. rejecting any scientific inference from this blocked engineering smoke.
