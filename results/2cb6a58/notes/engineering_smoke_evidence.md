# M1 local engineering-smoke evidence

Date: 2026-07-13

This packet is engineering evidence only. It does not answer the force-state
observability question and contains no convergence, multi-seed, statistical, or
scientific claim.

## Verdict

- **Paved-road authoring, emission, and stop/resume conformance:** **PASS**. The
  compact content-pinned base, exact four-row matrix, registered lowerers, resolved
  semantics, execution capsule, batch-50 operational stop, strict same-row resume,
  checkpoint custody, lineage, registration, and certificates all pass.
- **Initial-training plausibility:** **PASS, limited to the measured engineering
  screen**. All four batch-50 and batch-100 losses are finite and each batch-100 loss
  is lower than its batch-50 loss. Raw first/last ten-batch windows, endpoint behavior,
  and action-energy criteria are unmeasured; this packet does not infer them.
- **Downstream standard manifests and reports:** **BLOCKED**. Post-run mapping,
  evaluation, analysis, figure, diagnostics, and report paths have not run.
- **Scientific evidence:** **NONE**. One seed and 100 batches cannot answer the
  force-state observability question or support convergence, robustness, statistical,
  or neuroscience conclusions.

No alternate executor, synthetic source checkpoint, fresh-start override, callback,
compiler edit, registry addition, direct durable writer, cloud launch, or extra seed
was used.

## Dependency and environment identity

| Item | Identity |
|---|---|
| RLRMP materializer commit | `b1b28c396fe4651c47157255ddb88e80d23c1ac3` |
| Protected/pinned Feedbax develop | `060d65d285969ec11e4a284712913550c462ba18` |
| exact Feedbax runtime dependency for all accepted M1 runs | `a86f6b8685d5ce6a2761d26a814b65528b9dee1a` |
| current clean signed Feedbax staging head (not used by accepted M1 runs) | `257573ea7642b6570d12afac8a71ee913256e93a` |
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
| portable matrix sidecar SHA-256 | `d4dd25031e8a483d7c735288c068fa3f0db3b8e7919ce1819bedae1bed818f07` |
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

The original absolute-path sidecar was preserved in commit
`a5a24dcc34065918bc11dd5d03c294c2d2269de5` with file SHA-256
`375feeefe15720a0a54a8d1a7fcca82cb1991ec24a5627bbf558362fe8bf6c25`.
Issue `e093cd9` then made the reference portable in commit
`46dbf51d44fea970b93a0cd6beb8010def3ece35` (merged locally as `07282d2c`),
producing the current `repo://results/2cb6a58/runs/matrix.json` sidecar without
changing the matrix bytes or artifact identity.

## Exact row identities

| Row | Planned run ID | Authored payload hash | Lowered execution hash | Feedback | Training |
|---|---|---|---|---:|---|
| `force_visible__nominal_seed42_smoke100` | `feedbax-training-run:13ba53f325a05f24be910385774c1872` | `11c7e36b1d22bc43a5be4464a1c3eb12b6cc3bfd620394f4063325bf02b3c245` | `8f66b87fbe36194c3a2a7e2af983ecdd4ff5533b5f569914c19c43171b94936e` | 6D | nominal |
| `force_hidden__nominal_seed42_smoke100` | `feedbax-training-run:97c76892178bd32eadcc8eefb834bfd6` | `451e80039143c7a2370979bdbb0ddc1fa295aad1b2b46ef6f9429a3faa167bcf` | `2e9dbe0f7ddbcc4a12861a35ff4ae8d0ad83a9814cc0210af3d3652f58b89796` | 4D | nominal |
| `force_visible__broad_pgd_seed42_smoke100` | `feedbax-training-run:99ef061bf8b05f8761db7483e75a2512` | `364a47a1847676f3562064b987c1afa0a61d080fcbfcca69d3710bd12f7e1e4e` | `34aacd8a72762084eea0a33b9b6994d0abf6fa3ac6114aaec53dc1984b1c2c36` | 6D | broad-epsilon PGD |
| `force_hidden__broad_pgd_seed42_smoke100` | `feedbax-training-run:6ad196b423dec55afbf1816bc012c76d` | `88e1607b1ee21d784d2233d8f526bc5cf960895d1a579c2a76e6c73a458ea1bf` | `8959f4d5ff683ce8b01800e5edda362890f9c640a0c060b1a49c34be66bff775` | 4D | broad-epsilon PGD |

Every row has seed 42, 100 batches, and checkpoint interval 50. Nominal rows use
`target_relative`, `objective.partial`, and the registered architecture lowerer;
PGD rows additionally use the registered `broad_epsilon_pgd` lowerer. All lowerer
versions are `v1`.

## Exact lifecycle commands

All lifecycle commands ran serialized from the issue-linked `wt` worktree with the
reviewed staging checkout first on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/src:/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/integration__509368b-feedbax-staging" \
UV_CACHE_DIR="$PWD/.uv-cache" \
FEEDBAX_JAX_COMPILATION_CACHE_DIR="$PWD/.jax-cache" \
uv run --no-sync python scripts/launch_training.py execute \
  results/2cb6a58/runs/matrix.json --row <row-id> \
  --stop-after-batches 50 --driver local

PYTHONPATH="$PWD/src:/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/integration__509368b-feedbax-staging" \
UV_CACHE_DIR="$PWD/.uv-cache" \
FEEDBAX_JAX_COMPILATION_CACHE_DIR="$PWD/.jax-cache" \
uv run --no-sync python scripts/launch_training.py execute \
  results/2cb6a58/runs/matrix.json --row <row-id> --resume --driver local
```

The `<row-id>` values were used exactly in the order shown below. No matrix/spec
coordinate changed between stop and resume.

| Stage | Command summary | Result |
|---|---|---|
| targeted conformance | `scripts/dev_tests.sh tests/test_training_matrix_row_relowering.py tests/test_training_matrix_native_provenance.py tests/test_feedbax_ref_pin.py` | pass: 5 tests |
| KPI/path guards | `scripts/dev_tests.sh tests/test_paths.py::test_experiment_marginal_cost_kpi_record_is_committable tests/test_experiment_kpi_gates.py::test_experiment_kpi_is_revision_pinned_and_deterministic` | pass: 3 tests |
| schema validation | `uv run --no-sync python scripts/launch_training.py validate results/2cb6a58/runs/matrix.intent.json` | pass |
| deterministic planning | `uv run --no-sync python scripts/launch_training.py dry-run results/2cb6a58/runs/matrix.intent.json` | pass: four exact row/run IDs |
| governed emission | `uv run --no-sync python scripts/emit_training_run_matrix.py results/2cb6a58/runs/matrix.intent.json --output results/2cb6a58/runs/matrix.json --custody-root _artifacts/2cb6a58/spec-storage` | pass |
| staging import proof | `uv run --no-sync python -c 'import feedbax; print(feedbax.__file__)'` with staged `PYTHONPATH` | pass: staging checkout |
| staged sidecar emission | same governed emitter, recorded in `_artifacts/2cb6a58/spec-emission-feedbax-cb3f606e.json` | pass; matrix SHA before/after `547efe4d…a1f5` |
| four stop segments | exact command above with `--stop-after-batches 50` | all exit 0; row/registration `stopped`; all eight certificate checks pass |
| four resume segments | exact command above with `--resume` | all exit 0; row/registration `completed`; all eight certificate checks pass |

## Direct lifecycle evidence

Issue `2412353` owns the missing mapped-run evidence packet, so these run-set paths
are cited directly rather than claiming a post-run mapping that did not occur.

| Row | Stop / resume run set | Stop / resume transaction | Checkpoint manifest SHA-256 | Collected `TrainingRunManifest` SHA-256 | Certificate SHA-256 | Loss 50 -> 100 |
|---|---|---|---|---|---|---:|
| visible nominal | `_artifacts/orchestration/2026-07-13-6bae06ab` / `_artifacts/orchestration/2026-07-13-b5e80253` | `tx-c29c9b098f364575a970f0f23ba889bf` / `tx-3868327ebce5417aa8eeb169cb6d2cc8` | `fe9eb894513e0d5c680c539a8875924bfcdff53beb3b8ea16faaa6a0294e0443` / `e8c4843823d98196b9b050f6046787671afa6be460c75490528b9e3e85f52a33` | `91ae385abf88712dd3ddf6c54bf6137d78cf8aeea8265b644d6d147ea3d0b375` / `36412dcf4db037094151f506afa9c2c86d24e9fae91b9bcb6e7fb34cefd6ea5a` | `8b1fe848ef3f5c120a5d615a9a8d035bbb28d1ea9805882607fcf8460a011a50` / `cd538785381ca8bb301cce923295ca89cd0dd2862c5a8f3e863d2ad7dde0b97f` | `76456.77734375 -> 23244.3765625` (-69.60%) |
| hidden nominal | `_artifacts/orchestration/2026-07-13-1a170b75` / `_artifacts/orchestration/2026-07-13-1ac1bcee` | `tx-f1f12722e4394a9388b8b6f586af7956` / `tx-44f070e42cce42afb488669e95465c84` | `58ad2642bd5f9610ab3dec8f150a1f026e7e2ae728decdd3cc75ef60be6046c6` / `92dd062c195fe112d1ba26ac392610302f7738a46ecdca0e2a56fb3ba7752758` | `c60e56e1a40876478496b9a0798633cf5c8287fb7fec2c8a528c8ebb3570e5c8` / `2fe9f046392f66c72ddfdcc2a60ba36e6c0a5784f30952d2ba7360b8f31b843f` | `dcf8fc099e55b0118f1644fb7c83128a11d36087c2bc5fa6f70c427c319bfd19` / `ff86dbdc8e2cb33fd976969c26420079fe20cf7a1524bd81006d9e1978333596` | `72068.46484375 -> 22282.5359375` (-69.08%) |
| visible broad PGD | `_artifacts/orchestration/2026-07-13-7afcafb8` / `_artifacts/orchestration/2026-07-13-bd90c6fc` | `tx-9716697ea7b541f5b6cdd32b01942a4e` / `tx-41d81f97bb8447f097f906d0a9f094d9` | `5db864dfc7555ff85f1c1d68362e1755183348788b03c0a92f42a9a6f9a68e86` / `44d9a65164943d5788c10d480ee3f444b0a0aff5fd4a704e604064f53f09c22c` | `c3ad10f04c35852d9e2785e182b5d21012a9c5838d2094100529b3449fa431c5` / `0c475344e96ae9a902bc98ddcdb2862e01d4b08c6d070cc2325adce8fb5a4d80` | `35892a7c0e701b7d3f6e43127380ae0f20b582eaf84a15698a1a0aefa05e0cbc` / `44513844ef88cc8eb1b4ac8426c0d0eafde68034d846e80190d9fb10012a0889` | `71876.63203125 -> 28446.34140625` (-60.42%) |
| hidden broad PGD | `_artifacts/orchestration/2026-07-13-3d2417d7` / `_artifacts/orchestration/2026-07-13-43c9cd35` | `tx-bdd90762159d41f3ba9249dd294bc5ab` / `tx-4f269ed689754c039cb4c2f4e44a095c` | `b78d7972fcf7046227e03de273921904194a1460bbc7fcfd7b92d1df1e991aed` / `ef6b2d99466e82e941ebb04f555115326370008511f7ce1f24b4d9b961417103` | `28ae38f97777eeea91a55aead108a45ef4cd95b9b31dfe5316dbddb8781d38c1` / `c87b389fcd4e61afc4de761c39c9f02598c70bffd366a3f7aa73cf4b6e3cd503` | `836436bd9b0009a24550d561fc52c5f6eb894a93bee5b36ff26540c2ff1a38eb` / `8674de3d313895250e8efbec5b0e70c46a343e000444eea37df5655860f3f534` | `73391.73828125 -> 32210.2875` (-56.11%) |

For every row, the stop transaction has no parent and segment lineage
`start_batch=0`, `segment_batch_count=50`. The resume transaction names the stop
transaction as `resume_parent` and records `start_batch=50`,
`segment_batch_count=50`. Stop diagnostics record context
`schedule_origin_step/current_step/optimizer_count=0/0/0`; resume diagnostics record
`0/50/50`. Stop LR evidence covers coordinates 0, 50, 99, and 100; resume evidence
covers 50, 99, and 100, all at `0.003000000026077032`. Checkpoint cadence is exactly
one coordinate at 50 per segment. Every run-set state records an empty
`event_discrepancies` list.

## Required products and raw plausibility measurements

| Product or measurement | Status |
|---|---|
| finite checkpoint loss | pass for all four rows; exact values in the lifecycle table |
| first/last/best ten-batch windows | `unmeasured`: retained raw batch history does not prove the frozen window criterion |
| endpoint-distance improvement fraction | `unmeasured`: downstream evaluation is blocked |
| action-energy min/median/max/nonzero fraction | `unmeasured`: downstream evaluation is blocked |
| batch-50 checkpoint transaction | pass for all four rows; exact transaction and manifest hashes above |
| strict resume coordinate and batch-100 completion | pass for all four rows with explicit parent lineage |
| `TrainingRunManifest` | pass: eight collected manifests; exact hashes above |
| `EvaluationRunManifest` | `blocked_not_generated` |
| `AnalysisRunManifest` | `blocked_not_generated` |
| `FigureManifest` and render | `blocked_not_generated` |
| custody-routed report renders | `blocked_not_generated` |

Downstream evaluation, analysis, figure, and report stages were not invoked. All four
training rows completed and registered through their stop/resume lifecycle before the
lane stopped at the separately owned downstream gaps.

## KPI and bypass inventory

`marginal_cost_input.json` classifies the two authored JSON documents and the emitted
matrix separately. The tracked revision-pinned report already exists for revision
`9866740fb1fd21f12a05e8c7e0219c595b1facfd` and records `c1=60`, `c2=0`, `c3=0`,
`c4=0`, and `c5=0`. Those counts are evidence for that exact authoring revision, not
for the current later documentation head. A final-head KPI report still requires
regeneration after the final packet commit if closeout requires the report to name that
new revision.

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
| extra batch, seed, or tuning retry | 0; one seed and the frozen 100-batch budget only |

## Bounded repair stack and remaining gaps

The completed smoke-specific repair stack is independently issue-scoped:

- RLRMP `52bacb3`: governed fresh-matrix execution (`b1d53cb5`, `d627cb0d`);
- RLRMP `ebd5d02`: typed schedule realization (`addb5e0a`, merged `ab9db1e6`);
- RLRMP `0a97038`: canonical LR diagnostics (`e962304e`, merged `feabe55a`);
- RLRMP `c37df92`: completed-batch stop control (`d04f2fe4`, merged `e944a482`);
- RLRMP `d000e9e`: stop-limit conformance handoff (`b3e176d2`, merged `914e4140`);
- RLRMP `1310669`: strict same-row resume (`53efee07`, merged `25471e6c`);
- RLRMP `e4d9cba`: resume diagnostics context (`7aa76c57`, merged `5fa23fa5`);
- RLRMP `a4f114e`: same-row LR evidence (`ce2222d7`, merged `b0ab8bbb`);
- Feedbax `0e257d0`: pipless fingerprint (`438af19b`, staged `cb3f606e`);
- Feedbax `b9ddd04`: canonical seed provenance (`62e69cf7`, staged `4eaf7c71`);
- Feedbax `0fa46bf`: terminal/sentinel reconciliation (`5d32cece`, staged `601b4581`);
- Feedbax `2a1e5e7`: semantic terminal event status (`23f414ef`, staged `ff3cc320`);
- Feedbax `ce10142`: authorized cancelled-batch conformance (`91bfed21`, staged
  `86fc9e17`);
- Feedbax `a7e443e`: stopped registration status (`0916eff6`, staged `d26d72a1`);
- Feedbax `b1e4f95`: signed-zero binding normalization (`aaba898a`, staged
  `3a536b66`); and
- Feedbax `912f861`: same-row resume segment lineage (`bc13e851`, accepted runtime
  staging `a86f6b86`);
- Feedbax `7e4cf6b`: staged analysis-bundle CLI (`3de3e09a`, staged `6e0352ab`);
- Feedbax `ca2f937`: resolved evaluation inputs (`f1171d27`, current live staging
  intermediate merge `c2932138`); and
- Feedbax `d81a868`: checkpoint-custody resolver (`220d13af`, clean signed staging
  merge `257573ea`).

These later Feedbax issues are implemented, but they do not retroactively change the
accepted runtime dependency or constitute the still-missing M1 acceptance products.
Current live blockers are `deadff5` (native post-run identity), `37d13e1` (executable
GRU post-run), `639e30f` (native evaluation checkpoints), `8776106` (typed training
diagnostics), and `986a0bf` (orchestration manifest index). Issue `2412353` owns the
mapped-run evidence gap. None is repaired or bypassed in this documentation pass.

An independent reviewer can reproduce or falsify this packet by:

1. checking RLRMP and Feedbax SHAs plus the pin file;
2. recomputing the base canonical hash and all file/custody SHA-256 values;
3. validating that the matrix contains exactly the four frozen row IDs and seed 42;
4. rerunning the three targeted test files;
5. rerunning `validate` and `dry-run` and comparing all planned IDs;
6. decoding the resolved snapshot and checking 4D/6D feedback, nominal/PGD mode,
   100 batches, interval 50, and the listed lowerers for every row;
7. rerunning the governed emission and comparing the storage identities;
8. rerunning each row's serialized stop command and then its resume command, confirming
   the listed run identity, transaction lineage, contexts, LR samples, and passing
   certificate;
9. confirming all four rows have batch-50 and batch-100 training outputs while all
   downstream evaluation, analysis, figure, and report stages remain absent;
10. comparing the experiment-owned M1 matrix and intent bytes with baseline
    `bd529256`, confirming their hashes remain `547efe4d…a1f5` and
    `68705858…ec45`, and attributing the bounded `src/`, `scripts/`, and `tests/`
    changes to the issue-scoped repair commits enumerated above rather than claiming a
    whole-repository zero diff;
11. verifying the existing KPI report against revision
    `9866740fb1fd21f12a05e8c7e0219c595b1facfd`, and regenerating it after the final
    packet commit only if closeout requires a final-head KPI revision;
12. rejecting any scientific inference from this engineering smoke.
