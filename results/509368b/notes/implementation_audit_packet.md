# Two-family marginal-cost implementation audit

This packet audits the owner-selected engineering-smoke families under
[issue:509368b]: [issue:2cb6a58] (M1, force-state observability by robust
training) and [issue:4eb51ee] (A1, mixed certificate modes across evaluation
lenses). It separates authored-road conformance, smoke plausibility, and
scientific evidence. The current packet is an audit-by-construction result:
static authoring exposed real road gaps before any invalid matrix, bypass, or
training run was created.

## Verdict

| Claim | M1 | A1 |
|---|---|---|
| Static packet accurately describes the selected family | pass | pass |
| Dependency pin and ordered-lowerer imports | pass | pass |
| Current road can author and execute the selected family | blocked | blocked |
| Local 100-batch smoke plausibility | blocked, not run | blocked, not run |
| Scientific evidence | not established | not established |

No hypothesis is answered. No model was trained, no checkpoint was written, and
no evaluation, analysis, figure, or report manifest was generated. The blocker
finding is itself road-conformance evidence; it is not smoke or neuroscience
evidence.

## Revision and environment snapshot

| Item | Identity |
|---|---|
| Authoring baseline | `bd5292565ea56148734384ef8ee3393dce73832b` |
| M1 packet commit | `4a86e53cb043a483592bf7e0e7c6e938323ee01c` |
| A1 packet commit | `d5f96d8ff6d631abf67cb15579cffb42423ed4f0` |
| Pin integration commit | `6d74a6d81a929f060319dcb1c1582f40e3ae7ee0` |
| Feature merge commit | `03ceacbeb40902bf5620f05cbde9ccd7f9b3ffe0` |
| Protected Feedbax `develop` | `e7f3ef1eb0b631bb475407b900b54c6256135fca` |
| Feedbax feature parent | `fc819ff0cd73c7550fb300b35f3ff8e6159213f9` |
| `uv.lock` SHA-256 before the feature merge | `1c5e08022cd1eb54f32a84c01afb22638d63ee6dada161915a78fbd8b50b45e4` |
| Host | macOS Darwin 25.5.0, arm64, Apple M4 Pro |
| Tool/runtime observed by targeted tests | `uv 0.9.27`; Python 3.13.5; pytest 9.0.3 |
| Execution tier | local only, non-billable |

The Feedbax auth request was [issue:bcf5d8a]/[issue:427cffa] delivery
`d63e7780`, completed successfully with signed merge `e7f3ef1e`. The tracked
`ci/feedbax-ref.toml` and local Feedbax `develop` checkout both resolved to that
merge before dependency-sensitive tests ran. `fc819ff0` is recorded only as the
feature parent, never as the protected dependency identity.

## Frozen row and path identities

### M1: force-state observability

The intended compact matrix is blocked and therefore has no honest content
hash, planned-run ID, or resolved override path. All four identities are frozen
without placeholder digests:

| Row | Force state | Training | Seed | Batches | Matrix/run identity |
|---|---|---|---:|---:|---|
| `force_visible__nominal_seed42_smoke100` | visible | nominal | 42 | 100 | `blocked_not_generated` |
| `force_hidden__nominal_seed42_smoke100` | hidden | nominal | 42 | 100 | `blocked_not_generated` |
| `force_visible__broad_pgd_seed42_smoke100` | visible | broad-epsilon PGD | 42 | 100 | `blocked_not_generated` |
| `force_hidden__broad_pgd_seed42_smoke100` | hidden | broad-epsilon PGD | 42 | 100 | `blocked_not_generated` |

The conceptual axes are `force_filter_feedback` and
`broad_epsilon_pgd_training`. They do not yet have valid executable
`TrainingRunSpec` pointers because the matrix road patches an already-lowered
spec. [issue:5816bf0] owns governed per-row authoring-intent re-lowering.

### A1: mixed certificate modes

Tracked intent: `results/4eb51ee/runs/cohort.intent.json`.

| Row | Architecture | Certificate mode | Training | Seed | Base identity |
|---|---|---|---|---:|---|
| `sg_nominal_s42` | static-gain linear | `static_gain` | nominal | 42 | `a1.static_gain_linear.matched.v1` |
| `sg_robust_s42` | static-gain linear | `static_gain` | broad-epsilon PGD | 42 | `a1.static_gain_linear.matched.v1` |
| `alr_nominal_s42` | linear recurrent | `augmented_linear` | nominal | 42 | `a1.augmented_linear_recurrent.matched.v1` |
| `alr_robust_s42` | linear recurrent | `augmented_linear` | broad-epsilon PGD | 42 | `a1.augmented_linear_recurrent.matched.v1` |
| `gru_nominal_s42` | GRU | `empirical_nonlinear` | nominal | 42 | `a1.gru.matched.v1` |
| `gru_robust_s42` | GRU | `empirical_nonlinear` | broad-epsilon PGD | 42 | `a1.gru.matched.v1` |

The intent contains 24 concrete RFC 6901 pointers covering three architecture
values, three certificate-mode values, six base IDs, six training distributions,
and six seeds. Independent review resolved all 24 to non-null values. Executable
`TrainingRunSpec` override pointers remain explicitly null with status
`blocked_not_resolved` under [issue:427d0d8]; no schema or hash is invented.

The four evaluation-only lenses are `nominal_clean`, `riccati_epsilon`,
`process_noise`, and `held_out_validation`. They are not training axes.

## Tracked file identities

These are revision-pinned Git blob IDs followed by SHA-256 of the committed
bytes.

| File | Git blob | SHA-256 |
|---|---|---|
| `results/2cb6a58/README.md` | `ccb9b4b5be70f5075f641ce558dfaa188789abef` | `eca1fac1a09c6df8cf5fd4f78311cba60eca991677b14c8a8d5b7a9abb113f41` |
| `results/2cb6a58/RUN_PLAN.md` | `f025747d0b895f15267b929231240bb48d5e7fc7` | `f5afba943d61698fec70f7d4b4a2ce97ec94e41280133e18c979289c02f3c927` |
| `results/2cb6a58/notes/static_authoring_gap.md` | `126726e0d2e1a9f5216b65a2a52d94fe4d43e0d5` | `cb0f93f7dd697d8062b121ea752737a517304bcd36b80d7acd929e3eb212ad55` |
| `results/4eb51ee/README.md` | `b8dab54a974ef7dcceafa688620f0fc5f5c096a1` | `1b37bade484ad34036018f06d29b23008f87cbfa4fc39d1c0dc59edfd70396bd` |
| `results/4eb51ee/RUN_PLAN.md` | `5582623f374f6e17cdea77077fed861725d83e9e` | `86d060efe13b9ba6585eaf09c3e1416e33404b1cbe9d85189090b5f6988f5dc0` |
| `results/4eb51ee/runs/cohort.intent.json` | `35633373a61d123c8d7c79d54410a67de39132da` | `8e061879f59d9a12d2d77e9cc6ca3e73dd97effd624cdd2cbc9683b67cfd6ce1` |
| `results/4eb51ee/analysis/cross_lens/spec.json` | `0672c58ad6266333563691c0dd627af31083f7fc` | `230073780755b23e0b5daa37094e642ebca2ba59fd7afd57f6de59d947c98201` |

## Stage evidence

| Stage | M1 | A1 | Evidence or blocker |
|---|---|---|---|
| Static JSON/Markdown authoring | pass | pass | Independent reviewer accepted both packets. |
| Feedbax protected pin | pass | pass | `ci/feedbax-ref.toml` and local `develop` resolve to `e7f3ef1e`. |
| Ordered science lowerers | pass | pass | Targeted lowering tests passed after the pin merge. |
| Executable training matrix | blocked | blocked | [issue:5816bf0] for M1; [issue:427d0d8] for A1 linear rows. |
| Grouped heterogeneous certificate adapter | not required | blocked | Remaining narrow [issue:e6a32b8] delta and robust training-distribution vocabulary. Low-level three-mode components already landed via `7d0a77a0`. |
| Mode-aware report renderer | existing | existing | Landed via [issue:8583faa], commits `3b4d710f` and `7d701a0e`; [issue:9c342ba] reconciliation is non-blocking. |
| KPI input/report tracking | blocked | blocked | [issue:fddd87a]; required JSON filenames are ignored by the role whitelist. |
| Training and checkpoint/resume | blocked | blocked | Platform road blockers above are not integrated. |
| Evaluations and analysis | blocked | blocked | No valid training manifests exist. |
| Figures and reports | blocked | blocked | No analysis manifest exists. |
| Scientific interpretation | not established | not established | One-seed, 100-batch smoke is never scientific evidence. |

## Manifest, artifact, and custody inventory

Every required runtime record is explicit rather than omitted:

| Record | M1 expected | A1 expected | Current ref/hash/custody status |
|---|---:|---:|---|
| Authored matrix/spec | 1 | architecture-specific matrices/bases plus the tracked cohort intent | M1 `blocked_not_generated`; A1 executable bases `blocked_not_resolved` |
| `TrainingRunManifest` | 4 | 6 | `blocked_not_generated`; no custody ref |
| `EvaluationRunManifest` | stock perturbation/feedback stages | 24 (six rows by four lenses) | `blocked_not_generated`; no custody ref |
| `AnalysisRunManifest` | grouped standard outputs | 1 heterogeneous certificate analysis | `blocked_not_generated`; no custody ref |
| `FigureManifest` | response-norm figure(s) | 1 certificate-agreement figure | `blocked_not_generated`; no custody ref |
| Report manifest/render | GRU post-run and certificate reports | 1 certificate report | `blocked_not_generated`; no `report_render` custody ref |
| Checkpoint lineage | four batch-50 to batch-100 resumes | six batch-50 to batch-100 resumes | `not_run`; no transaction or digest |

Absence is a blocker, not `not_applicable`. `not_applicable` is reserved for
structurally invalid certificate components. In A1, GRU global linear transition,
quadratic value, and Bellman-Hessian components are reason-coded
`not_applicable`; augmented-linear rows must use the full plant-plus-recurrent
basis and may not fall back to plant-state static gain.

## Forbidden-surface proof and escape inventory

The independent review compared both child packets with authoring baseline
`bd529256`. The child commits contain only seven files under
`results/2cb6a58/**` and `results/4eb51ee/**`. They contain no changes to:

- `src/rlrmp/train/training_configs.py`;
- `src/rlrmp/train/run_spec_authoring.py`;
- `src/rlrmp/train/config_materialization.py`;
- any compiler, materializer, registry, callback, source, test, or script; or
- `_artifacts`, the shared `.venv`, dependency locks, or pins.

Current escape-hatch count is zero for both children. Explicitly refused pressure:

| Refused escape | Count |
|---|---:|
| Inline or materialized matrix base | 0 |
| Legacy payload mode | 0 |
| Fresh-start or checkpoint-parity skip | 0 |
| Compiled-field patch forest | 0 |
| GRU-to-static certificate coercion | 0 |
| Plant-state fallback for augmented recurrence | 0 |
| Manual manifest normalization/join | 0 |
| Direct durable write | 0 |
| Result-local plot or rollout-rerunning analysis | 0 |
| Forced Git add of ignored KPI records | 0 |

## KPI evidence

The current blocked packets support `c2=c3=c4=c5=0`: the diff has no registry
record, callback, escape invocation, script, or control flow. These counts apply
only to the blocked child packets. Future road work under [issue:5816bf0],
[issue:427d0d8], or [issue:e6a32b8] remains separately attributable and must not
silently preserve a zero cost in the final cross-family verdict.

| KPI | M1 | A1 |
|---|---|---|
| Authored production/spec LOC | unavailable: no valid matrix | pending tool run over the two committed JSON specs |
| c1 | unavailable: no valid matrix | expected `100` distinct JSON keys; independently recounted |
| c2 registry entries | 0 | 0 for current packet |
| c3 callbacks | 0 | 0 |
| c4 escapes | 0 | 0 |
| c5 script control flow | 0 | 0 |
| Revision-pinned report | blocked | blocked |

Canonical `marginal_cost_input.json` drafts exist locally but are ignored by the
current role whitelist and are intentionally not force-added. Their SHA-256 values
at audit time were `885284e0ecddafe41e6abf1cb1316a55d783f8536462fa8559245bbef5b09dce`
(M1) and `8ba030bcb9ce1091623c593539898fca72c7067480a22d26835ed2d8262179df`
(A1). [issue:fddd87a] must make the required input and report normally trackable;
then `scripts/experiment_kpi.py` must run against the final child revision.

## Frozen smoke plausibility and raw measurements

The protocol is one seed (`42`), exactly 100 batches, local-only, with a standard
checkpoint after batch 50 and strict resume to batch 100. The following criteria
were frozen before results:

- all total/per-term losses, states, actions, and endpoint errors are finite;
- final-window median loss is no more than 1.25 times the first-window median;
- M1 records whether any later 10-batch window improves at least 5% and whether
  at least 50% of evaluation trials finish closer to target than they start;
- A1 records a finite nominal-clean median endpoint error no greater than 0.20 m;
- action energy is finite and non-zero, with raw min/median/max/non-zero fraction;
- checkpoint transaction/content digests exist, completed batches are monotonic,
  and resume continues at the predicted next batch without identity drift; and
- every required manifest and custody reference agrees on row, spec, execution,
  architecture, certificate mode, training distribution, and lens.

All raw measurements are currently `not_run`. A future failure does not authorize
extra batches, seeds, tuning, or scientific interpretation.

## Commands and results

Executed from the issue-linked `wt` worktree:

```text
git merge --no-ff --no-edit integration/7f532e3-conformance
scripts/dev_tests.sh tests/test_feedbax_ref_pin.py tests/test_science_lowering.py tests/test_training_spec_storage_adoption.py
```

Result: merge commit `03ceacbe`; `21 passed` in 4.33 seconds. The lowering tests
emitted float64-to-float32 warnings under the default JAX x64-disabled setting;
there were no test failures. No RLRMP full suite ran.

Static review also used `jq empty`, RFC 6901 pointer resolution, `sha256sum`, Git
blob lookup, `git diff --check`, issue reports/links, and forbidden-path diff scans.

Commands intentionally not defined or run yet:

- matrix validation/emission for M1, because no governed matrix can exist before
  [issue:5816bf0];
- architecture-specific A1 matrix emission, because content-pinned linear bases do
  not exist before [issue:427d0d8];
- heterogeneous analysis execution, because the narrow [issue:e6a32b8] adapter
  delta is not integrated; and
- training, checkpoint/resume, evaluation, analysis, figures, reports, KPI reports,
  and the serialized full suite.

The final audit must record exact command argument vectors only after those paths
are governed; inventing commands now would be another bypass.

## Independent reproduction and falsification checklist

1. Resolve every linked issue and verify blocker commits are ancestors of this
   branch before generating matrices or artifacts.
2. Confirm `ci/feedbax-ref.toml`, protected Feedbax `develop`, and the local editable
   checkout all resolve to the same protected SHA.
3. Run `jq empty` on every JSON; resolve all 24 A1 RFC 6901 pointers and require
   non-null results.
4. Recompute committed Git blob IDs and SHA-256 values from the named revisions.
5. Require content-pinned authored bases and real planned run IDs/hashes; reject
   placeholders, inline bases, or historical fixture/runtime inputs.
6. Diff from `bd529256` and require zero experiment-lane edits to forbidden
   compiler/materializer/registry/callback/script surfaces.
7. Reconcile c1 from recursive JSON keys and independently substantiate c2-c5 from
   registry, callback, escape, and control-flow diffs. Attribute platform-gap cost
   to the owning gap issue.
8. Require four M1 and six A1 training manifests. For A1 require 24 evaluation
   manifests, then grouped analysis, figure, and report identities with materialized
   custody refs.
9. Require strict batch-50 checkpoint lineage and batch-100 resume continuity. Fail
   on a fresh restart, duplicated/missing boundary step, or identity drift.
10. Fail A1 if a lens becomes a training axis, a structural absence lacks a reason,
    a GRU is coerced to static gain, or augmented recurrence falls back to plant
    state.
11. Fail road conformance on any direct write, manual join, result-local plot,
    rollout rerun, untracked durable output, or unaccounted registry/callback/escape.
12. Keep the verdicts separate: road conformance, smoke plausibility, and scientific
    evidence. The scientific verdict remains `not_established` for this smoke.
