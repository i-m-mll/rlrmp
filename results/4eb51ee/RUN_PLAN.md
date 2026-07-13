# Frozen local engineering-smoke protocol

Tracking issue: [issue:4eb51ee]

Status: **training blocked after successful authoring preflight**. The exact
six-row matrix and portable sidecar are ready, but the downstream evaluation and
figure road cannot yet produce the frozen acceptance packet. No A1 training is
authorized while that block remains.

Every eventual result is **non-scientific engineering smoke**. One seed and 100
batches may establish road conformance and weak initial plausibility only. They
cannot answer the cross-architecture certificate question.

## Frozen identities and boundary

- RLRMP authoring branch head at consolidation: `9dc3c5db`.
- Protected/pinned Feedbax: `060d65d285969ec11e4a284712913550c462ba18`.
- Exact accepted-run Feedbax staging: `a86f6b8685d5ce6a2761d26a814b65528b9dee1a`.
- Current clean signed Feedbax staging:
  `257573ea7642b6570d12afac8a71ee913256e93a`, including [issue:7e4cf6b]
  merge `6e0352ab`, [issue:ca2f937] merge `c2932138`, and [issue:d81a868]
  checkpoint-resolver merge `257573ea`.
- Matrix SHA-256: `78108ca2286af701583e5c4eb87a92736820b5c9260129637722c61831a9e52f`.
- Sidecar: `repo://results/4eb51ee/runs/matrix.json`, with the same SHA-256.
- Analysis-intent SHA-256:
  `7ada9db0fc412e9cd19b0e8a77308e7d295151c08cf05ee3fb0c54c02cbf62b6`.

[issue:238eaea] proved that tracked authoring reproduces the frozen matrix bytes.
[issue:e093cd9] made the sidecar checkout-independent and hash-verifiable. Neither
completion authorizes training by itself.

Local hardware only; no cloud, pod, Modal, push, protected auth, extra seed,
expanded batch budget, tuning, or full-suite run is part of this protocol.

## Minimum cohort

| Row ID | Architecture | Certificate mode | Training distribution | Batches | Seed |
|---|---|---|---|---:|---:|
| `sg_nominal_s42` | static-gain linear | `static_gain` | nominal | 100 | 42 |
| `sg_robust_s42` | static-gain linear | `static_gain` | broad-epsilon PGD | 100 | 42 |
| `alr_nominal_s42` | linear recurrent | `augmented_linear` | nominal | 100 | 42 |
| `alr_robust_s42` | linear recurrent | `augmented_linear` | broad-epsilon PGD | 100 | 42 |
| `gru_nominal_s42` | GRU | `empirical_nonlinear` | nominal | 100 | 42 |
| `gru_robust_s42` | GRU | `empirical_nonlinear` | broad-epsilon PGD | 100 | 42 |

Evaluation lenses remain separate from training axes:
`nominal_clean`, `riccati_epsilon`, `process_noise`, and
`held_out_validation`.

## Current preflight block

Training must remain held until all of the following are integrated and reviewed:

1. [issue:0d6c2ae] supplies governed same-basis augmented-state reference action,
   transition, value, and Bellman evidence.
2. [issue:0be2b69] emits canonical `standard_certificate_rows` from real
   `TrainingRunManifest`/`EvaluationRunManifest` lineage for static-gain,
   augmented-linear, and empirical-nonlinear rows.
3. [issue:6fa0431] replaces the descriptive cross-lens intent with an executable
   `AnalysisBundleSpec` and certificate-agreement `FigureSpec`.

[issue:7e4cf6b] staged-bundle CLI execution and [issue:ca2f937] resolved
evaluation inputs are implemented on current staging. [issue:d81a868] is also
implemented and `done` at signed merge `257573ea`; none is an active blocker.
Live worker states are `in_progress` for [issue:0d6c2ae], and `blocked` for
[issue:0be2b69] and [issue:6fa0431].

The grouped certificate adapter and mode-aware report renderer already exist. The
missing pieces are production inputs and executable bundle/figure wiring. Tests that
inject `standard_certificate_rows` prove the consumer, not this experiment road.

The grouped analysis must receive exactly 24 `EvaluationRunManifest` parents and
must not include direct training bundle inputs. Static and empirical producers must
be registered; augmented-linear lineage and reference identities must be governed,
not caller assertions. The figure must preserve architecture, mode, training
distribution, lens, and reason-coded `not_applicable` cells.

## Checkpoint, evaluation, and outputs after the gate opens

Each row must stop after batch 50 through the standard control, materialize a
complete checkpoint transaction, then resume strictly to batch 100 without a fresh
restart or identity drift.

Each completed row then produces four cached evaluations, for 24 total
`EvaluationRunManifest` records. One grouped certificate analysis consumes those
manifests without rerunning rollouts and produces:

- one `AnalysisRunManifest`;
- one declarative certificate-agreement `FigureManifest` and render; and
- one Feedbax-custody bridge-certificate report manifest/render.

`static_gain` and `augmented_linear` components fail closed when their canonical
inputs are absent. `augmented_linear` uses the plant-plus-recurrent basis and never
falls back to plant-state gain. `empirical_nonlinear` may mark global linear
transition/value/Bellman components `not_applicable` only with an explicit structural
reason.

## Frozen plausibility checks

Record raw values before assigning a result:

1. all train/validation losses, states, actions, and endpoint errors are finite;
2. median loss over batches 91-100 is at most 1.25 times batches 1-10;
3. nominal-clean median endpoint error is finite and at most `0.20 m`;
4. action energy is finite and non-zero, with min/median/max/non-zero fraction;
5. checkpoint and resume identities agree at the batch-50 boundary; and
6. all required manifests and custody references agree on row, spec, execution,
   architecture, mode, training distribution, and lens.

A failed check remains visible and does not authorize more compute. No A1 check has
yet been observed, because no A1 training has run.

## Reproduction before release

Verify the protected pin and staging identity separately; reproduce the exact matrix
and portable sidecar through the public heterogeneous emitter:

```text
FEEDBAX_STAGING="$HOME/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/integration__509368b-feedbax-staging"
PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python \
  scripts/emit_heterogeneous_training_matrix.py \
  --base-intent results/4eb51ee/runs/base.intent.json \
  --matrix-authoring results/4eb51ee/runs/matrix.authoring.json \
  --issue 4eb51ee \
  --output results/4eb51ee/runs/matrix.json
```

Then run targeted validation/dry-run;
then prove the four issue partitions above are integrated. Only after those checks
may the lane replay the frozen local stop-50/resume-100 lifecycle. The RLRMP full
suite remains parent-serialized and outside this experiment lane.
