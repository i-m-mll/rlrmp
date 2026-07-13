# Frozen local engineering-smoke protocol

This protocol is frozen before execution. Every result must be labelled **non-scientific
engineering smoke**. It may establish road conformance and initial plausibility only; it
must not be used to claim that the cross-architecture hypothesis is answered.

## Gate and execution boundary

- Local hardware only; no cloud, pod, Modal, or other billable execution.
- One seed (`42`) and exactly `100` training batches where a row requires training.
- Do not import RLRMP/Feedbax, emit matrices, train, or write custody artifacts until the
  ordered lowerer is on protected Feedbax `develop` and `ci/feedbax-ref.toml` pins that
  resulting SHA.
- Do not execute until issue `427d0d8` supplies content-pinned canonical training bases
  for the static-linear and linear-recurrent paths and the dependency-gated matched GRU
  base is emitted through its existing authoring road. Inline bases, legacy payload modes, fresh-start/parity skips,
  private writes, manual manifest joins, and hand-normalized certificate rows are not
  substitutes.
- Do not execute grouped analysis until a registered manifest-native adapter consumes
  heterogeneous `EvaluationRunManifest` rows without losing architecture, certificate
  mode, nominal-versus-robust distribution, or evaluation lens. Its distribution contract
  must represent robust/broad-epsilon rather than squeezing it into the current narrower
  `BridgeTrainingDistribution` vocabulary.

## Minimum cohort

| Row ID | Architecture | Certificate mode | Training distribution | Batches | Seed |
|---|---|---|---|---:|---:|
| `sg_nominal_s42` | static-gain linear | `static_gain` | nominal | 100 | 42 |
| `sg_robust_s42` | static-gain linear | `static_gain` | broad-epsilon PGD | 100 | 42 |
| `alr_nominal_s42` | linear recurrent | `augmented_linear` | nominal | 100 | 42 |
| `alr_robust_s42` | linear recurrent | `augmented_linear` | broad-epsilon PGD | 100 | 42 |
| `gru_nominal_s42` | GRU | `empirical_nonlinear` | nominal | 100 | 42 |
| `gru_robust_s42` | GRU | `empirical_nonlinear` | broad-epsilon PGD | 100 | 42 |

The plant, task, loss, optimizer, batch size, reach support, disturbance budget, and seed
must be identical except where the architecture or nominal/robust training distribution
requires a declared difference. Evaluation lenses are not training axes.

## Checkpoint and resume

Each trainable row must checkpoint after batch 50, stop cleanly, resume from that
checkpoint, and finish at batch 100. The resumed execution must retain the authored,
resolved, and execution identities and record checkpoint lineage in its
`TrainingRunManifest`. A fresh start after the stop is a failure, not an acceptable smoke
shortcut.

## Evaluation and standard outputs

Each of the six completed rows is evaluated through four distinct cached-evaluation
stages: `nominal_clean`, `riccati_epsilon`, `process_noise`, and `held_out_validation`.
This yields 24 `EvaluationRunManifest` records. The grouped certificate analysis must
consume those manifests without rerunning rollouts and produce an `AnalysisRunManifest`,
declarative `FigureManifest`, and Feedbax-custody `ReportManifest`/report render.

The low-level certificate components landed with `e6a32b8` core (`7d0a77a0`) and remain
native to each row:

- `static_gain`: action, transition, value, and Bellman components are required when the
  canonical static model supplies their inputs.
- `augmented_linear`: action sensitivity, transition, value, and Bellman components use
  the augmented state `[plant_state; recurrent_state]`; plant-state fallback is forbidden.
- `empirical_nonlinear`: action/response evidence is required, while a global linear
  transition, value gap, and Bellman-Hessian residual are explicitly `not_applicable`
  unless a separately governed local-linear certificate is declared.

## Frozen plausibility checks

Record raw values before assigning a pass/fail status.

1. Every logged train and validation loss is finite; every row reaches batch 100.
2. The median total loss over batches 91-100 is no greater than 1.25 times the median over
   batches 1-10. This is a permissive divergence screen, not a convergence claim.
3. The final nominal-clean median endpoint error is finite and no greater than `0.20 m`;
   the final action sequence contains no NaN or infinity.
4. The batch-50 checkpoint is materialized, loadable through the standard resume route,
   and its resumed lineage is present in the training manifest.
5. All required manifests and custody-routed reports exist, materialize, and agree on row,
   spec, execution, architecture, certificate-mode, training-distribution, and lens
   identities.
6. Every structurally invalid component is `not_applicable` with a reason. `missing` is a
   failure when the component is valid and its canonical inputs should exist.

A smoke pass says only that the paved road executed coherently and showed non-catastrophic
early behavior. It is not evidence of convergence, architecture agreement, robustness,
or any neuroscience conclusion.

The custody-routed certificate report renderer is already landed (`8583faa`, commits
`3b4d710f` and `7d701a0e`). Remaining `9c342ba` mixed-mode test/documentation and
stale-open reconciliation are verification/hygiene work, not an execution blocker for
this packet once the grouped adapter emits the standard structured rows.
