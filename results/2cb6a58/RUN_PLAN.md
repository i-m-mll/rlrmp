# Frozen local engineering-smoke plan

Tracking issue: `2cb6a58`

Status: **authored, emitted, preflighted, and provisioned; blocked before batch 1 by local environment fingerprinting**.
The protocol remains frozen and every outcome is non-scientific engineering-smoke
evidence; no hypothesis is answered by this protocol.

## Intended rows

| Row identity | Force/filter state visible | Training regime | Seed | Batches |
|---|---:|---|---:|---:|
| `force_visible__nominal_seed42_smoke100` | yes | nominal | 42 | 100 |
| `force_hidden__nominal_seed42_smoke100` | no | nominal | 42 | 100 |
| `force_visible__broad_pgd_seed42_smoke100` | yes | broad-epsilon PGD | 42 | 100 |
| `force_hidden__broad_pgd_seed42_smoke100` | no | broad-epsilon PGD | 42 | 100 |

The matrix now exists at `runs/matrix.json`, backed by compact authored intent and a
Feedbax resolved-semantics snapshot plus execution capsule. Exact row and content
identities are recorded in `notes/engineering_smoke_evidence.md`. No training-run
manifest exists because governed execution fails before batch 1.

The KPI input is now normally trackable and classifies the compact authored base and
matrix intent separately from the generated matrix materialization. Its report must
be generated against the final child commit so the revision is immutable.

## Fixed authoring inputs

The only scientific variation points are:

1. `force_filter_feedback`: `true` or `false`; and
2. `broad_epsilon_pgd_training`: `false` or `true`.

All rows must otherwise share the current canonical C&S GRU training choices:

- target-relative multitarget task with `const_band16` support;
- fixed simple reach, 15 cm target, C&S LSS plant, and five-step feedback delay;
- GRU hidden size 180 with the initial-hidden-state encoder enabled;
- batch size 64, AdamW learning rate `3e-3`, global gradient clip 5;
- stochastic preset `cs2019-rollout`, training diagnostics enabled;
- one seed, 42, and exactly 100 total batches; and
- local execution only, with cloud, RunPod, Modal, and billable compute forbidden.

The PGD cells use the registered direct full-state epsilon mechanism with the
standard moderate budget, hard-L2 objective, projected-gradient ascent, three inner
steps, and step-size fraction 0.25. No callback, registry entry, compiler edit,
manual materializer, inline base, fresh-start override, or parity skip is permitted.

## Checkpoint and resume protocol

The eventual authored run contract must checkpoint at batch 50. Each row runs to
the first checkpoint, stops through the standard operational stop control, verifies
that the checkpoint transaction is complete, and resumes through the standard
resume path to batch 100. Evidence must include the checkpoint transaction ID,
content digests, completed-batch coordinate before and after resume, and a loss
history with no missing or duplicated boundary step.

## Plausibility criteria frozen before results

These checks are deliberately weak engineering checks, not convergence criteria:

- every total and per-term training loss value is finite;
- the median loss over batches 91-100 is no more than 1.25 times the median over
  batches 1-10, and at least one later 10-batch window improves by 5% relative to
  the first window;
- evaluation states, actions, and endpoint errors are finite;
- at least 50% of smoke evaluation trials end closer to the target than they start;
- action energy is finite and nonzero, with the raw minimum, median, maximum, and
  nonzero fraction reported; and
- the resume boundary preserves completed-batch monotonicity and produces the same
  next-batch coordinate as strict resume verification predicts.

Record the raw first/last-window losses, best window, endpoint-distance improvement
fraction, action-energy summary, and checkpoint coordinates for every row. A row may
fail these checks without invalidating road conformance; the failure must remain
visible and must not trigger extra batches, seeds, tuning, or a scientific claim.

## Standard post-run route

Every row must register a `TrainingRunManifest` with metadata selecting the existing
`rlrmp/training_diagnostics` and `rlrmp/gru_postrun` bundles. The grouped evaluation
must reuse cached rollout states and produce `EvaluationRunManifest` records for the
stock perturbation bank and feedback ablation. Downstream execution must produce:

- grouped `AnalysisRunManifest` records for the standard certificate, perturbation
  aggregation, feedback ablation, and response-norm comparison;
- a declarative response-norm `FigureManifest` and its custody-routed HTML render;
- the stock `rlrmp.report.gru_postrun_summary` and
  `rlrmp.report.bridge_certificate_notes` report renders in Feedbax custody; and
- a tracked note only as a downstream export of the report artifact.

The bundle and report recipes are existing registered surfaces; this experiment adds
no callback or registry entry. Artifact references, manifest IDs, hashes, and custody
materialization checks belong in the final audit packet. No direct durable write may
substitute for a required manifest or `report_render` artifact.

## Gate before execution can resume

Protected Feedbax `060d65d285969ec11e4a284712913550c462ba18` is pinned, the
pin guard passes, and the governed row re-lowering, fresh-matrix, and typed-optimizer
routes are integrated. Validation, lowering, dry-run identity generation, custody
emission, assembly, preflight, and provisioning pass. Execution can resume only
after the Feedbax local driver fingerprints a uv-managed interpreter without
assuming that `pip` is importable, while preserving deterministic provenance and
fail-closed behavior. [issue:feedbax/0e257d0] owns that correction; [issue:ebd5d02]
and [issue:52bacb3] are repaired predecessors.

The full RLRMP suite remains parent-serialized and is forbidden in this child lane.
