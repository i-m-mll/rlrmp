# Legacy analysis materializers

This note inventories the analysis materializers frozen by [issue:64d5f13].
They are retained for provenance and possible future porting, not as current
patterns for new analysis work. The active Feedbax-backed output-feedback
rollout-recovery spec runner is deliberately outside this list.

## C&S analytical game card

Files: `src/rlrmp/analysis/math/cs_game_card.py` writer surface and
`scripts/materialize_analytical_game_card.py`.

Purpose: This materializer wrote the Phase 0 C&S analytical reference bundle:
the deterministic game card, Riccati/LQR reference objects, and sidecar arrays
used by later bridge checks. The reference math remains central to live
recipes, but the direct tracked-note/NPZ writer predates the Feedbax analysis
recipe, bundle, and manifest contracts.

Used by: [issue:cb98e58] (closed), with downstream open consumers in the bridge
stack.

Keep-signal: Keep as provenance for the canonical reference target and as a
porting source if the report-stage manifest work needs to regenerate the old
artifact shape.

Banner status: Scoped math-module pointer added; `write_outputs` has the
legacy writer banner. The top-level script has the full module banner, while
`materialize_reference` is explicitly left LIVE library code.

## Adversary equivalence

Files: `src/rlrmp/analysis/math/adversary_equivalence.py` writer surface and
`scripts/materialize_adversary_equivalence.py`.

Purpose: This materializer compared the Riccati-implied state-feedback
disturbance with optimized open-loop epsilon sequences for the same analytical
game. It records the Phase 1 analytical decision surface used to interpret what
kind of adversary the game-equivalence bridge was testing.

Used by: [issue:a7dad8a] (closed), with downstream references from the open
linear round-trip and bridge-certificate work.

Keep-signal: Keep as historical evidence for the disturbance-object choice and
as a source for any later report-native explanation of the same comparison.

Banner status: Scoped math-module pointer added; `write_outputs` has the
legacy writer banner. The top-level script has the full module banner.

## Linear round trip

Files: `src/rlrmp/analysis/math/linear_round_trip.py` writer surface and
`scripts/materialize_linear_round_trip.py`.

Purpose: This materializer trained local time-varying full-state gains against
the Phase 0 analytical game and audited them under held-out adversary search.
It is a Phase 3 same-game round-trip artifact writer, not a Feedbax-native
analysis recipe.

Used by: [issue:6f5c79e] (open).

Keep-signal: Keep while the open Phase 3 bridge work still uses the result as
context for what the local linear objective did and did not establish.

Banner status: Scoped math-module pointer added; `write_outputs` has the
legacy writer banner. The top-level script has the full module banner.

## Linear equivalence certificate

Files: `src/rlrmp/analysis/math/linear_equivalence_certificate.py` writer
surface and `scripts/materialize_linear_equivalence_certificate.py`.

Purpose: This materializer wrote the state-weighted certificate for linear
controllers from the Phase 3 round trip. It asks whether trained controllers
match the analytical reference in state/action directions that matter for clean
and disturbance-relevant behavior.

Used by: [issue:d01c35a] (open), consuming [issue:6f5c79e] (open).

Keep-signal: Keep because open output-feedback and recurrent bridge rows still
refer to the standard certificate semantics even though the writer itself is
not contract-native.

Banner status: Scoped math-module pointer added; `write_outputs` has the
legacy writer banner. The top-level script has the full module banner.

## Robust Bellman diagnostics

Files: `src/rlrmp/analysis/math/robust_bellman.py` writer surface and
`scripts/materialize_robust_bellman.py`.

Purpose: This materializer wrote robust Bellman diagnostics for the linear
same-game gate, including deterministic and output-feedback Bellman views. It
produces tracked notes and manifests directly rather than through Feedbax
analysis custody.

Used by: [issue:583d764] (open), with gamma-sweep context from
[issue:97604a8] (open).

Keep-signal: Keep as explanatory bridge evidence until the report-stage system
can either port the diagnostic or make the old writer unnecessary.

Banner status: Scoped math-module pointer added; `write_outputs` has the
legacy writer banner. The top-level script has the full module banner.

## Output-feedback analytical lane

Files: `src/rlrmp/analysis/math/output_feedback.py` writer/driver surfaces and
`scripts/materialize_output_feedback_lane.py`.

Purpose: This materializer wrote the combined Phase 0B, Phase 1, and Phase 3
estimator-in-loop output-feedback lane. The estimator and controller math in
the module remains live, but the phase driver and artifact writer are older
direct materialization surfaces.

Used by: [issue:83fc5b5] (open), with related historical work on
[issue:60d105d] (closed) and [issue:4008843] (closed).

Keep-signal: Keep because it captures the old operative gap between the C&S
released-code information structure and the deterministic full-state bridge.

Banner status: Module pointer added; `analyze_phase0b_output_feedback`,
`analyze_phase1_output_feedback`, `analyze_phase3_output_feedback`, and
`write_outputs` carry scoped legacy banners. The top-level lane script has the
full module banner, and the output-feedback math core is explicitly LIVE.

## Output-feedback gamma sweep

Files: `src/rlrmp/analysis/math/output_feedback.py` gamma-sweep writer and
`scripts/materialize_output_feedback_gamma_sweep.py`.

Purpose: This materializer wrote the gamma-penalized robust feasibility sweep
used to choose a coherent output-feedback target for later bridge work. It is a
certificate-sidecar writer, not a current Feedbax analysis recipe.

Used by: [issue:97604a8] (open).

Keep-signal: Keep as provenance for why later output-feedback rows used their
selected gamma factor.

Banner status: `write_gamma_sweep_outputs` carries a scoped legacy banner in
the math module. The top-level gamma-sweep script has the full module banner.

## C&S stochastic Phase 1

Files: `src/rlrmp/analysis/pipelines/cs_stochastic_phase1.py` and
`scripts/materialize_cs_stochastic_phase1.py`.

Purpose: This materializer evaluated the C&S game card under sampled
released-code-style forward noise for the Phase 1 fidelity correction. It
materializes Monte Carlo summaries, tracked notes, and bulk arrays directly.

Used by: [issue:dd232cd] (open), with deterministic comparator
[issue:a7dad8a] (closed).

Keep-signal: Keep because it records how the released-code stochastic lens was
compared with the deterministic Phase 1 analytical target.

Banner status: Full module banner added to the pipeline and top-level script.

## C&S stochastic Phase 3

Files: `src/rlrmp/analysis/pipelines/cs_stochastic_phase3.py` and
`scripts/materialize_cs_stochastic_phase3.py`.

Purpose: This materializer evaluated Phase 3 controllers under released-code
stochastic rollout-recovery conditions. It also ran process-noise sweep rows
around the output-feedback bridge, but wrote artifacts directly.

Used by: [issue:dd232cd] (open), consuming [issue:7a459bb] (open).

Keep-signal: Keep as the old stochastic comparison layer until equivalent
report-stage or bundle-native materialization exists.

Banner status: Full module banner added to the pipeline and top-level script.

## SISU perturbation comparison

Files: `src/rlrmp/analysis/pipelines/sisu_perturbation_comparison.py`.

Purpose: This materializer compares perturbation-class responses across SISU
levels for trained GRU rows. It has been used from experiment-local scripts to
write comparison JSON and Markdown directly under experiment result folders.

Used by: [issue:e4800d6] (open) and [issue:7c1f7ed] (open, worker status
done).

Keep-signal: Keep because it preserves the SISU-conditioned perturbation
comparison lens used by existing result notes, but do not extend it with new
ad hoc custody.

Banner status: Full module banner added to the pipeline module.

## Bridge aggregation

Files: `src/rlrmp/analysis/pipelines/bridge_aggregation.py`.

Purpose: This materializer reads bridge manifests and writes summary JSON or
Markdown tables for analytical bridge runs. It is useful for aggregating old
bridge-row outputs, but it predates a report-stage product that would own this
summary as a first-class artifact.

Used by: open bridge work including [issue:7a459bb], [issue:d01c35a], and
[issue:c45adde].

Keep-signal: Keep as a compatibility summarizer for existing bridge manifests
until report-stage aggregation replaces it or makes it unnecessary.

Banner status: Full module banner added to the pipeline module.

## Failure decomposition helper and writer

Files: `src/rlrmp/analysis/pipelines/failure_decomposition.py` and
`scripts/materialize_output_feedback_failure_decomposition.py`.

Purpose: This surface explains failed standard-certificate rows by comparing
objectives, gradients, interpolation curves, and visited-state gain errors. The
pipeline helper defines reusable calculations, while the top-level script wrote
the output-feedback companion diagnostic directly.

Used by: [issue:c45adde] (open), consuming [issue:7a459bb] (open) and
[issue:d01c35a] (open).

Keep-signal: Keep because failed-row explanations remain scientifically useful,
but the writer needs a contract-native home before it becomes a pattern.

Banner status: Full module banner added to the helper module and the top-level
failure-decomposition script.

## Delayed diagnostic bundle

Files: `src/rlrmp/analysis/pipelines/delayed_diagnostic_bundle.py`.

Purpose: This dormant materializer builds delayed-reach direction-split and
peak-decay diagnostic bundles. The audit found a registered schema family but
no production caller beyond tests.

Used by: no production issue was found in the audit; the custody allowlist owner
was [issue:c223bb8] (closed).

Revival: [issue:21f4638] covers reviving exactly this bundle.

Keep-signal: Keep only as dormant provenance and schema context unless a future
delayed-reach report explicitly revives it.

Banner status: Full module banner added to the pipeline module.

## Output-feedback affine tracker

Files: `src/rlrmp/analysis/pipelines/output_feedback_affine_tracker.py` and
`scripts/materialize_output_feedback_affine_tracker.py`.

Purpose: This materializer fit affine tracker variants for the output-feedback
bridge and wrote bridge rows, notes, manifests, and arrays directly. It splits
feedforward replay from feedback correction in the older bridge stack.

Used by: [issue:50c260d] (open), with links to [issue:87edaae] (open),
[issue:d01c35a] (open), and [issue:c45adde] (open).

Keep-signal: Keep as evidence for the affine-tracker diagnostic until the
report-stage era decides whether to port or delete it.

Banner status: Full module banner added to the pipeline and top-level script.

Retired 2026-07-09 by [issue:dd8523c] on
`feature/dd8523c-frozen-of-bridge-retirement`. Deletion commit: this branch's
retirement commit; final hash is recorded in the issue closeout. Last-tree
commit carrying the code: `acbdf8d7008a073b3fe9375b5b915288ac05183d`, also
tagged locally as `legacy/output-feedback-materializers-retired`.

Deleted files:
- `src/rlrmp/analysis/pipelines/output_feedback_affine_tracker.py`
- `scripts/materialize_output_feedback_affine_tracker.py`
- `tests/analysis/pipelines/test_output_feedback_affine_tracker.py`

Recovery shape: `git show legacy/output-feedback-materializers-retired:<path>`
or `git show acbdf8d7008a073b3fe9375b5b915288ac05183d:<path>`.

## Output-feedback interpolated starts

Files: `src/rlrmp/analysis/pipelines/output_feedback_interpolated_starts.py`
and `scripts/materialize_output_feedback_interpolated_starts.py`.

Purpose: This materializer probed output-feedback bridge behavior under
interpolated initial conditions derived from rollout-recovery artifacts. It
writes the comparison directly and is not a Feedbax-native runner.

Used by: [issue:7cea1b7] (open), consuming [issue:7a459bb] (open) and
[issue:1c014e5] (open).

Keep-signal: Keep as provenance for the interpolated-starts question and for
later comparison with any native bridge bundle.

Banner status: Full module banner added to the pipeline and top-level script.

## Output-feedback optimizer basin

Files: `scripts/materialize_output_feedback_optimizer_basin_diagnostic.py`.

Purpose: This top-level writer compared optimizer-basin fits for the free
output-feedback bridge using saved rollout-recovery and interpolated-starts
context. It is a historical diagnostic writer rather than a registered recipe.

Used by: [issue:1c014e5] (open), with inputs from [issue:7cea1b7] (open) and
[issue:7a459bb] (open).

Keep-signal: Keep as historical evidence for optimizer-basin interpretation
until the bridge report either ports it or drops it.

Banner status: Full module banner added to the top-level script.

## Output-feedback sweep certificates

Files: `scripts/materialize_output_feedback_sweep_certificates.py`.

Purpose: This writer materialized standard certificate rows and sidecars for
the output-feedback sweep. It remains an active-reference legacy writer for the
old bridge rows, but it is not contract-native.

Used by: [issue:7a459bb] (open), with standard-certificate semantics from
[issue:d01c35a] (open).

Keep-signal: Keep because other legacy output-feedback diagnostics still align
to these certificate rows while math and mode generalization remains unsettled.

Banner status: Full module banner added to the top-level script.

## Output-feedback observer-error coverage

Files: `scripts/materialize_output_feedback_observer_error_coverage.py`.

Purpose: This top-level writer generated an observer-error coverage sweep used
to interpret output-feedback rollout-recovery rows. It feeds notes under the
rollout-recovery issue but is not itself a Feedbax-backed spec runner.

Used by: [issue:3becdec] (open), feeding [issue:7a459bb] (open).

Keep-signal: Keep as historical coverage-sweep evidence for the rollout-recovery
bridge notes.

Banner status: Full module banner added to the top-level script.

## Output-feedback linear recurrent

Files: `src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py` and
`scripts/materialize_output_feedback_linear_recurrent.py`.

Purpose: This materializer fit phase-aware linear recurrent controllers for the
output-feedback bridge and wrote augmented-state bridge rows. It is deliberately
auditable, but its artifact writing predates the Feedbax contracts.

Used by: [issue:5e55f69] (open), with substrate work from [issue:4ded904]
(closed).

Keep-signal: Keep because it records the first linear recurrent bridge attempt
and still informs recurrent certificate interpretation.

Banner status: Full module banner added to the pipeline and top-level script.

Retired 2026-07-09 by [issue:dd8523c] on
`feature/dd8523c-frozen-of-bridge-retirement`. Deletion commit: this branch's
retirement commit; final hash is recorded in the issue closeout. Last-tree
commit carrying the code: `acbdf8d7008a073b3fe9375b5b915288ac05183d`, also
tagged locally as `legacy/output-feedback-materializers-retired`.

Deleted files:
- `src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py`
- `scripts/materialize_output_feedback_linear_recurrent.py`
- `tests/analysis/pipelines/test_output_feedback_linear_recurrent.py`

Recovery shape: `git show legacy/output-feedback-materializers-retired:<path>`
or `git show acbdf8d7008a073b3fe9375b5b915288ac05183d:<path>`.

## Output-feedback phase-modulated recurrent

Files:
`src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py` and
`scripts/materialize_output_feedback_phase_modulated_recurrent.py`.

Purpose: This materializer fit phase-modulated recurrent output-feedback rows,
including oracle, supervised, projection, and reward-control conditions. It
writes the bridge rows directly rather than through a report-native bundle.

Used by: [issue:d6d25d6] (open), with related open rows on [issue:a06307d],
[issue:007087e], and [issue:ad309f5].

Keep-signal: Keep as a provenance source for recurrent bridge comparisons and
for deciding which rows deserve a native report-stage successor.

Banner status: Full module banner added to the pipeline and top-level script.

Retired 2026-07-09 by [issue:dd8523c] on
`feature/dd8523c-frozen-of-bridge-retirement`. Deletion commit: this branch's
retirement commit; final hash is recorded in the issue closeout. Last-tree
commit carrying the code: `acbdf8d7008a073b3fe9375b5b915288ac05183d`, also
tagged locally as `legacy/output-feedback-materializers-retired`.

Deleted files:
- `src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py`
- `scripts/materialize_output_feedback_phase_modulated_recurrent.py`
- `tests/analysis/pipelines/test_output_feedback_phase_modulated_recurrent.py`

Recovery shape: `git show legacy/output-feedback-materializers-retired:<path>`
or `git show acbdf8d7008a073b3fe9375b5b915288ac05183d:<path>`.

## Output-feedback time constrained

Files: `src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py` and
`scripts/materialize_output_feedback_time_constrained.py`.

Purpose: This materializer fit smooth time-basis output-feedback bridge rows
and wrote notes, manifests, coverage sidecars, and arrays directly. It is a
legacy bridge writer that should be ported before being extended.

Used by: [issue:87edaae] (open), consuming [issue:7a459bb] (open) and
[issue:1c014e5] (open).

Keep-signal: Keep as source material for future spec-runner conversion and for
publication-time delete-or-port decisions.

Banner status: Full module banner added to the pipeline and top-level script.

Retired 2026-07-09 by [issue:dd8523c] on
`feature/dd8523c-frozen-of-bridge-retirement`. Deletion commit: this branch's
retirement commit; final hash is recorded in the issue closeout. Last-tree
commit carrying the code: `acbdf8d7008a073b3fe9375b5b915288ac05183d`, also
tagged locally as `legacy/output-feedback-materializers-retired`.

Deleted files:
- `src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py`
- `scripts/materialize_output_feedback_time_constrained.py`
- `tests/analysis/pipelines/test_output_feedback_time_constrained.py`

Recovery shape: `git show legacy/output-feedback-materializers-retired:<path>`
or `git show acbdf8d7008a073b3fe9375b5b915288ac05183d:<path>`.
