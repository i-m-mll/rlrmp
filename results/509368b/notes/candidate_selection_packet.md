# Candidate selection packet

Issue `509368b` tests whether the post-consolidation experiment road can carry two
genuinely new families—one median-complexity and one deliberately awkward—at a
marginal cost of a few hundred authored lines, without compiler edits. This packet
freezes the measurement before authoring and leaves scientific priority with the
project owner.

## Frozen acceptance measurements

The selected families will each receive their own experiment issue and therefore
their own `results/<issue>/notes/marginal_cost_input.json` and revision-pinned
`marginal_cost.json`. Issue `509368b` will carry only the cross-family verdict. This
keeps the KPI's required unit—one record per experiment issue—unambiguous.

For each selected family, freeze the following before its first authored commit:

- the content-pinned authored base, matrix schema version, expected row identities,
  variation paths, and expected composed training-mode/lowerer identities;
- `authored_production_paths` and its `authored_spec_paths` subset; the KPI counts the
  complete immutable Git blobs at the recorded revision, not a hand-estimated diff;
- c1 as distinct authored JSON keys plus explicit concepts for non-JSON formats;
  c2 as new registry records; c3 as newly authored framework callbacks; c4 as
  explicit escape-hatch invocations with a reason; and c5 as non-boilerplate
  experiment-script control flow;
- generated expanded rows separately as materialization LOC, never authored LOC;
- zero edits to
  `src/rlrmp/train/{training_configs,run_spec_authoring,config_materialization}.py`,
  and zero RLRMP or Feedbax compiler/materializer edits;
- standard identity and custody outputs from matrix emission, then
  `TrainingRunManifest`, cached `EvaluationRunManifest`, grouped
  `AnalysisRunManifest`, declarative `FigureManifest`, and custody-routed report
  renders/JSON summaries. Any tracked note is a downstream export, not a substitute
  for the report artifact.

Governed rows, axes, deltas, and override patches are declarative authoring, not c4
or c5. Inline matrix bases, fresh-start/parity skips, legacy payload modes, manual
joins, direct durable writes, result-local plotting, or analyses that rerun rollouts
are escape pressure and must not be hidden. A forced bypass or compiler edit becomes
a defect against the owning surface rather than an unreported exception.

Non-billable preflight may prove schema validity, content identity, materialization,
row composition, custody routing, and standard bundle selection. It cannot establish
a neuroscience result. No training or billable compute is authorized by this packet;
any later launch needs the usual owner-confirmed run-spec table.

## Median-complexity candidates

| ID | Candidate and scientific question | Required variation points | Callback and standard evidence | Useful evidence threshold |
|---|---|---|---|---|
| M1 | **Force-state observability × robust training.** Does the robust feedback phenotype require direct force/filter-state observation, or can a GRU reconstruct it from delayed position and velocity? | Force/filter feedback present vs absent × nominal vs broad-epsilon PGD; match plant, task, budget, optimizer, and seeds. | Likely no new callback. Standard training diagnostics, feedback ablation, recurrent Jacobian, perturbation bank, certificate, response-norm figure, and GRU post-run report. | Authorability is non-billable; all causal effect estimates require new training. Expected cost: about 120–220 spec LOC, c2–c5 all zero. |
| M2 | **Angular support-gap × robust training.** Does robustification improve interpolation into unseen reach directions, or only harden trained directions? | Narrow vs wide held-out angular bands × nominal vs PGD, with matched support metadata and seeds. | Existing seen/held-out summaries suffice for the aggregate claim; an optional registered angle-to-boundary analysis would be about 80–140 callback LOC. Standard center-out, perturbation, certificate, response-norm, and report outputs. | Support/materialization proof is non-billable; the generalization interaction requires training. Expected cost: about 140–230 spec LOC without the optional callback. |
| M3 | **Segregated populations × robust training.** Can robustness emerge when sensing, recurrence, and readout are assigned to distinct populations rather than overlapping units? | Overlapping vs input-only/recurrent-only/readout-only partition × nominal vs PGD; match total width and report parameter-count differences. | Standard recurrent-Jacobian, policy, perturbation, certificate, loss, response-norm, and report outputs. A subgroup-block mechanism analysis would add roughly 100–160 callback LOC and is optional. | Topology and routing are non-billable; competence, robustness, and mechanism claims require training. Expected cost: about 150–250 spec LOC without the optional callback. |
| M4 | **Catch-hazard prior × robust training.** Under explicitly normalized catch/no-catch loss, does robust training change how the controller uses catch prevalence as an environmental prior? | Low vs high catch probability × nominal vs PGD, with normalized loss and fixed delayed-reach contract. | Likely no callback: delayed-reach bank, catch/no-catch velocity, hold drift, perturbation, certificate, and report outputs. | Semantics are inspectable non-billably; the hazard-prior claim requires training. Expected cost: about 130–230 spec LOC. Known delayed-reach mismatch makes this the riskiest median option. |

M1 is the cleanest callback-free test. M2 has the cleanest geometric-generalization
interpretation. These are coordinator recommendations only, not selections.

## Deliberately awkward falsifier candidates

| ID | Candidate and scientific question | Required variation points | Callback and standard evidence | Useful evidence threshold |
|---|---|---|---|---|
| A1 | **Mixed certificate-mode, cross-lens cohort.** Do robust-control conclusions agree across static-gain linear, augmented-linear recurrent, and nonlinear GRU controllers, or depend on architecture/certificate mode? | Architecture/certificate mode × nominal vs robust training; evaluate nominal-clean, Riccati-epsilon, process-noise, and held-out lenses without treating lenses as training axes. | Prefer stock certificate, perturbation, phenotype, figure, and report recipes with explicit `not_applicable` cells. At most one 40–70 LOC registered concordance analysis. | Composition can be demonstrated non-billably; a controlled matched cohort requires training. Expected cost: about 180–300 spec LOC, c4=0. This most directly tests semantics the road already promises. |
| A2 | **Signed-transfer wedge.** Does robustness learned against one disturbance sign/target sector transfer to the mirrored sign and held-out sectors, or memorize experienced geometry? | Clockwise-only, counter-clockwise-only, balanced, and nominal training × same-sign, opposite-sign, and held-out-sector evaluation. | Likely one 50–90 LOC registered grouped sign-by-sector contrast; standard perturbation evaluations, heatmap/bar and shared-y profiles, and report outputs. | Authoring is non-billable; causal transfer needs asymmetric training. Expected cost: about 140–230 spec LOC plus the callback, c4=0. |
| A3 | **Feedback contingency × perturbation timing.** Is a robustness benefit mediated by a specific feedback channel, and only for early versus late disturbances? | Nominal vs robust training × intact/position/velocity/force feedback × early/mid/late perturbations at calibrated severity. | One 60–100 LOC grouped interaction analysis over cached manifests; standard ablation and perturbation evaluations, response/profile figures, and report. Conditional roles must be declarative. | Existing checkpoints may yield useful non-billable evaluation evidence; new feedback-corrupted training claims require confirmation. Expected cost: about 170–260 spec LOC plus callback, c4=0. |
| A4 | **Prespecified sparse evidence ladder.** Are conclusions stable when cheap lenses cover all rows while expensive exact/certificate evidence is scheduled only for design-eligible strata? | Method/architecture × mandatory nominal/perturbation lenses × predeclared, outcome-independent eligibility for worst-case and exact audits. | Prefer conditional stock stages and reason-coded absent cells; at most one 30–60 LOC evidence-coverage audit. | Existing checkpoints can provide useful non-billable evidence. It qualifies as the awkward family only if the owner accepts an analysis/evaluation family as satisfying “new experiment family”; otherwise choose A1–A3. |

A1 is the strongest falsifier of the existing road. A2 is the sharper asymmetric
scientific experiment. A3 is the strongest conditional-composition stress. These
rankings do not choose the owner's scientific priority.

## Preconditions and owner decision

Implementation verification must first see the ordered-lowerer Feedbax API on the
published/pinned dependency path. The current RLRMP branch imports that API, while
`ci/feedbax-ref.toml` still points to its parent commit. Do not work around this in
RLRMP. Also avoid making calibrated composite-mode labels an acceptance dependency
until the capability-owned perturbation lowerer preserves the calibrated identity.

The owner must now choose exactly one median ID (`M1`–`M4`) and one awkward ID
(`A1`–`A4`). For any candidate with an optional callback, the decision must also say
whether to keep the first implementation callback-free or authorize the named small
registered analysis. Selecting A4 must explicitly confirm that an analysis/evaluation
family satisfies the issue's “new experiment family” requirement. Selection authorizes
authoring and non-billable validation only; it does not authorize training or billable
compute.
