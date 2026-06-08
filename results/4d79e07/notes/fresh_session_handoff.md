# Fresh-Session Handoff: lr=3e-3 PGD Broad-Epsilon Replication

Issue: [issue:4d79e07].

This handoff is for continuing the PGD broad-epsilon robustness path in a fresh
session without reconstructing the recent conversation. It is specifically about
expanding the promising lr=3e-3 PGD result, not about the teacher/distillation
path and not about proving PGD is identical to the Riccati feedback adversary.

## Current State

- `main` includes the recent integration branch `integration/recent-robustness-diagnostics`
  via merge commit `6c5c38a`.
- That integration brought in:
  - feedback-control quality lens documentation from [issue:abe33da];
  - PGD broad-epsilon training/diagnostic machinery and run records under
    [issue:020a65b] and [issue:b8aa38e];
  - worst-case epsilon audit, broad-epsilon attribution, feedback ablation,
    perturbation-response bank, objective-comparator, standard-certificate, and
    H-infinity phenotype sidecar materials;
  - compact tracked `results/` artifacts, with large perturbation-response
    detail manifests kept under `_artifacts/`.
- New related issues:
  - [issue:9d8eb39] - adversary formalism and same-channel GRU robustness path.
  - [issue:6f1ffa5] - Riccati-epsilon teacher curriculum for C&S GRUs.
  - [issue:4d79e07] - this PGD 3e-3 replication/expansion issue.

## Scientific Framing

The current PGD lane is a same-channel broad/full-state epsilon surrogate:
optimize an open-loop `T x 8` epsilon sequence on the C&S `B_w=[I_8;0]`
channel and train/evaluate the GRU under that adversarial rollout loss. This is
not automatically the exact finite-horizon H-infinity Riccati adversary, because
the analytical adversary is a state-dependent feedback policy. Treat PGD as a
robust-discovery training tool and same-channel audit, while [issue:9d8eb39]
owns the formal PGD-vs-Riccati adequacy/equivalence path.

The next experiment should ask a narrower question:

> Does the promising lr=3e-3 PGD broad-epsilon GRU result replicate across more
> seeds and sensible PGD pressure/budget settings when compared to matched
> force/filter-feedback non-PGD baselines?

## Essential Context To Read First

Start from these issue reports and artifacts:

- `mandible issue report 4d79e07 --json`
- `mandible issue report 020a65b --json`
- `mandible issue report b8aa38e --json`
- `mandible issue report b35595c --json`
- `mandible issue report c99ad9d --json`
- `results/c99ad9d/notes/perturbation_taxonomy.md`
- `results/c99ad9d/notes/adversarial_epsilon_robustification_plan.md`
- `results/c99ad9d/notes/pgd_bank_alpha_response_plan_20260607.md`
- `docs/diagnostic_stack.md`
- `results/020a65b/notes/gru_worst_case_epsilon_audit_pgd_bank_four_rows_validation_selected.md`
- `results/020a65b/notes/hinf_phenotype_sidecar_pgd_bank_four_rows_validation_selected.md`
- `results/020a65b/notes/pgd_bank_four_rows_validation_selected_broad_epsilon_attribution.md`
- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_broad_validation_selected.md`
- `results/b35595c/notes/calibrated_perturb_level_screen_summary.md`

The high-level vocabulary is in `perturbation_taxonomy.md`: broad epsilon is the
canonical C&S disturbance channel; calibrated perturbation-bank rows are
behavioral feedback/transfer probes; random broad epsilon is not the same as a
worst-case adversary; PGD is useful but must be labeled as an open-loop
same-channel surrogate unless the formal path validates it.

## Current Implementation Assumptions

Verify these live before launching new training:

- Force/filter feedback should be included by default for this lane. The GRU
  input should include the existing target-relative position/velocity feedback
  plus the force/filter-state feedback channels.
- PGD should operate on the broad/full-state epsilon channel with shape `T x 8`
  and the C&S `B_w=[I_8;0]` mapping.
- The corrected PGD hyperparameters should round-trip through run specs. In
  particular, nested inner-maximizer fields such as step count and step size
  must not silently fall back to defaults.
- Current semantics from `pgd_bank_alpha_response_plan_20260607.md`: if
  calibrated perturbation training and PGD are both enabled, sample nominal or
  calibrated-bank trials from the training distribution, then add PGD broad
  epsilon to that sampled trial. There is not yet a separate explicit alpha
  coefficient unless a future implementation adds one.
- PGD should report inner-loop diagnostics: declared radius, achieved epsilon
  norm/radius, pre/post inner loss, best-seen objective, projection saturation,
  and epsilon energy by time/component where available.

## Proposed First Matrix

Keep the first continuation small and interpretable:

| Row | lr | feedback | calibrated bank | PGD | purpose |
|---|---:|---|---|---|---|
| baseline | 3e-3 | force/filter on | selected level, likely small or moderate | off | matched non-robust comparator |
| PGD | 3e-3 | force/filter on | same as baseline | on | replicate promising robust-discovery row |

Then expand only after the first pair is healthy:

- more random seeds/replicates for the 3e-3 PGD row;
- moderate versus strong same-channel epsilon budget;
- PGD pressure sweep by inner steps/restarts/step size;
- optional 1e-3 anchor if optimizer-specificity needs interpretation.

The issue body currently asks to expand beyond the default five replicates and
to predeclare the final seed count before launch. A conservative first
replication would use at least 10 replicates for the primary 3e-3 PGD condition
if local runtime is acceptable; otherwise run five first, document the reason,
and treat it as a gate rather than a final replication.

## Smoke Gate Before Full Runs

Before any 12k batch run:

1. Run a short 1k-batch smoke for the exact baseline/PGD configuration.
2. Confirm the reach is present, losses are finite, and training metrics are
   emitted.
3. Confirm PGD training metrics are nontrivial and not silently defaulted:
   achieved radius, inner loss improvement, best-seen objective, gradient norms,
   clipping fraction, and update/parameter ratio.
4. Confirm post-run materializers can load the smoke checkpoint:
   standard certificate, objective comparator, perturbation-response bank,
   worst-case epsilon audit, H-infinity phenotype sidecar, feedback ablation,
   map-error decomposition, velocity profiles, and loss figures.
5. Only then launch full 12k rows.

## Required Post-Run Diagnostics

Run the full diagnostic set by default, validation-selected checkpoints primary
and feedback-selected checkpoints audit-only unless a later issue changes that:

- training diagnostics, including gradient norm, clipping fraction,
  update/parameter ratio, LR schedule, and PGD inner-loop metrics;
- standard certificate;
- objective comparator with extLQG and robust output-feedback analytical rows
  where defined;
- perturbation-response bank with calibrated rows, class/timing summaries, delta
  x/u metrics, attenuation metrics, extLQG ratios, and robust analytical
  comparator where available;
- same-channel worst-case epsilon audit;
- H-infinity phenotype sidecar;
- feedback ablation/lens bundle;
- task-aligned/covariance-weighted map-error decomposition;
- velocity profile and loss figures;
- all-replicate tables and checkpoint-selection summaries.

Do not omit final-checkpoint summaries if they are cheap, but the primary
interpretation should use validation-selected best checkpoints.

## Comparison Baselines

Compare the new rows against:

- the matched non-PGD force/filter-feedback baseline from the same new matrix;
- the prior `b35595c` calibrated perturbation screen for non-proprio,
  non-PGD feedback competence;
- the prior `b8aa38e` proprioceptive/robust overnight rows;
- the analytical extLQG controller for nominal/feedback competence;
- the output-feedback robust analytical controller for same-channel robust
  comparisons where the diagnostic defines that comparator.

For robustness interpretation, side-by-side tables should show absolute metrics
and ratios. In particular, include worst-case full-Q/R/Q_f cost, delta cost,
peak/mean `delta x`, peak/mean `delta u`, endpoint error, attenuation-style
metrics, and GRU/analytical ratios. Do not rely on a single scalar.

## Stop Gates

Stop and report before launching the full matrix if any of these happen:

- PGD diagnostics are missing, zero, deterministic in a suspicious way, or do
  not reflect the configured budget/step count.
- The calibrated-bank rows are not actually present in the training distribution
  when the row name/spec says they are.
- Force/filter feedback channels are not wired into the GRU input.
- Command-input or process-epsilon perturbation rows materialize as zero-effect
  rows despite nonzero payloads.
- Validation-selected checkpoint materialization cannot be reproduced from the
  run specs and `_artifacts` checkpoint directories.

## Artifact Discipline

- Keep run specs, compact notes, diagnostic summaries, and figure specs under
  `results/4d79e07/`.
- Keep checkpoints, full perturbation-response detail manifests, raw training
  histories, large NPZs, and HTML figure renders under `_artifacts/4d79e07/`.
- Use regeneration specs for figure/diagnostic materializers when possible.
- If a review packet is needed, create it under a temporary directory and keep
  bulky files out unless explicitly requested.

## Suggested First Commands

```bash
cd "/Users/mll/Main/10 Projects/10 PhD/rlrmp"
mandible issue report 4d79e07 --json
mandible issue report 020a65b --json
mandible issue report c99ad9d --json
rg -n "pgd|broad_epsilon|force_filter|proprio|inner_max" src scripts tests results/020a65b results/b8aa38e results/c99ad9d
```

Then create a new feature worktree from current `main`, inspect the available
training CLI/run-spec helpers, and write the exact run-spec names before smoke
testing.

## Open Questions For The Next Agent

- Final replicate count for the first 3e-3 PGD replication: five as a gate, or
  at least ten as the first serious replication?
- Which calibrated-bank level should be paired with PGD first: small or
  moderate? The previous plan leans toward calibrated-bank plus PGD, but the
  exact bank level should be predeclared.
- Should the first expansion vary budget, PGD pressure, or replicate count
  first? The safest order is replicate the promising row, then budget/pressure.
- Should a future run use an explicit PGD alpha/mix fraction? Not for the first
  continuation unless the user redirects; current plan keeps PGD on every
  sampled trial for a clean comparison.

