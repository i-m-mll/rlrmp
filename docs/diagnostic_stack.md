# Diagnostic Stack for C&S GRU Robustness Work

Issue: `abe33da` - H-infinity phenotype sidecar for GRU robustness
interpretation

Coordination: `4d38c15` for analysis surfaces, `c99ad9d` for training-method
and perturbation-family vocabulary.

Last updated: 2026-06-04

## Purpose

This document is the durable registry for the diagnostic stack used to interpret
C&S GRU robustness results. It separates certificate gates, objective
comparators, perturbation-response diagnostics, feedback-map diagnostics,
training diagnostics, and phenotype sidecars so future agents do not have to
reconstruct the stack from issue comments and scattered notes.

The H-infinity phenotype sidecar is an interpretive aggregation layer. It is not
the standard certificate, not a replacement for the objective comparator or
perturbation-response bank, and not a formal robust-game proof. Formal
H-infinity claims still require the declared game card, disturbance channel,
gamma or budget, information pattern, and exact audit or induced-gain evidence
for the policy being claimed.

## Reading Order

Use this stack from bottom to top:

1. Confirm the run contract and checkpoint-selection rule.
2. Read the standard certificate before sidecars.
3. Use the objective comparator to interpret scalar cost claims.
4. Use the perturbation-response bank and feedback ablation to test recovery
   behavior.
5. Use the feedback-control quality lens bundle to judge whether the controller
   actually recovers from perturbations.
6. Use map-error decomposition to identify local feedback-law failures.
7. Use training diagnostics to explain why a row may have learned or failed.
8. Use the H-infinity phenotype sidecar only as an integrated interpretation.

If a higher layer conflicts with a lower certificate or provenance guardrail,
the lower layer wins.

## Layer Registry

| Layer | What it answers | Applicable when | Must not be used for | Checkpoint-selection status | Provenance and inputs | Likely result paths |
|---|---|---|---|---|---|---|
| Run contract and review artifacts | What was trained, what objective was optimized, what validation bins selected checkpoints, and which artifacts exist. | Every run or sweep. Required before interpreting any diagnostic. | Inferring robustness or same-game equivalence by itself. | Owns checkpoint selection when the run spec says so. Analytical action, I/O, Jacobian, map, and phenotype metrics are audit-only unless a later issue explicitly changes the run contract. | `results/<issue>/runs/<variant>/run.json`, model graph manifests, postrun materialization manifests, run review notes. | `results/aacb9ed/runs/*/run.json`, `results/ba82f3d/runs/*/run.json`, `results/<issue>/notes/*review*.md`, `_artifacts/<issue>/runs/<variant>/` for bulk outputs. |
| Standard certificate | Whether a controller satisfies the declared same-game certificate components available for that architecture: action behavior, response maps, transition/value/Bellman components where meaningful, and explicit `not_applicable` where they are not. | Required for linear and recurrent bridge claims. For GRUs, current rows are `empirical_nonlinear`: clean action behavior and observation-to-action maps can be reported, while same-coordinate transition, value, and Bellman components are not applicable unless a separate local-linear certificate is defined. | Replacing unavailable components with plant-state static-gain surrogates, treating raw gain mismatch as the gate, or calling partial GRU evidence a full certificate. | Audit-only unless the run contract explicitly selects by certificate metrics. Current C&S GRU rows use validation-selected checkpoints from rollout losses, not certificate metrics. | Validation-selected checkpoint, extLQG or H-infinity reference, shared observation contract, rollout states/actions, fitted response maps, certificate manifest. | `results/aacb9ed/notes/gru_standard_certificates_*_validation_selected.md`, `results/ba82f3d/notes/gru_standard_certificates_*_validation_selected.md`, older linear/output-feedback notes under `results/43e8728/` descendants. |
| Objective comparator | Whether scalar costs are apples-to-apples across GRU and analytical references, including deterministic full-Q/R/Q_f, covariance-inclusive expected cost, shared-rollout stress rows, and split-bank rescores. | When cost ratios, full-Q/R/Q_f values, or "near extLQG objective" claims are reported. | Treating covariance-inclusive expected cost as comparable to realized GRU validation scalars, treating stress-bank shared rollouts as expected cost without the required sanity checks, or using cost-sidecar values as certificate gates. | Diagnostic only for current rows. GRU values may be validation-selected realized scalars, but post-hoc shared-rollout and split-bank blocks are audit-only. | Run spec objective contract, validation scalar records, extLQG deterministic and expected-cost terms, shared initial-state/process-epsilon banks, per-term scorer status. | `results/aacb9ed/notes/objective_comparator_*_validation_selected.md`, `results/ba82f3d/notes/objective_comparator_*_validation_selected.md`. |
| Standard perturbation-response bank | How nominal and trained controllers respond to predeclared perturbation families: initial-state offsets, process/load epsilon, sensory feedback, delayed observation, command input where supported, target stream where meaningful, and transfer rows. | When evaluating feedback competence, recovery, disturbance attenuation, perturbation-family transfer, or stress-bank failures. | Calling a restricted perturbation family H-infinity-equivalent unless the game card embeds that channel; using perturbation rows as checkpoint selectors unless declared by the run spec. | Usually audit-only after validation selection. Some run contracts select by aggregate rollout loss over held-out perturbation bins; the bank's post-hoc analytical rows remain diagnostic unless explicitly declared. | Perturbation taxonomy, bank manifest, controller-visible channels, extLQG comparator availability, full-Q/R/Q_f rescoring, perturbation amplitudes and seeds. | `results/3992394/notes/gru_perturbation_response_fullqrf_validation_selected.md`, `results/aacb9ed/notes/gru_perturbation_response_*_validation_selected.md`, `results/ba82f3d/notes/gru_perturbation_response_*_validation_selected.md`, taxonomy in `results/c99ad9d/notes/perturbation_taxonomy.md`. |
| Feedback ablation | Whether performance depends on feedback input, recurrent state, or a specific perturbation/recovery signal rather than only feedforward reach timing. | Useful for GRU rows with suspiciously good nominal behavior but weak recovery or response-map evidence. | Proving same-game equivalence or H-infinity optimality. Ablation says a signal matters, not that the implemented law matches the analytical law. | Audit-only unless a future training contract makes ablation robustness a selector. | Paired evaluations with feedback, hidden state, observation, or channel interventions; same validation/task bins as the source run when possible. | Expected under `results/<run-issue>/notes/*feedback_ablation*.md` or folded into `results/<run-issue>/notes/gru_evaluation_diagnostics_*_validation_selected.json` until a stable note exists. |
| Feedback-control quality lens bundle | Whether feedback control is good enough for the claim being made, combining absolute recovery, feedback-dependence, and reference-relative perturbation cost. | Standard interpretation layer for C&S GRU rows evaluated under perturbations, especially when extLQG map identity is not required. | Replacing the standard certificate, hiding per-class failures in one scalar, or claiming extLQG policy identity from good perturbation cost alone. | Audit-only unless a future run contract explicitly selects on perturbation-bin validation loss. The post-hoc lens bundle itself is not a hidden selector. | Standard perturbation-response bank, feedback ablation, objective comparator on shared perturbation banks, evaluation diagnostics, optional response plots, and standard certificate context. | Usually assembled from `results/<issue>/notes/gru_perturbation_response_*`, `results/<issue>/notes/*feedback_ablation*`, `results/<issue>/notes/objective_comparator_*`, and response figures/specs when requested. |
| Map-error decomposition | Where the GRU observation-history-to-action response map differs from the reference: raw norm, covariance-weighted mismatch, task-aligned directions, norm ratio, cosine, scalar gain, residual, singular directions, and weakly visited directions. | When standard certificates report poor observation-to-action maps, or when good nominal cost coexists with poor perturbation recovery. | Rescuing a failed certificate by explaining it away. Decomposition classifies the failure; it does not change the bridge gate. | Audit-only for current C&S GRU rows. Analytical map metrics are not checkpoint selectors unless a guided lane explicitly declares them as training or selection objectives. | Fitted GRU and reference response maps, observation covariance, task-aligned basis/probes, row identity, validation-selected checkpoint. | `results/aacb9ed/notes/gru_map_error_decomposition_*_validation_selected.md`, `results/aacb9ed/notes/*_aligned.md`, `results/ba82f3d/notes/gru_map_error_decomposition_*_validation_selected.md`, `results/ba82f3d/notes/*_aligned.md`. |
| Training diagnostics | Why a row did or did not learn: loss curves, validation-bin breakdowns, pre-go drift, reach kinematics, perturbation-bin losses, optimizer stability, hidden-state support, and supervision/teacher terms when present. | Every training run, especially when a sidecar suggests robustness or when a certificate fails despite good nominal behavior. | Treating training loss improvement as proof of analytical equivalence, or comparing methods without separating training axes from evaluation lenses. | Owns checkpoint selection only through the predeclared validation rule in the run spec. Kinematic and analytical diagnostics are sidecars unless declared. | `run.json`, training logs, validation histories, checkpoint manifests, postrun materialization, training-method taxonomy. | `results/<issue>/runs/<variant>/run.json`, `results/<issue>/notes/validation_selected_checkpoints*.json`, `results/<issue>/notes/gru_postrun_materialization*.json`, bulk logs under `_artifacts/<issue>/runs/<variant>/`. |
| H-infinity phenotype sidecar | Whether the row looks robust-control-like in the behavioral sense: nominal efficiency, recovery competence, feedback gain/signature, disturbance attenuation, Delta-v or peak-forward-velocity inflation, early acceleration, induced-gain/exact-audit sidecars where available, and paired baseline-vs-robust contrasts. | After the lower layers have produced enough provenance to aggregate. Most useful for GRU interpretation and method triage. | Calling a row a standard certificate pass, formal H-infinity solution, same-game proof, or robust-game proof. Missing components must be `unavailable` or `not_applicable`, not silently omitted. | Audit-only for current issue `abe33da`. It must not select checkpoints unless a future issue explicitly changes the claim and run contract. | Pointers back to the standard certificate, objective comparator, perturbation-response bank, feedback ablation, map-error decomposition, induced-gain/exact-audit outputs, and evaluation diagnostics. | Expected under `results/abe33da/notes/h_infinity_phenotype_*_validation_selected.md` and matching JSON manifests; may aggregate existing `results/aacb9ed/`, `results/ba82f3d/`, `results/3992394/`, and future robust-training rows. |
| Figures and narrative review | Human-readable figures, plots, reviewer packets, and synthesis notes that summarize the above layers. | Useful for communication, issue comments, and papers. | Treating a plot or narrative as the source of truth when the manifest or diagnostic table disagrees. | Never selects checkpoints unless the underlying run spec says the plotted metric selects. | Figure specs, source manifests, analysis notes, reviewer packet inputs. | `results/<issue>/figures/<topic>/spec.json`, `_artifacts/<issue>/figures/<topic>/figure.html`, `results/<issue>/notes/*.md`, manuscript figure dirs when promoted. |

## H-infinity Phenotype Sidecar Contract

The `abe33da` sidecar should aggregate, not recompute or relabel, the lower
diagnostics. Each row should preserve these fields or explicit unavailable
markers:

- Row identity: source issue, run variant, checkpoint selector, replicate or
  aggregate scope, training objective family, perturbation/adversary channel,
  and evaluation lens.
- Nominal efficiency: endpoint error, terminal speed, command energy, peak
  velocity, time-to-peak, and full-Q/R/Q_f scalar where comparable.
- Feedback competence: perturbation-response bins, recovery/displacement/action
  response, disturbance attenuation, feedback-ablation deltas, and deferred
  channel notes.
- Local feedback law: raw response-map mismatch, covariance-weighted mismatch,
  task-aligned mismatch/probes, map norm ratio, cosine, scalar gain, residual,
  and top error directions.
- H-infinity-like phenotype: Delta-v or peak-forward-velocity inflation,
  early-acceleration signature, feedback-gain magnitude, induced-gain or exact
  audit metrics where available, worst-case or CVaR disturbed loss where
  available, and paired nominal-vs-robust contrasts.
- Provenance links: every aggregate field points to the lower diagnostic file
  or manifest that produced it.

Interpretation labels should be conservative. For example, a row can be labeled
`robust_phenotype_candidate` only when it has a coherent pattern across nominal
quality, recovery, and feedback-law evidence. It should be labeled
`phenotype_only_no_formal_game_proof` unless the game card, gamma/budget,
disturbance channel, information pattern, and exact audit or induced-gain check
are present. If exact-L2/gamma or induced-gain sidecars improve while standard
action/value/transition/reference-equivalence metrics still fail, use the
existing `sidecar_improving_non_equivalent` framing rather than a pass label.

## Feedback-Control Quality Lens Bundle

This is a standard interpretation bundle, not a new single diagnostic metric.
Use it when the question is whether a controller has useful feedback control
under perturbations, especially when exact extLQG observation-to-action map
identity is not the claim. The bundle combines three lenses:

1. Absolute recovery behavior. Check whether perturbation responses are stable,
   vigorous enough, and task-useful. Relevant quantities include endpoint error,
   terminal speed, recovery time, overshoot, post-peak sign changes, command
   norm, first-few-step command norm, command jerk, displacement reduction, and
   class-binned full-Q/R/Q_f perturbation cost. Response plots are legitimate
   evidence here, but the source of truth remains the perturbation-response
   manifest and evaluation table.
2. Feedback-dependence / ablation. Check whether the controller actually uses
   feedback rather than replaying an open-loop motor tape. Normal rollouts should
   degrade under feedback removal, delayed-observation scrambling, sensory
   channel masking, or related ablations in the perturbation bins where feedback
   should matter. This demonstrates feedback sensitivity, not analytical-law
   identity.
3. Reference-relative perturbation performance. Compare GRU and extLQG on the
   same sampled initial-state/noise/perturbation bank with the same full-Q/R/Q_f
   scoring objective, reported separately by perturbation class. A GRU can be a
   good feedback controller if its perturbation cost is near extLQG even when
   its local I/O map differs. Conversely, nominal kinematics or good absolute
   recovery do not by themselves establish extLQG-level feedback performance if
   same-bank cost is much worse.

Report these lenses together. Do not collapse them into one pass/fail scalar
unless a specific issue defines a task gate. A typical summary should say:

- whether absolute perturbation recovery is stable and task-useful;
- whether feedback ablation shows genuine feedback dependence;
- how GRU cost compares with extLQG on the same perturbation bank, by class;
- whether failures are concentrated in initial state, process/load, sensory,
  delayed-observation, command-input, target-stream, or combined perturbations;
- whether the standard certificate and map decomposition show policy identity,
  partial mismatch, or a different but effective feedback law.

This bundle is allowed to judge feedback-control performance without requiring
deep extLQG I/O identity. It is not allowed to turn good feedback performance
into a formal H-infinity claim; that still requires the game card, disturbance
channel, gamma or budget, and exact audit or induced-gain evidence.

## Checkpoint-Selection Guardrail

Current C&S GRU rows are selected by rollout validation rules declared in
`run.json`, such as aggregate validation loss over held-out perturbation bins or
target-relative validation bins. Analytical action, I/O, response-map,
Jacobian, phenotype, and H-infinity sidecar metrics are audit-only unless a
future issue explicitly changes both the claim and the run contract.

When reporting a selected checkpoint, always state:

- the selector name and validation bins;
- whether nominal quality is a sidecar/gate or a selector;
- which diagnostics were materialized after selection;
- whether any component was unavailable, not applicable, or stress-test-only.

## Delayed-Reach and h0/Pre-roll Guardrail

Delayed-reach task changes and h0/pre-roll support answer different questions.
Issue `6c36536` covers the delayed-reach task contract: changing when the reach
starts, how target visibility and movement epochs are represented, and which
validation bins define task performance. Issue `643f101` covers initial-state
recovery and h0 support: whether the recurrent state and observation history are
conditioned consistently when initial mechanics are perturbed or when a pre-roll
is needed.

Do not interpret an initial-position recovery failure as only a delayed-reach
problem unless h0/pre-roll support is already controlled. Conversely, do not
treat improved h0/pre-roll handling as a delayed-reach training result unless
the delayed task contract and validation selector were actually used.

## Provenance Rules

- Use `results/` for tracked run specs, diagnostic notes, manifests, figure
  specs, and narratives.
- Use `_artifacts/` for bulk checkpoints, logs, arrays, rendered figures, and
  large outputs.
- Prefer issue-hash result roots: `results/<7-char-issue>/...`.
- Keep run specs flat as `results/<issue>/runs/<variant>.json` unless the run
  already has additional per-run tracked files.
- Do not copy numerical tables into a synthesis note as the only source of
  truth. Point to the diagnostic note or manifest that generated the numbers.
- Keep training axes separate from evaluation lenses. A robust-trained row can
  be evaluated nominally, and a nominal row can be evaluated under perturbation
  or H-infinity-inspired lenses.

## Issue Map

- `abe33da` - H-infinity phenotype sidecar for GRU robustness interpretation.
- `4d38c15` - Project analyses coordination; comment here for analysis-stack
  tier changes and cross-cutting analysis decisions.
- `c99ad9d` - Project training-methods coordination; comment here for
  perturbation-family, adversary, and robustness-training vocabulary changes.
- `e4800d6` - Robust-control strategy ladder for C&S GRUs.
- `aacb9ed` - Task-breadth diagnostics for C&S GRU feedback-policy
  identification.
- `ba82f3d` - Target-relative multi-target C&S GRU training contract.
- `0fc6814` - Task-aligned map decomposition for C&S GRU diagnostics.
- `643f101` - Initial-state recovery and h0 support for C&S GRUs.
- `6c36536` - Delayed-reach C&S GRU task contract.
- `c314267` - Guided feedback-response supervision for C&S GRU
  identification.
- `db35426` - H-infinity Riccati teacher: behavioural cloning and fine-tune.
- `43e8728` - cs2019-to-RNN game-equivalence umbrella.
- `020a65b` - Full-state epsilon adversary class matching the C&S Eq. 13
  H-infinity disturbance.
- `a8462de` - Apples-to-apples objective comparator sidecar for C&S GRU rows.
- `3992394` - Standard perturbation-response diagnostic bank.

Update this map when a diagnostic layer becomes canonical, moves issue ownership,
or gets replaced by a newer contract.
