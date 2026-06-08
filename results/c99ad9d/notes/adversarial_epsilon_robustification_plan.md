# Adversarial Epsilon Robustification Plan

Issue: `c99ad9d` - Project training-methods coordination

Related issues: `020a65b`, `e4800d6`, `abe33da`, `b8aa38e`, `b35595c`,
`3992394`, `1ad3c16`, `a7dad8a`, `cb98e58`

Date: 2026-06-06

Status: plan only. No new robustification runs are authorized by this note.

## Purpose

This note records the next robustification plan after the overnight
`b8aa38e` broad-epsilon/proprioceptive-feedback screen and the subsequent
external critique. It is intended to keep the robust-control interpretation
durable instead of leaving it only in chat context.

## Evidence Reviewed

- External critique in
  `/Users/mll/.codex/attachments/6907d0e9-4624-4a62-ad11-25b2ce8067db/pasted-text.txt`.
- Project taxonomy in `results/c99ad9d/notes/perturbation_taxonomy.md`.
- Diagnostic-stack registry in `docs/diagnostic_stack.md`.
- Adversary-equivalence note in `results/a7dad8a/notes/adversary_equivalence.md`.
- Current issue state for `c99ad9d`, `4d38c15`, `020a65b`, `e4800d6`,
  `abe33da`, and `b8aa38e`.
- Independent read-only formalism review by a high-effort subagent.

## Current Interpretation

The broad/full-state epsilon lane remains the correct formal C&S H-infinity
disturbance lane for this project. In the current game card, epsilon is an 8D
additive disturbance entering the current physical-state block through
`B_w[:8, :] = I_8`, with no direct writes into delay-buffer history.

The `b8aa38e` broad-epsilon result should not be read as "full-state epsilon
failed." It should be read as:

1. The implemented broad-epsilon training pressure was weak for H-infinity
   purposes.
2. The force/filter feedback observation contract is a real lead for competent
   feedback control.
3. No row yet supports a formal H-infinity claim, because same-channel
   worst-case or induced-gain evidence is still missing.

## Why Random Broad Epsilon Was Not Enough

The overnight broad-epsilon rows sampled iid normal epsilon over time and all
8 components, projected each trial to one flattened rollout L2 budget, and
optimized expected loss over that sampled distribution.

That is not the same mathematical claim as H-infinity. Random sampled training
estimates something like expected disturbed cost under a distribution. The
H-infinity game asks for a worst-case or induced-gain statement over the
declared disturbance channel and budget.

The problem is especially sharp here because the disturbance object is the full
`T x 8` epsilon sequence. A random L2-projected vector spreads energy across
many coordinates and need not align with the directions that maximize value
growth or recovery difficulty. Therefore matching the total Riccati-derived
energy budget is not enough to make the training pressure equivalent to the
Riccati adversary.

PGD is not uniquely mandatory. The mandatory property is same-channel
worst-case optimization or certification. Valid routes include:

- exact Riccati or robust-Bellman inner maximization where available;
- exact induced-gain or gamma audit for a frozen policy;
- PGD over the full `T x 8` epsilon sequence under the declared L2 budget;
- a validated open-loop surrogate whose adequacy has been checked for the
  relevant controller class and rollout setting.

CVaR/top-k over sampled epsilon is a useful intermediate training pressure, but
it should be labeled as tail-risk stochastic robustness unless it is tied to a
validated worst-case approximation.

## Force/Filter Feedback Interpretation

The force/filter feedback rows are important because they change the controller
information contract. The GRU no longer has to infer the force/filter state
only from delayed kinematic history; it receives two additional delayed
force/filter coordinates.

That should be treated as a richer proprioceptive feedback baseline, not as an
H-infinity method by itself. Comparisons should therefore be matched by
observation contract:

- kinematic-only GRU versus kinematic-only GRU;
- force/filter-feedback GRU baseline versus force/filter-feedback robustified
  GRU;
- distilled or analytical-anchor controllers, when used, should declare the
  same observation contract or be explicitly labeled as different.

The current best next baseline lane is probably force/filter feedback with
none/small/moderate calibrated perturbation training, with stress treated as an
aggressive candidate only after response curves rule out overreaction.

## Immediate Work Before More Overnight Runs

### 1. Add Paired Broad-Epsilon Attribution Diagnostics

For the same model and same batch, compute:

- loss with broad epsilon active;
- loss with epsilon zeroed;
- per-term loss delta under full Q/R/Q_f;
- gradient norm with and without epsilon;
- gradient cosine or update-direction agreement;
- contribution of broad epsilon to clipping and update/parameter ratio when
  available.

This answers whether broad epsilon materially changes the training update or
only adds a small diffuse loss term.

### 2. Add Same-Channel Worst-Case Epsilon Audit

For existing rows before retraining, evaluate a frozen controller under a
worst-case epsilon search over the full `T x 8` sequence and the same declared
budget. The first practical implementation should be projected gradient ascent
with multiple restarts, retaining the best incumbent across the optimization
path.

Report at least:

- worst-case full-Q/R/Q_f cost and delta cost;
- epsilon energy and budget compliance;
- peak and AUC delta x;
- peak and AUC delta u;
- endpoint error and terminal speed;
- induced-gain/gamma sidecars where available;
- comparison to random iid projected epsilon at the same budget;
- comparison to Riccati-realized epsilon where the analytical reference
  supplies one.

This is an audit first. It should be run on selected existing rows before it
becomes a training objective.

### 3. Preserve Existing Diagnostic Layers

Do not replace the current perturbation-response bank, feedback ablation,
map-error decomposition, objective comparator, or H-infinity phenotype sidecar.
The new worst-case epsilon audit should feed the phenotype sidecar as another
formal/near-formal evidence component.

Validation-selected and feedback-selected checkpoints should continue to be
reported separately when both are materialized. Analytical action/I/O/map
metrics remain audit-only unless a future run explicitly declares them as
selection criteria.

## Training Ladder After Audit

The next training matrix should be smaller than another broad exploratory
overnight screen and should use force/filter feedback as the main baseline
contract.

### Stage A: Matched Feedback Baselines

Train or select matched force/filter-feedback baselines:

- no calibrated perturbation training;
- small calibrated perturbation training;
- moderate calibrated perturbation training;
- stress only if response curves show it is not an overreactive baseline.

Use these to define "good feedback control without formal robust pressure."

### Stage B: Tail-Risk Broad Epsilon

Train a CVaR/top-k broad-epsilon variant on the same force/filter-feedback
contract. This should optimize the worst sampled fraction of full-state epsilon
rollouts, not the mean alone.

Label this as stochastic tail-risk robustness unless the follow-up worst-case
audit shows that it approximates same-channel worst-case behavior.

### Stage C: PGD Broad Epsilon

Train a projected-gradient broad-epsilon variant over the full `T x 8` epsilon
sequence. Start with conservative smoke tests:

- low PGD step count;
- multiple restarts only after the single-restart path is stable;
- strict budget projection;
- gradient clipping and update diagnostics active;
- short-run postrun materialization before launching full rows.

This is the first practical GRU training row that can plausibly support
H-infinity-like evidence, subject to postrun exact/induced-gain audits.

### Stage D: Optional Budget Scale Diagnostic

If the same-channel audit shows the current moderate/strong budgets produce
negligible worst-case effect even under PGD, run a budget-scale diagnostic
before interpreting broad-epsilon failure:

- current budget;
- 2x to 5x radius;
- 10x radius only as a stress diagnostic if lower scales remain too weak.

This should be framed as a budget sensitivity test, not as a new formal C&S
game unless the game card is deliberately changed.

## Evaluation Plan

Every robustification row should report:

- nominal kinematics and full-Q/R/Q_f objective;
- standard certificate components available for GRUs;
- objective comparator with any same-bank caveats explicit;
- standard perturbation-response bank with calibrated families/timing bins;
- feedback ablation;
- task-aligned map-error decomposition;
- H-infinity phenotype sidecar;
- new same-channel worst-case epsilon audit;
- all-replicate reporting and validation-selected versus feedback-selected
  checkpoint comparison where applicable.

The robustness claim should be tiered:

1. competent feedback baseline: nominal behavior plus absolute recovery and
   feedback-dependence evidence;
2. tail-risk robust candidate: improved CVaR/top-k and perturbation response,
   no formal H-infinity claim;
3. H-infinity-like candidate: improved same-channel worst-case epsilon audit,
   induced-gain/gamma sidecar, and phenotype evidence;
4. formal same-game claim: only if game card, channel, gamma/budget,
   information pattern, and exact audit or induced-gain check all support it.

## Issue Handling

No duplicate issue is needed just to name this plan.

- `c99ad9d` remains the coordination surface for the training-method taxonomy
  and this plan.
- `020a65b` already owns the full-state epsilon adversary class and should own
  implementation of PGD/full-epsilon training or audit machinery.
- `e4800d6` owns the robust-control strategy ladder and matched
  baseline-vs-robust comparisons.
- `abe33da` owns phenotype-sidecar integration of any new worst-case epsilon
  audit metrics.
- If implementation scope becomes too broad, file a focused child issue for
  "same-channel worst-case epsilon audit and PGD training objective" and
  cross-reference it from `020a65b`, `e4800d6`, `abe33da`, and `c99ad9d`.

## Dotfiles Planning-Workflow Follow-Up

The requested general workflow change was delegated separately in the dotfiles
repo. The worker created issue `b82e4e1`, branch
`feature/persist-planning-artifacts`, and commit
`72fced5d9ac0334f7f800d7327f1c371b2b3dbdc`, updating the installed Codex
AGENTS file and repo-local Claude instructions to require durable Markdown
planning artifacts when the user asks for a plan about how to proceed.

## Recommended Next Action

If the user says to proceed, do not immediately launch another 12-run overnight
matrix. First implement the paired broad-epsilon attribution diagnostic and the
same-channel worst-case epsilon audit, run them on the existing `b8aa38e`
rows, and use those results to choose the first CVaR/top-k versus PGD training
screen on the force/filter-feedback baseline.

## 2026-06-07 Audit Update

The two pre-training audits above were implemented and materialized on the
existing `b8aa38e` checkpoints.

### Paired Broad-Epsilon Attribution

Outputs:

- `results/b8aa38e/notes/gru_broad_epsilon_attribution_validation_selected.md`
- `results/b8aa38e/notes/gru_broad_epsilon_attribution_validation_selected.json`
- `results/b8aa38e/notes/gru_broad_epsilon_attribution_validation_selected.csv`
- `_artifacts/b8aa38e/broad_epsilon_attribution/gru_broad_epsilon_attribution_validation_selected/`

Corrected semantics: the active condition is sampled through the run's actual
training task stack. The paired condition replays the same target and
calibrated-perturbation sampler branches but removes only the broad-epsilon
draw. It does not zero unrelated graph-channel perturbations or non-broad
process-epsilon inputs.

Result summary over the eight non-smoke broad-epsilon rows:

| broad level | calibrated perturbation level | active-minus-without-broad loss delta |
|---|---|---:|
| moderate | none | 3.64 |
| moderate | small | 3.05 |
| moderate | moderate | 4.35 |
| moderate | stress | 7.11 |
| strong | none | 9.26 |
| strong | small | 7.59 |
| strong | moderate | 10.26 |
| strong | stress | 16.47 |

Interpretation: broad epsilon is present and contributes to the sampled
training objective and raw gradient direction, but the mean active-minus-paired
loss deltas are modest relative to the total losses in these rows. This
supports the earlier concern that random projected broad epsilon is a weak,
diffuse training pressure rather than an H-infinity-like inner maximizer.

### Same-Channel Worst-Case Epsilon Audit

Outputs:

- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_broad_validation_selected.md`
- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_broad_validation_selected.json`
- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_proprio_moderate_validation_selected.md`
- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_proprio_moderate_validation_selected.json`
- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_proprio_strong_validation_selected.md`
- `results/b8aa38e/notes/gru_worst_case_epsilon_audit_proprio_strong_validation_selected.json`

Audit method: projected gradient ascent over the full `T x 8` epsilon sequence
under a flattened rollout L2 ball. Broad-epsilon rows use their declared
moderate or strong budget. Proprioceptive-feedback rows did not train with
broad epsilon, so they were audited under explicit moderate and strong budget
overrides on the same channel.

Broad-row result summary:

| declared budget | zero cost range | optimized cost range | optimized delta range |
|---|---:|---:|---:|
| moderate, L2 0.00123243 | 4610-4667 | 5423-5513 | 792-859 |
| strong, L2 0.00232849 | 4617-4682 | 6734-6870 | 2117-2213 |

Force/filter-proprio row result summary:

| audit budget override | zero cost range | optimized cost range | optimized delta range |
|---|---:|---:|---:|
| moderate, L2 0.00123243 | 4629-4665 | 5454-5544 | 825-879 |
| strong, L2 0.00232849 | 4629-4665 | 6785-6894 | 2156-2229 |

In all audited rows, the optimized epsilon beat the best iid projected random
epsilon candidate at the same budget. This is the important sanity check: the
audit is finding same-channel high-cost directions that random broad-epsilon
sampling would usually miss.

### Hyperparameter Consequences

Use the analytical budget levels as the first principled robustification
screen:

- moderate: `gamma_factor = 1.4`, `closed_loop_epsilon_l2_15cm =
  0.0012324305441740995`;
- strong: `gamma_factor = 1.05`, `closed_loop_epsilon_l2_15cm =
  0.0023284905801002004`;
- keep reach-length scaling on, so the radius scales linearly with trial reach
  length relative to 15 cm.

Do not invent an unanchored larger budget as the next first move. The same-
channel audit shows that the declared moderate and strong budgets already have
large worst-case effects when optimized adversarially. The problem is therefore
not that the budget is obviously too small; it is that random projected
sampling does not concentrate training on the bad directions.

Recommended next robustification screen:

1. Keep force/filter feedback enabled by default for all new C&S GRU rows unless
   the row is explicitly a historical/kinematic-only ablation.
2. Implement a training-time same-channel adversarial epsilon lane using
   projected gradient ascent over the full `T x 8` epsilon sequence.
3. Smoke locally before full runs:
   - `batch_size = 64`;
   - peak learning rate `1e-3`, warmup plus cosine decay as in `b8aa38e`;
   - `clip = 5`;
   - force/filter feedback enabled;
   - full Q/R/Q_f objective;
   - moderate budget first;
   - 1 restart, 2-3 PGD steps, step size `0.25 * radius`;
   - checkpoint and diagnostics at about 1000 batches before launching full
     12000-batch rows.
4. If the smoke is stable, run a small matrix rather than another broad sweep:
   - nominal force/filter-feedback baseline;
   - calibrated perturbation-only baseline at the best current physical level
     (likely moderate unless stress is deliberately being tested);
   - adversarial-epsilon moderate;
   - adversarial-epsilon strong.
5. For each full row, report validation-selected and feedback-selected
   checkpoints if they differ, and include the same-channel worst-case audit,
   perturbation-response bank, feedback ablation, H-infinity phenotype sidecar,
   nominal kinematics, loss plots, training diagnostics, and all-replicate
   summaries.

CVaR/top-k over sampled broad epsilon remains a useful parallel lane, but it is
second priority after the PGD lane because these audits show random projected
sampling misses the high-cost directions even at the correct budget.
