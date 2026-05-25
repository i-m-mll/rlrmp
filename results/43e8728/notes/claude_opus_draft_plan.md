# Plan: cs2019-to-RNN Game-Equivalence Umbrella (`43e8728`)

## 1. Scope and framing

This umbrella supersedes the broader "robustness reproduction" framing carried by older issues. The work is not "train an RNN and check whether it looks like humans on cs2019." It is:

> Given a plant, cost, horizon, observation structure, delay treatment, and disturbance channel for which a finite-horizon robust-control solution predicts coupled speed/gain modulation, can the rlrmp/feedbax training+evaluation stack reproduce the same closed-loop input-output game solved by the analytical H∞ Riccati recursion? Then, and only then, does a trained recurrent controller (GRU) reproduce, alter, or dissociate that signature?

Three things that are routinely conflated must be separated:

1. **Analytical sufficiency** — does the formal game produce the cs2019-like signature on the specified plant/cost?
2. **Training-stack validity** — does gradient-based minimax training in feedbax recover the known game in the *linear* case?
3. **Architectural effect** — once (1) and (2) hold, does a GRU still couple speed and feedback gain, or does it solve the game by a different policy?

Behavioral reproduction is downstream. It is not the first experiment.

## 2. Authority and priority order

When sources conflict, resolve in this order:

1. **GPT-5.5 Pro external review** of the cs2019-to-RNN robustness plan (controlling).
2. **Issue `0b1f109`** (planning/synthesis basis).
3. **Older issues** (`35f64be`, `6f783fa`, etc.) — subordinate only. v1/v2 of the robustness plan are historical; do not treat as canonical.

If any subordinate issue's framing conflicts with (1) or (2), update the subordinate issue with a back-reference and proceed under the higher-authority framing. Do not silently reinterpret older issues.

## 3. Central question, decomposed

Replace any "does Δv go positive?" framing with a **composite signature**, jointly held across:

- Nominal forward velocity / trajectory shape.
- Feedback-response gain under standardized perturbations (step loads, lateral pushes).
- Induced gain (disturbance-to-cost operator norm, or its surrogate).
- Lateral displacement / standardized perturbation-following error.
- Nominal cost decomposition (control energy, terminal error, time-varying state cost).
- Where the controller has muscle-like or antagonist-force variables, a co-contraction analog.

A controller that increases speed without an induced-gain improvement and sane cost decomposition has *not* reproduced the H∞ signature. Δv alone is not diagnostic.

## 4. Major unresolved formal choices (STOP-AND-ASK gates)

Agents working under this umbrella must **pause execution, flag the user, request explicit consent, and suggest a strong-model second opinion** before resolving any of the following. Do not adopt a default. Do not pick "what the older code does." Capture the decision as a comment on `43e8728` with a one-line summary of consent.

### 4.1 Open-loop ε trajectory vs. state-dependent H∞ adversary  *(highest priority)*

The current rlrmp training uses an open-loop per-timestep ε disturbance with a rollout-integrated L2 budget and global projection. The analytical H∞ object is a *dynamic game* whose worst-case disturbance is **state-dependent**, derived from the Riccati value matrix. The two adversaries are not obviously the same game. PGD on an open-loop ε can certify one nominal trajectory but may not certify the closed-loop disturbance-to-cost operator.

Decision required before Phase 1 training:
- Train against (a) open-loop ε only, (b) state-dependent Riccati adversary only, or (c) both as separate arms with a planned equivalence check.
- If (a) only: explicitly state and defend why open-loop is taken to be equivalent to the Riccati game for this plant/cost.
- Default recommendation in this plan: **(c) both**, with the analytical state-feedback adversary derived from the same Riccati recursion. Failing this gate under (a) but passing it under (c) would indicate a training parameterization limitation, not a target-side failure.

### 4.2 Eight-state vs six-state plant for the analytical target

Equivalence between the cs2019-faithful eight-state model and the six-state simplification is **not** a harmless implementation choice. The plan requires:
- Materialize the eight-state target first.
- Rerun Riccati on the six-state variant.
- Demonstrate equivalence on the H∞ quantities of interest **across multiple γ values**, not at one point.
- Re-check equivalence separately for the broad-ε analytical gate and for restricted physical-field contrasts (state dropped under one adversary class may still matter under another).

Pause and ask if any equivalence margin breaks beyond pre-declared tolerance.

### 4.3 Disturbance channel B_w

Which entries of the state does ε enter — positions, velocities, force states, all physical states, a weighted subset? This must be locked into the game card (§5) before any PGD or Riccati solve. Document the exact selection and norm convention.

### 4.4 γ → ε-budget mapping for PGD training

How is the analytical γ translated into the training-time ε budget? Does the norm include timestep scaling? Inconsistency here is the most common source of silent same-game failures in stochastic-control implementations.

### 4.5 Restricted-field arm: Experiment-1-like, Experiment-2-like, or both

cs2019 Experiment 1 (curl-only, interleaved CW/CCW) and Experiment 2 (step loads + curl + orthogonal velocity-dependent fields) are not the same protocol. The Experiment-2 block-level velocity contrast is weaker; trial-history and feedback-gain signatures carry more weight there. The "restricted-field baseline" must be one of:
- E1-like (curl interleaving, expected-cost / domain-randomization training).
- E2-like (mixed perturbation menu, expected-cost training).
- Worst-case-over-restricted-fields (different question; do not call this "human-protocol-like").

Pause and ask before collapsing these into a single comparison arm.

### 4.6 Feedback-gain measurement for nonlinear policies

For GRUs, "feedback gain" is not a single number. Choices:
- Local Jacobian of action w.r.t. observation along nominal trajectory.
- Standardized step-response amplitude.
- Induced-gain estimates (held-out adversary).
- All of the above as a panel.

The plan's default is all three, reported jointly. Confirm before Phase 2.

### 4.7 Pass/fail tolerances must be predeclared

Tolerances for gain-match, trajectory-match, cost-match, induced-gain, Δv, and held-out adversary loss are set from numerical and optimization noise once the **target artifact** exists, **before** looking at any GRU outcome. Lock the thresholds in writing and reference them when reporting the gate result.

## 5. Phase 0 — Analytical game card  *(blocking artifact)*

Before any training, materialize a single compact artifact (one Markdown + one JSON, under `results/43e8728/notes/game_card.md` and `results/43e8728/figures/game_card/spec.json`). It must contain:

- Discrete-time state vector (eight-state and six-state forms, delay augmentation explicit).
- Discrete-time convention (zero-order hold, Δt).
- Cost schedule: running state cost, control penalty scaling, terminal penalty, time-varying weights.
- Disturbance channel B_w with explicit row selection.
- ε norm definition and time-integration convention.
- γ* (smallest γ for which Riccati has a stabilizing solution) and chosen γ (one or several).
- Analytical LQR gain matrices (warmup target).
- Analytical H∞ gain matrices (adversary-trained target).
- Nominal trajectories (state, action, cost time course) under both controllers.
- Analytical worst-case disturbance characterization (state-feedback form).
- Closed-loop induced gain or direct DP induced-gain computation.
- Reference Δv (H∞-minus-LQR) on standardized reaches.

The card is the single reference object every later comparison audits against. Phase 1 training cannot begin until the card exists and is reviewed.

## 6. Phase 1 — Same-game linear gate

A time-varying linear regulator is trained in feedbax under the same plant/cost/disturbance/budget as the game card. Two adversary arms (per §4.1 decision):

**Gate 1a — Open-loop ε arm**
- Warmup: pure LQR; must match analytical LQR gain to predeclared tolerance.
- Minimax: trained with PGD on open-loop ε trajectories under the rollout-integrated L2 budget.
- Pass criteria (all conjunctive):
  - Gain matrix matches analytical H∞ gain within tolerance.
  - Nominal trajectories match within tolerance.
  - Cost time course matches.
  - Held-out PGD adversary loss matches.
  - Δv matches reference within declared band.

**Gate 1b — Closed-loop / state-feedback adversary arm**
- Same as 1a but trained against the Riccati-implied state-feedback disturbance (or evaluated against a direct closed-loop induced-gain computation).
- Pass criteria as above; *additionally* the closed-loop induced gain must match the analytical value.

**Gate-equivalence check**: do 1a and 1b yield the same controller within tolerance? If yes, the open-loop ε surrogate is certified as same-game on this plant. If no, that disagreement is itself a load-bearing result and must be reported.

**Frontier requirement**: run the gate at ≥3 γ values spanning the robust regime. A single-point pass is necessary but not sufficient. Report a γ-sweep of {Δv, gain norm, induced gain, control energy, terminal error}.

Failing Phase 1 stops the pipeline. Do not proceed to GRU training until the linear gate passes.

## 7. Phase 2 — Matched GRU under the certified game

Once Phase 1 passes, train a GRU under the identical plant/cost/disturbance/budget/evaluation as Phase 1. Same observation structure, same delay treatment, same ε channel and budget, same hold-out evaluation protocol.

Required reporting:
- Composite signature (§3) reported jointly, never as a single scalar.
- γ-sweep frontier (Phase 1 plot, with GRU overlaid).
- Local-Jacobian gain, standardized step-response gain, induced-gain — all three (§4.6).
- Cost decomposition, terminal error, control energy.
- Held-out adversary performance under both open-loop ε and state-dependent ε.

Interpretation rules:
- "Speed inflation present" + "induced gain improved" + "step-response gain increased" + "no pathological cost trade" ⇒ same-game signature reproduced.
- "Speed inflation present" + "induced gain not improved" ⇒ urgency / cost-misspecification artifact, *not* H∞.
- "Induced gain improved" + "speed inflation absent" ⇒ GRU dissociates speed from gain; substantive architectural result; report as such.

## 8. Phase 3 — Adversary-class contrast (matched conditions)

With plant, task, cost, architecture, budget, and evaluation held fixed, train arms differing only in **uncertainty class**:

- **A. Broad ε** (analytical-matched, Phase 2 result).
- **B. E1-like restricted fields** (curl interleaving, expected-cost).
- **C. E2-like restricted fields** (mixed perturbation menu, expected-cost).
- **D. State-multiplicative ΔA** (model-class perturbation).
- **E. Force-channel perturbations** (input-instance into actuator).

**Reporting object: full transfer matrix.** Every trained controller evaluated on every perturbation family above, plus held-out adversary searches. "Generalization" means measurable transfer (robustness to unseen families), not resemblance to a summary statistic.

This is the place where the "broad-defense prior" hypothesis is actually tested. Possible verdicts include:
- B/C alone reproduce the signature ⇒ broadening is not necessary.
- Only A reproduces the signature ⇒ broad adversary class is the load-bearing variable.
- All produce the same frontier under matched conditions ⇒ adversary class is *not* load-bearing; earlier negative-Δv from ΔA was about something else.

## 9. Phase 4 — Architecture ladder

Do not jump from time-varying linear regulator straight to full GRU as the only architectural contrast. Add intermediate models trained under the certified game:

- Affine feedforward trajectory + linear feedback.
- Constrained RNN / GRU with explicit feedback pathway.
- GRU with muscle-like force states / antagonist decomposition (where the plant supports it).

This distinguishes "GRU is a flexible universal approximator" from "production plant lacks biomechanical constraints that force speed/gain coupling."

## 10. Phase 5 — Trial-history / meta-policy  *(deferred)*

cs2019's central human result includes trial-history dynamics: a single perturbation increases next-trial speed/gain, decaying over ~10 unperturbed trials. The static-policy gate above intentionally does not model this. Add a later context-inference arm:

- Train/evaluate in block sequences with changing perturbation probability or latent uncertainty.
- Ask whether a single perturbation induces next-trial speed/gain increase + decay.
- Whether to train a context-conditional policy, a meta-policy, or to evaluate a fixed policy under sequence drift is itself a stop-and-ask decision.

This phase is not gating for the methodological paper. It is required only for the strongest version of the human-linking claim.

## 11. Falsification criteria

These are the observations that would constitute genuine pressure against the framing, not implementation bugs:

- **Falsifies "broad ε is model-matched"**: Phase 1 fails even after channel/state/norm/cost/delay/adversary are matched, or open-loop ε cannot reproduce the state-dependent Riccati game.
- **Falsifies "broad-adversary training is sufficient for GRUs"**: Phase 1 passes, Phase 2 GRU achieves low held-out induced gain, but robustly *dissociates* feedback gain from speed across seeds and γ.
- **Falsifies "broadening is necessary"**: Phase 3 arm B or C produces the same signature as arm A under matched conditions.
- **Falsifies "adversary class is load-bearing"**: All of A–E in Phase 3 produce the same frontier.
- **Falsifies "speed increase is robust-control-like"**: Phase 2 speed increases without induced-gain improvement or with pathological cost.
- **Falsifies the human-linking interpretation**: Phase 5 sequence model trained only on narrow physical perturbation history reproduces the trial-by-trial signature without broad-ε exposure.

Each falsification scenario has a defined action: report the result, update the umbrella body with the verdict, and re-plan downstream phases on the user's instruction. Do not patch and rerun without consent.

## 12. Code-review focus  *(deferred until Phase 0 exists)*

Code review is unproductive against an undefined target. Once the game card is materialized, narrow audit to:

1. Riccati solver and γ/γ* convention.
2. Feedbax disturbance injection and state-coordinate map.
3. Adversary inner loop, especially projection over rollout-integrated L2.
4. Evaluation code for induced gain, held-out adversaries, feedback Jacobians, Δv baselines.

## 13. Core subordinate issues (this umbrella's load-bearing children)

- `020a65b`
- `6ec6b19`
- `daa48c8`
- `1ad3c16`
- `63cec06`
- `8fcb6c7`
- `cf56e1e`
- `f7b1b17`
- `b41c940`

Each must carry a body-level back-reference to `43e8728` and align to the phase it serves (Phase 0–4 above). If a core issue's current scope predates this plan, comment on it with the phase mapping and any scope deltas; do not silently re-scope.

## 14. Supplementary / later issues

- `ac06736`
- `a5e1450`
- `31043a5`
- `a3edc0c`
- `6d62018`
- `65156e8`
- `297260c`
- `0af472c`

These are not blocking for the same-game linear gate or the matched GRU phase. They feed Phase 3 (adversary-class contrast), Phase 4 (architecture ladder), Phase 5 (trial-history), or analyses peripheral to the central question. They should be sequenced *after* Phase 1 passes; promoting any of them ahead of the linear gate requires explicit user consent.

## 15. Ops / hygiene / dependencies

- `fdad09d`
- `e75ddd7`
- `2ef67ca`
- `2092cb5`
- `6d5c906`
- `76d3a8e`
- `216b368`
- `a8ed10f`
- `3bd407b`
- `f7d40f1`
- `f350f58`

These are tooling, refactor, asset-strip, run-spec migration, and infrastructure items. They are not on the scientific critical path. Schedule opportunistically — particularly when an ops change reduces friction in the phase about to be executed. Do not block Phase 0 or Phase 1 on any of them unless one is shown to corrupt the game-card or adversary inner loop (in which case promote to blocking and flag the user).

## 16. Issue disposition notes

- **Core (§13)**: in-scope, phase-mapped. Update bodies to reference `43e8728` and the phase served. Where pre-existing implementation predates the game card, the issue's deliverable must be re-stated as "consistent with the game card" not "as previously implemented."

- **Supplementary (§14)**: in-scope but sequenced after Phase 1. Comment on each with "Deferred until Phase 1 pass; will be re-sequenced into Phase 3/4/5 per `43e8728` plan."

- **Ops / hygiene (§15)**: out-of-scientific-scope; opportunistic. No back-reference required. Promote to blocking only on demonstrated interference with the game card or training stack.

- **Superseded / historical as central drivers**: `35f64be`, `6f783fa`, `753508c`, `84ee4ff`, `ce34c2c`, `83fc5b5`.
  - These are no longer central drivers. Comment on each: "Superseded as central driver by `43e8728` (cs2019-to-RNN game equivalence umbrella). Retained as historical reference; do not treat v1/v2 framing as canonical." Do not close.
  - If any of these issues contain still-useful analysis (e.g. a specific diagnostic notebook, a useful figure), the analysis should be either (a) ported into a Phase-3/4/5 child issue with a clear scope statement, or (b) left in place with a "historical analysis, not in critical path" tag. Decide per-issue; do not bulk-port.

## 17. Reporting discipline

- Every Phase 1/2/3 result reports the **frontier** over γ (or ε-budget), not a single point.
- Every comparison reports the **composite signature** (§3), not Δv alone.
- Every Phase 3 result reports the **full transfer matrix** across perturbation families.
- Tolerances are predeclared in the game card; pass/fail logic is locked before any GRU output is inspected.
- All scoping language stays narrow: "broad-ε training is *sufficient* to induce the cs2019-like signature in this controller under this plant/cost" — never "humans inferred or optimized over full-state ε disturbances."

## 18. Agent operating rules under this umbrella

1. Do not begin Phase 1 training before Phase 0 game card exists and has been reviewed.
2. Do not interpret any GRU result before Phase 1 passes.
3. On encountering any of the §4 unresolved choices, stop, flag the user, request consent, and suggest strong-model (GPT-5.5 Pro, Gemini 2.5 Pro) feedback before resolving.
4. Do not silently re-interpret a subordinate issue under the new framing; comment on it with the re-scope and proceed.
5. Failures of pre-declared gates are reported and routed through this umbrella — no in-place patch-and-rerun without consent.
6. Frontier and composite-signature reporting are mandatory; single-point Δv claims are not acceptable outputs from this umbrella.
