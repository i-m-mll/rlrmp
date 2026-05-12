# Synthesis — decoupling framing consultation (2026-05-11)

External consultations of the regulator/tracker decoupling puzzle (issue `d448c9d`) with Gemini 3.1 Pro and GPT-5.5 via codex. Codex was given a second round after the initial synthesis to critique the recommended path. This document captures the agreement, disagreement, and final recommended ordering.

## Glossary (brief)

- **Δv (peak velocity inflation)**: `(peak_v(arch_adversarial) - peak_v(arch_baseline)) / peak_v(arch_baseline)`, signed and projected on the reach axis. C&S 2019 predicts Δv > 0 as a robustness signature for analytical H∞ controllers.
- **Trivial x_nom**: when the tracker's nominal state is a *constant* (e.g. the target), the formula `u = u_ff(t) - K_t(x - x_nom)` collapses algebraically to `regulator + bias`. No structural decoupling possible.
- **Decoupling-via-parameterisation**: hypothesis that a tracker controller can increase feedback stiffness K_t for robustness without changing nominal motion, because u_ff(t) handles the nominal motion and K_t only acts on deviations from it.
- **Velocity deflation for safety**: alternative hypothesis (Gemini) that the GRU intentionally slows its nominal trajectory under adversarial training to leave actuator headroom — this would produce Δv < 0, not just Δv = 0.
- **Objective mismatch**: alternative hypothesis (Codex) that the GRU and the analytical Riccati simply optimize different cost functions, so their Δv signatures are not comparable. In particular: rlrmp's training adds hold/running/late losses, `nn_output`/`nn_hidden` regularization, catch trials, motor and sensory noise — none of which appear in the Riccati setup.
- **Adversary mismatch**: alternative hypothesis (Codex) that flavor-B's `ΔA·x` Frobenius-bounded adversary class is not equivalent to the analytical H∞ free additive disturbance with full-state `B_w`. Robustness to one ≠ matching the H∞ controller's behavior under the other.
- **Matched-objective ladder**: Codex's flagship recommendation — train every architecture (Riccati controller → trained LTV regulator → true LTV tracker → affine recurrent → GRU) under *identical* plant/cost/horizon/adversary/delay. Only then are architectural Δv differences interpretable.
- **Pareto diagnostic**: Codex's preferred primary plot — x-axis is nominal peak velocity or movement time, y-axis is a robustness metric (induced gain, worst-case adversarial loss, perturbation-rejection impulse response). Decoupling means "for equal or better robustness, the adversarial model does not need higher nominal peak velocity."

## Gemini's position (abstract)

The framing is fundamentally correct: Δv is the right behavioral signature for distinguishing a pure regulator from a decoupled tracker. The MVP's structural degeneracy (constant x_nom) explains why it failed to show the discriminator split. A minimum viable tracker must parameterize a time-varying nominal trajectory, with `u_ff(t)` computed by inverse dynamics. The recommended path is a 3-phase plan: (A) prove decoupling is mechanically possible by training a frozen-trajectory tracker, (B) prove the optimizer can find decoupling end-to-end by training a learnable tracker warm-started from the teacher, (C) directly test the hypothesis by linearizing trained GRUs around their nominal trajectories and comparing K_local and u_ff between baseline and adversarial. Gemini additionally suggests three explanations for the GRU's *negative* Δv (rather than just neutral): velocity deflation for safety, observer-gain shift, and bimodality across two local minima.

## Codex's position (abstract)

The framing is too narrow. Δv is one point on a Pareto diagnostic, not "proof of decoupling." The GRU result should be interpreted as "the GRU is not converging to the same H∞ optimum as the Riccati/linear-controller game" — tracker decoupling is one candidate mechanism, but cost mismatch, adversary mismatch, and optimizer basin effects are at least as plausible. The 30× discrepancy between the trained LTV regulator (Δv = +78% from the MVP) and analytical Riccati (Δv = +1-27% on the same regime) is a louder alarm than the tracker/regulator story and must be closed first. Before any GRU representational analysis, run a trajectory and cost audit across all four "controllers" (baseline GRU, flavor-B GRU, trained linear, Riccati) to surface the unmatched axes. Then the matched-objective ladder. Direction-dependent `u_ff(t, target)` is essential for center-out; a single world-frame `u_ff(t)` cannot represent direction-specific feedforward.

## Disagreement (short)

Gemini accepts the regulator-vs-tracker framing and prescribes architecture engineering. Codex argues the framing may itself be premature — until plant/cost/horizon/adversary/delay are matched across all controllers, no architectural comparison is interpretable. Both converge on the same concrete experiments; they diverge on *ordering* and *what counts as conclusive evidence*.

## Codex's second-opinion update

After receiving my initial synthesis (which proposed GRU affine decomposition as the first move), Codex pushed back: (1) GRU affine decomposition is observational and local — it tells you whether the GRU "looks tracker-like" near its nominal trajectory, not whether tracker structure *caused* Δv decoupling. (2) The nominal/feedback power split is ill-defined on the undisturbed nominal trajectory because the local feedback correction is zero by construction — must measure under standardized perturbations or worst-case w*. (3) `induced_gain.py` is useful but not a complete GRU affine-decomposition tool; a new wrapper exposing `u_nominal`, `∂u/∂x_plant`, `∂u/∂h`, residual nonlinearity, and perturbation-response decompositions is needed. (4) Most importantly: promote the Riccati round-trip to the *first decisive move*. If trained LTV regulator still gives +78% where Riccati gives +1-27% under matched plant/cost/horizon, **stop doing architecture comparisons** — the training objective/evaluation pipeline is the puzzle.

I accept this update. The ordering below reflects it.

## Recommended path (final)

**Tier 0 — Trajectory and cost audit** — `b6084c7`
Cheapest first check. For each existing checkpoint (baseline GRU, flavor-B GRU, trained linear regulator, analytical Riccati): nominal movement time, peak velocity, state-cost time course, control-cost time course, adversary loss vs Riccati `w → z` loss, exact delay/noise/catch-trial/task-input differences. Surface the unmatched axes before doing anything else.

**Tier 1 — Matched-objective Riccati round-trip** — sub-item 2.2 of `66a3374`
Single highest-priority experiment. Make the trained LTV regulator reproduce the analytical Riccati Δv (+1-27% target). If it gives +78% under the matched objective, stop. Architectural comparisons are meaningless until this closes.

**Tier 2 — GRU affine decomposition + bimodality clustering** — `0c95d6b`
Now demoted to "useful diagnostic" rather than "highest leverage." Linearize baseline and flavor-B GRUs around their nominal trajectories; compute `u_nominal`, `∂u/∂x_plant`, `∂u/∂h`, residual nonlinearity, perturbation-response decompositions (under standardized perturbations, NOT on undisturbed trajectories). Cluster bimodal flavor-B replicates by features. Likely needs a new analysis wrapper rather than just `induced_gain.py`.

**Tier 3 — Fixed-direction matched-objective sweep with tracker variants** — sub-item 2.1 of `66a3374`
Linear architecture engineering: regulator + true LTV tracker (non-trivial x_nom) + K-frozen tracker + u_ff-frozen tracker, all under matched objective. Direction-dependent `u_ff(t, target)` is required if generalising beyond fixed direction.

**Tier 4 — GRU under matched objective** — `0083c5f`
Conclusive architecture test. Train a GRU with the exact same matched plant/cost/horizon/adversary/delay as Tier 3. If GRU still gives Δv ≈ 0 while linear gives Δv > 0 under matched objective, decoupling-via-architecture is the explanation. If GRU also gives Δv > 0, the original empirical gap was driven by objective mismatch, not architecture.

## Confounds and gotchas flagged

- **Direction-dependent u_ff** is essential for center-out reaching; a single world-frame `u_ff(t)` is structurally insufficient.
- **Nominal/feedback power split** must be measured under standardized perturbations or worst-case `w*`, never on the undisturbed trajectory.
- **Delay matters**: a delayed GRU tracker may solve a fundamentally different problem than full-state Riccati even with nominally identical costs.
- **Mixing notions of robustness**: empirical adversary rollouts vs analytical induced norms are not interchangeable. Pick one and use it consistently across the matched-objective ladder.
- **Bimodality of flavor-B**: averages across replicates are misleading. Use replicate-level analysis throughout. Do not overfit narratives to PCA clusters unless the positive-Δv group also differs on loss/robustness/Jacobian/trajectory metrics.

## Literature pointers (from the consultations)

- C&S 2019 (analytical H∞ velocity inflation): PMC10.1152/jn.00031.2019, ModelDB code at model 258846.
- Todorov & Jordan 2002 (optimal feedback control; "not necessarily tracking a desired trajectory" caution).
- DGKF / standard H∞ state-space theory (Doyle, Glover, Khargonekar, Francis 1989).
- Mastrogiuseppe & Ostojic 2018 (low-rank RNN dynamics — useful for interpreting GRU solution classes rather than treating the 180D state opaquely).
- Anderson & Moore, *Optimal Control* (Gemini's reference for LQT vs Regulator formulation distinction).

## Open questions to revisit after Tier 0–1

- Is the negative Δv of flavor-B GRUs an artifact of objective mismatch, or a genuine "velocity deflation for safety" signature?
- Is the flavor-B bimodality an attractor-class story (slow-conservative vs H∞-like) or an artifact of unmatched evaluation geometry / adversary?
- Does the trained linear regulator under matched objective recover the analytical Δv sign and magnitude, or does it stay at +78%?
- If the analytical bridge for the rlrmp cost (`cea2e62`) is done, do its predictions match the trained linear regulator under matched objective?
