# Claude-side synthesis (as delivered to user)

This is the summary Claude (Opus 4.7) delivered to the user upon receiving Codex's response. It restates Codex's conclusions in the user-facing form, with Claude's interpretation triage at the end. The verbatim Codex response is in `codex_response.md`; the prompt that produced it is in `prompt.md`.

## On the user's original concern — Codex's pushback, then validation

> "A fixed adversary at convergence is not, by itself, non-adversarial. In H-infinity/LQ games the controller also converges to a fixed policy against a fixed worst-case solution. C&S explicitly say the uncertainty is random across trials but not within a trial; the formal controller is robust to an unknown disturbance set, not to perpetual novelty.
>
> But the user's worry is well-founded **for the current GRU training pipeline**. The failure mode is not 'fixed adversary' in the abstract. It is: nonconvex RNN + one learned adversary per replicate + weakly verified PGD inner loop + no held-out worst-case audit can converge to a local co-adapted pair that is not robust to the intended disturbance class. The right test is not philosophical; freeze the controller, re-run strong independent adversary search with many restarts/step counts, and ask whether the saved adversary was actually worst-case. That makes `89891ab` a blocker for trusting minimax claims."

## Codex's central reframe

The load-bearing axis is **not** "saddle point vs co-adaptation." It's:

1. What uncertainty channel is being robustified?
2. What cost schedule makes speed useful rather than wasteful?
3. Is the policy forced to couple nominal movement speed to feedback gain, or can it decouple them?
4. Has the inner maximization actually found the worst case?

**Current `LinearDynamicsAdversary` (flavor-B) is a different game from cs2019 Eq 13.** C&S's analytical setup reads as full-state additive ε H∞, not flavor-B ΔA. The empirical −25% Δv from flavor-B is unsurprising: a state-multiplicative adversary makes "keep state small / move slower" optimal.

## Multi-signature prediction table

| Training regime | Speed Δv | Feedback-gain modulation | Why |
|---|---:|---:|---|
| Standard expected-cost + perturbation training | no / tiny | yes, modest | LQG-like; improves corrections without changing nominal |
| CVaR / APT / PAI-ASF | no / tiny | yes, **stronger** | Matches `844ef95` empirical observation |
| Gaussian bump minimax | unlikely | likely | Force adversary handled by corrective FB or memorized timing |
| **Current ΔA flavor-B** | **mixed / negative mean** | unclear / channel-specific | State-coupled adversary rewards small state excursions; bimodality real |
| LEQG alone | maybe small | yes | Better risk tilt, but only as good as disturbance channel |
| **Full-state ε H∞-matched** | **most likely yes** | **most likely yes** | Closest to C&S Eq. 13 + passing Riccati tests |
| H∞ Riccati teacher distillation | (highest existence check) | yes if architecture can express it | Decisive capacity check, not a training method per se |

Codex's verdict: **feedback-gain modulation is broad and easy; speed inflation is narrow** — it appears when robust-gain inflation is tied to nominal steering by the plant/cost/controller parameterisation. RNNs may learn tracker-like decompositions that preserve FB robustness while leaving nominal speed flat.

## Must-do — the single next experiment

**Implement the full-state ε adversary matching C&S Eq. 13**, with C&S-style `(t/N)^6` cost schedule + matched horizon/delay. **This folds `020a65b`, `89891ab`, `6ec6b19`, `daa48c8` into one decisive experiment.**

- Adversary: free time-varying full-state ε_t, L2-bounded over rollout
- Training: minimax, with PGD step sweep `{5, 20, 50}` and independent restarts
- Arms: **linear regulator first as a round-trip gate; one GRU arm only after the linear arm matches Riccati**
- Evaluation: **no group means alone** — replicate-conditioned Δv, feedback-gain/step-response, induced gain by channel, endpoint/lateral deviation, held-out adversary loss
- **Success**: linear regulator recovers Riccati's +Δv; GRU either also recovers OR cleanly dissociates gain from speed
- **Failure**: linear fails → training/eval pipeline is not the H∞ game; stop GRU interpretation until fixed

**Blocker**: cheap no-new-training audit `b6084c7` before launching compute.

## Deprioritise

- More flavor-B η sweeps
- Population adversaries (`--n-adversaries K > 1`)
- Regulator-vs-tracker architecture comparisons — **until the matched ε round-trip works**
- Muscle/biomechanics work
- LEQG-alone
- Broad reach-geometry sweeps
- Interpretability-first GRU decomposition

Codex's rationale: *"those become valuable after you have at least one trained GRU with a verified robustness signature. Right now the project needs a matched training game more than another explanation of an unmatched one."*

## Claude-side triage takeaways

1. **The user's original worry was real but mis-attributed**: not about "fixed = not adversarial," but about "we never verified the PGD inner loop actually finds the worst case." → `89891ab` is the load-bearing gate.
2. **The current LinearDynamicsAdversary is the wrong game** for inducing the C&S signature — by design, not by training failure.
3. **The work needed**: implement `020a65b` (full-state ε adversary), with `6ec6b19` (cost schedule) baked in, evaluated under `89891ab` (PGD audit), reported per-replicate (`daa48c8` framing). Linear regulator first as a Riccati round-trip gate. Only after that gate passes: GRU arm.
4. **What you don't have to worry about right now**: regulator-vs-tracker, more flavor-B sweeps, interpretability, biomechanics. All deferred.
5. **Cheap unblocker before compute**: `b6084c7` Tier-0 audit (~2h on existing artifacts).
