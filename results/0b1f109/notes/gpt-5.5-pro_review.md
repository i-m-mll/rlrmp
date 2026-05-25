External review of the cs2019-to-RNN robustness plan

1. Overall verdict

The plan is scientifically coherent and substantially stronger than the broader motivating synthesis. Its best feature is that it refuses to interpret GRU behavior until the training pipeline has passed a same-game linear round trip against an analytical Riccati target. That is the right load-bearing move. The packet correctly identifies the main danger: comparing “robust” controllers trained under different plants, costs, horizons, delays, disturbance channels, and evaluation protocols can create plausible-looking but uninterpretable Δv stories. The proposed audit, analytical target materialization, linear gate, matched GRU test, and staged bridge are all methodologically sensible.  

The main weakness is that the current “broad epsilon” training game is still not yet guaranteed to be the same game as the cs2019 analytical H-control model. In particular, the packet itself flags the important open question: a practical optimized per-timestep epsilon trajectory may not be equivalent to the state-dependent worst-case disturbance strategy implied by the Riccati game. If that mismatch is unresolved, the linear gate can fail for a deep formal reason, not just because the implementation is wrong.  

My strongest recommendation is to make the first experiment a game-equivalence experiment, not merely a behavioral reproduction experiment. The question should be: Can the rlrmp/feedbax training and evaluation stack reproduce the same closed-loop input-output game solved by the finite-horizon H∞ Riccati recursion? Only after that is established should the project ask whether a GRU reproduces or dissociates the speed/gain phenotype.

2. Best-case interpretation of the plan

The strongest version of the plan is not “train an RNN and see whether it looks like humans.” It is:

Given a plant, cost, horizon, observation structure, delay treatment, and disturbance channel for which a finite-horizon robust-control solution predicts increased nominal movement speed and increased perturbation-response gain, test whether the same formal game induces the same behavioral coupling in a flexible trained recurrent controller. Then contrast that with controllers trained on narrower, human-protocol-like perturbation distributions under otherwise matched conditions.

That is a clean scientific question. It separates three things that are too often conflated:

1. Analytical sufficiency: Does the H∞/broad-epsilon linear game produce the cs2019-like speed/gain signature on the specified plant and cost?
2. Training-stack validity: Can gradient-based minimax training recover that known game in the linear case?
3. Architectural effect: Once the game is certified, does a GRU still couple speed and feedback gain, or does it solve the game by a behaviorally dissociated strategy?

The packet already contains this structure: first certify the broad epsilon game against an analytical target, then train the GRU under the same certified game, then compare broad-epsilon training against restricted physical-field distributional training with plant/task/cost/architecture/training/evaluation held fixed.  

That is the strongest framing because it does not require the risky claim that humans explicitly solve an H∞ game. It only claims that humans may behave as if they have access to a broad-defense policy, and that a GRU trained only on a narrower perturbation distribution may not acquire that policy unless the broader adversary class is supplied.  

3. What cs2019 actually gives you

The cs2019 paper supports a composite target, not a single scalar target.

In Experiment 1, participants made 15 cm forward reaches with 0.6–0.8 s arrival-time feedback; during the peri-exposure phase, 50 null-field trials and 10 curl-field trials per block were randomly interleaved, with clockwise and counterclockwise curl fields.   The clearest block-level velocity effect is in this experiment: peri-exposure peak forward velocity increased relative to pre/post, and trial-by-trial analysis showed that a single force-field perturbation was followed by increased forward velocity that decayed over roughly ten undisturbed trials.  

Experiment 2 is different and should not be collapsed into “curl-field exposure.” It included step loads, curl fields, and orthogonal velocity-dependent fields; the step loads were used to probe feedback responses and EMG.   The phase-level velocity contrast was weaker or absent in Experiment 2, but trial-history analysis still showed velocity modulation and, more importantly, stronger perturbation responses: trials classified as “Early” after a perturbation had higher forward velocity at perturbation onset and smaller lateral displacement after the same step load.    

The model prediction in cs2019 is also composite. The robust controller had larger feedback/control gains; because the linear controller uses the same gains to steer the limb and respond to perturbations, this produced both faster nominal forward velocity and more vigorous perturbation responses.   The discussion explicitly frames the human data as evidence for a robust control strategy that is dissociable from pure impedance/co-contraction stiffening, while acknowledging that LQG and robust control are both state-feedback control laws and that one could fit some gain modulation by changing LQG costs.  

So the target should not be “positive Δv.” It should be a conjunction:

* nominal speed/trajectory change;
* feedback-response gain increase under standardized perturbations;
* lower perturbation-induced displacement or induced gain;
* no pathological terminal-error or cost tradeoff;
* preferably a model-appropriate co-contraction or antagonist-force analog when the plant/controller has variables that make that meaningful.

This matches the packet’s proposed evaluation suite and is exactly the right instinct.  

4. Major formal and methodological risks

Risk 1: open-loop epsilon trajectory is not obviously the same as the H∞ game

The most important unresolved formal issue is the adversary. The packet proposes a per-timestep epsilon disturbance with a rollout-integrated L2 budget and global projection over the whole flattened trajectory. That is a reasonable training object. But the H∞ solution is usually expressed as a dynamic game whose worst-case disturbance depends on the current state and the Riccati value matrix. The packet explicitly notes this tension.  

A fixed optimized epsilon trajectory can be a valid adversary for one nominal rollout, but it may not certify the same closed-loop disturbance-to-cost operator. In a linear deterministic single-trajectory setting, an open-loop disturbance sequence can reproduce some worst-case behavior, but a state-dependent adversary is the more faithful object when the controller is evaluated under perturbations, delays, state deviations, target variations, or nonlinear policies.

For the linear gate, I would not accept “strong PGD on epsilon trajectories” as sufficient by itself. I would add either:

1. an analytical state-feedback worst-case disturbance policy derived from the same Riccati recursion, or
2. a direct dynamic-programming / closed-loop induced-gain computation against the trained linear controller.

If the open-loop PGD adversary and the state-dependent H∞ adversary disagree materially, that is not an implementation detail. It means the training game is not the analytical game.

Risk 2: broad additive epsilon may be model-matched, but it is not the human perturbation protocol

The packet handles this well. Humans were exposed to restricted physical fields and step loads, not full-state arbitrary epsilon disturbances. The broad-epsilon arm is model-matched to the analytical robust-control reference; the restricted-field arm is human-protocol-like. Those should not be conflated.  

The practical implication is that “broad epsilon reproduces cs2019” would mean:

This broad adversary class is sufficient to induce the cs2019-like signature in this trained controller under this plant/cost/task.

It would not mean:

Humans inferred or optimized over full-state epsilon disturbances.

The packet already states that narrower interpretation, and it should remain the published framing.  

Risk 3: the six-state simplification is defensible only as a demonstrated equivalence, not as a convenience

The eight-state-to-six-state reduction could be fine, but it is not a harmless implementation simplification. The plan says the eight-state cs2019-faithful analytical target should be materialized first, then the Riccati should be rerun on the six-state variant, and only then should equivalence be claimed for the relevant H∞ quantities. That is exactly the right standard.  

The equivalence demonstration should be done across more than one gamma/budget setting. If the dropped disturbance-mediator/integrator states are inert only near one chosen gamma, the simplification is fragile. It should also be checked separately for the broad-epsilon analytical gate and the restricted physical-field contrast, because a state that is irrelevant to full-state epsilon attack might still matter for velocity-dependent physical fields.

Risk 4: speed inflation is not diagnostic by itself

The packet is right that speed increase can arise from robust control, urgency, cost misspecification, endpoint tolerance, overshoot tolerance, unstable dynamics, or optimization pathology.   This is not a minor caution. In these tasks, the time-varying state cost, arrival-time constraint, terminal penalty, control penalty, and delay compensation can all change the movement-speed optimum.

A successful model must show speed inflation and a robustness improvement against relevant disturbances, with sane trajectories and cost decomposition. A model that moves faster but has worse held-out induced gain has not reproduced the H∞ signature.

Risk 5: the human result includes trial-history dynamics; the first gate does not

The narrow movement-only gate is appropriate, but it does not model the adaptive/contextual part of cs2019: a perturbation changes the next trial’s policy and that policy decays over subsequent unperturbed trials. cs2019’s trial-history result is central, especially because Experiment 2’s block-level velocity effect is not the cleanest signature.    

This is not a flaw in the gate, as long as the gate is described as a policy-form validation, not a full human-protocol model. Later, if the project wants to claim closer contact with the human data, it needs a context-inference or meta-policy arm: train/evaluate controllers over trial sequences with changing perturbation probability or latent uncertainty, and ask whether a single perturbation induces the next-trial speed/gain increase and decay.

5. Is the cs2019-to-RNN mapping well posed?

Yes, but only in a layered sense.

The well-posed version is:

Given a cs2019-inspired plant and cost whose analytical robust controller has a known speed/gain signature, can a trained controller recover, alter, or dissociate that signature under the same formal game?

That is well posed.

The less well-posed version would be:

Can a GRU trained with an adversary reproduce human behavior in cs2019?

That is too broad unless the training protocol includes trial history, uncertainty inference, physical perturbation distributions, catch-trial structure, and the arrival-time/feedback constraints of the human task.

The packet’s staged bridge is therefore a strength. The initial gate is intentionally movement-only, single-reach, no hold epoch, no catch trials, no go cue, no mixed targets, and same observation/information structure as the analytical target.   That narrowness is not a problem; it is what makes the first result interpretable. But the claims should be scoped to that narrowness.

6. Is restricted-field versus broad-epsilon the right contrast?

Yes, with one adjustment: use a transfer matrix, not a single contrast.

The clean contrast should hold plant, task, cost, architecture, budget, and evaluation fixed while changing only the uncertainty class. The packet states this correctly.   But “restricted-field” should probably not be one arm. I would use at least two restricted-field baselines:

1. Experiment-1-like restricted distribution: randomly interleaved CW/CCW curl fields, expected-cost or domain-randomized training.
2. Experiment-2-like mixed distribution: step loads + curl fields + orthogonal velocity-dependent fields, again under expected cost unless the explicit question is worst-case restricted-field control.

A worst-case restricted-field arm is useful, but it answers a different question: “What if the same physical fields are adversarially selected?” That should not be presented as the human-protocol-like baseline unless the paper explicitly frames it that way. The packet already flags this distinction.  

A decisive design would evaluate every trained controller on all major perturbation families:

* clean nominal reach;
* step-load feedback response;
* curl fields;
* orthogonal velocity-dependent fields;
* broad epsilon induced gain;
* state-multiplicative / ΔA-style perturbations;
* held-out adversary searches.

Then report the whole transfer matrix. “Generalization” should mean measurable transfer: robustness to perturbation families not used in training, not merely resemblance to a human summary statistic. The packet also notes that generalization needs an observable definition, likely robustness transfer to unseen perturbation families.  

7. Is the linear same-game gate sufficient?

It is necessary and almost sufficient for validating the training stack, but not sufficient for the scientific interpretation by itself.

The proposed gate is strong: LQR warmup must match the analytical LQR; adversarial training must match the analytical H∞ target; evaluation must include gain matrix match or equivalent representation, nominal trajectory, cost time course, induced gain, held-out adversary loss, and Δv within a declared band.   Treating this as conjunctive is correct. A controller that matches Δv but fails gain, trajectory, or robustness checks should fail the gate.  

But I would add three criteria.

First, the gate should include a closed-loop adversary equivalence check, not only held-out PGD on the same open-loop epsilon parameterization. For the linear system, compute the analytical closed-loop worst-case disturbance or induced gain directly.

Second, tolerances should be set from numerical and optimization noise after target materialization, but the pass/fail logic should be predeclared before looking at GRU outcomes. The packet already proposes declaring tolerances after the target artifact establishes scale/noise.  

Third, the gate should sweep gamma or epsilon budget. A single value such as 1.5×gamma-star is useful, but a monotone robustness frontier is more diagnostic. If the trained linear controller matches one point but not the frontier, it may be exploiting the training setup rather than recovering the game.

8. Suggested modifications to make the plan more decisive

Modification A: materialize an explicit “game card”

Before training, produce a compact artifact with the exact equations:

* state vector, including delay augmentation;
* discrete-time convention;
* cost schedule and terminal/running penalties;
* control penalty scaling;
* disturbance channel B_w;
* epsilon norm and time integration convention;
* gamma-star and chosen gamma;
* LQR and H∞ gains;
* nominal trajectories;
* analytical worst-case disturbance characterization;
* induced gain and Δv.

The packet already lists most of this target materialization.   The value of a “game card” is that every later comparison can be audited against one object.

Modification B: split the broad-epsilon gate into open-loop and feedback adversary variants

Run the initial linear gate twice:

1. Open-loop trajectory epsilon: the practical PGD surrogate currently planned.
2. State-dependent H∞ adversary: the Riccati-implied disturbance policy or equivalent induced-gain computation.

Passing both is a strong certificate. Passing only the open-loop gate is not enough. Failing only the open-loop gate but passing the analytical induced-gain check would suggest the training parameterization is insufficient, not that the robust-control target is wrong.

Modification C: report a robustness-speed-gain frontier

Do not rely on one gamma. Sweep gamma or epsilon budget and plot:

* nominal Δv;
* feedback gain / standardized step response;
* induced gain;
* nominal cost;
* terminal error;
* control energy.

For a true H∞-like family, the interesting object is the frontier, not a single point. This also makes the GRU result easier to interpret: a GRU that lacks speed inflation at one budget may still lie on the same frontier in a different region, or may define a different frontier entirely.

Modification D: add a minimal architecture ladder

After the linear gate, do not jump directly from time-varying linear regulator to full GRU as the only architectural contrast. Add one or two intermediate models:

* affine feedforward trajectory + linear feedback;
* constrained RNN/GRU with explicit feedback pathway;
* GRU with or without muscle-like force states / antagonist decomposition.

This would help distinguish “GRUs can decouple because they are nonlinear universal function approximators” from “the production plant lacks biomechanical constraints that force speed/gain coupling.”

Modification E: add a later trial-history/meta-policy experiment

Once the static policy gate is done, train or evaluate a controller in block sequences with perturbation probability/hazard and history. The human result is not merely that a robust policy exists; it is that unexpected perturbations transiently shift control strategy on subsequent trials. cs2019 explicitly reports immediate next-trial increases and gradual decay over unperturbed trials.  

That later experiment would ask whether the model learns a policy-switching rule rather than just a robust policy.

9. What would falsify the current framing?

Here are the observations I would treat as genuine pressure against the framing, not merely implementation bugs.

Falsifies “broad epsilon is model-matched to the analytical target”: the linear same-game gate fails even after the disturbance channel, state vector, norm, cost, delay, and adversary form are matched, or the open-loop epsilon surrogate cannot reproduce the state-dependent Riccati game. That would mean the proposed training game is not the cs2019 analytical game.

Falsifies “broad adversary-class training is sufficient to induce the cs2019-like signature in GRUs”: the linear gate passes, the GRU demonstrably minimizes the same broad-epsilon robust objective and achieves low held-out induced gain, but it robustly dissociates feedback gain from nominal speed across seeds and gamma values. That would be a substantive result: flexible recurrent architectures can solve the broad game without the human-like speed/gain coupling.

Falsifies “broadening is necessary relative to restricted fields”: restricted-field expected-cost training, under matched plant/task/cost/architecture/training/evaluation, yields the same speed/gain/robustness signature as broad epsilon. Then the broad-defense-prior story is not needed for the model.

Falsifies “adversary class is the load-bearing variable”: broad epsilon, restricted fields, state-multiplicative ΔA, and force-channel perturbations all produce the same frontier once cost, delay, budget, and architecture are controlled. Then the earlier negative Δv from the state-multiplicative adversary was probably not about uncertainty class.

Falsifies “speed increase is robust-control-like”: speed increases without improved held-out induced gain, without stronger standardized perturbation responses, or with pathological endpoint/cost trajectories. That would be an urgency/cost/optimization artifact, not a robust-control signature.

Falsifies the human-linking interpretation: a human-protocol-like sequence model trained only on narrow physical perturbation history reproduces the trial-by-trial cs2019 phenotype without broad-epsilon exposure, especially if it also transfers well to held-out perturbation families. That would not invalidate the linear H∞ result, but it would weaken the claim that broad adversary-class training is needed to explain the human-like phenotype.

10. Separate notes on synthesis-5.md

The synthesis is useful background, but the packet is more careful and should supersede it for the experimental plan.

The strongest part of synthesis-5.md is the insistence that “adversarial” is not one operation. It distinguishes input-instance perturbations, model/structural perturbations, and risk-sensitive/LEQG-style distributional tilts, and it emphasizes that these routes can produce different controllers and guarantees.   That conceptual move directly supports the packet’s refusal to treat state-multiplicative ΔA, broad additive epsilon, force-channel perturbations, and restricted physical fields as interchangeable.  

The biggest tension is that the synthesis often labels the motor-control case as straightforwardly “flavor-b / model-class / structural,” while the packet’s current target is a broad additive epsilon disturbance entering the state, chosen as the model-matched H∞ training game. The cs2019 model does motivate the disturbance by grouping fixed model errors such as force-field terms with process noise, but the discrete robust-control problem is expressed with an unknown additive disturbance term in the state update.     The packet’s narrower “broad additive epsilon” wording is therefore safer than the synthesis’s stronger “this is model-class perturbation” language.

The statement in the synthesis that input-instance attacks are a strict subset of model-class perturbations is too broad unless the disturbance sets, channels, budgets, and outputs are defined so that the broader game literally contains the narrower one.   A controller robust to a structural-uncertainty family is not automatically robust to every additive input disturbance unless the structural family includes the same additive channel with comparable budget and performance output. The subset claim can be retained as a useful schematic only after adding those qualifications.

The LEQG/H∞ connection should also be scoped more tightly. The synthesis correctly notes that the cost-distribution variance term is not itself a Hessian, and that the Hessian enters through the Riccati/value function in the LQ slice.   But claims like “humans move along the Whittle axis” should be softened. cs2019 supports behavior consistent with robust-control predictions; it does not prove that humans transition along a Whittle one-parameter family.  

The Goodhart, trauma, and prior-precision material is much more speculative than the motor-control plan. Some of it is framed as conceptual bridge or hypothesis, which helps, but claims like “no separate trauma circuit” or “Goodhart is mostly flavor-b” are not needed for the rlrmp experiment and could distract reviewers.   I would quarantine that material from the methodological paper unless the project explicitly becomes a cross-domain theory paper.

Finally, the synthesis’s “natural experiment” claim—cs2019 versus prior rlrmp additive-adversary failures—is rhetorically useful but not yet decisive. The packet’s planned matched contrast is the correct way to turn that intuition into evidence. Until then, “no amount of flavor-a training can produce the signature” is too strong. The current plan should instead test whether broad-epsilon training is sufficient and whether restricted-field/domain-randomized training is insufficient under matched conditions.    

11. Concrete questions to answer before implementation

1. What is the exact discrete-time state vector for the eight-state, six-state, and delay-augmented analytical targets?
2. What is the exact disturbance channel B_w, and does epsilon enter positions, velocities, force states, all physical states, or a weighted subset?
3. Is the training adversary an open-loop epsilon trajectory, a state-dependent adversary, or both? If only open-loop, why is that equivalent to the Riccati game?
4. How is gamma translated into an epsilon budget for PGD training, and does the norm include timestep scaling?
5. What are the pass/fail tolerances for gain, trajectory, cost, induced gain, Δv, and held-out adversary loss?
6. Which restricted-field baseline is being modeled: Experiment 1 curl-only, Experiment 2 mixed perturbations, or both?
7. Is restricted-field training expected-cost/domain randomization, minimax over restricted fields, or both as separate arms?
8. How will feedback gain be measured for nonlinear GRUs—local Jacobians, standardized step responses, induced-gain estimates, or all of these?
9. What is the definition of generalization: transfer to unseen physical fields, transfer to full-state epsilon, transfer to state-multiplicative ΔA, or a full cross-evaluation matrix?
10. What observation would make the project abandon the broad-defense-prior interpretation rather than add another architectural or training fix?

12. Where source-code review would matter

I agree with the prompt that this does not yet need a full implementation audit. If code review becomes useful, it should focus narrowly on four places:

1. the Riccati solver and gamma/gamma-star convention;
2. the feedbax disturbance injection and state-coordinate map;
3. the adversary inner loop, especially projection over rollout-integrated L2 norm;
4. evaluation code for induced gain, held-out adversaries, feedback Jacobians, and Δv baselines.

That code review should happen after the analytical “game card” exists; otherwise implementation details will be hard to judge against the intended formal target.