# GPT 5.5 Pro Review: Phase 3 Linear Certificate

Imported from `/Users/mll/Downloads/Untitled.md` on 2026-05-28.

Context: external GPT 5.5 Pro analysis of the Phase 3 review packet for
`6f5c79e`, focused on gain recovery, clean objective/behavioral recovery, and
what should constitute a disturbance-relevant linear equivalence certificate.
Follow-up implementation issue: `d01c35a`.

According to the attached Phase 3 archive from May 28, 2026, my bottom line is: do not let this checkpoint pass yet, but also do not make raw Frobenius gain recovery the law of the land. The current result should be classified as “clean behavioral/objective recovery achieved; disturbance-relevant closed-loop equivalence not yet certified.” The attached archive is here for reference: rlrmp_phase3_gain_review_packet_20260528.zip. The earlier plan was right that the linear gate should be a same-game certificate, not merely positive Δv or clean trajectory resemblance; it explicitly allowed either gain-matrix match or an explainable equivalent representation, but required trajectory, cost, induced-gain, and held-out-adversary checks as part of the certificate.

1. Bottom-line recommendation

Treat the current Phase 3 result as a failed exit certificate, but a successful diagnostic.

The key facts from the Phase 3 bundle are:

Controller	Clean result	Raw gain error	Held-out adversary audit
Analytical LQR	clean cost ≈ 4288.75	0	held-out cost ≈ 4789.86
Teacher LQR fit	clean cost ≈ 4288.75	≈ 1.2e-16	held-out cost ≈ 4784.08
Adam + L-BFGS-B LQR fit	clean cost ≈ 4290.39	≈ 0.989	held-out cost ≈ 5266.18

So the L-BFGS-B-refined controller is almost indistinguishable on the clean canonical reach, but it is not equivalent off trajectory. Its held-out adversary cost is roughly 10% higher than the teacher/analytical LQR audit. That makes this more than a cosmetic gain-metric issue.

The right reading is:

The current objective-trained controller has learned a clean-reach-equivalent policy, not the Riccati feedback law, and not yet a disturbance-relevant equivalent of that law.

That means: do not proceed to GRU/RNN same-game interpretation under the current certificate. But also: do not spend the next week simply trying to beat raw ||K_fit - K_ref||_F / ||K_ref||_F below 0.05 unless you first decide that exact Riccati gain recovery is the scientific target.

It probably should not be the target.

2. Classification of the evidence

Not a representation/plumbing failure

The teacher-fit result is decisive on this narrow point. The same tensor shape can hold the analytical LQR and H∞ gains to numerical precision. So the gain mismatch is not because the parameterization cannot represent the Riccati controller, nor because the reporting metric is pointed at the wrong array.

Not a pure optimizer failure anymore

Adam alone failing could have been an optimizer story. But Adam-warm-started L-BFGS-B gets the clean weighted objective ratio to about 1.0088 and clean behavior almost exactly back to analytical LQR, while the gain error remains ≈0.989. That pattern is not “the optimizer simply did nothing.” It says the objective has found a low-cost basin far from the raw Riccati gain tensor.

However, L-BFGS-B stopped at the iteration limit, so optimizer failure is not ruled out. You still need gradient norm, longer runs, and restarts. But the result has moved from blocked_on_optimizer to something more subtle: blocked_on_identifiability/certificate.

Strong evidence for weak identifiability under the current rollout objective

The current clean objective is an empirical rollout objective over a finite deterministic initial-state ensemble. That is not the same object as the Riccati dynamic-programming solution “for all states at all times.” It can make many columns or time slices of K_t weakly identified, especially with:

* a 48D delay-augmented state;
* finite initial-state support;
* delay-lag states that may wash out or become constrained to a reachable manifold;
* a cost schedule dominated by late movement/terminal terms;
* canonical reach overweighting;
* raw state coordinates with different units and scales.

Even with positive and negative basis states at t=0, this does not guarantee that every K_t column is strongly identified at every later time.

But not harmless non-identifiability

If all the gain mismatch lived in genuinely irrelevant null directions, the held-out adversary audit should look like analytical/teacher LQR. It does not. The L-BFGS-B controller’s clean behavior matches, but its adversarial audit differs substantially.

So the current result is best classified as:

Weakly identified / ill-conditioned clean objective, plus an insufficiently targeted certificate, with residual optimizer uncertainty.

Not “metric problem only.” Not “optimizer problem only.”

3. Why raw Frobenius gain error is the wrong central certificate

For finite-horizon LQR, the Riccati feedback matrix is meaningful. But raw Frobenius error over the whole tensor,

\frac{\|K_{\text{fit}} - K_{\text{ref}}\|_F}{\|K_{\text{ref}}\|_F},

is a poor scientific certificate by itself.

It is not invariant to coordinate scaling. If you rescale velocity, force, or delay-lag coordinates, the Frobenius error changes even if the closed-loop input-output behavior does not. It also weights unvisited, weakly visited, and scientifically important state directions equally. In a 48D delayed system, that is almost guaranteed to be misleading.

The better object is not K in isolation. It is the action and closed-loop map induced by K on the state distributions that matter.

The useful LQR identity is the performance-difference form:

J(K) - J(K^*) \approx
\sum_t
\mathbb{E}_{x_t}
\left[
\big((K_t - K_t^*)x_t\big)^\top
H_t
\big((K_t - K_t^*)x_t\big)
\right],

where

H_t = R_t + B^\top P^*_{t+1}B.

This says the meaningful error is not the raw matrix difference. It is the state-weighted, control-cost-weighted action error:

(K_t - K_t^*)x_t.

A high Frobenius error can be irrelevant if it lives in state directions never reached or never adversarially excited. But a small-looking behavioral error can be dangerous if the mismatch lives in directions reached by perturbations. Your held-out adversary audit suggests the latter may be happening.

4. Why clean behavioral replication is also too weak

Clean canonical behavior is necessary, but it is not the point of the Phase 3 gate. The point of the gate is to prevent exactly this failure mode: a controller that looks right on the nominal reach but has the wrong off-trajectory sensitivity. The previous review packet explicitly warned that speed increase or nominal behavior alone can be caused by robust control, urgency, cost misspecification, overshoot tolerance, unstable dynamics, or optimization artifacts; it should not be interpreted alone.

For this project, the relevant scientific object is feedback sensitivity / robustness, because the C&S reference result is not just “movement speed changed.” The robust-control model predicts faster movements and stronger feedback responses, and the empirical paper emphasizes the speed-plus-feedback-gain signature.

So if a trained linear controller matches clean movement but has a different perturbation-response map, it has not passed the spirit of the same-game gate.

5. The diagnostic package I would run next

A. State-weighted action mismatch

For each controller and each time step, compute:

E^{u}_t =
\frac{
\mathbb{E}_{x \sim \mathcal{D}_t}
\| (K_t^{\text{fit}} - K_t^{\text{ref}})x \|^2_{R_t}
}{
\mathbb{E}_{x \sim \mathcal{D}_t}
\| K_t^{\text{ref}}x \|^2_{R_t}
}.

Do this for several \mathcal{D}_t:

1. clean canonical trajectory plus small perturbations;
2. current full-rank training ensemble trajectories;
3. independent validation ensemble trajectories;
4. Riccati worst-case / held-out adversary-induced states;
5. per-time isotropic or coordinate-normalized state probes.

If the raw K error is high but E^u_t is tiny on clean and adversarial states, raw Frobenius is too strict. If E^u_t is large on adversarial states, the controller is not equivalent.

Given the held-out audit, I expect the latter.

B. Closed-loop transition mismatch

Compare:

M_t^{\text{fit}} = A - BK_t^{\text{fit}}

to

M_t^{\text{ref}} = A - BK_t^{\text{ref}}.

But again, do not use unweighted Frobenius as the main score. Use:

\frac{
\mathbb{E}_{x \sim \mathcal{D}_t}
\| (M_t^{\text{fit}} - M_t^{\text{ref}})x \|^2
}{
\mathbb{E}_{x \sim \mathcal{D}_t}
\| M_t^{\text{ref}}x \|^2
}.

This directly asks whether the fitted controller induces the same local state update.

C. Policy-evaluation value matrices

For the fitted controller, compute its exact finite-horizon policy-evaluation matrices:

P_T^K = Q_f,

P_t^K =
Q_t + K_t^\top R_t K_t
+
(A-BK_t)^\top P_{t+1}^K (A-BK_t).

Then compare P_t^K to the Riccati P_t^*. This gives you an all-state value-function certificate that does not depend on a finite rollout sample.

Useful metrics:

\frac{\operatorname{tr}((P_0^K - P_0^*)\Sigma_0)}
{\operatorname{tr}(P_0^*\Sigma_0)}

for several declared \Sigma_0, plus eigen-analysis of P_0^K - P_0^* in physically and adversarially meaningful subspaces.

If P^K is close to P^*, the raw gain mismatch may be ignorable. If P^K is not close, the controller has not recovered the LQR law in the relevant sense.

D. Bellman one-step residual

For each time step, compute the one-step Riccati optimality residual:

\Delta K_t =
K_t^{\text{fit}}
-
(R_t + B^\top P_{t+1}^{\text{ref}}B)^{-1}
B^\top P_{t+1}^{\text{ref}}A.

That is just K_t^{\text{fit}} - K_t^{\text{ref}}, but now you can weight it by the Bellman Hessian:

\|\Delta K_t\|_{H_t,\Sigma_t}^2
=
\operatorname{tr}
\left(
\Delta K_t^\top H_t \Delta K_t \Sigma_t
\right).

This is the principled gain-error metric. Raw Frobenius is the unweighted, coordinate-dependent version of this.

E. Identifiability Gramian

For the training objective, collect the state matrix at each time step:

X_t = [x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(N)}].

Compute singular values/effective rank of:

\Sigma_t = X_t X_t^\top.

Then decompose the gain error into components inside and outside the span of X_t:

K_{\text{err},t}^{\parallel}
=
(K_t^{\text{fit}} - K_t^{\text{ref}})\Pi_{\text{span}(X_t)},

K_{\text{err},t}^{\perp}
=
(K_t^{\text{fit}} - K_t^{\text{ref}})(I-\Pi_{\text{span}(X_t)}).

If most error is perpendicular to visited states, the clean objective is under-identifying K. If much of it is inside the visited/adversarial span, the optimizer has not found the right local law.

F. Optimizer-convergence checks

Because L-BFGS-B hit the iteration limit, you still need the boring checks:

* gradient norm at the final K;
* objective and gradient norm at K_{\text{ref}};
* interpolation curve between K_{\text{fit}} and K_{\text{ref}};
* longer L-BFGS-B runs;
* L-BFGS-B from zero, from K_{\text{ref}} plus noise, and from multiple random starts;
* maybe second-order/Lanczos curvature around K_{\text{ref}} and K_{\text{fit}}.

But these should be used to classify the failure, not to rescue the current certificate by assertion.

6. Proposed replacement certificate

I would split Phase 3 into two explicit sub-gates.

Phase 3A: clean LQR behavior/value gate

This can pass without raw Frobenius gain recovery if all of the following hold:

1. canonical clean trajectory matches analytical LQR;
2. clean cost ratio is within tolerance;
3. terminal error and time-to-peak are within tolerance;
4. exact policy-evaluation value gap is small under declared state covariances;
5. state-weighted action error is small on clean and validation state distributions.

The current L-BFGS-B result probably passes 1–3, but 4–5 are not yet established.

Phase 3B: disturbance-relevant LQR equivalence gate

This is the gate that matters before adversarial/H∞ training.

Require:

1. held-out epsilon audit close to analytical/teacher LQR under the same audit protocol;
2. action mismatch small on adversary-induced states;
3. standardized perturbation responses close to analytical/teacher LQR;
4. closed-loop transfer / induced-gain metrics close to analytical/teacher LQR;
5. gain mismatch, if large, explained as lying in unvisited and disturbance-irrelevant subspaces.

The current L-BFGS-B result appears to fail this gate because the held-out adversary audit is far from teacher/analytical LQR.

Phase 3C: adversarial/H∞ linear gate

Only after 3A and 3B should you train the linear controller with the broad epsilon adversary. Then the pass criterion should not be “does it move faster?” It should be:

1. clean Δv near analytical H∞;
2. clean cost and terminal behavior near analytical H∞;
3. held-out adversary audit near analytical H∞;
4. value/closed-loop transfer metrics near analytical H∞;
5. state-weighted action/gain metrics near analytical H∞ on clean, perturbed, and adversary-induced states.

This matches the older methodological plan: the linear arm is the load-bearing gate, and GRU behavior should not be interpreted if the linear gate cannot certify the intended game.

7. What this means for moving to GRUs

A GRU will not expose a K_t tensor. So a Phase 3 certificate based solely on raw K equality is not portable to the nonlinear phase. The real purpose of the linear phase should be to calibrate metrics that will also make sense for GRUs:

* closed-loop trajectories;
* local perturbation responses;
* local Jacobians around relevant states;
* disturbance-to-cost transfer;
* held-out adversary performance;
* state-weighted action or next-state response errors where a local linearization is available.

So the correct spirit is:

Use the linear Riccati gain as an oracle to validate the closed-loop metrics, not as the only object the trained policy must literally reproduce entry by entry.

That said, the current trained linear controller has not yet validated those closed-loop metrics. It only validated clean canonical behavior.

8. Risks if you proceed now

If you proceed to GRU/RNN training now, you risk a false interpretation in either direction.

If the GRU matches behavior, you will not know whether it learned the H∞ game or merely found another clean-trajectory solution with hidden off-trajectory differences.

If the GRU fails to match behavior, you will not know whether the architecture failed or whether the training objective/certificate was never implementing the same game.

If the GRU dissociates speed and feedback gain, you will not know whether that is a substantive architectural result or just the same weak-identifiability problem reappearing in nonlinear form.

That is exactly what the same-game gate was supposed to prevent.

9. My concrete decision rule

For the current checkpoint, I would record:

Phase 3 clean LQR nominal behavior: provisionally passed.
Phase 3 Riccati feedback recovery: failed under raw Frobenius, but raw Frobenius is not yet accepted as the right gate.
Phase 3 disturbance-relevant closed-loop equivalence: failed or at least not passed, because held-out adversary behavior differs materially.
Proceed to GRU: no.

The next work item should not be “make Frobenius gain error small” in the abstract. It should be:

Build a state-weighted, value-weighted, disturbance-relevant linear equivalence certificate. Then decide whether the current controller fails because the optimizer did not converge, because the empirical rollout objective under-identifies the Riccati law, or because the raw gain metric was asking the wrong question.

If that certificate says the large gain error is off-subspace and harmless, drop raw Frobenius as a hard gate. If it says the error appears on adversarially reachable states, the current controller is not the same game, no matter how good the clean reach looks.
