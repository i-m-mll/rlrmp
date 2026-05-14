I would push back on the premise this way: a fixed adversary at convergence is not, by itself, non-adversarial. In H-infinity/LQ games the controller also converges to a fixed policy against a fixed worst-case solution. C&S explicitly say the uncertainty is random across trials but not within a trial; the formal controller is robust to an unknown disturbance set, not to perpetual novelty.

But the user’s worry is well-founded for the current GRU training pipeline. The failure mode is not “fixed adversary” in the abstract. It is: nonconvex RNN + one learned adversary per replicate + weakly verified PGD inner loop + no held-out worst-case audit can converge to a local co-adapted pair that is not robust to the intended disturbance class. The right test is not philosophical; freeze the controller, re-run strong independent adversary search with many restarts/step counts, and ask whether the saved adversary was actually worst-case. That makes `89891ab` a blocker for trusting minimax claims.

**My Central Reframe**
The load-bearing axis is not “saddle point vs co-adaptation.” It is:

1. What uncertainty channel is being robustified?
2. What cost schedule makes speed useful rather than wasteful?
3. Is the policy forced to couple nominal movement speed to feedback gain, or can it decouple them?
4. Has the inner maximization actually found the worst case?

C&S’s analytical setup is now best read as full-state additive epsilon H-infinity, not current flavor-B ΔA. The current `LinearDynamicsAdversary` is a different game, and its empirical negative Δv is not surprising: a state-multiplicative adversary can make “keep state small / move slower” an optimal conservative strategy.

**Multi-Signature Prediction**
My prediction table:

| Regime | Speed Δv | Feedback gain / perturbation rejection | Why |
|---|---:|---:|---|
| Standard expected-cost perturbation training | no / tiny | yes, modest | LQG-like; improves corrections without changing nominal trajectory |
| CVaR / APT / PAI-ASF | no / tiny | yes, stronger | Already observed in `results/844ef95`: robustness without peak-velocity change |
| Gaussian bump minimax | unlikely | likely | Force-profile adversary can be handled by corrective feedback or memorized timing |
| Current ΔA flavor-B | likely mixed/negative mean | unclear / channel-specific | State-coupled uncertainty rewards small state excursions; observed bimodality matters |
| LEQG alone | maybe small | yes | Better risk tilt, but still only as good as its disturbance channel |
| Full-state ε H∞-matched training | most likely | most likely | Closest to C&S Eq. 13 and the passing Riccati tests |
| H∞ Riccati teacher distillation | highest existence check | yes if architecture can express it | Not a training method answer, but decisive capacity check |

So I do not expect one universal “adversarial training” signature. Feedback-gain modulation is broad and easy. Speed inflation is narrower: it appears when robust gain inflation is tied to nominal steering by the plant/cost/controller parameterization. RNNs may learn tracker-like decompositions that preserve feedback robustness while leaving nominal speed flat.

**Single Next Experiment**
Must-do: implement the full-state ε adversary matching C&S Eq. 13, with C&S-style `(t/N)^6` cost schedule and matched horizon/delay, then run a small round-trip experiment:

- Adversary: free time-varying full-state ε_t, L2-bounded over rollout.
- Training: minimax, with PGD step sweep `{5, 20, 50}` and independent restarts.
- Arms: linear regulator first as a round-trip gate; one GRU arm only after the linear arm matches Riccati.
- Evaluation: no group means alone. Report replicate-conditioned Δv, feedback-gain/step-response metrics, induced gain by channel, endpoint/lateral deviation, and adversary held-out loss.
- Success: linear regulator recovers Riccati-sign positive Δv; GRU either also recovers it or cleanly dissociates gain from speed.
- Failure: linear regulator fails the Riccati round-trip. Then the training/evaluation pipeline is not the H∞ game yet, and GRU interpretation should stop.

This folds `020a65b`, `89891ab`, `6ec6b19`, and `daa48c8` into one decisive experiment. The cheap no-new-training audit `b6084c7` is still a practical blocker before launching compute.

**What I Would Deprioritise**
I would not spend central effort on more current flavor-B η sweeps, population adversaries, or regulator-vs-tracker architecture comparisons until the matched ε round-trip works. They may be interesting, but they do not answer whether RNNs can reproduce the C&S formal regime.

I would also defer muscle/biomechanics work, LEQG-alone, broad reach-geometry sweeps, and interpretability-first GRU decomposition. Those become valuable after you have at least one trained GRU with a verified robustness signature. Right now the project needs a matched training game more than another explanation of an unmatched one.
