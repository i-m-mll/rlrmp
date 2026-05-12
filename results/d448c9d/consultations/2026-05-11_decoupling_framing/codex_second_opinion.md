Yes, your ordering mostly makes sense, but I’d change the emphasis.

**Main Pushback**
- Tier 1 is not as decisive as you imply. GRU affine decomposition is useful, but it is observational and local. It can tell you whether the trained GRU *looks tracker-like near its nominal path*, not whether tracker structure caused Δv decoupling.
- The most dangerous hidden assumption is that “GRU Δv ≈ 0/negative” and “linear Δv +78%” are comparable. They currently are not cleanly comparable unless plant, horizon, cost, adversary class, delay, task input, noise, and training objective all match.
- I would move the **matched-objective Riccati/trained-linear round-trip** ahead of most representational analysis. A 30× Δv discrepancy is a louder alarm than the tracker/regulator story.

**Tier 0 I’d Add**
- **Trajectory/cost audit on existing checkpoints:** for baseline GRU, flavor-B GRU, trained linear regulator, and Riccati:
  - nominal movement time / peak velocity / terminal error
  - state-cost time course `x_t^T Q_t x_t`
  - control-cost time course `u_t^T R_t u_t`
  - adversary loss actually optimized vs Riccati `w -> z` loss
  - exact delay/noise/catch-trial/task-input differences
- This is cheaper than GRU Jacobians and may immediately show “not the same game.”

**Single Highest-Priority Experiment**
- Make the trained LTV regulator reproduce Riccati behavior under the exact same clean setup.
- If trained LTV still gives `Δv ≈ +78%` where Riccati gives `+1-27%`, stop doing architecture comparisons. The training objective/evaluation pipeline is the puzzle.
- If it matches Riccati, then proceed to tracker/GRU architecture tests.

**On Your Tiers**
- **1.1 GRU affine decomposition:** good, but interpret cautiously. For a GRU, instantaneous `∂u/∂x_feedback` at fixed hidden state is not the whole feedback law; the hidden state is part of the closed-loop state. Use augmented-state linearization and distinguish:
  - `u_nom(t)`
  - direct sensory/readout sensitivity
  - hidden-state-mediated sensitivity
  - induced closed-loop sensitivity
- **1.2 clustering:** yes. Do it per replicate. But do not overfit narratives to PCA clusters unless the positive-Δv group also differs on loss/robustness/Jacobian/trajectory metrics.
- **2.1 fixed-direction sanity test:** good. I’d keep this, but only after the Riccati round-trip is trusted.
- **2.2 Riccati round-trip:** I’d promote this to Tier 0/Tier 1.
- **3.1 matched GRU:** yes, this is the conclusive test, but only after the clean linear round-trip works.

**Flaws / Confounds**
- Direction-dependent `u_ff(t, target)` is essential for center-out. A single world-frame `u_ff(t)` is not a real center-out tracker.
- “Nominal/feedback power split” is ill-defined on the undisturbed nominal trajectory: the local feedback correction is zero by construction. Measure it under standardized perturbations or worst-case `w*`.
- Existing flavor-B bimodality means means are almost useless. Use replicate-level analysis throughout.
- Delay matters. A delayed GRU tracker may solve a different problem than full-state Riccati even with nominally identical costs.
- If GRU robustness is measured under empirical adversary rollouts but Riccati robustness is measured as an induced norm, you are mixing notions of robustness.

**`induced_gain.py`**
- It is useful machinery, but not a complete GRU affine-decomposition tool.
- `linearise_trajectory(..., z_channel="control")` can give `Cz_t = ∂u/∂x_aug` and `u_nominal`.
- However, real trained SimpleFeedback models apparently need the network-only adapter from `scripts/run_induced_gain_part2_5.py`; `feedbax_graph_controller` is not sufficient for those checkpoints.
- You’ll likely want a small new analysis wrapper that exposes `u_nominal`, `∂u/∂x_plant`, `∂u/∂h`, residual nonlinearity, and perturbation-response decompositions directly.

Bottom line: I agree with your path, but I’d demote GRU representational analysis from “highest leverage” to “useful diagnostic,” and promote the Riccati-vs-trained-linear matched-objective round-trip to the first decisive move. The current biggest risk is explaining an architecture effect before proving all controllers are playing the same control game.
