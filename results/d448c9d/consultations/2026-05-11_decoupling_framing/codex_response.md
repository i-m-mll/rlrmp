**Short Answer**

The framing is close, but too weak as stated. `Δv` is a useful behavioral signature, not a direct signature of “regulator/tracker decoupling.” The decisive object is the tradeoff between nominal movement speed and closed-loop sensitivity: can adversarial training reduce worst-case sensitivity without increasing nominal forward velocity? Your current MVP does not test that cleanly, because [LinearTrackerController](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/src/rlrmp/networks/linear_controllers.py:225) is effectively `u_ff(t) - K_t(pos-target, vel)`, with `x_nom = target` and `u_ff` not target-direction dependent.

My view: the GRU result should not yet be interpreted as “the GRU discovered a tracker.” It is stronger to say: **the GRU is not converging to the same H∞ optimum as the Riccati/linear-controller game.** Tracker-like decoupling is one candidate mechanism, but cost mismatch, adversary mismatch, and optimizer basin effects are at least as plausible.

**1. Is `Δv` The Right Signature?**

Use `Δv`, but demote it from “proof of decoupling” to one point on a Pareto diagnostic.

The corrected local definition in [decoupling_acid_test_mvp.md](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/results/410d7ac/notes/decoupling_acid_test_mvp.md:11) is right:

```text
Δv_arch = (peak_v(arch_adversarial) - peak_v(arch_baseline)) / peak_v(arch_baseline)
```

It matches the C&S-style nominal rollout comparison. But it conflates several things:

- nominal timing policy
- effective control-cost weight
- feedback gain / stiffness
- adversary class
- local robustness
- optimizer basin

The better primary plot is:

```text
x-axis: nominal peak forward velocity or movement time
y-axis: robustness metric
```

where robustness is one or more of:

- induced gain from disturbance to terminal/running error
- worst-case adversarial loss after a strong held-out adversary
- local closed-loop sensitivity around the nominal trajectory
- perturbation-rejection impulse response: area/peak of position error after standardized force pulses

Then decoupling means: **for equal or better robustness, the adversarial model does not need higher nominal peak velocity.**

For tracker evidence, also measure:

```text
u(t, x) ≈ u0(t, target) - K_local(t, target) · (x - x_nom(t, target)) + residual
```

For GRUs, estimate:

- `u0`: rollout output on clean nominal feedback
- `K_local`: Jacobian `-∂u/∂x_feedback` along nominal trajectory
- residual nonlinearity: error of local affine reconstruction under small perturbations
- nominal/feedback power split: `||u0||²` vs `||K_local e||²`

That decomposition is more diagnostic than `Δv` alone.

**2. Is The Linear MVP Valid?**

As a software test, yes. As a decoupling test, no.

The current implementation explicitly says `x_nom(t) = 0` in target-relative coordinates [linear_controllers.py](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/src/rlrmp/networks/linear_controllers.py:20), and the tracker computes:

```python
e_pos = feedback_flat[:2] - target_pos
e_vel = feedback_flat[2:]
u = u_ff_t - K_t @ e
```

at [linear_controllers.py](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/src/rlrmp/networks/linear_controllers.py:307). That is not a trajectory tracker. It is an affine LTV regulator-to-target.

There is a second important flaw: for random center-out directions, a world-frame `u_ff(t)` cannot represent direction-specific feedforward. A single time-indexed vector cannot drive all reach directions unless the task is fixed-direction or the feedforward depends on target direction.

Minimum viable tracker:

```text
x_nom(t, target) = [p_nom(t, target), v_nom(t, target)]
p_nom = p0 + s(t) · (target - p0)
v_nom = sdot(t) · (target - p0)
u_ff = inverse_dynamics(x_nom)
u = u_ff(t, target) - K_t · (x - x_nom(t, target))
```

For the point mass with damping and no actuator filter:

```text
u_ff(t) = m · a_nom(t) + damping · v_nom(t)
```

If the motor filter is active, either deconvolve the force command or initialize `u_ff` from a teacher and let training adjust it.

For multi-direction reaches, make the tracker reach-aligned:

```text
u_ff(t, target) = û(t) · reach_axis
K_t = R(target) · K_reach_frame(t) · R(target)^T
```

or let `u_ff(t, target)` be a small linear map of target displacement. Do not use a single global `u_ff(t)` for all directions.

**3. Best Path Forward**

Priority order:

1. **Do a fixed-direction sanity test first.**  
   One start, one target, no target-direction generalization. Train:
   - LTV regulator: `u = -K_t x_target_error`
   - true trajectory tracker: `u = u_ff(t) - K_t(x - x_nom(t))`
   - tracker with `K` frozen / `u_ff` trainable
   - tracker with `u_ff` frozen / `K` trainable

   Evidence for decoupling: adversarial tracker improves held-out robustness mostly by raising `K_local`, while nominal `u_ff` and peak velocity remain near baseline.

2. **Teacher-initialize the tracker.**  
   Use minimum-jerk or LQR nominal trajectory. Initialize `u_ff` from inverse dynamics. Initialize `K_t` from LQR or small stabilizing gains. Zero initialization biases toward the K-only basin; your MVP saw exactly that, with `u_ff` still an order of magnitude smaller than `K e` [decoupling_acid_test_mvp.md](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/results/410d7ac/notes/decoupling_acid_test_mvp.md:55).

3. **Run a matched-cost linear/Riccati round trip.**  
   Before more GRU work, make the trained linear regulator reproduce the analytical sign and approximate magnitude under the same:
   - plant
   - horizon
   - Q schedule
   - R/control penalty
   - disturbance channel
   - delay treatment

   If trained linear controllers give +78% while the Riccati target is +10 to +27%, the training objective is not the Riccati game.

4. **Then repeat with GRU under the same matched objective.**  
   Current training defaults are not C&S-faithful: 140 steps, 0.5 m eval reach, feedback delay/noise, flat schedules by default, hold/running losses, `nn_output`/`nn_hidden` regularization [train_minimax.py](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/scripts/train_minimax.py:303). That is not a clean H∞ controller-cost clone.

5. **Analyze the bimodality before averaging more runs.**  
   The flavor-B result is not “GRU = −25%”; it is “mixture of positive and negative solutions” [peak_velocity_cross_method_comparison.md](/Users/mll/Main/10%20Projects/10%20PhD/rlrmp/results/c723082/notes/peak_velocity_cross_method_comparison.md:38). Cluster by:
   - nominal velocity profile
   - final loss
   - adversarial loss
   - local feedback Jacobian norm
   - hidden-state trajectory PCA
   - readout norm / hidden norm
   - pre-go drift

   The positive-Δv minority may already be the H∞-like basin.

6. **Run the GRU affine decomposition.**  
   For each trained replicate, fit/measure:
   - `x_nom(t)`
   - `u0(t)`
   - `K_local(t)`
   - nonlinear residual
   - alignment of `K_local` with Riccati `K_t`

   If negative-Δv GRUs have high robustness but low `K_local`, the tracker hypothesis weakens. If they have high `K_local` but reduced `u0`, then they are robust but deliberately slower. If positive-Δv replicates align with Riccati gains, the puzzle becomes optimizer basin selection.

**4. Alternative Hypotheses**

Most likely alternatives, ranked:

1. **Objective mismatch.**  
   This is my top concern. C&S robust control predicts increased speed and feedback gain under a specific finite-horizon quadratic game. Your GRU training loss includes delayed reach structure, hold losses, running/late losses, noise, catch trials, and regularizers. The latest synthesis already flags this as central.

2. **Adversary mismatch.**  
   C&S analytical H∞ uses free additive disturbance with full-state `B_w`; your flavor-B training uses Frobenius-bounded `ΔA x`. Those are not equivalent. A controller can be robust to the trained structural perturbations while not behaving like the full-state H∞ controller.

3. **The GRU is robust in a weaker/narrower sense.**  
   “Does not catastrophically fail” is not the same as small induced gain. Evaluate with a stronger held-out adversary and report robustness as a curve, not a binary.

4. **Slow strategy exploits the cost schedule.**  
   If terminal timing is weak or horizon is forgiving, moving slower can reduce motor noise, actuator effort, and adversarial leverage while still satisfying hold error. That is not decoupling; it is a different optimum.

5. **Basin / regularization effects.**  
   The GRU may have two attractor classes: fast stiff H∞-like and slow conservative. The bimodality strongly suggests this. Averaging across seeds is actively hiding the mechanism.

6. **Hidden-state feedforward exists, but it is not “tracker” feedforward.**  
   The hidden state may encode timing, target, and disturbance context, but the output may still be a nonlinear policy rather than `u_ff - K e`. The affine-local analysis is required.

**5. Missing Pieces**

The biggest missing experiment is not a bigger tracker. It is a **matched-objective ladder**:

```text
Riccati controller
→ trained LTV regulator
→ true LTV tracker
→ affine recurrent controller
→ GRU
```

All under the same plant, cost, horizon, and disturbance channel. Only then interpret architecture differences.

Literature anchors I would keep in view:

- C&S 2019 for the empirical robust-control claim and model framing: increased movement speed and feedback gains under unpredictable disturbance ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6786821/), [ModelDB code](https://modeldb.science/showmodel?model=258846)).
- Todorov & Jordan 2002 for optimal feedback control and the “not necessarily tracking a desired trajectory” caution ([Nature Neuroscience](https://www.nature.com/articles/nn963)).
- DGKF / standard H∞ state-space theory for what the Riccati game is actually certifying ([IEEE CSS summary](https://ieeecss.org/paper/state-space-solutions-standard-h2-and-h-infinity-control-problems)).
- Low-rank RNN dynamics work for interpreting GRU solution classes rather than treating the 180D state opaquely ([Mastrogiuseppe & Ostojic 2018](https://pubmed.ncbi.nlm.nih.gov/30057201/)).

My hard pushback: do not spend the next phase trying only “longer training of the existing tracker.” The existing tracker lacks a real trajectory reference and target-dependent feedforward, so more compute mostly tests optimizer persistence in a degenerate parameterization. The next clean step is a true trajectory tracker plus matched-objective linear round trip.
