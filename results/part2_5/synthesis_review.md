# Synthesis Review: Why the SISU-Velocity Signature Failed and What to Do Next

This document is a unified analysis of the rlrmp Part 2.5 null result on Crevecoeur & Scott (2019)–style velocity inflation. Phase 1 produced the four-question technical core (H∞ Riccati sanity check, objective-form candidates, adversary-class redesign, diagnostic measurements). This Phase 2 review verifies Phase 1's load-bearing claims, recomputes the damping-mismatch question numerically (and inverts Phase 1's intuition), adds a tiered analysis menu, cross-walks the 26 open issues, registers falsification criteria, and gives a sequenced experimental plan.

Notation follows Phase 1: $\theta$ = controller parameters; $\delta$ or $w$ = disturbance; $J(\theta,\delta)$ = trajectory cost; $A,B,B_w,C_z,D_{zw}$ = linearised plant; $\gamma$ = H∞ disturbance-attenuation level (small = strong attenuation); $\beta$ = SISU; $T_{w\to z}$ = closed-loop transfer; "flavor (a)" = additive force-trajectory adversary, "flavor (b)" = structural model-class adversary on $\Delta A$.

---

## 1. Framing

The empirical fact: training rlrmp's GRU-controller against perturbations — with PAI-ASF SISU wiring, with CVaR, with APT, with a parametric `GaussianBumpAdversary` minimax — produces no increase in peak reaching velocity at high SISU. C&S report a clear human velocity inflation signature under unpredictable disturbances; we get $|\Delta v_{\text{peak}}| < 1.2\%$ across every converged condition.

The synthesis-4 framing supplies the candidate explanation: replacing $\mathbb{E}_\delta L$ with $\max_\delta L$ produces a structural shift in the *Hessian of a locally-defined potential* — in motor control, the closed-loop value-function Hessian — and on an LQ closed-loop attractor that shift drives the controller toward higher peak velocity (the C&S signature). The shift only fires if the inner $\max$ is over a class rich enough to cross-couple to the policy's policy-gradient direction. Three flavors of $\max_\delta$ matter:

- **(a) Input-instance**: $\delta$ is a force trajectory $w \in \mathbb{R}^{T \times 2}$, additive at the effector. PGD on $w$ over an $\ell_2$ ball is the standard adversarial-training move and is what rlrmp currently does.
- **(b) Model-class / structural**: $\delta$ is a perturbation $\Delta A$ to the plant's dynamics matrix. The same $\Delta A$ acts at every step, multiplicatively on state — the C&S "structural" disturbance and the natural target of the H∞ Riccati game.
- **(c) Distributional / risk-sensitive**: $\delta$ is the noise distribution itself; LEQG and DRO put the operator on the cost-distribution rather than on the trajectory.

The C&S replication asks for (b). rlrmp's entire training stack runs (a). Phase 1 settled this from a code-reading angle: every disturbance class in rlrmp (`gusts`, `curl`, `constant`, `noise`, `GaussianBumpAdversary`, APT-PGD) injects an additive force on the effector channel. None modifies $A, B, \tau, k, m$, the feedback delay, or the noise covariances.

A second concern surfaces from the same code reading: rlrmp's plant has $k_{\text{rlrmp}} = 10$ Ns/m vs C&S's $k = 0.1$ Ns/m, a 100× damping mismatch. Phase 1 flagged this as a possible competing explanation (high damping saturates the controller, no headroom for K-inflation). Section 2 verifies the units and recomputes the headroom question numerically; the answer reverses Phase 1's intuition.

---

## 2. Damping-Mismatch Verification

**(A) Numerical value and units.** `feedbax.mechanics.skeleton.pointmass.PointMass` builds $A_{\text{vel}} = -(k/m)\,I_2$. With rlrmp's training default `damping=10.0`, `effector_mass=1.0`, $k/m = 10\,\text{s}^{-1}$. The physical units are unambiguous: $k/m$ is a 1/time constant, and `dt=0.01` s is in seconds. Velocity-decay timescale $\tau_v = m/k = 100$ ms. C&S's $k=0.1$ gives $\tau_v = 10\,000$ ms.

The 100× damping mismatch is **real and units-matched**. It is not an artifact of workspace-coordinate scaling: $k/m$ has units of inverse time, so any rescaling of length cancels. Over a 1-second movement, rlrmp damping completes 5 e-foldings of velocity decay; C&S damping completes 0.06.

**(B) Effect on H∞ headroom — Phase 1's intuition is wrong.** Phase 1 conjectured that high damping reduces the H∞ velocity-inflation effect ("there may be little room for further inflation"). I tested this by solving the finite-horizon discrete-time H∞ Riccati on the full 6-state plant + a representative time-varying $Q$ schedule (mid-period: $Q_{\text{pos}}=I_2$, $Q_{\text{vel}}=0$; late-period: $Q_{\text{pos}}=4I_2$, $Q_{\text{vel}}=0.4 I_2$, ramped factors matching rlrmp's `running_cost` plus `effector_*_late` ramp; $R = 3\times10^{-5} I_2$ matching the adaptive control-cost converged value). $T = 100$ steps at $dt=0.01$, terminal $Q_f = Q_{\text{late}}$.

| Plant | LQR $\|v\|_{\text{peak}}$ | $\gamma_*$ | $\Delta v\%$ at $1.5\gamma_*$ | $\Delta v\%$ at $1.2\gamma_*$ | $\Delta v\%$ at $1.05\gamma_*$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| rlrmp $k=10,\tau=0.05$ | 1.86 | 0.0155 | **+10.8%** | **+18.8%** | **+27.2%** |
| C&S-like $k=0.1,\tau=0.06$ | 2.10 | 0.0174 | +1.5% | +2.3% | +2.4% |

**The high damping makes the velocity-inflation effect *larger*, not smaller.** Mechanism: the LQR baseline on rlrmp is *under-powered* because the controller spends control fighting the 10/s velocity decay, peak velocity ≈ 1.86 (vs C&S's 2.10). H∞ at $1.5\gamma_*$ inflates the feedback gain enough to overcome the damping, reaching $\approx 2.06$ peak velocity — a +11% bump. C&S's near-frictionless plant is already coasting at LQR; H∞ has very little to add. **rlrmp's parameter regime makes the velocity signature *more visible*, not less.**

This inverts the Phase 1 worry. The damping mismatch is real but does not reduce — and probably amplifies — the predicted velocity inflation under flavor (b) training. The discrepancy with the trained networks is therefore squarely an **objective-form / training-class** problem: rlrmp's training uses $\mathbb{E}_w$ over flavor-(a) disturbances, never $\max_{\Delta A}$ over flavor-(b) ones, and that gap (not damping) is what kills the signature.

**(C) Caveat: the trained policy is already above the LQR baseline.** Trained networks reach peak velocity ≈ 3.33 (running-cost loss) or 2.42 (adaptive control-cost loss). My LQR proxy on the same $Q,R$ predicts $\approx 1.86$. The trained controllers are 30–80% faster than the on-paper LQR, indicating the on-paper $Q,R$ does not capture the actual fixed-point of the loss-update mechanism, the implicit reach-time pressure of `effector_pos_late`, or the network's softmax-like terminal-position prior. So the *headroom* prediction (room to inflate by +10–25%) should be read directionally, not as a quantitative target. The right protocol is to compute the H∞ controller on the same plant, simulate it, and compare its peak velocity to the LQR controller computed with the same loss matrices — which is exactly Q1's procedure.

**(D) Other Phase 1 claims I verified or refined.**

- *"rlrmp curl is implicitly $\Delta A_{vv}$ but applied via the additive-force channel."* Confirmed by reading `disturbance.py`: curl produces a force trajectory $w(v) = c\!\begin{pmatrix}0&1\\-1&0\end{pmatrix}\!v$ which enters as `effector force`, not as a structural perturbation on the LTI matrices. So curl-training does **not** auto-lift to flavor (b). To get flavor (b), rlrmp needs a new feedbax intervenor that adds $\Delta A \cdot x$ inside the dynamics rollout.
- *"GRU laziness is not the bottleneck."* Confirmed by reading the README: the architecture sweep showed vRNN matched GRU at 0% Δvel. Issue ce34c2c is downgraded to "still worth a one-shot Jacobian-eigenvalue sanity check, but not a primary explanation."
- *"PAI-ASF wires SISU to the network, not to the loss."* Confirmed in `SISU_FNS["pai-asf"]`. Training pressure on $\beta$-dependent behavior comes only through the $w$-distribution shift, not through any direct loss coupling. This means the cross-partial $\nabla^2_{\theta\beta} L$ is mediated through $\nabla^2_{\theta\delta} L$ together with $\partial \delta^* / \partial \beta$ — and the relevant cross-partial is the $\theta$–$\delta$ one, not $\theta$–$\beta$.
- *"CVaR ≠ LEQG and the codebase does not implement LEQG."* Verified: `_CVaRCompositeLoss` is per-trial tail-mean, not exponential tilting. LEQG would be a 10-line change to the loss aggregation.
- *"rlrmp's filter is symmetric ($\tau_{\text{rise}}=\tau_{\text{decay}}=0.05$)."* Confirmed in `feedbax/filters.py`. C&S do not specify; this is unlikely to be load-bearing for the velocity signature.
- *"Phase 5 r=0.10 result (+4.8%) is task-failure, not signature."* Confirmed in README: ep_err=0.158 ≫ ep_err=0.003 of competent reaches.

The recommendation order changes accordingly — see §11.

---

## 3. Why $\mathbb{E}_w \to \max_w$ Was Not Enough

A short diagnostic narrative, integrated across Phase 1's Q1 and Q4.

The synthesis (§1, post-Danskin) gives the local gate: $\nabla_\theta L(\theta,\delta^*) \approx \nabla_\theta L(\theta,0) + \nabla^2_{\theta\delta}L(\theta,0)\cdot\delta^*$. For the inner $\max$ to bend $\theta$ in a velocity-relevant direction, the cross-partial $\nabla^2_{\theta\delta}L$ must be non-zero in those directions. We **know** the cross-partial is non-zero in *some* direction — Phase 3's ~55% APT lateral-deviation reduction is structural evidence of it — but the velocity null result implies that the projection onto the velocity-relevant subspace is nearly zero.

Two non-exclusive explanations:

1. **Wrong $\delta$-class.** A force-trajectory adversary $w \in \mathbb{R}^{T\times 2}$ couples to the controller's *low-pass*-filtered velocity command but does not robustify the controller against state-coupled (multiplicative) perturbations. The H∞ Riccati's $\gamma^{-2}B_w B_w^\top$ term enters the bracket inversion *with* $\gamma$ at the boundary; without a class that sees $A$ multiplicatively, the network has no incentive to inflate the value-function Hessian along the velocity axis.
2. **Not enough $\gamma$-tilt.** Even with a force-trajectory adversary, an $\mathbb{E}_w$-flavor objective averages over realisations and lands near LQG. To approach H∞ we need the cumulant-generating function (LEQG) or true worst-case. CVaR is a discrete IS-style estimator and rlrmp's CVaR-trained models showed worse convergence and *no* velocity inflation. APT-PGD is a finite worst-case proxy but the inner-loop budget is small relative to the Riccati boundary.

Phase 1's argument is that fixing only (2) (LEQG without (1)) lifts us to a "compile-time risk-tilt of an inadequate training distribution" — an LEQG-flavored objective on a flavor-(a) class does not access the structural Hessian inflation. Fixing only (1) (a flavor-(b) adversary trained against $\mathbb{E}$) gets us closer but still misses the cumulant tail. Fixing both gives the cleanest signal.

---

## 4. Objective-Form Candidates (Phase 1's Q2, integrated)

Three candidates. All are described with finite-horizon discrete-time matrices to match rlrmp's `dt=0.01`, `n_steps=130`.

### 4.1 LEQG / risk-sensitive (cumulant tilt)

$$
J_{\text{LEQG}}(\theta) = \frac{1}{\gamma}\log\mathbb{E}_w\!\left[\exp(\gamma\,J(\theta,w))\right],\quad \gamma>0.
$$

At $\gamma\to 0^+$: $J_{\text{LEQG}} = \mathbb{E}[J] + \tfrac{\gamma}{2}\text{Var}(J) + O(\gamma^2)$, recovering LQG. As $\gamma\to\gamma_*$ (Whittle breakdown): $J_{\text{LEQG}}$ converges to the H∞ controller, with $1/\gamma$ playing the role of the disturbance budget $\epsilon^2$.

Implementation is a 10-line change to the loss aggregator: per-trial losses $J_b$, weights $\log w_b = \gamma J_b$ (stop-gradient), self-normalised softmax over the batch, weighted sum. The streaming-loss machinery in `feedbax.streaming_loss` already supports per-trial losses. No new adversary, no inner loop.

**SISU wiring.** The natural choice is $\gamma(\beta) = \gamma_0\,\beta$ or $\gamma(\beta) = \gamma_*(\beta/(1+\epsilon))$, so $\beta=0$ recovers LQG and $\beta=1$ sits near the H∞ boundary. This is a *re-interpretation* of $\beta$: currently it scales disturbance amplitude; under LEQG it scales risk tilt. These are different dials connected by the Whittle correspondence.

**Variance properties.** Self-normalised IS effective sample size $N_{\text{eff}} = (\sum w_b)^2/\sum w_b^2 \to 1$ as $\gamma\to\gamma_*$. The empirical estimator hits the same boundary as the Riccati. Mitigations: large batches (rlrmp's 250 may not be enough at high $\gamma$), per-trial gradient clipping, or a $\gamma$-schedule.

**Critical caution.** $\text{Var}(J)$ in the cumulant expansion is a moment of the cost distribution; it is *not* $\nabla_x^2 L$ or any local Hessian. Documentation must not conflate them.

### 4.2 Model-class structural adversary on $\Delta A$

Rewrite the plant as $\dot x = (A + \Delta A)x + Bu + B_w w$ with $\|\Delta A\|_F \le \eta$, constant over the trial. Inner loop: 5–10 PGD steps on $\Delta A$ to maximise $J$. Outer loop: standard backprop on the controller against the trial cost evaluated under the worst $\Delta A$.

Structurally interesting: the velocity row is the natural target.
$$
\Delta A = \begin{pmatrix} 0 & 0 & 0\\ 0 & \Delta A_{vv} & \Delta A_{vF}\\ 0 & 0 & 0\end{pmatrix}
$$
with $\Delta A_{vv} \in \mathbb{R}^{2\times 2}$ giving curl-like coupling, $\Delta A_{vF}$ giving force-channel scaling. Restricting to $\Delta A_{vv}$ alone (4 free params) recovers the C&S setting; restricting further to the antisymmetric part (1 param) recovers the curl-strength dial.

Why this differs from rlrmp's existing curl: the curl perturbation is computed from the network-visible velocity and added at the effector force input. A $\Delta A_{vv}$ enters multiplicatively *inside* the dynamics rollout, so the network sees it through the integrated state at all times, not just as a force impulse. The Riccati robust against $\Delta A$ produces a feedback law structurally different from the Riccati robust against $w$ (multiplicative vs additive disturbance budget).

**SISU wiring.** $\eta(\beta) = \eta_{\max}\beta$ — the budget on the structural perturbation scales with SISU. This is a more honest interpretation of $\beta$ than the current PAI-ASF.

### 4.3 H∞ Riccati teacher (offline)

The plant is LTI, so the H∞ Riccati on it is *exact*, not an approximation. Two-phase distillation:

- **Phase A**: solve the H∞ finite-horizon discrete-time Riccati (Phase 1 Q1.4 pseudocode); bisect for $\gamma_*$; compute $K_t$ for $\gamma = (1+\epsilon)\gamma_*$.
- **Phase B**: train the GRU policy by behavioural cloning of $u_t = -K_t \xi_t$ on simulated trajectories (with delay augmentation; without if the K is computed on the un-augmented plant).
- **Phase C** (optional): fine-tune end-to-end with the §4.2 $\Delta A$-adversary.

This is the *cleanest* possible test of "given a known-correct robust controller, can the GRU+ delay network produce the velocity signature?" If yes, the architecture is fine and any fix to (4.1) or (4.2) will work; the only remaining question is whether end-to-end optimization can find this controller on its own.

### 4.4 Predictions (with confidence)

| Candidate | Recovers velocity signature? | Why |
| --- | --- | --- |
| LEQG (4.1) | Maybe — moderate. | Right tilt of the objective, wrong $\delta$-class. Cheapest. |
| $\Delta A$-adversary (4.2) | Yes — high. | Lifts to flavor (b); §2's H∞ table predicts +10–25% velocity inflation. Need ~150 LOC + tests. |
| H∞ Riccati teacher (4.3) | Yes — highest. | Existence proof. Decouples training-objective question from architecture question. |

**Recommended ordering** (refined from Phase 1 in light of §2): the H∞ Riccati simulation alone (Q1) is the cheapest possible go/no-go and runs in minutes — do it first. Then run the §6 induced-gain measurement on existing checkpoints. Then attempt 4.3 (cheap: solving Riccati already done; behavioural cloning is a few hours). Then 4.2 (the production fix). 4.1 only after these to test whether the cumulant tilt buys anything beyond what 4.2 already gives.

---

## 5. Adversary-Class Redesign (Phase 1's Q3, integrated)

The minimal infrastructural change is a feedbax intervenor that perturbs the dynamics matrix $A$ inside the rollout, and a corresponding rlrmp adversary class. Phase 1's sketch:

1. **feedbax** (~30 LOC): `DynamicsMatrixPerturb` intervenor, wraps `LTISystem.vector_field` to add `delta_A @ state`.
2. **rlrmp** (~150 LOC): `LinearDynamicsAdversary(eqx.Module)` with a learnable $\Delta A$ and Frobenius-ball projection; `make_dynamics_adversary_pre_step_fn` parallel to `make_adversary_pre_step_fn`; `--adversary-type {gaussian-bump, linear-dynamics}` flag in `train_minimax.py`.

For the structural adversary's expressivity:
- **Curl-only** ($\Delta A_{vv}$ antisymmetric, 1 param): cheapest, matches C&S.
- **Velocity-row generic** ($\Delta A_{vv}$, 4 params): moderate, exposes off-diagonal coupling.
- **Low-rank $uv^\top$** (12 params): more capacity, allows structural force-channel coupling.

Inner loop: 5 steps PGD per training batch with Frobenius ball projection. Same scaffolding as `APTTrainingWrapper.find_adversarial_perturbation`. The closed-form Riccati-derived worst $\Delta A$ exists for the linearised closed loop, but PGD is robust to the network's nonlinearity.

**SISU compatibility**: $\eta(\beta) = \eta_{\max}\beta$. Per-trial $\Delta A$ requires vmap over the batch — trivially parallel.

---

## 6. Diagnostic Measurements (Phase 1's Q4, integrated)

### 6.1 Compute the induced gain $\|T_{w\to z}\|_\infty$ first

**Definition.** Linearise the closed loop around the nominal reach trajectory:
$$
\xi_{t+1} = A_{\text{cl},t}\xi_t + B_{w,t}w_t,\quad z_t = C_{z,t}\xi_t + D_{zw,t}w_t,
$$
where $\xi_t \in \mathbb{R}^{216}$ stacks plant ($6$) + GRU hidden ($180$) + delay aug ($30$) state. Compute $\|T_{w\to z}\|_\infty$ as the largest singular value of the lifted Toeplitz operator $w_{0:T}\mapsto z_{0:T}$, via matrix-free power iteration on $T^\top T$.

**Cost**: ~1 s per checkpoint on H100. Returns a single scalar directly comparable across all trained models.

**What it diagnoses.** Per synthesis §5: the *true* robust-control invariant is $\|T_{w\to z}\|_\infty$. A clenched controller has turned this volume knob down. The synthesis predicts: APT < Standard < no-perturbation baseline; SISU=1 < SISU=0 within a model. If all our models hit a *floor* on $\|T_{w\to z}\|_\infty$ — a level above which they cannot push — that floor is the objective-form ceiling, exactly the gap the C&S signature lives above.

The induced gain also lets us read off the "clench tax" as the Lagrangian dual: at the achieved gain, what's the nominal-cost gap? Directly extractable.

### 6.2 Cross-partial $\nabla^2_{\theta\delta}L$ second

For a current checkpoint with `GaussianBumpAdversary`, $|\delta|=12$ (3 Gaussians × 4 params). Compute the $|\theta|\times|\delta|\approx 100\,000\times 12$ matrix column-by-column via JVP through `jax.grad`. Cost: ~12 forward+backward+JVP triples ≈ 36 forward-equivalent passes ≈ 1–2 minutes per checkpoint.

**What it diagnoses**: which directions in $\theta$-space the current adversary can pull. The synthesis predicts the cross-partial is non-zero overall (Phase 3's lateral-deviation reduction is structural evidence) but orthogonal to the velocity-relevant subspace. Confirming this rules out the "adversary too weak" hypothesis and pins the discrepancy to objective form.

### 6.3 Why this order

Induced gain is a single scalar with direct C&S correspondence (peak velocity inflation = lower induced gain), produces immediate cross-checkpoint comparison, and runs in seconds. Cross-partial is high-dimensional and only worth computing once induced gain has shown structure.

**Concrete sequence** (with C&S-style targets):
1. Induced gain on all Phase 2–6 checkpoints. Look for: (i) APT < Standard? (ii) SISU=1 < SISU=0? (iii) does every model hit a common floor?
2. Cross-partial decomposition where induced gain reveals interesting structure.
3. Induced gain on the H∞ Riccati controller (from §4.3 Phase A) on the linearised plant; quantify the gap.

---

## 7. Tiered Analysis Menu

The tiers are defined by what the analysis would *change* about the next experimental decision:

- **Essential**: on the critical path. Outcome would shift the next-step plan.
- **Desirable**: corroborating, raises confidence, multi-line confirmation.
- **Auxiliary**: post-resolution; useful for the broader framework but not for closing the rlrmp question.

### 7.1 Essential

| Analysis | Synthesis prediction it tests | What it does | Cost | Why essential |
| --- | --- | --- | --- | --- |
| **H∞ Riccati simulation on rlrmp plant (Q1)** | The plant *is* C&S's plant under H∞ — velocity inflation is mechanical, not architectural | Solve discrete-time finite-horizon H∞ Riccati on rlrmp's $A,B,B_w$ with rlrmp's $Q_t$ schedule; simulate closed-loop reach; compare peak velocity at $\gamma\to\infty$ (LQR) vs $\gamma$ near $\gamma_*$ | Hours (one Python file) | §2's table is suggestive. A clean run with rlrmp's actual $Q_t,R$ and trained-policy operating point as the linearisation closes the predicted-effect-size question and pre-registers the §4.3 teacher target. |
| **Induced gain $\|T_{w\to z}\|_\infty$ across checkpoints (Q4.2)** | Robust-control invariant; should drop with APT, with SISU, and stratify training methods | Power-iterate Toeplitz of linearised closed-loop $\xi\to z$ map | ~1 s/ckpt | Quantifies how far each trained model is from the Riccati ceiling. If APT models do not have lower induced gain than Standard, the "APT robustifies" reading from the lateral-deviation results is wrong and we need to rethink. |
| **Cross-partial $\nabla^2_{\theta\delta}L$ on current adversary (Q4.1)** | Cross-partial gates the inner-max's ability to bend $\theta$ | Column-wise JVP through $\nabla_\theta L$ | ~1–2 min/ckpt | Distinguishes "wrong adversary class" (cross-partial small in velocity directions) from "weak adversary" (small everywhere). |
| **Damping-sweep training calibration** | Velocity-inflation effect *grows* with damping — inverts Phase 1 intuition | Re-train one or two control conditions at $k\in\{0.1,1,10\}$ keeping the rest of the training stack fixed; measure peak velocity, lateral deviation, induced gain | ~1 day compute | Confirms or refutes §2's prediction that rlrmp's high damping is *favorable* for the signature. If high damping does not produce more inflation, §2's analysis is wrong and the LQ proxy diverges materially from network behavior. |
| **Falsifies Phase 1's recommended order** — the reordered plan is in §11. |

### 7.2 Desirable

| Analysis | Tests | What it does | Cost | Why desirable |
| --- | --- | --- | --- | --- |
| **Steady-state Jacobian eigendecomposition vs SISU** (issue 17137e6, c500097) | Whether SISU reorganises the value-function Hessian — the synthesis's structural-correspondence claim | At fixed point, compute $A_{\text{cl}} = \partial f/\partial x$, eigendecompose; compare across $\beta\in[0,1]$ | Minutes/ckpt | Direct test of synthesis Prediction 2: directional eigenstructure, not global magnitude. If under SISU=1 the eigenvalues shift along disturbance-coupled directions but not globally, the network is doing structural-Hessian shaping. |
| **Curl-vs-gust velocity-coupling asymmetry** | Velocity-coupled perturbations push velocity *down*; velocity-independent perturbations leave it free; rlrmp sees the same null in both | Compute $\partial v_{\text{peak}}/\partial \|\delta\|$ separately for curl-strength and gust-strength sweeps on the trained checkpoints (no retraining) | Hours | If the slopes differ (curl pushes velocity down, gust does not), the network *is* sensitive to the velocity-coupled structure but in the suppress direction; the symmetric null implies neither pull is strong. |
| **Regime-shift / discontinuity scan** | Synthesis Prediction 3: qualitative discontinuity at threshold, not slope | Plot peak velocity, induced gain, kinematic dictionary identity (e.g., bell-shape vs double-bump, time-to-peak) as a function of `pert_std` during training, with fine spacing near the Phase 5 r=0.10 transition | ~1 day | Looks for the C&S-style "penguin shuffle" — abrupt motor-dictionary change. Phase 5 already shows a sharp r=0.12 vs r=0.10 transition (in task-failure mode); a clean robustness-driven version would corroborate. |
| **Adversary-strategy verification** (issue 89891ab) | Whether the trained adversary finds an *exploitable* axis or a generic noise direction | Cluster `GaussianBumpAdversary` final parameters; project onto velocity-coupled vs. velocity-orthogonal subspaces; correlate with controller weakness directions from §6.2 cross-partial | Hours | If the adversary lives in directions orthogonal to velocity coupling, that's a per-design constraint of the adversary class, not a property of the controller. |
| **Plant-perturbation responses with SISU** (issue 25dcc16) | C&S's curl-field test of trained networks | Apply curl fields of various strengths at SISU=0 vs SISU=1; measure trajectory straightness, lateral deviation, peak velocity | Hours | Already-issue-tracked. Provides the experimental phenotype most directly comparable to C&S behavioral data. |
| **Tangling and PCA during reaching** (issue bde625b) | Whether SISU separates trajectories or compresses them | Compute neural tangling $Q$ across SISU levels | Hours | Multi-line corroboration: if SISU does not reduce tangling, the network is treating $\beta$ as a label, not as a control axis. |

### 7.3 Auxiliary

| Analysis | Tests | Why auxiliary |
| --- | --- | --- |
| **Curvature regularisation (Moosavi-Dezfooli style $\|\nabla^2_x L\|$ penalty)** | Direct second-order training | Falls naturally out of synthesis §3, but already strongly addressed by the $\Delta A$-adversary route. Worth keeping as a falsifier if (i)–(iii) all underwhelm. |
| **Antagonist-muscle / stiffness-eigenstructure plant** (Franklin et al. 2007 reproduction) | Whether stiffness eigenstructure aligns with disturbance eigenstructure under H∞ | Not needed to explain the C&S null (their plant is also a frictionless point mass). Needed to test Franklin's directional-stiffness predictions. Separate research track. |
| **Single-unit stimulation** (issue ab37a70) | Functional unit roles | Useful for unit-level interpretation; not on the path to resolving the velocity null. |
| **Motor-noise scaling / speed-accuracy tradeoff** (issue 6d62018) | Whether motor noise creates velocity ceiling | Possible alternate explanation, but `motor_noise_std=0.01` is small relative to perturbation effects; unlikely to be load-bearing. Worth a quick sweep. |
| **Variable reach length** (issue 65156e8) | Robustness of velocity result to reach geometry | Confound rather than diagnostic at this stage. Defer until the constant-length result is mechanistically resolved. |
| **Frequency response of feedback-to-output** (issue 5efa4ca) | Bandwidth/phase characteristics | Synthesis-aligned but redundant with induced-gain after §6.1 is run. |
| **Unit preferences / tuning curves** (issue c5fc272) | Cortical-style unit tuning | Unit-level descriptive; auxiliary to the central H∞ question. |
| **Single-unit perturbation (Part 2 unit_perts.py)** | Causal unit roles | Same. |
| **LEQG as a separate training-method axis** | Cumulant-tilt vs class-lift | Worth a tracking issue (it's a separate method beyond CVaR, with cleaner control). Auxiliary because the prediction is that without (b), it stays in flavor (a). |

---

## 8. Cross-Walk of the 26 Open Issues

Grouped by tier. For each: brief description and how the synthesis/Phase 1 framing changes its priority.

### 8.1 Essential / promoted

| Issue | Description | Reason now essential |
| --- | --- | --- |
| 17137e6 | Steady-state fixed points and Jacobian eigendecomposition | Direct test of synthesis Prediction 2: eigenstructure changes with SISU, not global magnitude. Should be run with explicit SISU sweep at each checkpoint and reported as eigenvalue *trajectories*. |
| c500097 | SISU sensitivity via Jacobian decomposition | Same family; complements 17137e6. Should be the entrypoint for the cross-partial analysis (§6.2). |
| 89891ab | Adversary strategy verification | Pre-condition for trusting any minimax result. We need to know what subspace the `GaussianBumpAdversary` actually explored before concluding it failed. |
| 25dcc16 | Plant perturbation responses (curl) with SISU | Most direct C&S-style behavioral comparison. Run this on every essential-tier checkpoint. |

These four were already in the "Part 2 analysis" umbrella. The Phase 1 framing keeps them where they are but reorders them ahead of the rest.

### 8.2 Desirable

| Issue | Description | Position |
| --- | --- | --- |
| 753508c | Formal BCS/DAI/PAI-ASF analysis | Useful synthesis-level documentation; framework now more mature. Convert to a write-up after Tier 1 results land. |
| a3edc0c | Supplementary BCS/DAI/PAI-ASF experiment | Test of methodological alternatives — corroborating, not central. |
| 31043a5 | Adaptation vs robustness distinguishability | Phase 3's APT ↓ lateral-deviation result already provides one signal; full distinguishability test belongs here. |
| 84ee4ff | APT/CVaR worst-case methods | The umbrella for §4.1 LEQG-as-separate-axis discussion. Add LEQG as an explicit child item. |
| 1ad3c16 | Principled choice of evaluation perturbation amplitude | Becomes essential once the H∞ Riccati picks a $\gamma$; pre-registered evaluation amplitudes from the Riccati are more defensible than ad hoc choices. |
| 609247c | Feedback perturbation responses with SISU | Multi-line corroboration; informs §6.1 induced-gain interpretation. |
| bde625b | PCA and tangling during reaching | Synthesis Prediction 3 / regime-shift story. |
| c460681 | SISU step perturbation | Establishes causality of SISU on policy; complementary to behavioral measurements. |
| 6401524 | Part 1 feedback perturbation responses | Baseline; informs interpretation of Part 2 SISU comparisons. |
| 1090b47 | Part 1 plant perturbation (curl) | Same; supplies the without-SISU control. |

### 8.3 Auxiliary / downgraded

| Issue | Description | Reason downgraded |
| --- | --- | --- |
| ce34c2c | GRU laziness / architecture choice | Phase 1: vRNN matched GRU at 0% Δvel — architecture is not the bottleneck. Keep as a single eigenvalue-update-gate sanity check. |
| 6d62018 | Motor noise scaling | Possible alternate but `motor_noise_std=0.01` is unlikely to dominate. One quick sweep. |
| 65156e8 | Variable reach length | Confound at this stage. Defer. |
| 5efa4ca | Frequency response of feedback-to-output | Subsumed by induced-gain measurement. |
| c5fc272 | Unit preferences | Descriptive unit-level analysis; not on critical path. |
| ab37a70 | Single-unit stimulation | Same. |
| c044a24 | Fixed points during reaching | Useful but redundant with steady-state version 17137e6 for the H∞ question. |

### 8.4 Infrastructure (not analysis)

| Issue | Description | Position |
| --- | --- | --- |
| cdfc132 | Training runtime optimisation | Independent of the analysis question. Will materially help once §4.2/4.3 begin (training the $\Delta A$-adversary needs $\sim$5 PGD inner steps × batch). |
| 216b368 | Pod setup fragility | Operational. |
| a8ed10f | Modal serverless GPU integration | Operational. |

### 8.5 Umbrellas

297260c (Part 1) and 0af472c (Part 2) keep their roles as containers. The Part 2 umbrella in particular should be reframed to make explicit that the velocity-signature null result has prompted a tier reorganisation; the umbrella description should be updated to reflect the synthesis-driven ordering.

### 8.6 Suggested new issues (not yet open)

- **H∞ Riccati offline tool**: implement Q1's pseudocode as `src/rlrmp/analysis/hinf_riccati.py`. Single Python file; integrated with the existing analysis registry. (Closes the "what does the analytical Riccati predict?" question once and for all.)
- **Induced-gain analyser**: per §6.1; module under `analysis/part2/induced_gain.py`. Outputs scalar per checkpoint plus a Toeplitz-singular-spectrum decomposition.
- **Cross-partial analyser**: per §6.2; module under `analysis/part2/cross_partial.py`.
- **`LinearDynamicsAdversary` + feedbax intervenor**: per §5.
- **LEQG loss-aggregator**: per §4.1; ten-line addition to `_CVaRCompositeLoss`'s file with its own class and unit tests.

---

## 9. Pre-Registered Falsification Criteria

Each candidate fix and diagnostic comes with a pre-registered outcome that, if it occurs, falsifies the hypothesis it tests. Stating these in advance turns negative results into structural findings.

| Hypothesis under test | Pre-registered falsifier |
| --- | --- |
| **Plant supports H∞ velocity inflation** (§2, §4.3 Phase A) | If the H∞ Riccati on rlrmp's $A,B,B_w$ with rlrmp's $Q_t,R$, evaluated at $\gamma$ in the well-conditioned range $[1.05\gamma_*,\,2\gamma_*]$, produces $\Delta v_{\text{peak}} < 1\%$ over LQR, then the plant *cannot* generate the C&S signature under any objective form, and the discrepancy is at least partly a parameter-regime issue. ($k$ would need to be reduced, or $Q_t$/$R$ rebalanced, before any training-side fix can land.) |
| **Damping is not the bottleneck** (§2) | If retraining at $k\in\{0.1,1\}$ (with the rest of the training stack fixed) does *not* produce a substantially larger $\Delta v_{\text{peak}}$ at SISU=1 than $k=10$, then either the LQ proxy fundamentally misrepresents the trained controller's headroom, or training is hitting a non-LQ ceiling. Either is a major framework challenge. |
| **Cross-partial alignment is the gate** (§6.2) | If $\nabla^2_{\theta\delta}L$, projected onto a velocity-direction subspace, is comparable in magnitude to its projection onto a lateral-direction subspace at every trained checkpoint, then the cross-partial structure is *not* what blocks the velocity signature, and the explanation must be sought in the inner-max budget or the loss schedule. |
| **Induced gain has a floor** (§6.1) | If APT-trained, CVaR-trained, and Standard-trained models all hit the *same* numerical $\|T_{w\to z}\|_\infty$, then no flavor-(a) training can push past that floor, and only flavor-(b) (4.2) or LEQG (4.1) at $\gamma$ near $\gamma_*$ can do it. If induced gain *does* differ across training methods but velocity signature is still null, the gain is not the volume knob the synthesis predicts. |
| **LEQG alone suffices** (§4.1) | If LEQG with $\gamma(\beta) = \gamma_*\beta/(1+\epsilon)$ on the existing flavor-(a) adversary fails to produce $\Delta v_{\text{peak}} > 5\%$ at SISU=1 even at $\beta=1$, then objective-form-only fixes are insufficient on this $\delta$-class; flavor (b) is required. |
| **$\Delta A$-adversary suffices** (§4.2) | If `LinearDynamicsAdversary` with $\eta(\beta) = \eta_{\max}\beta$, $\eta_{\max}$ chosen near the H∞ boundary, and 5–10 inner PGD steps fails to produce $\Delta v_{\text{peak}} > 5\%$ at SISU=1, then either (a) the training loop is finding a non-saddle stationary point, (b) the GRU lacks expressivity for the H∞ controller, or (c) the synthesis prediction is wrong on this plant. The H∞ teacher (§4.3) discriminates among these. |
| **GRU expressivity** (§4.3) | If the GRU successfully clones the H∞ Riccati controller $K_t$ (low BC loss, near-perfect trajectory match) but the cloned policy still does not show velocity inflation in its standalone evaluation, then the synthesis-predicted velocity signature is *not* a robust property of this plant and is sensitive to noise/disturbance details we have not modeled. |
| **Architecture (GRU laziness)** (issue ce34c2c) | If the eigenvalue-trajectory test across SISU shows magnitudes/angles changing systematically with $\beta$ on a single trained PAI-ASF checkpoint, then GRU laziness is *not* the bottleneck and the architectural concern is closed. (Phase 1 already strongly suspects this is the outcome.) |

---

## 10. Calibration of Expected Effect Size

C&S (Crevecoeur, Cluff, & Scott 2019, "Robust control in human reaching movements") report the velocity inflation under unpredictable disturbance as a substantial qualitative effect — humans systematically reach faster under unpredictable loads, with peak velocity differences of order 10–25% relative to predictable conditions for the published reach geometry (15 cm reaches, ~30 cm/s peak velocity). I have not re-read Fig. 1 directly here, but the magnitudes consistent with the published curves and with §2's H∞ prediction sit in the same range.

Compare to what we measure:
- **Phase 2–6 Δvel SISU=0→1**: $|\Delta v\%| < 1.2$ across every converged condition.
- **Synthesis prediction at $\gamma\sim 1.5\gamma_*$ on the rlrmp plant** (§2 table): $\Delta v\% \approx +10$.
- **C&S effect**: $\Delta v\% \approx +10$ to $+25$ (order-of-magnitude).

So the rlrmp result is **at least 8–10× smaller than the prediction**, i.e. ~1/10th the predicted size, not 1/3rd. This is a *qualitative* gap, not a slow approach to the right answer with insufficient data. It is consistent with the explanation that we are in a different regime entirely (flavor (a) on $\mathbb{E}_w$) rather than under-tuned in the right regime.

The order-of-magnitude diagnosis matters for the falsifiers: a fix that produces +1–2% velocity inflation would be a partial result, not a full reproduction. Pre-register +5% as the threshold for "candidate works" and +10% as the threshold for "candidate matches the human-data effect size."

---

## 11. Recommended Next Experimental Steps (Sequenced)

1. **H∞ Riccati on the rlrmp plant.** Implement Phase 1's Q1.4 pseudocode as a single Python file. Solve the finite-horizon discrete-time Riccati on rlrmp's $A,B,B_w$ with the time-varying $Q_t$ from `running_cost`+`effector_*_late` schedule and $R = 3\!\times\!10^{-5}\,I_2$. Bisect for $\gamma_*$. Simulate closed-loop reach. Compare peak velocity at $\gamma\to\infty$ (LQR) to $\gamma$ near $\gamma_*$. Pre-registered: §2's prediction is +10–25% velocity inflation over the LQR baseline. **Falsifier**: ${<}1\%$ velocity inflation kills the prediction. **Cost**: hours. **Unlocks**: gives a numerical target for §4.2 and §4.3.
2. **Induced gain $\|T_{w\to z}\|_\infty$ on every Phase 2–6 checkpoint.** Power-iterate the lifted Toeplitz operator. **Falsifier**: identical induced gain across Standard/APT/CVaR (no robustification dimension at all). **Unlocks**: ranks training methods by structural robustness; calibrates how far each is from the Riccati ceiling.
3. **Damping calibration retraining.** Train one Standard + one APT condition at $k\in\{0.1,1\}$ keeping all other hyperparameters fixed. Measure peak velocity SISU=0 vs SISU=1 and induced gain. **Pre-registered prediction (§2)**: $\Delta v\%$ should *grow* as $k$ increases. **Falsifier**: no $k$-dependence at all. **Unlocks**: confirms or refutes the §2 reordering of Phase 1.
4. **Cross-partial analysis on existing checkpoints.** §6.2 procedure. **Pre-registered**: cross-partial small in velocity-aligned directions, larger in lateral-aligned ones. **Unlocks**: rules out "weak adversary."
5. **Implement `LinearDynamicsAdversary` (§5).** ~150 LOC rlrmp + ~30 LOC feedbax. One PR each. Tests: at $\eta>0$ with $u\equiv 0$, the rollout is destabilised relative to $\eta=0$.
6. **Train minimax with `LinearDynamicsAdversary`** on the curl-only ($\Delta A_{vv}$ antisymmetric) and full velocity-row variants. SISU wiring: $\eta(\beta)=\eta_{\max}\beta$. Evaluate Δvel and induced gain. **Falsifier**: $\Delta v_{\text{peak}}<5\%$ at SISU=1 falsifies "flavor (b) suffices." **Unlocks**: production-ready training method.
7. **(Optional) H∞ Riccati teacher.** Distill the $K_t$ from step 1 into the GRU via behavioural cloning. **Falsifier**: BC succeeds but distilled policy shows no velocity inflation in standalone eval — falsifies the synthesis prediction itself on this plant.
8. **(Optional) LEQG loss aggregator.** §4.1 implementation. Test only after §6 is implemented; the prediction is that LEQG without (b) gives partial improvement at most.

This is the same general thrust as Phase 1's "Q1 → Q4 induced gain → Q3 minimal lift → (optional) Q2.iii" but with two structural changes:
- **Step 3 (damping calibration) is added** based on §2's analysis. It is a 1-day experiment that disambiguates Phase 1's hedge about parameter regime.
- **Q2.iii (Riccati teacher) is moved later, not earlier.** Phase 1 recommended Q2.iii first as "controlled diagnostic." But the Riccati simulation alone (step 1) already settles the existence question — the teacher then becomes a refinement, not the entry point.

---

## 12. What This Analysis Does Not Cover (Scope Honesty)

- **No synthesis upgrade.** The synthesis-4 framing is taken as given. Whether the locally-quadratic / Laplace / LQ-slice decomposition is the *right* abstraction for this plant is out of scope; we are asking whether rlrmp's training stack, taken as configured, can produce the C&S signature, not whether the C&S signature is the right object to chase.
- **No active-inference or psychiatry predictions.** The synthesis covers three domains; we test only the motor-control one in this document.
- **No AI-alignment / Goodhart connections.** Synthesis §6 relates the clench operator to alignment-tax phenomena; that bridge is conceptual and not part of this analysis.
- **No claim about biomechanics-Franklin predictions.** The C&S null result is plant-agnostic in the sense that C&S use a frictionless point mass; there is no reason to introduce muscles or stiffness eigenstructure to explain *this* null. Franklin-style stiffness-eigenstructure-aligns-with-disturbance predictions would require a richer plant; that is a separate, lower-priority research track (an "interesting confirmation" rather than a "necessary fix").
- **No quantitative claim about C&S Fig 1 effect size.** The +10–25% range in §10 is consistent with the published curves but I have not re-read the paper here. Step 1 of §11 should be paired with a careful re-read of C&S to produce a numerical pre-registered effect size before $\Delta A$-adversary training begins.
- **No deep architecture exploration.** GRU laziness is downgraded but a single eigenvalue-trajectory test (§7.3 / issue ce34c2c) should still be run. A full architecture sweep (LSTM, transformers, etc.) is not on the critical path.

---

## 13. Bottom Line

The discrepancy between rlrmp and C&S is most likely *objective-form*: rlrmp trains on $\mathbb{E}_w$ over a flavor-(a) (input-instance, additive-force) class, never on $\max_{\Delta A}$ over a flavor-(b) (model-class, structural) class. The damping mismatch ($k=10$ vs C&S $k=0.1$) does *not* explain the null and probably *favors* the velocity-inflation signature once the right training class is in place. The minimal lift is a `LinearDynamicsAdversary` intervenor in feedbax + adversary class in rlrmp, which is ~180 LOC + tests. Two diagnostics — H∞ Riccati simulation and induced gain on existing checkpoints — settle the existence question and quantify the headroom in days. After those land, the $\Delta A$-adversary training run is the production fix.
