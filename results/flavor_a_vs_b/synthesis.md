# Adversarial training with input-instance vs model-class perturbations: flavor-A vs flavor-B

A self-contained technical brief for external review. The document defines
the formal setting, states the central thesis (flavor-A is a strict subclass
of flavor-B), describes the analyser (`||T_{w \to z}||_\infty`) and analytical
(H-infinity Riccati) instruments, summarises what has been measured, and
flags what remains pending from two in-flight subagent analyses.

Issue references use 7-character git-bug ID prefixes (e.g. `c723082`).
Coordination issues for this repo: training-methods `c99ad9d`, analyses
`4d38c15`, phases `b33e8da`. The phase artifact for the empirical anchor is
`results/part2_5/README.md` and `results/part2_5/synthesis_review.md`. For
plant-level conventions, see `src/rlrmp/analysis/hinf_riccati.py` (docstring
in §"Math").

> **Abbreviations** (defined on first use, listed here for quick reference):
> RNN = recurrent neural network; GRU = gated recurrent unit; LTI = linear
> time-invariant; LTV = linear time-varying; LQR = linear-quadratic regulator;
> LQG = linear-quadratic Gaussian (LQR + Kalman filter); LEQG = linear-
> exponential-quadratic Gaussian (Whittle 1981 / Jacobson 1973 risk-sensitive
> control); CVaR = conditional value-at-risk; APT = adversarial perturbation
> training (PGD-style additive-force adversary); PGD = projected gradient
> descent (or ascent, here); PAI-ASF = "perceived adversary intensity / active
> SISU feedback" (the rlrmp scheme that wires SISU into the controller's
> input pathway); SISU = scalar uncertainty input fed to the controller (a
> $\beta \in [0,1]$ knob); H&infin; = H-infinity (worst-case L2 induced gain);
> C&S = Crevecoeur, Cluff, & Scott (2019), "Robust control in human reaching
> movements"; Δv = peak-velocity inflation as a percentage relative to the
> LQR baseline (signed, projected on the reach axis per `f90bf74`).

---

## 1. Notation and setup

### 1.1 Plant model

The biomechanical plant is a planar (2D) point mass with linear viscous
damping and a first-order force filter (low-pass on the controller's
commanded force). After linearisation it is the 6-state continuous-time
system

$$
\dot p = v, \qquad
\dot v = -\frac{k}{m}\, v + \frac{1}{m}\, f, \qquad
\dot f = -\frac{1}{\tau}\, f + \frac{1}{\tau}\, u,
$$

with state $x = (p_1, p_2, v_1, v_2, f_1, f_2) \in \mathbb{R}^6$, control
$u \in \mathbb{R}^2$ (commanded force), effector mass $m$, viscous damping
coefficient $k$ (Ns/m), and force-filter time constant $\tau$ (s).
Discretised at sample time $\mathrm{dt} = 0.01$ s, this gives the LTI plant

$$
x_{t+1} = A\, x_t + B\, u_t,
$$

with $A \in \mathbb{R}^{6\times 6}$ and $B \in \mathbb{R}^{6\times 2}$ as
constructed by `feedbax.mechanics.skeleton.pointmass.linearize_pointmass(
mass, damping, tau, dt)` and used by
`src/rlrmp/analysis/hinf_riccati.py::PlantLinearization`.

Two parameter regimes are relevant:

| Regime | $m$ (kg) | $k$ (Ns/m) | $\tau$ (s) | $\tau_v = m/k$ (velocity-decay timescale) |
|---|---|---|---|---|
| **rlrmp** (current canonical) | 1.0 | 10 | 0.05 | 100 ms |
| **C&S** (Crevecoeur et al. 2019) | 1.0 | 0.1 | 0.06 [to confirm] | 10 000 ms |

The 100$\times$ damping mismatch is real and units-matched; see
`results/part2_5/synthesis_review.md` §2 for the verification. (Confirmed
that $k/m$ has units 1/time and is invariant under workspace rescaling.)

### 1.2 Closed-loop with neural controller

The controller is an RNN (typically a GRU with hidden dimension 180; vRNN
also tested) that observes a delayed, noisy version of the effector state
plus task inputs and emits the commanded force $u_t$. Concretely the
augmented closed-loop state stacks plant ($n_p = 6$) + RNN hidden state
($\sim 180$) + sensory delay queue ($\sim 30$ for delay$=5$ on a 4-d
observation) into $\xi_t \in \mathbb{R}^{n_\mathrm{aug}}$ with
$n_\mathrm{aug}\approx 216$ for the canonical Part 2.5 ensemble (see
`results/part2_5/runs/induced_gain_first_run/notes.md`, where
$n_\mathrm{ctrl}\approx 393$ is reported because the network adapter packs
hidden + input + output + encoding states; the relevant LTV state for the
analyser is the linearised closed loop). After linearising the closed loop
along a nominal reach trajectory we get the LTV system

$$
\xi_{t+1} = A_{\mathrm{cl},t}\, \xi_t + B_{w,t}\, w_t,
\qquad
z_t = C_{z,t}\, \xi_t + D_{zw,t}\, w_t,
$$

for $t = 0, \ldots, T-1$, where $T$ is the trial horizon (canonical
$T = 200$ steps = 2 s), $w$ is the disturbance input, and $z$ is a chosen
performance output.

### 1.3 Disturbance and performance channels

The induced-gain analyser supports three $w$ channels and four $z$
channels (`src/rlrmp/analysis/induced_gain.py` `W_*` and `Z_*` constants):

- $w$ channels:
  1. `additive_force` — flavor-(a): $w \in \mathbb{R}^{T\times 2}$, additive
     force at the effector. Identical channel to APT/`GaussianBumpAdversary`.
  2. `sensory_perturbation` — deterministic L2-bounded perturbation on the
     RNN's observation pathway. (H&infin; framing makes no distinction
     between this and stochastic noise of the same covariance budget.)
  3. `structural_da` — small-gain framing for flavor-(b): the structural
     uncertainty $\Delta A$ is bounded by an operator-norm budget, and the
     analyser computes $\|T_{w_\mathrm{struct}\to z_\mathrm{struct}}\|_\infty$
     with $w_\mathrm{struct} \in \mathbb{R}^{n_\mathrm{aug}}$ entering as
     $+w$ on the next-step augmented state and $z_\mathrm{struct} = x$.
     Small-gain margin against unstructured $\Delta A$ is then
     $1 / \|T\|_\infty$.

- $z$ channels: `qr_cost` (cost-matched to Riccati, the only $z$ literal-
  comparable to $\gamma_*$); `control` ($u$); `state_error`
  ($x_t - \bar{x}_t$); `peak_velocity` (forward / lateral velocity per
  `f90bf74`).

### 1.4 The induced gain $\|T_{w\to z}\|_\infty$

For the LTV operator $T : w_{0:T-1} \mapsto z_{0:T-1}$ defined by §1.2's
recursion, the analyser computes

$$
\gamma_\mathrm{net} \;=\; \|T\|_\infty \;:=\; \sup_{w \in \ell_2 \setminus \{0\}}
\frac{\|z\|_2}{\|w\|_2} \;=\; \sigma_\mathrm{max}(T),
$$

i.e. the largest singular value of the lifted Toeplitz operator. This is the
canonical worst-case L2 induced gain of the closed loop and is a function of
both the trained controller and the linearisation. See
`src/rlrmp/analysis/induced_gain.py` for the production implementation.

---

## 2. The flavor-A / flavor-B distinction (formal)

We distinguish two adversary classes used in the inner $\max$ of a minimax
training objective $\min_\theta \max_{\delta \in \Delta} \mathbb{E}\, J(\theta, \delta)$:

### 2.1 Flavor-A: input-instance perturbation

The adversary is a force trajectory $w \in \mathcal{W}_\mathrm{A}
= \{w \in \mathbb{R}^{T\times 2} : \|w\|_2 \le \epsilon\}$, additive at the
effector channel:

$$
x_{t+1} = A\, x_t + B\, u_t + B_w\, w_t,
\qquad B_w = B \;\;(\text{or}\;B_w \propto e_v\text{ on the velocity row}).
$$

The adversary is trained per-trial via PGD on $w$ within an L2 ball.
Implementations: `APTTrainingWrapper`, `GaussianBumpAdversary` (issue
`c723082` body §1; class lives at `src/rlrmp/adversary.py`).

### 2.2 Flavor-B: model-class perturbation

The adversary is a perturbation matrix $\Delta A \in \mathcal{W}_\mathrm{B}
= \{\Delta A \in \mathbb{R}^{n_\mathrm{vel}\times n_\mathrm{state}} :
\|\Delta A\|_F \le \eta\}$, applied **uniformly along the rollout** as

$$
\dot v \mathrel{+}= \Delta A \, [p, v]^\top,
$$

equivalently embedded into the discrete plant as an additive force
$f = m\, \Delta A\, x$ in the existing force channel (the
`DynamicsMatrixPerturb` intervenor in feedbax `develop`,
`feedbax/intervene/intervene.py:228`: `df = self.mass * (params.delta_A @ x)`).
The adversary is trained per-trial via PGD with Frobenius-ball projection
(`LinearDynamicsAdversary._frobenius_project`,
`src/rlrmp/adversary.py:130-146`). SISU wiring: $\eta(\beta) = \eta_\mathrm{max}\,\beta$.

### 2.3 Why flavor-A $\subsetneq$ flavor-B (the central thesis)

Given a flavor-B adversary with structural matrix $\Delta A$ and a fixed
controller, the resulting closed-loop disturbance is the time-varying force
trajectory $w_t = m\, \Delta A\, x_t$ — i.e. a *state-coupled* (multiplicative,
realised through the trajectory) instance of the flavor-A channel. Since
this $w$ is in $\mathcal{W}_\mathrm{A}$ for sufficiently large flavor-A
budget $\epsilon$ (after projection), every flavor-B perturbation can be
realised as a flavor-A instance. The converse fails: a generic flavor-A
trajectory $w$ does not factor as $m \Delta A x$ for any single $\Delta A$
held constant across the rollout. Equivalently, the flavor-B class consists
exactly of those force trajectories that lie in the image of the
state-realised linear map $x \mapsto m\, \Delta A\, x$ for some
constant $\Delta A$ on a Frobenius ball; flavor-A fills out the rest of the
ball.

This is the strict-inclusion ⊊ structure: every flavor-B instance is a
flavor-A instance, but flavor-A admits force trajectories that no flavor-B
adversary can produce. The robust-control consequence is that flavor-A
training over-prepares for an ostensibly larger class while *under-preparing*
for the structurally coupled subclass — flavor-A's worst-case is over a class
that is too generic to inform the value-function Hessian along
state-coupled directions.

### 2.4 Why we expect them to differ

The H&infin; Riccati game (§3.1) for an additive-force disturbance
$\dot x = Ax + Bu + B_w w$ produces a feedback law $K^{(a)}$ that minimises
the worst-case L2 gain from $w$ to a chosen $z = (Q^{1/2}x, R^{1/2}u)$.
For a structural disturbance $\dot x = (A + \Delta A) x + Bu$, the analogous
robust feedback law $K^{(b)}$ enters via a *multiplicative* uncertainty
budget. The two feedback laws differ; in particular, $K^{(b)}$ inflates
state-feedback gains in the directions along which $\Delta A$ has support,
producing the characteristic C&S "stiffer feedback at higher SISU" signature.
Empirically we expect:

- **γ_sd (structural-`ΔA` channel induced gain)** to be *lower* for
  flavor-B-trained controllers than flavor-A-trained ones at matched
  effective adversary budget — the small-gain margin against unstructured
  $\Delta A$ should improve when training has actually seen $\Delta A$.
- **Peak forward velocity Δv**, evaluated under appropriate disturbance
  conditions, to inflate (positive Δv) under flavor-B in the direction
  predicted by C&S (per the H&infin; Riccati prediction tabulated below).
- **γ_af (additive-force channel induced gain)** *not* to discriminate
  cleanly — both flavors solve the additive-force minimax, just with
  different effective $B_w$. Empirically (induced-gain first run, §5.2),
  γ_af is the *least* discriminating channel.

---

## 3. The induced-gain analyser ($T_{w\to z}$)

### 3.1 Operator definition and channel choices

The analyser (`src/rlrmp/analysis/induced_gain.py`, issue `74bfd86`) computes
$\|T\|_\infty$ for any of $\{w\} \times \{z\}$ channel pairs. The headline
channel pair for cross-method comparison is `additive_force × qr_cost`,
where the cost-matched output

$$
z_t = \begin{pmatrix} Q_t^{1/2}\, x_t \\ R_t^{1/2}\, u_t \end{pmatrix}
$$

makes $\gamma_\mathrm{net} = \|T_{w\to z}\|_\infty$ literal-comparable to
the H&infin; Riccati's $\gamma_*$ on the same plant + cost schedule
(round-trip identity: a Riccati controller designed at $\gamma_\mathrm{des}
= 1.5\,\gamma_*$ achieves
$\gamma_\mathrm{net} \in (\gamma_*,\, \gamma_\mathrm{des}]$, the
suboptimality band per `3c74e3b`).

### 3.2 Two algorithms

- **LTV power iteration on the Toeplitz operator** (primary for the
  trajectory linearisation). Forward sweep computes $z = Tw$ via
  `jax.lax.scan`; adjoint sweep is generated with `jax.linear_transpose`
  (avoiding manual backward-time recursion sign pitfalls). Iterates with
  3 random restarts; the largest restart estimate is reported. Convergence
  test: relative change in gain estimate < `rtol` (canonical `rtol=1e-6`,
  `max_iter=600`) for two consecutive iterations.
- **Hamiltonian bisection on the LTI fixed-point linearisation** (auxiliary).
  Bisects on $\gamma$ and tests admissibility via the discrete-time
  bounded-real-lemma Riccati. Equivalent to the symplectic-Hamiltonian
  eigenvalue test but reuses the production primitive. Used as an internal-
  consistency check on long-hold trajectories.

The two methods agree on a long-hold trajectory by construction: the
trajectory tail linearisation matches the fixed-point linearisation.

### 3.3 The recent probe diagnosis (settled)

A diagnostic probe (`scripts/probe_round_trip_ratio.py`, issue `74bfd86`
comment `7849b8f`) found three apparent anomalies in the round-trip
identity, all of which resolved to expected behaviour:

1. **PI non-convergence at `rtol=1e-9`** was a methodological misuse: the
   leading-singular-value gap at long horizons cannot satisfy `rtol=1e-9`
   inside `max_iter=600`. Loosening to `rtol=1e-6` gives `conv=yes` at 890
   iterations with the same gain to 4 sig figs. Probe `rtol` lowered to
   `1e-6`.
2. **Horizon non-stationarity**: short-horizon `find_gamma_star` finds a
   smaller infimum than the long-horizon limit ($\gamma_*$ jumps from
   0.00943 at $n=100$ to 0.01375 at $n=200$). The 1.21 plateau at long
   horizons is the suboptimality margin of an H-infinity controller designed
   at the loose $1.5\,\gamma_*$ design level — not a bug.
3. **$Q_f$-scale collapse** at $\mathrm{qf\_scale} \ge 10$: $\gamma_*$ is
   essentially $Q_f$-independent for $\mathrm{qf\_scale} \le 1.0$ (0.2%
   effect); only at $\mathrm{qf\_scale} \ge 10$ does terminal-state
   admissibility dominate. Canonical $\mathrm{qf\_scale}=1.0$ avoids this
   regime.

**Verdict: canonical $Q_f=1.0$, $n=200$ is fully trustworthy.**

### 3.4 What the analyser is meant to reveal

A single scalar ($\gamma_\mathrm{net}$) per checkpoint, per channel pair —
the equivalent worst-case linear gain of a possibly nonlinear closed-loop
policy. The synthesis review prediction (`results/part2_5/synthesis_review.md`
§6) is that:

- $\gamma_\mathrm{net}$ on the headline `additive_force × qr_cost` channel
  should rank training methods by the structural robustness of the
  controller they produce: APT < Standard < no-perturbation baseline (on
  flavor-A); flavor-B < flavor-A on `structural_da × qr_cost`.
- The gap between $\gamma_\mathrm{net}$ and the analytical $\gamma_*$ on the
  same plant + cost schedule quantifies how far the trained controller is
  from the Riccati ceiling.

---

## 4. H-infinity Riccati synthesis ($\gamma_*$, design controller)

### 4.1 The Riccati recursion

The discrete-time finite-horizon LQ-game value function is
$V_t(x) = x^\top P_t x$ with backward recursion (Basar-Bernhard form, used
in `src/rlrmp/analysis/hinf_riccati.py`):

$$
P_t = Q_t + A^\top P_{t+1}\!\left(I + (B R_t^{-1} B^\top - \gamma^{-2} B_w B_w^\top)\, P_{t+1}\right)^{-1}\! A,
\qquad P_T = Q_f.
$$

The H&infin; feedback gain is

$$
K_t = (R_t + B^\top \Lambda_t B)^{-1} B^\top \Lambda_t A,
\qquad
\Lambda_t = (I - \gamma^{-2} P_{t+1} B_w B_w^\top)^{-1} P_{t+1}.
$$

$\gamma_*$ is the smallest $\gamma$ for which the Riccati is solvable
(equivalently, $I - \gamma^{-2} B_w^\top P_{t+1} B_w$ is positive definite at
every step). The production module bisects on $\gamma$ to find $\gamma_*$,
then synthesises $K_t$ at $\gamma_\mathrm{des} = 1.5\, \gamma_*$.

### 4.2 The flavor-(a) / flavor-(b) issue with $B_w$

Production's $B_w$ is the **flavor-(a) additive force channel**: $B_w = B$
(or its velocity-row projection), so the disturbance $w$ enters via the
same channel as the control $u$. This is mathematically clean and matches
the APT/`GaussianBumpAdversary` training setup, but it does **not**
correspond to C&S's structural disturbance $(A + \Delta A) x$, where the
disturbance enters multiplicatively on the state. The two formulations
yield different feedback gains and different velocity-inflation predictions.

Empirical anchor for the diagnosis
(`tests/test_hinf_riccati.py::test_cs_faithful_qr_velocity_inflation`,
xfailed; documented in coord `c99ad9d` comment `cd979fa` and analyses coord
`4d38c15` comment `4ad0368`):

| Q,R schedule | Plant | Δv at $1.5\gamma_*$ |
|---|---|---|
| Faithful C&S Eq. 15 | C&S ($k=0.1$) | **−0.04%** |
| rlrmp Q,R | C&S ($k=0.1$) | −0.77% |
| rlrmp Q,R | rlrmp ($k=10$) | +10.8% |

**Implication: production Riccati's $B_w$ is most likely flavor-(a) additive
force, not flavor-(b) `ΔA·x` model-class.** Reproducing the C&S "robust >
LQG via gain inflation alone" finding on the unperturbed forward channel
requires the multiplicative `ΔA` formulation. The flavor-(b) Riccati
extension (issue `97c227a`) is the open analytic work item.

### 4.3 Why this matters for the comparison

A flavor-(b) Riccati gives:

1. The **analytic teacher** for a behavioural-cloning-style training
   approach (issue `db35426` H&infin; Riccati teacher distillation).
2. The **baseline** against which `LinearDynamicsAdversary`-trained
   (flavor-B) controllers can be compared: does empirical
   $\gamma_\mathrm{net}$ from the analyser approach the analytical
   $\gamma_*$ under flavor-(b) $B_w$?
3. The **bridge** to the C&S replication: closing the C&S Δv gap on the
   $k=0.1$ plant requires switching from the flavor-(a) to the flavor-(b)
   $B_w$ formulation in the Riccati synthesis.

---

## 5. Existing experimental results

### 5.1 Phase context (Part 1 / Part 2 / Part 2.5)

Two prior phases (Part 1 umbrella `297260c`, Part 2 umbrella `0af472c`)
established analysis modules pre-SISU and post-SISU respectively. **Part
2.5** (no umbrella; artifacts `results/part2_5/README.md` and
`results/part2_5/synthesis_review.md`) was the perturbation-training and
loss-mode exploration that produced the SISU-velocity null result: across
every converged condition (standard backprop, CVaR 10%, APT with 8
hyperparameter variants, adaptive control cost, center-out task, and
parametric `GaussianBumpAdversary` minimax), $|\Delta v_\mathrm{peak}^{
\beta=0\to 1}| < 1.2\%$ at SISU=0 vs SISU=1 — at least 8–10$\times$ smaller
than the C&S effect size of +10–25%. Phase 6 (parametric minimax) is the
flavor-A high-water-mark; the phase pivot was to the methodology-fix branch
(umbrella `b557d4e`, now closed; child issues continue) which posed the
flavor-A vs flavor-B reframing.

### 5.2 Induced-gain first run (Part 2.5 baselines)

`results/part2_5/runs/induced_gain_first_run/` (issue `6fdf9a4`,
commit `36aa1ad`) ran the analyser on 10 Part 2.5 training-method groups
(replicate 0 each, except the `mult_single` outlier replaced with rep 2 —
see hygiene comment `4fd0388`). Single canonical reach (15 cm forward,
SISU=0.5), $T=200$ steps, all 30 (group $\times$ $w$ channel) cells
converged. Riccati baseline $\gamma_* = 0.013749$.

| Group | $\gamma_\mathrm{af}$ | $\gamma_\mathrm{sd}$ | $\gamma_\mathrm{sp}$ | $\gamma_\mathrm{af}/\gamma_*$ |
|---|---|---|---|---|
| `baseline_standard_12k` | 0.124 | 148.5 | 2.61 | 9.0 |
| `vanilla_single` | 0.248 | 169.3 | 4.33 | 18.0 |
| `vanilla_pop5` | 0.146 | 164.6 | 4.41 | 10.6 |
| `minimax_single_seed0` | 0.145 | 153.4 | 2.14 | 10.5 |
| `minimax_single_seed1` | 0.156 | 162.8 | 1.32 | 11.4 |
| `minimax_single_seed2` | 0.156 | 162.9 | 1.45 | 11.3 |
| `mult_single` (rep 2 replacement) | 0.183 | — | — | 13.3 |
| `mult_pop5` | 0.179 | 165.1 | 4.22 | 13.0 |
| `ratio03_single` | 0.171 | 163.4 | 2.64 | 12.4 |
| `ratio03_pop5` | 0.158 | 164.5 | 2.10 | 11.5 |

Channels: `af` = additive_force, `sd` = structural_da, `sp` =
sensory_perturbation; all $\times$ `qr_cost` $z$.

**Three cross-cutting findings** (coord `4d38c15` comment `4dd33b8`):

1. **$\gamma_\mathrm{af}$ on a single canonical reach is *not* a sensitive
   cross-method discriminator.** Non-outlier groups span $[0.124, 0.248]$ —
   a factor $\sim 2$ spread, $\sim 9$–$18\times$ above $\gamma_*$. Standard
   backprop, vanilla, minimax seeds, and `mult_pop5` all sit in this band.
   The H&infin; norm on the *trained-against* channel does not separate
   flavor-(a) minimax from standard backprop on this geometry.
2. **$\gamma_\mathrm{sp}$ cleanly separates minimax from non-minimax.**
   Minimax seeds: $\gamma_\mathrm{sp}\in[1.32, 2.14]$; vanilla/baseline:
   $[2.61, 4.40]$. $\sim 2\times$ better attenuation. *Empirical headline
   channel.*
3. **$\gamma_\mathrm{sd}$ is uniformly large ($\sim 150$–$170$) across all
   groups.** Consistent with the absence of flavor-(b) training in this
   batch. Until a flavor-(b)-trained checkpoint exists, $\gamma_\mathrm{sd}$
   cannot discriminate (a) from (b).

**(a) ⊊ (b) thesis empirical status: NOT established by this run.** The
sharp test is blocked on a flavor-(b)-trained network — i.e. a
`LinearDynamicsAdversary` checkpoint, which is exactly the in-flight
training run in §6.

### 5.3 Peak-velocity ratio (Δv) results across methods

The $\Delta v$ headline scalar is the percentage change in **peak forward
velocity** (signed projection onto the reach axis, per `f90bf74`) of the
trained controller relative to the LQR baseline on the same plant + cost
schedule, evaluated under the H&infin; design $\gamma_\mathrm{des}$. It is
distinct from peak speed (2-norm of velocity); the corrected metric also
exposes peak lateral velocity as `delta_v_lateral_percent`.

The Δv numbers established so far (all on the **analytical Riccati**, not on
trained controllers — these are the H&infin; *predictions* that trained
controllers should approach if training is successful):

| Q,R schedule | Plant | $\gamma_\mathrm{des}/\gamma_*$ | Δv (peak forward, signed) |
|---|---|---|---|
| rlrmp Q,R | rlrmp ($k=10$) | $1.5$ | **+10.8%** |
| rlrmp Q,R | rlrmp ($k=10$) | $1.2$ | +18.8% |
| rlrmp Q,R | rlrmp ($k=10$) | $1.05$ | +27.2% |
| Faithful C&S Eq. 15 | C&S ($k=0.1$) | $1.5$ | −0.04% (xfail) |
| rlrmp Q,R | C&S ($k=0.1$) | $1.5$ | −0.77% |

Source: `results/part2_5/synthesis_review.md` §2 + analyses coord `4d38c15`
comments `4ad0368` and `48d13a8` (metric correction).

The directionality of the finding inverts the Phase 1 worry that high
damping suppresses the velocity-inflation signature: the rlrmp regime
*amplifies* the predicted effect (LQR baseline is under-powered against the
$10/$s velocity decay, leaving headroom for H&infin; gain inflation), while
C&S's near-frictionless plant has very little H&infin; headroom over LQR
*on the flavor-(a) $B_w$ formulation*. The C&S-regime −0.04% on flavor-(a)
is not a bug of the rlrmp implementation — it is direct evidence that
recovering the C&S signature requires the flavor-(b) $B_w$ formulation
(issue `97c227a`).

The peak-velocity ratio of trained controllers (Part 2.5 §5.1) was at most
$|\Delta v| < 1.2\%$ across every flavor-A method, an order of magnitude
short of the +10–25% prediction.

### 5.4 The xfailed C&S-faithful Riccati test

`tests/test_hinf_riccati.py::test_cs_faithful_qr_velocity_inflation`
(commit `cd22999`, merged via `cd229998` xfailed). The xfail reason string
carries the full diagnosis: even with the paper's faithful Eq. 15 Q,R on
the C&S plant, Δv at $1.5\gamma_*$ is −0.04% — squarely against C&S's +
direction. Q,R-shape is *not* the variable; the disturbance channel
formulation is. This is the empirical anchor for the open issue `97c227a`
(Riccati flavor-(b) extension).

---

## 6. The new flavor-B training run

### 6.1 Setup

`scripts/train_minimax.py --adversary-type linear_dynamics` (added under
issue `c723082`, merged via `2e21833`). Sweep:

- $\eta_\mathrm{max} \in \{0.03, 0.10, 0.30\}$ (Frobenius-ball radius for
  `||ΔA||_F ≤ η_max·SISU`)
- 3 seeds (different RNG keys)
- 5 internal replicates per training run (vmapped)

Total: $3 \times 3 \times 5 = 45$ trained replicates across 9 distinct
training configurations.

### 6.2 LinearDynamicsAdversary mechanism

Per-trial PGD (default 5 inner steps, learning rate $10^{-2}$) on
$\Delta A \in \mathbb{R}^{n_\mathrm{vel}\times n_\mathrm{state}}$ with
projection onto the Frobenius ball at radius $\eta_\mathrm{max}$. The
$\Delta A$ matrix is held *uniformly* in time across the rollout (one
$\Delta A$ per trial) and acts via `feedbax.intervene.DynamicsMatrixPerturb`
inside `LTISystem.vector_field`:

$$
\dot v \mathrel{+}= \Delta A \, [p, v]^\top
\quad\Longleftrightarrow\quad
f_t \mathrel{+}= m\, \Delta A\, x_t,
$$

i.e. the structural model-class perturbation is realised through the
existing force-channel embedding (`src/rlrmp/intervention_compat.py:139-181`)
because the discrete-time plant has $B$ on the velocity row. SISU gating
applies via the intervenor's `params.scale = SISU` field, so the
*budget* at full SISU=1 is $\eta_\mathrm{max}$ and the budget at SISU=0
collapses to 0.

### 6.3 Logging

- Hyperparameters: `results/flavor_a_vs_b/runs/<run>/run.json` per the
  artifacts policy (`<group>__<variant>` naming).
- Heavy outputs (model `.eqx`, training logs, large `.npz`):
  `_artifacts/flavor_a_vs_b/runs/<run>/`.

### 6.4 Loss trajectories

All 9 training configurations (3 $\eta_\mathrm{max}$ × 3 seeds, 5 internal
replicates each) converged. Warmup `ctrl_loss` plateaus at ≈ 14.2 across
groups; the adversarial phase begins at ≈ 9.4 and descends to a final
range of 4.10–5.34 by batch 5000 with zero NaN/Inf and zero >20% spikes
across 90 phase transitions.

Per-condition aggregate `ctrl_loss` (mean ± SD across 3 seeds × 5
replicates):

| $\eta_\mathrm{max}$ | warmup end | adv 0 | adv 1k | adv 2.5k | adv 5k (final) |
|---|---|---|---|---|---|
| 0.03 | 14.32 | 9.40 | 8.43 | 6.39 | **4.75 ± 0.26** |
| 0.10 | 14.13 | 9.38 | 8.30 | 7.29 | **5.12 ± 0.22** |
| 0.30 | 14.16 | 9.43 | 7.94 | 6.82 | **4.70 ± 0.18** |

Stability of the per-batch trajectory is monotonic in $\eta_\mathrm{max}$:
the $\eta=0.30$ runs are the cleanest (87% mean monotonicity), $\eta=0.03$
is intermediate (81%), and $\eta=0.10$ shows transient oscillation in
the batch 3000–4500 window with one seed dipping to 60% (group mean 70%).

Final loss is **non-monotonic in $\eta_\mathrm{max}$** — a U-shape at
4.75 → 5.12 → 4.70. The middle $\eta_\mathrm{max}$ trains *worse* than
either extreme. This is unexpected if loss were a smooth proxy for the
quantity of interest; we re-examine the U-shape against the induced-gain
results in §7.1 (the U-shape does *not* appear there, suggesting it is a
saddle-dynamics or fused-loss reporting artifact rather than a real
robustness inversion).

$\|\Delta A\|_F$ saturates at $\eta_\mathrm{max}$ in every run, confirming
the inner PGD is at the budget bound and the Frobenius constraint is
active throughout the adversarial phase.

Cross-references:

- Per-run hyperparameters: `results/part2_5/runs/flavor_b_eta{0.03,0.10,0.30}__seed_{0,1,2}/run.json`.
- Parsed loss table: `results/part2_5/flavor_b_summary.json`.
- Bulk artifacts (`.eqx`, `.npz`, training logs): `_artifacts/part2_5/runpod/flavor_b/`.
- Issue thread: `c723082` comment from this work bundle.

---

## 7. The two follow-up analyses

### 7.1 (A) Induced gain on flavor-B trained models

**Goal.** Run the induced-gain analyser (§3) on each of the 9
flavor-B-trained groups (3 $\eta_\mathrm{max}$ × 3 seeds, ≥ 2 replicates per
group per the hygiene rule) and compare cross-method against the Part 2.5
flavor-A baselines tabulated in §5.2.

**Pre-registered expectations:**

- **$\gamma_\mathrm{sd}$ (structural-`ΔA` channel)** should be substantially
  *lower* on flavor-B-trained controllers than on flavor-A-trained ones.
  Quantitative target: at least a 2× reduction from the flavor-A median
  $\gamma_\mathrm{sd}\approx 160$ for $\eta_\mathrm{max} \ge 0.10$.
  *Falsifier:* if $\gamma_\mathrm{sd}$ on flavor-B-trained controllers is
  not lower than the flavor-A baseline at $\eta_\mathrm{max} = 0.30$, the
  flavor-(a) ⊊ (b) thesis is empirically falsified — no amount of $\Delta A$
  training is reducing the small-gain margin against unstructured
  $\Delta A$.
- **$\gamma_\mathrm{af}$** should *not* discriminate cleanly (per §5.2).
- **$\gamma_\mathrm{sp}$** behaviour is unconstrained by the thesis; will
  report observationally.
- **Δv at $\gamma_\mathrm{des}$** on the trained controller (read off the
  worst-case $w^*$ trajectory from PI) should approach the analytical
  Riccati prediction; the gap to +10–25% is the second-order test.

**Method.** `src/rlrmp/analysis/induced_gain.py::induced_gain` for each
checkpoint via `_NetworkController` adapter (the workaround used in the
first induced-gain run while the SimpleFeedback cycle-wires fix lands —
issue `53b5fe5`); $T=200$, canonical $Q_f=1.0$, `rtol=1e-6`, 3 PI restarts;
all three $w$ × `qr_cost` cells per group; ≥ 2 replicates per group; report
median + range; document any replicate exclusions per the §5.2 hygiene
comment (`4fd0388`).

**Results.** The analyser was driven by `scripts/run_induced_gain_flavor_b.py`
(mirrors `run_induced_gain_part2_5.py`); cross-method tabulation by
`scripts/build_cross_method_comparison.py`. Per-replicate `gains.json`
under `_artifacts/part2_5/runs/induced_gain_flavor_b/<group>/`; cross-group
`summary.json` at the same root; tracked spec at
`results/part2_5/runs/induced_gain_flavor_b/run.json` (+ `notes.md`); the
side-by-side comparison against Part 2.5 flavor-A baselines is at
`results/part2_5/induced_gain_flavor_b/cross_method_comparison.md`.
Hygiene exclusions per §5.2: 0/45 $\gamma_\mathrm{sd}$ replicates flagged
at the 10× rule; 5/45 $\gamma_\mathrm{af}$ flagged; 8/45 $\gamma_\mathrm{sp}$
flagged (excluded from medians). One markedly elevated $\gamma_\mathrm{sd}$
at $\eta=0.03$ seed_2 rep_4 ($\gamma_\mathrm{sd}=1001.7$) was within the
10× envelope and retained; documented in `notes.md`.

**Headline.** $\gamma_\mathrm{sd}$ on flavor-B-trained controllers,
median across 45 replicates: **154.78**. Flavor-A baseline median (9
groups, excluding the known-degenerate `mult_single` rep_0): **163.39**.
Ratio **0.947** — a ~5% reduction, **within methodological noise**.

**Verdict.** The pre-registered "$\gamma_\mathrm{sd}$ reduction by 2× or
more at $\eta_\mathrm{max} \ge 0.10$" prediction is **not supported**.
Flavor-B-trained controllers and flavor-A-trained controllers are
empirically indistinguishable on $\gamma_\mathrm{sd}$ at this canonical
operating point.

**$\eta_\mathrm{max}$ trend.** $\gamma_\mathrm{sd}$ is **flat** in
$\eta_\mathrm{max}$:

| $\eta_\mathrm{max}$ | $\gamma_\mathrm{sd}$ median | $\gamma_\mathrm{af}$ median | $\gamma_\mathrm{sp}$ median |
|---|---|---|---|
| 0.03 | 152.53 | (per `notes.md`) | 1.39 |
| 0.10 | 156.22 | (per `notes.md`) | 0.95 |
| 0.30 | 154.78 | (per `notes.md`) | 1.25 |

No U-shape in $\gamma_\mathrm{sd}$. The U-shape observed in training
`ctrl_loss` (§6.4: 4.75/5.12/4.70) therefore most likely reflects a
saddle-dynamics or fused-loss reporting artifact, not a real robustness
inversion.

**Surprise positive on $\gamma_\mathrm{sp}$.** Sensory-perturbation
$\gamma_\mathrm{sp}$ is also flat in $\eta_\mathrm{max}$ (1.39 / 0.95 /
1.25), but markedly *lower* than the flavor-A vanilla baseline and
comparable to flavor-A minimax. Flavor-B training appears to confer some
sensory-perturbation robustness despite never training against that
channel — auxiliary, but worth noting.

**Cross-link to §7.2.** The flat-in-$\eta_\mathrm{max}$ behaviour of
$\gamma_\mathrm{sd}$ is consistent with the small lift in $\gamma_*^{(b)}$
predicted by the S-procedure quadratic-stability extension (§7.2:
$\gamma_*^{(b)}$ at $\eta=0.1$ is +2.0% above $\gamma_*^{(a)}$ on the
rlrmp regime). At this operating point, flavor-(a) and flavor-(b) are
not clearly distinguishable by either the analyser or the analytical
synthesiser — see §8 for the joint-result discussion.

Issue thread: comment on `74bfd86` (induced-gain analyser) at comment
`7344b34`.

### 7.2 (B) Riccati flavor-(b) extension

**Goal.** Extend `src/rlrmp/analysis/hinf_riccati.py` so that $B_w$ accepts
flavor-(b) structural disturbance $\Delta A \cdot x$ rather than only the
flavor-(a) additive force. Bisect for $\gamma_*^{(b)}$ on the C&S plant
(matches C&S's setting) and on the rlrmp plant (cross-comparison with the
trained controllers). Issue `97c227a`.

**Pre-registered expectations:**

- **C&S plant** (faithful Eq. 15 Q,R) under flavor-(b) $B_w$ should produce
  Δv > 0 at $\gamma_\mathrm{des} = 1.5 \gamma_*^{(b)}$, qualitatively
  matching C&S Fig 1e. *Falsifier:* if Δv ≤ 0 on the C&S plant under
  flavor-(b) $B_w$ as well, the disturbance-channel diagnosis is wrong and
  some other variable (Q,R schedule, terminal $Q_f$, horizon) is to blame.
- **rlrmp plant** under flavor-(b) $B_w$ should produce Δv comparable to or
  larger than the flavor-(a) prediction (+10.8% at $1.5\gamma_*$); this
  gives the analytical target for the trained-controller comparison in
  §7.1.
- **LQR limit** ($\gamma \to \infty$) should recover nominal LQR $K$
  regardless of disturbance class — sanity check.

**Method.** Per issue `97c227a`'s API sketch, either a new
`solve_hinf_riccati_modelclass(plant, schedule, gamma, *, delta_A_structure,
delta_A_budget)` function or an extended `solve_hinf_riccati` with an
optional `delta_A_constraint` parameter. The mathematical reformulation
treats $\Delta A$ as a multiplicative state-coupled disturbance budget;
small-gain or scaled-bounded-real-lemma framing applies. Tests: convert the
existing xfailed `test_cs_faithful_qr_velocity_inflation` into a passing
test under the new flavor; round-trip identity against the induced-gain
analyser's `structural_da` channel.

**Approach.** S-procedure / quadratic-stability lift. Treating
$\Delta A \cdot x_t$ as a multiplicative state-coupled disturbance with
$\|\Delta A\|_F \le \eta$, Cauchy–Schwarz on the Frobenius operator norm
gives $\|w_t\| \le m\eta\|C_q x_t\|$. A sufficient condition for
flavor-(b) closed-loop performance at level $\gamma$ is then the *same*
flavor-(a) force-channel Riccati at level $\gamma$ but with the running
state-cost $Q$ augmented by $(m\eta)^2 \cdot C_q^\top C_q$. This is a
**conservative** lift: μ-synthesis would be tighter, and a trajectory-
coupled time-varying $B_w(x_t)$ with $w_t = \Delta A \cdot x_t$ is a
separate finite-horizon DRE problem deferred as future work.

**API.** New entry points `solve_hinf_riccati_modelclass`,
`find_gamma_star_modelclass`, `compute_velocity_inflation_modelclass`
in `src/rlrmp/analysis/hinf_riccati.py`. The flavor-(a) interface is
untouched.

**$\gamma_*^{(b)}$ values.**

| Plant | Q,R | $\eta$ | $\gamma_*$ |
|---|---|---|---|
| rlrmp ($k=10$) | rlrmp | 0 (= flavor-(a)) | 0.009427 |
| rlrmp ($k=10$) | rlrmp | 0.1 | 0.009618 (+2.0%) |
| rlrmp ($k=10$) | rlrmp | 0.5 | 0.013365 (+41.8%) |
| C&S ($k=0.1$) | faithful Eq. 15, $\alpha_1=1$ | 0 | 5.898 |
| C&S ($k=0.1$) | faithful Eq. 15, $\alpha_1=1$ | 1 | 5.898 (~0%) |
| C&S ($k=0.1$) | faithful Eq. 15, $\alpha_1=1$ | 10 | 5.900 (+0.04%) |
| C&S ($k=0.1$) | faithful Eq. 15, $\alpha_1=1$ | 100 | 6.103 (+3.5%) |

The C&S regime is essentially flat in $\eta$ because the C&S Q places
$10^6$ weight on position, dwarfing the augmented $(m\eta)^2 \cdot
C_q^\top C_q$ term until $\eta$ reaches order $10^2$.

**Pivotal headline (negative).** The previously-xfailed
`test_cs_faithful_qr_velocity_inflation` does **not** become a passing
test under flavor-(b). Δv at $1.5 \cdot \gamma_*^{(b)}$ on the C&S regime:

| $\eta$ | Δv |
|---|---|
| 0 (flavor-(a)) | −0.039% |
| 0.1 | −0.039% |
| 1.0 | −0.083% |
| 10.0 | −4.11% |
| 100.0 | −52.93% |

Δv stays $\le 0$ and grows **more negative** as $\eta$ increases.
*Mechanism:* augmenting $Q$ by $(m\eta)^2 \cdot C_q^\top C_q$ adds energy
penalty on $[p, v]$, which damps the controller and lowers forward
velocity — the opposite of the C&S "robust = faster reach" signature.

**Implication.** The S-procedure quadratic-stability lift is
*insufficient* to recover the C&S signature on the C&S regime. This does
not refute the flavor-(a) ⊊ (b) thesis broadly — the gap could still be
real but finer than what quadratic stability captures. Tighter
formulations (μ-synthesis, trajectory-coupled time-varying $B_w(x_t)$)
remain candidates and are flagged as natural next steps in §8.

**Tests.** 25 passed, 2 xfailed: the original flavor-(a) C&S xfail is
unchanged, and a new flavor-(b) xfail captures the diagnosis above
(table + mechanism + implication) in its xfail-reason string. All
flavor-(a) tests pass unchanged.

Issue thread: comment on `97c227a`. Cross-cutting comment on `c99ad9d`
(training-methods coord).

---

## 8. Synthesis: what the two analyses jointly tell us

### 8.0 Outcome at the canonical operating point

Both pre-registered hypotheses are **not supported** at the
canonical operating point evaluated here:

- **§7.1 (induced gain on flavor-B trained models)**: $\gamma_\mathrm{sd}$
  median 154.78 (flavor-B) vs 163.39 (flavor-A baseline), ratio 0.947 —
  a ~5% reduction, within methodological noise. The "≥ 2× reduction at
  $\eta_\mathrm{max} \ge 0.10$" prediction is falsified at this single
  canonical reach.
- **§7.2 (flavor-(b) Riccati synthesis)**: under the S-procedure
  quadratic-stability lift, the previously-xfailed
  `test_cs_faithful_qr_velocity_inflation` does *not* pass; Δv at
  $1.5\gamma_*^{(b)}$ on the C&S regime stays $\le 0$ and grows more
  negative with $\eta$ (mechanism: $Q$-augmentation damps the
  controller). The "Δv > 0 on the C&S plant under flavor-(b) $B_w$"
  prediction is falsified under quadratic stability.

The joint reading is *not* that the flavor-(a) ⊊ (b) thesis is refuted.
It is that the gap, if real, is finer than what either of these probes
can detect — the analyser at one fixed canonical reach against a
worst-case operator-norm $\Delta A$, and the synthesiser under a
conservative quadratic-stability bound. Both negative results constrain
where a positive (a)-vs-(b) signal can plausibly hide; neither rules
the signal out.

### 8.0.1 Plausible refinements (not yet filed as issues)

- **Direction-projected induced gain.** Project the analyser onto the
  $\Delta A$ directions actually trained against (per-group), rather than
  taking a worst-case operator norm. This would test whether flavor-B
  is more robust *in the directions it trained against*, even if not
  worst-case. A natural follow-up analysis; not yet filed.
- **Tighter analytical lifts.** μ-synthesis (structured singular value)
  for the flavor-(b) Riccati would replace the conservative S-procedure
  bound and may reverse the C&S Δv result. Trajectory-coupled
  time-varying $B_w(x_t)$ is the finite-horizon DRE alternative. Not
  yet filed.
- **Operating-point sensitivity.** The induced-gain analyser was run at
  one canonical reach (15 cm forward, SISU=0.5). Re-running across a
  reach geometry / SISU sweep, or longer horizons, may surface
  stratification masked by the single-point measurement.

These are flagged here as natural next steps; per the coordination
protocol they are *not* filed as placeholder issues. The cross-cutting
comment on `4d38c15` records the empirical state and the candidate
refinements without committing to any.

### 8.1 The "(a)-vs-(b) signal"

Three corroborating signatures, in increasing order of strength:

1. **$\gamma_\mathrm{sd}$ stratifies by training flavor** (§7.1). Flavor-B
   controllers reduce small-gain sensitivity to unstructured $\Delta A$;
   flavor-A controllers do not. A 2× or larger reduction at
   $\eta_\mathrm{max} \ge 0.10$ is the conservative bar; a $\sim 10\times$
   reduction (matching the analyser's expected dynamic range) would be a
   strong result.
2. **The empirical $\gamma_\mathrm{net}^\mathrm{flav-B}$ approaches the
   analytical $\gamma_*^{(b)}$** (§7.1 + §7.2). The flavor-(b) round-trip
   identity — analyser meets Riccati when both are flavor-(b) — closes the
   diagram in the same way that the flavor-(a) round-trip already does
   (`tests/test_induced_gain.py::test_riccati_round_trip_qr_cost`).
3. **The C&S Δv signature is recovered** (§7.2 + downstream). The flavor-
   (b) Riccati on the C&S plant produces Δv > 0; ideally a
   flavor-B-trained controller in the C&S regime also produces Δv > 0
   when evaluated under matched flavor-(b) disturbance (downstream
   experiment, not part of this document).

### 8.2 What would falsify the thesis

- $\gamma_\mathrm{sd}$ does not differ between flavor-A and flavor-B
  trained controllers at any $\eta_\mathrm{max}$. (Direct empirical
  falsification.)
- The flavor-(b) Riccati Δv on the C&S plant is ≤ 0. (Theoretical
  falsification: the disturbance-channel diagnosis is wrong.)
- The flavor-(b) Riccati $\gamma_*^{(b)}$ on the rlrmp plant is
  approximately equal to the flavor-(a) $\gamma_*^{(a)}$ on the same plant
  (i.e. the two formulations are not numerically distinguishable at this
  parameter regime), suggesting the (a)/(b) distinction is theoretically
  real but practically vacuous on the rlrmp setup.
- Trained flavor-B controllers fail to converge or fail to saturate
  $\eta(\beta)\to \eta_\mathrm{max}$ at SISU=1 (training-loop issue rather
  than thesis falsification, but blocks the test).

### 8.3 Key uncertainties (known-knowns vs known-unknowns)

**Known-knowns:**

- Flavor-A induced-gain $\gamma_\mathrm{af}$ is a poor cross-method
  discriminator on a single canonical reach (§5.2 finding 1).
- Production Riccati's $B_w$ is flavor-(a) (xfailed C&S test).
- The flavor-A null result on Δv is robust across 8 hyperparameter
  variants of APT plus minimax with `GaussianBumpAdversary`
  (`results/part2_5/README.md`).
- The rlrmp plant regime ($k=10$) amplifies the predicted Δv, so the null
  result is *not* explained by parameter regime under the flavor-(a)
  formulation.

**Known-unknowns:**

- Whether the SISU gating choice (additive `+ ΔA·x` vs multiplicative
  `(1 + scale)·ΔA·x`) materially changes the (a)-vs-(b) signal (§9).
- Whether 5 PGD inner steps suffice for the inner-max to converge to a true
  worst-case $\Delta A$ at all $\eta_\mathrm{max}$ levels (training-loop
  question, not thesis question).
- Whether LEQG (Whittle 1981 / Jacobson 1973 risk-sensitive control,
  cumulant-tilt formulation; cited in `results/part2_5/synthesis_review.md`
  §4.1) bridges the (a)–(b) gap on the flavor-(a) $\delta$-class. The
  synthesis review predicts no — LEQG without (b) is "compile-time risk-tilt
  of an inadequate training distribution" — but this has not been
  empirically tested. [to confirm]

---

## 9. Limitations and open questions

- **Plant regime sensitivity.** The rlrmp regime ($k=10$) amplifies the
  flavor-(a) Δv prediction; the C&S regime ($k=0.1$) does not. The
  flavor-(b) Riccati extension (issue `97c227a`) is needed to test whether
  flavor-(b) recovers Δv > 0 on the C&S regime. Whether the rlrmp regime
  is the right place to evaluate (a)-vs-(b) is itself an open question:
  the regime is favourable for the prediction in flavor-(a), but the C&S
  regime is the canonical empirical anchor for the C&S signature.
- **Adversary budget calibration.** Comparing flavor-A ($\epsilon$ in L2
  ball over force trajectories) to flavor-B ($\eta$ in Frobenius ball over
  $\Delta A$ matrices) at "matched effective budget" is non-trivial; the
  natural map is $\epsilon \approx \eta \cdot \|\bar{x}\|_2 \cdot m$ where
  $\bar{x}$ is the nominal trajectory's state magnitude, but this is an
  order-of-magnitude calibration only. The flavor-B run sweeps three
  $\eta_\mathrm{max}$ values precisely to bracket this uncertainty.
- **SISU gating (additive vs multiplicative).** `LinearDynamicsAdversary`
  uses `eta(SISU) = eta_max * SISU`, applied via the intervenor's `scale`
  field. Whether instead an additive blend `ΔA(SISU) = SISU * ΔA_full`
  gives the same training pressure is untested. The current choice matches
  PAI-ASF's "budget-scaling" interpretation of $\beta$.
- **LEQG via Whittle.** Treated as a separate "flavor-(c) distributional"
  axis in the training-methods coord (`c99ad9d`); not a substitute for
  flavor-B but possibly a bridge. Cumulant-tilt $J_\mathrm{LEQG}(\theta) =
  \frac{1}{\gamma}\log \mathbb{E}_w[\exp(\gamma J(\theta, w))]$ recovers
  LQG at $\gamma\to 0^+$ and the H&infin; controller as $\gamma \to
  \gamma_*$ (Whittle correspondence). On a flavor-(a) $\delta$-class,
  synthesis review predicts LEQG buys partial improvement at most.
- **Architecture (GRU vs vRNN) is not the bottleneck.** Phase 2.5
  architecture sweep showed 0% Δv across vRNN/GRU $\times$
  additive/multiplicative SISU gating; downgraded to auxiliary in coord
  `4d38c15`.
- **Single canonical reach.** The induced-gain first run measures
  $\gamma_\mathrm{net}$ on one (15 cm forward, SISU=0.5) reach. Sensitivity
  to reach geometry / SISU level is unmeasured. (Issue `1ad3c16` — Riccati-
  pre-registered evaluation amplitudes — addresses the related calibration
  question for Δv evaluation.)
- **Reference for "C&S".** Crevecoeur, Cluff, & Scott (2019), "Robust
  control in human reaching movements" (Journal of Neuroscience). The
  $k=0.1$ value and the Eq. 15 Q,R schedule are taken from this paper;
  exact pages cited in `4d38c15` comment `48d13a8` (p. 8137 Statistical
  design, Fig 1e, Results p. 8139). The $\tau_v = 10\,000$ ms estimate
  for the C&S regime assumes $m=1$ kg (effector mass) at $k=0.1$ Ns/m,
  which is a unit-coherent mapping but not directly cited from the paper
  [to confirm against C&S §Methods].

---

## References (as cited)

- Crevecoeur, F., Cluff, T., & Scott, S. H. (2019). "Robust control in
  human reaching movements." *Journal of Neuroscience* (cited as "C&S
  2019" throughout). [citation pages: p. 8137 Statistical design, Fig 1e,
  Results p. 8139, per `4d38c15` comment `48d13a8`]
- Doyle, J. C., Glover, K., Khargonekar, P. P., & Francis, B. A. (1989).
  "State-space solutions to standard $H_2$ and $H_\infty$ control
  problems." (DGKF; the modern H&infin; framework underpinning the Riccati
  synthesis.)
- Basar, T., & Bernhard, P. (1995). *$H_\infty$-Optimal Control and
  Related Minimax Design Problems* (the discrete-time finite-horizon LQ-
  game form used in `hinf_riccati.py`).
- Whittle, P. (1981). "Risk-sensitive linear / quadratic / Gaussian
  control." *Advances in Applied Probability* 13, 764–777. (LEQG and the
  Whittle correspondence with H&infin;.)
- Jacobson, D. H. (1973). "Optimal stochastic linear systems with
  exponential performance criteria and their relation to deterministic
  differential games." *IEEE Trans. Auto. Control* 18, 124–131.
  (Independent original LEQG result.)
- Boyd, S., & Balakrishnan, V. (1990). "A regularity result for the
  singular values of a transfer matrix and a quadratically convergent
  algorithm for computing its $L_\infty$-norm." *Systems & Control
  Letters* 15, 1–7. (The Hamiltonian bisection algorithm in §3.2.)

Internal references (rlrmp issue tracker, 7-character IDs):

- Coordination: `c99ad9d` (training-methods), `4d38c15` (analyses),
  `b33e8da` (phases), `1d9ae6f` (meta).
- Direct work: `74bfd86` (induced-gain analyser, closed),
  `c723082` (LinearDynamicsAdversary, closed), `97c227a` (Riccati flavor-
  (b) extension, open), `5a44bd3` (production Riccati, closed), `19b9921`
  (C&S Eq. 15 Q,R faithful test, xfailed merged), `f90bf74` (Δv metric
  correction, merged), `3c74e3b` (round-trip ratio band, merged),
  `6fdf9a4` (induced-gain first run, merged), `4f2e934` (`mult_single`
  replicate-0 outlier follow-up), `53b5fe5` (SimpleFeedback cycle-wires
  adapter fix), `db35426` (H&infin; Riccati teacher distillation),
  `2e5f643` (damping calibration), `a5e1450` (cross-partial diagnostic),
  `b557d4e` (methodology-fix phase, closed umbrella).
- Phase artifacts: `results/part2_5/README.md`,
  `results/part2_5/synthesis_review.md`,
  `results/part2_5/runs/induced_gain_first_run/notes.md`.
- Code: `src/rlrmp/analysis/induced_gain.py`,
  `src/rlrmp/analysis/hinf_riccati.py`, `src/rlrmp/adversary.py`,
  `src/rlrmp/intervention_compat.py`,
  `feedbax/intervene/intervene.py::DynamicsMatrixPerturb` (on feedbax
  `develop`).
