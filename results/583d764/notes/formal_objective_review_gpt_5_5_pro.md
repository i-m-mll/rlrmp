Classification

Separated robust filter/control theorem needed, with an information-state Bellman—not a single Bellman over z=[x,\hat x].

The most plausible formal object is a finite-horizon, delayed-observation H_\infty output-feedback dynamic game whose controller state is not z_t, but the robust information state

\mathcal I_t(x)
=
(x-\hat x_t)^\top \Sigma_t^{-1}(x-\hat x_t)

together with a backward full-information control storage P_t. This is exactly the kind of two-Riccati / separation-like structure that standard output-feedback H_\infty theory uses: a control Riccati, a filtering Riccati, and a coupling/product condition. The classical DGKF state-space solution is explicitly a two-Riccati construction with a separation structure reminiscent of LQG, and solvability involves positivity plus a spectral-radius product condition; modern hinfsyn documentation describes the same Riccati-based algorithmic pattern with X,Y and \rho(XY)<1.    

The released C&S/ModelDB code should therefore be treated as an observer-form robust output-feedback controller, not as the fixed point of the simple public-state Bellman diagnostic. The ModelDB/GitHub source identifies the C&S code as MATLAB code for robust/stochastic reaching simulations, including minmaxfc_pointMass.m, and the ModelDB description frames the robust controller as a min-max / worst-case strategy.    

⸻

1. Formal game object I would use

Use the packet’s delayed augmented plant:

x_{t+1}=Ax_t+Bu_t+B_w\epsilon_t,\qquad y_t=Hx_t+v_t.

For the packet’s realized deterministic audits, v_t=0. But for the robust estimator recursion to be formally meaningful, the derivation must include a measurement disturbance v_t with unit quadratic penalty, because the H^\top H term in the robust filter is exactly the information-form measurement penalty. Without v_t, the robust filter is not the filter of the stated game.

The controller information pattern should be:

u_t=\mu_t(y_0,\ldots,y_{t-1},u_0,\ldots,u_{t-1})

if you preserve the packet’s timing convention that y_t updates \hat x_{t+1}, not the command u_t. Equivalently, \hat x_t,\Sigma_t are the pre-y_t predictor information state.

A normalized disturbance-attenuation form is:

\sum_{t=0}^{T-1}
\left(x_t^\top Q_t x_t+u_t^\top R_tu_t\right)
+x_T^\top Q_fx_T
-
\gamma^2
\left[
(x_0-\hat x_0)^\top \Sigma_0^{-1}(x_0-\hat x_0)
+\sum_{t=0}^{T-1}
\left(\epsilon_t^\top\epsilon_t+v_t^\top v_t\right)
\right].

Equivalently, divide by \gamma^2: the robust filter sees unit penalties for process and measurement inconsistency and a negative \gamma^{-2}Q_t term. This is exactly why the packet’s robust estimator precision has the form

\Sigma_t^{-1}+H^\top H-\gamma^{-2}Q_t.

So \gamma is still the H_\infty attenuation/penalty parameter, not an L2 budget.

⸻

2. Backward robust-control Riccati

Keep the existing full-information H_\infty control Riccati. With

P_T=Q_f,

define

\Lambda_{t+1}
=
\left(I-\gamma^{-2}P_{t+1}B_wB_w^\top\right)^{-1}P_{t+1},

M_t
=
R_t+B^\top\Lambda_{t+1}B,

K_t^x
=
M_t^{-1}B^\top\Lambda_{t+1}A,

and

P_t
=
Q_t
+
A^\top\Lambda_{t+1}A
-
A^\top\Lambda_{t+1}B
M_t^{-1}
B^\top\Lambda_{t+1}A.

This is equivalent to the packet’s full-state H_\infty Riccati form. Solvability requires at least

\gamma^2I-B_w^\top P_{t+1}B_w\succ 0.

For R_t=I, the code’s base gain

B^\top
\left(
P_{t+1}^{-1}+BB^\top-\gamma^{-2}B_wB_w^\top
\right)^{-1}
A

is the same object as K_t^x, up to the usual matrix-inversion identity.

⸻

3. Forward robust-filter information recursion

The robust information state is

\mathcal I_t(x)
=
(x-\hat x_t)^\top\Sigma_t^{-1}(x-\hat x_t).

After observing y_t, define the posterior information cost

\mathcal I_t^+(x)
=
\mathcal I_t(x)
+
(y_t-Hx)^\top(y_t-Hx)
-
\gamma^{-2}x^\top Q_tx.

Completing the square gives

\Omega_t
=
\Sigma_t^{-1}+H^\top H-\gamma^{-2}Q_t,

M_t^\Sigma
=
\Omega_t^{-1},

and

\bar x_t
=
\hat x_t
+
M_t^\Sigma
\left[
\gamma^{-2}Q_t\hat x_t
+
H^\top(y_t-H\hat x_t)
\right].

Then the predictor update is

\hat x_{t+1}
=
A\bar x_t+Bu_t,

\Sigma_{t+1}
=
A M_t^\Sigma A^\top+B_wB_w^\top.

This exactly matches the packet/code structure:

xhat_{t+1}
=
A xhat_t
+
B u_t
+
A M_t^\Sigma
\left[
\gamma^{-2}Q_t xhat_t
+
H^\top(y_t-Hxhat_t)
\right].

The required filter-side solvability check is

\Omega_t\succ 0.

⸻

4. Coupling and output-feedback command law

The missing coupling term is the robust hidden-state maximizer

x_t^\circ
=
\arg\max_x
\left\{
x^\top P_tx
-
\gamma^2(x-\hat x_t)^\top\Sigma_t^{-1}(x-\hat x_t)
\right\}.

The maximizer is

x_t^\circ
=
\left(I-\gamma^{-2}\Sigma_tP_t\right)^{-1}\hat x_t,

provided

I-\gamma^{-2}\Sigma_tP_t

is nonsingular with the appropriate positive-definite/spectral-radius margin.

Then the separated robust output-feedback law is

u_t
=
-K_t^x x_t^\circ
=
-
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_t\right)^{-1}
\hat x_t.

So the formal finite-horizon law should be

K_t^{OF}
=
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_t\right)^{-1}.

This is the clean formal target. It uses time-indexed P_t in the coupling correction.

The released-code-compatible law with persistent p_idx=0 is therefore

K_{t,\mathrm{C\&S}}^{OF}
=
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_0\right)^{-1}.

I would not treat p_idx=0 as the finite-horizon theorem unless you can show that the original MATLAB variable is intentionally representing a stationary/infinite-horizon P. With a time-varying ramped Q_t, P_0 is not generally interchangeable with P_t. So:

\boxed{\text{formal target: }p\_idx=t}

\boxed{\text{C\&S code-fidelity target: }p\_idx=0}

Both are useful, but they should be named separately.

⸻

5. A Bellman-style objective that should recover the formal law

For each t, fix P_{t+1}, \Sigma_t, and \hat x. Define

\Lambda_{t+1}
=
\left(I-\gamma^{-2}P_{t+1}B_wB_w^\top\right)^{-1}P_{t+1},

L_t^x
=
Q_t+A^\top\Lambda_{t+1}A,

N_t
=
A^\top\Lambda_{t+1}B,

M_t^u
=
R_t+B^\top\Lambda_{t+1}B.

For a command u that only sees \hat x, define the one-step information-state robust Bellman objective

\mathscr B_t(u;\hat x,\Sigma_t)
=
\sup_x
\left[
x^\top L_t^xx
+
2x^\top N_tu
+
u^\top M_t^u u
-
\gamma^2(x-\hat x)^\top\Sigma_t^{-1}(x-\hat x)
\right].

This is the control Bellman after the process disturbance \epsilon_t has already been maximized in closed form. If

D_t
=
\gamma^2\Sigma_t^{-1}-L_t^x
\succ 0,

then the inner supremum has the closed form

\mathscr B_t(u;\hat x,\Sigma_t)
=
u^\top M_t^u u
-
\gamma^2\hat x^\top\Sigma_t^{-1}\hat x
+
\left(N_tu+\gamma^2\Sigma_t^{-1}\hat x\right)^\top
D_t^{-1}
\left(N_tu+\gamma^2\Sigma_t^{-1}\hat x\right).

Now train a linear command model

u=-L_t\hat x

by minimizing

\mathcal L_{\mathrm{infoBell}}(L)
=
\sum_{t=0}^{T-1}
\mathbb E_{\hat x\sim \mathcal D_t}
\left[
\mathscr B_t(-L_t\hat x;\hat x,\Sigma_t)
\right],

where each \mathcal D_t has full-rank covariance.

The unique minimizer is

L_t^\star
=
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_t\right)^{-1}.

This gives you an L-BFGS-compatible training objective that recovers the formal time-indexed robust output-feedback law without directly supervising actions. It is not the failed z=[x,\hat x] Bellman; it is a Bellman objective over the robust information state.

Implementation note: before trusting this objective, log

\lambda_{\min}\left(\gamma^2\Sigma_t^{-1}-L_t^x\right).

If that matrix is not positive definite, this particular upper-value Bellman is unbounded for some commands. In that case the separated theorem may still be expressible through a more delicate completion-of-squares/product-condition proof, but this simple one-step \inf_u\sup_x objective should not be used as an optimizer.

⸻

6. Guaranteed L-BFGS training recipe matching the analytical recursion

If the immediate engineering goal is: “train a linear model and get exactly the analytical recursion’s controller,” then use one of these two objectives.

6.1 Formal robust target

First compute

L_t^\star
=
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_t\right)^{-1}.

Then train L_t by full-rank action matching:

\mathcal L_{\mathrm{match}}(L)
=
\sum_t
\mathbb E_{\hat x\sim\mathcal D_t}
\left[
\|L_t\hat x-L_t^\star\hat x\|_2^2
\right].

If

\mathbb E_{\mathcal D_t}[\hat x\hat x^\top]\succ0,

then the unique minimizer is exactly

L_t=L_t^\star.

This is boring but formally clean. It tests whether the trainable linear model class and optimizer can represent the separated robust controller.

6.2 C&S code-fidelity target

If the target is the released-code-compatible controller, define

L_{t,\mathrm{C\&S}}^\star
=
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_0\right)^{-1}.

Then use the same full-rank action-matching objective:

\mathcal L_{\mathrm{match,C\&S}}(L)
=
\sum_t
\mathbb E_{\hat x\sim\mathcal D_t}
\left[
\|L_t\hat x-L_{t,\mathrm{C\&S}}^\star\hat x\|_2^2
\right].

This will recover the C&S-compatible analytical recursion with L-BFGS, but the certificate should be called something like:

C&S code-fidelity gain-recovery certificate

not

finite-horizon output-feedback H_\infty Bellman optimality certificate.

⸻

7. Why the simple joint-state Bellman fails

The failed diagnostic uses

z_t=[x_t,\hat x_t]

and asks for a quadratic value

V_t(z)=z^\top S_tz

with control restricted to

u_t=-K_t\hat x_t.

That is solving a different game.

The missing object is the robust information cost

-\gamma^2(x_t-\hat x_t)^\top\Sigma_t^{-1}(x_t-\hat x_t).

This term is not a cosmetic regularizer. It is the accumulated historical energy certificate saying how expensive it was for the adversary to make x_t differ from \hat x_t while remaining consistent with the observation history. The robust estimator recursion exists precisely to propagate that history cost.

The joint-state Bellman also omits the measurement-disturbance channel v_t. But the robust estimator’s H^\top H term is a measurement information term. So the diagnostic says “the adversary only chooses \epsilon_t,” while the estimator recursion says “the observation history has a unit measurement-energy penalty.” Those are not the same game.

Finally, the diagnostic treats z=[x,\hat x] as an ordinary Markov state for value fitting. But arbitrary sampled pairs (x,\hat x) are not equally admissible under the robust filter. Two pairs with the same z can have different historical feasibility if their \Sigma_t or information cost differs. Thus the correct Markov object is not z_t; it is at least

(\hat x_t,\Sigma_t)

plus the hidden-state cost-to-come

\mathcal I_t(x).

That is why the joint policy-improvement fit can achieve a lower objective and still miss the C&S gains: it is improving the wrong Bellman objective.

⸻

8. Conflicts among the current certificates

Deterministic full-state H_\infty

Valid and internally coherent for

u_t=-K_tx_t,\qquad \epsilon_t=F_tx_t.

But it is a full-state game. It does not by itself certify delayed output feedback.

LQG/Kalman separation

The clean LQG comparator is fine as an output-feedback baseline, but LQG separation is stronger and simpler than H_\infty separation. In H_\infty, the filter and controller are coupled by

I-\gamma^{-2}\Sigma_tP_t.

So “Kalman estimator + robust full-state gain” is not the same formal object.

Exact flattened epsilon audit

This is exact for a frozen controller and frozen estimator dynamics:

\epsilon_{0:T-1}\mapsto
\text{cost}(\epsilon)

is truly a finite-dimensional quadratic. The condition

\gamma^2I-H_\epsilon\succ0

is a valid frozen-policy penalized open-loop certificate. But it is not a dynamic-game synthesis proof, because the adversary is an open-loop trajectory and the estimator/controller are frozen.

C&S robust output feedback

The time-indexed version

K_t^x(I-\gamma^{-2}\Sigma_tP_t)^{-1}

has a direct separated robust filtering/control interpretation.

The persistent-index version

K_t^x(I-\gamma^{-2}\Sigma_tP_0)^{-1}

should be treated as C&S code fidelity unless a stationary or infinite-horizon reinterpretation is explicitly derived.

⸻

9. Recommended next implementation steps

1. Add a second output-feedback target named something like formal_separated_hinf_of, with correction

p\_idx=t.

2. Keep the current one as cs_matlab_persistent_index_hinf_of, with

p\_idx=0.

3. For both targets, log the three margins:

\lambda_{\min}(\gamma^2I-B_w^\top P_{t+1}B_w),

\lambda_{\min}(\Sigma_t^{-1}+H^\top H-\gamma^{-2}Q_t),

\lambda_{\min}(I-\gamma^{-2}\Sigma_tP_{\pi(t)}).

For the information-Bellman optimizer, also log

\lambda_{\min}(\gamma^2\Sigma_t^{-1}-L_t^x).

4. Implement the L-BFGS objective

\mathcal L_{\mathrm{infoBell}}(L)
=
\sum_{t,i}
\mathscr B_t(-L_t\hat x_{t,i};\hat x_{t,i},\Sigma_t)

for the formal p\_idx=t target.

5. Implement the guaranteed gain-recovery objective

\mathcal L_{\mathrm{match}}(L)
=
\sum_{t,i}
\|L_t\hat x_{t,i}-L_t^\star\hat x_{t,i}\|^2

for both the formal and C&S-fidelity targets.

6. Rename the certificates:

* full_state_hinf_bellman_recovery: deterministic full-state game.
* formal_separated_output_feedback_hinf_recovery: time-indexed P_t, robust information-state Bellman.
* cs_output_feedback_code_fidelity_recovery: persistent P_0, action/gain matching.
* frozen_open_loop_epsilon_audit: exact fixed-controller vulnerability.
* gamma_penalized_frozen_policy_feasibility: \lambda_{\max}(H_\epsilon)/\gamma^2<1, frozen policy only.

⸻

Bottom line

A single Bellman objective over

z=[x,\hat x]

is the wrong object.

A usable formal objective is available if you promote the state to the robust information state

\mathcal I_t(x)=(x-\hat x_t)^\top\Sigma_t^{-1}(x-\hat x_t)

and couple it to the backward P_t control storage. That gives the separated law

u_t
=
-
K_t^x
\left(I-\gamma^{-2}\Sigma_tP_t\right)^{-1}
\hat x_t.

For the exact released-code-compatible law using P_0, I would not claim finite-horizon Bellman optimality. I would train it by full-rank action matching or gain-residual minimization and label it as a C&S code-fidelity target.