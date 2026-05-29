# Formal Objective Review: Output-Feedback H-infinity Game

## 1. The Correct Information State
The simple joint state $z_t = [x_t, \hat{x}_t]$ is incorrect for the finite-horizon output-feedback dynamic game. The correct state for the controller's Bellman recursion is the **information state** (or belief state). For linear systems under bounded disturbances (or Gaussian noise in the risk-sensitive LEQG framework), the information state is fully summarized by the worst-case conditional mean $\hat{x}_t$ and the robust estimation error covariance $\Sigma_t$. 

Because $\Sigma_t$ is deterministic (independent of the specific observation history $y_{0:t}$), the effective state variable for the optimal controller's dynamic programming equation is just $\hat{x}_t$. The value function associated with this state is $V_t(\hat{x}_t) = \hat{x}_t^T P_t \hat{x}_t + c_t$. The simple joint-state diagnostic failed because it evaluates the policy over $z_t$, effectively changing the information pattern to one where the adversary has direct access to the controller's internal estimate, which breaks the separation principle.

## 2. Disturbance Policy Class and Information Pattern
In the formal $H_\infty$ output-feedback game, the optimal disturbance policy $w_t$ (or $\epsilon_t$) is adapted to the true plant state $x_t$ (it acts as the worst-case noise pushing the plant into high-cost configurations), while the controller only has access to the observation history $y_{0:t}$. This asymmetrical information pattern is exactly what necessitates the robust filter (which estimates $x_t$ assuming the disturbance acts to maximize the error) and the robust controller. The C&S released-code structures are highly compatible with this standard Basar-Bernhard / Whittle framework.

## 3. Finite-Horizon Saddle Point Derivation
The robust output-feedback law used in the codebase is fundamentally derived from the separated **Whittle Risk-Sensitive LEQG / $H_\infty$ Optimal Control** framework (e.g., Basar & Bernhard 1995, Theorem 5.1). The exact finite-horizon saddle point consists of:
1. **The Full-Information Control Riccati** (run backwards) yielding $P_t$.
2. **The $H_\infty$ Filter Riccati** (run forwards) yielding $\Sigma_t$. This update intrinsically couples with the cost via the $-\gamma^{-2} Q_t$ term.
3. **A Coupling Condition** requiring the spectral radius $\rho(\Sigma_t P_{t+1}) < \gamma^2$ at all times.
4. **The Robust Command Gain** which modifies the certainty-equivalence gain by a coupling correction: $K_t^{robust} = K_t^{FI} (I - \gamma^{-2} \Sigma_t P_{t+1})^{-1}$.

## 4. Why the Simple Joint-State Bellman Diagnostic Fails
The simple joint-state diagnostic evaluates the cost using a Markov game over $z_t = [x_t, \hat{x}_t]$. This fails to recover the released-code gains for three reasons:
1. It ignores the asymmetric information pattern, treating $\hat{x}_t$ as fully observable by the adversary.
2. The value function is not a simple quadratic in $z_t$, but a function of the information state.
3. (Crucially) The C&S reference code uses a constant $P_{p\_idx}$ slice ($p\_idx=0$), which is mathematically sub-optimal for the true finite-horizon game, making it impossible for the Bellman diagnostic (which computes exact finite-horizon gradients) to recover it as a fixed point.

## 5. The $M(:,:,k)$ / $p\_idx = 0$ Convention
The use of the first Riccati slice ($P_0$) for all timesteps $t$ in the command correction $(I - \gamma^{-2} \Sigma_t P_0)^{-1}$ **does not have a formal finite-horizon interpretation**. It is a steady-state approximation or a bug in the MATLAB release. To form a mathematically rigorous finite-horizon target, this should be corrected to use the time-varying $P_{t+1}$ (or $P_t$, depending on exact step timing). The persistent-index law should be relegated to an empirical "C&S-code-fidelity" variant, not the formal target.

## 6. Interpretation of Gamma
The full-state deterministic Riccati $\gamma_{star}$ is a lower bound. Because output feedback restricts the controller's information, the system is more vulnerable, meaning the achievable attenuation bound $\gamma_{OF}$ is strictly larger than $\gamma_{star}$. A factor like $1.35 \times \gamma_{star}$ is correct in principle, representing the increased penalty required to ensure the spectral radius condition $\rho(\Sigma_t P_{t+1}) < \gamma^2$ remains valid across the horizon.

## 7. Exact Flattened Epsilon Audit as a Training Objective
The exact flattened epsilon audit calculates the precise worst-case cost $\max_w (\text{cost}(K, w) - \gamma^2 \|w\|^2)$ for a frozen linear controller. This is mathematically equivalent to computing the $H_\infty$ closed-loop norm (or the risk-sensitive LEQG objective). 

Therefore, **this exact audit is the perfect unified objective to train a linear model**. If you define a linear RNN controller (with learnable estimator matrices and gains) and optimize this exact flattened cost using L-BFGS, the parameters will naturally converge to the exact separation-principle $H_\infty$ robust filter and controller. This avoids having to explicitly encode the coupled Riccati equations into the training loop.

## 8. Final Recommendations for the Project
- **Unified Objective:** The exact flattened epsilon audit $\max_w (\text{cost} - \gamma^2 \|w\|^2)$ is the formal $H_\infty$ objective for any fixed policy. Train linear models by directly minimizing this analytical maximum.
- **Analytical Reference Certificate:** Implement the true Basar-Bernhard separated recursions (using time-varying $P_{t+1}$ in the coupling correction) as the mathematically rigorous analytical comparator.
- **C&S Fidelity Variant:** Keep the $p\_idx=0$ version only to document the behavior of the historical MATLAB code, acknowledging it as a stationary approximation of the true finite-horizon game.
