# C&S Fidelity Audit: Output-Feedback & Rollout

I have completed the read-only audit of the output-feedback lanes, focusing heavily on the covariance recursions and forward rollouts as requested. 

Here are the definitive findings comparing `rlrmp` to `cs2019_modeldb`:

## 1. Covariance Recursion Fidelity: Exact Mathematical Match
I performed a line-by-line comparison of the robust estimator covariance updates. Despite structural differences in how the matrices are constructed, `rlrmp` perfectly implements the C&S mathematical operations.

**C&S Implementation (`minmaxfc_pointMass.m` line 254):**
```matlab
Sigma(:,:,i+1) = Aest*(Sigma(:,:,i)^-1+H'*(E*E')^-1*H-gamma^-2*Q(:,:,i))^-1*Aest'+D*D';
```

**`rlrmp` Implementation (`output_feedback.py` line 202-204):**
```python
precision = jnp.linalg.inv(Sigma) + H.T @ H - inv_gamma2 * schedule.Q[t].astype(jnp.float64)
middle = jnp.linalg.inv(precision)
Sigma = A @ middle @ A.T + Q_proc  # Q_proc = plant.Bw @ plant.Bw.T
```

**Why they are algebraically identical:**
1. **Sensory Noise `(E*E')^-1` omission:** In C&S, `E` is initialized as `E = eye(8,1)'` (line 76), which is a `1x8` vector `[1, 0, ..., 0]`. Therefore, `E*E'` is the scalar `1`. `rlrmp` correctly recognizes that `(E*E')^-1` drops out of the equation entirely, leaving just `H.T @ H`. (Note: This implies C&S's robust estimator mathematically assumes *identity* sensory noise covariance during the minimax Riccati recursion, even though it later injects scaled sensory noise during forward simulation).
2. **Process Noise `D*D'` vs `Bw @ Bw.T`:** In C&S, `D` is initialized as `zeros` but then `D(1:8, 1:8) = eye(8)` (line 115). This means `D*D'` acts purely as a selector to inject noise into the 8 physical state variables at the head of the delay chain. `rlrmp`'s `Bw` matrix does exactly the same thing. 

**Verdict on Recursions:** PASS. No fidelity gaps here.

## 2. Forward Simulation Gap: Confirmed
Your candidate hypothesis is **correct**. There is a strict divergence in how the actual forward rollouts are executed.

In **C&S (`minmaxfc_pointMass.m`)**, the forward simulation is heavily stochastic. Every timestep samples three distinct noise sources:
1. `sensoryNoise` (sampled from `Omega` via `mvnrnd`)
2. `motorNoise` (sampled from `Oxi` via `mvnrnd`)
3. `sdn` (signal-dependent noise, via `normrnd(0,1)*Csdn(:,:,isdn)*u(i,:)'`)

These are explicitly added to the forward and observation equations:
```matlab
yx = H*currentX + sensoryNoise;
...
currentX = Aest*currentX + B*u(i,:)'+D*wx + motorNoise + sdn;
```

In **`rlrmp` (`output_feedback.py`)**, `rollout_with_robust_estimator` and `rollout_with_kalman_estimator` only use the covariance matrices to compute the estimator gains (`K_est`). The actual forward propagation of the state uses `epsilon` (an explicit deterministic adversarial budget, defaulting to zero):
```python
xhat_next = plant.A @ xhat_t + plant.B @ u_t + plant.A @ middle @ robust_correction
x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t
```
`rlrmp` does not sample `normrnd` or `mvnrnd` anywhere in these rollout pipelines.

**Impact of this gap:**
Because `rlrmp`'s rollouts are deterministic, your tracked artifacts (like the Phase 0B cost of `4496.785` and the peak forward velocity of `0.77046417` in `output_feedback_lane.md`) are perfectly "clean" L2 exact trajectories. If you were to run the exact C&S MATLAB code, the resulting empirical metrics would show variance and likely slightly higher mean costs due to stochastic noise inflation during the reach.

## 3. Bellman Diagnostics & Gamma Sweeps
I reviewed `robust_bellman.md` and `output_feedback_gamma_sweep.md`. The pipeline to find the robust linear target (gamma factor 1.35) and verify it through exact-inner information-state Bellman objectives is mathematically sound under the deterministic-rollout paradigm. The fact that the exact-inner objective hits a max gain relative error of `0.00035479` against the persistent target (when configured with the C&S persistent index) confirms your optimizer is successfully navigating the estimator-in-loop landscape.

However, be aware that any subsequent Phase 3 / RNN training that trains against these "clean" deterministic rollouts will lack the signal-dependent noise (`sdn`) that fundamentally drives Todorov-style/Flavor-B velocity suppression (as highlighted in your Gemini synthesis document).
