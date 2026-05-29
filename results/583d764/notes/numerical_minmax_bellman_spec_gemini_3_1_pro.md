# Fully Numerical Min-Max Information-State Bellman Optimization

## 1. Formal Setup
The objective is to find the robust output-feedback $H_\infty$ (or risk-sensitive) command gain at each timestep without relying on exact analytical inner-maximization formulas. We operate in the formal framework where the dynamic programming state is the **robust information state**, defined by the conditional mean $\hat{x}_t$ and the deterministic robust filter covariance $\Sigma_t$.

We assume the necessary Riccati/filter matrices are available for the current timestep:
- $\Sigma_t$: Robust estimation covariance
- $L_t^x$: Uncontrolled state cost-to-go
- $N_t$: State-control coupling cost
- $M_t^u$: Control cost

## 2. The Single Bellman Objective
For a given controller action $u$ and an adversarial hidden-state selection $x_{adv}$, the one-step robust Information-State Bellman objective is a single algebraic expression:

$$ \mathscr{B}_t(u, x_{adv}; \hat{x}_t, \Sigma_t) = \underbrace{x_{adv}^\top L_t^x x_{adv} + 2x_{adv}^\top N_t u + u^\top M_t^u u}_{\text{Cost-to-go}} - \underbrace{\gamma^2 (x_{adv} - \hat{x}_t)^\top \Sigma_t^{-1} (x_{adv} - \hat{x}_t)}_{\text{Information Penalty}} $$

This objective balances the future game cost against the accumulated historical penalty of the adversary forcing the true state $x$ away from the estimate $\hat{x}_t$.

## 3. Dual Linear Model Parameterization
Instead of computing the exact analytical maximizer $x_{adv}^* = \arg\max \mathscr{B}_t$, we explicitly parameterize two independent linear models for each timestep:

1. **Controller Model:** $u_t = -K_t \hat{x}_t$
2. **Adversary Model:** $x_{adv, t} = M_t \hat{x}_t$

We define the batched objective function over a full-rank dataset of state estimates $\mathcal{D}_t$:

$$ \mathcal{L}_t(K_t, M_t) = \mathbb{E}_{\hat{x} \sim \mathcal{D}_t} \left[ \mathscr{B}_t(-K_t \hat{x}, M_t \hat{x}; \hat{x}, \Sigma_t) \right] $$

The goal is to find the saddle point:
$$ K_t^*, M_t^* = \arg\min_{K_t} \max_{M_t} \mathcal{L}_t(K_t, M_t) $$

## 4. Formal Guarantees
Assuming the standard robust feasibility condition holds—namely, that the matrix $D_t = \gamma^2 \Sigma_t^{-1} - L_t^x$ is strictly positive definite—the function $\mathcal{L}_t(K_t, M_t)$ is a strongly-convex, strongly-concave quadratic.

Due to the geometry of strict convex-concave quadratics, the saddle point is **unique**. Therefore, any numerical optimization procedure that successfully converges to this saddle point will, with 100% mathematical certainty, converge to identical gains as the exact analytical separation-principle Riccati solution.

## 5. Numerical Optimization Methods
Naively applying simultaneous gradient descent to $K_t$ and $M_t$ on a min-max quadratic can cause the parameters to "rotate" and diverge to infinity (blow up). To guarantee stable convergence to the unique saddle point purely numerically, the optimization must be structured using one of the following methods:

### Method A: Inner-Loop / Outer-Loop Optimization
This method replicates the exact analytical ordering but performs it via numerical simulation.

1. **Inner Loop (Adversary):** Freeze the controller weights $K_t$. Perform gradient ascent on the adversary weights $M_t$ to maximize $\mathcal{L}_t$:
   $$ M_t \leftarrow M_t + \alpha \nabla_{M_t} \mathcal{L}_t(K_t, M_t) $$
   Iterate until $M_t$ converges. Because the objective is strongly concave with respect to $M_t$, gradient ascent (or L-BFGS) will stably and quickly find the peak without diverging.
2. **Outer Loop (Controller):** Freeze $M_t$ at its converged peak. Take a gradient descent step on the controller weights $K_t$ to minimize $\mathcal{L}_t$:
   $$ K_t \leftarrow K_t - \beta \nabla_{K_t} \mathcal{L}_t(K_t, M_t) $$
3. Repeat the inner and outer loops until both $K_t$ and $M_t$ achieve a stable fixed point.

### Method B: Extragradient Descent
If a simultaneous (single-loop) update is required, use the Extragradient algorithm, which mathematically suppresses the rotational divergence inherent to min-max quadratics by utilizing a lookahead step.

1. **Lookahead Step:** Compute temporary "lookahead" weights for both models using standard gradients:
   $$ \tilde{K}_t = K_t - \eta \nabla_{K_t} \mathcal{L}_t(K_t, M_t) $$
   $$ \tilde{M}_t = M_t + \eta \nabla_{M_t} \mathcal{L}_t(K_t, M_t) $$
2. **Update Step:** Evaluate the gradients at the lookahead positions, and apply these gradients to update the actual model weights:
   $$ K_t \leftarrow K_t - \eta \nabla_{K_t} \mathcal{L}_t(\tilde{K}_t, \tilde{M}_t) $$
   $$ M_t \leftarrow M_t + \eta \nabla_{M_t} \mathcal{L}_t(\tilde{K}_t, \tilde{M}_t) $$

By evaluating the gradient at the anticipated future position $(\tilde{K}_t, \tilde{M}_t)$, the optimization trajectory avoids the infinite outward spiral, guaranteeing smooth convergence to the saddle point of the Information-State Bellman equation.
