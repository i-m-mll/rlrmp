# Analytical damage recursions for paired reference targets

## Target quantity

For adaptive damage matching, the analytical reference target is the paired
within-controller damage

```text
D_beta_ref = E_i[J_i(K_beta, F_beta) - J_i(K_beta, 0)].
```

Here `K_beta` is the analytical H-infinity controller at the selected beta
level, and `F_beta` is the matched analytical adversarial disturbance law from
the same synthesis. The clean comparator uses the same analytical controller,
the same task convention, and no adversarial disturbance. The trial index `i`
may include target condition, initial state, horizon convention, mask, and
nominal stochastic realization or distribution.

The paired target is valid only after the following conventions are fixed:

- The cost `J_i` is the task cost used for the comparison: state cost `Q`,
  control cost `R`, and terminal cost `Q_f`, with the same masks, reductions,
  target centering, and state basis as the intended training/evaluation
  comparison.
- The analytical controller and adversary are fixed before evaluation. The
  recursions below are post-synthesis cost-evaluation recursions, not the
  H-infinity synthesis recursion.
- The closed-loop dynamics are finite-horizon linear or affine after
  substituting the fixed controller and disturbance law. Affine terms can be
  handled either by augmenting the state with a constant coordinate or by using
  explicit quadratic-linear-constant value terms.
- Additive noise, when present, has known mean and covariance and is represented
  in the augmented dynamics. The simple trace formulas below assume zero-mean
  additive noise independent of the current state; nonzero-mean noise should be
  absorbed into the affine state update or propagated explicitly.

This target measures extra task cost caused by the matched analytical
disturbance against its own matched analytical controller. It is not nominal
conservatism, such as `J(K_beta, 0) - J(K_ref, 0)`, and it is not total burden
relative to a nominal clean controller, such as
`J(K_beta, F_beta) - J(K_ref, 0)`.

## Full-state deterministic recursion

In the full-state case, after fixing `K_beta` and `F_beta`, write the
comparison in an augmented state `z_t`. At minimum, `z_t` contains the plant
state used by the analytical controller. It should also contain target or
reference coordinates, delay blocks, and a constant coordinate if these are
needed to express offsets or affine dynamics linearly.

For each time step, substitute the fixed controller into the control law:

```text
u_t = U_t z_t.
```

For the adversarial condition, substitute the matched disturbance law:

```text
w_t = F_t z_t.
```

For the clean condition, set only the adversarial disturbance to zero:

```text
w_t = 0.
```

This gives two closed-loop transition matrices:

```text
z_{t+1}^{adv} = M_t^{adv} z_t
z_{t+1}^{0}   = M_t^{0}   z_t
```

for deterministic, noise-off evaluation. If the original dynamics are affine,
the equations have the same form after adding a constant coordinate to `z_t`.

The stage cost after substituting the fixed controller is a quadratic form in
the augmented state:

```text
c_t(z_t) = z_t^T C_t z_t
C_t = X_t^T Q_t X_t + U_t^T R_t U_t.
```

`X_t` selects the true plant state or task-error coordinates that receive the
state cost. The terminal cost is

```text
c_T(z_T) = z_T^T C_T z_T
C_T = X_T^T Q_f X_T.
```

For either closed-loop condition `a in {adv, 0}`, compute the deterministic
cost-to-go matrices backward:

```text
S_T^a = C_T
S_t^a = C_t + (M_t^a)^T S_{t+1}^a M_t^a.
```

For a deterministic initial augmented state `z_0`, the two fixed-policy costs
are

```text
J^a(z_0) = z_0^T S_0^a z_0,
```

and the paired analytical damage for that initial condition is

```text
d_beta(z_0) = z_0^T (S_0^{adv} - S_0^0) z_0.
```

This recursion is a deterministic finite-horizon cost evaluation for two fixed
closed loops. It does not optimize a new adversary during the backward pass.
All adversarial structure enters through the already fixed matrix `F_t` inside
`M_t^{adv}`.

## Expectation from initial moments and additive noise

If the initial augmented state has mean `mu_0` and covariance `Sigma_0`, then
the deterministic-recursion expectation for noise-off dynamics is

```text
E[J^a] =
    tr(S_0^a Sigma_0) + mu_0^T S_0^a mu_0

D_beta_ref =
    tr((S_0^{adv} - S_0^0) Sigma_0)
    + mu_0^T (S_0^{adv} - S_0^0) mu_0.
```

With zero-mean additive noise in the transition,

```text
z_{t+1}^a = M_t^a z_t + eta_t^a,
E[eta_t^a] = 0,
Cov(eta_t^a) = Omega_t^a,
```

the backward value recursion remains

```text
S_T^a = C_T
S_t^a = C_t + (M_t^a)^T S_{t+1}^a M_t^a,
```

but the expected cost includes additive-noise trace terms:

```text
E[J^a] =
    tr(S_0^a Sigma_0) + mu_0^T S_0^a mu_0
    + sum_{t=0}^{T-1} tr(S_{t+1}^a Omega_t^a).
```

Therefore

```text
D_beta_ref = E[J^{adv}] - E[J^0],
```

using the corresponding closed-loop matrices and noise covariances for each
condition. If the nominal additive-noise law is the same in clean and
adversarial rollouts, the same `Omega_t` should be used in both expectations.
If the output-feedback estimator or plant model injects condition-specific
noise through different augmented channels, the condition-specific covariance
seen by `z_{t+1}` must be used.

The same expected costs can also be computed by forward moment propagation:

```text
mu_{t+1}^a = M_t^a mu_t^a
Sigma_{t+1}^a = M_t^a Sigma_t^a (M_t^a)^T + Omega_t^a

E[J^a] =
    sum_{t=0}^{T-1} [
        tr(C_t Sigma_t^a) + (mu_t^a)^T C_t mu_t^a
    ]
    + tr(C_T Sigma_T^a) + (mu_T^a)^T C_T mu_T^a.
```

The backward and forward formulas should agree when their timing convention and
noise injection convention are the same. The forward form is often easier to
audit because it exposes the evolving means and covariances; the backward form
is compact and convenient for deterministic initial-state sweeps.

## Output-feedback augmented recursion

The output-feedback case uses the same paired target, but the augmented state
must include both the true plant state and the estimator/filter state. A minimal
schematic state is

```text
z_t = [x_t; xhat_t],
```

with additional coordinates added as needed for delayed observations,
force-filter states, target/reference coordinates, and affine constants. The
exact contents of `z_t` are part of the information contract. The full-state
formula should not be reused on plant state alone unless the controller and
adversary truly have full-state access.

The controller is a map from the augmented state to the estimate-based control:

```text
u_t = U_t z_t.
```

In the simple estimator-state case, `U_t` selects `xhat_t` and applies the
analytical feedback gain, for example `u_t = -K_t xhat_t`. The matched
output-feedback adversary is also represented in the same augmented basis:

```text
w_t = F_t z_t.
```

The adversarial and clean closed loops are then

```text
z_{t+1}^{adv} = M_t^{adv} z_t + eta_t^{adv}
z_{t+1}^{0}   = M_t^{0}   z_t + eta_t^{0}.
```

`M_t^{adv}` includes the plant, estimator, controller, and matched analytical
disturbance law. `M_t^0` keeps the same controller and estimator but sets the
adversarial disturbance to zero. The clean output-feedback comparator is not a
perfect-state controller and not a different estimator.

The cost remains the true task cost. State and terminal costs are evaluated on
the true plant state or true task-error coordinates:

```text
C_t = X_t^T Q_t X_t + U_t^T R_t U_t
C_T = X_T^T Q_f X_T.
```

Here `X_t z_t` selects the true plant/task-error coordinates, not the estimated
state, while `U_t z_t` is the actual estimate-based control that receives the
`R_t` cost. With these `C_t`, `C_T`, and the output-feedback closed-loop
matrices, the same deterministic, backward-noise-trace, or forward moment
propagation formulas above compute `D_beta_ref`.

The output-feedback recursion is exact under the same linear-Gaussian
post-synthesis assumptions as the full-state recursion. It is not exact if the
estimator is nonlinear, if the adversary is clipped or reoptimized online, if
noise is state-dependent, or if delayed/target coordinates are omitted from the
augmented state used by the matrices.

## Relation to the H-infinity Riccati game value

These recursions should be kept separate from the H-infinity Riccati game value.
The Riccati synthesis solves a game whose value convention may include a
disturbance-energy term such as

```text
J - gamma^2 sum_t ||w_t||^2.
```

The paired damage target is not that Lagrangian value. It is

```text
J(K_beta, F_beta) - J(K_beta, 0)
```

using the task cost only. No `gamma^2 ||w||^2` term is subtracted in the damage
target. Disturbance energy can be logged as a sidecar,

```text
E_beta_ref = E_i sum_t ||w_{beta,i,t}||^2,
```

and ratios such as `D_beta_ref / E_beta_ref` can be useful diagnostics, but
they are not the primary paired damage definition.

## Practical verification protocol

The implementation should verify the recursion against explicit rollouts before
using it as an adaptive-lambda target.

1. Matrix assembly checks.
   Confirm that every matrix is in the declared augmented basis; that `M_t^0`
   differs from `M_t^{adv}` only by removal of the adversarial disturbance
   contribution; that `C_t` and `C_T` use the intended Q/R/Qf timing; and that
   the same controller and estimator are present in clean and adversarial
   output-feedback conditions.

2. Deterministic paired rollout check.
   Turn off additive noise and evaluate a set of fixed initial states or fixed
   trials. For each condition, explicitly roll out the closed-loop dynamics and
   sum the stage and terminal costs. The rollout cost should match
   `z_0^T S_0^a z_0` for the corresponding recursion within ordinary numerical
   precision. The paired rollout damage should match
   `z_0^T (S_0^{adv} - S_0^0) z_0`.

3. One-step transition check.
   For representative augmented states, compare the explicit plant/controller/
   estimator update against multiplication by `M_t^{adv}` and `M_t^0`. This
   catches basis ordering, target-offset, delay-block, and estimator-injection
   mistakes before they are hidden inside horizon sums.

4. Moment-propagation check.
   With additive noise enabled, compute the expected clean and adversarial costs
   from forward moment propagation and from the backward recursion plus
   additive-noise trace terms. These should agree when the same timing and noise
   convention is used.

5. Monte Carlo convergence check.
   Draw initial states and nominal noise from the stated distribution and
   compare Monte Carlo averages of `J(K_beta, F_beta)`,
   `J(K_beta, 0)`, and their paired difference against the moment-propagation
   expectations. Using common random numbers for the clean and adversarial
   rollouts reduces estimator variance for the difference; it should not change
   the expected value when the marginal nominal-noise law is the same.

6. Zero-disturbance sanity check.
   Setting `F_beta` to zero or replacing `M_t^{adv}` with `M_t^0` should produce
   zero paired damage up to numerical precision. A nonzero result in this check
   usually indicates a mismatch in cost timing, target offsets, stochastic
   pairing, or clean/adversarial controller state.

## Common failure modes

- Target offsets are omitted from the augmented state. If target-centered costs
  or affine dynamics are represented as if they were origin-centered, the
  recursion can match the wrong task even while dimensions look correct.
- Delay blocks are ordered or advanced incorrectly. Output-feedback and delayed
  force-filter states require the same shift convention in the controller,
  estimator, adversary, and cost selector.
- Estimator noise injection is missing or placed in the wrong coordinate. In
  output feedback, measurement, sensory, estimator, and process noise may enter
  different parts of the augmented state.
- The state cost uses the estimated state. `Q` and `Q_f` should be applied to
  the true plant/task-error coordinates unless the stated task objective
  deliberately defines otherwise. The `R` cost applies to the actual control,
  which is estimate-based in output feedback.
- The adversarial output-feedback law is replaced by a full-state law. If the
  matched analytical solution gives `w_t = F_t z_t`, using a different
  full-state-only `F_t x_t` evaluates a different adversary and a different
  reference target.
- Off-by-one control/state cost timing. The implementation must decide whether
  `c_t` charges `x_t, u_t` before the transition or a shifted state/control
  pair, then use that convention in both rollouts and recursions.
- Terminal cost is missing, duplicated, or uses a different basis from the
  rollout. `Q_f` should be applied once at the terminal state selected by the
  finite-horizon convention.
- Clean and adversarial stochastic rollouts are not paired when estimating a
  finite-sample difference. Unpaired seeds do not change the target expectation
  under the same nominal-noise law, but they can make the sampled damage noisy
  enough to obscure recursion errors or adaptive-lambda diagnostics.
