# Soft-Constraint Adversary Lambda Formalism

## Purpose

This note records the formal method for RLRMP soft-constraint adversary
families and lambda calibration. It is a methods artifact for future run specs,
audits, and reviews. It defines the shared soft game, the local curvature and
gradient-pressure estimators, the adversary-family distinctions, required
diagnostics, and the stale constraints that must not be promoted into future
lambda recommendations.

The motivating scientific object is an H-infinity-like disturbance-energy
penalty applied to the implemented RLRMP controller and task loss. The method is
not a theorem that a trained nonlinear controller has a particular analytical
H-infinity level. It is a controlled way to assign and audit soft adversary
pressure inside the actual GRU training objective.

Notation used below:

- `Q/R/Qf`: state, action, and terminal-state cost weights in the finite-horizon
  task objective.
- `HVP`: Hessian-vector product, used to estimate curvature without forming a
  dense Hessian.
- `Lanczos`: an iterative eigensolver used with HVPs to estimate leading
  generalized-eigen directions.
- `KKT`: Karush-Kuhn-Tucker optimality conditions; here, local checks that a
  selected adversary is stationary or correctly constrained.
- `BPTT`: backpropagation through time for recurrent adversaries.

## Scope and Non-Goals

This document is not launch authorization. It does not approve training rows,
cloud runs, auth requests, or issue closure. A run remains blocked until a
separate no-launch spec is accepted and the user explicitly approves launch.

This document is not a chronology. Historical rows and prior audits appear only
where they affect the formal method, for example when a prior constraint is now
known stale or invalid.

This document separates three method classes:

- Fixed-lambda rows: lambda is chosen before the run and remains fixed.
- Preplanned schedules: lambda follows a schedule specified before launch.
- Adaptive-curriculum rows: lambda is updated online from diagnostics,
  constraint violation, phenotype error, or a dual update.

Only fixed-lambda rows support the cleanest H-infinity-like interpretation:
the controller is trained against one fixed soft game. Adaptive rows may be
useful engineering methods, but they are different scientific objects and must
be labeled as adaptive curricula.

This document also separates adversary families. A direct-epsilon lambda is not
automatically valid for a restricted shared finite policy, an affine policy, an
MLP adversary, or a recurrent adversary. A common physical energy price across
families can be a deliberate experiment, but it is not the same as
family-calibrated beta pressure.

## Common Mathematical Framework

For a frozen controller with parameters `psi` and an adversary family `A` with
parameters `eta`, define the inner soft game as:

```text
maximize_eta  J_psi(eta) - lambda * E(eta)
```

Definitions:

- `psi`: controller parameters, typically a frozen or training GRU controller.
- `eta`: adversary parameters. These may be a direct epsilon tensor, finite
  gain matrices, affine gain/bias parameters, MLP weights, or GRU adversary
  weights.
- `epsilon_eta[i,t,d]`: realized disturbance inserted into the plant or process
  channel for trial `i`, time `t`, and disturbance component `d`.
- `J_psi(eta)`: task loss under controller `psi` and realized disturbance
  `epsilon_eta`. For H-infinity-like calibration this is the same Q/R/Qf
  horizon loss used by training, with the same target centering, masks, and
  reductions.
- `E(eta)`: realized disturbance energy. The default contract is batch mean of
  per-trial horizon/component sums:

```text
E(eta) = mean_i sum_t,d mask[i,t] * epsilon_eta[i,t,d]^2
```

The batch reduction must match the task-loss reduction. If the task loss is
`mean_i J_i`, the soft energy must also be `mean_i E_i`. A batch-summed energy
term changes the effective per-trial lambda by the batch size.

Do not divide energy by time, disturbance dimension, or reach length unless the
scientific objective deliberately changes. Time masks and trial masks are part
of the energy operator. Safety caps, trust radii, and optimizer rejection rules
are diagnostics or numerical safeguards; they are not the scientific energy
definition unless explicitly promoted and justified.

The zero incumbent is always part of the inner maximization:

```text
eta = eta_zero
epsilon_eta = 0 or near zero
objective = J_psi(0)
```

An adversary update is selected only if it improves the penalized objective over
the zero incumbent. Non-improving candidates should be reported as zero rather
than treated as an implicit perturbation.

## Curvature and Lambda Formalism

For a frozen controller and calibration batch, write the stacked disturbance
vector as:

```text
e_eta = vec(epsilon_eta[i,t,d])
```

Around a zero-output or near-zero-output adversary initialization `eta0`, the
adversary output has local linearization:

```text
e_(eta0 + delta_eta) ~= e_eta0 + D * delta_eta
```

where `D` is the disturbance-output Jacobian of the adversary family.

With the mask/reduction energy operator `W`, the local energy metric is:

```text
G = D^T W D
E(eta0 + delta_eta) ~= delta_eta^T G delta_eta
```

This simplified energy expansion assumes exact zero realized disturbance at the
linearization point, `e_eta0 = 0`. If initialization is only near zero, the
first-order energy term must be carried:

```text
E(eta0 + delta_eta) ~= E0
                         + 2 * e_eta0^T W D delta_eta
                         + delta_eta^T G delta_eta
```

Near-zero initialization is therefore a distinct approximation. It is useful
for MLP/GRU adversaries when exact zero output blocks useful gradients, but it
must report the output scale, the linear energy term, and sensitivity to that
scale.

The task loss has local expansion:

```text
J(eta0 + delta_eta) ~= J0
                         + g^T delta_eta
                         + 0.5 * delta_eta^T H delta_eta
```

The local soft objective is:

```text
J0 + g^T delta_eta + 0.5 * delta_eta^T H delta_eta
   - lambda * delta_eta^T G delta_eta
```

The curvature boundary for adversary family `A` is:

```text
lambda_star_A_curv = 0.5 * lambda_max(H, G)
```

where `lambda_max(H, G)` is the largest generalized eigenvalue:

```text
H v = mu G v
```

Equivalently, on the range of `G`:

```text
lambda_star_A_curv =
    0.5 * lambda_max(G^(-1/2) H G^(-1/2))
```

Use the largest algebraic eigenvalue, not the largest absolute eigenvalue. A
negative curvature direction does not define the soft-adversary blow-up
boundary.

For neural adversaries and any nonlinear output map, distinguish the exact
parameter Hessian from the realized-epsilon tangent curvature. The preferred
calibration object is usually the tangent or Gauss-Newton-style curvature:

```text
H_A_tan = D^T M D
M = d2 J / d epsilon^2 at the linearization point
```

This estimates the dangerous disturbance-output directions available to the
local adversary. It does not include every term in the full parameter Hessian
when the output map is nonlinear and `grad_epsilon J` is nonzero. If a row uses
the full parameter Hessian instead, the report must say so and explain why raw
parameter curvature is the intended object.

Curvature alone is insufficient when the local adversary is driven by the
linear term. With a trust radius or diagnostic radius `r`, the gradient-pressure
scale is:

```text
lambda_star_A_grad = sqrt(g^T G^+ g) / (2 * r)
```

Here `r` is a realized-epsilon norm radius:

```text
r = sqrt(delta_eta^T G delta_eta)
```

so the corresponding energy is `r^2`. The formula says which lambda makes the
local linear optimum have realized norm on the order of that radius. It is a
pressure diagnostic, not a substitute for the curvature boundary.

A practical family-specific local scale is:

```text
lambda_star_A =
    p90 over calibration trials/batches/checkpoints of
    max(lambda_star_A_curv, lambda_star_A_grad)
```

The report should also include median, p75, p90, max, finite/nonfinite counts,
and sensitivity to the selected ridge or nullspace policy. The p90 summary is a
continuity and caution choice; it is not mathematically mandatory. A median
scale is more typical, while p90 is more conservative.

For a row labeled by beta:

```text
lambda_A(beta) = s_A * beta^2 * lambda_star_A
```

where:

- `beta = 1` means the local family-specific boundary scale.
- `beta > 1` means a stronger energy price and therefore a weaker adversary,
  all else equal.
- `beta < 1` means a weaker energy price and therefore a stronger adversary,
  all else equal.
- `s_A = 1` is the formal local-curvature calibration.
- `s_A != 1` is an explicit second-stage correction, such as a phenotype scale
  or finite-radius activity correction. It must be named in row labels and run
  specs.

Avoid calling `beta` an analytical gamma ratio unless the lambda-to-gamma
mapping is stated for the row. In this method record, beta is a row label for
relative soft energy price around the estimated local boundary.

## Tangent Convention and Feature Dependence

The Jacobian `D` must match the scientific object being calibrated. There are
two common conventions:

```text
fixed-feature tangent:
    D = partial epsilon_eta(features_clean) / partial eta

live closed-loop tangent:
    D = d epsilon_eta(rollout_eta) / d eta
```

The fixed-feature tangent treats features as frozen, usually from a clean or
precomputed rollout. It is useful for debugging, replayable frozen-batch
audits, and checking whether a policy family has any useful direction in a
given feature basis. It is not by itself proof that the live closed-loop
training game is calibrated.

The live closed-loop tangent differentiates through the way adversary outputs
change the rollout state and later features. It is the relevant object when the
training row claims a feedback adversary evaluated on perturbed-rollout
features. If the implementation uses static clean-rollout features, the row
must be labeled as a clean-feature or static-feature policy, not as a true live
closed-loop adversary.

Future reports should state which tangent convention was used for each lambda
estimate. A fixed-feature lambda may be a launch blocker or diagnostic, but it
should not be silently promoted to a live closed-loop lambda without a live
rollout check or an explicit approximation argument.

### Singular or Ill-Conditioned Energy Metrics

`G` is often singular or ill-conditioned. This is expected for restricted or
overparameterized adversaries: many parameter directions may not change the
realized disturbance on a calibration batch.

Acceptable treatments:

- Restrict generalized eigenvalue and gradient-pressure calculations to the
  numerical range of `G`.
- Use a documented ridge:

```text
G_rho = G + rho * I
```

- Report ridge value, rank estimate, condition estimate, and sensitivity over a
  small ridge grid.
- Use JVP/VJP definitions of `G v = D^T W D v` rather than dense matrices when
  dimensions are large.

Unacceptable treatments:

- Compute lambda in raw parameter norm and treat it as realized-epsilon lambda.
- Ignore null directions that create optimizer instability.
- Use the direct-epsilon `G = I` estimate for a restricted policy without
  projecting through that policy's realized-epsilon map.

## Adversary Families

### Direct Epsilon

Definition:

```text
eta = epsilon[i,t,d]
epsilon_eta = eta
```

The adversary chooses an independent disturbance sequence for each trial in the
batch. This is the largest local adversary space among the families described
here.

Energy metric:

```text
G = W
E(epsilon) = mean_i sum_t,d mask[i,t] * epsilon[i,t,d]^2
```

Lambda estimator:

```text
M = d2 J / d epsilon^2 at the linearization point
lambda_star_direct_curv = 0.5 * lambda_max(M, W)
lambda_star_direct_grad = sqrt(g_epsilon^T W^+ g_epsilon) / (2r)
```

In practice, compute per-trial or small-batch HVP/Lanczos estimates in epsilon
coordinates, then summarize over trials, target conditions, seeds, and
checkpoints.

Diagnostics:

- Selected energy and selected norm/cap.
- Raw task-loss gain over zero.
- Energy penalty and penalty-to-raw-gain ratio.
- Penalized objective gain over zero.
- Zero fraction and cap-bound fraction.
- KKT/radial margin at selected epsilon.
- Gradient norm at zero.
- Batch-size and reduction invariance.

Training risks:

- Direct epsilon can become cap-dominated if lambda is too small or the cap is
  too low.
- Direct epsilon may be too permissive relative to feedback-like analytical
  adversaries because it can choose trial-specific open-loop sequences.
- Direct epsilon lambda does not determine the lambda for restricted shared
  policies or neural adversaries.

Difference from other families:

Direct epsilon tests the most flexible per-trial disturbance sequence. It is a
useful reference and calibration substrate, but it does not imply that a shared
linear, affine, MLP, or recurrent adversary has the same local curvature scale.

### Linear No-Bias Finite Policy

Definition:

```text
epsilon[i,t] = K_t phi[i,t]
eta = {K_t}
```

`phi[i,t]` is the live feature vector used by the finite policy. It may include
target-centered plant state, delayed feedback, controller-visible variables, or
other declared features. The gain sequence is shared across the calibration
batch unless the row is explicitly per-trial.

Energy metric:

Let `theta = vec({K_t})`. The map from parameters to realized epsilon is
linear for fixed features:

```text
epsilon = A theta
G_linear = A^T W A
E(theta) = theta^T G_linear theta
```

Lambda estimator:

```text
lambda_star_linear_curv = 0.5 * lambda_max(H_theta, G_linear)
lambda_star_linear_grad = sqrt(g_theta^T G_linear^+ g_theta) / (2r)
lambda_star_linear = p90 max(curv, grad)
```

Diagnostics:

- `sqrt(g^T G^+ g)` at zero.
- Effective beta under any reused lambda:

```text
beta_effective_linear = sqrt(lambda_used / lambda_star_linear)
```

- Per-trial linear versus shared-batch linear comparison.
- Feature mean, RMS, scaling, and target/phase coverage.
- Ridge-fit projection from direct epsilon into the linear basis.
- Energy-normalized line search along `G^+ g`, top generalized eigenvectors,
  known directions, and random energy-normalized directions.

Training risks:

- No-bias shared policies can be first-order dead even when direct epsilon is
  harmful.
- Batch sharing and target heterogeneity can average the gradient away.
- Raw parameter-space gradient steps are poorly aligned with realized epsilon
  energy.
- Zero reinitialization each batch may repeatedly miss finite-policy basins.
- If features are not centered or not scaled, the policy can be ill-conditioned
  or structurally unable to express useful directions.

Difference from other families:

Linear no-bias is a restricted feedback-like family. It is closer in spirit to
analytical feedback adversaries than direct epsilon, but only if the live
features are the right state basis. It lacks an open-loop or bias term and
therefore cannot express arbitrary trial-specific direct-epsilon components.

### Affine Finite Policy

Definition:

```text
epsilon[i,t] = K_t phi[i,t] + b_t
eta = {K_t, b_t}
```

The affine family adds a shared time-varying bias or open-loop component.

Energy metric:

For each time step:

```text
theta_t = [vec(K_t), b_t]
A[i,t] = [phi[i,t]^T kron I, I]
epsilon[i,t] = A[i,t] theta_t
```

Then:

```text
G_affine = mean/sum over i,t of A[i,t]^T W[i,t] A[i,t]
E(theta) = theta^T G_affine theta
```

Lambda estimator:

```text
lambda_star_affine_curv = 0.5 * lambda_max(H_theta, G_affine)
lambda_star_affine_grad = sqrt(g_theta^T G_affine^+ g_theta) / (2r)
lambda_star_affine = p90 max(curv, grad)
```

Diagnostics:

- Linear contribution energy.
- Bias contribution energy.
- Cross-term or total-energy reconciliation.
- Bias-to-total energy ratio.
- Raw pre-cap policy output norm.
- Selected/clipped norm if a cap is used as a diagnostic.
- Known direction checks: mean direct-epsilon bias, ridge linear fit plus bias,
  and scalar amplitude line search.

Training risks:

- A positive affine result may mean the no-bias restriction was too strong, not
  that affine lambda should set linear lambda.
- If the bias dominates, the row is mixed open-loop plus feedback rather than a
  pure feedback analogue.
- Projection or clipping can hide raw policy blow-up if the penalty is computed
  on selected/clipped epsilon rather than raw epsilon.

Difference from other families:

Affine is a diagnostic decomposition and a candidate adversary family in its own
right. It should not be used to set the linear no-bias lambda. It answers
whether a shared open-loop component is needed for useful finite-policy
adversarial pressure.

### MLP Adversary

Definition:

```text
epsilon[i,t] = a_eta(phi[i,t])
```

An MLP adversary is an amortized disturbance function. It shares parameters
across trials and times according to its feature design.

Required method definition:

- Feature set and timing.
- Whether features are controller-visible, analysis-only, delayed, or live
  perturbed-rollout features.
- Time or phase encoding.
- Target encoding.
- Output channel and coordinate convention.
- Output transform, clipping, or smooth saturation.
- Initialization and whether zero output is exact or approximate.
- Optimizer and whether adversary state carries across batches.

Energy metric:

```text
D_MLP = d vec(epsilon_eta) / d eta at eta0
G_MLP = D_MLP^T W D_MLP
```

Prefer function-space tangent curvature:

```text
H_MLP_tan = D_MLP^T M D_MLP
```

where `M = d2 J / d epsilon^2` at the linearization point. This estimates
dangerous disturbance functions the local MLP can generate, not arbitrary raw
parameter curvature.

Lambda estimator:

```text
lambda_star_MLP_curv = 0.5 * lambda_max(H_MLP_tan, G_MLP)
lambda_star_MLP_grad = sqrt(g_MLP^T G_MLP^+ g_MLP) / (2r)
lambda_star_MLP = p90 max(curv, grad)
lambda_MLP(beta) = s_MLP * beta^2 * lambda_star_MLP
```

Diagnostics:

- Feature normalization and feature RMS table.
- Output transform derivative and saturation fraction.
- Readout-only tangent versus full-network tangent if final layer starts at
  zero.
- Selected energy, raw gain, penalty, penalized gain, zero fraction, cap-bound
  fraction, KKT/radial margin, and nonfinite status.
- Optimizer agreement between Adam, line search, local quadratic proposal, and
  HVP/top-direction probes where feasible.

Training risks:

- Exact zero-output initialization may make only the final readout layer active
  in the tangent space.
- Small-output full-network initialization can make lambda depend on the
  initialization scale.
- Missing time or target features can make the MLP much weaker than direct
  epsilon.
- Poor feature scaling can turn Adam parameter geometry into the dominant
  method.
- Saturating output transforms can make raw parameters blow up while selected
  epsilon remains capped.

Difference from other families:

An MLP is not direct epsilon with fewer parameters. It is a shared function
class. Its lambda must be estimated in the realized-epsilon tangent space
induced by its feature set and output transform.

### GRUadv Adversary

Definition:

Use `GRUadv` to avoid confusion with the controller GRU.

```text
h[i,0] = h0
h[i,t+1] = F_eta(h[i,t], phi[i,t])
epsilon[i,t] = O_eta(h[i,t], phi[i,t])
```

The hidden reset convention, horizon, observation timing, and feature set are
part of the adversary family definition.

Energy metric:

```text
D_GRUadv = d vec(epsilon_eta) / d eta at eta0
G_GRUadv = D_GRUadv^T W D_GRUadv
```

`D_GRUadv` includes recurrence and backpropagation through time over the full
horizon.

Lambda estimator:

```text
H_GRUadv_tan = D_GRUadv^T M D_GRUadv
lambda_star_GRUadv_curv = 0.5 * lambda_max(H_GRUadv_tan, G_GRUadv)
lambda_star_GRUadv_grad = sqrt(g_GRUadv^T G_GRUadv^+ g_GRUadv) / (2r)
lambda_star_GRUadv = p90 max(curv, grad)
lambda_GRUadv(beta) = s_GRUadv * beta^2 * lambda_star_GRUadv
```

Diagnostics:

- Hidden reset convention.
- Hidden norm and hidden-state stability.
- Output saturation and clipping diagnostics.
- Timing and observation-access audit.
- Carry-over versus zero-start optimizer comparison.
- Finite/nonfinite gradients through BPTT.
- Same selected-energy/raw-gain/penalty/KKT diagnostics used for other
  families.

Training risks:

- The recurrent adversary can exploit simulator timing or observations not
  intended for the H-infinity analogue.
- Hidden state can become large while output energy remains bounded.
- Reinitializing the adversary optimizer from zero every batch can make a
  recurrent policy appear dead.
- A powerful GRUadv can become a co-adaptive opponent rather than a local
  worst-case disturbance generator.

Difference from other families:

GRUadv is a learned recurrent disturbance generator. It may be less expressive
than direct epsilon at initialization, but more structured or more dangerous
after adversary optimization because it can use history and state. Its lambda
therefore must be architecture-specific.

## Diagnostics and Tests

Every future soft-adversary row or no-launch spec should report enough data to
distinguish a valid soft optimum from zero suppression, cap domination,
optimizer failure, and stale scale transfer.

Minimum frozen-batch audit diagnostics:

- `J(0)` and zero incumbent objective.
- Selected raw task loss and raw task-loss gain over zero.
- Realized energy `E(selected)`.
- Energy penalty `lambda * E(selected)`.
- Penalized gain over zero.
- Selected norm, norm/cap, and cap-bound fraction if a cap is present.
- Zero fraction.
- Raw gain, selected/clipped gain, and raw-to-selected ratio for policy
  adversaries with caps or output transforms.
- Gradient norm at zero in the realized-energy metric.
- `sqrt(g^T G^+ g)` or ridged equivalent.
- Curvature estimate, gradient-pressure estimate, and selected quantile.
- KKT or radial margin at selected epsilon:

```text
radial_margin = <grad_epsilon J(epsilon) - 2 lambda W epsilon,
                 epsilon / sqrt(epsilon^T W epsilon)>
```

- Interior KKT residual when selected epsilon is not cap-bound.
- Optimizer agreement across at least two independent methods when the result
  will gate launch.
- Finite/nonfinite status for objective, gradients, HVPs, selected epsilon, and
  policy raw output.

Interpretation rules:

- Zero selected adversary with near-zero gradient and no positive curvature
  means the family may be locally dead on that batch.
- Zero selected adversary with nonzero energy-metric gradient usually indicates
  lambda too large, optimizer failure, or bad parameter geometry.
- Cap-bound selected adversary with positive radial margin means the soft game
  still wants to move outward; lambda is too small for that cap or the cap is
  defining the game.
- Cap-bound selected adversary with negative radial margin suggests the
  optimizer or projection path may have selected the boundary incorrectly.
- Positive known-direction line search means a prior zero result cannot support
  a no-useful-policy claim.
- Direct epsilon nonzero while shared no-bias linear is zero does not by itself
  indicate a bug; it may reflect restricted expressivity, batch sharing,
  missing bias, or family-specific lambda mismatch.

Required invariance and replay tests:

- Batch-size invariance of the corrected soft objective.
- Regression showing batch-summed energy would change effective lambda by `B`.
- Same-batch replay of frozen audit inputs when comparing families.
- Same controller checkpoint, PRNG keys or deterministic batch descriptor,
  target support, masks, and optimizer config for family comparisons.
- Batch-mean versus per-trial selection tests on toy separable objectives.
- HVP finite-difference validation along top directions.
- Singular `G` ridge/range sensitivity.
- Live perturbed rollout versus clean-rollout feature policy check for
  closed-loop claims.

## Training Considerations

Optimizer geometry matters. The penalty is on realized epsilon energy, not raw
parameter norm. Direct epsilon can use epsilon-space gradients; finite policies
and neural adversaries need either energy-metric normalization, line search in
realized-epsilon norm, G-preconditioned proposals, or optimizer diagnostics that
show Adam/L-BFGS found the same objective region.

Adam, L-BFGS, and reference solvers have distinct roles:

- A line search along known directions is reference evidence that useful
  directions exist.
- L-BFGS or local quadratic proposals are strong frozen-audit reference solvers.
- Adam is training-relevant when the actual inner maximizer will use Adam.
- Agreement is about selected energy, cap-bound fraction, objective gain, and
  dominant components, not exact equality of epsilon tensors.

Initialization is part of the method. Direct epsilon can start at zero. Finite
policies may need energy-normalized random starts, fitted direct-epsilon starts,
projected analytical starts, or carry-over. MLP/GRUadv zero-output
initialization can restrict the initial tangent class to the readout layer.

Feature scaling and timing must be explicit. For finite and neural policies,
record feature means, RMS, target centering, time/phase inputs, delayed feedback
conventions, and whether features are clean-rollout or live perturbed-rollout
features. A feedback-looking policy evaluated only on clean rollout features is
not a live closed-loop adversary.

Output transforms and saturation must be included in `D`. If epsilon is
generated through `tanh`, clipping, projection, or a smooth radial squash, the
lambda estimator and diagnostics must state whether the penalty is computed on
raw output, transformed output, or clipped selected output. Penalizing clipped
output can let raw policy parameters blow up without paying additional energy.

Batch sharing and target heterogeneity are scientific choices. A shared policy
over a heterogeneous batch can average away useful directions. Compare direct,
per-trial finite, shared finite, and affine variants before treating a shared
no-bias zero result as structural.

Adaptive lambda is a separate method. It may be implemented through local
lambda re-estimation, realized-energy targets, cost/energy ratio targets, or
dual updates. Such rows should be labeled adaptive and compared against at least
one fixed-lambda row.

Phenotype scale is a second-stage correction:

```text
lambda_A(beta) = s_A * beta^2 * lambda_star_A
```

Use `s_A` only after the formal family-specific local scale and frozen audits
are coherent. A phenotype scale can match peak velocity or another behavioral
summary, but it is not a stronger proof of analytical gamma equivalence.

### Adaptive Lambda Contract

An adaptive-lambda row must define:

- target signal: realized energy, cap-bound fraction, cost/energy ratio,
  phenotype error, or another explicit diagnostic;
- update cadence: per inner step, per batch, per checkpoint, or per phase;
- smoothing and bounds;
- whether lambda updates receive gradients or are stop-gradient controller
  logic;
- how the row initializes lambda;
- a fixed-lambda comparator row.

A typical log-space update has the form:

```text
log lambda_(k+1) =
    (1 - rho) * log lambda_k
    + rho * log(beta^2 * lambda_star_hat_A,k)
```

or, for a dual-style constraint:

```text
lambda_(k+1) = [lambda_k + alpha * constraint_violation_k]_+
```

Both are adaptive curricula. They should not be described as fixed-beta
H-infinity-equivalent rows.

### Phenotype Scale Contract

A phenotype scale row must define:

- phenotype metric: peak velocity, time-to-peak, endpoint quality, perturbation
  response, or another predeclared summary;
- calibration set and held-out check;
- fit rule for `s_A`;
- allowed range or clipping of `s_A`;
- whether `s_A` is shared across adversary families or family-specific;
- row labels that expose the correction.

Use labels such as:

```text
linear_no_bias beta1p4 policyHVP s1p0
linear_no_bias beta1p4 policyHVP pheno_s0p5
MLP beta1p4 policyHVP s1p0
GRUadv beta1p4 policyHVP pheno_s0p5
```

This keeps formal local calibration and behavior-matching correction separate.

Artifact and replay requirements:

- Persist run specs with active optimizer, lambda source, family, beta, `s_A`,
  masks, reductions, cap policy, and output transform.
- Persist selected epsilon arrays or compact replayable representations when
  direct-to-policy projection or same-batch replay may matter.
- Persist checkpoint/pre-update model identity, deterministic batch descriptor,
  target metadata, PRNG seeds, optimizer config, and finite/nonfinite flags.
- Keep bulk arrays outside tracked result notes, but keep scalar summaries and
  replay pointers durable.

## Stale or Invalidated Constraints

The batch-sum versus batch-mean bug invalidated early soft-PGD rows for
scientific interpretation. Those rows priced the adversary energy as a batch sum
while task loss was a batch mean, making the effective per-trial lambda larger
by the batch size. Their diagnostics can explain operational behavior, but they
must not be used as evidence that a corrected soft objective failed or
succeeded.

Cap/trust-radius-derived floors are stale as launch-facing lambda
recommendations. A cap-derived threshold answers a different question:

```text
At what lambda does an optimizer find useful perturbations that are interior
to this particular borrowed cap?
```

That is not the same as:

```text
What lambda calibrates the soft objective for this adversary family?
```

The old radius-pressure and cap-to-interior quantities should remain
diagnostic-only unless a future spec explicitly justifies a scientific cap.
They can be useful for detecting cap domination and optimizer failures, but
they do not define fixed-lambda launch rows.

The inherited `0.0045455`-scale cap is therefore a trust-region or diagnostic
sidecar in this method record, not a formal lambda recommendation.

## Literature Note

[arXiv:2404.14405v2](https://arxiv.org/abs/2404.14405) is relevant as precedent
for learned state-conditioned disturbers and adaptive H-infinity-inspired
constraints. It should not be used as a fixed-lambda recipe for RLRMP rows.

The useful distinction is:

- The literature precedent supports learned adversaries and adaptive constraint
  or dual-update formulations.
- RLRMP fixed-lambda rows require a precomputed family-specific lambda for the
  implemented controller, loss, masks, and disturbance channel.
- A dual-adaptive RLRMP row would be a different method from a fixed-beta HVP
  row and should be labeled accordingly.

## Implementation Checklist for Future Rows and Specs

Before a row is launch-facing:

1. Define the adversary family exactly: direct, linear no-bias, affine, MLP, or
   GRUadv; include features, timing, output transform, optimizer, and reset or
   carry-over conventions.
2. State the soft objective:

```text
mean_i [J_i(epsilon_i) - lambda * E_i(epsilon_i)]
```

3. Verify task loss and energy use matching batch reductions.
4. Compute or cite `lambda_star_A` for the active family using the realized
   epsilon energy metric.
5. Report curvature, gradient-pressure, p50/p75/p90/max, finite/nonfinite
   counts, and ridge/range handling for singular `G`.
6. State the row mapping:

```text
lambda_A(beta) = s_A * beta^2 * lambda_star_A
```

7. Label any `s_A != 1` as phenotype or finite-radius correction.
8. Run frozen-batch audits at the proposed lambda on the same objective used by
   training.
9. Report selected energy, raw gain, penalty, penalized gain, zero fraction,
   cap-bound fraction, KKT/radial margin, gradient norm at zero, and optimizer
   agreement.
10. Confirm closed-loop rows use live perturbed-rollout features rather than
    clean reference features, or label them as clean-feature policies.
11. Keep cap/trust radius as a diagnostic sidecar unless a scientific cap is
    explicitly justified.
12. Persist replayable selected epsilon or compact policy artifacts when later
    direct-to-policy projection or same-batch comparison may be needed.
13. State whether the row is fixed-lambda, preplanned schedule, or adaptive
    curriculum.
14. Include a no-launch statement unless the document is a separately approved
    run spec.

## Provenance

This methods record was prepared from prior RLRMP soft-adversary review packets,
formalism notes, frozen-audit summaries, and follow-up synthesis artifacts.
Those source materials are retained separately as issue-linked evidence. The
main body above is the normative method record and should be read independently
of the review sequence that produced it.
