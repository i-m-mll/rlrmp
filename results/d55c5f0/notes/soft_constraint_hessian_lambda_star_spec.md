# Soft-constraint Hessian lambda-star and closed-loop adversary spec

Issue: d55c5f0

Status: planning/spec only. Do not treat this as launch approval.

## Summary

The first soft-PGD rows used a defensible scientific idea, but the implemented
scale was not yet in the same reduction convention as the analytical game. The
task loss scalar is a batch mean over per-trial analytical Q/R/Q_f sums, while
the soft adversary penalty currently uses a batch sum over epsilon energy. For
batch size 64 this makes the per-trial penalty 64x too large, or the effective
gamma about 8x larger than intended.

After that reduction bug is fixed, the remaining question is not "should lambda
equal gamma squared?" in the abstract. It is whether the GRU's implemented
epsilon-to-loss curvature has the same boundary scale as the analytical
output-feedback model. The clean next step is to estimate a frozen-GRU local
soft-game boundary,

```text
lambda_star_GRU = 0.5 * largest_eigenvalue(d2 J_GRU(epsilon=0) / d epsilon^2)
```

where `J_GRU` is the exact per-trial analytical Q/R/Q_f task loss used by
training, evaluated with the exact epsilon channel, time mask, coordinates, and
batch reduction used by the soft adversary.

Then a gamma-factor row should use

```text
lambda(beta) = beta^2 * lambda_star_GRU
```

and report

```text
c_star = lambda_star_GRU / gamma_star_analytic^2
```

as the conversion between the analytical gamma provenance and the implemented
GRU soft-game scale.

## Repo evidence

The current soft objective in
`src/rlrmp/train/cs_perturbation_training.py` evaluates:

```python
task_loss = loss_func(candidate_states, candidate, model).total
energy = jnp.sum(jnp.square(masked_delta))
objective = task_loss - soft_energy_lambda * energy
```

The full analytical Q/R/Q_f loss in `src/rlrmp/loss.py` returns a per-trial
horizon sum:

```python
sum_t x_t.T Q_t x_t + u_t.T R_t u_t + x_T.T Q_f x_T
```

Feedbax `TermTree.total` uses `aggregate()` with leaf default `jnp.mean`, so the
scalar task loss is a batch mean. Therefore the current soft objective is:

```text
(1 / B) * sum_i J_i(delta_i) - lambda * sum_i E_i(delta_i)
```

Multiplying by `B` gives the equivalent per-trial penalty:

```text
sum_i J_i(delta_i) - (B * lambda) * sum_i E_i(delta_i)
```

For the d55c5f0 rows with `B = 64`, the intended gamma factors 1.05, 1.4, and
1.8 were effectively priced like 8.4, 11.2, and 14.4 times the chosen
`gamma_star` scale. This is consistent with the observed suppression of selected
epsilon energy in the completed rows.

The analytical output-feedback diagnostic uses the same mathematical soft-game
condition in its own coordinates: the flattened cost is

```text
epsilon.T @ H @ epsilon + 2 * g.T @ epsilon + constant
```

and the gamma-penalized maximization is finite when

```text
gamma^2 * I - H
```

is positive definite. The GRU procedure below estimates the analogous `H` for
the implemented GRU objective rather than assuming it equals the analytical
model's value.

## Correct soft objective contract

The soft adversary objective should be implemented as:

```text
maximize_delta mean_i [J_i(delta_i) - lambda * E_i(delta_i)]
```

with:

```text
J_i = sum_t x_i,t.T Q_t x_i,t
    + sum_t u_i,t.T R_t u_i,t
    + x_i,T.T Q_f x_i,T

E_i = sum_t,d epsilon_delta_i,t,d^2
```

Do not divide `E_i` by time, epsilon dimension, or reach length unless the
analytical game is deliberately changed to do that. The fixed 15 cm target does
not require any lambda rescaling. Reach-dependent safety caps should remain
separate from scientific energy scaling.

Implementation implication:

```python
per_trial_energy = jnp.sum(jnp.square(masked_delta), axis=(-2, -1))
energy = jnp.mean(per_trial_energy)
objective = task_loss - lambda_ * energy
```

If the current batch-summed implementation were left in place, the algebraic
patch would be `lambda_code = lambda / B`. That is not preferred because it
hides the reduction contract inside a parameter.

## GRU lambda-star estimation

### What is being estimated

For a frozen controller and a fixed trial distribution, write the local expansion
around zero disturbance as:

```text
J(epsilon) ~= J(0) + grad.T epsilon + 0.5 * epsilon.T M epsilon
```

The local soft adversary is finite and locally concave when:

```text
lambda * I - 0.5 * M
```

is positive definite. Therefore:

```text
lambda_star_GRU = 0.5 * largest_algebraic_eigenvalue(M)
```

If another code path represents the quadratic as:

```text
J(epsilon) ~= J(0) + 2 * g.T epsilon + epsilon.T H epsilon
```

then:

```text
lambda_star_GRU = largest_algebraic_eigenvalue(H)
```

The spec and run metadata must state which convention was used.

### Can this use the no-PGD baseline?

Yes. The no-PGD baseline is a frozen differentiable GRU controller. It was not
trained with PGD, but the task can still receive epsilon inputs, roll the model
forward, evaluate the same Q/R/Q_f loss, and differentiate that loss with
respect to epsilon. That is exactly the right first calibration point because it
measures the vulnerability scale before adversarial training changes the model.

After training, the same audit can be repeated on robust rows as a diagnostic,
but the first lambda scale should not be fitted from an already-hardened model.

### Calibration set

Use the current c92/d55 task regime:

- 6D no-integrator epsilon channel.
- Fixed 15 cm target support / const-band16 support as in the current moderate
  calibrated-perturbation rows.
- H0 GRU with the same validation-selected no-PGD baseline checkpoint used as
  the comparison baseline for c92 moderate rows.
- The same full analytical Q/R/Q_f loss profile used by d55 soft rows.
- The same epsilon time mask used during training.
- A small but representative held-out calibration set over seeds, target
  directions, and perturbation-bank conditions.

The report should include the calibration set definition, checkpoint path,
model hash/spec, epsilon shape, time mask, loss objective, and reduction
convention.

### Computation

Do not materialize a full batch Hessian if the flattened dimension is large.
Use Hessian-vector products:

```python
def hvp(epsilon0, vector):
    return jax.jvp(jax.grad(objective), (epsilon0,), (vector,))[1]
```

where `objective` returns either one per-trial `J_i` or a consistently reduced
mean objective. Prefer per-trial estimates first because the soft objective is
separable across trials once both loss and energy are per-trial and then
averaged.

For each calibration trial or small batch:

1. Evaluate `J(0)` and `grad J(0)`.
2. Estimate the largest algebraic eigenvalue with power iteration or Lanczos
   using HVPs.
3. Validate selected estimates with finite-difference curvature probes along the
   estimated top eigenvector.
4. Record `lambda_star_i = 0.5 * eigmax_i` under the ordinary Hessian
   convention.
5. Summarize median, p75, p90, max, and bootstrap uncertainty.

The initial training lambda can use a conservative quantile, such as p75 or p90,
but the chosen quantile must be recorded as a scientific choice. A median scale
is more representative; a p90 scale is more cautious.

### Linear term and predicted energy

The boundary alone does not predict how much epsilon the adversary will use. The
linear term matters. Under the local quadratic model:

```text
J(epsilon) ~= J(0) + 2 * g.T epsilon + epsilon.T H epsilon
```

the soft optimum is:

```text
epsilon_star(lambda) = (lambda * I - H)^-1 g
```

for `lambda > lambda_star_GRU`. Use conjugate gradients or a small projected
solve to estimate the predicted energy curve:

```text
E_pred(lambda) = ||epsilon_star(lambda)||^2
```

This gives an objective-level activity prediction before any full GRU training
run. It is preferable to directly matching endpoint kinematics as the first
calibration method.

## Improved soft inner optimizer

The current inner optimizer remains hard-PGD-like: it normalizes per-trial
gradients and takes a fixed fraction of the safety radius. That is suitable for
hard-radius PGD but can miss a soft optimum near zero.

The improved optimizer should:

1. Include zero as an incumbent.
2. Use the corrected mean of per-trial soft objectives.
3. Use batch-mean objective only for gradients.
4. Use per-trial objective values for incumbent selection where the candidate
   family is separable across trials.
5. Evaluate a log-spaced radial search along the normalized gradient direction,
   including radii near zero.
6. Optionally add unnormalized gradient ascent, Adam, L-BFGS, or a quadratic
   HVP proposal after the radial-search baseline is tested.
7. Keep the trust radius as a safety cap only, never as the defining scientific
   budget.
8. Reject non-improving candidates and report best-vs-final endpoint gaps.

Minimum radial-search candidate set:

```text
0, 1e-7, 3e-7, 1e-6, 3e-6, ..., radius_cap
```

scaled to the actual epsilon units and time mask. The exact values can be
configurable, but the grid should resolve the `1e-5` scale where the first d55
rows showed selected perturbations.

Diagnostics to add:

- `task_loss_initial`
- `task_loss_candidate`
- `task_loss_gain`
- `epsilon_energy_mean`
- `epsilon_energy_max`
- `lambda_times_energy_mean`
- `penalty_to_task_gain_ratio`
- `inner_objective_initial`
- `inner_objective_best`
- `inner_objective_selected`
- `selected_zero_fraction`
- `selected_radius_mean`
- `selected_radius_max`
- `cap_fraction`
- `gradient_norm_at_zero_mean`
- `gradient_norm_at_zero_max`
- `lambda_star_GRU`
- `lambda_over_lambda_star_GRU`
- `effective_beta = sqrt(lambda / lambda_star_GRU)`

Tests should include:

- batch-size invariance of the corrected soft objective;
- a direct regression proving the old batch-sum penalty would differ by `B`;
- per-trial incumbent selection on a toy separable objective;
- a quadratic toy where radial search selects an interior optimum;
- preservation of the hard-L2 objective path;
- finite diagnostics in a tiny local training smoke.

## Closed-loop adversary without a persistent neural policy

### Open-loop versus closed-loop

The current broad-epsilon PGD adversary optimizes a tensor
`epsilon[batch, time, dim]` before rollout. The controller then responds to that
fixed sequence. This is open-loop with respect to the perturbed trajectory: the
epsilon at time `t` is not recomputed from the live state or observation reached
after earlier perturbations.

A closed-loop finite-dimensional PGD adversary would optimize a feedback law and
apply it inside the rollout:

```text
epsilon_t = f_phi(z_t, t)
```

where `z_t` is a live feature vector from the perturbed rollout.

This does not require a persistent neural adversary. The optimized object can be
an ephemeral per-batch feedback parameter `phi` discarded after each inner
maximization.

### Linear and affine variants

Start with time-varying linear feedback:

```text
epsilon_t = K_t z_t
```

where `z_t` should be centered in task coordinates. For target-relative plant
features, centering means subtracting the goal position from the relevant
position coordinates, as the full analytical loss already does for the state
cost. If `z_t` is fully centered, a no-bias linear gain is closest to the
classical H-infinity analogy.

The affine variant is:

```text
epsilon_t = K_t z_t + b_t
```

or, equivalently,

```text
epsilon_t = K_t (z_t - c_t) + b_t
```

with a declared center `c_t`.

The bias is principled when the chosen feature vector is not perfectly centered
or when the local adversarial optimum has a nonzero open-loop component from
the linear term in the cost expansion. However, if the bias term dominates the
energy, the row should be interpreted as mixed open-loop plus feedback rather
than as a pure H-infinity-like linear adversary.

Required affine diagnostics:

- linear contribution energy;
- bias contribution energy;
- cross-term or total energy reconciliation;
- bias-to-total energy ratio;
- feature mean and feature RMS;
- whether features were target-relative or raw;
- whether the adversary used GRU-visible features, task-visible plant state, or
  analysis-only analytical state.

### Implementation surface

The existing memoryless policy-adversary path is not automatically a true
closed-loop PGD adversary if it computes perturbations from clean rollout
features before the perturbed rollout. A true closed-loop implementation needs
the epsilon generator to run inside the differentiable rollout and see live
perturbed features.

Likely implementation options:

1. Add a Feedbax/task-level online epsilon generator hook that computes
   `epsilon_t` from current rollout features.
2. Add a separate differentiable rollout helper for c92/d55 perturbation
   training that applies the feedback adversary at each step.
3. Use a clean-rollout feature policy only as a labeled intermediate baseline,
   not as the closed-loop row.

The implementation must preserve the distinction in metadata:

- `open_loop_sequence`: optimized `epsilon[t]` tensor, fixed before rollout.
- `clean_feature_policy`: optimized feedback-looking function evaluated on a
  clean reference rollout only.
- `closed_loop_feature_policy`: optimized function evaluated online on the live
  perturbed rollout.

## Initial follow-up experiment plan

Do not launch another full training batch until the offline audit passes.

### Phase 0: offline calibration and optimizer audit

On the frozen no-PGD c92 moderate baseline:

1. Fix or locally prototype the corrected soft reduction.
2. Estimate `lambda_star_GRU` over the calibration set.
3. Run the improved soft inner optimizer over candidate `beta` values.
4. Confirm that selected epsilons are nonzero when expected, are not frequently
   cap-bound, and produce objective-improving candidates.
5. Compare predicted local-quadratic energy with actual selected PGD energy.

Candidate beta values for audit:

```text
1.05, 1.2, 1.4, 1.8
```

Use `beta = 1.4` as the first likely training ratio only after the audit shows
that it is neither effectively zero nor cap-dominated.

### Phase 1: local tests and tiny smokes

Before any remote run:

- targeted unit tests for reduction, optimizer, diagnostics, and metadata;
- tiny local smoke with corrected open-loop soft PGD;
- tiny local or CPU-limited smoke for closed-loop finite feedback if feasible;
- a 500 to 1000 batch gate before any 12000 batch continuation.

### Phase 2: first corrected training rows

After user lock-in, train a compact comparison at one ratio:

| Row label | Meaning |
| --- | --- |
| `soft_ol_beta1p4` | Corrected open-loop epsilon sequence, improved soft optimizer |
| `soft_cl_lin_beta1p4` | Closed-loop centered linear finite feedback adversary |
| `soft_cl_aff_beta1p4` | Closed-loop affine finite feedback adversary |

All rows should use:

- the same c92 moderate calibrated-perturbation GRU task regime;
- the same 6D no-integrator epsilon channel;
- the same H0 GRU and training schedule as the current c92 moderate row unless
  live spec evidence requires otherwise;
- `lambda = 1.4^2 * lambda_star_GRU`;
- corrected per-trial soft objective reduction;
- safety cap diagnostics but no hard projection as the scientific constraint.

Rows at `beta = 1.05` and `beta = 1.8` should wait until the `beta = 1.4`
optimizer and closed-loop comparison is interpretable.

## Outputs

Expected planning and audit outputs:

- `results/d55c5f0/notes/soft_constraint_hessian_lambda_star_spec.md`
- `results/d55c5f0/notes/soft_constraint_lambda_star_audit.md`
- `results/d55c5f0/notes/soft_constraint_lambda_star_audit.json`
- `_artifacts/d55c5f0/lambda_star_audit/` for bulky spectra, HVP traces, and
  diagnostic arrays

Expected run outputs after launch approval:

- `results/d55c5f0/runs/soft_ol_beta1p4.json`
- `results/d55c5f0/runs/soft_cl_lin_beta1p4.json`
- `results/d55c5f0/runs/soft_cl_aff_beta1p4.json`
- `_artifacts/d55c5f0/runs/<row>/` for checkpoints, diagnostics, logs, and
  training summaries

## Open risks and decisions

- `lambda_star_GRU` is a local frozen-model calibration scale, not a formal
  nonlinear GRU H-infinity certificate.
- The Hessian may be indefinite; use the largest algebraic eigenvalue and
  report the spectrum edge, not an absolute eigenvalue.
- The top curvature can vary across target directions and seeds; the selected
  quantile is a scientific choice.
- A local Hessian at zero may not describe large disturbances; pair it with the
  predicted-energy curve and actual PGD audit.
- The closed-loop adversary needs an online rollout hook. A clean-rollout
  feature policy should not be mislabeled as closed-loop.
- Affine bias is useful when centering is imperfect or when the local optimum
  contains an open-loop term, but bias dominance weakens the H-infinity-like
  interpretation.
- Reach scaling should remain a safety-cap choice for the fixed 15 cm rows, not
  a silent lambda scaling.
- The original analytical `gamma_star` remains provenance for the model family;
  the GRU-local `lambda_star_GRU` is the principled experimental conversion
  needed before choosing training lambdas.
