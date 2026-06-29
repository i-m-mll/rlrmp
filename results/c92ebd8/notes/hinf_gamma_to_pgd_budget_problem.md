# H-Infinity Gamma-Derived Budgets for GRU PGD Training

## Purpose

This note states the conceptual problem in using analytical H-infinity gamma
values to define hard L2 PGD budgets for training recurrent neural-network
controllers. The goal is to separate the formal H-infinity object from the
practical adversarial-training object, and to identify where the translation
between them is principled, heuristic, or potentially misleading.

## Analytical H-Infinity Object

The analytical controller is defined by a finite-horizon linear-quadratic
H-infinity game. For a plant, cost schedule, information structure, and
attenuation parameter `gamma`, the game has the form

```text
min_controller max_disturbance
    sum_t (x_t^T Q_t x_t + u_t^T R_t u_t - gamma^2 ||w_t||^2)
    + x_T^T Q_f x_T.
```

Here `x_t` is the analytical state, `u_t` is the control command, and `w_t` is
the disturbance. The disturbance term is a soft energy penalty inside the
maximization objective, not a hard L2 constraint.

For a `gamma` where the finite-horizon Riccati recursion remains well defined
for this plant, cost, disturbance channel, and information structure, the
dynamic-programming construction gives both an analytical robust controller and
an analytical worst-case disturbance policy. In a full-state game these are
time-varying linear policies of the form

```text
u_t = -K_t x_t
w_t = F_t x_t.
```

In the output-feedback or estimator-in-loop formulation used here, the
controller acts only on the state estimate, while the analytical disturbance
policy is represented as feedback on the joint plant and estimator state:

```text
u_t = -K_t xhat_t
w_t = F_t [x_t, xhat_t].
```

Thus the analytical adversary is not an informal perturbation budget. It is an
idealized feedback policy implied by the same finite-horizon game that defines
the robust controller.

## Meaning of Gamma

`gamma` is an attenuation parameter for the analytical H-infinity game. Smaller
feasible values of `gamma` impose a stricter robustness requirement. The value
`gamma_star` denotes the approximate feasibility boundary for the specified
plant, cost, horizon, disturbance channel, and information structure. Below that
boundary the Riccati game is not well posed for that setup.

The relationship between `gamma` and disturbance strength is therefore not the
same as the relationship between a PGD radius and perturbation strength. In the
analytical game, `gamma` changes:

- the controller produced by the Riccati recursion;
- the adversary feedback policy produced by the same recursion;
- the closed-loop trajectory under that controller/adversary pair;
- the realized disturbance energy on a particular rollout.

The value of `gamma` is not itself a hard disturbance budget.

## GRU Training Setup

The learned controller is a GRU trained on a Crevecoeur-and-Scott-style reaching
task. For the rows discussed in this note, the no-integrator reaching setup is:

- the physical plant has 6 physical coordinates: 2D position, 2D velocity, and
  2D force-filter/load state;
- the analytical delayed plant state contains the current physical block plus
  five lag blocks, giving a 36D delay-augmented state;
- the GRU is not fed the full 36D analytical state;
- the GRU receives delayed target-relative feedback;
- in H0 force-filter rows, the feedback also includes delayed force-filter
  coordinates, giving a 6D controller input consisting of delayed
  target-relative 2D position, delayed target-relative 2D velocity, and delayed
  2D force-filter state;
- process disturbance enters through the plant process channel, with `B_w`
  injecting into the current physical block and not directly into lag blocks;
- sensory noise is applied on the delayed feedback channel after the delayed LSS
  feedback selector, not as direct noise on the full analytical state.

The training objective is the task loss over sampled targets, stochastic
rollouts, and calibrated perturbation conditions. PGD training adds an inner
maximization over a hard L2 ball:

```text
min_GRU max_delta, ||delta||_2 <= r
    task_loss(GRU, delta).
```

The PGD radius `r` is a hard constraint on the perturbation energy available to
the training adversary. It is therefore a direct adversarial-training pressure
parameter in a way that `gamma` is not.

## Rollout-Derived Budget Procedure

One proposed bridge from the analytical H-infinity game to GRU PGD training is:

1. Select a gamma factor, such as `1.4 * gamma_star` or `1.05 * gamma_star`.
2. Solve the analytical H-infinity output-feedback game for that `gamma`.
3. Roll out the resulting analytical controller with the corresponding
   analytical worst-case disturbance feedback policy.
4. Compute the per-rollout disturbance norm, without a `dt` factor and summed
   over time and disturbance-channel dimensions:

   ```text
   r_gamma = sqrt(sum_t ||w_t||^2).
   ```

5. Use `r_gamma` as the hard L2 radius for PGD training of the GRU under the
   same process-disturbance channel convention.

This procedure answers a specific calibration question:

```text
What hard PGD radius has the same numeric summed L2 norm as the analytical
disturbance sequence w_t for this gamma, plant, horizon, information structure,
initial condition, and process-disturbance channel convention?
```

This is a provenance-matched calibration. It is not a formal equivalence between
the H-infinity game and the PGD training game.

## What the Procedure Preserves

The rollout-derived PGD radius preserves:

- the total L2 energy of one analytical worst-case disturbance rollout;
- the selected analytical plant and cost schedule;
- the selected analytical information structure;
- the selected gamma factor;
- the selected initial condition and horizon;
- a direct link to the analytical controller/adversary pair used to generate the
  disturbance.

This makes the resulting PGD row interpretable as a GRU trained with a radius
matched to a specific analytical H-infinity rollout.

## What the Procedure Does Not Preserve

The rollout-derived PGD radius does not preserve:

- the analytical adversary feedback law `F_t`;
- the temporal profile of the analytical disturbance;
- the state or estimator dependence of the analytical disturbance;
- the soft energy-penalty structure of the H-infinity game;
- the induced-gain interpretation of `gamma`;
- a monotone relationship between gamma factor and hard PGD radius;
- a theorem that the GRU trained with radius `r_gamma` has H-infinity level
  `gamma`.

The PGD adversary is a hard-radius optimizer against the nonlinear GRU training
loss. The analytical H-infinity adversary is a linear feedback policy derived
from a linear-quadratic game with a soft energy penalty. Matching one rollout's
total L2 energy does not make these adversaries equivalent.

## The Awkward Case

A stricter analytical gamma can produce a smaller rollout-derived L2 radius.
For example, in the 6D no-integrator output-feedback rollout used for this c92
budget question, the computed per-rollout process-disturbance radii were:

```text
gamma factor 1.4  -> r_gamma = 0.004545011406169036
gamma factor 1.05 -> r_gamma = 0.0017513324974961241
```

This can occur because changing `gamma` changes the whole closed-loop analytical
solution. The controller, estimator, adversary policy, and trajectory all change
together. A controller from a stricter game may suppress the state components
that drive the analytical adversary, so the realized disturbance energy on the
chosen rollout can be smaller even though the analytical robustness target is
stricter.

This creates a practical ambiguity for GRU PGD training. For a fixed loss,
perturbation family, and PGD optimizer, increasing `r` expands the feasible set
of the inner maximization. Operational training effects can still be nonmonotone
once perturbations become destabilizing, behaviorally irrelevant, or too large
for the controller to compensate. Therefore, if a gamma closer to `gamma_star`
yields a smaller `r_gamma`, then the resulting GRU row is analytically stricter
by provenance but weaker by the hard-radius PGD pressure parameter.

Both statements can be true at the same time:

```text
gamma = 1.05 * gamma_star is analytically closer to the robust feasibility
boundary than gamma = 1.4 * gamma_star.
```

and

```text
the PGD radius inferred from the gamma-1.05 rollout is smaller than the PGD
radius inferred from the gamma-1.4 rollout.
```

The first statement concerns the analytical H-infinity game. The second concerns
the hard-radius PGD training problem.

## Interpretation Risk for GRU Experiments

If a GRU trained with the gamma-1.05-derived radius performs worse than a GRU
trained with the gamma-1.4-derived radius on the specified robustness metrics,
the result has multiple possible interpretations. Relevant metrics include
perturbation-response attenuation, perturbation-response area under the curve,
endpoint error, velocity-profile distortion, and steady-state hold diagnostics.
Possible interpretations include:

- the gamma-1.05 analytical provenance may not transfer well to GRU training;
- the output-feedback analytical model may be the wrong comparator for this GRU
  setup;
- the perturbation families used during training may not match the analytical
  disturbance channel;
- the target-support distribution may change the robustness phenotype;
- the smaller gamma-1.05-derived hard PGD radius may simply provide weaker
  adversarial training pressure.

The gamma-1.05 row should therefore be described as analytically closer to the
feasibility boundary, not as stronger PGD pressure, unless its derived hard L2
radius is actually larger.

More precise row descriptions are:

```text
PGD trained with a hard L2 radius matched to the 6D output-feedback
H-infinity rollout at gamma factor 1.05.
```

and

```text
PGD trained with a hard L2 radius matched to the 6D output-feedback
H-infinity rollout at gamma factor 1.4.
```

## Clean Experimental Separation

The analytical-provenance question and the adversarial-training-pressure
question should be separated.

Analytical-provenance axis:

```text
gamma factor -> analytical output-feedback rollout -> derived PGD radius.
```

This tests whether H-infinity-derived disturbance energies are useful for GRU
training.

Training-pressure axis:

```text
fixed hard PGD radius sweep independent of gamma provenance.
```

This tests how GRU robustness depends on adversarial perturbation strength.

The two axes answer different questions. A gamma-derived row is interpretable as
a PGD row with analytical provenance matching, not as a monotone PGD-strength
row.

## Recommended Reporting Language

Use:

```text
output-feedback gamma-derived PGD radius
```

or:

```text
hard L2 PGD radius matched to the analytical output-feedback rollout at gamma
factor X.
```

Avoid:

```text
larger gamma budget
```

or:

```text
gamma-1.05 is stronger PGD.
```

The relevant distinction is:

```text
gamma controls the analytical H-infinity game.
r controls the hard-radius PGD training pressure.
```

The rollout-derived bridge maps one to the other for a specified analytical
setup, but the bridge is a calibration convention rather than a formal identity.
