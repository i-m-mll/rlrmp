# SISU Conditioning and Soft-Gamma Inner Maximization

## Purpose

This note defines the problem of using an analytical H-infinity `gamma` value
inside neural-network adversarial training, with particular attention to SISU
controllers. It separates three related objects:

- the analytical H-infinity game, where `gamma` is a soft energy-penalty
  parameter;
- the current GRU PGD training path, where adversarial strength is a hard L2
  radius;
- SISU-conditioned GRU training, where the controller receives a scalar input
  that currently represents a hard-budget energy fraction.

The goal is to state what a soft-gamma inner maximization would mean, whether
it is a coherent experiment for the current models, and how the SISU scalar
should be interpreted if the inner maximization is changed from a hard budget
to a soft energy penalty.

## Analytical Object

The finite-horizon H-infinity controller is defined by a linear-quadratic
dynamic game:

```text
min_controller max_disturbance
    sum_t x_t^T Q_t x_t + u_t^T R_t u_t - gamma^2 ||w_t||^2
    + x_T^T Q_f x_T.
```

Here `x_t` is the analytical plant or augmented plant state, `u_t` is the
control command, and `w_t` is the process disturbance. The term
`gamma^2 ||w_t||^2` is a soft price on disturbance energy inside the adversary's
objective. It is not a hard bound of the form `||w|| <= r`.

For a fixed plant, horizon, disturbance channel, cost schedule, and information
structure, `gamma_star` is the lower feasibility boundary for the Riccati game.
The finite-horizon solution is well posed only for feasible values above that
boundary, for example `gamma = 1.05 * gamma_star` or
`gamma = 1.4 * gamma_star`. Within a fixed soft-penalty objective, smaller
feasible `gamma` means a smaller disturbance-energy price and therefore a
stronger adversary pressure. Moving `gamma` closer to `gamma_star` also changes
the analytical controller, the analytical adversary, and the closed-loop
trajectory. Therefore a rollout-derived disturbance norm from the analytical
solution is a trajectory-dependent realized quantity, not a direct definition
of the adversary's budget.

## Current GRU Training Object

The c92-style learned controller is a GRU trained on a calibrated
Crevecoeur-and-Scott-style reaching task. In the rows relevant here:

- the physical plant is the 6D no-integrator plant: 2D position, 2D velocity,
  and 2D force-filter/load state;
- H0 GRU rows receive delayed target-relative sensory feedback, including
  force-filter state where applicable;
- process disturbances enter through the same 6D process-disturbance channel
  used by the calibrated perturbation setup;
- training samples target conditions, stochastic rollouts, calibrated
  perturbation conditions, and, when enabled, adversarial perturbations;
- broad-epsilon PGD currently treats the adversarial perturbation as a tensor
  over time and process-disturbance dimensions.

The current broad-epsilon PGD path is a hard-radius inner maximization:

```text
min_theta max_delta task_loss(theta, delta)
subject to ||delta||_2 <= r.
```

The implementation projects the perturbation onto a per-trial flattened
time-component L2 ball. The radius `r` is therefore an actual training-pressure
parameter: for the same model, loss, mask, and optimizer, a larger radius gives
the inner maximizer a larger feasible set.

## Current SISU Semantics

SISU rows expose a scalar input to the controller. That scalar is also used to
select the adversary's budget during training. In the current hard-radius PGD
implementation:

```text
SISU in [0, 1]
SISU = fraction of max disturbance energy
epsilon_l2_radius = max_l2_radius * sqrt(SISU)
```

This square-root rule is intentional. L2 energy scales as squared radius, so a
radius fraction of `sqrt(SISU)` gives an energy fraction of `SISU`.
The energy convention is the squared L2 norm of the masked adversarial epsilon
tensor after flattening the active time and process-disturbance dimensions for
each trial:

```text
energy_fraction =
    ||masked_delta||_2^2 / max_l2_radius^2
```

There is no `dt` factor in this convention. The radius constraint is applied per
trial before any batch averaging of the training loss, so the SISU value is not
a batch-reduced energy fraction.

The practical interpretation is:

- `SISU = 0`: no adversarial perturbation budget;
- `SISU = 0.25`: one quarter of the max energy, half of the max radius;
- `SISU = 1`: the selected maximum hard L2 budget.

For non-delayed SISU rows, the scalar may be carried on the ordinary
`trial_specs.inputs["input"]` channel. For delayed rows, the ordinary input is
already used for the go-cue or task stream, so SISU is carried on a separate
`trial_specs.inputs["sisu"]` key. In both cases the controller-visible scalar is
currently an energy-fraction label for a hard-budget adversary.

## Why Gamma-Derived Hard Radii Are Awkward

A common bridge from the analytical game to hard PGD is:

1. choose a gamma factor, such as `1.4 * gamma_star` or
   `1.05 * gamma_star`;
2. solve the analytical H-infinity game;
3. roll out the analytical controller and analytical adversary;
4. compute the realized disturbance norm;
5. use that norm as the hard PGD radius for GRU training.

This gives a provenance-matched hard radius:

```text
r_gamma = sqrt(sum_t ||w_t||^2)
```

It answers the question:

```text
What hard PGD radius has the same numeric L2 norm as this analytical
disturbance rollout?
```

It does not answer:

```text
What hard PGD radius is mathematically equivalent to this gamma?
```

The distinction matters because a stricter analytical gamma can produce a
smaller realized rollout disturbance. In the c92 output-feedback 6D
no-integrator rollout, the current concrete radii are:

```text
gamma factor 1.4  -> active L2 radius 0.004545011406169036
gamma factor 1.05 -> active L2 radius 0.0017513324974961241
```

The gamma-1.05 row is analytically closer to the H-infinity feasibility
boundary, but it gives the hard-radius PGD adversary a smaller feasible set.
That is not a contradiction. It means the current mapping has mixed semantics:
it is stricter by analytical provenance and weaker by hard PGD radius.

## Soft-Gamma Inner Maximization

A more direct H-infinity-style training objective would keep gamma as an energy
penalty inside the inner maximization:

```text
min_theta max_delta
    task_loss(theta, delta) - lambda * ||delta||^2.
```

The analytical correspondence suggests `lambda` should be proportional to
`gamma^2`:

```text
lambda = c * gamma^2.
```

The constant `c` should be treated explicitly. It is exactly `1` only if the
neural-network training objective, disturbance coordinates, loss reductions,
time weighting, and units match the analytical finite-horizon game. That is a
strong condition. The current models are close enough to make a soft-gamma
experiment scientifically meaningful, because the training task is deliberately
parameterized to match the analytical reaching model. However, direct numerical
equivalence should not be claimed without checking the following:

- the training loss uses the same `Q_t`, `R_t`, and `Q_f` geometry as the
  analytical game;
- the process-disturbance tensor uses the same physical disturbance channel;
- the L2 energy convention uses the same time weighting, including whether
  there is no `dt` factor;
- the training loss reduction over batch, time, target, and replicate axes does
  not introduce an unrecorded scale factor;
- any stochastic sensory noise, target sampling, perturbation-bank mixture, and
  nonlinear GRU policy effects are treated as departures from the analytical
  linear-quadratic game, not as exact H-infinity equivalences.

Under those conditions, a soft-gamma inner maximization is a coherent and useful
experiment. It should be reported as an H-infinity-style energy-penalized PGD
objective, not as a formal H-infinity certificate for the GRU.

## Numerical Safety Cap

A pure soft-penalty inner maximization has no hard radius. In practice, a first
implementation should still use a large numerical safety cap or trust-region
mechanism. That cap should be documented as optimizer stabilization, not as the
definition of the game.

The reason is practical rather than conceptual. If the chosen penalty is too
small relative to the nonlinear training loss curvature, finite-step PGD can
drive perturbations into numerically unstable regions. A cap also makes it
possible to diagnose whether the soft optimum is interior or repeatedly hitting
the safety boundary.

The diagnostics for a soft-gamma row should record:

- raw controller task loss;
- perturbation energy;
- penalty coefficient `lambda`;
- penalty contribution `lambda * ||delta||^2`;
- penalized inner objective;
- perturbation norm;
- whether the numerical safety cap was active;
- NaN or overflow failures, if any.

## SISU Under a Soft-Gamma Objective

The current SISU scalar should not keep the name or interpretation
`energy_fraction` when the adversary no longer has a hard radius. Under a
soft-gamma inner maximization, the scalar should represent adversary pressure
through the penalty coefficient.

A clean convention is to define SISU as a pressure fraction, where pressure is
the inverse of the soft energy penalty:

```text
pressure_fraction = lambda_ref / lambda
                  = gamma_ref^2 / gamma_eff^2
SISU = pressure_fraction
SISU = 0 means no adversary
SISU = 1 means strongest selected soft adversary
```

For `SISU > 0`, one natural mapping is:

```text
gamma_eff(SISU) = gamma_ref / sqrt(SISU)
lambda(SISU) = c * gamma_eff(SISU)^2
             = c * gamma_ref^2 / SISU.
```

Here `gamma_ref` is a row-contract parameter: the smallest feasible gamma that
the SISU-conditioned row is meant to expose, usually written as
`gamma_ref = gamma_factor_ref * gamma_star`. It defines the strongest selected
soft adversary for that row. The corresponding reference penalty is
`lambda_ref = c * gamma_ref^2`.

This mapping gives SISU a direct soft-penalty meaning:

- `SISU = 1` gives the reference gamma and the strongest selected adversary;
- `SISU = 0.25` gives twice the gamma and four times the energy penalty;
- as `SISU` approaches zero, the penalty approaches infinity and the optimizer
  is discouraged from using disturbance energy.

The exact zero row should be special-cased:

```text
if SISU == 0:
    skip adversarial optimization or force delta = 0
else:
    use lambda = c * gamma_ref^2 / SISU
```

This avoids representing infinity numerically and keeps the old behavioral
meaning that `SISU = 0` is the no-adversary condition.

An alternative is log interpolation between two finite gamma factors:

```text
gamma_eff(SISU) =
    gamma_weak^(1 - SISU) * gamma_strong^SISU
```

where `gamma_strong < gamma_weak` because smaller feasible gamma means stronger
soft adversary pressure. This is useful if the experiment needs bounded nonzero
gamma values at every positive SISU level. It is less directly tied to the
inverse-energy-pressure interpretation unless `gamma_weak` and `gamma_strong`
are stated as part of the row contract.

## Recommended Naming

The hard-budget and soft-gamma schedules should use different metadata names.
Suggested names are:

```text
hard-radius schedule:
    budget_schedule = sisu_energy_fraction
    mapping_rule = epsilon_l2_radius = max_l2_radius * sqrt(SISU)

soft-gamma schedule:
    budget_schedule = sisu_gamma_pressure
    mapping_rule = gamma_eff = gamma_ref / sqrt(SISU), SISU > 0
    zero_rule = SISU == 0 disables adversarial perturbation
```

This avoids a false comparison between SISU values across two different game
definitions. A `SISU = 0.5` hard-radius row means half of max disturbance
energy. A `SISU = 0.5` soft-gamma row means twice the reference energy penalty,
or equivalently `sqrt(2) * gamma_ref`, under the inverse-penalty mapping.

## Prior Policy-Adversary Precedent

The prior H0 policy-adversary comparison was tracked under issue
`e901a20`. It used a learned memoryless adversary policy with fixed inner
updates. The important constraint detail is:

- the `plain` policy-adversary row used a hard fixed L2 projection radius;
- the `energy` policy-adversary row used the same hard fixed L2 projection
  radius and added an energy penalty to the adversary objective;
- neither row was a pure soft-gamma adversary without a hard budget.

The fixed radius in those row specs was `0.004545500088363065` in 15 cm task
units, recorded as `active_max_l2_radius_15cm` with source
`effective_020a65b_pgd_training_radius`. The norm was the same per-trial
flattened time-component L2 geometry used by the hard broad-epsilon PGD path,
with no `dt` factor. In the `energy` row, the energy penalty was an additional
objective term after projection; the projection radius remained the feasibility
constraint.

The policy-adversary `energy` row is therefore a hybrid: hard budget first,
soft energy stabilizer second. It is closer to a budget-constrained adversary
than to a true H-infinity soft-constraint adversary. It is relevant precedent
for implementation mechanics, but it does not resolve the gamma-to-soft-PGD
question.

## Practical Experiment Proposal

A conservative implementation sequence is:

1. Add a broad-epsilon PGD objective mode such as `soft_energy_penalty`.
2. Keep the existing hard-radius PGD path unchanged.
3. Add explicit run metadata for `lambda`, `gamma_ref`, `gamma_factor`,
   `gamma_star`, the scale constant `c`, and any numerical safety cap.
4. Start with non-SISU rows so the gamma-soft objective can be diagnosed without
   a controller-visible budget scalar.
5. After the non-SISU behavior is stable, add a separate
   `sisu_gamma_pressure` schedule for SISU-conditioned soft-gamma rows.

The first soft-gamma experiment should be described as an ablation of the
training adversary objective:

```text
hard-radius PGD:
    max_delta task_loss(theta, delta)
    subject to ||delta|| <= r

soft-gamma PGD:
    max_delta task_loss(theta, delta) - lambda ||delta||^2
    with optional safety cap reported as optimizer stabilization
```

The result would answer whether using gamma as an energy price produces a
clearer robustness signal than converting gamma rollouts into hard PGD radii.

## Interpretation Boundaries

The following claims are justified:

- A soft-gamma inner maximization more directly mirrors the H-infinity objective
  than a rollout-derived hard PGD radius.
- Smaller `gamma` means a smaller disturbance-energy price, so it should be a
  stronger adversarial pressure within a fixed soft-penalty training objective.
- SISU can be adapted to soft-gamma training by redefining it as an inverse
  penalty pressure rather than a hard-budget energy fraction.
- The prior policy-adversary energy row was not a pure soft-gamma experiment
  because it retained a hard L2 projection.

The following claims are not justified without additional evidence:

- `lambda = gamma^2` has exactly the same numeric scale in the GRU training
  objective as in the analytical Riccati game;
- a soft-gamma-trained GRU has a certified H-infinity level;
- SISU values have the same quantitative meaning under hard-radius and
  soft-gamma schedules;
- a rollout-derived hard radius is the budget mathematically equivalent to a
  selected gamma.

## Code and Artifact Pointers

Relevant implementation and artifact surfaces:

- `src/rlrmp/train/cs_perturbation_training.py`: broad-epsilon PGD config,
  hard-radius budget schedule, SISU budget schedule, policy-adversary
  projection, and policy-adversary energy penalty;
- `src/rlrmp/train/cs_nominal_gru.py`: training CLI, SISU input selection, and
  planned policy-adversary rows;
- `results/c92ebd8/notes/hinf_gamma_to_pgd_budget_problem.md`: companion note
  on why gamma-derived hard PGD radii are provenance-matched calibrations rather
  than formal equivalences;
- `results/e901a20/runs/h0_policy_adversary__plain.json`: prior hard-budget
  policy-adversary row;
- `results/e901a20/runs/h0_policy_adversary__energy.json`: prior hard-budget
  policy-adversary row with an added energy stabilizer.
