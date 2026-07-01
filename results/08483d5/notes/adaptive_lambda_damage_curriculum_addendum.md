# Adaptive Lambda Damage-Curriculum Addendum

Date: 2026-07-01

## 1. Purpose

This addendum supplements the adaptive-lambda formalism complement for issue
`08483d5`. It records a curriculum-specific methods contract for using
analytical H-infinity damage references without treating low final equilibrium
damage as the whole training-pressure target.

This note is planning material. It is not a launch approval, not a pod request,
not a protected-branch merge request, and not a verdict on the experiment. Any
launch-facing run still needs a locked run spec and explicit user approval.

The central addition is the separation of three quantities:

- analytical equilibrium damage, which is a final reference outcome;
- training adversarial pressure, which is controlled by the soft-adversary
  price, adversary mechanism, optimizer, and outer loss weight;
- learned behavior, which is assessed through damage and phenotype diagnostics
  after or during training.

## 2. Definitions

Let `K_beta` be the analytical output-feedback H-infinity controller at gamma
ratio `beta`, and let `F_beta` be its matched analytical adversary. Let
`J(K, F)` denote the same task-cost functional used in training, including the
same Q/R/Qf terms, state basis, horizon, target convention, nominal noise
convention, masks, and reduction.

The paired analytical equilibrium damage is:

```text
D_beta_eq = E[J(K_beta, F_beta) - J(K_beta, 0)].
```

This measures how much the matched analytical adversary damages the already
solved analytical robust controller. It is an outcome of the solved robust
closed-loop pair, not a direct measure of how much training pressure a learned
controller should receive at every point in training.

For a learned controller `psi`, the soft direct-epsilon adversary selected at a
given lambda is:

```text
epsilon_star(lambda; psi, B) =
    arg max_epsilon [
        J(psi, epsilon; B) - lambda * E(epsilon; B)
    ],
```

where `B` is the evaluation or training batch and `E` is the corrected realized
epsilon energy with the same batch reduction convention as the task loss.

The paired learned damage estimate is:

```text
D_hat(lambda; psi, B) =
    mean_B [J(psi, epsilon_star(lambda); B) - J(psi, 0; B)].
```

The outer adversarial weight is a separate curriculum variable:

```text
L_outer =
    (1 - rho_t) * L_clean
    + rho_t * L_adv(epsilon_star(lambda_t)).
```

`rho_t` applies only to the optimized-epsilon adversarial loss term. It does
not change the randomized perturbation-bank training distribution, and it does
not define the soft adversary's energy price. Randomized perturbation training
can remain enabled on top of this curriculum under its own contract.

## 3. Equilibrium damage is not training pressure

Lower `D_beta_eq` near beta close to 1 can reflect greater achieved robustness
of the analytical controller. Because changing beta changes the controller, the
estimator, the adversary law, the closed-loop trajectory, and the realized
disturbance together, `D_beta_eq` need not be monotone in beta.

Therefore a low final analytical damage target should not automatically be used
from the start of training. A learned controller can make damage low by pricing
the adversary out:

```text
lambda increases -> selected epsilon weakens -> measured damage decreases.
```

That is not the same outcome as an analytical robust controller having low
damage despite a meaningful matched adversary. A launch-facing adaptive row
must therefore diagnose whether low final damage comes with a nonzero,
finite, behaviorally meaningful adversary.

The low beta-1.05 analytical damage values remain useful as final references:

- deterministic beta-1.05 output-feedback damage: about `447.89`;
- paired-noise beta-1.05 output-feedback damage: about `1911.89`.

They should not by themselves define the early training target.

## 4. Damage curriculum

A damage curriculum defines a time-varying reference:

```text
D_target(t).
```

For the first adaptive soft-epsilon curriculum candidate, the current planning
choice is:

```text
D_target(t):
    weak initial challenge
    -> recoverable high challenge around 3500
    -> final lower challenge around 1000.
```

One concrete no-launch planning schedule is:

- continue from the clean 6D no-PGD H0 `const_band16` baseline checkpoint;
- run `7500` additional controller batches;
- ramp the target damage up over the first `2500` batches;
- then cosine-anneal target damage over the next `5000` batches;
- use `D_peak ~= 3500`;
- use `D_final = 1000` for the first test.

A schedule may start with effectively no adversarial outer pressure, but the
lambda update rule should not divide by a zero target. If `D_target(t)` is zero
or below a configured numerical floor, the run spec should either freeze lambda
or use a named positive damage floor for the adaptive controller.

The beta damage curve and its narrow spike region should not be used as an
automatic curriculum generator. The spike near beta `1.342` is a stress-test or
conditioning diagnostic under the implemented deterministic equations, not a
recoverable training target unless a separate scientific reason is supplied.

## 5. Adaptive lambda under a moving target

Lambda controls the adversary's energy price. For direct epsilon, lower lambda
usually allows stronger disturbances, and higher lambda usually weakens the
selected disturbance.

A conservative adaptive rule can use a smoothed damage estimate:

```text
D_ema,k = (1 - alpha) * D_ema,k-1 + alpha * D_hat_k
```

with log-space updates:

```text
if abs(D_ema,k / D_target,k - 1) <= deadband:
    log_lambda_k+1 = log_lambda_k
else:
    log_lambda_k+1 =
        log_lambda_k
        + clip(
            eta * log(D_ema,k / D_target,k),
            -max_log_step,
            max_log_step
        )
```

The intended sign is:

- if measured damage is above target, increase lambda to make disturbances more
  expensive;
- if measured damage is below target, decrease lambda to allow a stronger
  adversary.

The first conservative frozen-replay settings were:

- update cadence: every `50` controller batches;
- evaluation batch size: `64`;
- replicate aggregation: all `5` replicates;
- EMA alpha: `0.1`;
- eta: `0.1`;
- max log-lambda step: `0.1`;
- deadband: `10%`.

A run spec should define `lambda_min`, `lambda_max`, and the action taken when
these bounds are reached. These bounds are adaptive-controller bounds, not old
safety caps, projection radii, or trust-region defaults. Historical cap/radius
values must not enter this row as active controls.

## 6. Outer adversarial-weight curriculum

The outer adversarial weight `rho_t` controls how much the optimized-epsilon
loss affects controller updates. It is independent of `D_target(t)` and
`lambda_t`, even when the first row chooses matching time windows for practical
stability.

For the first candidate row:

```text
rho_t:
    0 -> 1 over the first 2500 batches,
    then stay at 1 during damage annealing.
```

This means the adversary can be selected and monitored while its contribution
to controller gradients ramps in gradually. Once the model reaches the high
challenge phase, all optimized-epsilon trials can contribute at full
adversarial weight while the damage target anneals downward.

The schedule should be parameterized independently:

- `rho_start`;
- `rho_final`;
- `rho_ramp_batches`;
- `rho_schedule_shape`;
- whether rho is held, annealed, or frozen after the ramp.

Using `rho_t` is a stability curriculum. It should not be described as a hidden
baseline task change. It also should not be applied to the randomized
perturbation bank.

## 7. Candidate method families

| Family | Control variable | Intended use | Main risk | Current role |
|---|---|---|---|---|
| Fixed final-damage servo | `lambda_t` tracks `D_beta_eq` from batch 0 | Simple adaptive target | Can make the adversary disappear instead of training robustness | Not preferred as first launch row |
| Damage-only curriculum | `lambda_t` tracks `D_target(t)` | Tests whether a moving damage target can train robustness | May destabilize early training if every adversarial trial has full weight | Useful ablation after hybrid row |
| Hybrid damage plus outer-weight curriculum | `lambda_t` tracks `D_target(t)`, `rho_t` ramps outer adversarial contribution | Separates adversary strength from how much adversarial examples drive controller updates | More moving parts; needs clear logging | Preferred first curriculum candidate |
| Lambda schedule plus damage monitoring | Preplanned `lambda_t`, damage is diagnostic | Keeps training pressure more directly tied to price | Needs a reliable lambda scale and may miss target damage | Possible comparator or fallback |
| Phenotype matching | Match behavior sidecars such as peak velocity, endpoint quality, recovery, or perturbation response | Targets the final behavior more directly | Can overfit phenotype while losing soft-game interpretation | Separate family or second-stage correction |
| Beta-indexed curriculum | Schedule beta and map to damage or lambda references | Ties row labels to analytical references | Beta damage is nonmonotone and has a spike region | Use cautiously as provenance, not a default knob |
| Fixed-lambda comparator | Hold lambda fixed from a cap-independent source | Baseline for interpreting adaptive rows | May go inactive late | Required comparator/control |

## 8. Phenotype matching

Phenotype matching replaces or augments scalar damage matching with a vector of
behavioral targets:

```text
phi_hat(psi) ~= phi_ref(beta).
```

Candidate phenotype components include:

- peak velocity;
- time to peak velocity;
- forward-velocity RMSE;
- endpoint error;
- terminal velocity;
- perturbation recovery shape;
- clean/adversarial trajectory-shape distance;
- selected adversary energy and zero fraction as sidecars.

The phenotype objective should be explicit:

```text
M_phi =
    sum_j w_j * normalize_j(phi_hat_j - phi_ref_j)^2.
```

Phenotype matching can be useful if damage matching trains the wrong behavior
or if multiple lambda settings produce similar damage with different movement
phenotypes. It should be labeled as phenotype-calibrated adaptive training, not
as proof that the trained GRU implements the analytical H-infinity controller.

For the first launch-facing row, phenotype quantities are better treated as
post-run diagnostics and possible second-stage corrections, not as the primary
adaptive controller.

## 9. Pass/fail interpretation

A hybrid damage-curriculum row should be considered interpretable only if these
conditions are checked under predeclared tolerances:

- selected epsilon remains nonzero after the initial ramp and late in training;
- damage follows the scheduled target without persistent lambda runaway;
- final low damage is not achieved by lambda exceeding the empirical
  `lambda_zero` region or by making the adversary inactive;
- no safety cap, inherited radius, projection radius, or trust-region value
  sets the selected disturbance scale;
- nominal task performance remains comparable to the baseline contract;
- adversarial damage remains recoverable rather than entering the spike-like
  blow-up regime;
- outer-weight effects are logged separately from lambda effects;
- randomized perturbation-bank effects are logged separately from optimized
  epsilon effects.

The row should be bracketed or failed if:

- the adversary goes zero for most late checkpoints;
- the only way to hit `D_target(t)` is to use a guard or hidden bound;
- the damage target is met while the learned movement phenotype is plainly
  inconsistent with the intended robust-control behavior;
- lambda oscillation or clipping dominates the adaptive state;
- the result cannot separate the effect of the damage curriculum from the
  effect of the outer adversarial-weight curriculum.

## 10. First run-spec implications

The first launch-facing adaptive curriculum spec should update the older
`adaptive_pn_b1p05` lock. The prior lock used paired-noise beta-1.05 damage as
the controller-driving target. Under this addendum, that value is better
treated as a reference sidecar rather than the from-start training target.

The first updated row should instead specify:

- baseline checkpoint: clean 6D no-PGD H0 `const_band16`;
- mechanism: cap-free `direct_epsilon`;
- total adaptive phase length: `7500` batches;
- damage schedule: ramp to about `3500` over `2500` batches, then cosine anneal
  to `1000` over `5000` batches;
- outer adversarial weight: ramp from `0` to `1` over a separately configured
  interval, initially matching the `2500`-batch damage ramp, then hold at `1`;
- lambda update: conservative EMA/deadband/log-space controller unless a new
  frozen replay motivates different values;
- forbidden active controls: no safety cap, inherited radius, projection
  radius, trust region, or hard energy budget;
- diagnostics: beta-1.05 and beta-1.4 analytical damage references, phenotype
  sidecars, zero fraction, epsilon energy, lambda trajectory, outer-weight
  trajectory, and clean/adversarial task decompositions.

No numerical target in this addendum should be treated as launch authorization.
The run spec must still lock the exact implementation branch, command, resource
plan, post-run diagnostics, stop criteria, and user approval gate.
