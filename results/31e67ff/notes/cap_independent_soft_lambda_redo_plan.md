# Cap-Independent Soft-Lambda Redo Plan

Issue: 31e67ff. Parent umbrella: 54389a4.

This is a no-launch planning artifact. It does not approve training, pod
creation, protected-branch auth, or issue closure.

## Plain-Language Goal

Redo the soft-adversary lambda calibration without using the inherited
`0.0045455` hard cap as the rule for choosing lambda. The cap can still be
reported as a diagnostic, meaning "how large was the selected perturbation
relative to an old safety boundary?", but it must not decide whether a lambda
passes.

The main scale should be the distribution of per-trial local curvatures under
the corrected frozen-GRU objective at zero perturbation. In plain terms:
estimate, trial by trial, how sharply the frozen model's loss curves upward
when a small epsilon perturbation is introduced. The reviewer shorthand was:

```text
lambda_star_i = 0.5 * largest algebraic local curvature of J_i(epsilon_i)
                at epsilon_i = 0
```

The implemented scale should then summarize the per-trial `lambda_star_i`
distribution with median, p75, p90, and max. The redo keeps p90 as the primary
continuity summary because the previous corrected-objective work centered its
sweep on p90, and the user confirmed that continuity choice for this redo.

Then beta is just a multiplier around the chosen per-trial distribution summary:

```text
lambda = beta^2 * lambda_star_summary
```

Beta greater than 1 means a stronger penalty than the local estimate. Beta less
than 1 means a weaker penalty than the local estimate. The redo may include
`beta = 0.95` as a user-approved diagnostic curiosity test; it should be
labelled as a weaker-penalty instability probe, not as a launch candidate unless
later evidence supports that.

## Evidence Already Fixed

The cap-lineage audit in
`results/301d1f0/notes/cap_conditioned_lambda_lineage.md` found that the
previous threshold chain is stale for launch planning:

- `results/093d949/` contains useful corrected-objective diagnostics, but its
  curvature estimate used bounded finite directions rather than HVP/power or
  Lanczos.
- `results/1697bdc/` selected "lowest valid" lambda by requiring both useful
  objective gain and interiority relative to the borrowed hard cap.
- `results/f3c5db9/` matched Adam against those cap-conditioned 1697bdc
  references.
- `results/54389a4/NO_LAUNCH_SPEC_LOCK.md` copied those cap-conditioned rows
  into no-launch candidates, so those candidates are stale.

## Archaeology Verdict

I have not found a completed true HVP/Lanczos/power estimate of the corrected
per-trial soft-lambda scale in the tracked artifacts searched so far. The
evidence points to a spec plus finite-direction approximations:

- `results/d55c5f0/notes/soft_constraint_hessian_lambda_star_spec.md` specifies
  the intended HVP/power-or-Lanczos method, per-trial estimates, finite
  difference validation, and median/p75/p90/max summaries.
- `results/093d949/notes/soft_lambda_sweep.md` explicitly says the curvature
  estimate used bounded finite directions, not HVP or power iteration.
- `results/093d949/scripts/materialize_soft_lambda_sweep.py` records
  `hvp_power_curvature` as not used and implements random finite-direction
  plus/minus probes.
- `results/0a46652/scripts/materialize_soft_adversary_audit.py` is older and
  also uses finite directional curvature with a gradient-pressure floor.

The reusable starting points are the frozen-batch loading and per-trial slicing
from `results/093d949/scripts/materialize_soft_lambda_sweep.py`, the closed-loop
audit harness from `results/3b850d6/scripts/materialize_closed_loop_policy_audit.py`,
the search harness from `results/1697bdc/scripts/materialize_critical_lambda_search.py`,
and the Adam reliability harness from
`results/f3c5db9/scripts/materialize_frozen_adam_audit_tuning.py`. The
1697bdc and f3c5db9 criteria must be changed because they currently depend on
cap-interiority.

## What Is Justified

| choice | status | source or reason |
|---|---|---|
| Correct objective | justified | Use `mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]`, with per-trial energy summed over time and epsilon dimensions. |
| Primary lambda scale | justified | Estimate per-trial `lambda_star_i` values from the largest algebraic Hessian eigenvalue of the corrected frozen-GRU objective at zero, then summarize the distribution. |
| HVP/power or Lanczos | justified | The d55c5f0 spec says not to materialize the full Hessian and to use Hessian-vector products with power iteration or Lanczos. |
| Per-trial first | justified | The corrected objective is separable across trials before averaging, so per-trial estimates are preferred first. |
| Summary statistics | justified | Report median, p75, p90, max, and uncertainty. Use p90 as the primary continuity summary for this redo. |
| Largest algebraic eigenvalue | justified | Use the largest signed/algebraic Hessian eigenvalue. Do not use spectral radius if that means largest absolute eigenvalue of an indefinite Hessian. |
| Finite-difference validation direction | justified | Validate selected estimates along the estimated top eigenvector. |
| Cap handling | justified as diagnostic only | The hard cap may be reported as a sidecar norm diagnostic, not as a pass/fail criterion. |
| Objective-level sweep fields | justified | Report finite optimizer behavior, nonzero perturbation, positive penalized gain over zero, energy, penalty/gain relation, and norm sidecars. |
| First training beta after audit | partially justified | d55c5f0 names beta 1.4 as a first likely training ratio only after the audit shows it is neither zero nor cap-dominated. |
| Later beta 1.05 and 1.8 rows | partially justified | d55c5f0 says these should wait until beta 1.4 is interpretable. |
| Beta 0.95 | user-approved diagnostic | Include as a weaker-penalty curiosity test and instability probe, not as a launch candidate by default. |
| Finite-difference validation step grid | user-approved diagnostic | Start with `h = 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5` along the unit top-eigenvector direction. Treat these as numerical validation probes only, not a training bound. |

## Resolved Decisions

The following choices were explicitly resolved by the user after the first
blocked plan:

1. Finite-difference probe step sizes.

   Use a geometric local-validation grid along the unit top-eigenvector
   direction:

   ```text
   h = 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5
   ```

   These are finite-difference validation probes only. They are not hard caps
   and are not a training perturbation budget. If this grid does not show a
   stable curvature plateau, expand one notch smaller or larger and report that
   the first grid was inconclusive.

2. Whether to include beta below 1 in the first redo sweep.

   Include `beta = 0.95` as a diagnostic curiosity test. It probes a slightly
   weaker penalty than the p90 local-curvature scale because
   `0.95^2 = 0.9025`. It should be labelled as a weaker-penalty instability
   probe, not as a launch candidate by default.

3. Whether p75 or p90 is the launch-facing quantile.

   Keep p90 as the primary continuity summary for this redo. Still report
   median, p75, p90, max, and uncertainty so the result shows how conservative
   p90 is relative to the rest of the distribution.

## Proposed Work After User Decision

### 1. Lambda estimate

Implement or reuse a corrected estimator that:

- loads the same frozen no-PGD c92/d55 calibration rows;
- evaluates the corrected per-trial objective;
- uses HVP power iteration or Lanczos to estimate the largest algebraic
  curvature per trial, not the largest absolute curvature if the Hessian is
  indefinite;
- records `lambda_star_i = 0.5 * eigmax_i` under the ordinary Hessian
  convention;
- reports median, p75, p90, max, and uncertainty over the per-trial
  `lambda_star_i` values, using p90 as the primary continuity summary;
- validates representative estimates with finite-difference probes along the
  top direction using `1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5`;
- writes tracked summaries under `results/31e67ff/` and bulky traces under
  `_artifacts/31e67ff/`.

### 2. Direct-epsilon objective sweep

Redo direct-epsilon sweeps around the p90 per-trial distribution summary of
the corrected local estimate. The primary readout should be raw soft-objective
behavior:

- finite optimizer status;
- nonzero selected perturbation where expected;
- positive penalized objective gain over zero;
- task-loss gain, energy, and penalty-to-gain relation;
- selected norm diagnostics, including old-cap ratio as a sidecar only.

No row should pass or fail because it is inside the old `0.0045455` cap.

### 3. Closed-loop linear and affine checks

Reuse the 3b850d6 and 1697bdc harness ideas, but compare linear and affine
closed-loop policies against the corrected reference scale. Keep raw policy
output diagnostics separate from any clipped or cap-ratio diagnostics.

### 4. Adam reliability redo

Redo Adam matching only after the corrected direct and closed-loop references
exist. Do not match Adam against the cap-conditioned 1697bdc rows.

### 5. New no-launch spec lock

Regenerate a new no-launch spec only after the corrected chain is coherent. It
must say plainly that no training is approved yet and must list:

- p90 as the chosen continuity quantile, with median/p75/max sidecars;
- the beta values and whether each is candidate, diagnostic, or instability
  probe, including beta 0.95 as diagnostic only;
- the optimizer evidence;
- the cap sidecars as diagnostics only;
- unresolved risks before any smoke or full training run.

## Current Go Point

I have not found a completed true HVP/Lanczos/power soft-lambda estimate in the
tracked artifacts. The user has now approved the missing choices needed to
start implementation: keep p90, include beta 0.95 as a diagnostic test, and use
the finite-difference plateau grid above as local numerical validation.
