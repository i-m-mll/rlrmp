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
distribution, for example with median, p75, p90, and max. The previous work
centered the sweep on the p90 summary. Keeping p90 would preserve that cautious
aggregation choice, but it still needs to be justified explicitly in the redo
artifact; changing it requires the same level of justification.

Then beta is just a multiplier around the chosen per-trial distribution summary:

```text
lambda = beta^2 * lambda_star_summary
```

Beta greater than 1 means a stronger penalty than the local estimate. Beta
less than 1 means a weaker penalty than the local estimate; it is not forbidden,
but it should be labelled as an instability probe unless evidence justifies it
as a candidate training value.

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
| Summary statistics | justified | Report median, p75, p90, max, and uncertainty. The selected quantile is a scientific choice and must be named. |
| Largest algebraic eigenvalue | justified | Use the largest signed/algebraic Hessian eigenvalue. Do not use spectral radius if that means largest absolute eigenvalue of an indefinite Hessian. |
| Finite-difference validation direction | justified | Validate selected estimates along the estimated top eigenvector. |
| Cap handling | justified as diagnostic only | The hard cap may be reported as a sidecar norm diagnostic, not as a pass/fail criterion. |
| Objective-level sweep fields | justified | Report finite optimizer behavior, nonzero perturbation, positive penalized gain over zero, energy, penalty/gain relation, and norm sidecars. |
| First training beta after audit | partially justified | d55c5f0 names beta 1.4 as a first likely training ratio only after the audit shows it is neither zero nor cap-dominated. |
| Later beta 1.05 and 1.8 rows | partially justified | d55c5f0 says these should wait until beta 1.4 is interpretable. |

## What Is Not Yet Justified

These choices should block implementation until the user confirms or an
artifact with stronger evidence is found.

1. Finite-difference probe step sizes.

   The d55c5f0 spec requires finite-difference probes along the top eigenvector,
   but it does not name the step sizes. Here "radii" would mean small signed
   step sizes along a unit top-eigenvector direction for validating local
   curvature. They would not be a training hard bound. I have not found a
   source that justifies exact values.

2. Whether to include beta below 1 in the first redo sweep.

   Beta below 1 is conceptually allowed as a weaker-than-local penalty and may
   expose instability, but I have not found a reviewer artifact that promotes
   a specific below-1 beta value to a launch-facing candidate.

3. Whether p75 or p90 is the launch-facing quantile.

   The d55c5f0 spec says p75 or p90 can be used and that the choice is
   scientific. The previous work centered on p90, so p90 is the continuity
   default, but I have not found a source that makes p90 mandatory for the
   cap-independent redo.

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
  `lambda_star_i` values;
- validates representative estimates with finite-difference probes along the
  top direction, once probe step sizes are approved;
- writes tracked summaries under `results/31e67ff/` and bulky traces under
  `_artifacts/31e67ff/`.

### 2. Direct-epsilon objective sweep

Redo direct-epsilon sweeps around the chosen per-trial distribution summary of
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

- the chosen quantile and why it was chosen;
- the beta values and whether each is candidate, diagnostic, or instability
  probe;
- the optimizer evidence;
- the cap sidecars as diagnostics only;
- unresolved risks before any smoke or full training run.

## Current Stop Point

I have not found a completed true HVP/Lanczos/power soft-lambda estimate in the
tracked artifacts. I also have not found an artifact that fixes the
finite-difference probe step sizes or chooses p75 versus p90 for the redo.

Implementation should wait for user judgment on those choices, unless a later
archaeology pass finds an already-justified source.
