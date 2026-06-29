# Cap-Conditioned Lambda Lineage Audit

Issue: 301d1f0. Parent umbrella: 54389a4.

## Verdict

**User is correct.** The recent frozen-audit lambda refinement should not be
used as a launch-facing minimum-lambda calibration without redo. The initial
per-trial p90 lambda estimates in 093d949 are useful diagnostic estimates, but
the downstream "minimum lambda", "critical lambda", Adam-match, and no-launch
training-candidate values were selected by a finite-radius cap/interiority
criterion. That cap was the borrowed
`ofb_6d_no_integrator_gamma_1p4_rollout_radius` value, approximately
`0.0045455`, from the c92/output-feedback rollout-budget lineage. It was
documented as a safety or diagnostic trust radius, not as a newly derived
scientific soft-lambda boundary.

The contaminated claim is not "the code cannot optimize a corrected
soft-energy objective." The contaminated claim is narrower and important:
the specific selected multipliers in 1697bdc, f3c5db9, and
results/54389a4/NO_LAUNCH_SPEC_LOCK.md are cap-conditioned thresholds. They
answer "where does the optimized perturbation become useful while remaining
inside this borrowed 0.0045455 cap?" rather than "what lambda should a
soft-adversary training row use without a hard boundary?"

## Evidence Read

I inspected Mandible reports for 54389a4, 093d949, 1697bdc, f3c5db9, 27dece3,
301d1f0, 06a4dc8, 3c5836c, b413bb0, 0a46652, d55c5f0, 020a65b, c92ebd8, and
the linked closed-loop reference 3b850d6. I also inspected the tracked result
notes, JSON/CSV outputs, and materializer scripts under results/093d949,
results/3b850d6, results/1697bdc, results/f3c5db9, results/27dece3,
results/54389a4, results/c92ebd8, and results/d55c5f0.

Key source files:

- `results/d55c5f0/notes/soft_constraint_hessian_lambda_star_spec.md`
- `results/093d949/notes/soft_lambda_sweep.md`
- `results/093d949/scripts/materialize_soft_lambda_sweep.py`
- `results/3b850d6/notes/closed_loop_policy_audit.md`
- `results/3b850d6/scripts/materialize_closed_loop_policy_audit.py`
- `results/1697bdc/notes/critical_lambda_search.md`
- `results/1697bdc/scripts/materialize_critical_lambda_search.py`
- `results/f3c5db9/notes/frozen_adam_audit_tuning.md`
- `results/f3c5db9/scripts/materialize_frozen_adam_audit_tuning.py`
- `results/54389a4/NO_LAUNCH_SPEC_LOCK.md`
- `results/54389a4/scale_sanity_summary.json`
- `src/rlrmp/train/cs_perturbation_training.py`

## Answers

### 1. Initial lambda estimate

The original reviewer/planning direction was Hessian/curvature based. The
d55c5f0 spec defines the intended frozen-GRU local soft-game boundary as

```text
lambda_star_GRU = 0.5 * largest_eigenvalue(d2 J_GRU(epsilon=0) / d epsilon^2)
```

under the corrected per-trial mean soft objective. It recommends HVP/power or
Lanczos estimation, per-trial estimates first, finite-difference validation,
and median/p75/p90/max summaries. It explicitly says the selected quantile is a
scientific choice.

The implemented 093d949 child did a bounded approximation, not a full HVP power
estimate. It reported:

- old batch-mean estimator;
- batch-corrected comparison estimator;
- per-trial p90 estimator;
- finite-direction curvature plus a gradient-pressure floor.

The actual scalar center used for the direct sweep was
`per_trial_p90.lambda_beta`, not the batch-corrected comparison value. The
reported per-trial p90 beta lambdas were:

| row | per-trial p90 beta lambda |
|---|---:|
| open_loop_small | 1.11397e+08 |
| open_loop_moderate | 1.02932e+08 |
| open_loop_stress | 1.35806e+08 |

Those values remain useful as local frozen-model scale diagnostics, with the
caveat that they used finite directional curvature rather than the originally
recommended HVP/power curvature pass.

### 2. Intended honing method

Yes. The implemented honing explicitly used finite-radius sweeps and then
binary-search/bracket logic around a cap-to-interior transition.

In 093d949, the direct-epsilon sweep was centered on the per-trial p90 estimate
and interpreted by cap behavior. The note says the center landed cap-dominated
and the grid bracketed the transition above it:

| row | last cap-dominated multiplier | first interior multiplier |
|---|---:|---:|
| open_loop_small | 2.0 | 4.0 |
| open_loop_moderate | 2.0 | 4.0 |
| open_loop_stress | 1.0 | 2.0 |

In 1697bdc, the definition became explicit: practical `lambda_crit` was the
smallest tested multiplier where the optimized adversary was both "interior"
and "useful". Interior was defined as `cap_bound_fraction = 0.0` and
`max_norm_over_cap <= 0.99`. Bisection then searched that valid/invalid
boundary.

### 3. Hard-cap provenance

The hard cap was `0.004545500088363065`, surfaced in the training module as
`EFFECTIVE_020A65B_PGD_RADIUS_15CM` and in
`PGD_SISU_MAX_RADIUS_SOURCES["ofb_6d_no_integrator_gamma_1p4_rollout_radius"]`.
The metadata describes it as a 6D no-integrator C&S output-feedback H-infinity
rollout L2 radius for `gamma_factor=1.4`, with source issue c92ebd8. The older
020/c92 archaeology says this radius was an output-feedback/estimator-in-loop
budget, not the raw full-state gamma 1.4 radius and not a new lambda-derived
training scale.

The planning intent was diagnostic/safety use. The d55c5f0 spec says the trust
radius should remain a safety cap only and never define the scientific budget.
The c92 soft-energy implementation comments likewise described the safety cap
as stabilization only. The later threshold searches did not keep that
separation: they used interiority relative to this cap as a validity criterion.

### 4. Minimum lambda values selected by cap interiority

Yes. The selected 1697bdc critical-lambda values, the f3c5db9 Adam matches, and
the 54389a4 no-launch training candidates are cap-conditioned. The 1697bdc
reference values used downstream were:

| row | mechanism | reference optimizer | selected multiplier | selected lambda | status |
|---|---|---|---:|---:|---|
| open_loop_small | direct_epsilon | pgd_projected_epsilon | 2.82843 | 3.15077e+08 | cap-conditioned |
| open_loop_small | linear_no_bias | lbfgsb | 2.0 | 2.22793e+08 | cap-conditioned |
| open_loop_small | affine | lbfgsb | 2.18102 | 2.42958e+08 | cap-conditioned |
| open_loop_moderate | direct_epsilon | pgd_projected_epsilon | 3.08442 | 3.17484e+08 | cap-conditioned |
| open_loop_moderate | linear_no_bias | lbfgsb | 2.0 | 2.05863e+08 | cap-conditioned |
| open_loop_moderate | affine | lbfgsb | 2.18102 | 2.24495e+08 | cap-conditioned |
| open_loop_stress | direct_epsilon | pgd_projected_epsilon | 2.18102 | 2.96194e+08 | cap-conditioned |
| open_loop_stress | linear_no_bias | lbfgsb | 1.41421 | 1.92058e+08 | cap-conditioned |
| open_loop_stress | affine | lbfgsb | 1.68179 | 2.28397e+08 | cap-conditioned |

f3c5db9 then defined an Adam match as finding a finite, useful, interior point
at the 1697bdc reference lambda multiplier. Therefore the optimizer setting
recommendation is valid only relative to those cap-conditioned reference
regions.

The 54389a4 no-launch spec copied these values into proposed rows:

- `direct_epsilon_calibrated`: small 2.83x, moderate 3.08x, stress 2.18x.
- `linear_no_bias_calibrated`: small 2x, moderate 2x, stress 1.41x.
- `affine_calibrated`: small 2.18x, moderate 2.18x, stress 1.68x.

Those proposed rows should be marked stale/pivot-needed until redone.

### 5. What remains valid without the cap

Several pieces remain useful:

- The corrected soft objective semantics are still the right target:
  `mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]`.
- The batch-reduction diagnosis remains valid: the old d55/c92 soft rows
  effectively over-penalized epsilon by the batch size.
- The 093d949 per-trial p90 estimates remain a bounded local frozen-model
  scale diagnostic, but not a final training scale.
- The 3b850d6 raw closed-loop policy audit remains evidence that nonzero finite
  policy directions can improve the raw objective; it already separated raw
  policy output from selected/clipped cap diagnostics.
- The 1697bdc and f3c5db9 scripts are useful test harnesses for objective
  evaluation, finite optimizer plumbing, and table rendering.
- The relative ordering of some lambda regions may remain suggestive, but it is
  only suggestive because the selection target mixed objective usefulness with
  a borrowed cap.

What does not remain valid is any launch-facing claim that the listed
multipliers are the minimum scientific lambdas for soft-adversary training.

### 6. Redo needed

The redo should be no-launch until the user explicitly approves training.

Minimum redo:

1. Recompute or strengthen the initial lambda estimate using the intended
   Hessian/HVP contract: per-trial corrected objective, HVP power/Lanczos or a
   clearly bounded approximation, finite-difference validation along top
   directions, and median/p75/p90/max summaries.
2. Separate unbounded/raw soft objective evaluation from diagnostic trust-region
   reporting. The primary selection criterion should be objective-level:
   finite optimizer behavior, positive penalized objective gain, predicted and
   realized energy, and nonzero selected epsilon where expected. Cap-bound
   status should be a sidecar, not the pass/fail gate.
3. Direct-epsilon redo: sweep lambda across the p90/HVP-derived region without
   defining validity by `0.0045455` interiority. Report raw objective gain,
   energy, penalty-to-gain ratio, selected norm distribution, and any numerical
   trust-region hits separately.
4. Closed-loop redo: rerun finite linear no-bias and affine policy brackets
   with the same objective-level criterion. Keep raw policy output diagnostics
   distinct from clipped/selected cap diagnostics.
5. Adam reliability redo: match Adam against the redone objective-level
   reference rows, not against the 1697bdc cap-conditioned rows. The current
   Adam grid can be reused as a harness.
6. No-launch spec redo: regenerate the scale sanity summary and
   NO_LAUNCH_SPEC_LOCK only after the above references are available. The spec
   should explicitly state whether any safety cap is merely numerical,
   diagnostic, or newly scientifically justified.
7. Tests/artifacts: preserve or add focused tests for batch-size invariance,
   soft objective reduction, cap-diagnostic-only metadata, objective-level
   reference-row selection, and Adam matching against the new reference rows.

## Affected Analyses

| issue/artifact | intended role | cap/interiority usage | dependency on 0.0045455 | conclusion status | redo needed |
|---|---|---|---|---|---|
| d55c5f0 / `soft_constraint_hessian_lambda_star_spec.md` | Reviewer/planning theory for corrected soft lambda scale | Says trust radius is safety only, never scientific budget | Mentions safety caps conceptually, not as selected threshold | Valid planning source | Use as redo contract; implement HVP/curvature estimate more faithfully |
| 093d949 / `soft_lambda_sweep.*` | Estimate old, batch-corrected, and per-trial p90 lambda; direct finite-radius sweep | Sweep read is explicitly cap-to-interior; p90 center landed cap-dominated | Uses frozen contract cap from `ofb_6d_no_integrator_gamma_1p4_rollout_radius` | Estimator diagnostic valid; transition claims cap-conditioned | Recompute strengthened estimate and direct sweep with cap as sidecar |
| 3b850d6 / `closed_loop_policy_audit.*` | Show nonzero finite closed-loop directions exist; raw-vs-clipped policy audit | Lambda choices use 093d949 transition; cap diagnostics reported after raw policy output | Hard-coded cap source and radius in script and JSON | Raw expressivity evidence mostly valid; lambda choices cap-influenced | Reuse harness, choose lambda grid from redone objective-level estimate |
| 1697bdc / `critical_lambda_search.*` | Practical frozen-audit lambda thresholds for direct, linear, affine mechanisms | Defines valid as useful plus interior; bisection searches cap-to-interior boundary | Inherits center/cap from 093d949 and frozen radius | Critical/minimum lambda conclusions contaminated | Redo thresholds with objective-level finite/useful criterion and cap sidecar |
| f3c5db9 / `frozen_adam_audit_tuning.*` | Tune Adam to match frozen-audit reference regions | Match means finite/useful/interior at 1697bdc reference lambda | Depends on 1697bdc cap-conditioned references | Adam implementation evidence valid; training-facing setting recommendation stale | Redo Adam match after new references exist |
| 27dece3 / `materialize_scale_spec_lock.py` | Materialize no-launch scale sanity and spec lock | Renders "lowest valid" as finite/useful/interior | Reads cap radius from 093d949 frozen contract and copies selected rows | Materializer logic needs criteria change | Update to render objective-level criteria and cap sidecar |
| 54389a4 / `NO_LAUNCH_SPEC_LOCK.md`, `scale_sanity_summary.json` | User-facing no-launch approval packet and proposed training rows | Proposed rows are selected because finite/useful/interior | States safety cap radius is 0.0045455 from OFB gamma 1.4 rollout radius | Launch-facing lambda recommendations contaminated | Regenerate after redo; mark current packet stale for lambda approval |
| c92ebd8 / output-feedback budget lineage | Source of OFB rollout budget and 0.0045455 cap | Original role was PGD/output-feedback budget provenance, not soft-lambda threshold | Direct source of cap metadata | Valid as budget archaeology, not as new soft training scale | Do not use as threshold gate unless a new scientific cap is justified |
| 020a65b / PGD budget clarification | Older H0 PGD budget provenance clarification | Clarifies active radius provenance | Shows 0.0045455 was effective OFB/estimator-in-loop radius | Valid provenance | None for soft lambda except citation/provenance separation |

## Bottom Line

The redo should not start from the 54389a4 candidate rows as approved lambda
scales. It should start from the d55c5f0 contract, keep the corrected objective,
recompute or strengthen the frozen-GRU lambda estimate, and then evaluate
direct/closed-loop/Adam behavior with cap diagnostics reported but not used as
the primary validity boundary.
