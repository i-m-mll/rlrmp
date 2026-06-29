# Cap-Independent Soft-Lambda No-Launch Spec Lock

Issue: `31e67ff`. Parent umbrella: `54389a4`.

> No training launch is approved by this artifact. This is a corrected
> inner-adversary spec lock for review and possible later smoke planning. A
> billable or non-smoke run still requires a separate explicit user-approved
> run spec with training length, seeds, hardware, and launch command.

## Evidence Used

- HVP/Lanczos lambda source: `results/06a4dc8/canonical_soft_lambda_hvp.json`
  and `results/06a4dc8/notes/canonical_soft_lambda_hvp.md`.
- Direct-epsilon corrected reference:
  `results/7180984/direct_epsilon_soft_lambda_redo.json` and
  `results/7180984/notes/direct_epsilon_soft_lambda_redo.md`.
- Closed-loop corrected reference:
  `results/6cfa892/closed_loop_soft_lambda_redo.json` and
  `results/6cfa892/notes/closed_loop_soft_lambda_redo.md`.
- Adam reliability redo: `results/d469108/adam_soft_lambda_redo.json` and
  `results/d469108/notes/adam_soft_lambda_redo.md`.

## Corrected Scale Contract

- The lambda scale is per trial under the corrected per-trial soft objective.
- For each trial, use the largest algebraic Hessian eigenvalue estimated by
  HVP-backed Lanczos, then compute `lambda_star_i = 0.5 * eigmax_i`.
- The launch-facing continuity summary remains the per-substrate p90 of
  `lambda_star_i`.
- Beta means `lambda(beta) = beta^2 * lambda_star_p90`.
- Beta `0.95` is diagnostic only: it probes a weaker-than-p90 penalty and must
  not be treated as a recommended training scale without a later decision.
- Old hard-cap ratios are sidecars only. They are not criteria for lambda
  choice, optimizer success, or launch readiness.

## Corrected Lambda Values

| substrate | lambda p90 | beta 0.95 diagnostic | beta 1.05 | beta 1.2 | beta 1.4 | beta 1.8 sidecar |
|---|---:|---:|---:|---:|---:|---:|
| `open_loop_small` | 2.55916e+08 | 2.30965e+08 | 2.82148e+08 | 3.68520e+08 | 5.01596e+08 | 8.29169e+08 |
| `open_loop_moderate` | 2.30908e+08 | 2.08394e+08 | 2.54576e+08 | 3.32507e+08 | 4.52579e+08 | 7.48141e+08 |
| `open_loop_stress` | 2.12539e+08 | 1.91816e+08 | 2.34324e+08 | 3.06056e+08 | 4.16576e+08 | 6.88626e+08 |

## Candidate No-Launch Smoke Grid

The corrected first-pass smoke grid is beta `1.05`, `1.2`, and `1.4` for each
mechanism below, with beta `0.95` allowed only as a diagnostic add-on because
the user asked to see it. Direct-epsilon beta `1.8` is not in the candidate
grid: the corrected direct-epsilon reference selects zero/no-positive-gain at
that beta, while the Adam redo finds small positive nonzero solutions, so it is
a mismatch to investigate rather than a launch-facing row.

| proposed row | mechanism | beta values | inner optimizer setting | evidence summary | status |
|---|---|---|---|---|---|
| `direct_epsilon_hvp_p90_beta_sweep` | direct epsilon | diagnostic `0.95`; candidates `1.05`, `1.2`, `1.4` | zero-start Adam, `steps=8`, `lr=1e-5` for candidate rows | corrected direct reference is finite and positive for beta `1.05`-`1.4`; Adam matches those corrected classifications | candidate_no_launch_smoke |
| `linear_no_bias_hvp_p90_beta_sweep` | closed-loop linear no-bias | diagnostic `0.95`; candidates `1.05`, `1.2`, `1.4` | zero-start Adam, `steps=8`, `lr=1e-5` for candidate rows | corrected closed-loop reference is finite and positive; Adam matches candidate classifications | candidate_no_launch_smoke |
| `affine_hvp_p90_beta_sweep` | closed-loop affine | diagnostic `0.95`; candidates `1.05`, `1.2`, `1.4` | zero-start Adam, `steps=8`, `lr=1e-5` for candidate rows | corrected closed-loop reference is finite and positive; Adam matches candidate classifications | candidate_no_launch_smoke |

The `steps=8`, `lr=1e-5` setting is not a new theoretical criterion. It is the
representative matching Adam setting selected by the corrected Adam redo for
all candidate beta `1.05`-`1.4` groups across the three c92 substrates and three
mechanisms.

## What This Replaces

The older launch-facing rows in `results/54389a4/NO_LAUNCH_SPEC_LOCK.md` are
stale for launch planning because they were conditioned on the inherited
`0.0045455` cap boundary. This lock replaces the lambda scale, beta grid, and
Adam matching basis with the corrected HVP/p90 chain above.

## Still Not Decided

- Whether a later smoke should include the diagnostic beta `0.95` rows or keep
  them as audit-only outputs.
- Whether closed-loop beta `1.8` deserves a separate stress probe. It is
  positive in the closed-loop reference, but it is omitted from the unified
  first-pass grid because direct epsilon beta `1.8` has the corrected-reference
  mismatch described above.
- Training-run details: `n_batches`, warmup/adversary schedule, seeds,
  replicate count, GPU/cloud, output paths, and exact launch command.

## Launch Gate

This artifact does not authorize training. Before any launch, create a separate
run spec table that names the selected rows, training schedule, seeds, hardware,
expected artifact paths, monitor plan, and stop criteria, then get explicit user
approval in the current conversation.
