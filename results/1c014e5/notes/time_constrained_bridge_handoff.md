# Handoff: Time-Constrained Output-Feedback Bridge

This handoff starts from integration branch `integration/7a459bb-certificate-results`.
The optimizer-basin experiment itself is commit `8d7c29a`, and the next-stage
planning issue is `87edaae`.

## Relevant Issues

- `43e8728` - umbrella for the cs2019-to-RNN game-equivalence plan.
- `7a459bb` - free time-varying output-feedback rollout recovery.
- `1c014e5` - Adam/AdamW optimizer-family bridge for Phase 3 rollout recovery.
- `7cea1b7` - K-alpha interpolated-start basin diagnostic.
- `d01c35a` - standard bridge certificate.
- `c45adde` - standard failure-decomposition companion.
- `87edaae` - next-stage time-constrained output-feedback bridge.

## Current Verdict

The free time-varying output-feedback controller is not representationally
blocked. The successful row is the Bellman-initialized preservation row:
`adamw_bellman_polish_lr_0p01__bellman_init`.

That row passes the practical bridge target:

| metric | value |
|---|---:|
| objective ratio | `1.0000000012` |
| clean action mismatch | `6.73e-06` |
| exact-L2 ratio | `0.9999916` |
| lambda/gamma^2 | `1.5551161` |
| gain relative error | `1.33e-04` |

This does not mean from-scratch training is solved. From scratch remains the
real discovery test. Best scratch AdamW still fails:

| metric | value |
|---|---:|
| objective ratio | `1.0079189` |
| clean action mismatch | `0.0306647` |
| exact-L2 ratio | `1.1616916` |
| lambda/gamma^2 | `2.0434034` |

K-alpha starts are diagnostic, not a bridge solution. The best K-alpha L-BFGS
row at `alpha=0.75` nearly passes, but only from a start already close to the
analytical gain and with optimizer non-convergence at the cap:

| metric | value |
|---|---:|
| objective ratio | `1.0000684` |
| clean action mismatch | `5.43e-04` |
| exact-L2 ratio | `1.0096479` |
| lambda/gamma^2 | `1.5923224` |

## Interpretation

Do not describe the next step as GRU. The planned sequence is:

`free time-varying -> time-constrained optimization -> linear recurrent -> GRU`.

The free stage says:

- preservation from a structured Bellman start works;
- scratch discovery under the clean rollout objective still fails;
- high-alpha interpolation suggests the reference basin is narrow;
- the failure is not simple gain-class representability.

Therefore the time-constrained bridge should ask whether adding the planned
time/structure constraint improves scratch discovery while preserving the
same certified game.

## What To Run Next

Use issue `87edaae` as the implementation issue. Start from integration branch
`integration/7a459bb-certificate-results` unless a newer integration branch has
superseded it.

Minimum matrix:

1. Time-constrained output-feedback controller from scratch.
2. Time-constrained output-feedback controller from the structured/Bellman
   sanity start, if that initialization is meaningful for the constrained
   parameterization.
3. AdamW-family optimizer path only where it is part of the planned training
   method or a necessary optimizer-family diagnostic.
4. L-BFGS or AdamW-to-L-BFGS polish as a numerical aid, but keep it labeled.

The primary success claim must come from the from-scratch row. Bellman init is
a sanity anchor: it checks preservation and implementation correctness, not
practical discovery.

## Required Analyses

Every retained row must get the standard certificate from `d01c35a` and the
failure decomposition from `c45adde`. This is not optional or a new decision.

At minimum, keep:

- nominal-clean and Riccati-epsilon evaluation lenses;
- state-weighted action mismatch;
- closed-loop transition mismatch;
- value-policy gap;
- Bellman-Hessian weighted residual;
- visited-subspace diagnostics;
- exact-L2 and finite-gamma sidecar metrics;
- optimizer metadata;
- failure classification and gain-error subspace decomposition.

Keep raw controller and rollout arrays under `_artifacts/<issue>/...` so the
certificate and failure decomposition can be rerun without retraining.

## Do Not Do

- Do not jump to GRU from the free time-varying result.
- Do not treat Bellman-initialized success as scratch training success.
- Do not gate the next phase on recovering the analytical gain alone; use the
  standard certificate and failure decomposition.
- Do not introduce new coverage/noise sweeps as a substitute for the planned
  architectural bridge.
- Do not use direct Riccati teacher cloning or action supervision as the primary
  bridge claim.

## Useful Existing Outputs

- `results/1c014e5/notes/output_feedback_optimizer_basin.md`
- `results/1c014e5/notes/output_feedback_optimizer_basin_manifest.json`
- `_artifacts/1c014e5/output_feedback_optimizer_basin/output_feedback_optimizer_basin.npz`
- `results/7a459bb/notes/output_feedback_sweep_standard_certificates.md`
- `results/7a459bb/notes/output_feedback_failure_decomposition.md`

## Suggested First Agent Actions

1. Read issues `87edaae`, `7a459bb`, `1c014e5`, `7cea1b7`, `d01c35a`, and
   `c45adde`.
2. Inspect the current output-feedback recovery materializers before designing
   new code.
3. Identify the existing or intended time-constrained parameterization in the
   codebase. If none exists, implement it as a narrow extension with its own
   materializer and tests.
4. Run a small smoke that writes to `/tmp` before the full materialization.
5. Materialize tracked markdown and manifest outputs under `results/87edaae/`
   and bulk arrays under `_artifacts/87edaae/`.
6. Post a compact verdict back to `87edaae`, `7a459bb`, and `43e8728`.
