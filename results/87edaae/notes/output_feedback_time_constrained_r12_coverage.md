# Smooth Time-Basis Output-Feedback Bridge

Issue: `87edaae`. Umbrella: `43e8728`.
Source issue: `7a459bb`.

Scope: Focused r=12 state-coverage follow-up for the smooth spline time-basis output-feedback bridge. Coverage rows test whether state-eigenspectrum or observer-error state coverage changes scratch discovery relative to the no-coverage r=12 baseline.

Non-goals: No trajectory eigenspectrum coverage, broader rank sweep, GRU, linear recurrence, robust training variants, or direct teacher-cloning claims.

Runtime: `2697.46` seconds.

Rank grid: `[12]`.
Retained fit ranks: `[12]`.

## Coverage Follow-Up Verdict

State coverage changed the r=12 scratch solutions substantially, but it did not
produce a standard bridge pass. All 24 standard-certificate rows are full rows,
and the failure-decomposition classifications are `mixed` for all coverage rows
and `optimizer_basin` for the no-coverage/preservation rows.

The best exact-L2 sidecar was the state-eigenspectrum `m=4`, `scale=3` row
(`0.896405` LQR ratio, `lambda/gamma^2=1.25830`), but it also had large gain
error (`1.60809`), worse clean mismatch (`0.0344251`), high final projected
gradient (`148.731`), and very large state-weighted action/Bellman residual
ratios on the standard certificate lens. The small observer-error state row
(`m=1`, `scale=0.3`) was less extreme (`0.924709` exact-L2 ratio,
`lambda/gamma^2=1.62450`) but still classified as `mixed`.

Interpretation: r=12 state coverage can find lower exact-L2 sidecars than the
no-coverage r=12 scratch row, so coverage is not inert. It is not a clean rescue
of scratch discovery, because the improved sidecars come with non-equivalent
gain/action structure and non-converged optimizer diagnostics.

## Projection-Only Representability

| rank | label | projection residual | objective ratio | clean mismatch | exact L2 ratio | lambda/gamma^2 |
|---:|---|---:|---:|---:|---:|---:|
| 12 | spline_r12__projection | 0.016400292 | 1.0001115 | 0.016311589 | 1.000041 | 1.555085 |

## Training Rows

| label | objective ratio | gain rel err | clean mismatch | under-eps ratio | exact L2 ratio | lambda/gamma^2 | iters | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| spline_r12__scratch_lbfgsb_whitened | 1.0372154 | 0.97886565 | 0.19027455 | 1.0905809 | 1.1564461 | 2.0862795 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_lr_0p01 | 1.0087644 | 0.97119125 | 0.034537595 | 1.0635035 | 1.128581 | 1.9532402 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0046942 | 0.96887533 | 0.014332299 | 1.0623097 | 1.1438936 | 2.020065 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.0000276 | 0.35235045 | 0.0022358883 | 1.0354746 | 1.0594093 | 1.7657788 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m1_s0p3_w0p1 | 0.99679969 | 1.339711 | 0.0087128636 | 0.93505406 | 0.94949493 | 1.6617418 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m1_s1_w0p1 | 0.96818643 | 1.464953 | 0.01554636 | 0.90456302 | 0.98147459 | 1.8219678 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m1_s3_w0p1 | 0.93557227 | 1.7403535 | 0.030844748 | 0.90847024 | 4.6456721 | 14.012298 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m4_s0p3_w0p1 | 0.99866845 | 1.5750984 | 0.0080207459 | 0.96541554 | 0.95829548 | 1.4214227 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m4_s1_w0p1 | 0.98000436 | 1.6051631 | 0.017316326 | 0.93476306 | 0.91084464 | 1.2771936 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m4_s3_w0p1 | 0.94566554 | 1.6080924 | 0.034425091 | 0.92142398 | 0.8964052 | 1.2583037 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_observer_error_state_m1_s0p3_w0p1 | 0.99462806 | 1.2012954 | 0.0086452596 | 0.92164424 | 0.9247087 | 1.6244962 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01_observer_error_state_m1_s1_w0p1 | 0.95433857 | 1.4581353 | 0.01495834 | 0.90401037 | 1.0843535 | 2.1304779 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
