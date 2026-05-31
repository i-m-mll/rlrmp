# Smooth Time-Basis Output-Feedback Bridge

Issue: `87edaae`. Umbrella: `43e8728`.
Source issue: `7a459bb`.

Scope: Focused r=20 state-coverage closure for the smooth spline time-basis output-feedback bridge. Coverage rows are restricted to state-eigenspectrum m=4 at scales 1 and 3 plus observer-error state m=1 at scale 0.3, all with weight 0.1.

Non-goals: No broader rank sweep, trajectory eigenspectrum coverage, affine tracker, recurrent controller, GRU, robust training variants, or direct teacher-cloning claims.

Runtime: `1575.56` seconds.

Rank grid: `[20]`.
Retained fit ranks: `[20]`.

## Coverage Closure Verdict

The focused r=20 coverage closure reproduces the r=12 interpretation: coverage
changes the scratch basin and improves disturbance sidecars, but it does not
produce a standard bridge pass. All 16 standard-certificate rows are full rows.
Failure decomposition classified the two no-coverage scratch AdamW rows and the
two preservation-anchor rows as `optimizer_basin`, the no-coverage scratch
L-BFGS-B row as `under_identification`, and all six coverage rows as `mixed`.

The best exact-L2 sidecar is the state-eigenspectrum `m=4`, `scale=3` row
(`0.897207` LQR ratio, `lambda/gamma^2=1.25329`), but it also has large gain
error (`1.63524`), elevated clean mismatch (`0.0277476`), and high final
projected gradient (`162.869`). The state-eigenspectrum `m=4`, `scale=1` row is
similar (`0.909730` exact-L2 ratio, `lambda/gamma^2=1.27592`) with even larger
gain error (`1.72751`). The observer-error state `m=1`, `scale=0.3` row is less
extreme (`0.969593` exact-L2 ratio, `lambda/gamma^2=1.78048`) and still
classified as `mixed` on both lenses.

Interpretation: r=20 state coverage remains diagnostic rather than a promoted
bridge method. It can lower the disturbance sidecars relative to the no-coverage
r=20 scratch row, but the improvement comes with non-equivalent gain/action
structure and non-converged optimizer diagnostics.

## Projection-Only Representability

| rank | label | projection residual | objective ratio | clean mismatch | exact L2 ratio | lambda/gamma^2 |
|---:|---|---:|---:|---:|---:|---:|
| 20 | spline_r20__projection | 0.0050816221 | 1.0000102 | 0.005080895 | 1.0000043 | 1.5551193 |

## Training Rows

| label | objective ratio | gain rel err | clean mismatch | under-eps ratio | exact L2 ratio | lambda/gamma^2 | iters | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| spline_r20__scratch_lbfgsb_whitened | 1.0304336 | 0.98224499 | 0.156001 | 1.085746 | 1.1532355 | 2.0451869 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__scratch_adamw_lr_0p003 | 1.0173432 | 0.9600289 | 0.088976545 | 1.0634614 | 1.1243609 | 1.9431357 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r20__scratch_adamw_lr_0p01 | 1.0069733 | 0.94668997 | 0.032596581 | 1.0616696 | 1.1399774 | 2.0314223 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r20__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0035628 | 0.94540278 | 0.014164998 | 1.065966 | 1.1579516 | 2.0758552 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.0000003 | 0.005098315 | 0.00035033918 | 0.9999903 | 0.99998549 | 1.555106 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m4_s1_w0p1 | 0.97986855 | 1.7275123 | 0.013847832 | 0.93306385 | 0.90973009 | 1.2759221 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__scratch_adamw_then_lbfgsb_lr_0p01_eigen_state_m4_s3_w0p1 | 0.94395392 | 1.6352441 | 0.027747585 | 0.92273325 | 0.89720671 | 1.2532923 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__scratch_adamw_then_lbfgsb_lr_0p01_observer_error_state_m1_s0p3_w0p1 | 0.99544104 | 1.0760016 | 0.0096023926 | 0.94142589 | 0.96959273 | 1.7804796 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
