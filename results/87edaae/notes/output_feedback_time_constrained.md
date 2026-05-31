# Smooth Time-Basis Output-Feedback Bridge

Issue: `87edaae`. Umbrella: `43e8728`.
Source issue: `7a459bb`.

Scope: Smooth spline time-basis output-feedback bridge. Projection rows check representability; scratch rows test discovery; Bellman-projected rows are preservation anchors only.

Non-goals: No GRU, linear recurrence, coverage/noise sweeps, robust training variants, or direct teacher-cloning claims.

Runtime: `2200.69` seconds.

Rank grid: `[3, 5, 8, 12, 20, 60]`.
Retained fit ranks: `[3, 5, 8, 12, 20, 60]`.

## Projection-Only Representability

| rank | label | projection residual | objective ratio | clean mismatch | exact L2 ratio | lambda/gamma^2 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | spline_r3__projection | 0.10564412 | 1.0483376 | 0.24478979 | 1.1037202 | 1.5660355 |
| 5 | spline_r5__projection | 0.074452984 | 1.0720537 | 0.30527175 | 0.99735208 | 1.5539947 |
| 8 | spline_r8__projection | 0.041844579 | 1.0058504 | 0.10973524 | 0.99950956 | 1.5544568 |
| 12 | spline_r12__projection | 0.016400292 | 1.0001115 | 0.016311589 | 1.000041 | 1.555085 |
| 20 | spline_r20__projection | 0.0050816221 | 1.0000102 | 0.005080895 | 1.0000043 | 1.5551193 |
| 60 | spline_r60__projection | 1.2305548e-15 | 1 | 3.9057289e-15 | 1 | 1.5551183 |

## Training Rows

| label | objective ratio | gain rel err | clean mismatch | under-eps ratio | exact L2 ratio | lambda/gamma^2 | iters | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| spline_r3__scratch_lbfgsb_whitened | 1.0309586 | 0.93062127 | 0.13757623 | 0.99860077 | 0.9973058 | 1.6318721 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r3__scratch_adamw_lr_0p003 | 1.0129411 | 0.88683645 | 0.026209956 | 1.0350297 | 1.0601797 | 1.7518834 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r3__scratch_adamw_lr_0p01 | 1.0056436 | 0.81787972 | 0.016417567 | 1.0303776 | 1.0649185 | 1.8298974 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r3__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0026158 | 0.80256697 | 0.015301109 | 1.0343856 | 1.0755536 | 1.8727064 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r3__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.00007 | 0.26119327 | 0.0043825852 | 1.0189993 | 1.0306687 | 1.6507433 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r5__scratch_lbfgsb_whitened | 1.0140868 | 0.93224193 | 0.031119178 | 1.0384654 | 1.0640014 | 1.8860853 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r5__scratch_adamw_lr_0p003 | 1.0140097 | 0.95410293 | 0.040419544 | 1.0022236 | 1.0331478 | 1.8345909 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r5__scratch_adamw_lr_0p01 | 1.0083123 | 0.96201207 | 0.023400683 | 1.0361593 | 1.0809184 | 1.819361 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r5__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0042966 | 0.95287966 | 0.021349772 | 1.0335303 | 1.0694082 | 1.8091821 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r5__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.0000504 | 0.38485815 | 0.0022331271 | 1.0146366 | 1.0232943 | 1.6462515 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r8__scratch_lbfgsb_whitened | 1.0195202 | 0.96682467 | 0.057709457 | 1.077953 | 1.1418076 | 2.0380326 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r8__scratch_adamw_lr_0p003 | 1.0136122 | 0.96983222 | 0.047773552 | 1.0151554 | 1.0316537 | 1.7885929 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r8__scratch_adamw_lr_0p01 | 1.0071771 | 1.044431 | 0.025830645 | 1.0580413 | 1.1227529 | 1.9348702 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r8__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0038399 | 1.040907 | 0.018260241 | 1.0696477 | 1.1412535 | 2.001867 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r8__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.0000432 | 0.42993896 | 0.0019165908 | 1.017239 | 1.0311411 | 1.6753177 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_lbfgsb_whitened | 1.0372154 | 0.97886565 | 0.19027455 | 1.0905809 | 1.1564461 | 2.0862795 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__scratch_adamw_lr_0p003 | 1.014255 | 0.95481286 | 0.052712802 | 1.0303045 | 1.0595372 | 1.750627 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r12__scratch_adamw_lr_0p01 | 1.0087644 | 0.97119125 | 0.034537595 | 1.0635035 | 1.128581 | 1.9532402 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r12__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0046942 | 0.96887533 | 0.014332299 | 1.0623097 | 1.1438936 | 2.020065 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r12__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.0000276 | 0.35235045 | 0.0022358883 | 1.0354746 | 1.0594093 | 1.7657788 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__scratch_lbfgsb_whitened | 1.0304336 | 0.98224499 | 0.156001 | 1.085746 | 1.1532355 | 2.0451869 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__scratch_adamw_lr_0p003 | 1.0173432 | 0.9600289 | 0.088976545 | 1.0634614 | 1.1243609 | 1.9431357 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r20__scratch_adamw_lr_0p01 | 1.0069733 | 0.94668997 | 0.032596581 | 1.0616696 | 1.1399774 | 2.0314223 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r20__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0035628 | 0.94540278 | 0.014164998 | 1.065966 | 1.1579516 | 2.0758552 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r20__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1.0000003 | 0.005098315 | 0.00035033918 | 0.9999903 | 0.99998549 | 1.555106 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r60__scratch_lbfgsb_whitened | 1.0133966 | 0.98180596 | 0.019863756 | 1.0844434 | 1.1584429 | 2.1262755 | 2000 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r60__scratch_adamw_lr_0p003 | 1.014407 | 0.97481973 | 0.10059054 | 1.0860757 | 1.1727692 | 2.0799962 | 5000 | AdamW completed 5000 full-batch steps (lr=0.003, clip=10000.0) |
| spline_r60__scratch_adamw_lr_0p01 | 1.0089183 | 0.96317147 | 0.062405411 | 1.0885763 | 1.1764069 | 2.1159924 | 5000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0) |
| spline_r60__scratch_adamw_then_lbfgsb_lr_0p01 | 1.0029899 | 0.962388 | 0.011571566 | 1.0877357 | 1.1742759 | 2.1092838 | 6000 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_r60__bellman_projected_adamw_then_lbfgsb_lr_0p01 | 1 | 0.00013081059 | 5.3918198e-06 | 0.99999525 | 0.99999186 | 1.5551146 | 5661 | AdamW completed 5000 full-batch steps (lr=0.01, clip=10000.0); L-BFGS-B polish maxiter=1000: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
