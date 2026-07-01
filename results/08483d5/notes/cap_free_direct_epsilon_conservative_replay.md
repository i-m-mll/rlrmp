<!-- AUTO-GENERATED: cap_free_direct_epsilon_conservative_replay -->
# Cap-Free Direct-Epsilon Conservative Replay

## Headline

This no-launch replay loads the clean 6D no-PGD H0 `const_band16` baseline, uses normal 15 cm validation trials with varying reach directions, and aggregates each damage estimate over all five GRU checkpoint replicates.

The inner objective is `mean(task_cost) - lambda * mean(sum_t,d epsilon[t,d]^2)`. No projection, safety cap, inherited radius, or trust-region value is used.

## Inputs

- Command: `PYTHONPATH=src uv run --no-sync python results/08483d5/scripts/run_cap_free_direct_epsilon_conservative_replay.py --include-beta14`.
- Runtime: `25.0` seconds.
- Run spec: `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json`.
- Checkpoint: `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest/model.eqx`.
- Lambda source: `results/06a4dc8/canonical_soft_lambda_hvp.json`.
- Batch: 64 trials from 72 validation trials; repeated singleton: `False`.
- Reach length mean: `0.15` m; std: `3.65e-09`.
- Replicates: all 5.

## Optimizer And Adaptive Rule

- Optimizer steps: `12`; Adam learning rate: `2e-05`; gradient clip L2: `1000000.0`.
- `lambda0` beta 1.05: `281032999`.
- EMA initializes from the first aggregate damage. EMA alpha, lambda eta: `0.1`.
- Max log lambda step: `0.1`; deadband: `+/-10%`.

## Target: beta1p05_deterministic_output_feedback

- Reference damage: `447.8904024` from `results/08483d5/notes/output_feedback_damage_estimate_beta1p05.json`.
- First raw damage: `1765.9537`; last raw damage: `647.26395`.
- First EMA damage: `1765.9537`; last EMA damage: `1541.3931`.
- Max raw step change: `444.26286`; max EMA step change: `99.347685`.
- Last decision in deadband: `False`; target reached: `False`; directional/smoothness criterion: `True`.

| iter | lambda | raw damage | EMA damage | damage/ref | EMA/ref | decision | log step | next lambda | energy/trial | finite | nonzero |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|:---:|
| 0 | 2.81033e+08 | 1765.95 | 1765.95 | 3.9428 | 3.9428 | increase_lambda | 0.1 | 3.10589e+08 | 4.40648e-06 | True | True |
| 1 | 3.10589e+08 | 1545.76 | 1743.93 | 3.4512 | 3.8937 | increase_lambda | 0.1 | 3.43254e+08 | 3.66326e-06 | True | True |
| 2 | 3.43254e+08 | 1322.24 | 1701.76 | 2.9521 | 3.7995 | increase_lambda | 0.1 | 3.79355e+08 | 2.96177e-06 | True | True |
| 3 | 3.79355e+08 | 1091.53 | 1640.74 | 2.437 | 3.6633 | increase_lambda | 0.1 | 4.19252e+08 | 2.2842e-06 | True | True |
| 4 | 4.19252e+08 | 647.264 | 1541.39 | 1.4451 | 3.4415 | increase_lambda | 0.1 | 4.63345e+08 | 1.11677e-06 | True | True |

Per-replicate damage on the final row:

| replicate | mean | median | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0 | 644.35137 | 641.67233 | 130.45481 | 405.89712 | 928.99751 |
| 1 | 647.88863 | 636.05046 | 129.94671 | 396.89871 | 902.01369 |
| 2 | 653.48454 | 647.8196 | 130.82305 | 417.84869 | 962.7512 |
| 3 | 641.85003 | 600.09301 | 138.31948 | 409.64068 | 996.87162 |
| 4 | 648.74518 | 648.02826 | 109.35797 | 430.13475 | 898.09567 |

Final-row replicate-mean summary: mean `647.26395`, median `647.88863`, std `3.9767735`.

## Target: beta1p05_paired_nominal_noise_output_feedback

- Reference damage: `1911.893097` from `results/08483d5/notes/output_feedback_damage_estimate_beta1p05.json`.
- First raw damage: `1765.9537`; last raw damage: `1765.9537`.
- First EMA damage: `1765.9537`; last EMA damage: `1765.9537`.
- Max raw step change: `0`; max EMA step change: `0`.
- Last decision in deadband: `True`; target reached: `True`; directional/smoothness criterion: `True`.

| iter | lambda | raw damage | EMA damage | damage/ref | EMA/ref | decision | log step | next lambda | energy/trial | finite | nonzero |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|:---:|
| 0 | 2.81033e+08 | 1765.95 | 1765.95 | 0.92367 | 0.92367 | unchanged_deadband | 0 | 2.81033e+08 | 4.40648e-06 | True | True |
| 1 | 2.81033e+08 | 1765.95 | 1765.95 | 0.92367 | 0.92367 | unchanged_deadband | 0 | 2.81033e+08 | 4.40648e-06 | True | True |
| 2 | 2.81033e+08 | 1765.95 | 1765.95 | 0.92367 | 0.92367 | unchanged_deadband | 0 | 2.81033e+08 | 4.40648e-06 | True | True |
| 3 | 2.81033e+08 | 1765.95 | 1765.95 | 0.92367 | 0.92367 | unchanged_deadband | 0 | 2.81033e+08 | 4.40648e-06 | True | True |
| 4 | 2.81033e+08 | 1765.95 | 1765.95 | 0.92367 | 0.92367 | unchanged_deadband | 0 | 2.81033e+08 | 4.40648e-06 | True | True |

Per-replicate damage on the final row:

| replicate | mean | median | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0 | 1741.5201 | 1793.2488 | 320.91774 | 1147.8177 | 2295.6919 |
| 1 | 1763.6555 | 1805.2833 | 314.88945 | 1176.2254 | 2255.5706 |
| 2 | 1796.6964 | 1851.422 | 327.67904 | 1089.1333 | 2401.3904 |
| 3 | 1760.2935 | 1761.3311 | 327.60697 | 1158.6057 | 2427.7438 |
| 4 | 1767.6032 | 1862.3612 | 291.7342 | 1049.156 | 2225.812 |

Final-row replicate-mean summary: mean `1765.9537`, median `1763.6555`, std `17.78849`.

## Target: beta1p4_deterministic_output_feedback

- Reference damage: `3704.963269` from `results/08483d5/notes/output_feedback_damage_estimate.json`.
- First raw damage: `1765.9537`; last raw damage: `2185.1503`.
- First EMA damage: `1765.9537`; last EMA damage: `1877.7569`.
- Max raw step change: `167.55245`; max EMA step change: `34.154824`.
- Last decision in deadband: `False`; target reached: `False`; directional/smoothness criterion: `True`.

| iter | lambda | raw damage | EMA damage | damage/ref | EMA/ref | decision | log step | next lambda | energy/trial | finite | nonzero |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|:---:|
| 0 | 2.81033e+08 | 1765.95 | 1765.95 | 0.47665 | 0.47665 | decrease_lambda | -0.074098 | 2.60962e+08 | 4.40648e-06 | True | True |
| 1 | 2.60962e+08 | 1933.51 | 1782.71 | 0.52187 | 0.48117 | decrease_lambda | -0.073154 | 2.42553e+08 | 5.01242e-06 | True | True |
| 2 | 2.42553e+08 | 2065.24 | 1810.96 | 0.55743 | 0.48879 | decrease_lambda | -0.071581 | 2.25797e+08 | 5.50911e-06 | True | True |
| 3 | 2.25797e+08 | 2137.36 | 1843.6 | 0.57689 | 0.4976 | decrease_lambda | -0.069795 | 2.10575e+08 | 5.7896e-06 | True | True |
| 4 | 2.10575e+08 | 2185.15 | 1877.76 | 0.58979 | 0.50682 | decrease_lambda | -0.06796 | 1.9674e+08 | 5.98523e-06 | True | True |

Per-replicate damage on the final row:

| replicate | mean | median | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0 | 2142.0049 | 2189.0001 | 232.22845 | 1259.1529 | 2547.1682 |
| 1 | 2180.917 | 2184.0817 | 225.55857 | 1317.388 | 2535.8341 |
| 2 | 2228.5143 | 2259.1134 | 235.05495 | 1268.8346 | 2677.9063 |
| 3 | 2186.5414 | 2176.6405 | 239.16739 | 1265.8009 | 2747.1976 |
| 4 | 2187.7741 | 2218.2719 | 197.56951 | 1432.688 | 2546.4647 |

Final-row replicate-mean summary: mean `2185.1503`, median `2186.5414`, std `27.454339`.

## Target: beta1p4_paired_nominal_noise_output_feedback

- Reference damage: `6131.690676` from `results/08483d5/notes/output_feedback_damage_estimate.json`.
- First raw damage: `1765.9537`; last raw damage: `2238.5885`.
- First EMA damage: `1765.9537`; last EMA damage: `1896.5213`.
- Max raw step change: `222.11152`; max EMA step change: `38.007465`.
- Last decision in deadband: `False`; target reached: `False`; directional/smoothness criterion: `True`.

| iter | lambda | raw damage | EMA damage | damage/ref | EMA/ref | decision | log step | next lambda | energy/trial | finite | nonzero |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|:---:|
| 0 | 2.81033e+08 | 1765.95 | 1765.95 | 0.288 | 0.288 | decrease_lambda | -0.1 | 2.54289e+08 | 4.40648e-06 | True | True |
| 1 | 2.54289e+08 | 1988.07 | 1788.16 | 0.32423 | 0.29163 | decrease_lambda | -0.1 | 2.3009e+08 | 5.21653e-06 | True | True |
| 2 | 2.3009e+08 | 2121.48 | 1821.5 | 0.34599 | 0.29706 | decrease_lambda | -0.1 | 2.08194e+08 | 5.7267e-06 | True | True |
| 3 | 2.08194e+08 | 2191.67 | 1858.51 | 0.35743 | 0.3031 | decrease_lambda | -0.1 | 1.88382e+08 | 6.01285e-06 | True | True |
| 4 | 1.88382e+08 | 2238.59 | 1896.52 | 0.36509 | 0.3093 | decrease_lambda | -0.1 | 1.70455e+08 | 6.22065e-06 | True | True |

Per-replicate damage on the final row:

| replicate | mean | median | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0 | 2193.4899 | 2239.8563 | 232.55985 | 1308.4423 | 2598.2795 |
| 1 | 2233.599 | 2237.803 | 228.72857 | 1323.9212 | 2589.7606 |
| 2 | 2282.855 | 2313.5461 | 236.87943 | 1296.5833 | 2728.8254 |
| 3 | 2241.0524 | 2233.6483 | 239.83524 | 1311.0926 | 2804.8568 |
| 4 | 2241.9463 | 2273.3487 | 197.51249 | 1487.6035 | 2604.6408 |

Final-row replicate-mean summary: mean `2238.5885`, median `2241.0524`, std `28.410033`.

## Interpretation

The conservative rule is less jumpy if the EMA step changes are smaller than the raw damage changes and lambda changes stay bounded by the 0.1 log-step cap. It still fails as a target-tracking rule if the raw or EMA sequence moves away from the requested damage target or never enters the deadband.

## Residual Uncertainty

- This is a local frozen replay, not controller training.
- The first 64 of 72 validation trials are used to satisfy the requested batch size; reach length is fixed at 15 cm but directions vary.
- Optimizer iterations are kept practical for CPU; batch size and replicate aggregation are not reduced.
- The unconstrained inner optimizer can still pick very large epsilon if the soft objective rewards it.

## Checks Run

- `PYTHONPATH=src uv run --no-sync python results/08483d5/scripts/run_cap_free_direct_epsilon_conservative_replay.py --include-beta14`
<!-- /AUTO-GENERATED -->
