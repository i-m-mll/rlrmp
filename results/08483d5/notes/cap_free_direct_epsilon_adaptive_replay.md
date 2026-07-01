<!-- AUTO-GENERATED: cap_free_direct_epsilon_adaptive_replay -->
# Cap-Free Direct-Epsilon Adaptive Replay

## Headline

This replay loads the clean 6D no-PGD H0 `const_band16` baseline and runs a cap-free direct-epsilon inner optimizer with frozen controller weights.

No projection, inherited radius, trust-region value, or cap/guard criterion is used. Selected epsilon energy and norm are reported only as outcomes.

## Inputs

- Run spec: `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json`.
- Checkpoint: `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest/model.eqx`.
- Lambda source: `results/06a4dc8/canonical_soft_lambda_hvp.json`.
- Damage references: `results/08483d5/notes/output_feedback_damage_estimate.json`.
- Batch: 2 repeated fixed +x 15 cm validation trials; replicate 0 of 5 was used for CPU cost.

## Optimizer

- Method: `cap_free_adam_ascent` from zero epsilon.
- Objective: `mean(task_cost) - lambda * mean(sum_t,d epsilon[t,d]^2)`.
- Steps: `35`; Adam learning rate: `2e-05`; gradient clip L2: `1000000.0`.
- `lambda_curv_p90`: `254905215`; `lambda0` beta 1.05: `281032999`.

## Activity Bracket

This is a batch- and optimizer-dependent activity bracket, not a theoretical lambda_zero estimate.

| lambda | damage | objective gain | energy/trial | norm/trial | finite | nonzero |
|---:|---:|---:|---:|---:|:---:|:---:|
| 2.54905e+10 | 0 | 0 | 0 | 0 | True | False |
| 7.64716e+09 | 7.17864 | 2.96746 | 5.50685e-10 | 2.20506e-05 | True | True |
| 2.54905e+09 | 34.3608 | 16.2836 | 7.09175e-09 | 8.25656e-05 | True | True |
| 7.64716e+08 | 162.49 | 71.6286 | 1.18818e-07 | 0.000343184 | True | True |
| 2.54905e+08 | 7101.76 | 1252.43 | 2.29471e-05 | 0.00476062 | True | True |
| 7.64716e+07 | 16805.3 | 11464.6 | 6.98396e-05 | 0.00835544 | True | True |
| 2.54905e+07 | 17904 | 15327.7 | 0.00010107 | 0.0100509 | True | True |
| 7.64716e+06 | 17943 | 17083.6 | 0.000112382 | 0.010598 | True | True |
| 2.54905e+06 | 17895.2 | 17590.7 | 0.000119434 | 0.0109237 | True | True |

## Adaptive Target: deterministic_output_feedback

- Reference damage: `3704.963269`.
- Pass frozen criterion: `False`.
- First damage: `4590.7041`; last damage: `842.75089`.
- First absolute error: `885.74085`; last absolute error: `2862.2124`.

| iter | lambda | clean cost | selected cost | damage | damage/ref | energy/trial | norm/trial | finite | nonzero | objective gain | next lambda |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|
| 0 | 2.81033e+08 | 4405.52 | 8996.23 | 4590.7 | 1.2391 | 1.37012e-05 | 0.00368658 | True | True | 740.228 | 3.12828e+08 |
| 1 | 3.12828e+08 | 4405.52 | 7011.61 | 2606.09 | 0.7034 | 6.91654e-06 | 0.00258691 | True | True | 442.401 | 2.62366e+08 |
| 2 | 2.62366e+08 | 4405.52 | 10545 | 6139.51 | 1.6571 | 1.9299e-05 | 0.00437796 | True | True | 1076.1 | 3.3774e+08 |
| 3 | 3.3774e+08 | 4405.52 | 5976.79 | 1571.27 | 0.4241 | 3.64478e-06 | 0.00186076 | True | True | 340.279 | 2.19946e+08 |
| 4 | 2.19946e+08 | 4405.52 | 15135 | 10729.5 | 2.896 | 3.72014e-05 | 0.00601608 | True | True | 2547.21 | 3.74294e+08 |
| 5 | 3.74294e+08 | 4405.52 | 5248.27 | 842.751 | 0.22747 | 1.56853e-06 | 0.00122648 | True | True | 255.66 | 1.78513e+08 |

Blockers:
- damage did not move closer to the target over the adaptive sequence.

## Adaptive Target: paired_nominal_noise_output_feedback

- Reference damage: `6131.690677`.
- Pass frozen criterion: `False`.
- First damage: `4590.7041`; last damage: `10294.321`.
- First absolute error: `1540.9866`; last absolute error: `4162.6302`.

| iter | lambda | clean cost | selected cost | damage | damage/ref | energy/trial | norm/trial | finite | nonzero | objective gain | next lambda |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|
| 0 | 2.81033e+08 | 4405.52 | 8996.23 | 4590.7 | 0.74868 | 1.37012e-05 | 0.00368658 | True | True | 740.228 | 2.43168e+08 |
| 1 | 2.43168e+08 | 4405.52 | 13200.1 | 8794.57 | 1.4343 | 2.95294e-05 | 0.00536826 | True | True | 1613.95 | 2.91222e+08 |
| 2 | 2.91222e+08 | 4405.52 | 8302.64 | 3897.12 | 0.63557 | 1.1277e-05 | 0.00333508 | True | True | 613.005 | 2.3217e+08 |
| 3 | 2.3217e+08 | 4405.52 | 14320 | 9914.45 | 1.6169 | 3.39399e-05 | 0.00574796 | True | True | 2034.64 | 2.95223e+08 |
| 4 | 2.95223e+08 | 4405.52 | 8042.49 | 3636.97 | 0.59314 | 1.03804e-05 | 0.00319493 | True | True | 572.433 | 2.27368e+08 |
| 5 | 2.27368e+08 | 4405.52 | 14699.8 | 10294.3 | 1.6789 | 3.54537e-05 | 0.00587567 | True | True | 2233.29 | 2.94604e+08 |

Blockers:
- damage did not move closer to the target over the adaptive sequence.

## Residual Uncertainties

- This is a small local frozen replay, not a training launch.
- The activity bracket is optimizer- and batch-dependent.
- CPU cost kept the replay to one replicate and a small repeated validation batch.
- The direct-epsilon optimizer is unconstrained; if damage grows sharply at lower lambda, that is reported as optimizer behavior rather than clipped away.
<!-- /AUTO-GENERATED -->
