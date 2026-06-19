# Matched force-filter/perturbation start-position hold addendum

This addendum reruns the start-position hold rows with the delayed timing/hold lane
contract restored: force-filter feedback enabled, calibrated movement-age perturbation
training enabled, physical perturbation level `small`, full analytical Q/R/Qf movement
objective, go cue 10..30, and `p_catch_trial=0.5`. The earlier `hold_start_pos_*` norm
rows are retained as historical evidence, but are not directly comparable to
`hold__start_pos_zero_vel` because their launch planner omitted the force-filter feedback
and calibrated perturbation-training flags.

## No-catch fixed-bank readout

All rows below were evaluated on the same no-catch fixed delayed bank used by the velocity
profile materializer: 20 directions, go cues 10..30, 0.15 m reach length, final checkpoint.

| Row | Peak velocity (m/s) | Time of peak (s) | Endpoint error (m) | Terminal speed (m/s) | Path length (m) | Final forward vel (m/s) | Pre-go peak speed (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hold__start_pos_zero_vel` | 0.676107 | 0.16 | 0.019641 | 0.011868 | 0.140514 | 0.005957 | 0.023624 |
| `hold_start_pos_l2_ffpert__w1e6_lr3e-3` | 0.656566 | 0.15 | 0.016405 | 0.010652 | 0.139295 | 0.005505 | 0.104795 |
| `hold_start_pos_l2_ffpert__w1e8_lr3e-3` | 0.661617 | 0.16 | 0.025068 | 0.017161 | 0.139150 | 0.008263 | 0.036130 |
| `hold_start_pos_l1_ffpert__w1e6_lr3e-3` | 0.676777 | 0.16 | 0.033312 | 0.010360 | 0.138432 | 0.003625 | 0.015537 |
| `hold_start_pos_l1_ffpert__w1e5_lr3e-3` | 0.657027 | 0.16 | 0.023770 | 0.016043 | 0.137279 | 0.008368 | 0.038227 |
| `hold_start_pos_l2_ffpert__w1e8_lr1e-2` | 0.677548 | 0.16 | 0.024156 | 0.015427 | 0.143171 | 0.004250 | 0.038262 |
| `hold_start_pos_l1_ffpert__w1e5_lr1e-2` | 0.679120 | 0.16 | 0.021174 | 0.017226 | 0.142945 | 0.008157 | 0.042579 |
| `hold__start_pos_zero_vel_lr1e-2` | 0.683531 | 0.16 | 0.017386 | 0.009713 | 0.143314 | 0.004239 | 0.019454 |

The matched addendum does not solve the peak-velocity depression relative to the 8D extLQG
reference peak (0.731057 m/s), but the high-LR zero-velocity row is the strongest row in
this addendum: it has the highest no-catch peak, lower endpoint error than the original
zero-velocity row, and lower terminal speed. Among start-position-only rows, L1 1e5 at
LR 1e-2 has the highest peak, while L2 1e6 at LR 3e-3 has the smallest endpoint error but
substantially more pre-go leakage.

## Artifacts

- Tracked run specs: `results/ef9c882/runs/hold_start_pos_*_ffpert__*.json` and
  `results/ef9c882/runs/hold__start_pos_zero_vel_lr1e-2.json`
- Bulk run artifacts: `_artifacts/ef9c882/runs/<run>/`
- Velocity figure spec:
  `results/ef9c882/figures/start_pos_hold_norm_matched_ffpert_velocity_profiles/spec.json`
- No-catch velocity summary:
  `_artifacts/ef9c882/figures/start_pos_hold_norm_matched_ffpert_velocity_profiles/no_catch/velocity_profile_summary.json`
- Catch velocity summary:
  `_artifacts/ef9c882/figures/start_pos_hold_norm_matched_ffpert_velocity_profiles/catch/velocity_profile_summary.json`
