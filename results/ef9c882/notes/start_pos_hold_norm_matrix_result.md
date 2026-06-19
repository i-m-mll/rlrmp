# Start-Position Hold Norm Matrix Result

This note records the first fixed-bank readout for the five delayed pre-go
start-position-only hold rows requested under issue `ef9c882`. All rows used
target-visible delayed reach, go cues 10-30, `p_catch_trial=0.5`, 5 GRU
replicates, hidden size 180, batch size 64, 12000 training batches, full
movement-period C&S Q/R/Qf, no PGD, no `nn_output_pre_go`, no pre-go
zero-velocity hold, and no force/filter hold. The fixed evaluation bank uses 20
uniform target directions at 0.15 m reach length.

## Fixed-Bank No-Catch Readout

| Row | Hold norm | Weight | Peak velocity (m/s) | Time to peak (s) | Endpoint error (m) | Terminal speed (m/s) | Path length (m) | Final forward displacement (m) | Pre-go peak speed (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hold_start_pos_l2__w1e4` | L2 | 1e4 | 0.536715 | 0.14 | 0.014617 | 0.013716 | 0.121241 | 0.141913 | 0.243493 |
| `hold_start_pos_l2__w1e6` | L2 | 1e6 | 0.648427 | 0.14 | 0.021153 | 0.015973 | 0.140501 | 0.138104 | 0.063057 |
| `hold_start_pos_l2__w1e8` | L2 | 1e8 | 0.666288 | 0.15 | 0.022336 | 0.022096 | 0.142716 | 0.136639 | 0.006792 |
| `hold_start_pos_l1__w1e8` | L1 | 1e8 | 0.679563 | 0.18 | 0.031241 | 0.034507 | 0.139385 | 0.129507 | 0.003859 |
| `hold_start_pos_l1__w1e6` | L1 | 1e6 | 0.674028 | 0.16 | 0.028532 | 0.015766 | 0.139844 | 0.132874 | 0.002262 |

Reference sidecar from the same velocity materialization: the 8D extLQG
output-feedback comparator peaks at 0.731057 m/s at 0.16 s and has terminal
position error 0.003262 m; the 4D comparator peaks at 0.730759 m/s at 0.16 s
and has terminal position error 0.003300 m.

Catch-bank target-radial velocity stayed flat for all five rows in this
fixed-bank readout. The weak L2 1e4 row has the smallest endpoint error among
the five new rows, but it is not usable as anti-anticipation evidence because
its mean pre-go peak speed is 0.243493 m/s. The strongest L1/L2 holds suppress
pre-go speed much better, but the improved no-catch peak velocity remains below
the extLQG reference and comes with larger endpoint error or terminal speed.

Compared with the previous `hold__start_pos_zero_vel` row from the same issue
(peak 0.676107 m/s, endpoint error 0.018476 m, terminal speed 0.012682 m/s,
pre-go peak speed 0.005704 m/s), removing the zero-velocity hold and switching
to strong L1 can recover a little peak velocity, but it worsens endpoint quality.

## Durable Outputs

- Run specs: `results/ef9c882/runs/hold_start_pos_*.json`
- Bulk run artifacts: `_artifacts/ef9c882/runs/hold_start_pos_*`
- Velocity figure spec: `results/ef9c882/figures/start_pos_hold_norm_velocity_profiles/spec.json`
- No-catch velocity summary: `_artifacts/ef9c882/figures/start_pos_hold_norm_velocity_profiles/no_catch/velocity_profile_summary.json`
- Catch velocity summary: `_artifacts/ef9c882/figures/start_pos_hold_norm_velocity_profiles/catch/velocity_profile_summary.json`
