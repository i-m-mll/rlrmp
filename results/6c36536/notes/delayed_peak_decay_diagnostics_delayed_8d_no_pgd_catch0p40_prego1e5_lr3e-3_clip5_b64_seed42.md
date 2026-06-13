# Delayed Peak/Support-Decay Diagnostics

Run: `delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42`

Evaluation lens: corrected uniform 20-direction, 0.15 m delayed fixed banks over go cues 10..30. GRU values are one seeded stochastic rollout per bank trial and replicate, pooled over 5 replicates, 21 go cues, and 20 directions. The extLQG reference is deterministic: zero rollout draws with the standard C&S covariance-derived Kalman gains.

## Support-Decay Onset

| Metric | raw initial GRU/ext | first <95% | first <90% | first <85% | ratio steps 5-9 | ratio steps 10-14 |
|---|---:|---:|---:|---:|---:|---:|
| command | 0.843 | n/a | n/a | n/a | 1.61 | n/a |
| force/filter | 0.968 | 9 | 11 | 12 | 0.964 | 0.796 |
| acceleration | 1.01 | 5 | 9 | 11 | 0.924 | 0.757 |
| velocity | 1.1 | 6 | 8 | 15 | 0.905 | 0.868 |
| effector force | 0 | n/a | n/a | n/a | n/a | n/a |

Threshold ratios use positive target-radial support profiles after normalizing each GRU/extLQG profile by its own initial-launch window. Command uses steps 0..4; force/filter and velocity use steps 1..5 because support starts near zero at movement onset.

## Checkpoint Sweep

| Checkpoint | peak vel | t_peak | vel RMSE | shape err/ext peak | endpoint@go+59 | pre-go peak vel | catch peak vel | catch endpoint drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_0001000` | 0.6864 | 17 | 0.0679 | 0.0778 | 0.02750 | 0.00777 | 0.01688 | 0.00212 |
| `checkpoint_0002000` | 0.7060 | 16 | 0.0402 | 0.0435 | 0.02696 | 0.00689 | 0.02048 | 0.00225 |
| `checkpoint_0003000` | 0.6960 | 16 | 0.0339 | 0.0249 | 0.02742 | 0.00649 | 0.01734 | 0.00163 |
| `checkpoint_0004000` | 0.6861 | 16 | 0.0339 | 0.0172 | 0.02861 | 0.00655 | 0.02386 | 0.00129 |
| `checkpoint_0005000` | 0.6771 | 16 | 0.0358 | 0.0148 | 0.02958 | 0.00479 | 0.02107 | 0.00129 |
| `checkpoint_0006000` | 0.6825 | 16 | 0.0322 | 0.0141 | 0.02880 | 0.00584 | 0.02015 | 0.00095 |
| `checkpoint_0007000` | 0.6832 | 16 | 0.0325 | 0.0162 | 0.02839 | 0.00549 | 0.01882 | 0.00147 |
| `checkpoint_0008000` | 0.6855 | 16 | 0.0318 | 0.0188 | 0.02852 | 0.00625 | 0.01441 | 0.00094 |
| `checkpoint_0009000` | 0.6800 | 16 | 0.0339 | 0.0196 | 0.02983 | 0.00726 | 0.02397 | 0.00147 |
| `checkpoint_0010000` | 0.6744 | 15 | 0.0361 | 0.0203 | 0.02927 | 0.00902 | 0.02415 | 0.00097 |
| `checkpoint_0011000` | 0.6719 | 15 | 0.0393 | 0.0249 | 0.02892 | 0.00734 | 0.01846 | 0.00116 |
| `checkpoint_0012000` | 0.6773 | 15 | 0.0381 | 0.0278 | 0.02764 | 0.00836 | 0.02524 | 0.00146 |
| `final` | 0.6773 | 15 | 0.0381 | 0.0278 | 0.02764 | 0.00903 | 0.02579 | 0.00148 |

## Read

- Final checkpoint: peak `0.6773 m/s`, time-to-peak step `15`, shape error `0.0278` ext-peak units.
- Best shape checkpoint: `checkpoint_0006000` with shape error `0.0141` ext-peak units.
- Recommendation: checkpoint_0006000 is materially closer in scaled velocity shape than final by 0.0137 ext-peak units while passing the leakage screen.

Outputs:
- JSON: `_artifacts/6c36536/diagnostics/delayed_peak_decay_delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/delayed_peak_decay_diagnostics.json`
- Arrays: `_artifacts/6c36536/diagnostics/delayed_peak_decay_delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/delayed_peak_decay_profiles.npz`
