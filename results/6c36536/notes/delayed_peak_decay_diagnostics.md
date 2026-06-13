# Delayed Peak/Support-Decay Diagnostics

Run: `delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b64_seed42`

Evaluation lens: corrected uniform 20-direction, 0.15 m delayed fixed banks over go cues 10..30. GRU values are one seeded stochastic rollout per bank trial and replicate, pooled over 5 replicates, 21 go cues, and 20 directions. The extLQG reference is deterministic: zero rollout draws with the standard C&S covariance-derived Kalman gains.

## Support-Decay Onset

| Metric | raw initial GRU/ext | first <95% | first <90% | first <85% | ratio steps 5-9 | ratio steps 10-14 |
|---|---:|---:|---:|---:|---:|---:|
| command | 0.848 | n/a | n/a | n/a | 1.72 | n/a |
| force/filter | 0.975 | 10 | 12 | 13 | 0.98 | 0.852 |
| acceleration | 0.995 | 9 | 12 | 13 | 0.96 | 0.83 |
| velocity | 1.03 | 8 | 18 | 24 | 0.955 | 0.932 |
| effector force | 0 | n/a | n/a | n/a | n/a | n/a |

Threshold ratios use positive target-radial support profiles after normalizing each GRU/extLQG profile by its own initial-launch window. Command uses steps 0..4; force/filter and velocity use steps 1..5 because support starts near zero at movement onset.

## Checkpoint Sweep

| Checkpoint | peak vel | t_peak | vel RMSE | shape err/ext peak | endpoint@go+59 | pre-go peak vel | catch peak vel | catch endpoint drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_0001000` | 0.6856 | 18 | 0.0830 | 0.0989 | 0.02761 | 0.00624 | 0.01616 | 0.00176 |
| `checkpoint_0002000` | 0.6935 | 17 | 0.0518 | 0.0559 | 0.02748 | 0.00698 | 0.02074 | 0.00192 |
| `checkpoint_0003000` | 0.6949 | 16 | 0.0383 | 0.0334 | 0.02822 | 0.00508 | 0.01837 | 0.00163 |
| `checkpoint_0004000` | 0.6891 | 16 | 0.0344 | 0.0208 | 0.02855 | 0.00550 | 0.01916 | 0.00181 |
| `checkpoint_0005000` | 0.6890 | 16 | 0.0318 | 0.0160 | 0.02888 | 0.00655 | 0.01808 | 0.00185 |
| `checkpoint_0006000` | 0.6858 | 16 | 0.0316 | 0.0146 | 0.02855 | 0.00529 | 0.02937 | 0.00144 |
| `checkpoint_0007000` | 0.6886 | 16 | 0.0303 | 0.0163 | 0.02869 | 0.00553 | 0.01787 | 0.00092 |
| `checkpoint_0008000` | 0.6890 | 16 | 0.0307 | 0.0192 | 0.02861 | 0.00481 | 0.01804 | 0.00125 |
| `checkpoint_0009000` | 0.6841 | 16 | 0.0329 | 0.0206 | 0.02922 | 0.00507 | 0.01575 | 0.00096 |
| `checkpoint_0010000` | 0.6892 | 16 | 0.0315 | 0.0228 | 0.02896 | 0.00506 | 0.01452 | 0.00072 |
| `checkpoint_0011000` | 0.6873 | 15 | 0.0321 | 0.0232 | 0.02887 | 0.00516 | 0.01557 | 0.00085 |
| `checkpoint_0012000` | 0.6885 | 15 | 0.0327 | 0.0251 | 0.02882 | 0.00528 | 0.01667 | 0.00094 |
| `final` | 0.6886 | 15 | 0.0327 | 0.0252 | 0.02884 | 0.00528 | 0.01948 | 0.00097 |

## Read

- Final checkpoint: peak `0.6886 m/s`, time-to-peak step `15`, shape error `0.0252` ext-peak units.
- Best shape checkpoint: `checkpoint_0006000` with shape error `0.0146` ext-peak units.
- Recommendation: checkpoint_0006000 is materially closer in scaled velocity shape than final by 0.0106 ext-peak units while passing the leakage screen.

Outputs:
- JSON: `_artifacts/6c36536/diagnostics/delayed_peak_decay/delayed_peak_decay_diagnostics.json`
- Arrays: `_artifacts/6c36536/diagnostics/delayed_peak_decay/delayed_peak_decay_profiles.npz`
