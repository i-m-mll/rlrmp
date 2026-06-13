# Delayed Peak/Support-Decay Diagnostics

Run: `normloss_3e3_s42`

Evaluation lens: corrected uniform 20-direction, 0.15 m delayed fixed banks over go cues 10..30. GRU values are one seeded stochastic rollout per bank trial and replicate, pooled over 5 replicates, 21 go cues, and 20 directions. The extLQG reference is deterministic: zero rollout draws with the standard C&S covariance-derived Kalman gains.

## Support-Decay Onset

| Metric | raw initial GRU/ext | first <95% | first <90% | first <85% | ratio steps 5-9 | ratio steps 10-14 |
|---|---:|---:|---:|---:|---:|---:|
| command | 0.852 | n/a | n/a | n/a | 1.58 | n/a |
| force/filter | 0.98 | 8 | 11 | 12 | 0.95 | 0.791 |
| acceleration | 1.02 | 5 | 9 | 11 | 0.909 | 0.752 |
| velocity | 1.12 | 6 | 7 | 13 | 0.896 | 0.854 |
| effector force | 0 | n/a | n/a | n/a | n/a | n/a |

Threshold ratios use positive target-radial support profiles after normalizing each GRU/extLQG profile by its own initial-launch window. Command uses steps 0..4; force/filter and velocity use steps 1..5 because support starts near zero at movement onset.

## Checkpoint Sweep

| Checkpoint | peak vel | t_peak | vel RMSE | shape err/ext peak | endpoint@go+59 | pre-go peak vel | catch peak vel | catch endpoint drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_0001000` | 0.6980 | 17 | 0.0586 | 0.0674 | 0.02547 | 0.00880 | 0.01768 | 0.00172 |
| `checkpoint_0002000` | 0.6940 | 16 | 0.0387 | 0.0332 | 0.02573 | 0.00660 | 0.01831 | 0.00224 |
| `checkpoint_0003000` | 0.6895 | 16 | 0.0333 | 0.0188 | 0.02576 | 0.00769 | 0.03067 | 0.00190 |
| `checkpoint_0004000` | 0.6810 | 16 | 0.0346 | 0.0152 | 0.02665 | 0.00665 | 0.01725 | 0.00088 |
| `checkpoint_0005000` | 0.6849 | 16 | 0.0331 | 0.0175 | 0.02731 | 0.00690 | 0.01676 | 0.00126 |
| `checkpoint_0006000` | 0.6882 | 16 | 0.0308 | 0.0178 | 0.02788 | 0.00597 | 0.01982 | 0.00123 |
| `checkpoint_0007000` | 0.6862 | 16 | 0.0315 | 0.0188 | 0.02853 | 0.00632 | 0.02175 | 0.00103 |
| `checkpoint_0008000` | 0.6843 | 16 | 0.0320 | 0.0194 | 0.02831 | 0.00695 | 0.01694 | 0.00104 |
| `checkpoint_0009000` | 0.6851 | 15 | 0.0321 | 0.0210 | 0.02904 | 0.00569 | 0.01792 | 0.00094 |
| `checkpoint_0010000` | 0.6829 | 15 | 0.0342 | 0.0236 | 0.02883 | 0.00743 | 0.01951 | 0.00100 |
| `checkpoint_0011000` | 0.6816 | 15 | 0.0349 | 0.0251 | 0.02960 | 0.00757 | 0.01945 | 0.00085 |
| `checkpoint_0012000` | 0.6809 | 15 | 0.0367 | 0.0284 | 0.02861 | 0.00949 | 0.02132 | 0.00099 |
| `final` | 0.6809 | 15 | 0.0367 | 0.0284 | 0.02856 | 0.00963 | 0.02263 | 0.00101 |

## Read

- Final checkpoint: peak `0.6809 m/s`, time-to-peak step `15`, shape error `0.0284` ext-peak units.
- Best shape checkpoint: `checkpoint_0004000` with shape error `0.0152` ext-peak units.
- Recommendation: checkpoint_0004000 is materially closer in scaled velocity shape than final by 0.0132 ext-peak units while passing the leakage screen.

Outputs:
- JSON: `_artifacts/6c36536/diagnostics/delayed_peak_decay_normloss_3e3_s42/delayed_peak_decay_diagnostics.json`
- Arrays: `_artifacts/6c36536/diagnostics/delayed_peak_decay_normloss_3e3_s42/delayed_peak_decay_profiles.npz`
