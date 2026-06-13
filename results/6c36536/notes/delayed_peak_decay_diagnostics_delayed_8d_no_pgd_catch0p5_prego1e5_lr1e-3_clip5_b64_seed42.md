# Delayed Peak/Support-Decay Diagnostics

Run: `delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42`

Evaluation lens: corrected uniform 20-direction, 0.15 m delayed fixed banks over go cues 10..30. GRU values are one seeded stochastic rollout per bank trial and replicate, pooled over 5 replicates, 21 go cues, and 20 directions. The extLQG reference is deterministic: zero rollout draws with the standard C&S covariance-derived Kalman gains.

## Support-Decay Onset

| Metric | raw initial GRU/ext | first <95% | first <90% | first <85% | ratio steps 5-9 | ratio steps 10-14 |
|---|---:|---:|---:|---:|---:|---:|
| command | 0.748 | n/a | n/a | n/a | 2.88 | n/a |
| force/filter | 0.814 | n/a | n/a | n/a | 1.22 | 1.34 |
| acceleration | 0.773 | n/a | n/a | n/a | 1.29 | 1.42 |
| velocity | 0.706 | n/a | n/a | n/a | 1.19 | 1.32 |
| effector force | 0 | n/a | n/a | n/a | n/a | n/a |

Threshold ratios use positive target-radial support profiles after normalizing each GRU/extLQG profile by its own initial-launch window. Command uses steps 0..4; force/filter and velocity use steps 1..5 because support starts near zero at movement onset.

## Checkpoint Sweep

| Checkpoint | peak vel | t_peak | vel RMSE | shape err/ext peak | endpoint@go+59 | pre-go peak vel | catch peak vel | catch endpoint drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_0001000` | 0.6587 | 20 | 0.1304 | 0.1550 | 0.02662 | 0.00799 | 0.01974 | 0.00172 |
| `checkpoint_0002000` | 0.6911 | 18 | 0.0818 | 0.0991 | 0.02416 | 0.00639 | 0.01586 | 0.00141 |
| `checkpoint_0003000` | 0.6978 | 17 | 0.0628 | 0.0768 | 0.02456 | 0.00656 | 0.01732 | 0.00173 |
| `checkpoint_0004000` | 0.6969 | 17 | 0.0513 | 0.0598 | 0.02635 | 0.00500 | 0.01841 | 0.00151 |
| `checkpoint_0005000` | 0.6907 | 17 | 0.0451 | 0.0486 | 0.02865 | 0.00517 | 0.01702 | 0.00175 |
| `checkpoint_0006000` | 0.6942 | 16 | 0.0394 | 0.0417 | 0.02718 | 0.00508 | 0.01905 | 0.00166 |
| `checkpoint_0007000` | 0.6946 | 16 | 0.0351 | 0.0347 | 0.02723 | 0.00599 | 0.01758 | 0.00134 |
| `checkpoint_0008000` | 0.6935 | 16 | 0.0325 | 0.0294 | 0.02693 | 0.00558 | 0.01880 | 0.00139 |
| `checkpoint_0009000` | 0.6932 | 16 | 0.0310 | 0.0269 | 0.02759 | 0.00507 | 0.01925 | 0.00140 |
| `checkpoint_0010000` | 0.6930 | 16 | 0.0301 | 0.0250 | 0.02748 | 0.00555 | 0.01871 | 0.00135 |
| `checkpoint_0011000` | 0.6953 | 16 | 0.0281 | 0.0230 | 0.02752 | 0.00487 | 0.02151 | 0.00144 |
| `checkpoint_0012000` | 0.6958 | 16 | 0.0272 | 0.0223 | 0.02769 | 0.00536 | 0.01965 | 0.00136 |
| `final` | 0.6955 | 16 | 0.0273 | 0.0222 | 0.02775 | 0.00581 | 0.01861 | 0.00137 |

## Read

- Final checkpoint: peak `0.6955 m/s`, time-to-peak step `16`, shape error `0.0222` ext-peak units.
- Best shape checkpoint: `final` with shape error `0.0222` ext-peak units.
- Recommendation: Checkpoint choice is not the main explanation: the final/latest checkpoint is effectively as close in velocity shape as the best leakage-acceptable checkpoint.

Outputs:
- JSON: `_artifacts/6c36536/diagnostics/delayed_peak_decay_delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42/delayed_peak_decay_diagnostics.json`
- Arrays: `_artifacts/6c36536/diagnostics/delayed_peak_decay_delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42/delayed_peak_decay_profiles.npz`
