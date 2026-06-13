# Delayed Peak/Support-Decay Diagnostics

Run: `delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42`

Evaluation lens: corrected uniform 20-direction, 0.15 m delayed fixed banks over go cues 10..30. GRU values are one seeded stochastic rollout per bank trial and replicate, pooled over 5 replicates, 21 go cues, and 20 directions. The extLQG reference is deterministic: zero rollout draws with the standard C&S covariance-derived Kalman gains.

## Support-Decay Onset

| Metric | raw initial GRU/ext | first <95% | first <90% | first <85% | ratio steps 5-9 | ratio steps 10-14 |
|---|---:|---:|---:|---:|---:|---:|
| command | 0.862 | n/a | n/a | n/a | 1.09 | n/a |
| force/filter | 1 | 6 | 6 | 7 | 0.843 | 0.573 |
| acceleration | 1.12 | 5 | 5 | 5 | 0.749 | 0.499 |
| velocity | 1.57 | 6 | 6 | 6 | 0.708 | 0.611 |
| effector force | 0 | n/a | n/a | n/a | n/a | n/a |

Threshold ratios use positive target-radial support profiles after normalizing each GRU/extLQG profile by its own initial-launch window. Command uses steps 0..4; force/filter and velocity use steps 1..5 because support starts near zero at movement onset.

## Checkpoint Sweep

| Checkpoint | peak vel | t_peak | vel RMSE | shape err/ext peak | endpoint@go+59 | pre-go peak vel | catch peak vel | catch endpoint drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_0001000` | 0.6891 | 16 | 0.0402 | 0.0342 | 0.02606 | 0.01050 | 0.01597 | 0.00224 |
| `checkpoint_0002000` | 0.6817 | 16 | 0.0360 | 0.0182 | 0.02549 | 0.00847 | 0.01907 | 0.00205 |
| `checkpoint_0003000` | 0.6809 | 16 | 0.0337 | 0.0171 | 0.02448 | 0.00838 | 0.02017 | 0.00214 |
| `checkpoint_0004000` | 0.6806 | 15 | 0.0347 | 0.0210 | 0.02448 | 0.00912 | 0.02460 | 0.00212 |
| `checkpoint_0005000` | 0.6785 | 15 | 0.0356 | 0.0228 | 0.02451 | 0.00977 | 0.01801 | 0.00136 |
| `checkpoint_0006000` | 0.6842 | 15 | 0.0333 | 0.0242 | 0.02385 | 0.01172 | 0.02127 | 0.00111 |
| `checkpoint_0007000` | 0.6831 | 15 | 0.0343 | 0.0261 | 0.02357 | 0.01400 | 0.03110 | 0.00146 |
| `checkpoint_0008000` | 0.6812 | 15 | 0.0372 | 0.0314 | 0.02243 | 0.02228 | 0.03917 | 0.00150 |
| `checkpoint_0009000` | 0.6787 | 15 | 0.0395 | 0.0353 | 0.02220 | 0.02242 | 0.04255 | 0.00144 |
| `checkpoint_0010000` | 0.6716 | 15 | 0.0441 | 0.0399 | 0.02206 | 0.02926 | 0.05821 | 0.00147 |
| `checkpoint_0011000` | 0.6665 | 15 | 0.0476 | 0.0438 | 0.02194 | 0.03064 | 0.05750 | 0.00155 |
| `checkpoint_0012000` | 0.6647 | 15 | 0.0508 | 0.0488 | 0.02165 | 0.03856 | 0.06498 | 0.00209 |
| `final` | 0.6648 | 15 | 0.0508 | 0.0488 | 0.02162 | 0.03806 | 0.06568 | 0.00212 |

## Read

- Final checkpoint: peak `0.6648 m/s`, time-to-peak step `15`, shape error `0.0488` ext-peak units.
- Best shape checkpoint: `checkpoint_0003000` with shape error `0.0171` ext-peak units.
- Recommendation: checkpoint_0003000 is materially closer in scaled velocity shape than final by 0.0317 ext-peak units while passing the leakage screen.

Outputs:
- JSON: `_artifacts/6c36536/diagnostics/delayed_peak_decay_delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/delayed_peak_decay_diagnostics.json`
- Arrays: `_artifacts/6c36536/diagnostics/delayed_peak_decay_delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/delayed_peak_decay_profiles.npz`
