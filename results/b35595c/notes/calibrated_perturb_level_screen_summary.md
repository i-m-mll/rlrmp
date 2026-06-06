# Calibrated Perturbation-Level Training Screen

Issue: `b35595c`

This screen trained six target-relative multi-target C&S GRU rows locally with
the full analytical Q/R/Q_f objective, batch size 64, clip 5, warmup plus cosine
learning-rate schedule, and validation-selected per-replicate checkpoint
evaluation. The crossed factors were learning rate (`1e-3`, `3e-3`) and
training perturbation level (`none`, calibrated `small`, calibrated `moderate`).
Calibrated perturbation rows used a per-trial 45% nominal, 45% single-family,
10% mild-combined mixture. The single-family set sampled initial-position,
initial-velocity, process-epsilon, command-input, sensory-feedback, and delayed-
observation perturbations with randomized component/axis, sign, and timing bin.

The first two rows were checkpoint-gated at 1000 batches, postrun diagnostics
were materialized there, and both rows were then resumed to 12000 batches after
the smoke diagnostics showed normal training and functioning standard
diagnostic hooks. The remaining four rows were trained directly to 12000
batches after that gate.

## Objective And Standard-Certificate Summary

All values below use validation-selected checkpoints. The deterministic split
ratio is the split-bank deterministic nominal full-Q/R/Q_f GRU/extLQG ratio.
The x0+eps ratio is the split-bank stress lens with shared initial-state and
process/load epsilon bank; it is a stress-test lens, not a checkpoint-selection
criterion.

| condition | val full-QRF | val/extLQG det | det split ratio | x0+eps split ratio | action mismatch | obs map | cov obs map | feedback ckpt diff mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none_lr1e-3 | 3996.1 | 0.915 | 1.063 | 6.78 | 1.817 | 1.684 | 1.684 | -9000 |
| none_lr3e-3 | 3813.6 | 0.873 | 1.018 | 17.15 | 1.817 | 1.939 | 1.892 | -1400 |
| cal_small_lr1e-3 | 3985.1 | 0.912 | 1.061 | 5.70 | 1.828 | 1.789 | 1.801 | -10200 |
| cal_small_lr3e-3 | 3795.6 | 0.869 | 1.015 | 24.16 | 1.825 | 2.020 | 2.051 | -5400 |
| cal_moderate_lr1e-3 | 3969.8 | 0.909 | 1.062 | 5.10 | 1.839 | 2.078 | 2.011 | -10200 |
| cal_moderate_lr3e-3 | 3790.4 | 0.868 | 1.011 | 15.38 | 1.826 | 2.174 | 2.292 | -9200 |

The deterministic nominal split lens is close to extLQG for all rows and is
closest for `lr=3e-3`. The validation scalar is below the deterministic extLQG
scalar in all rows, continuing the earlier warning that this scalar alone is not
a sufficient feedback-quality claim. The x0+epsilon split stress lens remains
substantially worse than extLQG, especially for `lr=3e-3` rows.

The standard action mismatch is almost unchanged across perturbation levels.
The observation-action map mismatch worsens as calibrated perturbation level
increases, especially at `lr=3e-3`. Calibrated perturbation training therefore
does not solve the map-agreement problem in this screen.

Feedback-selected checkpoint audit rows often selected much earlier checkpoints
than validation selection. This is audit-only, but it is informative: feedback
response quality can peak earlier than the nominal validation objective.

## Nominal Kinematics

All rows continued to perform ordinary nominal reaches. Values are means over
validation-selected evaluation rollouts.

| condition | endpoint mm | terminal speed mm/s | peak forward vel m/s | t peak s | overshoot mm | post-peak sign changes |
|---|---:|---:|---:|---:|---:|---:|
| none_lr1e-3 | 3.467 | 6.549 | 0.739 | 0.162 | 0.000 | 1.03 |
| none_lr3e-3 | 3.807 | 5.519 | 0.731 | 0.160 | 0.000 | 1.19 |
| cal_small_lr1e-3 | 3.451 | 6.384 | 0.740 | 0.162 | 0.000 | 1.11 |
| cal_small_lr3e-3 | 4.200 | 4.776 | 0.728 | 0.160 | 0.000 | 1.30 |
| cal_moderate_lr1e-3 | 3.474 | 6.162 | 0.740 | 0.161 | 0.000 | 1.22 |
| cal_moderate_lr3e-3 | 3.210 | 4.443 | 0.732 | 0.160 | 0.000 | 1.34 |

## Representative Perturbation-Response Metrics

All values are class-binned means over the standard perturbation bank with 64
rollout trials per replicate. `max dx` is reported in millimeters.

| condition | class | max dx mm | max du | endpoint delta mm | delta full-QRF cost | GRU/extLQG ratio |
|---|---|---:|---:|---:|---:|---:|
| none_lr1e-3 | initial position | 10.000 | 0.612 | 0.172 | 86.3 | 4.45 |
| cal_small_lr1e-3 | initial position | 10.000 | 0.648 | 0.142 | 77.9 | 4.02 |
| cal_moderate_lr1e-3 | initial position | 10.000 | 0.749 | 0.120 | 66.3 | 3.42 |
| none_lr3e-3 | initial position | 10.000 | 0.659 | 0.188 | 76.3 | 3.94 |
| cal_small_lr3e-3 | initial position | 10.000 | 0.667 | 0.197 | 66.1 | 3.41 |
| cal_moderate_lr3e-3 | initial position | 10.000 | 0.728 | 0.212 | 53.2 | 2.74 |
| none_lr1e-3 | force-state epsilon | 0.380 | 0.037 | 0.007 | 1.8 | 5.62 |
| cal_small_lr1e-3 | force-state epsilon | 0.370 | 0.038 | 0.007 | 1.7 | 5.31 |
| cal_moderate_lr1e-3 | force-state epsilon | 0.347 | 0.042 | 0.006 | 1.5 | 4.59 |
| none_lr3e-3 | force-state epsilon | 0.348 | 0.042 | 0.005 | 1.5 | 4.59 |
| cal_small_lr3e-3 | force-state epsilon | 0.328 | 0.049 | 0.004 | 1.2 | 3.87 |
| cal_moderate_lr3e-3 | force-state epsilon | 0.290 | 0.068 | 0.004 | 0.9 | 2.68 |
| none_lr1e-3 | sensory feedback | 1.623 | 0.655 | -0.803 | 30.7 | 1.45 |
| cal_small_lr1e-3 | sensory feedback | 1.302 | 0.302 | -0.120 | 7.9 | 0.37 |
| cal_moderate_lr1e-3 | sensory feedback | 1.277 | 0.346 | -0.162 | 8.8 | 0.41 |
| none_lr3e-3 | sensory feedback | 1.532 | 0.705 | -0.789 | 20.7 | 0.97 |
| cal_small_lr3e-3 | sensory feedback | 1.281 | 0.308 | -0.315 | 4.2 | 0.20 |
| cal_moderate_lr3e-3 | sensory feedback | 1.269 | 0.284 | -0.344 | 7.9 | 0.37 |
| none_lr1e-3 | delayed observation | 2.653 | 1.291 | -0.799 | 31.1 | 1.46 |
| cal_small_lr1e-3 | delayed observation | 2.763 | 1.376 | -0.882 | 35.0 | 1.65 |
| cal_moderate_lr1e-3 | delayed observation | 2.750 | 1.579 | -0.896 | 37.2 | 1.75 |
| none_lr3e-3 | delayed observation | 2.622 | 1.742 | -0.819 | 17.4 | 0.82 |
| cal_small_lr3e-3 | delayed observation | 2.720 | 1.750 | -0.904 | 10.2 | 0.48 |
| cal_moderate_lr3e-3 | delayed observation | 2.736 | 1.988 | -0.788 | 21.8 | 1.03 |

## Interpretation

Calibrated perturbation training did produce behaviorally visible changes in
several perturbation-response metrics. The cleanest pattern is that stronger
training perturbations reduce initial-position and force-state epsilon cost
ratios, with the strongest improvement in the `cal_moderate_lr3e-3` row. Sensory
feedback offsets improve sharply relative to the no-perturbation rows. Delayed
observation offsets are mixed: `lr=3e-3` improves under `cal_small`, but
`cal_moderate` partially gives that back; `lr=1e-3` does not improve delayed
observation costs.

The screen does not yet show the desired standard I/O-map generalization.
Indeed, the nominal observation-action map mismatch increases with perturbation
training level. This suggests that the current perturbation training mixture can
teach some local correction behavior without aligning the policy's observation-
to-action map with extLQG.

The most plausible next decision is therefore not simply "increase perturbation
level". A better next screen should separate feedback-quality gains from
robustness induction and map-alignment failures, likely by using the newly
materialized feedback checkpoint audit, perturbation-class metrics, and target-
relative map decomposition as explicit comparison lenses.

## Materialized Outputs

- Standard certificates:
  `results/b35595c/notes/gru_standard_certificates_calibrated_perturb_level_screen_validation_selected.md`
- Objective comparator:
  `results/b35595c/notes/objective_comparator_calibrated_perturb_level_screen_validation_selected.md`
- Perturbation-response bank:
  `results/b35595c/notes/gru_perturbation_response_calibrated_perturb_level_screen_validation_selected.md`
- Feedback ablation and feedback checkpoint audit:
  `results/b35595c/notes/gru_feedback_ablation_calibrated_perturb_level_screen_validation_selected.md`
- Map decomposition:
  `results/b35595c/notes/gru_map_error_decomposition_calibrated_perturb_level_screen_validation_selected.md`
- Evaluation diagnostics:
  `results/b35595c/notes/gru_evaluation_diagnostics_calibrated_perturb_level_screen_validation_selected.json`
- Velocity and loss figures:
  `_artifacts/b35595c/figures/gru_postrun_calibrated_perturb_level_screen_validation_selected/`
