# Calibrated Perturbation-Level Multi-Target Screen

## Rows

Shared settings:
- `--target-relative-multitarget`
- `--loss-objective full_analytical_qrf`
- `--batch-size 64`
- `--gradient-clip-norm 5`
- `--lr-warmup-batches 500`
- `--lr-warmup-init-fraction 0.1`
- `--lr-cosine-alpha 0.01`
- `--n-replicates 5`
- `--n-train-batches 12000`

Training conditions:
- `none`: no `--perturbation-training`.
- `cal_small`: `--perturbation-training --perturbation-calibrated-timing --perturbation-physical-level small`.
- `cal_moderate`: `--perturbation-training --perturbation-calibrated-timing --perturbation-physical-level moderate`.

Learning rates:
- `--controller-lr 0.001`
- `--controller-lr 0.003`

## Checkpoint Gate

The first two rows exercise both code paths and both learning rates:
- `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64`
- `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64`

They are launched with `--stop-after-batches 1000` and
`--checkpoint-interval-batches 1000`, preserving the 12000-batch run contract so
the same output directories can resume cleanly.

## Diagnostics

Primary checkpoint selection remains validation-selected per replicate.
Analytical action/I/O/Jacobian-like metrics, perturbation response, feedback
ablation, and feedback-selected checkpoints are audit-only sidecars.

After completion, materialize the full standard post-run bundle:
- standard certificate
- objective comparator and split-stress comparator
- map-error decomposition
- perturbation-response bank
- feedback ablation and feedback-selected checkpoint audit
- evaluation diagnostics
- loss and velocity profile figures

The final deliverable is a standalone review packet with relevant issue copies,
code copies, run specs, tracked diagnostics, and interpretation.
