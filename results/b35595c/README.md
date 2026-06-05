This issue records the calibrated perturbation-level target-relative multi-target
C&S GRU screen. The screen compares no perturbation training, calibrated small
perturbation training, and calibrated moderate perturbation training at peak
learning rates 1e-3 and 3e-3, using the full analytical Q/R/Q_f objective,
batch size 64, gradient clip 5, warmup+cosine learning rate schedule, and five
replicates per row. Bulk checkpoints and figures live under `_artifacts/b35595c`;
tracked run specs, notes, and reviewer-facing summaries live here.
