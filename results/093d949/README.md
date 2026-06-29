# Soft lambda estimator and direct-epsilon sweep

This issue-local experiment repairs the frozen soft-adversary lambda calibration from the c92 open-loop no-PGD substrates by comparing the old batch-mean estimator, a batch-corrected gradient-pressure comparison, and a per-trial p90 estimator. The tracked outputs summarize whether a narrow direct-epsilon soft-energy sweep centered on the per-trial estimate predicts the interior-to-cap transition without launching new training.
