# Baseline Feedback Controller Decision Lens

This note records the decision lens motivated by the calibrated perturbation-level
screen and the external review critique. It is not a replacement for the
standard certificate, and it is not a formal H-infinity robustness test. It is a
baseline-feedback lens for deciding whether a GRU is a useful non-robust
feedback-controller starting point before a separate robustification phase.

## Selection Status

The corrected packet reports both validation-selected and feedback-selected
checkpoints. Validation-selected checkpoints remain the primary interpretation
path for this screen. Feedback-selected checkpoints are audit-only because the
feedback score was defined after training and after seeing this diagnostic
family. A future use of feedback-aware checkpointing should predeclare the
feedback-validation score, select on a held-out feedback bank, and evaluate on a
separate perturbation bank.

## Pass-Criterion Lens

A baseline feedback GRU should be judged by whether it has:

- good nominal target-relative reaching;
- no obvious high-energy or robust-like nominal phenotype;
- clear feedback dependence under feedback ablation;
- family-balanced attenuation of small and moderate calibrated online
  perturbations;
- stable response curves without late oscillatory overcorrection;
- acceptable sensory-feedback and delayed-observation offset handling;
- initial-state results separated by information consistency rather than pooled
  into a single undifferentiated score.

It does not need to match the extLQG observation-action map. ExtLQG remains a
useful comparator for perturbation performance and response shape, but low
observation-action map mismatch is not the pass criterion for this baseline
track.

## Plot-Based Checks

The response-curve figures introduced after this screen should be read as a
shape diagnostic for this lens. In particular, they should show whether lower
aggregate cost comes from prompt correction, delayed correction, overshoot, or a
different control-energy tradeoff. The plots should therefore keep timing bins
visible instead of averaging early, mid, and late perturbations into one curve.

## Robustification Boundary

Robustification is a separate phase. A robustification screen should use stress
perturbations, CVaR or worst-bin objectives, and explicit robustness sidecars.
Its success should be judged relative to the baseline feedback controller by
stress attenuation, worst-case cost, induced-gain-like summaries, and any
nominal effort/kinematic tradeoff. Those are not required for this baseline
feedback pass lens.
