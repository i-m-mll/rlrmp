# Diagnostic: Combo Cell Hold-Period Motor Outputs

## Purpose

Is the 4.6 mm residual hold-period drift in the combo cell
(`gru__jerk_motor_smooth_combo`) driven by non-zero motor commands
during the hold epoch, or by residual plant mechanics?

See `_artifacts/scratchpad/residual_anticipation_proposal.md` §5 for framing.

## Method

- Loaded adversarial_model.eqx for combo cell (5 replicates).
- Ran inference on 8 validation trials at SISU=0.5, pert_scale=0.
- Extracted `states.efferent.output` (motor commands) per step.
- Computed ‖motor‖ = L2 norm of 2D force command per step.
- Hold epoch = steps before go cue; movement epoch = steps from go cue onward.

## Results

| Metric | Value |
|--------|-------|
| Mean ‖motor‖ during hold (all reps, all trials) | 0.8160 N |
| Mean max ‖motor‖ during hold (all reps, all trials) | 1.2694 N |
| Mean ‖motor‖ during movement (all reps, all trials) | 4.5875 N |
| Hold / movement mean ratio | 0.1769 (17.7%) |

Per-replicate (mean over 8 trials):

| Replicate | Hold mean (N) | Move mean (N) | Ratio |
|-----------|--------------|--------------|-------|
| Rep 0 | 0.7985 | 4.5562 | 0.1744 (17.4%) |
| Rep 1 | 0.8692 | 4.6363 | 0.1863 (18.6%) |
| Rep 2 | 0.7982 | 4.5942 | 0.1730 (17.3%) |
| Rep 3 | 0.7941 | 4.5822 | 0.1722 (17.2%) |
| Rep 4 | 0.8201 | 4.5686 | 0.1787 (17.9%) |

## Verdict

**MOTOR COMMANDS NON-ZERO DURING HOLD.** Hold-period motor magnitude is 17.7% of movement-period magnitude (mean ‖motor‖ during hold = 0.8160 N, max = 1.2694 N). Drift is likely motor-command-driven — the nn_output_pre_go penalty is reducing but not eliminating pre-go motor activity. Increasing nn_output_pre_go weight is the targeted lever.

## Figure

See `figures/combo_hold_motor_diagnostic/figure.html` for per-trial time traces.
