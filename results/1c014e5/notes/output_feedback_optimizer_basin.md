# Output-Feedback Optimizer-Basin Diagnostic

Issue: `1c014e5`. Interpolated-start issue:
`7cea1b7`. Source issue: `7a459bb`.
Umbrella: `43e8728`.

This diagnostic asks whether the failed free time-varying output-feedback row is
mainly an optimizer-family or basin-access failure. It does not change the task,
the reference controller, or the bridge gate. It screens full-batch AdamW,
polishes the best AdamW starts with L-BFGS-B, repeats the same optimizer-family
check from the Bellman start, and tests K_alpha starts between the failed
scratch controller and the analytical LQR controller.

Runtime: `1386.10` seconds.

Grid: `{"adamw_clip_norm": 10000.0, "adamw_lrs": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03], "adamw_steps": 5000, "alpha_adamw_steps": 3000, "alpha_lbfgs_maxiter": 2000, "best_lrs_selected_for_followup": [0.01, 0.003], "interpolated_alphas": [0.1, 0.25, 0.5, 0.75], "polish_maxiter": 1000}`.

## Verdict

At least one bounded optimizer-basin row reaches the practical bridge target: `adamw_bellman_polish_lr_0p01__bellman_init`.

## Best Rows

| group | label | objective ratio | clean mismatch | exact L2 ratio | lambda/gamma^2 |
|---|---|---:|---:|---:|---:|
| adamw_scratch | adamw_fixed_lr_0p01__scratch | 1.0079189 | 0.03066474 | 1.1616916 | 2.0434034 |
| adamw_followup | adamw_bellman_polish_lr_0p01__bellman_init | 1 | 6.730258e-06 | 0.99999156 | 1.5551161 |
| k_alpha_lbfgs | strong_optimizer_whitened__k_alpha_0p75 | 1.0000684 | 0.0005426134 | 1.0096479 | 1.5923224 |
| k_alpha_adamw_polish | k_alpha_adamw_polish_lr_0p01__k_alpha_0p75 | 1.0000714 | 0.00091320451 | 1.0313292 | 1.6991765 |

## Failure Labels

`{'mixed': 12, 'not_failure': 2, 'optimizer_basin': 26}`
