# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation_selected_per_replicate checkpoints for C&S GRU runs: cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64, cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64.

This is an objective-lens diagnostic, not a standard-certificate gate.

## Objective lenses

| lens | status | comparability |
|---|---|---|
| deterministic extLQG | available | deterministic full-Q/R/Q_f initial-state term; comparable only to full-Q/R/Q_f realized scalars |
| covariance-inclusive extLQG expected cost | available | not directly comparable to realized GRU validation scalars |
| realized GRU validation | available for full-Q/R/Q_f scalar rows | validation-selected audit metric, not checkpoint selection input |
| full same-noise-bank Monte Carlo | not_implemented | full shared sensory/command/motor noise is not exposed for both arms; see the partial shared-rollout comparator below |
| realized per-term full-Q/R/Q_f scoring | not_implemented | requires scorer output for running state, terminal, command, force/filter, and disturbance-integrator terms |
| shared-rollout comparator | available | shared initial-state and process/load epsilon bank; sensory/command noise limits declared |
| standard split-bank comparator | available | deterministic nominal, component-specific x0/process-epsilon, x0 position+velocity, and x0+epsilon audit-only lenses |

## extLQG decomposition

| component | value | lens |
|---|---:|---|
| deterministic initial-state term | 4368.5107 | comparable to realized/validation full-QRF values |
| initial covariance trace term | 7775.5302 | expected-cost sidecar only |
| accumulated noise scalar | 57.383523 | expected-cost sidecar only |
| total expected cost | 12201.424 | not directly comparable to GRU validation values |
| x0-only realized sanity | not_applicable | realized extLQG x0-only cost vs deterministic + initial-covariance-trace expectation |

## GRU comparison

| run | row comparability | mean selected validation | deterministic extLQG | selected/deterministic | total expected cost | selected/total | per-term scoring |
|---|---|---:|---:|---:|---:|---:|---|
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4040.7949 | 4368.5107 | 0.92498227 | 12201.424 | 0.33117403 | not_implemented |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4081.411 | 4368.5107 | 0.93427973 | 12201.424 | 0.33450283 | not_implemented |

## Caveats

- `selected/total` is retained only as a labeled non-apples-to-apples diagnostic for continuity with the provisional sidecar.
- The partial x0+epsilon shared-rollout comparator is stress-test-only; expected-cost wording is allowed only when an extLQG x0-only sanity check passes. Current status: `not_applicable`.
- The apples-to-apples scalar for the available GRU validation records is restricted to rows whose run spec declares the full analytical Q/R/Q_f objective; the deterministic extLQG term is not interchangeable with the covariance-inclusive expected cost.
- This sidecar is diagnostic only and is not a standard-certificate gate.
- GRU values are validation-selected realized full-QRF scalars; the shared-rollout and split-bank blocks are audit-only post-hoc rescores and are not used for checkpoint selection.
- The x0+epsilon shared-rollout block is stress-test-only unless the extLQG x0-only sanity check supports expected-cost wording.
- Split-bank GRU hidden states are initialized from the checkpoint model default rather than conditioned on the perturbed x0, so x0 lenses are recovery stress tests rather than expected-cost comparisons.

Full same-noise-bank Monte Carlo: `not_implemented` - full shared sensory/command/motor noise is not exposed for both arms. Partial shared-rollout replacement: `available_with_limitations` - shared-rollout comparator materialized common random inputs for initial state and process/load epsilon; sensory and command/motor noise are explicitly not shared under the current GRU graph contract

Per-term realized scoring: `not_implemented` - validation checkpoint manifests currently expose scalar full-QRF objectives, not running-state, terminal-state, command, force/filter, and disturbance-integrator contributions

## Shared-rollout comparator

Bank `cs_lss_shared_x0_epsilon_v1` uses 32 trials, seed `20260603`, shared initial states, and shared process/load epsilon.

Limitation: This is a shared initial-state plus process/load epsilon comparator. Sensory and command/motor noise are explicitly not claimed as shared.

| run | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | 40472.132 | 11090.472 | 3.6492704 | 4.9475972 | 4.8927324 | 2.5039959 | 2.7319964 | 1 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | 50457.034 | 11090.472 | 4.5495839 | 7.978536 | 30.517787 | 1.5790386 | 1.7451718 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `deterministic_nominal` | 4741.6384 | 4363.51 | 1.0866569 | 0.91094593 | 0.41510355 | 1.2447173 | 1.2130507 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `x0_position_only` | 40155.837 | 11027.369 | 3.6414702 | 4.8971432 | 2.8910406 | 2.5261923 | 2.765013 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `x0_velocity_only` | 5047.8943 | 4399.7482 | 1.1473144 | 1.0016749 | 0.41529072 | 1.2796255 | 1.2440713 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4744.7458 | 4366.7422 | 1.0865642 | 0.91245924 | 0.41498855 | 1.242971 | 1.2121102 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4996.8479 | 4374.524 | 1.1422609 | 1.0219633 | 5.8057643 | 1.2367911 | 1.2047982 | 1 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4741.6384 | 4363.51 | 1.0866569 | 0.91094593 | 0.41510355 | 1.2447173 | 1.2130507 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4741.6384 | 4363.51 | 1.0866569 | 0.91094593 | 0.41510355 | 1.2447173 | 1.2130507 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4742.2592 | 4363.7822 | 1.0867314 | 0.91097286 | 0.42596689 | 1.2447503 | 1.2131149 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4764.1062 | 4364.8752 | 1.0914645 | 0.91969105 | 0.8629437 | 1.2447991 | 1.2132507 | 1 |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `x0_position_velocity` | 40323.638 | 11076.817 | 3.6403634 | 4.9245299 | 2.8980191 | 2.5115528 | 2.7406756 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__effective_020a65b_pgd_radius_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 40472.132 | 11090.472 | 3.6492704 | 4.9475972 | 4.8927324 | 2.5039959 | 2.7319964 | 1 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `deterministic_nominal` | 4601.856 | 4363.51 | 1.0546225 | 0.93759837 | 0.39490082 | 1.1537529 | 1.1560365 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `x0_position_only` | 49606.179 | 11027.369 | 4.49846 | 7.851112 | 27.957854 | 1.5753672 | 1.7441024 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `x0_velocity_only` | 4981.603 | 4399.7482 | 1.1322473 | 1.0617087 | 0.68515232 | 1.1918953 | 1.1908369 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4605.0718 | 4366.7422 | 1.0545784 | 0.93869104 | 0.39698021 | 1.1526334 | 1.1551611 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4845.6771 | 4374.524 | 1.1077039 | 1.043225 | 5.4480647 | 1.1469466 | 1.1486354 | 1 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4601.856 | 4363.51 | 1.0546225 | 0.93759837 | 0.39490082 | 1.1537529 | 1.1560365 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4601.856 | 4363.51 | 1.0546225 | 0.93759837 | 0.39490082 | 1.1537529 | 1.1560365 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4602.2984 | 4363.7822 | 1.0546581 | 0.9375788 | 0.40330692 | 1.1537651 | 1.1560833 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4625.0881 | 4364.8752 | 1.0596152 | 0.94671985 | 0.84606596 | 1.1538459 | 1.1562422 | 1 |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `x0_position_velocity` | 50163.895 | 11076.817 | 4.5287284 | 7.9358041 | 27.741212 | 1.5818257 | 1.7486945 | not_comparable |
| `cs_gru_h0_sisu_spectrum_targetfix__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 50457.034 | 11090.472 | 4.5495839 | 7.978536 | 30.517787 | 1.5790386 | 1.7451718 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
