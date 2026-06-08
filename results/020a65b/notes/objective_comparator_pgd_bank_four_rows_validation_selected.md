# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64.

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
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 5239.2192 | 4368.5107 | 1.1993147 | 12201.424 | 0.42939406 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4747.3943 | 4368.5107 | 1.0867306 | 12201.424 | 0.38908526 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 5070.065 | 4368.5107 | 1.1605935 | 12201.424 | 0.41553058 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4048.7078 | 4368.5107 | 0.92679361 | 12201.424 | 0.33182255 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | 72134.744 | 11090.472 | 6.5042085 | 12.383316 | 154.22735 | 1.1369247 | 1.2344218 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 59651.339 | 11090.472 | 5.3786113 | 9.5950651 | 118.13654 | 1.5493232 | 1.4766629 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | 184789.51 | 11090.472 | 16.662006 | 32.030567 | 639.79801 | 1.7739731 | 2.3811914 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 42593.997 | 11090.472 | 3.8405936 | 3.9846501 | 7.5351835 | 3.7277923 | 3.667934 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `deterministic_nominal` | 4621.7193 | 4363.51 | 1.0591747 | 1.1125341 | 1.6737121 | 1.0042375 | 1.0330761 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `x0_position_only` | 70771.125 | 11027.369 | 6.4177705 | 12.162809 | 148.72389 | 1.1340189 | 1.2353283 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `x0_velocity_only` | 5782.49 | 4399.7482 | 1.3142775 | 1.6667557 | 11.038104 | 0.99284821 | 0.99634605 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4625.0582 | 4366.7422 | 1.0591553 | 1.1107188 | 1.6690793 | 1.0059336 | 1.0341723 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5050.6024 | 4374.524 | 1.154549 | 1.3033097 | 8.4886821 | 0.9987551 | 1.0273105 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4621.7193 | 4363.51 | 1.0591747 | 1.1125341 | 1.6737121 | 1.0042375 | 1.0330761 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4621.7193 | 4363.51 | 1.0591747 | 1.1125341 | 1.6737121 | 1.0042375 | 1.0330761 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4621.5004 | 4363.7822 | 1.0590584 | 1.1123089 | 1.6687068 | 1.0042438 | 1.0331161 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4647.1737 | 4364.8752 | 1.0646751 | 1.1230761 | 2.0630538 | 1.004212 | 1.0331571 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `x0_position_velocity` | 71565.404 | 11076.817 | 6.4608277 | 12.293123 | 149.70797 | 1.1384959 | 1.2367578 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 72134.744 | 11090.472 | 6.5042085 | 12.383316 | 154.22735 | 1.1369247 | 1.2344218 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `deterministic_nominal` | 4412.0168 | 4363.51 | 1.0111164 | 1.0281068 | 0.92567535 | 0.99376002 | 1.0055842 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_position_only` | 57490.882 | 11027.369 | 5.2134721 | 9.2214071 | 110.02061 | 1.5482899 | 1.4860922 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_velocity_only` | 5016.9293 | 4399.7482 | 1.1402765 | 1.2629088 | 5.0731728 | 1.0370883 | 1.002787 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4415.5867 | 4366.7422 | 1.0111856 | 1.0279943 | 0.92220478 | 0.99416034 | 1.0054387 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4780.7851 | 4374.524 | 1.0928698 | 1.1920019 | 6.9295123 | 0.98808082 | 0.99990099 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4412.0168 | 4363.51 | 1.0111164 | 1.0281068 | 0.92567535 | 0.99376002 | 1.0055842 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4412.0168 | 4363.51 | 1.0111164 | 1.0281068 | 0.92567535 | 0.99376002 | 1.0055842 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4412.563 | 4363.7822 | 1.0111786 | 1.0282353 | 0.93163856 | 0.99374338 | 1.0055771 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4437.3936 | 4364.8752 | 1.0166141 | 1.0386766 | 1.3148717 | 0.99374497 | 1.0056583 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_position_velocity` | 59140.022 | 11076.817 | 5.339081 | 9.5125245 | 114.23516 | 1.5517137 | 1.4793626 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 59651.339 | 11090.472 | 5.3786113 | 9.5950651 | 118.13654 | 1.5493232 | 1.4766629 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `deterministic_nominal` | 5574.3632 | 4363.51 | 1.2774952 | 1.331861 | 1.2586928 | 1.215461 | 1.2727339 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `x0_position_only` | 175281.11 | 11027.369 | 15.895098 | 30.426766 | 572.64935 | 1.7912257 | 2.4024781 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `x0_velocity_only` | 8733.1617 | 4399.7482 | 1.9849231 | 2.7110393 | 22.25186 | 1.307675 | 1.3638811 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `x0_force_filter_only` | 5580.4267 | 4366.7422 | 1.2779382 | 1.332395 | 1.2714369 | 1.2155988 | 1.2736301 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 6057.5442 | 4374.524 | 1.3847322 | 1.5406276 | 9.990289 | 1.2113233 | 1.2663308 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 5574.3632 | 4363.51 | 1.2774952 | 1.331861 | 1.2586928 | 1.215461 | 1.2727339 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 5574.3632 | 4363.51 | 1.2774952 | 1.331861 | 1.2586928 | 1.215461 | 1.2727339 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 5575.7748 | 4363.7822 | 1.277739 | 1.3322817 | 1.2732169 | 1.2155019 | 1.2728197 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 5613.5983 | 4364.8752 | 1.2860845 | 1.3475122 | 2.0057743 | 1.2157231 | 1.2730686 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `x0_position_velocity` | 183788.06 | 11076.817 | 16.592137 | 31.883622 | 628.34032 | 1.7764733 | 2.384907 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 184789.51 | 11090.472 | 16.662006 | 32.030567 | 639.79801 | 1.7739731 | 2.3811914 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `deterministic_nominal` | 4743.8732 | 4363.51 | 1.0871691 | 0.94480197 | 0.39826681 | 1.2119614 | 1.1991173 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_position_only` | 41909.603 | 11027.369 | 3.8005078 | 3.9117822 | 5.3914051 | 3.7180279 | 3.6648552 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_velocity_only` | 5027.0541 | 4399.7482 | 1.1425777 | 1.0126832 | 0.4078972 | 1.2614675 | 1.2276326 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4746.9347 | 4366.7422 | 1.0870655 | 0.94570388 | 0.3981357 | 1.2108959 | 1.198262 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4971.7643 | 4374.524 | 1.1365269 | 1.0446256 | 5.0484555 | 1.2043198 | 1.1912313 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4743.8732 | 4363.51 | 1.0871691 | 0.94480197 | 0.39826681 | 1.2119614 | 1.1991173 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4743.8732 | 4363.51 | 1.0871691 | 0.94480197 | 0.39826681 | 1.2119614 | 1.1991173 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4744.2056 | 4363.7822 | 1.0871775 | 0.94475227 | 0.40844301 | 1.2119413 | 1.1991275 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4764.0492 | 4364.8752 | 1.0914514 | 0.95257926 | 0.78944362 | 1.2120747 | 1.1993321 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_position_velocity` | 42462.331 | 11076.817 | 3.8334417 | 3.9656176 | 5.7109984 | 3.7352809 | 3.6757095 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 42593.997 | 11090.472 | 3.8405936 | 3.9846501 | 7.5351835 | 3.7277923 | 3.667934 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
