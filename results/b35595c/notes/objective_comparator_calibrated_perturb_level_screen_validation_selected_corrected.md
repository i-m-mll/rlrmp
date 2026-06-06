# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64.

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
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3969.8067 | 4368.5107 | 0.9087323 | 12201.424 | 0.32535601 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3790.4094 | 4368.5107 | 0.86766628 | 12201.424 | 0.31065303 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3985.0756 | 4368.5107 | 0.91222751 | 12201.424 | 0.32660741 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3795.6072 | 4368.5107 | 0.86885612 | 12201.424 | 0.31107903 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3996.1287 | 4368.5107 | 0.91475768 | 12201.424 | 0.32751329 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3813.6233 | 4368.5107 | 0.87298021 | 12201.424 | 0.31255559 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 56523.899 | 11090.472 | 5.0966179 | 7.9677107 | 124.20934 | 2.2554061 | 2.5454815 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 170592.97 | 11090.472 | 15.381939 | 28.439789 | 556.06073 | 2.869407 | 2.7949587 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 63193.504 | 11090.472 | 5.6979994 | 9.3792944 | 133.96989 | 2.1433848 | 2.4789329 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 267966.21 | 11090.472 | 24.16184 | 41.976642 | 1911.5972 | 3.3483154 | 3.5249717 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 75230.515 | 11090.472 | 6.7833464 | 11.505984 | 182.7532 | 2.2058497 | 2.5711277 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 190164.33 | 11090.472 | 17.14664 | 29.663262 | 1197.2752 | 2.9739397 | 3.1464059 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `deterministic_nominal` | 4632.4548 | 4363.51 | 1.061635 | 1.1140272 | 1.601218 | 1.0089868 | 1.0335789 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_position_only` | 53850.592 | 11027.369 | 4.883358 | 7.514533 | 113.16769 | 2.2501527 | 2.550423 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_velocity_only` | 5197.5309 | 4399.7482 | 1.1813246 | 1.3178701 | 3.681009 | 1.0595225 | 1.0654128 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4638.1232 | 4366.7422 | 1.0621472 | 1.1167537 | 1.622347 | 1.0075204 | 1.0324255 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4981.0391 | 4374.524 | 1.1386471 | 1.2684554 | 7.2442967 | 1.0037629 | 1.0283091 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4632.4548 | 4363.51 | 1.061635 | 1.1140272 | 1.601218 | 1.0089868 | 1.0335789 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4632.4548 | 4363.51 | 1.061635 | 1.1140272 | 1.601218 | 1.0089868 | 1.0335789 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4633.0355 | 4363.7822 | 1.0617018 | 1.1142084 | 1.5824941 | 1.0090266 | 1.0336695 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4657.5587 | 4364.8752 | 1.0670543 | 1.1243858 | 1.9954637 | 1.008954 | 1.0336488 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_position_velocity` | 56475.573 | 11076.817 | 5.0985382 | 7.9598221 | 123.14022 | 2.2628663 | 2.5544432 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 56523.899 | 11090.472 | 5.0966179 | 7.9677107 | 124.20934 | 2.2554061 | 2.5454815 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `deterministic_nominal` | 4412.2597 | 4363.51 | 1.0111721 | 1.0220383 | 1.2951202 | 1.0006586 | 1.0024672 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_position_only` | 179134.92 | 11027.369 | 16.244574 | 29.684039 | 710.48559 | 2.7831743 | 2.7957706 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_velocity_only` | 5522.7992 | 4399.7482 | 1.2552535 | 1.3806753 | 6.6031938 | 1.1650867 | 1.0630485 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4416.5299 | 4366.7422 | 1.0114016 | 1.0241631 | 1.3077347 | 0.9991865 | 1.0012882 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4678.6485 | 4374.524 | 1.0695217 | 1.1406556 | 5.5735254 | 0.99526601 | 0.99704664 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4412.2597 | 4363.51 | 1.0111721 | 1.0220383 | 1.2951202 | 1.0006586 | 1.0024672 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4412.2597 | 4363.51 | 1.0111721 | 1.0220383 | 1.2951202 | 1.0006586 | 1.0024672 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4412.9854 | 4363.7822 | 1.0112754 | 1.0221817 | 1.2907736 | 1.000738 | 1.0025995 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4428.9097 | 4364.8752 | 1.0146704 | 1.0286994 | 1.5340912 | 1.000701 | 1.0026176 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_position_velocity` | 171026.54 | 11076.817 | 15.440044 | 28.527442 | 555.85398 | 2.8814632 | 2.8010516 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 170592.97 | 11090.472 | 15.381939 | 28.439789 | 556.06073 | 2.869407 | 2.7949587 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `deterministic_nominal` | 4629.1978 | 4363.51 | 1.0608885 | 1.1174278 | 1.5566951 | 1.0045422 | 1.0304306 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_position_only` | 61057.221 | 11027.369 | 5.5368801 | 9.0052398 | 127.06424 | 2.1490834 | 2.494329 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_velocity_only` | 5236.3059 | 4399.7482 | 1.1901376 | 1.360973 | 3.6723008 | 1.0382911 | 1.0508386 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4635.7405 | 4366.7422 | 1.0616016 | 1.1206773 | 1.5871618 | 1.0029969 | 1.029141 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5039.7766 | 4374.524 | 1.1520743 | 1.2994691 | 8.1683623 | 0.99927518 | 1.0249797 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4629.1978 | 4363.51 | 1.0608885 | 1.1174278 | 1.5566951 | 1.0045422 | 1.0304306 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4629.1978 | 4363.51 | 1.0608885 | 1.1174278 | 1.5566951 | 1.0045422 | 1.0304306 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4629.9401 | 4363.7822 | 1.0609925 | 1.1176983 | 1.5364044 | 1.0045782 | 1.0305213 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4657.6074 | 4364.8752 | 1.0670654 | 1.129247 | 2.0048976 | 1.0045066 | 1.0305015 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_position_velocity` | 63073.525 | 11076.817 | 5.6941924 | 9.3632442 | 131.64849 | 2.1502918 | 2.4876626 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 63193.504 | 11090.472 | 5.6979994 | 9.3792944 | 133.96989 | 2.1433848 | 2.4789329 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `deterministic_nominal` | 4427.596 | 4363.51 | 1.0146868 | 1.0412689 | 2.1335257 | 0.98755124 | 0.99213497 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_position_only` | 262094.89 | 11027.369 | 23.767672 | 41.415014 | 1784.8595 | 3.2986661 | 3.4878446 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_velocity_only` | 5276.0876 | 4399.7482 | 1.1991794 | 1.318875 | 5.589138 | 1.0991883 | 1.0572197 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4432.7286 | 4366.7422 | 1.0151111 | 1.043822 | 2.1538096 | 0.98607222 | 0.9909174 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4752.449 | 4374.524 | 1.0863923 | 1.1858802 | 7.2796071 | 0.98227765 | 0.98681004 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4427.596 | 4363.51 | 1.0146868 | 1.0412689 | 2.1335257 | 0.98755124 | 0.99213497 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4427.596 | 4363.51 | 1.0146868 | 1.0412689 | 2.1335257 | 0.98755124 | 0.99213497 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4427.4725 | 4363.7822 | 1.0145952 | 1.0410777 | 2.1206611 | 0.98760911 | 0.99224804 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4447.8003 | 4364.8752 | 1.0189983 | 1.0495319 | 2.4207748 | 0.98755128 | 0.99225152 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_position_velocity` | 268278.34 | 11076.817 | 24.219805 | 42.096608 | 1888.7929 | 3.3626292 | 3.5407622 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 267966.21 | 11090.472 | 24.16184 | 41.976642 | 1911.5972 | 3.3483154 | 3.5249717 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `deterministic_nominal` | 4637.9806 | 4363.51 | 1.0629013 | 1.1240854 | 1.6419418 | 1.0036341 | 1.0253291 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_position_only` | 71650.592 | 11027.369 | 6.4975236 | 10.883877 | 170.41954 | 2.1992241 | 2.5701171 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_velocity_only` | 5333.1291 | 4399.7482 | 1.2121442 | 1.4127532 | 4.3603348 | 1.0346677 | 1.044007 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4644.2771 | 4366.7422 | 1.0635565 | 1.1272217 | 1.6712612 | 1.0020819 | 1.0240327 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5085.9811 | 4374.524 | 1.1626365 | 1.3230181 | 8.7332404 | 0.99847414 | 1.01995 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4637.9806 | 4363.51 | 1.0629013 | 1.1240854 | 1.6419418 | 1.0036341 | 1.0253291 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4637.9806 | 4363.51 | 1.0629013 | 1.1240854 | 1.6419418 | 1.0036341 | 1.0253291 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4639.682 | 4363.7822 | 1.0632249 | 1.1247826 | 1.6364057 | 1.0036705 | 1.0254319 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4667.4507 | 4364.8752 | 1.0693206 | 1.1365911 | 2.0449267 | 1.0036158 | 1.0254139 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_position_velocity` | 75191.986 | 11076.817 | 6.7882306 | 11.50553 | 180.3129 | 2.2138754 | 2.5817471 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 75230.515 | 11090.472 | 6.7833464 | 11.505984 | 182.7532 | 2.2058497 | 2.5711277 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `deterministic_nominal` | 4440.0954 | 4363.51 | 1.0175513 | 1.0463014 | 1.9756205 | 0.98796107 | 0.99653275 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_position_only` | 151645.35 | 11027.369 | 13.751725 | 23.735655 | 805.77017 | 2.8121907 | 2.9838071 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_velocity_only` | 5154.5797 | 4399.7482 | 1.1715624 | 1.2991622 | 4.9395423 | 1.0626218 | 1.0358382 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4444.7732 | 4366.7422 | 1.0178694 | 1.0486513 | 1.9923544 | 0.9864766 | 0.99528415 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4788.1145 | 4374.524 | 1.0945453 | 1.2011082 | 7.443843 | 0.9829012 | 0.99133782 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4440.0954 | 4363.51 | 1.0175513 | 1.0463014 | 1.9756205 | 0.98796107 | 0.99653275 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4440.0954 | 4363.51 | 1.0175513 | 1.0463014 | 1.9756205 | 0.98796107 | 0.99653275 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4441.3275 | 4363.7822 | 1.0177702 | 1.0467494 | 1.9729276 | 0.98801443 | 0.99665378 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4462.2581 | 4364.8752 | 1.0223106 | 1.0555384 | 2.2668614 | 0.98796812 | 0.99664714 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_position_velocity` | 191384.7 | 11076.817 | 17.277951 | 29.872809 | 1200.4128 | 2.9887054 | 3.1618351 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 190164.33 | 11090.472 | 17.14664 | 29.663262 | 1197.2752 | 2.9739397 | 3.1464059 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
