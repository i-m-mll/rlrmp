# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64.

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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3963.3188 | 4368.5107 | 0.90724713 | 12201.424 | 0.32482427 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3972.5232 | 4368.5107 | 0.90935412 | 12201.424 | 0.32557864 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4025.9989 | 4368.5107 | 0.92159531 | 12201.424 | 0.32996139 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4008.3379 | 4368.5107 | 0.91755251 | 12201.424 | 0.32851393 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3965.2718 | 4368.5107 | 0.90769419 | 12201.424 | 0.32498433 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 3974.9052 | 4368.5107 | 0.9098994 | 12201.424 | 0.32577387 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4027.2288 | 4368.5107 | 0.92187684 | 12201.424 | 0.33006219 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4004.7017 | 4368.5107 | 0.91672014 | 12201.424 | 0.32821592 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4761.6543 | 4368.5107 | 1.0899949 | 12201.424 | 0.39025397 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 5572.942 | 4368.5107 | 1.2757076 | 12201.424 | 0.4567452 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4031.247 | 4368.5107 | 0.92279665 | 12201.424 | 0.33039151 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 5606.7744 | 4368.5107 | 1.2834521 | 12201.424 | 0.45951802 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | 61968.119 | 11090.472 | 5.5875095 | 8.8267724 | 137.16503 | 2.3877437 | 2.7258125 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | 62225.662 | 11090.472 | 5.6107315 | 9.2216993 | 124.30005 | 2.1443682 | 2.4816311 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | 73203.566 | 11090.472 | 6.6005816 | 11.240216 | 228.91715 | 2.0220623 | 2.1207495 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | 63436.297 | 11090.472 | 5.7198914 | 9.5476542 | 138.01602 | 2.0425163 | 2.3391399 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | 64460.807 | 11090.472 | 5.812269 | 9.2033604 | 151.26938 | 2.4385031 | 2.7905681 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | 58566.895 | 11090.472 | 5.2808297 | 8.605261 | 110.99039 | 2.0961934 | 2.4227421 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 78458.383 | 11090.472 | 7.0743952 | 11.97031 | 283.66587 | 2.112708 | 2.2050225 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | 70196.276 | 11090.472 | 6.3294218 | 10.682764 | 165.93814 | 2.1172753 | 2.4570015 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | 50063.103 | 11090.472 | 4.5140641 | 8.0889349 | 48.917504 | 1.3608912 | 1.5373876 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | 68556.481 | 11090.472 | 6.1815655 | 11.727084 | 128.12 | 1.1678625 | 1.2817167 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 43006.484 | 11090.472 | 3.8777866 | 6.1320489 | 18.738758 | 1.9323883 | 2.0405743 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | 84965.004 | 11090.472 | 7.6610808 | 14.772871 | 202.23698 | 1.1255031 | 1.2172548 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `deterministic_nominal` | 4631.6421 | 4363.51 | 1.0614487 | 1.115952 | 1.5163257 | 1.0072406 | 1.0320779 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `x0_position_only` | 58823.716 | 11027.369 | 5.3343381 | 8.2876035 | 125.78432 | 2.3791038 | 2.7271195 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `x0_velocity_only` | 5203.8161 | 4399.7482 | 1.1827532 | 1.3189402 | 3.6090736 | 1.0613191 | 1.067757 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4637.4208 | 4366.7422 | 1.0619864 | 1.1187153 | 1.5380922 | 1.0057885 | 1.0309402 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4981.7371 | 4374.524 | 1.1388067 | 1.2710252 | 7.1801568 | 1.0020518 | 1.0268434 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4631.6421 | 4363.51 | 1.0614487 | 1.115952 | 1.5163257 | 1.0072406 | 1.0320779 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4631.6421 | 4363.51 | 1.0614487 | 1.115952 | 1.5163257 | 1.0072406 | 1.0320779 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4632.3381 | 4363.7822 | 1.061542 | 1.1161909 | 1.4968288 | 1.0072806 | 1.0321681 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4657.3994 | 4364.8752 | 1.0670178 | 1.1266135 | 1.9188036 | 1.0072072 | 1.0321467 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `x0_position_velocity` | 61952.379 | 11076.817 | 5.5929768 | 8.8271711 | 135.66153 | 2.395539 | 2.7353122 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 61968.119 | 11090.472 | 5.5875095 | 8.8267724 | 137.16503 | 2.3877437 | 2.7258125 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `deterministic_nominal` | 4628.6484 | 4363.51 | 1.0607626 | 1.1198824 | 1.4993737 | 1.0024441 | 1.0283434 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `x0_position_only` | 59860.025 | 11027.369 | 5.4283142 | 8.8158626 | 116.32656 | 2.1423418 | 2.4873647 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `x0_velocity_only` | 5225.4603 | 4399.7482 | 1.1876726 | 1.3557639 | 3.5358174 | 1.0385257 | 1.0510075 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4635.2898 | 4366.7422 | 1.0614984 | 1.1231584 | 1.5311481 | 1.0009136 | 1.0270738 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5041.0427 | 4374.524 | 1.1523637 | 1.3027842 | 8.1133333 | 0.99721242 | 1.0229573 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4628.6484 | 4363.51 | 1.0607626 | 1.1198824 | 1.4993737 | 1.0024441 | 1.0283434 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4628.6484 | 4363.51 | 1.0607626 | 1.1198824 | 1.4993737 | 1.0024441 | 1.0283434 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4629.7147 | 4363.7822 | 1.0609408 | 1.1203045 | 1.4814046 | 1.0024806 | 1.0284352 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4657.677 | 4364.8752 | 1.0670814 | 1.1319969 | 1.952474 | 1.0024095 | 1.0284143 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `x0_position_velocity` | 62091.977 | 11076.817 | 5.6055795 | 9.2029311 | 121.89103 | 2.1516127 | 2.4908589 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 62225.662 | 11090.472 | 5.6107315 | 9.2216993 | 124.30005 | 2.1443682 | 2.4816311 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `deterministic_nominal` | 4668.0696 | 4363.51 | 1.0697969 | 1.1331444 | 1.9707683 | 1.0096565 | 1.0245986 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `x0_position_only` | 68794.365 | 11027.369 | 6.238511 | 10.480445 | 206.53973 | 2.0150372 | 2.1402162 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `x0_velocity_only` | 5823.8356 | 4399.7482 | 1.3236748 | 1.5557871 | 9.4192948 | 1.1286021 | 1.0557668 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4672.7029 | 4366.7422 | 1.0700661 | 1.1350279 | 1.9828293 | 1.0084459 | 1.0238148 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4869.0344 | 4374.524 | 1.1130433 | 1.2223546 | 5.1053631 | 1.0048017 | 1.0198764 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4668.0696 | 4363.51 | 1.0697969 | 1.1331444 | 1.9707683 | 1.0096565 | 1.0245986 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4668.0696 | 4363.51 | 1.0697969 | 1.1331444 | 1.9707683 | 1.0096565 | 1.0245986 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4668.3537 | 4363.7822 | 1.0697953 | 1.1330991 | 1.968486 | 1.0097238 | 1.0247057 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4681.1391 | 4364.8752 | 1.0724566 | 1.1382708 | 2.1369024 | 1.0096487 | 1.0246923 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `x0_position_velocity` | 72931.3 | 11076.817 | 6.5841389 | 11.196808 | 226.23043 | 2.0268845 | 2.1259579 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 73203.566 | 11090.472 | 6.6005816 | 11.240216 | 228.91715 | 2.0220623 | 2.1207495 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `deterministic_nominal` | 4634.2168 | 4363.51 | 1.0620388 | 1.1226284 | 1.5570414 | 1.0035806 | 1.025133 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `x0_position_only` | 61611.097 | 11027.369 | 5.5871074 | 9.234434 | 131.91239 | 2.0466268 | 2.3501911 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `x0_velocity_only` | 5323.6942 | 4399.7482 | 1.2099998 | 1.4123312 | 4.3685145 | 1.0311361 | 1.0402712 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4640.5826 | 4366.7422 | 1.0627105 | 1.1257884 | 1.5873554 | 1.0020349 | 1.0238449 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5089.9306 | 4374.524 | 1.1635393 | 1.3248803 | 8.7838253 | 0.99845369 | 1.0198081 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4634.2168 | 4363.51 | 1.0620388 | 1.1226284 | 1.5570414 | 1.0035806 | 1.025133 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4634.2168 | 4363.51 | 1.0620388 | 1.1226284 | 1.5570414 | 1.0035806 | 1.025133 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4636.0207 | 4363.7822 | 1.0623859 | 1.1233754 | 1.5509498 | 1.0036166 | 1.0252382 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4664.1553 | 4364.8752 | 1.0685656 | 1.135349 | 1.9675073 | 1.0035598 | 1.0252164 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `x0_position_velocity` | 63170.807 | 11076.817 | 5.7029748 | 9.505544 | 134.96993 | 2.0491445 | 2.3476347 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 63436.297 | 11090.472 | 5.7198914 | 9.5476542 | 138.01602 | 2.0425163 | 2.3391399 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `deterministic_nominal` | 4631.395 | 4363.51 | 1.0613921 | 1.1167255 | 1.517459 | 1.0065117 | 1.0312655 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `x0_position_only` | 60941.229 | 11027.369 | 5.5263615 | 8.5951155 | 138.94595 | 2.4287514 | 2.7908759 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `x0_velocity_only` | 5208.2533 | 4399.7482 | 1.1837617 | 1.3200738 | 3.6929526 | 1.0619611 | 1.0683971 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4637.076 | 4366.7422 | 1.0619074 | 1.1194501 | 1.5383118 | 1.0050546 | 1.0301213 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4983.6294 | 4374.524 | 1.1392393 | 1.2727955 | 7.2167986 | 1.0012849 | 1.0259699 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4631.395 | 4363.51 | 1.0613921 | 1.1167255 | 1.517459 | 1.0065117 | 1.0312655 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4631.395 | 4363.51 | 1.0613921 | 1.1167255 | 1.517459 | 1.0065117 | 1.0312655 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4632.0738 | 4363.7822 | 1.0614815 | 1.1169597 | 1.497714 | 1.0065499 | 1.0313517 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4657.2417 | 4364.8752 | 1.0669817 | 1.1274284 | 1.921048 | 1.0064784 | 1.0313319 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `x0_position_velocity` | 64559.893 | 11076.817 | 5.8283797 | 9.2226243 | 150.5165 | 2.4471646 | 2.8012139 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 64460.807 | 11090.472 | 5.812269 | 9.2033604 | 151.26938 | 2.4385031 | 2.7905681 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `deterministic_nominal` | 4631.3184 | 4363.51 | 1.0613745 | 1.1186035 | 1.5107356 | 1.0046953 | 1.0302659 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `x0_position_only` | 56364.287 | 11027.369 | 5.1113086 | 8.2220631 | 103.91302 | 2.0983341 | 2.4333002 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `x0_velocity_only` | 5228.4496 | 4399.7482 | 1.188352 | 1.3567061 | 3.5621877 | 1.0391002 | 1.0509231 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4637.8884 | 4366.7422 | 1.0620935 | 1.1218518 | 1.5417844 | 1.0031579 | 1.0289917 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5047.5512 | 4374.524 | 1.1538515 | 1.3031744 | 8.2016749 | 0.9994332 | 1.0248399 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4631.3184 | 4363.51 | 1.0613745 | 1.1186035 | 1.5107356 | 1.0046953 | 1.0302659 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4631.3184 | 4363.51 | 1.0613745 | 1.1186035 | 1.5107356 | 1.0046953 | 1.0302659 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4632.2491 | 4363.7822 | 1.0615216 | 1.1189652 | 1.4911735 | 1.0047304 | 1.0303563 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4660.4294 | 4364.8752 | 1.067712 | 1.130747 | 1.9672416 | 1.0046596 | 1.0303349 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `x0_position_velocity` | 58428.51 | 11076.817 | 5.2748467 | 8.5854277 | 108.5639 | 2.1034002 | 2.4319974 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 58566.895 | 11090.472 | 5.2808297 | 8.605261 | 110.99039 | 2.0961934 | 2.4227421 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `deterministic_nominal` | 4691.3938 | 4363.51 | 1.0751422 | 1.1376091 | 2.1750518 | 1.0164377 | 1.0267789 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_position_only` | 73139.93 | 11027.369 | 6.6325819 | 11.07024 | 252.37853 | 2.1006615 | 2.2203055 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_velocity_only` | 5907.7649 | 4399.7482 | 1.3427507 | 1.5877741 | 10.032466 | 1.136132 | 1.0600532 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4696.1493 | 4366.7422 | 1.0754354 | 1.1395412 | 2.189073 | 1.0152264 | 1.0259846 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4897.609 | 4374.524 | 1.1195753 | 1.2292549 | 5.3640654 | 1.0115468 | 1.0220643 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4691.3938 | 4363.51 | 1.0751422 | 1.1376091 | 2.1750518 | 1.0164377 | 1.0267789 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4691.3938 | 4363.51 | 1.0751422 | 1.1376091 | 2.1750518 | 1.0164377 | 1.0267789 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4691.3015 | 4363.7822 | 1.075054 | 1.137398 | 2.1701113 | 1.0164991 | 1.0268823 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4704.6821 | 4364.8752 | 1.0778503 | 1.1428424 | 2.341681 | 1.016417 | 1.0268675 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_position_velocity` | 78291.423 | 11076.817 | 7.0680436 | 11.944376 | 281.46931 | 2.1181795 | 2.2110967 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 78458.383 | 11090.472 | 7.0743952 | 11.97031 | 283.66587 | 2.112708 | 2.2050225 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `deterministic_nominal` | 4640.491 | 4363.51 | 1.0634766 | 1.1225388 | 1.4454779 | 1.0069241 | 1.0275654 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `x0_position_only` | 67774.945 | 11027.369 | 6.1460665 | 10.261428 | 158.29151 | 2.1192353 | 2.4651862 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `x0_velocity_only` | 5334.9681 | 4399.7482 | 1.2125621 | 1.4132585 | 4.2607227 | 1.0354196 | 1.0444635 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4646.7835 | 4366.7422 | 1.0641305 | 1.1256736 | 1.4747632 | 1.0053687 | 1.0262687 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5095.6797 | 4374.524 | 1.1648535 | 1.3244361 | 8.7058874 | 1.0017573 | 1.0221929 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4640.491 | 4363.51 | 1.0634766 | 1.1225388 | 1.4454779 | 1.0069241 | 1.0275654 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4640.491 | 4363.51 | 1.0634766 | 1.1225388 | 1.4454779 | 1.0069241 | 1.0275654 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4642.4466 | 4363.7822 | 1.0638585 | 1.1233541 | 1.4403061 | 1.0069602 | 1.0276695 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4670.8977 | 4364.8752 | 1.0701103 | 1.1354585 | 1.8667842 | 1.0069032 | 1.0276492 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `x0_position_velocity` | 70067.45 | 11076.817 | 6.3255944 | 10.665831 | 163.25727 | 2.1246099 | 2.4665447 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 70196.276 | 11090.472 | 6.3294218 | 10.682764 | 165.93814 | 2.1172753 | 2.4570015 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `deterministic_nominal` | 4616.1396 | 4363.51 | 1.057896 | 1.1068985 | 1.4863825 | 1.0082591 | 1.0334555 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `x0_position_only` | 49121.076 | 11027.369 | 4.4544691 | 7.9398479 | 45.036992 | 1.3605846 | 1.5455523 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `x0_velocity_only` | 5262.4576 | 4399.7482 | 1.1960815 | 1.3902055 | 4.8561138 | 1.0244146 | 1.0265165 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4619.4073 | 4366.7422 | 1.0578612 | 1.1044778 | 1.4795388 | 1.0104967 | 1.034887 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5029.6506 | 4374.524 | 1.1497595 | 1.2905765 | 8.1512015 | 1.0026825 | 1.0277064 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4616.1396 | 4363.51 | 1.057896 | 1.1068985 | 1.4863825 | 1.0082591 | 1.0334555 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4616.1396 | 4363.51 | 1.057896 | 1.1068985 | 1.4863825 | 1.0082591 | 1.0334555 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4615.9939 | 4363.7822 | 1.0577966 | 1.1067266 | 1.4817858 | 1.0082461 | 1.033468 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4642.0293 | 4364.8752 | 1.0634965 | 1.1176384 | 1.8838266 | 1.0082364 | 1.0335336 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `x0_position_velocity` | 49559.732 | 11076.817 | 4.4741854 | 8.0068285 | 45.450379 | 1.3630794 | 1.5404697 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 50063.103 | 11090.472 | 4.5140641 | 8.0889349 | 48.917504 | 1.3608912 | 1.5373876 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `deterministic_nominal` | 4626.9135 | 4363.51 | 1.060365 | 1.110194 | 1.3918546 | 1.0102775 | 1.0357276 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `x0_position_only` | 67018.99 | 11027.369 | 6.0775139 | 11.470233 | 122.12422 | 1.1676384 | 1.2858709 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `x0_velocity_only` | 5902.3481 | 4399.7482 | 1.3415195 | 1.7101856 | 10.974199 | 1.0074501 | 1.0097208 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4630.6042 | 4366.7422 | 1.0604254 | 1.1086106 | 1.401274 | 1.011946 | 1.0365665 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5057.9471 | 4374.524 | 1.156228 | 1.3019145 | 8.25613 | 1.004741 | 1.0299073 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4626.9135 | 4363.51 | 1.060365 | 1.110194 | 1.3918546 | 1.0102775 | 1.0357276 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4626.9135 | 4363.51 | 1.060365 | 1.110194 | 1.3918546 | 1.0102775 | 1.0357276 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4626.9473 | 4363.7822 | 1.0603067 | 1.1100953 | 1.3852875 | 1.0102783 | 1.0357654 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4653.8211 | 4364.8752 | 1.066198 | 1.1213849 | 1.8071368 | 1.010249 | 1.0358089 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `x0_position_velocity` | 68081.434 | 11076.817 | 6.1462996 | 11.65323 | 124.29507 | 1.1694398 | 1.2841219 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 68556.481 | 11090.472 | 6.1815655 | 11.727084 | 128.12 | 1.1678625 | 1.2817167 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `deterministic_nominal` | 4656.384 | 4363.51 | 1.0671189 | 1.1284505 | 2.2264211 | 1.008537 | 1.0210469 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_position_only` | 41791.553 | 11027.369 | 3.7898027 | 5.940413 | 14.288145 | 1.9245668 | 2.0494579 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_velocity_only` | 5215.1828 | 4399.7482 | 1.1853367 | 1.2755195 | 3.7379716 | 1.1124924 | 1.0805844 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4660.7862 | 4366.7422 | 1.0673372 | 1.1264746 | 2.2291967 | 1.0108805 | 1.0224491 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5049.8085 | 4374.524 | 1.1543675 | 1.3048666 | 8.024462 | 1.0033397 | 1.0157439 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4656.384 | 4363.51 | 1.0671189 | 1.1284505 | 2.2264211 | 1.008537 | 1.0210469 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4656.384 | 4363.51 | 1.0671189 | 1.1284505 | 2.2264211 | 1.008537 | 1.0210469 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4656.6252 | 4363.7822 | 1.0671076 | 1.1284112 | 2.247562 | 1.0085311 | 1.0210281 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4679.3358 | 4364.8752 | 1.0720435 | 1.1380846 | 2.5059056 | 1.0085408 | 1.021172 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_position_velocity` | 42413.061 | 11076.817 | 3.8289936 | 6.030361 | 15.127712 | 1.9364555 | 2.045572 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 43006.484 | 11090.472 | 3.8777866 | 6.1320489 | 18.738758 | 1.9323883 | 2.0405743 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `deterministic_nominal` | 4677.7404 | 4363.51 | 1.0720132 | 1.1334817 | 1.338524 | 1.0112496 | 1.0407125 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `x0_position_only` | 82651.506 | 11027.369 | 7.4951246 | 14.36831 | 193.24679 | 1.1315182 | 1.2261665 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `x0_velocity_only` | 6715.8582 | 4399.7482 | 1.5264188 | 2.098624 | 18.037981 | 1.0011087 | 1.0108611 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4682.0676 | 4366.7422 | 1.0722107 | 1.1328337 | 1.3527408 | 1.0123148 | 1.0411706 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5166.9622 | 4374.524 | 1.1811485 | 1.3520186 | 8.7893585 | 1.0059238 | 1.0350059 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4677.7404 | 4363.51 | 1.0720132 | 1.1334817 | 1.338524 | 1.0112496 | 1.0407125 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4677.7404 | 4363.51 | 1.0720132 | 1.1334817 | 1.338524 | 1.0112496 | 1.0407125 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4678.3358 | 4363.7822 | 1.0720828 | 1.1336263 | 1.3400038 | 1.0112514 | 1.0407653 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4706.3589 | 4364.8752 | 1.0782345 | 1.145682 | 1.7110203 | 1.011235 | 1.0408079 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `x0_position_velocity` | 84300.39 | 11076.817 | 7.6105249 | 14.666266 | 197.21987 | 1.1270087 | 1.219562 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 84965.004 | 11090.472 | 7.6610808 | 14.772871 | 202.23698 | 1.1255031 | 1.2172548 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
