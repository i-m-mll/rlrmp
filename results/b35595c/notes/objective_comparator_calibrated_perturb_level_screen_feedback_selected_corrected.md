# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: fixed_bank_rescored_per_replicate checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64.

This is an objective-lens diagnostic, not a standard-certificate gate.

## Objective lenses

| lens | status | comparability |
|---|---|---|
| deterministic extLQG | available | deterministic full-Q/R/Q_f initial-state term; comparable only to full-Q/R/Q_f realized scalars |
| covariance-inclusive extLQG expected cost | available | not directly comparable to realized GRU validation scalars |
| realized selected-GRU checkpoint | available for full-Q/R/Q_f scalar rows | selected-checkpoint audit metric, not checkpoint selection input |
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
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4016.7615 | 4368.5107 | 0.91948075 | 12201.424 | 0.32920431 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3802.9344 | 4368.5107 | 0.8705334 | 12201.424 | 0.31167955 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4005.2424 | 4368.5107 | 0.91684392 | 12201.424 | 0.32826024 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3818.7422 | 4368.5107 | 0.87415198 | 12201.424 | 0.31297512 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4029.7158 | 4368.5107 | 0.92244613 | 12201.424 | 0.33026601 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3904.2465 | 4368.5107 | 0.89372485 | 12201.424 | 0.31998285 | not_implemented |

## Caveats

- `selected/total` is retained only as a labeled non-apples-to-apples diagnostic for continuity with the provisional sidecar.
- The partial x0+epsilon shared-rollout comparator is stress-test-only; expected-cost wording is allowed only when an extLQG x0-only sanity check passes. Current status: `not_applicable`.
- The apples-to-apples scalar for the available GRU validation records is restricted to rows whose run spec declares the full analytical Q/R/Q_f objective; the deterministic extLQG term is not interchangeable with the covariance-inclusive expected cost.
- This sidecar is diagnostic only and is not a standard-certificate gate.
- GRU values are selected-checkpoint realized full-QRF scalars; the shared-rollout and split-bank blocks are audit-only post-hoc rescores and are not used for checkpoint selection.
- The x0+epsilon shared-rollout block is stress-test-only unless the extLQG x0-only sanity check supports expected-cost wording.
- Split-bank GRU hidden states are initialized from the checkpoint model default rather than conditioned on the perturbed x0, so x0 lenses are recovery stress tests rather than expected-cost comparisons.

Full same-noise-bank Monte Carlo: `not_implemented` - full shared sensory/command/motor noise is not exposed for both arms. Partial shared-rollout replacement: `available_with_limitations` - shared-rollout comparator materialized common random inputs for initial state and process/load epsilon; sensory and command/motor noise are explicitly not shared under the current GRU graph contract

Per-term realized scoring: `not_implemented` - validation checkpoint manifests currently expose scalar full-QRF objectives, not running-state, terminal-state, command, force/filter, and disturbance-integrator contributions

## Shared-rollout comparator

Bank `cs_lss_shared_x0_epsilon_v1` uses 32 trials, seed `20260603`, shared initial states, and shared process/load epsilon.

Limitation: This is a shared initial-state plus process/load epsilon comparator. Sensory and command/motor noise are explicitly not claimed as shared.

| run | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | 67778.663 | 11090.472 | 6.1114317 | 9.7916262 | 187.03804 | 2.3916597 | 2.7219575 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | 177987.77 | 11090.472 | 16.04871 | 29.518668 | 618.76361 | 2.9880613 | 2.9455848 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | 69598.507 | 11090.472 | 6.2755224 | 10.443231 | 159.72307 | 2.2289209 | 2.5954406 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 300372.57 | 11090.472 | 27.08384 | 48.257244 | 1898.6323 | 3.5918652 | 3.5901176 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 115637.8 | 11090.472 | 10.426769 | 18.251947 | 422.73422 | 2.4984552 | 2.9634399 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | 190796.92 | 11090.472 | 17.203679 | 30.217045 | 1161.3352 | 2.7739266 | 2.8104117 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `deterministic_nominal` | 4669.5058 | 4363.51 | 1.0701261 | 1.1191191 | 2.1771954 | 1.0186749 | 1.0426091 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_position_only` | 63783.955 | 11027.369 | 5.7841498 | 9.1236046 | 166.89102 | 2.3784257 | 2.7159057 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_velocity_only` | 5248.5928 | 4399.7482 | 1.1929303 | 1.3333184 | 4.1830347 | 1.0660931 | 1.0730618 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4675.185 | 4366.7422 | 1.0706345 | 1.1218741 | 2.1992653 | 1.0171742 | 1.0414195 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5021.0293 | 4374.524 | 1.1477887 | 1.2751214 | 7.8101668 | 1.0133745 | 1.0372305 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4669.5058 | 4363.51 | 1.0701261 | 1.1191191 | 2.1771954 | 1.0186749 | 1.0426091 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4669.5058 | 4363.51 | 1.0701261 | 1.1191191 | 2.1771954 | 1.0186749 | 1.0426091 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4669.0132 | 4363.7822 | 1.0699464 | 1.1188115 | 2.1489919 | 1.0187188 | 1.0427066 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4693.6642 | 4364.8752 | 1.0753261 | 1.1290393 | 2.5511936 | 1.0186444 | 1.0426853 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_position_velocity` | 67860.121 | 11076.817 | 6.1263197 | 9.8061718 | 186.61763 | 2.3998054 | 2.7319141 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 67778.663 | 11090.472 | 6.1114317 | 9.7916262 | 187.03804 | 2.3916597 | 2.7219575 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `deterministic_nominal` | 4417.0855 | 4363.51 | 1.0122781 | 1.0211631 | 1.2677093 | 1.0031787 | 1.0061224 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_position_only` | 184149.01 | 11027.369 | 16.699269 | 30.353389 | 749.55965 | 2.9266996 | 2.9644466 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_velocity_only` | 5478.4987 | 4399.7482 | 1.2451846 | 1.3611436 | 6.1597041 | 1.1610786 | 1.069601 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4421.4193 | 4366.7422 | 1.0125213 | 1.0232903 | 1.2802053 | 1.0017268 | 1.004973 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4682.8549 | 4374.524 | 1.0704833 | 1.139453 | 5.5277094 | 0.99784911 | 1.0007684 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4417.0855 | 4363.51 | 1.0122781 | 1.0211631 | 1.2677093 | 1.0031787 | 1.0061224 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4417.0855 | 4363.51 | 1.0122781 | 1.0211631 | 1.2677093 | 1.0031787 | 1.0061224 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4417.9704 | 4363.7822 | 1.0124177 | 1.0213897 | 1.2630108 | 1.003255 | 1.0062531 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4434.5994 | 4364.8752 | 1.0159739 | 1.0282281 | 1.5144964 | 1.0032251 | 1.0062695 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_position_velocity` | 178358.19 | 11076.817 | 16.101936 | 29.59494 | 619.63743 | 2.9930852 | 2.95151 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 177987.77 | 11090.472 | 16.04871 | 29.518668 | 618.76361 | 2.9880613 | 2.9455848 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `deterministic_nominal` | 4645.938 | 4363.51 | 1.0647249 | 1.1177738 | 1.5190698 | 1.01166 | 1.0367452 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_position_only` | 66961.038 | 11027.369 | 6.0722586 | 9.982858 | 150.68089 | 2.2321155 | 2.6073041 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_velocity_only` | 5270.2611 | 4399.7482 | 1.1978552 | 1.369688 | 3.6725138 | 1.0452155 | 1.0577565 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4652.4505 | 4366.7422 | 1.0654283 | 1.1210202 | 1.5495235 | 1.0100976 | 1.0354392 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5053.4355 | 4374.524 | 1.1551967 | 1.2985945 | 8.0523767 | 1.0063507 | 1.0312394 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4645.938 | 4363.51 | 1.0647249 | 1.1177738 | 1.5190698 | 1.01166 | 1.0367452 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4645.938 | 4363.51 | 1.0647249 | 1.1177738 | 1.5190698 | 1.01166 | 1.0367452 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4646.401 | 4363.7822 | 1.0647647 | 1.1179096 | 1.4958919 | 1.0116975 | 1.0368386 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4673.9997 | 4364.8752 | 1.0708209 | 1.1294293 | 1.9638983 | 1.0116251 | 1.0368188 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_position_velocity` | 69580.677 | 11076.817 | 6.2816492 | 10.44573 | 157.75259 | 2.2364093 | 2.6049817 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 69598.507 | 11090.472 | 6.2755224 | 10.443231 | 159.72307 | 2.2289209 | 2.5954406 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `deterministic_nominal` | 4402.3767 | 4363.51 | 1.0089072 | 1.0121508 | 1.2644984 | 1.0042676 | 1.0080605 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_position_only` | 284822.19 | 11027.369 | 25.828661 | 46.041434 | 1709.7373 | 3.5159011 | 3.540151 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_velocity_only` | 5443.4007 | 4399.7482 | 1.2372073 | 1.3673419 | 6.3398418 | 1.1291424 | 1.0777089 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4406.7677 | 4366.7422 | 1.009166 | 1.0144047 | 1.2789506 | 1.0027379 | 1.006807 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4715.3867 | 4374.524 | 1.07792 | 1.1514431 | 6.3009708 | 0.99878892 | 1.0025161 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4402.3767 | 4363.51 | 1.0089072 | 1.0121508 | 1.2644984 | 1.0042676 | 1.0080605 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4402.3767 | 4363.51 | 1.0089072 | 1.0121508 | 1.2644984 | 1.0042676 | 1.0080605 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4402.8462 | 4363.7822 | 1.0089519 | 1.0122123 | 1.2503227 | 1.0043365 | 1.0081856 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4422.5409 | 4364.8752 | 1.0132113 | 1.0204256 | 1.5548218 | 1.0042753 | 1.0081847 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_position_velocity` | 300711.92 | 11076.817 | 27.147865 | 48.375098 | 1879.3297 | 3.6055452 | 3.6068781 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 300372.57 | 11090.472 | 27.08384 | 48.257244 | 1898.6323 | 3.5918652 | 3.5901176 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `deterministic_nominal` | 4672.2425 | 4363.51 | 1.0707532 | 1.1285657 | 2.0300614 | 1.0135054 | 1.0337169 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_position_only` | 107799.5 | 11027.369 | 9.7756317 | 16.930702 | 379.1918 | 2.4760808 | 2.9397703 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_velocity_only` | 5424.9937 | 4399.7482 | 1.2330237 | 1.4437255 | 4.7119645 | 1.0459006 | 1.0562829 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_force_filter_only` | 4678.6212 | 4366.7422 | 1.0714214 | 1.1317719 | 2.0588675 | 1.0119181 | 1.0323888 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5108.4304 | 4374.524 | 1.1677683 | 1.3225113 | 8.8932967 | 1.008248 | 1.0282022 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4672.2425 | 4363.51 | 1.0707532 | 1.1285657 | 2.0300614 | 1.0135054 | 1.0337169 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4672.2425 | 4363.51 | 1.0707532 | 1.1285657 | 2.0300614 | 1.0135054 | 1.0337169 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4673.0489 | 4363.7822 | 1.0708713 | 1.1288344 | 2.0211246 | 1.0135448 | 1.0338253 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4699.9935 | 4364.8752 | 1.0767762 | 1.1402833 | 2.405758 | 1.0134876 | 1.0338067 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_position_velocity` | 116122.62 | 11076.817 | 10.483393 | 18.349147 | 420.90672 | 2.5094226 | 2.9776795 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 115637.8 | 11090.472 | 10.426769 | 18.251947 | 422.73422 | 2.4984552 | 2.9634399 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `deterministic_nominal` | 4435.8316 | 4363.51 | 1.0165742 | 1.0295992 | 1.9806288 | 1.0006235 | 1.0073633 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_position_only` | 163436.65 | 11027.369 | 14.821001 | 25.951468 | 898.39843 | 2.6676438 | 2.7068758 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_velocity_only` | 5332.7438 | 4399.7482 | 1.2120566 | 1.3557912 | 6.7026476 | 1.0871217 | 1.050949 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4440.5897 | 4366.7422 | 1.0169114 | 1.0320188 | 2.0005353 | 0.99909328 | 1.006088 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4804.367 | 4374.524 | 1.0982605 | 1.1936173 | 7.7865343 | 0.99538302 | 1.0020089 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4435.8316 | 4363.51 | 1.0165742 | 1.0295992 | 1.9806288 | 1.0006235 | 1.0073633 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4435.8316 | 4363.51 | 1.0165742 | 1.0295992 | 1.9806288 | 1.0006235 | 1.0073633 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4436.2999 | 4363.7822 | 1.0166181 | 1.0296899 | 1.9684977 | 1.0006812 | 1.0074932 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4457.8329 | 4364.8752 | 1.0212968 | 1.0387461 | 2.2738778 | 1.0006325 | 1.0074792 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_position_velocity` | 191605.16 | 11076.817 | 17.297855 | 30.353967 | 1163.0883 | 2.7863546 | 2.8224983 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 190796.92 | 11090.472 | 17.203679 | 30.217045 | 1161.3352 | 2.7739266 | 2.8104117 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
