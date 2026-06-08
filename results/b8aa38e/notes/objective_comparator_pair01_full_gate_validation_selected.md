# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64.

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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4027.2288 | 4368.5107 | 0.92187684 | 12201.424 | 0.33006219 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4031.247 | 4368.5107 | 0.92279665 | 12201.424 | 0.33039151 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 78458.383 | 11090.472 | 7.0743952 | 11.97031 | 283.66587 | 2.112708 | 2.2050225 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 43006.484 | 11090.472 | 3.8777866 | 6.1320489 | 18.738758 | 1.9323883 | 2.0405743 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
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

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
