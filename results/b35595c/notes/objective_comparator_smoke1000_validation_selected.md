# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64.

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
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 5115.3641 | 4368.5107 | 1.1709629 | 12201.424 | 0.41924319 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 7027.1365 | 4368.5107 | 1.6085886 | 12201.424 | 0.57592756 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | 99254.226 | 11090.472 | 8.9495041 | 17.488984 | 197.08308 | 1.2053117 | 1.4575905 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | 176603.16 | 11090.472 | 15.923863 | 32.817331 | 433.0145 | 0.55308277 | 0.73505111 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `deterministic_nominal` | 5738.5998 | 4363.51 | 1.3151339 | 1.5931363 | 5.2422757 | 1.0671531 | 1.0784018 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_position_only` | 94912.144 | 11027.369 | 8.6069616 | 16.712562 | 183.54626 | 1.2044536 | 1.4540707 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_velocity_only` | 8494.9994 | 4399.7482 | 1.9307922 | 2.8882661 | 17.18014 | 1.0899185 | 1.110901 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_force_filter_only` | 5750.5597 | 4366.7422 | 1.3168993 | 1.5989157 | 5.3200128 | 1.0653434 | 1.0769427 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 6131.3771 | 4374.524 | 1.4016101 | 1.7683945 | 11.21419 | 1.061627 | 1.0724036 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 5738.5998 | 4363.51 | 1.3151339 | 1.5931363 | 5.2422757 | 1.0671531 | 1.0784018 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 5738.5998 | 4363.51 | 1.3151339 | 1.5931363 | 5.2422757 | 1.0671531 | 1.0784018 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 5742.4939 | 4363.7822 | 1.3159442 | 1.5947828 | 5.3071444 | 1.0672012 | 1.0785209 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 5768.2403 | 4364.8752 | 1.3215132 | 1.6055296 | 5.6014366 | 1.0671312 | 1.0785065 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_position_velocity` | 98410.084 | 11076.817 | 8.884329 | 17.347745 | 191.57718 | 1.2092459 | 1.4632912 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 99254.226 | 11090.472 | 8.9495041 | 17.488984 | 197.08308 | 1.2053117 | 1.4575905 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `deterministic_nominal` | 7295.8735 | 4363.51 | 1.6720194 | 2.527565 | 9.832744 | 0.90117375 | 1.0055331 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_position_only` | 166799.67 | 11027.369 | 15.125971 | 31.043161 | 395.43515 | 0.56013564 | 0.74412157 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_velocity_only` | 15997.263 | 4399.7482 | 3.6359496 | 6.6514369 | 83.730857 | 0.87347865 | 0.9802067 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_force_filter_only` | 7323.733 | 4366.7422 | 1.6771618 | 2.54049 | 9.9965329 | 0.89960883 | 1.0042517 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 8473.9632 | 4374.524 | 1.9371166 | 3.0516583 | 27.405558 | 0.89711109 | 1.0010039 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 7295.8735 | 4363.51 | 1.6720194 | 2.527565 | 9.832744 | 0.90117375 | 1.0055331 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 7295.8735 | 4363.51 | 1.6720194 | 2.527565 | 9.832744 | 0.90117375 | 1.0055331 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 7300.3132 | 4363.7822 | 1.6729325 | 2.5298292 | 9.8719901 | 0.90119706 | 1.0055426 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 7357.2006 | 4364.8752 | 1.6855466 | 2.5540703 | 10.542717 | 0.90115072 | 1.0055966 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_position_velocity` | 174791.69 | 11076.817 | 15.779956 | 32.519072 | 420.35729 | 0.55453399 | 0.73728671 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 176603.16 | 11090.472 | 15.923863 | 32.817331 | 433.0145 | 0.55308277 | 0.73505111 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
