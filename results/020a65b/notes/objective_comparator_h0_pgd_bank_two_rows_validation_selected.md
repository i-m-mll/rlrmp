# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation_selected_per_replicate checkpoints for C&S GRU runs: target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64, target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64.

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
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 6823.1293 | 4368.5107 | 1.5618891 | 12201.424 | 0.5592076 | not_implemented |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4008.1539 | 4368.5107 | 0.91751039 | 12201.424 | 0.32849885 | not_implemented |

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
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | 195566.17 | 11090.472 | 17.63371 | 35.159441 | 810.49923 | 0.60876489 | 0.56480856 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | 48200.712 | 11090.472 | 4.3461371 | 5.6233354 | 6.7512471 | 3.2090965 | 3.4568204 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `deterministic_nominal` | 4381.8356 | 4363.51 | 1.0041997 | 1.0038775 | 1.3055037 | 1.0035437 | 1.00338 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_position_only` | 192620.33 | 11027.369 | 17.467478 | 34.692398 | 791.42577 | 0.60541771 | 0.56320494 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_velocity_only` | 5036.0668 | 4399.7482 | 1.1446261 | 1.2858882 | 6.2550259 | 1.0131561 | 1.0103905 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4385.8807 | 4366.7422 | 1.0043828 | 1.0054614 | 1.3095029 | 1.002458 | 1.0025958 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4822.3937 | 4374.524 | 1.1023814 | 1.2011445 | 7.9441836 | 0.99821839 | 0.9978406 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4381.8356 | 4363.51 | 1.0041997 | 1.0038775 | 1.3055037 | 1.0035437 | 1.00338 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4381.8356 | 4363.51 | 1.0041997 | 1.0038775 | 1.3055037 | 1.0035437 | 1.00338 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4382.1638 | 4363.7822 | 1.0042123 | 1.003889 | 1.3062436 | 1.0035558 | 1.0034253 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4404.9011 | 4364.8752 | 1.00917 | 1.0135875 | 1.6118702 | 1.0035231 | 1.003477 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_position_velocity` | 193923.73 | 11076.817 | 17.507171 | 34.8965 | 795.47121 | 0.60962464 | 0.56571371 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 195566.17 | 11090.472 | 17.63371 | 35.159441 | 810.49923 | 0.60876489 | 0.56480856 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `deterministic_nominal` | 4741.4557 | 4363.51 | 1.0866151 | 0.94682325 | 0.96965085 | 1.209941 | 1.1884337 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_position_only` | 47825.251 | 11027.369 | 4.3369592 | 5.5672868 | 4.8152895 | 3.2331962 | 3.4914415 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_velocity_only` | 5058.4093 | 4399.7482 | 1.1497043 | 1.0305638 | 0.98635197 | 1.256456 | 1.2276994 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_force_filter_only` | 4746.91 | 4366.7422 | 1.0870598 | 0.94501184 | 0.97010399 | 1.2124643 | 1.190072 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5021.4595 | 4374.524 | 1.1478871 | 1.068078 | 6.945818 | 1.2021023 | 1.1801835 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4741.4557 | 4363.51 | 1.0866151 | 0.94682325 | 0.96965085 | 1.209941 | 1.1884337 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4741.4557 | 4363.51 | 1.0866151 | 0.94682325 | 0.96965085 | 1.209941 | 1.1884337 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4741.776 | 4363.7822 | 1.0866207 | 0.94674865 | 0.98901346 | 1.2099261 | 1.1884547 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4766.3928 | 4364.8752 | 1.0919883 | 0.95656388 | 1.4640506 | 1.2100561 | 1.18865 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_position_velocity` | 48222.905 | 11076.817 | 4.3534985 | 5.6370977 | 4.837961 | 3.2147957 | 3.4658901 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64` | `x0_plus_epsilon` | 48200.712 | 11090.472 | 4.3461371 | 5.6233354 | 6.7512471 | 3.2090965 | 3.4568204 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
