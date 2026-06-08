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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 7381.3103 | 4368.5107 | 1.6896629 | 12201.424 | 0.60495481 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 10003.107 | 4368.5107 | 2.289821 | 12201.424 | 0.81983112 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | 157847.72 | 11090.472 | 14.232732 | 29.124695 | 413.7934 | 0.57406344 | 0.75994361 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | 288415.74 | 11090.472 | 26.005723 | 52.483983 | 1202.7881 | 0.31219488 | 0.37934228 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `deterministic_nominal` | 8182.9155 | 4363.51 | 1.8753058 | 2.9812992 | 15.865287 | 0.88201551 | 0.96789392 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_position_only` | 151366.81 | 11027.369 | 13.726466 | 27.963909 | 388.29067 | 0.57984395 | 0.76805571 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_velocity_only` | 14129.387 | 4399.7482 | 3.2114082 | 5.8046428 | 65.981134 | 0.85789919 | 0.94064031 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_force_filter_only` | 8206.6688 | 4366.7422 | 1.8793573 | 2.9916844 | 16.052162 | 0.88057811 | 0.96680248 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 9213.7944 | 4374.524 | 2.1062393 | 3.437445 | 31.6931 | 0.87803131 | 0.96343923 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 8182.9155 | 4363.51 | 1.8753058 | 2.9812992 | 15.865287 | 0.88201551 | 0.96789392 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 8182.9155 | 4363.51 | 1.8753058 | 2.9812992 | 15.865287 | 0.88201551 | 0.96789392 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 8180.7789 | 4363.7822 | 1.8746992 | 2.9805231 | 15.897063 | 0.8820301 | 0.9679041 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 8227.6049 | 4364.8752 | 1.8849577 | 3.0001972 | 16.262092 | 0.88196077 | 0.96794124 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_position_velocity` | 155990.82 | 11076.817 | 14.08264 | 28.818199 | 400.30696 | 0.57544018 | 0.76214389 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 157847.72 | 11090.472 | 14.232732 | 29.124695 | 413.7934 | 0.57406344 | 0.75994361 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `deterministic_nominal` | 7566.6577 | 4363.51 | 1.7340759 | 2.7053571 | 11.75744 | 0.86551058 | 0.95313718 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_position_only` | 266162.88 | 11027.369 | 24.136571 | 48.592741 | 1075.7462 | 0.30977346 | 0.38141223 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_velocity_only` | 27555.1 | 4399.7482 | 6.2628812 | 11.775019 | 271.1526 | 0.87402587 | 0.93661761 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_force_filter_only` | 7816.694 | 4366.7422 | 1.7900516 | 2.8223002 | 13.818061 | 0.8636841 | 0.95439883 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 10578.729 | 4374.524 | 2.4182582 | 4.0434665 | 56.937041 | 0.86478193 | 0.95059251 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_position_only` | 7566.6577 | 4363.51 | 1.7340759 | 2.7053571 | 11.75744 | 0.86551058 | 0.95313718 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 7566.6577 | 4363.51 | 1.7340759 | 2.7053571 | 11.75744 | 0.86551058 | 0.95313718 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 7568.9234 | 4363.7822 | 1.734487 | 2.7066416 | 11.825481 | 0.86542688 | 0.95298844 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 7675.7257 | 4364.8752 | 1.7585212 | 2.7534413 | 13.081216 | 0.86548238 | 0.95305667 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_position_velocity` | 281576.35 | 11076.817 | 25.420331 | 51.317112 | 1154.5131 | 0.31155288 | 0.37986071 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64` | `x0_plus_epsilon` | 288415.74 | 11090.472 | 26.005723 | 52.483983 | 1202.7881 | 0.31219488 | 0.37934228 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
