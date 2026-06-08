# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: smoke__broad_strong_cal_small, smoke__proprio_cal_stress.

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
| `smoke__broad_strong_cal_small` | comparable_deterministic_full_qrf | 144402.31 | 4368.5107 | 33.055272 | 12201.424 | 11.834873 | not_implemented |
| `smoke__proprio_cal_stress` | comparable_deterministic_full_qrf | 145524.42 | 4368.5107 | 33.312136 | 12201.424 | 11.926839 | not_implemented |

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
| `smoke__broad_strong_cal_small` | 1095607.3 | 11090.472 | 98.788158 | 196.0167 | 5837.3484 | 5.4485083e-05 | 0.00017467556 | 1 |
| `smoke__proprio_cal_stress` | 1097011.9 | 11090.472 | 98.914807 | 196.25499 | 5848.2063 | 7.4278961e-05 | 0.00021074976 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `smoke__broad_strong_cal_small` | `deterministic_nominal` | 164976.44 | 4363.51 | 37.808194 | 74.848896 | 2173.3452 | 4.4023555e-05 | 8.6870591e-05 | not_comparable |
| `smoke__broad_strong_cal_small` | `x0_position_only` | 1042252.3 | 11027.369 | 94.515043 | 187.00977 | 5459.3629 | 4.9091199e-05 | 9.6661222e-05 | not_comparable |
| `smoke__broad_strong_cal_small` | `x0_velocity_only` | 208086.09 | 4399.7482 | 47.294999 | 93.990675 | 2855.1185 | 5.1364249e-05 | 0.00010134631 | not_comparable |
| `smoke__broad_strong_cal_small` | `x0_force_filter_only` | 165743.26 | 4366.7422 | 37.955815 | 75.154786 | 2183.6112 | 4.4004596e-05 | 0.00023747316 | not_comparable |
| `smoke__broad_strong_cal_small` | `x0_disturbance_integrator_only` | 178546.49 | 4374.524 | 40.815068 | 80.691055 | 2405.0095 | 4.4696088e-05 | 8.7931295e-05 | 1 |
| `smoke__broad_strong_cal_small` | `process_epsilon_position_only` | 164976.44 | 4363.51 | 37.808194 | 74.848896 | 2173.3452 | 4.4023555e-05 | 8.6870591e-05 | not_comparable |
| `smoke__broad_strong_cal_small` | `process_epsilon_velocity_only` | 164976.44 | 4363.51 | 37.808194 | 74.848896 | 2173.3452 | 4.4023555e-05 | 8.6870591e-05 | not_comparable |
| `smoke__broad_strong_cal_small` | `process_epsilon_force_filter_only` | 164727.18 | 4363.7822 | 37.748718 | 74.738335 | 2192.5358 | 4.405216e-05 | 0.00010600382 | not_comparable |
| `smoke__broad_strong_cal_small` | `process_epsilon_integrator_only` | 164870.65 | 4364.8752 | 37.772135 | 74.777433 | 2140.7931 | 4.407384e-05 | 8.6966118e-05 | 1 |
| `smoke__broad_strong_cal_small` | `x0_position_velocity` | 1076139.9 | 11076.817 | 97.152451 | 192.811 | 5661.8644 | 5.4097679e-05 | 0.00010665263 | not_comparable |
| `smoke__broad_strong_cal_small` | `x0_plus_epsilon` | 1095607.3 | 11090.472 | 98.788158 | 196.0167 | 5837.3484 | 5.4485083e-05 | 0.00017467556 | 1 |
| `smoke__proprio_cal_stress` | `deterministic_nominal` | 164097.08 | 4363.51 | 37.60667 | 74.463971 | 2158.1285 | 5.5238213e-05 | 0.0001081384 | not_comparable |
| `smoke__proprio_cal_stress` | `x0_position_only` | 1042119.4 | 11027.369 | 94.502996 | 186.98771 | 5458.2048 | 5.6390227e-05 | 0.00011034927 | not_comparable |
| `smoke__proprio_cal_stress` | `x0_velocity_only` | 209697.81 | 4399.7482 | 47.661321 | 94.690249 | 2884.5307 | 9.4758896e-05 | 0.00018540551 | not_comparable |
| `smoke__proprio_cal_stress` | `x0_force_filter_only` | 164717.99 | 4366.7422 | 37.721025 | 74.705203 | 2166.1407 | 5.7052447e-05 | 0.00025714447 | not_comparable |
| `smoke__proprio_cal_stress` | `x0_disturbance_integrator_only` | 177942.49 | 4374.524 | 40.676995 | 80.422486 | 2395.734 | 5.688133e-05 | 0.00011028657 | 1 |
| `smoke__proprio_cal_stress` | `process_epsilon_position_only` | 164097.08 | 4363.51 | 37.60667 | 74.463971 | 2158.1285 | 5.5238213e-05 | 0.0001081384 | not_comparable |
| `smoke__proprio_cal_stress` | `process_epsilon_velocity_only` | 164097.08 | 4363.51 | 37.60667 | 74.463971 | 2158.1285 | 5.5238213e-05 | 0.0001081384 | not_comparable |
| `smoke__proprio_cal_stress` | `process_epsilon_force_filter_only` | 163872.51 | 4363.7822 | 37.552863 | 74.364068 | 2177.6259 | 5.4889183e-05 | 0.00012746024 | not_comparable |
| `smoke__proprio_cal_stress` | `process_epsilon_integrator_only` | 163993.17 | 4364.8752 | 37.571102 | 74.393362 | 2125.8484 | 5.5580213e-05 | 0.00010872168 | 1 |
| `smoke__proprio_cal_stress` | `x0_position_velocity` | 1077665.6 | 11076.817 | 97.290189 | 193.07354 | 5672.6758 | 7.4286381e-05 | 0.00014572543 | not_comparable |
| `smoke__proprio_cal_stress` | `x0_plus_epsilon` | 1097011.9 | 11090.472 | 98.914807 | 196.25499 | 5848.2063 | 7.4278961e-05 | 0.00021074976 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
