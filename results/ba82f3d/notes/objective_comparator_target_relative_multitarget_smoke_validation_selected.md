# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_smoke.

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
| `target_relative_multitarget_fullqrf_smoke` | comparable_deterministic_full_qrf | 168935.6 | 4368.5107 | 38.671212 | 12201.424 | 13.845564 | not_implemented |

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
| `target_relative_multitarget_fullqrf_smoke` | 1291613.3 | 11090.472 | 116.46152 | 229.74669 | 7228.5745 | 0.0024805076 | 0.0049455691 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_smoke` | `deterministic_nominal` | 257142.1 | 4363.51 | 58.930104 | 114.94406 | 3830.3172 | 0.0046124922 | 0.0091102406 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `x0_position_only` | 1249394.1 | 11027.369 | 113.29938 | 222.72233 | 6919.435 | 0.0025271289 | 0.0049918275 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `x0_velocity_only` | 292623.5 | 4399.7482 | 66.509148 | 130.7691 | 4375.1664 | 0.0045232342 | 0.0089003961 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `x0_force_filter_only` | 258050.42 | 4366.7422 | 59.094495 | 115.29277 | 3841.9125 | 0.0046072465 | 0.0092672893 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `x0_disturbance_integrator_only` | 271934.42 | 4374.524 | 62.163203 | 121.27462 | 4079.0904 | 0.0045976716 | 0.0090821387 | 1 |
| `target_relative_multitarget_fullqrf_smoke` | `process_epsilon_position_only` | 257142.1 | 4363.51 | 58.930104 | 114.94406 | 3830.3172 | 0.0046124922 | 0.0091102406 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `process_epsilon_velocity_only` | 257142.1 | 4363.51 | 58.930104 | 114.94406 | 3830.3172 | 0.0046124922 | 0.0091102406 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `process_epsilon_force_filter_only` | 256718.72 | 4363.7822 | 58.829408 | 114.75704 | 3864.325 | 0.0046116768 | 0.0090844172 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `process_epsilon_integrator_only` | 256788.32 | 4364.8752 | 58.830622 | 114.75124 | 3769.8375 | 0.0046110685 | 0.0091073152 | 1 |
| `target_relative_multitarget_fullqrf_smoke` | `x0_position_velocity` | 1271212.7 | 11076.817 | 114.76336 | 226.41173 | 7035.5622 | 0.0024869273 | 0.0049060557 | not_comparable |
| `target_relative_multitarget_fullqrf_smoke` | `x0_plus_epsilon` | 1291613.3 | 11090.472 | 116.46152 | 229.74669 | 7228.5745 | 0.0024805076 | 0.0049455691 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
