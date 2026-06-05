# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64.

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
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4367.3012 | 4368.5107 | 0.99972314 | 12201.424 | 0.35793372 | not_implemented |

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
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 307479.24 | 11090.472 | 27.72463 | 55.09096 | 1470.781 | 0.45767711 | 0.49109855 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `deterministic_nominal` | 4392.2063 | 4363.51 | 1.0065764 | 1.0228575 | 1.1204789 | 0.98705687 | 1.0061132 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_only` | 300703.24 | 11027.369 | 27.26881 | 53.940062 | 1433.657 | 0.45817451 | 0.49485227 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_velocity_only` | 7666.4738 | 4399.7482 | 1.7424801 | 2.4764519 | 46.610395 | 0.99268678 | 1.00656 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_force_filter_only` | 4399.9303 | 4366.7422 | 1.0076002 | 1.0269599 | 1.1686177 | 0.98535372 | 1.004467 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4709.7298 | 4374.524 | 1.0766268 | 1.1628966 | 6.7219793 | 0.98133808 | 0.99938227 | 1 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4392.2063 | 4363.51 | 1.0065764 | 1.0228575 | 1.1204789 | 0.98705687 | 1.0061132 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4392.2063 | 4363.51 | 1.0065764 | 1.0228575 | 1.1204789 | 0.98705687 | 1.0061132 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4394.5851 | 4363.7822 | 1.0070588 | 1.0237158 | 1.1369572 | 0.98715558 | 1.0062944 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4418.461 | 4364.8752 | 1.0122766 | 1.0338208 | 1.4987204 | 0.98710175 | 1.0062828 | 1 |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_velocity` | 306734.78 | 11076.817 | 27.691601 | 54.998099 | 1457.5134 | 0.45838771 | 0.4926666 | not_comparable |
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_plus_epsilon` | 307479.24 | 11090.472 | 27.72463 | 55.09096 | 1470.781 | 0.45767711 | 0.49109855 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
