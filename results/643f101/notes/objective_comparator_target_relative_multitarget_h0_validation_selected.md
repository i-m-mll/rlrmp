# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64, target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64.

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
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4159.0686 | 4368.5107 | 0.95205641 | 12201.424 | 0.34086746 | not_implemented |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3974.4788 | 4368.5107 | 0.90980178 | 12201.424 | 0.32573892 | not_implemented |

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
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | 84449.03 | 11090.472 | 7.6145568 | 13.957918 | 232.7838 | 1.5784966 | 1.8055223 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | 76067.164 | 11090.472 | 6.858785 | 12.312229 | 210.47529 | 1.6335489 | 1.8426201 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `deterministic_nominal` | 4456.8953 | 4363.51 | 1.0214014 | 1.0587464 | 2.3268885 | 0.98301903 | 0.99329425 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_only` | 84184.711 | 11027.369 | 7.6341609 | 13.928649 | 230.60791 | 1.5931997 | 1.8254096 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_velocity_only` | 5033.8309 | 4399.7482 | 1.144118 | 1.2937828 | 3.9687495 | 1.0102167 | 1.0171029 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_force_filter_only` | 4463.4192 | 4366.7422 | 1.0221394 | 1.0621194 | 2.369356 | 0.98140706 | 0.99180373 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4905.9066 | 4374.524 | 1.1214721 | 1.2588978 | 9.2616345 | 0.9779162 | 0.98787946 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4456.8953 | 4363.51 | 1.0214014 | 1.0587464 | 2.3268885 | 0.98301903 | 0.99329425 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4456.8953 | 4363.51 | 1.0214014 | 1.0587464 | 2.3268885 | 0.98301903 | 0.99329425 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4458.7092 | 4363.7822 | 1.0217534 | 1.0594754 | 2.3346058 | 0.98306176 | 0.99340574 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4486.0959 | 4364.8752 | 1.0277719 | 1.0710292 | 2.7522061 | 0.98298746 | 0.99337712 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_velocity` | 84121.419 | 11076.817 | 7.5943677 | 13.904979 | 228.98835 | 1.5862116 | 1.8157968 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_plus_epsilon` | 84449.03 | 11090.472 | 7.6145568 | 13.957918 | 232.7838 | 1.5784966 | 1.8055223 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `deterministic_nominal` | 4448.3645 | 4363.51 | 1.0194464 | 1.0803392 | 1.70701 | 0.96647918 | 0.96622885 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_position_only` | 75052.565 | 11027.369 | 6.8060262 | 12.151242 | 207.2036 | 1.6365876 | 1.8469291 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_velocity_only` | 4918.4913 | 4399.7482 | 1.1179029 | 1.2486355 | 3.0661342 | 1.0063126 | 0.99962961 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_force_filter_only` | 4454.2507 | 4366.7422 | 1.0200398 | 1.0832521 | 1.7404675 | 0.96500741 | 0.96492872 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4884.0105 | 4374.524 | 1.1164667 | 1.2736844 | 8.6045181 | 0.96155573 | 0.9611696 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4448.3645 | 4363.51 | 1.0194464 | 1.0803392 | 1.70701 | 0.96647918 | 0.96622885 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4448.3645 | 4363.51 | 1.0194464 | 1.0803392 | 1.70701 | 0.96647918 | 0.96622885 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4449.666 | 4363.7822 | 1.0196811 | 1.0808833 | 1.6931234 | 0.96651303 | 0.96632034 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4477.6908 | 4364.8752 | 1.0258462 | 1.0926757 | 2.1408222 | 0.9664564 | 0.96631884 | 1 |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_position_velocity` | 75849.998 | 11076.817 | 6.847635 | 12.272776 | 208.9036 | 1.6408538 | 1.8518561 | not_comparable |
| `target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_plus_epsilon` | 76067.164 | 11090.472 | 6.858785 | 12.312229 | 210.47529 | 1.6335489 | 1.8426201 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
