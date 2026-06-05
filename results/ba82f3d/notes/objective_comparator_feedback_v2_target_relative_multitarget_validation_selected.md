# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64, target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64.

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
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4052.1743 | 4368.5107 | 0.92758714 | 12201.424 | 0.33210666 | not_implemented |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 3890.6316 | 4368.5107 | 0.89060823 | 12201.424 | 0.318867 | not_implemented |

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
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | 129482.15 | 11090.472 | 11.67508 | 20.335047 | 523.81174 | 2.7321715 | 3.215932 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | 437548.77 | 11090.472 | 39.452673 | 68.798171 | 3463.2759 | 4.1980991 | 4.3760866 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `deterministic_nominal` | 4676.7406 | 4363.51 | 1.0717841 | 1.1316843 | 1.5895472 | 1.0155124 | 1.0312948 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_only` | 123521.47 | 11027.369 | 11.201354 | 19.349759 | 487.69823 | 2.7195214 | 3.2090956 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_velocity_only` | 5263.7923 | 4399.7482 | 1.1963849 | 1.3574071 | 3.9393181 | 1.0546037 | 1.0574457 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_force_filter_only` | 4682.7644 | 4366.7422 | 1.0723702 | 1.1346428 | 1.6153902 | 1.013979 | 1.0300598 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 5135.5511 | 4374.524 | 1.173968 | 1.3338292 | 9.2929624 | 1.0101684 | 1.0258414 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4676.7406 | 4363.51 | 1.0717841 | 1.1316843 | 1.5895472 | 1.0155124 | 1.0312948 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4676.7406 | 4363.51 | 1.0717841 | 1.1316843 | 1.5895472 | 1.0155124 | 1.0312948 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4677.9363 | 4363.7822 | 1.0719913 | 1.1321628 | 1.5742273 | 1.0155501 | 1.0313863 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4707.8799 | 4364.8752 | 1.078583 | 1.1446544 | 2.0948641 | 1.0154729 | 1.0313572 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_velocity` | 130329.18 | 11076.817 | 11.765941 | 20.485393 | 526.39345 | 2.7442788 | 3.2311672 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_plus_epsilon` | 129482.15 | 11090.472 | 11.67508 | 20.335047 | 523.81174 | 2.7321715 | 3.215932 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `deterministic_nominal` | 4455.9003 | 4363.51 | 1.0211734 | 1.0463725 | 1.5367992 | 0.99510896 | 1.0066474 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_position_only` | 393933.35 | 11027.369 | 35.723239 | 62.083344 | 3036.2301 | 3.9827771 | 4.1670527 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_velocity_only` | 4999.547 | 4399.7482 | 1.1363257 | 1.2025136 | 3.3672264 | 1.0828609 | 1.0555542 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_force_filter_only` | 4460.4757 | 4366.7422 | 1.0214653 | 1.0485916 | 1.5505296 | 0.9936792 | 1.0055038 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4821.5819 | 4374.524 | 1.1021958 | 1.2072572 | 7.7783979 | 0.98986779 | 1.0013298 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4455.9003 | 4363.51 | 1.0211734 | 1.0463725 | 1.5367992 | 0.99510896 | 1.0066474 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4455.9003 | 4363.51 | 1.0211734 | 1.0463725 | 1.5367992 | 0.99510896 | 1.0066474 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4456.9306 | 4363.7822 | 1.0213458 | 1.0467205 | 1.5285631 | 0.99516674 | 1.0067571 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4481.8994 | 4364.8752 | 1.0268104 | 1.057062 | 1.958213 | 0.99511375 | 1.0067516 | 1 |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_position_velocity` | 441760.38 | 11076.817 | 39.881528 | 69.499823 | 3483.9256 | 4.2196277 | 4.3979485 | not_comparable |
| `target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_plus_epsilon` | 437548.77 | 11090.472 | 39.452673 | 68.798171 | 3463.2759 | 4.1980991 | 4.3760866 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
