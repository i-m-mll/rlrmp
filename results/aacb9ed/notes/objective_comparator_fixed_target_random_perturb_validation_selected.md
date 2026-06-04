# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation-selected checkpoints for C&S GRU runs: fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64, fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64.

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
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4366.1153 | 4368.5107 | 0.99945167 | 12201.424 | 0.35783652 | not_implemented |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4365.3277 | 4368.5107 | 0.99927139 | 12201.424 | 0.35777198 | not_implemented |

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
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | 176882.81 | 11090.472 | 15.949079 | 31.815697 | 719.16417 | 0.55035785 | 0.61982691 | 1 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | 174602.06 | 11090.472 | 15.743428 | 31.449843 | 680.78476 | 0.60346045 | 0.65747539 | 1 |

## Standard split-bank comparator

| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `deterministic_nominal` | 4388.2581 | 4363.51 | 1.0056716 | 1.0125214 | 1.2577039 | 0.99534123 | 1.0083563 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_only` | 171591.84 | 11027.369 | 15.560542 | 30.889344 | 694.28626 | 0.55416164 | 0.62925329 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_velocity_only` | 5394.076 | 4399.7482 | 1.2259965 | 1.4714867 | 9.6255319 | 0.99658597 | 1.0002305 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_force_filter_only` | 4393.7137 | 4366.7422 | 1.0061766 | 1.0156482 | 1.2669159 | 0.99358376 | 1.0067784 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4711.768 | 4374.524 | 1.0770927 | 1.1529207 | 7.5193424 | 0.98956005 | 1.0019979 | 1 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_position_only` | 4388.2581 | 4363.51 | 1.0056716 | 1.0125214 | 1.2577039 | 0.99534123 | 1.0083563 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_velocity_only` | 4388.2581 | 4363.51 | 1.0056716 | 1.0125214 | 1.2577039 | 0.99534123 | 1.0083563 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4390.0921 | 4363.7822 | 1.0060292 | 1.0130945 | 1.2702509 | 0.99546984 | 1.0085516 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `process_epsilon_integrator_only` | 4411.9672 | 4364.8752 | 1.0107889 | 1.0222888 | 1.5985043 | 0.99543207 | 1.008564 | 1 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_position_velocity` | 176565.95 | 11076.817 | 15.940135 | 31.781564 | 712.9528 | 0.55217497 | 0.62210737 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64` | `x0_plus_epsilon` | 176882.81 | 11090.472 | 15.949079 | 31.815697 | 719.16417 | 0.55035785 | 0.61982691 | 1 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `deterministic_nominal` | 4392.543 | 4363.51 | 1.0066536 | 1.010035 | 1.1286887 | 1.0036715 | 1.00287 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_position_only` | 171993.28 | 11027.369 | 15.596946 | 31.002436 | 673.51037 | 0.59947089 | 0.65497675 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_velocity_only` | 6455.5868 | 4399.7482 | 1.4672628 | 1.9159468 | 19.976409 | 1.0420837 | 1.0344329 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_force_filter_only` | 4397.6214 | 4366.7422 | 1.0070715 | 1.0129032 | 1.1392035 | 1.0019636 | 1.0013355 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_disturbance_integrator_only` | 4653.0574 | 4374.524 | 1.0636717 | 1.1237094 | 5.9670691 | 0.9982375 | 0.99701419 | 1 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_position_only` | 4392.543 | 4363.51 | 1.0066536 | 1.010035 | 1.1286887 | 1.0036715 | 1.00287 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_velocity_only` | 4392.543 | 4363.51 | 1.0066536 | 1.010035 | 1.1286887 | 1.0036715 | 1.00287 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_force_filter_only` | 4393.9885 | 4363.7822 | 1.006922 | 1.0104424 | 1.1339309 | 1.003791 | 1.0030641 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `process_epsilon_integrator_only` | 4411.7532 | 4364.8752 | 1.0107398 | 1.0179168 | 1.3743511 | 1.0037577 | 1.0030777 | 1 |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_position_velocity` | 173994.59 | 11076.817 | 15.707995 | 31.365623 | 672.60002 | 0.60587839 | 0.65996273 | not_comparable |
| `fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64` | `x0_plus_epsilon` | 174602.06 | 11090.472 | 15.743428 | 31.449843 | 680.78476 | 0.60346045 | 0.65747539 | 1 |

Fairness/residual notes:

- `initial_observation_history`: partially_consistent - The comparator replaces trial_specs.inits['mechanics.vector'] for x0 lenses. It does not separately rewrite any pre-existing observation-history buffers; the C&S LSS graph observes the perturbed initial mechanics state from the rollout start.
- `gru_hidden_state_initialization`: stress_test_only - GRU recurrent hidden state starts from the checkpoint/model default during eval_trials and is not conditioned on the perturbed x0.
- `noise_channels`: declared - Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; sensory and command/motor noise remain graph-internal for the GRU arm and explicit zero draws for the extLQG arm in this materialization.
