# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v5`.

Scope: validation-selected checkpoints for C&S GRU runs: fixed_target_perturbation_fullqrf_warmcos__lr1e-3_clip5_b64, fixed_target_perturbation_fullqrf_warmcos__lr3e-3_clip5_b64.

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
| standard split-bank comparator | available | deterministic nominal, x0-only, epsilon-only, and x0+epsilon audit-only lenses |

## extLQG decomposition

| component | value | lens |
|---|---:|---|
| deterministic initial-state term | 4368.5107 | comparable to realized/validation full-QRF values |
| initial covariance trace term | 7775.5302 | expected-cost sidecar only |
| accumulated noise scalar | 57.383523 | expected-cost sidecar only |
| total expected cost | 12201.424 | not directly comparable to GRU validation values |
| x0-only realized sanity | warning | realized extLQG x0-only cost vs deterministic + initial-covariance-trace expectation |

## GRU comparison

| run | row comparability | mean selected validation | deterministic extLQG | selected/deterministic | total expected cost | selected/total | per-term scoring |
|---|---|---:|---:|---:|---:|---:|---|
| `fixed_target_perturbation_fullqrf_warmcos__lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4367.4157 | 4368.5107 | 0.99974934 | 12201.424 | 0.3579431 | not_implemented |
| `fixed_target_perturbation_fullqrf_warmcos__lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4362.4926 | 4368.5107 | 0.99862241 | 12201.424 | 0.35753962 | not_implemented |

## Caveats

- `selected/total` is retained only as a labeled non-apples-to-apples diagnostic for continuity with the provisional sidecar.
- The partial x0+epsilon shared-rollout comparator is stress-test-only; expected-cost wording is allowed only when the extLQG x0-only sanity check passes. Current status: `warning`.
- The apples-to-apples scalar for the available GRU validation records is restricted to rows whose run spec declares the full analytical Q/R/Q_f objective; the deterministic extLQG term is not interchangeable with the covariance-inclusive expected cost.
- This sidecar is diagnostic only and is not a standard-certificate gate.
- GRU values are validation-selected realized full-QRF scalars; the shared-rollout and split-bank blocks are audit-only post-hoc rescores and are not used for checkpoint selection.
- The x0+epsilon shared-rollout block is stress-test-only unless the extLQG x0-only sanity check supports expected-cost wording.

Full same-noise-bank Monte Carlo: `not_implemented` - full shared sensory/command/motor noise is not exposed for both arms. Partial shared-rollout replacement: `available_with_limitations` - shared-rollout comparator materialized common random inputs for initial state and process/load epsilon; sensory and command/motor noise are explicitly not shared under the current GRU graph contract

Per-term realized scoring: `not_implemented` - validation checkpoint manifests currently expose scalar full-QRF objectives, not running-state, terminal-state, command, force/filter, and disturbance-integrator contributions

## Shared-rollout comparator

Bank `cs_lss_shared_x0_epsilon_v1` uses 32 trials, seed `20260603`, shared initial states, and shared process/load epsilon.

Limitation: This is a shared initial-state plus process/load epsilon comparator. Sensory and command/motor noise are explicitly not claimed as shared.

| run | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_target_perturbation_fullqrf_warmcos__lr1e-3_clip5_b64` | 335093.65 | 195186.48 | 1.7167872 | 1.7651281 | 1.6864984 | 0.44243377 | 0.46559148 | 1 |
| `fixed_target_perturbation_fullqrf_warmcos__lr3e-3_clip5_b64` | 365194.25 | 195186.48 | 1.8710018 | 1.9161722 | 1.8739123 | 0.58830634 | 0.59555708 | 1 |
