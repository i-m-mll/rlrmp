# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v4`.

Scope: validation-selected checkpoints for C&S GRU runs: lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64, lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64.

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

## extLQG decomposition

| component | value | lens |
|---|---:|---|
| deterministic initial-state term | 4368.5107 | comparable to realized/validation full-QRF values |
| initial covariance trace term | 7775.5302 | expected-cost sidecar only |
| accumulated noise scalar | 57.383523 | expected-cost sidecar only |
| total expected cost | 12201.424 | not directly comparable to GRU validation values |

## GRU comparison

| run | row comparability | mean selected validation | deterministic extLQG | selected/deterministic | total expected cost | selected/total | per-term scoring |
|---|---|---:|---:|---:|---:|---:|---|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | comparable_deterministic_full_qrf | 4367.3012 | 4368.5107 | 0.99972314 | 12201.424 | 0.35793372 | not_implemented |
| `lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64` | comparable_deterministic_full_qrf | 4342.6365 | 4368.5107 | 0.99407713 | 12201.424 | 0.35591226 | not_implemented |

## Caveats

- `selected/total` is retained only as a labeled non-apples-to-apples diagnostic for continuity with the provisional sidecar.
- The apples-to-apples scalar for the available GRU validation records is restricted to rows whose run spec declares the full analytical Q/R/Q_f objective; the deterministic extLQG term is not interchangeable with the covariance-inclusive expected cost.
- This sidecar is diagnostic only and is not a standard-certificate gate.
- GRU values are validation-selected realized full-QRF scalars; the shared-rollout block is an audit-only post-hoc rescore and is not used for checkpoint selection.

Full same-noise-bank Monte Carlo: `not_implemented` - full shared sensory/command/motor noise is not exposed for both arms. Partial shared-rollout replacement: `available_with_limitations` - shared-rollout comparator materialized common random inputs for initial state and process/load epsilon; sensory and command/motor noise are explicitly not shared under the current GRU graph contract

Per-term realized scoring: `not_implemented` - validation checkpoint manifests currently expose scalar full-QRF objectives, not running-state, terminal-state, command, force/filter, and disturbance-integrator contributions

## Shared-rollout comparator

Bank `cs_lss_shared_x0_epsilon_v1` uses 32 trials, seed `20260603`, shared initial states, and shared process/load epsilon.

Limitation: This is a shared initial-state plus process/load epsilon comparator. Sensory and command/motor noise are explicitly not claimed as shared.

| run | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 307479.24 | 195186.48 | 1.5753102 | 1.6206814 | 1.5203112 | 0.45767711 | 0.49109855 | 1 |
| `lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64` | 302971.21 | 195186.48 | 1.5522141 | 1.5988374 | 1.4790988 | 0.45866649 | 0.48003939 | 1 |
