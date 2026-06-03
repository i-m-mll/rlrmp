# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v3`.

Scope: full-QRF C&S GRU rows, validation-selected checkpoints.

This is an objective-lens diagnostic, not a standard-certificate gate.

## Objective lenses

| lens | status | comparability |
|---|---|---|
| deterministic extLQG | available | deterministic full-Q/R/Q_f initial-state term; comparable only to full-Q/R/Q_f realized scalars |
| covariance-inclusive extLQG expected cost | available | not directly comparable to realized GRU validation scalars |
| realized GRU validation | available for full-Q/R/Q_f scalar rows | validation-selected audit metric, not checkpoint selection input |
| same-noise-bank Monte Carlo | not_implemented | requires shared realized noise bank for GRU and extLQG |
| realized per-term full-Q/R/Q_f scoring | not_implemented | requires scorer output for running state, terminal, command, force/filter, and disturbance-integrator terms |

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
- GRU values are validation-selected realized full-QRF scalars; same-noise-bank extLQG realized values require separate Monte Carlo materialization.

Same-noise-bank Monte Carlo: `not_implemented` - same-noise-bank extLQG-vs-GRU realized comparison was not materialized; the available tracked source only contains validation-selected GRU realized full-QRF scalars and the analytical extLQG expected-cost decomposition

Per-term realized scoring: `not_implemented` - validation checkpoint manifests currently expose scalar full-QRF objectives, not running-state, terminal-state, command, force/filter, and disturbance-integrator contributions
