# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v2`.

Scope: validation-selected checkpoints for C&S GRU runs: lss_ablation_partial_net_force_filter__lr1e-3_clip5_b64.

This is an objective-lens diagnostic, not a standard-certificate gate.

## extLQG decomposition

| component | value | lens |
|---|---:|---|
| deterministic initial-state term | 4368.5107 | comparable to realized/validation full-QRF values |
| initial covariance trace term | 7775.5302 | expected-cost sidecar only |
| accumulated noise scalar | 57.383523 | expected-cost sidecar only |
| total expected cost | 12201.424 | not directly comparable to GRU validation values |

## GRU comparison

| run | mean selected validation | deterministic extLQG | selected/deterministic | total expected cost | selected/total |
|---|---:|---:|---:|---:|---:|
| `lss_ablation_partial_net_force_filter__lr1e-3_clip5_b64` | 3045.4034 | 4368.5107 | 0.69712624 | 12201.424 | 0.24959409 |

## Caveats

- `selected/total` is retained only as a labeled non-apples-to-apples diagnostic for continuity with the provisional sidecar.
- The apples-to-apples scalar for the available GRU validation records is the deterministic extLQG term, not the covariance-inclusive expected cost.
- This sidecar is diagnostic only and is not a standard-certificate gate.
- GRU values are validation-selected realized full-QRF scalars; same-noise-bank extLQG realized values require separate Monte Carlo materialization.

Same-noise-bank Monte Carlo: `not_implemented`.
