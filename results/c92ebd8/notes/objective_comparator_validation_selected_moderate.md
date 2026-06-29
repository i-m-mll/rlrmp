# Full-QRF objective comparator sidecar

Schema: `rlrmp.objective_comparator_sidecar.v6`.

Scope: validation_selected_per_replicate checkpoints for C&S GRU runs: open_loop_small, open_loop_moderate, open_loop_stress, closed_loop_small, closed_loop_moderate, closed_loop_stress, closed_loop_cmd_lateral_small, closed_loop_cmd_lateral_moderate, closed_loop_cmd_lateral_stress.

This is an objective-lens diagnostic, not a standard-certificate gate.

## Objective lenses

| lens | status | comparability |
|---|---|---|
| deterministic extLQG | available | deterministic full-Q/R/Q_f initial-state term; comparable only to full-Q/R/Q_f realized scalars |
| covariance-inclusive extLQG expected cost | available | not directly comparable to realized GRU validation scalars |
| realized GRU validation | available for full-Q/R/Q_f scalar rows | validation-selected audit metric, not checkpoint selection input |
| full same-noise-bank Monte Carlo | not_implemented | full shared sensory/command/motor noise is not exposed for both arms; see the partial shared-rollout comparator below |
| realized per-term full-Q/R/Q_f scoring | not_implemented | requires scorer output for running state, terminal, command, force/filter, and disturbance-integrator terms |
| shared-rollout comparator | blocked | shared initial-state and process/load epsilon bank; sensory/command noise limits declared |
| standard split-bank comparator | not_available | deterministic nominal, component-specific x0/process-epsilon, x0 position+velocity, and x0+epsilon audit-only lenses |

## extLQG decomposition

| component | value | lens |
|---|---:|---|
| deterministic initial-state term | 4368.5107 | comparable to realized/validation full-QRF values |
| initial covariance trace term | 7775.5302 | expected-cost sidecar only |
| accumulated noise scalar | 57.383523 | expected-cost sidecar only |
| total expected cost | 12201.424 | not directly comparable to GRU validation values |
| x0-only realized sanity | not_available | realized extLQG x0-only cost vs deterministic + initial-covariance-trace expectation |

## GRU comparison

| run | row comparability | mean selected validation | deterministic extLQG | selected/deterministic | total expected cost | selected/total | per-term scoring |
|---|---|---:|---:|---:|---:|---:|---|
| `closed_loop_moderate` | comparable_deterministic_full_qrf | 4397.6148 | 4368.5107 | 1.0066622 | 12201.424 | 0.36041815 | not_implemented |
| `closed_loop_cmd_lateral_moderate` | comparable_deterministic_full_qrf | 4404.6821 | 4368.5107 | 1.00828 | 12201.424 | 0.36099737 | not_implemented |
| `closed_loop_cmd_lateral_small` | comparable_deterministic_full_qrf | 4398.7804 | 4368.5107 | 1.0069291 | 12201.424 | 0.36051368 | not_implemented |
| `closed_loop_cmd_lateral_stress` | comparable_deterministic_full_qrf | 4414.5329 | 4368.5107 | 1.010535 | 12201.424 | 0.36180472 | not_implemented |
| `closed_loop_small` | comparable_deterministic_full_qrf | 4398.5879 | 4368.5107 | 1.006885 | 12201.424 | 0.36049791 | not_implemented |
| `closed_loop_stress` | comparable_deterministic_full_qrf | 4415.0578 | 4368.5107 | 1.0106551 | 12201.424 | 0.36184774 | not_implemented |
| `open_loop_moderate` | comparable_deterministic_full_qrf | 4392.4518 | 4368.5107 | 1.0054804 | 12201.424 | 0.35999501 | not_implemented |
| `open_loop_small` | comparable_deterministic_full_qrf | 4391.3999 | 4368.5107 | 1.0052396 | 12201.424 | 0.35990879 | not_implemented |
| `open_loop_stress` | comparable_deterministic_full_qrf | 4406.5364 | 4368.5107 | 1.0087045 | 12201.424 | 0.36114935 | not_implemented |

## Caveats

- `selected/total` is retained only as a labeled non-apples-to-apples diagnostic for continuity with the provisional sidecar.
- The partial x0+epsilon shared-rollout comparator is stress-test-only; expected-cost wording is allowed only when an extLQG x0-only sanity check passes. Current status: `not_available`.
- The apples-to-apples scalar for the available GRU validation records is restricted to rows whose run spec declares the full analytical Q/R/Q_f objective; the deterministic extLQG term is not interchangeable with the covariance-inclusive expected cost.
- This sidecar is diagnostic only and is not a standard-certificate gate.
- GRU values are validation-selected realized full-QRF scalars; the shared-rollout and split-bank blocks are audit-only post-hoc rescores and are not used for checkpoint selection.
- The x0+epsilon shared-rollout block is stress-test-only unless the extLQG x0-only sanity check supports expected-cost wording.
- Split-bank GRU hidden states are initialized from the checkpoint model default rather than conditioned on the perturbed x0, so x0 lenses are recovery stress tests rather than expected-cost comparisons.

Full same-noise-bank Monte Carlo: `not_implemented` - full shared sensory/command/motor noise is not exposed for both arms. Partial shared-rollout replacement: `not_implemented` - same-noise-bank extLQG-vs-GRU realized comparison was not materialized; the available tracked source only contains validation-selected GRU realized full-QRF scalars and the analytical extLQG expected-cost decomposition

Per-term realized scoring: `not_implemented` - validation checkpoint manifests currently expose scalar full-QRF objectives, not running-state, terminal-state, command, force/filter, and disturbance-integrator contributions

## Shared-rollout comparator

Status: `blocked` - operands could not be broadcast together with remapped shapes [original->remapped]: (1,36)  and requested shape (32,48)

## Standard split-bank comparator

Status: `not_available` - standard split-bank comparator was not supplied to the sidecar builder
