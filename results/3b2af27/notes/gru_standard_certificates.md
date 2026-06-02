# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `3b2af27`.

This materialization applies the standard certificate umbrella contract to the
two locally synced C&S stochastic GRU pilot rows. The rows use
`empirical_nonlinear` mode. Clean rollout action behavior is available;
same-coordinate transition, value, and Bellman components are explicitly
`not_applicable`. Observation-to-action response-map components are evaluated
under the shared 4D delayed position/velocity feedback contract; disturbance and
measurement-output response maps remain unavailable for these GRU rows.

## Observation-contract blocker

_None for observation-to-action maps._

## Rows

| run | status | action mismatch | obs-action map | transition | value | Bellman | class |
|---|---|---:|---:|---|---|---|---|
| lss_12k__no_hidden_penalty__nominal_clean | partial_standard_certificate_blocked | 0.840387 | 640.292 | not_applicable | not_applicable | not_applicable | mixed |
| lss_12k__hidden_penalty__nominal_clean | partial_standard_certificate_blocked | 1.33804 | 1.13018 | not_applicable | not_applicable | not_applicable | mixed |
