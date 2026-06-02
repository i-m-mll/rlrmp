# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `30f2313`.

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
| cs_stochastic_gru__no_hidden_penalty__nominal_clean | partial_standard_certificate_blocked | 2.62757 | 1.13514 | not_applicable | not_applicable | not_applicable | mixed |
| cs_stochastic_gru__hidden_penalty__nominal_clean | partial_standard_certificate_blocked | 1.47911 | 1.1489 | not_applicable | not_applicable | not_applicable | mixed |
