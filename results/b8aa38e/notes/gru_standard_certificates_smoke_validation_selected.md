# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `b8aa38e`.

This materialization applies the standard certificate umbrella contract to the
locally synced C&S stochastic GRU pilot rows. The rows use
`empirical_nonlinear` mode. Clean rollout action behavior is available;
same-coordinate transition, value, and Bellman components are explicitly
`not_applicable`. Observation-to-action response-map components are evaluated
under the shared 4D delayed position/velocity feedback contract; disturbance and
measurement-output response maps remain unavailable for these GRU rows.

## Observation-contract blocker

_None for observation-to-action maps._

## Rows

| run | status | action mismatch | obs-action map | cov-weighted obs-action | transition | value | Bellman | class |
|---|---|---:|---:|---:|---|---|---|---|
| smoke__broad_strong_cal_small__nominal_clean | partial_standard_certificate_blocked | 1.00029 | 1.00007 | 0.999958 | not_applicable | not_applicable | not_applicable | mixed |
| smoke__proprio_cal_stress__nominal_clean | partial_standard_certificate_blocked | 0.999946 | 0.99932 | 0.999429 | not_applicable | not_applicable | not_applicable | mixed |
