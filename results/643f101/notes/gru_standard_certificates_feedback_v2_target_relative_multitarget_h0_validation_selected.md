# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `643f101`.

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
| target_relative_multitarget_h0_fullqrf_warmcos__lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.77537 | 1.29938 | 1.46605 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_h0_fullqrf_warmcos__lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.79233 | 1.82774 | 1.82619 | not_applicable | not_applicable | not_applicable | mixed |
