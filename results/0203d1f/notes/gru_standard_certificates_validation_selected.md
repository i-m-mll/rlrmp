# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `0203d1f`.

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

| run | status | action mismatch | obs-action map | transition | value | Bellman | class |
|---|---|---:|---:|---|---|---|---|
| lss_ablation_partial_net_force_filter__lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 0.772181 | 0.939795 | not_applicable | not_applicable | not_applicable | mixed |
