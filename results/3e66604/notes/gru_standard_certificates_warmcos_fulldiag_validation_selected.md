# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `3e66604`.

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
| lss_stabilization_warmcos_fulldiag__lr3e-3_clip5_b250__nominal_clean | partial_standard_certificate_blocked | 5.97746 | 1.02541 | not_applicable | not_applicable | not_applicable | mixed |
| lss_stabilization_warmcos_fulldiag__lr1e-3_clip5_b250__nominal_clean | partial_standard_certificate_blocked | 0.0234901 | 0.940478 | not_applicable | not_applicable | not_applicable | mixed |
| lss_stabilization_warmcos_fulldiag__lr3e-3_clip5_b1000__nominal_clean | partial_standard_certificate_blocked | 5.98206 | 1.00932 | not_applicable | not_applicable | not_applicable | mixed |
| lss_stabilization_warmcos_fulldiag__lr1e-3_clip5_b1000__nominal_clean | partial_standard_certificate_blocked | 0.0689938 | 0.942003 | not_applicable | not_applicable | not_applicable | mixed |
