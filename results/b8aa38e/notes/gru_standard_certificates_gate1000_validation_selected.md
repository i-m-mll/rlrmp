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
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.78949 | 1.02767 | 1.4237 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.91627 | 1.01233 | 0.998155 | not_applicable | not_applicable | not_applicable | mixed |
