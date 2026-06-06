# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `b35595c`.

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
| target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.81695 | 1.68391 | 1.6835 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82824 | 1.78861 | 1.80129 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.83898 | 2.07784 | 2.01115 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.81663 | 1.93922 | 1.8916 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82477 | 2.01989 | 2.05112 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82568 | 2.17374 | 2.29158 | not_applicable | not_applicable | not_applicable | mixed |
