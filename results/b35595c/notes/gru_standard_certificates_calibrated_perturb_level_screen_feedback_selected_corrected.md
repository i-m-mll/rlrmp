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
| target_relative_multitarget_fullqrf_warmcos__none_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82691 | 1.65841 | 1.67396 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__none_lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.80509 | 1.75834 | 1.7809 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.83076 | 1.7832 | 1.79189 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_small_lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.81753 | 1.9086 | 1.97788 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.84395 | 2.06981 | 1.97739 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__cal_moderate_lr3e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82918 | 2.26786 | 2.34615 | not_applicable | not_applicable | not_applicable | mixed |
