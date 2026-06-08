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
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_none_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.81582 | 1.68266 | 1.67945 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_small_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82905 | 1.79324 | 1.80623 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_moderate_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.83803 | 2.0832 | 2.02882 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_moderate_cal_stress_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.84493 | 3.43322 | 3.10075 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_none_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.8175 | 1.68696 | 1.6809 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_small_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.82914 | 1.79869 | 1.80963 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_moderate_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.83692 | 2.08526 | 2.03142 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__broad_strong_cal_stress_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.85906 | 3.4879 | 3.0748 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__proprio_none_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.75902 | 1.04792 | 0.996859 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.73734 | 1.06775 | 0.991573 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_moderate_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.76777 | 1.10316 | 0.98584 | not_applicable | not_applicable | not_applicable | mixed |
| target_relative_multitarget_fullqrf_warmcos__proprio_cal_stress_lr1e-3_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.83853 | 1.51225 | 0.984367 | not_applicable | not_applicable | not_applicable | mixed |
