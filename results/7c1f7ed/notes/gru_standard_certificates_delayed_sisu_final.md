# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `7c1f7ed`.

This materialization applies the standard certificate umbrella contract to the
locally synced C&S stochastic GRU pilot rows. The rows use
`empirical_nonlinear` mode. Clean rollout action behavior is available;
same-coordinate transition, value, and Bellman components are explicitly
`not_applicable`. Observation-to-action response-map components are evaluated
under the shared 4D delayed position/velocity feedback contract; disturbance and
measurement-output response maps remain unavailable for these GRU rows.

## Observation-contract blocker

- Response-map components are blocked: the 7c1f7ed Feedbax GraphSpec (None) feeds the GRU 6D delayed position/velocity plus force-filter feedback (6D), while the current C&S output-feedback reference uses delayed_observation_matrix over the full physical block (8D). No approved 6D-to-8D projection or 6D analytical reference response-map contract is present.

## Rows

| run | status | action mismatch | obs-action map | cov-weighted obs-action | transition | value | Bellman | class |
|---|---|---:|---:|---:|---|---|---|---|
| delayed_sisu_spectrum__raw_strong_gamma_1p05_radius_lr1e-2_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.65003 | n/a | n/a | not_applicable | not_applicable | not_applicable | external_rollout_mismatch |
| delayed_sisu_spectrum__effective_020a65b_pgd_radius_lr1e-2_clip5_b64__nominal_clean | partial_standard_certificate_blocked | 1.69289 | n/a | n/a | not_applicable | not_applicable | not_applicable | external_rollout_mismatch |
