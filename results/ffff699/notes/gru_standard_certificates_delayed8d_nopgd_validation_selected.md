# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `ffff699`.

This materialization applies the standard certificate umbrella contract to the
locally synced C&S stochastic GRU pilot rows. The rows use
`empirical_nonlinear` mode. Clean rollout action behavior is available;
same-coordinate transition, value, and Bellman components are explicitly
`not_applicable`. Observation-to-action response-map components are evaluated
under the shared 4D delayed position/velocity feedback contract; disturbance and
measurement-output response maps remain unavailable for these GRU rows.

## Observation-contract blocker

- Response-map components are blocked: the ffff699 Feedbax GraphSpec (None) feeds the GRU feedback basis `target_relative_delayed_feedback_plus_force_filter` (6D), while the current standard response-map reference uses the approved delayed position/velocity observation basis (4D). No approved standard-certificate projection for this candidate feedback basis is present.

## Rows

| run | status | action mismatch | obs-action map | cov-weighted obs-action | transition | value | Bellman | class |
|---|---|---:|---:|---:|---|---|---|---|
| delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42__nominal_clean | partial_standard_certificate_blocked | 1.01315 | n/a | n/a | not_applicable | not_applicable | not_applicable | external_rollout_mismatch |
