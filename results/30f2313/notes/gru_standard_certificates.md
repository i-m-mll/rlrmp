# GRU Standard Certificates

Issue: `e6a32b8`. Source run issue: `30f2313`.

This materialization applies the standard certificate umbrella contract to the
two locally synced C&S stochastic GRU pilot rows. The rows use
`empirical_nonlinear` mode. Clean rollout action behavior is available;
same-coordinate transition, value, and Bellman components are explicitly
`not_applicable`; response-map components remain blocked rather than inferred
across incompatible observation contracts.

## Blocker

- Response-map components are blocked: the 30f2313 Feedbax GraphSpec (model.graph.json) feeds the GRU delayed position/velocity feedback (4D), while the current C&S output-feedback reference uses delayed_observation_matrix over the full physical block (8D). No approved 4D-to-8D projection or 4D analytical reference response-map contract is present.

## Rows

| run | status | action mismatch | transition | value | Bellman | class |
|---|---|---:|---|---|---|---|
| cs_stochastic_gru__no_hidden_penalty__nominal_clean | partial_standard_certificate_blocked | 0.371861 | not_applicable | not_applicable | not_applicable | io_map_mismatch |
| cs_stochastic_gru__hidden_penalty__nominal_clean | partial_standard_certificate_blocked | 0.850205 | not_applicable | not_applicable | not_applicable | io_map_mismatch |
