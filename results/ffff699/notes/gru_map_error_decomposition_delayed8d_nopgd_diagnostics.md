# GRU Map-Error Decomposition

Issue: `ddf7f43`. Source issue: `ffff699`.

This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action response map. It is diagnostic-only; the standard certificate gate remains the standard response-map/action evidence.

## Rows

| Row | norm ratio | cosine | scalar gain | scalar residual | top error | covariance | annotations |
|---|---:|---:|---:|---:|---:|---|---|
| delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42__nominal_clean | blocked | blocked | blocked | blocked | blocked | not_available | Response-map components are blocked: the ffff699 Feedbax GraphSpec (None) feeds the GRU feedback basis `target_relative_delayed_feedback_plus_force_filter` (6D), while the current standard response-map reference uses the approved delayed position/velocity observation basis (4D). No approved standard-certificate projection for this candidate feedback basis is present. |

## Top Singular Directions

### `delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42__nominal_clean`

Blocked: Response-map components are blocked: the ffff699 Feedbax GraphSpec (None) feeds the GRU feedback basis `target_relative_delayed_feedback_plus_force_filter` (6D), while the current standard response-map reference uses the approved delayed position/velocity observation basis (4D). No approved standard-certificate projection for this candidate feedback basis is present.
