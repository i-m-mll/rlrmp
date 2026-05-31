# Phase-Aware Linear Recurrent Output-Feedback Bridge

Issue: `5e55f69`. Umbrella: `43e8728`.

Scope: Phase-aware linear recurrent output-feedback rows using delayed observations plus explicit polynomial phase/time inputs.

Non-goals: No GRU training, robust/H-infinity training arm, formal game-card change, or affine tracker implementation.

Runtime: `3.54` seconds.

Verdict: The phase-aware least-squares reference-replay recurrence improves the no-coverage clean objective ratio versus scratch (1.005 vs 40.52), but remains an audit row rather than a formal static-gain certificate pass.

## Retained Rows

| row | status | train dist | objective ratio | action mismatch | spectral radius | hidden max | failure |
|---|---|---|---:|---:|---:|---:|---|
| linear_recurrent__no_coverage__scratch_seed_0 | recurrence_audit_not_formal_static_gain | none | 40.516514 | 2398.914 | 0.85 | 1.5901022 | optimizer_basin |
| linear_recurrent__no_coverage__reference_replay | recurrence_audit_not_formal_static_gain | nominal | 1.004933 | 45318.452 | 0.85 | 5.1728925 | uncertain |
| linear_recurrent__state_eigenspectrum_m4_s1_w0p1__reference_replay | recurrence_audit_not_formal_static_gain | eigenspectrum_state | 1.4219178 | 986451.79 | 0.85 | 5.8175194 | optimizer_basin |
| linear_recurrent__state_eigenspectrum_m4_s3_w0p1__reference_replay | recurrence_audit_not_formal_static_gain | eigenspectrum_state | 1.4282834 | 309803.37 | 0.85 | 6.0411545 | optimizer_basin |
| linear_recurrent__observer_error_state_m1_s0p3_w0p1__reference_replay | recurrence_audit_not_formal_static_gain | observer_error | 1.116801 | 445032.83 | 0.85 | 5.5971089 | optimizer_basin |

## Certificate Boundary

Formal static-gain components are not silently treated as passes for the
linear recurrence. They are explicit `not_applicable` rows because the
controller has recurrent hidden state and no global static gain over the
certificate state.

- `bellman_hessian_residual:not_applicable`: 5
- `closed_loop_transition_mismatch:not_applicable`: 5
- `optimizer_metadata:available`: 5
- `recurrence_gru_diagnostics:available`: 5
- `state_weighted_action_mismatch:available`: 5
- `value_policy_gap:not_applicable`: 5
- `visited_subspace_diagnostics:available`: 5

## Failure Diagnostics

The failure rows use a recurrence-compatible subset of `c45adde`: clean
objective ratio, state-weighted action mismatch, recurrence diagnostics, and
the standard failure classifier where its inputs are meaningful. Gain-subspace
decomposition is `not_applicable` for these retained recurrent rows.

## Phase/Time Input

Features: `['phase_bias', 'phase_tau', 'phase_tau_squared']`.
No-phase replay ablation training RMSE: `1.0595994`;
phase-aware training RMSE: `0.19500062`.
