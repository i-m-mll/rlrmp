# Phase-Aware Linear Recurrent Output-Feedback Bridge

Issue: `5e55f69`. Umbrella: `43e8728`.

Scope: Trainable phase-aware linear recurrent output-feedback rows using delayed observations, previous actions, and explicit polynomial phase/time inputs.

Non-goals: No GRU training, robust/H-infinity training arm, formal game-card change, or affine tracker implementation.

Runtime: `34.93` seconds.

Verdict: The d_h=48 trainable linear recurrence rows were materialized, but nominal bridge recovery is not established by the scratch reward rows (clean ratio 13.02, Riccati-epsilon ratio 98.87; imitation diagnostic ratio 314).

## Retained Rows

| row | status | train dist | objective ratio | action mismatch | spectral radius | hidden max | failure |
|---|---|---|---:|---:|---:|---:|---|
| linear_recurrent__linrec_clean_scratch_baseline | trainable_recurrence_pending_augmented_certificate | clean_nominal | 13.019353 | 1099766.9 | 0.80267814 | 4.7060338 | optimizer_basin |
| linear_recurrent__linrec_riccati_eps_scratch | trainable_recurrence_pending_augmented_certificate | riccati_epsilon | 98.865743 | 624907.84 | 1.0091333 | 6.1701856 | optimizer_basin |
| linear_recurrent__linrec_state_eig_scratch | trainable_recurrence_pending_augmented_certificate | state_eigenspectrum | 10.045708 | 6318914.5 | 0.92769028 | 7.8330119 | optimizer_basin |
| linear_recurrent__linrec_observer_error_scratch | trainable_recurrence_pending_augmented_certificate | observer_error | 13.019353 | 1099766.9 | 0.80267814 | 4.7060338 | optimizer_basin |
| linear_recurrent__linrec_mixed_scratch | trainable_recurrence_pending_augmented_certificate | mixed_deviation | 260.52215 | 1281896.9 | 0.97724659 | 6.2454409 | optimizer_basin |
| linear_recurrent__linrec_imitation_nominal | trainable_recurrence_pending_augmented_certificate | nominal | 313.99864 | 430663.55 | 0.97674886 | 18.898461 | optimizer_basin |
| linear_recurrent__linrec_imitation_mixed_then_rollout | trainable_recurrence_pending_augmented_certificate | mixed_deviation | 259.02639 | 7053035.8 | 0.94188948 | 13.161328 | optimizer_basin |

## Certificate Boundary

Formal static-gain components are not silently treated as passes for the
linear recurrence. They are explicit `not_applicable` rows because the
controller has recurrent hidden state and no global static gain over the
certificate state.

- `bellman_hessian_residual:not_applicable`: 7
- `closed_loop_transition_mismatch:not_applicable`: 7
- `optimizer_metadata:available`: 7
- `recurrence_gru_diagnostics:available`: 7
- `state_weighted_action_mismatch:available`: 7
- `value_policy_gap:not_applicable`: 7
- `visited_subspace_diagnostics:available`: 7

Augmented-state recurrent certificate components are pending the `b8b533e`
shared interface. This materializer saves plant states, observations, actions,
hidden states, and controller matrices (`A_h`, `B_y`, `B_u`, `B_phi`, `b`,
`C_h`, `D_y`, `D_phi`, `c`) for that adapter rather than inventing a competing
schema here.

## Failure Diagnostics

The failure rows use a recurrence-compatible subset of `c45adde`: clean
objective ratio, state-weighted action mismatch, recurrence diagnostics, and
the standard failure classifier where its inputs are meaningful. Gain-subspace
decomposition is `not_applicable` for these retained recurrent rows.

## Phase/Time Input

Features: `['phase_bias', 'phase_tau', 'phase_tau_squared']`.
No-phase replay ablation training RMSE: `1.0595994`;
phase-aware training RMSE: `0.26139985`.
