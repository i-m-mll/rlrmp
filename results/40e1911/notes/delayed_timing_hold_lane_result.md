# Delayed Timing / Pre-Go Hold Lane Result

This is a scoped phase experiment result for [issue:40e1911], not merely
associated delayed-reach context. It covers the second-dispatch delayed timing
and pre-go hold lane across [issue:6c36536], [issue:bf71d86], and
[issue:ef9c882].

## Run Contract

All six rows completed 12000/12000 training batches on the RunPod RTX 5090
lane and were post-run materialized locally. Shared defaults were: no H0
encoder, target-visible delayed reach, scalar go cue, force/filter feedback,
target-relative multi-target, full Q/R/Qf, 5 replicates, batch size 64, hidden
size 180, `p_catch_trial=0.5`, controller LR `3e-3`, warmup-cosine schedule,
500 warmup batches, cosine alpha `0.1`, final LR `3e-4`, gradient clip norm
`5`, and the movement-age 45/45/10 perturbation bank.

Rows:

- `6c36536/baseline__delayed_repeat`: corrected delayed movement-bank baseline
  repeat with `gradient_clip_norm=5` restored, `nn_output_pre_go=1e5`, no extra
  pre-go hold penalties.
- `bf71d86/timing__fixed_go10`: fixed go cue at step 10.
- `bf71d86/timing__fixed_go20`: fixed go cue at step 20.
- `bf71d86/timing__go10_15`: narrow go-cue range `[10, 15]`.
- `ef9c882/hold__force_filter`: `nn_output_pre_go=0`,
  `delayed_pre_go_force_filter_hold=1e5`.
- `ef9c882/hold__start_pos_zero_vel`: `nn_output_pre_go=0`,
  `delayed_pre_go_start_pos_hold=1e6`,
  `delayed_pre_go_zero_vel_hold=1e5`.

## Headline

No row solves the delayed-reach kinematic mismatch. The narrow go-cue range is
closest on no-catch peak velocity, but it is still about 5% below the 8D extLQG
reference. The start-position plus zero-velocity hold row is the most plausible
shape improvement by eye around early and late movement, but its peak remains
depressed and it introduces slight pre-go anticipation that the other rows do
not show. The force/filter hold row does not demonstrate the hoped-for filter
effect at matched `1e5` penalty scale and is worst on peak velocity.

No-catch target-radial velocity summary:

| Row | Peak velocity (m/s) | Time to peak (s) | Gap vs 8D extLQG |
|---|---:|---:|---:|
| `timing__go10_15` | 0.694837 | 0.16 | -4.95% |
| `baseline__delayed_repeat` | 0.688341 | 0.17 | -5.84% |
| `timing__fixed_go10` | 0.683701 | 0.16 | -6.48% |
| `timing__fixed_go20` | 0.680533 | 0.16 | -6.91% |
| `hold__start_pos_zero_vel` | 0.676107 | 0.16 | -7.52% |
| `hold__force_filter` | 0.672467 | 0.17 | -8.01% |

Reference sidecar: the 8D extLQG/output-feedback comparator peaks at
`0.731057` m/s at `0.16` s; the 4D extLQG/output-feedback comparator peaks at
`0.730759` m/s at `0.16` s.

## Interpretation

The corrected baseline repeat confirms that restoring gradient clipping alone
does not fix the depressed delayed movement peak. Timing regularization helps
slightly when the task is narrowed to go cues 10-15, but fixed go-cue timing is
not sufficient. Replacing `nn_output_pre_go` with force/filter hold at the same
`1e5` scale does not help, so the filter state penalty is not the missing
mechanism in this configuration. The start-position plus zero-velocity hold
constraint is the only row that appears to move the profile shape toward the
analytical comparator outside the peak, but the remaining peak depression and
small pre-go leak mean it is a bracketed result, not a solution.

Recommended phase read: keep this lane as negative/diagnostic evidence for the
delayed-reaching kinematics track. It argues against spending the next run only
on more fixed go-cue timing or matched force/filter hold scale. The most useful
follow-up, if delayed kinematics remain a priority, is likely a different
mechanism such as staged/curriculum acquisition, direct movement-period shaping,
or a targeted loss that raises peak velocity without reopening pre-go
anticipation.

## Durable Outputs

- Six-row velocity figure spec:
  `results/40e1911/figures/delayed_timing_hold_lane_velocity_profiles/spec.json`
- Six-row velocity manifest:
  `results/40e1911/figures/delayed_timing_hold_lane_velocity_profiles/manifest.json`
- Figure index:
  `results/40e1911/notes/delayed_timing_hold_lane_velocity_profiles.md`
- No-catch aggregate velocity plot:
  `_artifacts/40e1911/figures/delayed_timing_hold_lane_velocity_profiles/no_catch/forward_velocity_profiles_stochastic.html`
- No-catch by-replicate velocity plot:
  `_artifacts/40e1911/figures/delayed_timing_hold_lane_velocity_profiles/no_catch/forward_velocity_profiles_by_replicate_stochastic.html`
- Catch aggregate velocity plot:
  `_artifacts/40e1911/figures/delayed_timing_hold_lane_velocity_profiles/catch/forward_velocity_profiles_stochastic.html`
- Catch by-replicate velocity plot:
  `_artifacts/40e1911/figures/delayed_timing_hold_lane_velocity_profiles/catch/forward_velocity_profiles_by_replicate_stochastic.html`
- Raw perturbation-response rollout deletion manifest:
  `results/40e1911/notes/perturbation_response_npz_deletion_manifest.json`

Full calibrated post-run materialization notes and regeneration specs are under
`results/6c36536/notes/`, `results/bf71d86/notes/`, and
`results/ef9c882/notes/` with the `delayed_timing_hold_lane` suffix.
