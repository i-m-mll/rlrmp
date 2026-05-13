# 06f7faf — Go-cue alignment fix

Tracking issue: `06f7faf` (label: `error`, `important`).

Originally surfaced in a velocity-profile diagnostic on `3702f54`: the
forward-velocity figures and the `within_cell_vel_rmse` /
`vel_rmse_ratio` / `pos_rmse_ratio` headline metrics in
`scripts/analyse_pregomatrix.py`, `scripts/analyse_lit_replication_6cell.py`,
and `scripts/analyse_anti_anticipation_6cell_variance.py` averaged
trial-mean profiles in *absolute trial time* before reducing across
replicates. Because `centerout`'s target-on duration is randomised per
trial, this smeared the go cue across ~150 ms and produced biased
velocity-RMSE values.

## Scope

- New shared helper: `src/rlrmp/analysis/trial_alignment.py` provides
  `align_trials(profile, idx, pad=nan)` plus two aggregation helpers
  (`replicate_mean_curves`, `pooled_trial_mean_with_band`).
- Regression test: `tests/test_trial_alignment.py`.
- Patches: the four `analyse_*.py` scripts above call the helpers; their
  spec JSON transform names are updated to reflect the alignment step.
- Notes updated with post-fix metrics:
  - `results/3702f54/notes/analysis.md`
  - `results/f47abb1/notes/variance_analysis.md`
  - `results/2bc95fd/notes/variance_analysis.md`
  - `results/410d7ac/notes/decoupling_acid_test_mvp.md` (no quantitative
    correction needed; Δv unaffected)
- HTML summary: `_artifacts/06f7faf/recent_experiments_report.html`
  (mirror of the user-side report at `/private/tmp/...`).

## Effect on conclusions

All headline conclusions are preserved:

- **3702f54** — `full_trial_pl__prego_1` still wins decisively. Vel-RMSE
  values shifted modestly upward; the winner ordering and the
  ~50× pre-go-RMS suppression are unchanged.
- **f47abb1** — ordering preserved (powerlaw + jerk-off still wins);
  absolute vel-within-RMSE values up ~5–30 %.
- **2bc95fd** — the two clear winners (`gru__jerk_smooth_high`,
  `gru__jerk_motor_smooth_combo`) hold at ~0.04. The marginal third
  winner `gru__jerk_motor_pre` flipped from 0.461 (under threshold) to
  0.552 (over) — one winner-status change.
- **410d7ac** — Δv values *identical* pre- and post-fix (the metric
  used per-trial after-go masking before `max`, so the trial axis was
  never collapsed in absolute time).

## Cross-refs

- `bf71d86` (go-cue stratification diagnostic) — proper analysis of
  early-vs-late go-cue trials. The `align_trials` helper here is the
  upstream primitive that stratification can build on.
- `4d38c15` (analyses coord) — cross-cutting impact noted there.
