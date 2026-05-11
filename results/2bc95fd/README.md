# Anti-anticipation + loss-shape 6-cell matrix

Pre-reg: `2bc95fd` (closed/merged 2026-05-09).
Phase: `f695729` (variance-collapse / methodology-fix).

Tests whether anti-anticipation loss terms (`nn_output_pre_go`, `nn_hidden_derivative`) and historical loss-shape variants ((t/N)^6 + terminal cliff vs terminal velocity only) can collapse inter-replicate peak-velocity variance while suppressing pre-go effector drift. All cells share GRU architecture, jerk loss 1e5, n_replicates=5, 12,000 warmup batches, batch size 250, seed 42; varies in the anti-anticipation lever and loss-shape config.

**Headline (RMSE-primary)**: the combo cell (`gru__jerk_motor_smooth_combo`) wins on every measured axis — velocity-RMSE-ratio 0.036 (prior best: 0.758 in baseline GRU/jerk matrix), hold drift 4.6 mm, peak velocity 1.22 m/s. Three cells beat the 0.50 RMSE-ratio threshold: motor_pre (0.461), smooth_high (0.037), and combo (0.036). Loss-shape variants did not deliver. Hold-drift motor diagnostic confirms motor commands are non-zero during hold at 17.7% of movement magnitude — drift is motor-command-driven, not mechanical.

CV (SD/mean of peak vel) was reported as the primary metric in earlier writeups; RMSE-ratio supersedes it as the operative metric matching `baseline_jerk_vrnn_matrix`. See `notes/variance_analysis.md` for full table.

**Layout**:
- `RUN_PLAN.md` — operational plan with the 6 production CLI invocations
- `runs/<group>__<variant>/run.json` — per-cell hyperparameter specs
- `notes/variance_analysis.md` — variance analysis, RMSE-primary table
- `notes/combo_hold_motor_diagnostic.md` — hold-period motor output diagnostic
- `notes/cell2_spike_investigation.md` — debunk of pre-go-spike artifact + finding that the pre-go penalty saturates
- `figures/rmse_ratio_comparison/` — RMSE ratio bar chart (primary metric)
- `figures/<fig>/spec.json` — figure specs (HTML renders in gitignored `_artifacts/.../figures/<fig>/figure.html`)
