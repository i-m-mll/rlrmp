# Anti-anticipation + loss-shape 6-cell matrix

Pre-reg: `2bc95fd` (closed/merged 2026-05-09).
Phase: `f695729` (variance-collapse / methodology-fix).

Tests whether anti-anticipation loss terms (`nn_output_pre_go`, `nn_hidden_derivative`) and historical loss-shape variants ((t/N)^6 + terminal cliff vs terminal velocity only) can collapse inter-replicate peak-velocity variance while suppressing pre-go effector drift. All cells share GRU architecture, jerk loss 1e5, n_replicates=5, 12,000 warmup batches, batch size 250, seed 42; varies in the anti-anticipation lever and loss-shape config.

**Headline**: the combo cell (`gru__jerk_motor_smooth_combo`) wins on every measured axis — CV 0.007, hold drift 4.6 mm, peak velocity 1.22 m/s. Hidden smoothness 1e2 alone collapses replicate variance dramatically but increases anticipation drift by 80%; pre-go motor mask 1e-2 corrects that. Loss-shape variants (terminal velocity only; (t/N)^6 + cliff) did not deliver — deprioritized as anti-anticipation levers. See `notes/variance_analysis.md` for full table + methodological caveat (variance-metric mismatch with prior `baseline_jerk_vrnn_matrix`).

**Layout**:
- `RUN_PLAN.md` — operational plan with the 6 production CLI invocations
- `runs/<group>__<variant>/run.json` — per-cell hyperparameter specs
- `notes/variance_analysis.md` — peak-velocity + hold-drift analysis
- `notes/cell2_spike_investigation.md` — debunk of pre-go-spike artifact + finding that the pre-go penalty saturates rather than collapsing pre-go activity
- `figures/<fig>/spec.json` — figure specs (HTML renders in gitignored `_artifacts/.../figures/<fig>/figure.html`)

Note: this experiment lives at `results/part2_5/runpod/<exp>/` due to legacy nesting. Per `f485c26`, the `runpod/` subdir is non-canonical and should flatten in a future layout reorganization.
