# Adaptive Epsilon Target-Schedule Matrix

This is a no-launch spec lock for issue 3cd018b. It prepares the run recipes
and post-run analysis plan only. It does not authorize training, RunPod pod
creation, pushing, protected auth, protected merge, issue closure, or comments
on unrelated issues.

## Live Orientation

Current refactored RLRMP code can express the requested rows without a new
implementation pass. The adaptive-epsilon damage schedule is parameterized as
`start -> peak` over `damage_ramp_batches`, then `peak -> final` over
`damage_anneal_batches`, then final hold. Therefore:

- no-target-ramp rows use `start == peak` with `damage_ramp_batches = 1000`;
- constant rows use `start == peak == final` with the same phase lengths;
- the prior-schedule replication row uses `start = 0`, `peak = 3500`,
  `final = 1000`;
- epsilon scale uses the outer schedule `0 -> 1` over the first 1000 adaptive
  batches for every row.

The generated row specs are current-schema `train_cs_nominal_gru.py` run recipes
under `results/3cd018b/runs/`. The graph sidecars are adjacent directories under
`results/3cd018b/runs/<row>/`.

## Common Run Contract

| Field | Locked value |
|---|---|
| Baseline source | `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json` |
| Baseline checkpoint | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_0012000` |
| Baseline latest symlink | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest` |
| Baseline completed batches | 12000 |
| Total stop | global batch 16500 |
| Additional adaptive batches | 4500 |
| Phase semantics | 1000 batch target-ramp-or-hold, 2500 batch anneal, 1000 batch final hold |
| Batch size | 64 |
| Seed | 42 |
| Replicates | 5 |
| Controller | GRU, hidden size 180, initial-hidden encoder enabled |
| Plant/task | 6D no-integrator C&S LSS, force-filter feedback, target-relative `const_band16` |
| Loss | `full_analytical_qrf` |
| Controller optimizer | lr 0.003, 500-batch warmup from 0.1 fraction, cosine alpha 0.01, gradient clip 5 |
| Adversary | cap-free `direct_epsilon` soft-energy inner maximizer |
| Inner optimizer | Adam, 12 steps, lr 0.00002, zero initialization |
| Initial soft-energy lambda | 281032999.21861446 |
| Forbidden controls | no projection, no safety cap, no inherited radius, no trust region, no hard budget |
| Training method | `epsilon_scaled_outer_training` |
| Epsilon-scale schedule | linear `0 -> 1` over first 1000 adaptive batches, then hold 1 |
| Lambda update | every 50 batches, EMA alpha 0.1, eta 0.1, deadband 0.1, lambda min 1e-12, max log step 0.1 |
| Checkpoint interval | 500 batches |
| Training diagnostics | enabled, `npz+json_manifest` |
| Bulk artifact root | `_artifacts/3cd018b/runs/<row>/` |
| Tracked recipe path | `results/3cd018b/runs/<row>.json` |

Before launch, the run manager must stage the 12000-batch baseline checkpoint
into each row's output checkpoint tree, then run the row recipe with `--resume`.
The tracked specs deliberately keep row identity and scientific payload in the
recipe; launch-time flags should be runtime-only controls such as `--resume`,
progress settings, and optional smoke `--stop-after-batches`.

## Row Lock Table

| Row | Target schedule | Epsilon-scale schedule | Recipe | Bulk output |
|---|---|---|---|---|
| `hold3500_to1000` | 3500 held for 1000 adaptive batches, cosine anneal to 1000 over 2500, hold 1000 for 1000 | `0 -> 1` over first 1000, then 1 | `results/3cd018b/runs/hold3500_to1000.json` | `_artifacts/3cd018b/runs/hold3500_to1000/` |
| `hold1750_to1000` | 1750 held for 1000, anneal to 1000 over 2500, hold 1000 for 1000 | same | `results/3cd018b/runs/hold1750_to1000.json` | `_artifacts/3cd018b/runs/hold1750_to1000/` |
| `hold3500_to250` | 3500 held for 1000, anneal to 250 over 2500, hold 250 for 1000 | same | `results/3cd018b/runs/hold3500_to250.json` | `_artifacts/3cd018b/runs/hold3500_to250/` |
| `hold1750_to250` | 1750 held for 1000, anneal to 250 over 2500, hold 250 for 1000 | same | `results/3cd018b/runs/hold1750_to250.json` | `_artifacts/3cd018b/runs/hold1750_to250/` |
| `const1750` | 1750 throughout all 4500 adaptive batches | same | `results/3cd018b/runs/const1750.json` | `_artifacts/3cd018b/runs/const1750/` |
| `const1000` | 1000 throughout all 4500 adaptive batches | same | `results/3cd018b/runs/const1000.json` | `_artifacts/3cd018b/runs/const1000/` |
| `const250` | 250 throughout all 4500 adaptive batches | same | `results/3cd018b/runs/const250.json` | `_artifacts/3cd018b/runs/const250/` |
| `ramp3500_to1000` | 0 to 3500 over first 1000, anneal to 1000 over 2500, hold 1000 for 1000 | same | `results/3cd018b/runs/ramp3500_to1000.json` | `_artifacts/3cd018b/runs/ramp3500_to1000/` |

`ramp3500_to1000` is the sanity-check replication of the prior epsilon-scaled
short row under the current system. Interpret it first as a replication check
against issue 1ab1fef, not as a separate method claim.

## Diagnostics Lock

The current code and focused tests confirm that adaptive-epsilon diagnostics are
emitted through the training diagnostics manifest. Required fields for this
matrix are present:

| Need | Confirmed diagnostic fields |
|---|---|
| Target damage | `adaptive_epsilon_target_damage` |
| Lambda | `adaptive_epsilon_lambda_value` |
| Outer schedule / epsilon scale | `adaptive_epsilon_outer_weight`, `adaptive_epsilon_epsilon_scale_used`, `adaptive_epsilon_adaptive_update_epsilon_scale_used` |
| Full-strength threat damage | `adaptive_epsilon_training_batch_full_strength_damage_raw`, `adaptive_epsilon_adaptive_update_full_strength_damage_raw` |
| Applied scaled training exposure | `adaptive_epsilon_training_batch_applied_scaled_damage_raw`, `adaptive_epsilon_adaptive_update_applied_scaled_damage_raw` |
| Damage smoothing/control | `adaptive_epsilon_ema_damage`, `adaptive_epsilon_log_ratio_error`, `adaptive_epsilon_lambda_log_step` |
| Inner-adversary activity | `adaptive_epsilon_inner_selected_objective_gain_over_zero`, `adaptive_epsilon_adaptive_update_inner_selected_objective_gain_over_zero` |

If the smoke/debug run shows any of these fields missing from the materialized
diagnostics files, launch must stop and the missing diagnostic surface must be
implemented under a separate issue/worktree before the full matrix proceeds.

## Launch Gate

This plan is not launch approval. Before any billable launch, present the
current run table in chat and obtain explicit user confirmation in this child
thread. After confirmation:

1. Use a run-management subagent for execution and monitoring.
2. Acquire secure RTX 5090 capacity using the feedbax RunPod deploy scripts.
3. Retry capacity acquisition until successful within runbook and cost rules.
4. Run local smoke/debugging before cloud iteration.
5. Keep at most three rows running in parallel.
6. Use `uv run --no-sync python` on pod after CUDA-JAX setup.
7. Stop and remove pods after artifacts are synced.
8. Use `scripts/post_run.sh` dry-run first, then non-dry-run, for each completed
   row where the script covers the source shape.

Known pre-launch blocker: [issue:5d88b5c] tracks a loader registration gap found
during this spec lock. The eight generated row specs validate if the adaptive
method is registered explicitly before calling the CLI, but the normal
`--run-spec --dry-run` path currently fails before that registration happens.
Do not launch until the normal dry-run validation path succeeds for every row
without a manual pre-import.

## Post-Run Analysis Plan

The standard nominal behavior output is a velocity-profile comparison. For this
matrix, plot nominal forward or target-radial velocity over time in m/s with
shared y-axes for comparable panels. Show each trained row/checkpoint as a
solid trace. Include 6D analytical extLQG and 6D output-feedback H-infinity
nominal comparators as visually distinct reference traces where applicable.
Summaries should report peak velocity, time-to-peak, terminal position error,
terminal velocity or movement-quality stats, and any relevant loss or validation
quality statistics.

The adaptive-curriculum output is a damage/lambda plot over global training
batch. Plot target damage, measured damage, and smoothed/EMA damage together on
the damage axis. Plot adaptive lambda on a log-scaled secondary axis. For these
epsilon-scaled rows, also plot epsilon scale on a 0-to-1 scale and distinguish
full-strength threat damage from applied scaled training exposure. Labels and
line styles must make target, measured damage, EMA damage, lambda, epsilon
scale, full-strength damage, and applied-scaled damage visually separable.

The comparison set should include the prior epsilon-scaled short row from issue
1ab1fef, the earlier ramp-duration rows from issue 91a090c, and analytical
comparators where applicable. The final issue verdict should state whether the
matrix answered, bracketed, or superseded the target-schedule question.

## Reconciliation Addendum: Launched Recipe Lineage

The 2026-07-07 RunPod launch did not consume the committed row JSON files
directly. `_artifacts/3cd018b/runpod_remote_run_dir/main_20260707T164908Z/launch_row.sh`
ran `scripts/train_cs_nominal_gru.py --run-spec` against the pod-local
`/workspace/feedbax_runs/3cd018b-adaptive-target-matrix/float32_rebound_specs/results/3cd018b/runs/${row}.json`
copy. The tracked recipes have been reconciled to those pod-rebound specs with
portable `results/3cd018b/runs/<row>` path fields.

Lineage policy for future work: do not attempt to resume or fork these rows by
byte-matching the old checkpoint binding payloads. The launch already used a
new-lineage graft at `tx-ce8481ee` with `relationship: new_lineage_override`.
Future resumes/forks should use `fork_checkpoint_transaction` with
`allow_new_lineage_override`, and the fork metadata must explain that the source
checkpoint binding was produced from historical pod-local rebound specs while
the tracked recipe is the reconciled portable record.

Phase-program parity was rechecked on 2026-07-08 with
`PYTHONPATH=src uv run --no-sync python`: current
`adaptive_epsilon_method_contract().phase_program` hashes to `a3060c554b3cc7bd44e81217b7a96fa26d6967af5fc71a1d95436273e324565c`,
matching the stored checkpoint binding hash. The remaining mismatch was the
candidate recipe/projection, not adaptive-epsilon method-code drift.
