# RunPod delayed 8D no-PGD catch/prego rows, 2026-06-09

## Remote setup

- Pod: `5zjzbgmot5rqxv`
- Cloud/GPU: RunPod secure cloud, `EU-CZ-1`, `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204`
- SSH probe: `.ssh.ssh_command` populated, then `nvidia-smi` verified RTX 4090 with driver `595.71.05`
- Environment: synced current dirty rlrmp worktree, Feedbax `develop`, and jax-cookbook; patched editable paths to `/workspace/feedbax` and `/workspace/jax-cookbook`; ran `uv sync` under `nohup`; installed `jax[cuda12]`; verified `uv run --no-sync python -c 'import jax; print(jax.__version__); print(jax.devices())'` printed `0.10.1` and `[CudaDevice(id=0)]`

Secure 4090 allocation first failed in `EUR-IS-2` and `CA-MTL-3`; `EU-CZ-1` succeeded. The first launch attempt failed immediately because the Feedbax rsync excluded `feedbax/feedbax/web`, which current rlrmp imports for graph manifests. I synced that Python package, verified `from feedbax.web.models.graph import GraphSpec`, and relaunched both rows successfully.

## Training rows

Both rows ran in parallel on the single RTX 4090 with final-checkpoint artifacts retrieved locally:

```bash
uv run --no-sync python -m rlrmp.train.cs_nominal_gru \
  --output-dir _artifacts/6c36536/runs/<run_id> \
  --spec-dir results/6c36536/runs/<run_id> \
  --issue 6c36536 --seed 42 --n-train-batches 12000 --batch-size 64 \
  --controller-lr 0.003 --gradient-clip-norm 5 --n-replicates 5 \
  --hidden-size 180 --plant-backend cs_lss --stochastic-preset cs2019-rollout \
  --loss-objective full_analytical_qrf --target-relative-multitarget \
  --delayed-reach --delayed-reach-p-catch-trial <p_catch> \
  --force-filter-feedback --nn-output-pre-go <prego> --full-train \
  --training-diagnostics --checkpoint-interval-batches 1000 --log-step 100 \
  --quiet-progress
```

| Run ID | p_catch | pre-go penalty | Status |
|---|---:|---:|---|
| `delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42` | 0.50 | 1e4 | completed 12000/12000 |
| `delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42` | 0.40 | 1e5 | completed 12000/12000 |

Retrieved tracked specs:

- `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/run.json`
- `results/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/model.graph.manifest.json`
- `results/6c36536/runs/delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/run.json`
- `results/6c36536/runs/delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/model.graph.manifest.json`

Retrieved bulk artifacts:

- `_artifacts/6c36536/runs/delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42/`
- `_artifacts/6c36536/runs/delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42/`
- `_artifacts/6c36536/runpod_logs/`

The pod was stopped after retrieval. `runpodctl pod get 5zjzbgmot5rqxv` reported `desiredStatus: EXITED` and `Exited by user: Tue Jun 09 2026 22:31:00 GMT+0000`.

## Velocity figures

Generated with final checkpoints, not validation/fixed-bank selected checkpoints. The evaluation bank is the corrected fixed delayed-reach bank: separate `no_catch` and `catch`, `direction_source: uniform_grid`, 20 directions, `reach_length_m: 0.15`, `direction_source_inferred_from_validation_targets: false`, go-cue steps 10 through 30, and 10 pre-go context steps retained in the go-aligned figure window.

Command:

```bash
uv run --no-sync python scripts/materialize_gru_velocity_profiles.py \
  --experiment 6c36536 \
  --run-id delayed_8d_no_pgd_catch0p5_prego1e4_lr3e-3_clip5_b64_seed42 \
  --label 'p_catch=0.50 prego=1e4 lr=3e-3' \
  --run-id delayed_8d_no_pgd_catch0p40_prego1e5_lr3e-3_clip5_b64_seed42 \
  --label 'p_catch=0.40 prego=1e5 lr=3e-3' \
  --topic delayed_8d_no_pgd_catch_prego_new_rows_fixed_bank_velocity_final_checkpoints \
  --pre-go-context-steps 10 --delayed-eval-bank both --include-reference
```

Outputs:

- No-catch HTML with matched 0.15 m extLQG traces: `_artifacts/6c36536/figures/delayed_8d_no_pgd_catch_prego_new_rows_fixed_bank_velocity_final_checkpoints/no_catch/forward_velocity_profiles_stochastic.html`
- No-catch summary: `_artifacts/6c36536/figures/delayed_8d_no_pgd_catch_prego_new_rows_fixed_bank_velocity_final_checkpoints/no_catch/velocity_profile_summary.json`
- Catch HTML: `_artifacts/6c36536/figures/delayed_8d_no_pgd_catch_prego_new_rows_fixed_bank_velocity_final_checkpoints/catch/forward_velocity_profiles_stochastic.html`
- Catch summary: `_artifacts/6c36536/figures/delayed_8d_no_pgd_catch_prego_new_rows_fixed_bank_velocity_final_checkpoints/catch/velocity_profile_summary.json`

Focused checks confirmed both HTML files exist and the summaries record the bank kind, uniform source, 20 directions, 0.15 m reach length, final-checkpoint policy, and extLQG references on the no-catch output only.
