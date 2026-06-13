# Delayed No-PGD Trial-Type Normalized Loss RunPod Row

Run: `normloss_3e3_s42`

Purpose: test whether separating delayed catch/no-catch sampling from objective
weighting improves the delayed no-PGD velocity-profile match to extLQG.

## Setup

- GPU: RunPod secure RTX 4090, pod `erot70t5nrlpih`, image
  `runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204`.
- JAX: installed after `uv sync` with `uv pip install -U "jax[cuda12]"`;
  verified `jax==0.10.1` and `CudaDevice(id=0)`.
- Training command used `uv run --no-sync`.
- Row matched the prior best delayed no-PGD baseline except for the loss
  normalization flag: 8D C&S state, no PGD, no h0 encoder, no deliberate
  perturbation bank, `p_catch_trial=0.50`, `nn_output_pre_go=1e5`, `lr=3e-3`,
  `batch_size=64`, `n_replicates=5`, `hidden_size=180`, 12000 batches.
- New flag: `--delayed-reach-trial-type-normalized-loss`.
- Loss effect: `full_analytical_qrf` is split into
  `full_analytical_qrf_no_catch` and `full_analytical_qrf_catch`, each
  normalized over its own trial-type support before explicit weighting.
  `nn_output_pre_go` remains a separate pre-go output penalty.

## Training

- Completed: 12000/12000 batches.
- Total training duration: 736.10 s.
- Aggregate training rate: 16.30 batches/s.
- Steady-state chunk rate after first checkpoint: about 17.1-17.6 batches/s.
- Checkpoints: every 1000 batches through `checkpoint_0012000`, plus final.
- Training diagnostics: written and finite. Latest monitor summary reports
  train total mean 3427.59 and validation total mean 15987.8 across replicates.

## Velocity Figures

- No-catch final-checkpoint figure:
  `_artifacts/6c36536/figures/delayed_8d_no_pgd_catch0p5_prego1e5_normloss_velocity_final_checkpoints/no_catch/forward_velocity_profiles_stochastic.html`
- Catch final-checkpoint figure:
  `_artifacts/6c36536/figures/delayed_8d_no_pgd_catch0p5_prego1e5_normloss_velocity_final_checkpoints/catch/forward_velocity_profiles_stochastic.html`
- Evaluation bank: corrected delayed fixed bank, uniform 20 directions, 0.15 m
  reach length, go cues 10..30, go-aligned with 10 pre-go steps and 60 movement
  steps shown.
- Error band: mean +/- 1 SD over pooled stochastic trials; 2100 pooled samples
  per profile.

No-catch final velocity summary:

- GRU peak forward velocity: 0.6812 m/s at 0.15 s after go.
- extLQG 4D pos+vel reference peak: 0.7308 m/s at 0.16 s.
- extLQG 8D reference peak: 0.7311 m/s at 0.16 s.

Catch final velocity summary:

- GRU peak forward velocity: 0.00389 m/s.

## Decay / Checkpoint Sweep

Diagnostic summary:
`results/6c36536/notes/delayed_peak_decay_diagnostics_normloss_3e3_s42.md`

Key read:

- Final checkpoint peak: 0.6809 m/s, time-to-peak step 15, shape error 0.0284
  extLQG peak units.
- Best shape checkpoint: `checkpoint_0004000`, shape error 0.0152 extLQG peak
  units.
- Final remains peak-depressed relative to extLQG, so trial-type normalization
  did not solve the main kinematic discrepancy.
- Leakage remained small in the checkpoint sweep: pre-go peak velocity remained
  below 0.01 m/s, catch peak velocity around 0.02 m/s at final.

## Local Outputs

- Run spec: `results/6c36536/runs/normloss_3e3_s42/run.json`
- Model/artifacts:
  `_artifacts/6c36536/runs/normloss_3e3_s42/`
- Training monitor:
  `results/6c36536/notes/training_monitor_normloss_3e3_s42.txt`
- Remote training log:
  `_artifacts/6c36536/runpod_logs/normloss_train.log`

RunPod pod was stopped after artifact retrieval.
