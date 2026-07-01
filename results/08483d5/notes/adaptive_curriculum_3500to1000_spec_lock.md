# Adaptive Curriculum 3500-to-1000 No-Launch Spec Lock

This packet locks the initial adaptive soft-adversary training trial for issue
`08483d5`. It is a no-launch lock: no training, pod acquisition, protected auth
request, push, merge, issue closure, or comment on `2e60620` is authorized.

| Field | Locked value |
|---|---|
| Question | Can cap-free adaptive `direct_epsilon` soft-energy training keep the adversary active while first pushing damage toward a high but non-spike regime, then annealing to a moderate damage target without old cap/radius/trust-region values defining the effect? |
| Recommended row label | `adaptive_curriculum_3500to1000` |
| Code branch / commit | Local `main` at `95c14a87339d67bf54241bcceb88e01ba578b87e`, merge of `feature/08483d5-adaptive-curriculum`; auth request `8cf3442b-c8c2-4fd6-8bd4-f04fa384db52` is completed. |
| Checkpoint source | Continue from `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest` (`checkpoint_0012000`) with tracked source recipe `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json`. |
| Resume operation | Stage the baseline checkpoint tree into `_artifacts/08483d5/runs/adaptive_curriculum_3500to1000/checkpoints/` before launch, or add a supported source-checkpoint flag first. Current CLI resumes only from the selected output directory. |
| Task and controller | Preserve target-relative `const_band16`; 6D C&S no-integrator physical state; 36D delayed LSS state; nominal GRU hidden size 180; 5 replicates; h0/initial hidden encoder; full analytical Q/R/Qf; calibrated moderate perturbation-training baseline. |
| Training length | Resume from 12,000 completed baseline batches and stop at `n_train_batches=19500`, giving 7,500 additional adaptive batches. |
| Optimizer and base run parameters | Batch size 64; seed 42; controller lr `3e-3`; warmup-cosine schedule inherited from the baseline with 500-batch warmup from 0.1 fraction and cosine alpha `0.01`; gradient clip 5; checkpoint interval 500; log step 100. |
| Adaptive adversary | Enable `--broad-epsilon-pgd-training`, `--broad-epsilon-pgd-objective soft_energy`, `--broad-epsilon-pgd-mechanism direct_epsilon`, cap-free Adam inner optimizer, 12 steps, Adam lr `2e-5`, `b1=0.9`, `b2=0.999`, `eps=1e-8`, zero initialization. |
| Forbidden controls | No projection, safety cap, inherited radius, trust region, or hard budget. Epsilon norm/energy are diagnostics only. |
| Initial lambda | `281032999.21861446`, the beta 1.05 cap-independent soft-energy initialization from the HVP p90 lambda source. |
| Damage schedule | Target starts at `0`, linearly ramps to `3500` over the first 2,500 additional batches, then cosine-anneals to `1000` over the next 5,000 batches. Do not target the beta `~1.342` spike. |
| Outer adversarial weight | Starts at `0`, linearly ramps to `1` over the first 2,500 additional batches, then holds at `1`; applies only to optimized direct-epsilon loss. Randomized perturbation-bank training remains orthogonal. |
| Lambda update | Update every 50 controller batches; EMA alpha `0.1`; eta `0.1`; 10% deadband; `lambda_min=1e-12`; no `lambda_max`; `max_log_step=0.1` recommended for the first trial. The merged code uses clipped `eta * relative_error` as the log-lambda step. |
| Stopping criteria | Normal stop at 19,500 total batches. Abort on nonfinite controller loss, nonfinite adversary objective, nonfinite lambda, checkpoint write failure, or adaptive state restore failure. |
| Expected tracked artifacts | Future recipe `results/08483d5/runs/adaptive_curriculum_3500to1000.json`; future graph sidecars under `results/08483d5/runs/adaptive_curriculum_3500to1000/`; post-run note `results/08483d5/notes/adaptive_curriculum_3500to1000.md`. |
| Expected bulk artifacts | `_artifacts/08483d5/runs/adaptive_curriculum_3500to1000/`, including checkpoints, `training_summary.json`, training diagnostics NPZ/manifest, history chunks, final model, and logs. |
| Post-run analysis | Report adaptive trajectory, clean/adversarial cost decomposition, epsilon sidecars, nominal movement phenotype, comparison to source no-PGD baseline and beta 1.05 / beta 1.4 output-feedback damage values, and perturbation-bank behavior. |
| Budget / cost exposure | Current cost is zero: no pod or training was launched. Future cost begins at pod creation; secure cloud remains the default unless the user explicitly accepts another tier at launch time. Verify current GPU availability and image tag before any pod creation. |
| Launch gate | Stop here. A future session must present the launch command and receive explicit user confirmation before creating resources or starting training. |

## Validation Performed

- `mandible issue report 08483d5 --json`
- `mandible auth list --repo rlrmp --head feature/08483d5-adaptive-curriculum --json`
- `git status --short --branch`
- `git rev-parse HEAD`
- `git worktree list --porcelain`
- `find results/08483d5 -maxdepth 3 -type f -print`
- `find _artifacts/08483d5 -maxdepth 4 -print`
- `PYTHONPATH=src uv run --no-sync python scripts/train_cs_nominal_gru.py --help`
- `PYTHONPATH=src uv run --no-sync python` dry-run probe of the command shape with explicit `--spec-dir`

The dry-run probe confirmed the future recipe would write:

- `results/08483d5/runs/adaptive_curriculum_3500to1000.json`
- `results/08483d5/runs/adaptive_curriculum_3500to1000/model.graph.manifest.json`

It also confirmed `broad_epsilon_pgd_training.budget_contract.effective_l2_radius_15cm`
is `null`, `radius_bound_mode` is `false`, safety cap is disabled, and the inner
maximizer projection is `none_cap_free_direct_soft_energy`.

## Residual Uncertainties

- Resume staging is not first-class in the CLI. The future run manager must copy
  the baseline checkpoint tree into the new output directory or add a supported
  source-checkpoint option before launch.
- The adaptive damage signal is measured on the live stochastic training batch,
  not a fixed held-out damage-evaluation batch. Use fixed held-out replays
  post-run before interpreting scientific damage matching.
- The lambda update formula in merged code is relative-error based, not the
  older log-ratio formula. This lock accepts merged behavior and compensates
  with conservative `max_log_step=0.1`.
