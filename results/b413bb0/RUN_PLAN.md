# Beta 1.4 Soft-Adversary No-Launch Run Plan

Issue: `b413bb0`. Parent umbrella: `0a46652`. Lambda gate: `9bb676f`.

This is a prepared run lock, not launch authorization. The row specs and
commands below are ready for user review. Do not acquire a pod, start a training
process, push, request protected auth, merge, or close issues until the user
explicitly approves the locked spec.

## Readiness Status

Protected `main` contains the corrected 9bb lambda gate and the compositional
finite-adversary implementation merged at `3b35ef2`. No additional RLRMP
implementation is required before this no-launch lock. The remaining gate is
explicit user approval of the shared run details and launch budget.

The launch-facing finite rows now use `broad_epsilon_pgd_training` with
`inner_maximizer.method=adam`. For `linear_no_bias` and `affine`, that route
optimizes finite-policy parameters and evaluates them through the live
graph-component rollout. The older `policy_adversary_training` finite path is a
legacy/static-clean-rollout lane and is not used by this lock.

## Shared Training Contract

| field | value |
|---|---|
| status | `prepared_not_authorized` |
| script | `scripts/train_cs_nominal_gru.py` |
| launch branch | `feature/b413bb0-beta1p4-run-lock` |
| task | target-relative multitarget |
| target support | `const_band16` |
| plant state | 6D C&S process state without integrators (`--no-integrator-state`) |
| controller | C&S nominal GRU, hidden size 180, 5 replicates |
| feedback / encoder | force-filter feedback, initial hidden encoder |
| fixed perturbation training | enabled, calibrated timing, `--perturbation-physical-level moderate` |
| loss objective | `full_analytical_qrf` |
| adversary lane | broad-epsilon PGD soft-energy with Adam inner optimizer |
| reach scaling | disabled for broad-epsilon radius scaling |
| batches | 12000 |
| batch size | 64 |
| outer optimizer | AdamW, weight decay 0 |
| peak controller LR | `3e-3` |
| LR warmup | 500 batches from `3e-4` to `3e-3` |
| LR annealing | cosine after warmup, `alpha=0.01`, final LR `3e-5` |
| gradient clip | global norm 5 |
| checkpoint interval | 500 batches |
| log step | 100 |
| seed | 42 |
| GPU/cloud | RunPod secure cloud RTX 5090, pending user confirmation |
| deploy helper | Feedbax `scripts/deploy/runpod_deploy.sh`, pending user approval |
| poll helper | Feedbax `scripts/deploy/poll_run.sh` |

The trust-region radius remains present because the current training surfaces
need a projection/safety radius. Its role is numerical stabilization and
diagnostic sidecar metadata only. It does not set lambda values, optimizer
success, readiness, or row recommendations.

## Locked Rows

| row id | mechanism | training lane | beta | lambda / gamma | lambda provenance |
|---|---|---|---:|---:|---|
| `direct_epsilon` | `direct_epsilon` | broad-epsilon direct sequence, Adam | 1.4 | `550824678.4684843` | `9bb676f` direct cap-independent fixed-HVP/p90 candidate `281032999.21861446` scaled by `1.4^2` |
| `linear_no_bias` | `linear_no_bias` | broad-epsilon live finite policy, Adam | 1.4 | `704889898.0081824` | current merged `9bb676f` support-whitened generalized-HVP/Lanczos candidate `359637703.06539917` scaled by `1.4^2` |
| `affine` | `affine` | broad-epsilon live finite policy, Adam | 1.4 | `704817601.4292752` | current merged `9bb676f` support-whitened generalized-HVP/Lanczos candidate `359600817.05575264` scaled by `1.4^2` |

The finite-row unscaled candidates differ slightly from the pre-compositional
lock because the merged 9bb artifact was regenerated after the compositional
implementation. The old `~4.13e8` cap/trust-radius floors from `7ea17b8` remain
invalid for launch planning and are diagnostic-only.

## Row Mechanics

All three rows use `--broad-epsilon-pgd-training`,
`--broad-epsilon-pgd-inner-optimizer-method adam`,
`--broad-epsilon-pgd-adam-lr 0.001`, 8 ascent steps, step size fraction `0.25`,
and `--broad-epsilon-pgd-objective soft_energy`.

`direct_epsilon` optimizes the direct epsilon sequence. `linear_no_bias` and
`affine` install finite-policy inputs into `TaskTrialSpec.inputs` and evaluate
them through the live graph-component rollout. The moderate safety radius is
`0.0012324305441740995` at 15 cm from
`broad_epsilon_moderate_closed_loop_epsilon_l2_15cm_numeric_trust_region`; it
is a projection and diagnostic sidecar, not a lambda or readiness criterion.

## Approval Questions

The following shared settings still require explicit user confirmation before
launch:

- keep 12000 training batches with a 500-1000 batch early gate;
- keep seed `42`, 5 replicates, hidden size 180, batch size 64;
- keep the c92 6D no-integrator H0 GRU target-relative `const_band16` moderate contract;
- use RunPod secure cloud RTX 5090, or choose a different GPU/cloud tier;
- launch all three rows together or stage direct first as a control;
- accept the current compositional broad-PGD Adam implementation as the
  launch-facing finite-policy path;
- use the validation/evaluation packet described below as the post-run standard.

## Monitoring And Stop Criteria

After approval and launch, inspect startup/JIT at about 1 minute, then poll
every 5 minutes through the first 1000 batches. Once losses are steadily
descending, poll every 30 minutes. Stop a row if it produces nonfinite values,
obvious divergence, or adversary diagnostics that are implausible relative to
the other rows after first checking for a command or implementation mismatch.

The 500-1000 batch gate should inspect total loss plus broad-epsilon adversary
diagnostics: `adv_penalty`, `adv_energy`, `adv_objective`, `adv_gain`,
`adv_radius_ratio`, `adv_nonfinite`, and Adam/finite-policy diagnostic fields
where available. Cap-boundary and norm/radius quantities are diagnostic only.

## Expected Artifacts

Tracked run specs:

- `results/b413bb0/runs/direct_epsilon.json`
- `results/b413bb0/runs/linear_no_bias.json`
- `results/b413bb0/runs/affine.json`

Bulk run outputs after launch:

- `_artifacts/b413bb0/runs/direct_epsilon/`
- `_artifacts/b413bb0/runs/linear_no_bias/`
- `_artifacts/b413bb0/runs/affine/`

After completion, use `scripts/post_run.sh` from this issue worktree when
possible. The post-run packet should include training summaries, perturbation
loss diagnostics, adversary energy/objective diagnostics, early/mid/late reach
perturbation diagnostics, stabilization diagnostics, nominal behavior quality,
and direct comparison against the no-PGD H0 6D `const_band16` reference.
