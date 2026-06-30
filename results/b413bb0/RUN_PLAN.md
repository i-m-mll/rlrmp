# Beta 1.4 Soft-Adversary No-Launch Run Plan

Issue: `b413bb0`. Parent umbrella: `0a46652`. Lambda gate: `9bb676f`.

This is a prepared run lock, not launch authorization. The row specs and
commands below are ready for user review. Do not acquire a pod, start a training
process, push, request protected auth, merge, or close issues until the user
explicitly approves the locked spec.

## Readiness Status

Protected `main` now contains the corrected 9bb finite-policy Adam/lambda-gate
work and the earlier finite-policy graph-component clarification. No additional
RLRMP implementation is required before a no-launch lock. The remaining gate is
user approval of the shared run details and the finite-row caveat below.

Important caveat: current `policy_adversary_training` finite rows optimize
finite parameters with Adam and persist the adversary state, but the training
integration materializes epsilon from clean-rollout pre-step features. Do not
describe the `linear_no_bias` or `affine` Adam rows as true live-perturbed
closed-loop finite-policy training until a Feedbax live rollout hook exists.
The broad-PGD finite graph-component path exists separately; it is not the Adam
finite-policy lane used by these two rows.

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
| reach scaling | disabled for broad-epsilon / policy-adversary radius scaling |
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
| `direct_epsilon` | `direct_epsilon` | broad-epsilon PGD soft-energy | 1.4 | `452578932.66011757` | `06a4dc8` open-loop-moderate beta mapping, `lambda = beta^2 * p90(lambda_star_i)` |
| `linear_no_bias` | `linear_no_bias` | finite policy Adam, energy mode | 1.4 | `705034449.3625335` | `9bb676f` support-whitened generalized-HVP/Lanczos candidate `359711453.7563946` scaled by `1.4^2` |
| `affine` | `affine` | finite policy Adam, energy mode | 1.4 | `705123804.979113` | `9bb676f` support-whitened generalized-HVP/Lanczos candidate `359757043.3566903` scaled by `1.4^2` |

For the direct row, this lock uses the `open_loop_moderate` per-substrate
mapping from `06a4dc8` because the shared contract is the calibrated moderate
substrate. It does not use the pooled direct beta-1.4 value.

For finite rows, the 9bb cap-independent finite-policy candidates are treated as
lambda-star candidates under the current beta convention. The old `~4.13e8`
cap/trust-radius floors from `7ea17b8` are invalid for launch planning and are
diagnostic-only.

## Row Mechanics

`direct_epsilon` uses broad-epsilon PGD with zero start, 10 ascent steps, step
size fraction `0.25`, soft-energy objective, and the moderate safety radius
`0.0012324305441740995` at 15 cm from
`broad_epsilon_moderate_closed_loop_epsilon_l2_15cm_numeric_trust_region`.

`linear_no_bias` and `affine` use `policy_adversary_training` with Adam,
energy mode, 8 ascent steps per controller update, inner learning rate `0.001`,
and the same reference radius as a projection/diagnostic sidecar. Adam state is
checkpointed with the model.

## Approval Questions

The following shared settings still require explicit user confirmation before
launch:

- keep 12000 training batches with a 500-1000 batch early gate;
- keep seed `42`, 5 replicates, hidden size 180, batch size 64;
- keep the c92 6D no-integrator H0 GRU target-relative `const_band16` moderate contract;
- use RunPod secure cloud RTX 5090, or choose a different GPU/cloud tier;
- launch all three rows together or stage direct first as a control;
- keep the finite Adam caveat acceptable for this beta-1.4 comparison;
- use the validation/evaluation packet described below as the post-run standard.

## Monitoring And Stop Criteria

After approval and launch, inspect startup/JIT at about 1 minute, then poll
every 5 minutes through the first 1000 batches. Once losses are steadily
descending, poll every 30 minutes. Stop a row if it produces nonfinite values,
obvious divergence, or adversary diagnostics that are implausible relative to
the other rows after first checking for a command or implementation mismatch.

The 500-1000 batch gate should inspect total loss plus adversary diagnostics:
direct PGD `adv_penalty`, `adv_energy`, `adv_objective`, `adv_gain`,
`adv_radius_ratio`, and `adv_nonfinite`; finite Adam
`policy_adversary_*` optimizer/objective/energy/projection diagnostics where
available. Cap-boundary and norm/radius quantities are diagnostic only.

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
