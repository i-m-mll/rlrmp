# Corrected Soft-Lambda Smoke Run Plan

Issue: ae9f30f. Umbrella: 54389a4.

This is the first approved launch wave after the corrected cap-independent
soft-lambda redo. The user approved launching four rows in parallel on an RTX
5090: direct epsilon and linear no-bias at beta 1.05 and beta 1.4.

## Shared Training Contract

| field | value |
|---|---|
| script | `scripts/train_cs_nominal_gru.py` |
| launch branch | `feature/ae9f30f-corrected-soft-lambda-smoke` |
| task | target-relative multitarget |
| target support | `const_band16` |
| plant state | 6D C&S process state without integrators (`--no-integrator-state`) |
| controller | C&S nominal GRU, hidden size 180, 5 replicates |
| feedback / encoder | force-filter feedback, initial hidden encoder |
| fixed perturbation training | enabled, calibrated timing, `--perturbation-physical-level moderate` |
| loss objective | `full_analytical_qrf` |
| adversary | broad epsilon PGD, soft-energy objective |
| adversary substrate | open-loop-moderate corrected HVP/p90 lambda scale |
| reach scaling | disabled for broad epsilon PGD |
| inner PGD | zero start, 10 ascent steps, step size fraction 0.25 |
| soft-energy safety cap | `0.0012324305441740995` at 15 cm, source `broad_epsilon_moderate_closed_loop_epsilon_l2_15cm_numeric_trust_region` |
| safety-cap role | numerical trust-region only, not a pass/fail criterion |
| old hard-cap ratios | sidecar diagnostics only |
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
| GPU/cloud | RunPod secure cloud RTX 5090 |
| deploy helper | Feedbax `scripts/deploy/runpod_deploy.sh` with row manifest |
| poll helper | Feedbax `scripts/deploy/poll_run.sh` |

The safety cap is required by the current `soft_energy` training surface. It is
recorded as a numerical trust region, not as the scientific hard budget or a
success criterion.

## First-Wave Rows

| row id | mechanism | beta | lambda |
|---|---|---:|---:|
| `direct_epsilon_b1p05` | `direct_epsilon` | 1.05 | 254575649.62131616 |
| `direct_epsilon_b1p4` | `direct_epsilon` | 1.4 | 452578932.66011757 |
| `linear_no_bias_b1p05` | `linear_no_bias` | 1.05 | 254575649.62131616 |
| `linear_no_bias_b1p4` | `linear_no_bias` | 1.4 | 452578932.66011757 |

## Monitoring And Stop Criteria

Launch all four rows in parallel. Inspect startup/JIT, then check early progress
at roughly 1 minute and every 5 minutes through the first 1000 batches. After
losses are steadily descending, poll every 30 minutes.

At the 1000-batch gate, inspect total loss and the reported loss terms,
including `adv_penalty`, `adv_energy`, `adv_objective`, `adv_gain`,
`adv_radius_ratio`, and `adv_nonfinite`. If one row has nonfinite values,
obvious divergence, or adversary diagnostics that are implausible relative to
the other rows, first check that the launch command and mechanism implementation
match this plan. If no implementation or command error is found, stop the
affected row and leave the other rows running.

Do not use old hard-cap ratios as pass/fail criteria.

## Expected Artifacts

Tracked run specs:

- `results/ae9f30f/runs/direct_epsilon_b1p05.json`
- `results/ae9f30f/runs/direct_epsilon_b1p4.json`
- `results/ae9f30f/runs/linear_no_bias_b1p05.json`
- `results/ae9f30f/runs/linear_no_bias_b1p4.json`

Bulk run outputs:

- `_artifacts/ae9f30f/runs/direct_epsilon_b1p05/`
- `_artifacts/ae9f30f/runs/direct_epsilon_b1p4/`
- `_artifacts/ae9f30f/runs/linear_no_bias_b1p05/`
- `_artifacts/ae9f30f/runs/linear_no_bias_b1p4/`

After completion, run post-run materialization and diagnostics. The post-run
packet should include a nominal velocity profile figure with two shared-y-axis
subplots: direct epsilon and linear no-bias. Each subplot should show its two
trained beta rows plus the 6D no-integrator analytical extLQG and output-feedback
H-infinity references.

The final issue comment should summarize feedback/robustness diagnostics,
sensory and non-sensory perturbation loss gains and AUC diagnostics, early/mid/
late reach perturbation diagnostics, stabilization diagnostics, and direct
comparison against the equivalent no-PGD H0 6D `const_band16` model.

## Standalone Review Packet

After post-run materialization, create a standalone review packet for an
external reviewer. The packet should cover the run code, run specs, materialized
results, diagnostics, and issue context needed to evaluate why the linear
no-bias rows suppress the adversary. It should be framed open-endedly: possible
explanations include implementation error, overly large lambda for the training
task, Hessian/p90 calibration not transferring to the live training surface,
single-adversary-across-batch effects, fresh adversary initialization each
batch, missing adversary warmup or carry-over, or another mechanism not yet
identified.

The packet should include:

- A copy of the RLRMP repo state used for the experiments, including the bulky
  local context the user requested: `results/`, `_artifacts/`, `worktrees/`,
  and `.venv/`. This review packet is a local standalone bundle, not a tracked
  repo artifact.
- Separate bundled result artifacts and summaries sufficient for review.
- A subdirectory containing full text exports of relevant issues and comments.
- A plain-language diagnostic summary and the packet author’s interpretation,
  while preserving uncertainty and inviting alternative explanations.
- A reviewer prompt asking what to do about linear-case suppression and whether
  the evidence suggests an implementation regression or a methodological change
  such as lower lambda, different finite-policy optimizer, warm-start/carry-over,
  or mechanism-specific calibration.

The initial packet should be created by a subagent. A separate restricted-context
subagent should then review it for vagueness, missing context, undefined terms,
and insufficient epistemic humility before final delivery.
