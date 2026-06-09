# Post-Benchmark Delayed-Reach Next Steps

Date: 2026-06-08

Issues: `4d79e07`, `6c36536`

This note updates the original PGD-replication handoff after the provider-neutral
packing benchmark, RunPod 4090 secure benchmark, and delayed-reach task-contract
implementation.

## Current State

- `integration/4d79e07-6c36536-pre-runpod` contains the provider-neutral
  scenario-driven packing benchmark, delayed-reach task contract, RunPod benchmark
  report, and PGD benchmark hook fix.
- The RunPod secure RTX 4090 benchmark is complete and the pod was stopped.
- The benchmark correction matters: the first PGD attempt was invalid because the
  benchmark omitted the normal C&S PGD `pre_step_fn`. Commit `19d530f` wires the
  hook and adds a regression test.
- Delayed reach now means: target visible from trial start, go cue sampled from
  steps 10 to 30, one scalar go-cue input added to the current target-relative
  feedback stream, movement-age C&S Q/R/Qf losses, prep-only `nn_output_pre_go`,
  and movement-epoch-gated random broad epsilon / PGD.
- Local delayed no-PGD smoke completed 500/500 batches with finite loss decline.
- Benchmark throughput on one RTX 4090:

| Row | n=1 b/s | n=2 b/s | n=4 b/s | n=8 b/s | Practical packing |
|---|---:|---:|---:|---:|---|
| Immediate no-PGD | 11.07 | 18.87 | 26.28 | 27.81 | n=4 or n=8 |
| Immediate PGD | 3.52 | 3.98 | 4.09 | 4.08 | n=4, n=8 not useful |
| Delayed no-PGD | 9.39 | 15.61 | 19.12 | 19.13 | n=4, n=8 not useful |
| Delayed PGD | 2.50 | 2.69 | 2.67 | 2.72 | n=1 for latency, n=2 for slight aggregate gain |

## How I Would Proceed

### 1. Land the substrate before more runs

Submit one protected-branch auth request for
`integration/4d79e07-6c36536-pre-runpod`, referencing `4d79e07` and `6c36536`.
This branch is now the substrate for both immediate PGD replication and delayed
reach:

- provider-neutral packing benchmark;
- delayed-reach task contract;
- RunPod deployment/benchmark evidence;
- PGD pre-step hook fix for benchmarks;
- tracked reports and run specs.

Do not launch more cloud training from an unmerged substrate unless there is a
specific reason to keep iterating on the integration branch.

### 2. Add one small delayed-PGD local or short-GPU smoke

The existing local delayed smoke covered no-PGD. Before any 12k delayed PGD run,
run a short delayed-PGD smoke with the production task contract:

- target-relative force/filter feedback;
- batch 64 if on GPU, smaller if local CPU only;
- hidden size 180 on GPU, small hidden size acceptable for local wiring smoke;
- `full_analytical_qrf`;
- `broad_epsilon_pgd_training=true`;
- `broad_epsilon_pgd_steps=10`;
- movement-epoch-only PGD;
- `nn_output_pre_go=1.0`;
- 500 to 1000 training batches.

Acceptance for this smoke:

- finite loss decline;
- nonzero PGD inner-loop diagnostics;
- achieved epsilon radius is plausible;
- PGD perturbation is movement-epoch-only;
- pre-go drift remains small;
- go-cue bins 10 to 30 all materialize;
- validation/materializer code can load the checkpoint.

Add a small metadata/reporting improvement if needed: expose
`movement_epoch_only` in the compact PGD metadata emitted by the benchmark/run
summary so future artifacts make the gating explicit.

### 3. Run the first serious delayed pair, not a broad matrix

For the next real training launch, prioritize delayed reach directly:

| Row | Purpose |
|---|---|
| `delayed_no_pgd__proprio_cal_small_lr3e-3_b64` | matched delayed baseline |
| `delayed_pgd__proprio_cal_small_pgd_ofb_lr3e-3_b64` | first delayed robust-discovery row |

Use the current production C&S contract:

- C&S LSS plant, `cs2019-rollout`;
- target-relative multi-target;
- force/filter feedback;
- calibrated small proprioceptive perturbation training;
- full analytical Q/R/Qf loss;
- lr `3e-3`, warmup+cosine, gradient clip 5;
- hidden size 180;
- validation-selected checkpoints;
- all-replicate reporting.

For packing on one 4090:

- delayed no-PGD: run 4 parallel models per pod;
- delayed PGD: run 1 model per pod if wall-clock latency matters, or 2 parallel
  models if aggregate seed throughput matters. Avoid 4 or 8 for PGD unless the
  queueing convenience outweighs poor per-model throughput.

Replicate count should be predeclared before launch. I would use 10 total
replicates for the first serious delayed pair if budget allows; otherwise run 5
as a gate and explicitly label it as a gate, not final replication.

### 4. Analyze delayed reach before expanding PGD pressure

Do not immediately sweep PGD budgets/steps until the delayed task itself passes
behavioral checks. The first analysis should answer:

- Did delayed no-PGD learn the movement without anticipatory drift?
- Did delayed PGD remain trainable under the movement-only inner maximizer?
- Are movement kinematics still close to the immediate C&S-like case?
- Did robustness improve without destroying nominal reach quality?

Required delayed-specific diagnostics:

- pre-go RMS drift and endpoint drift by go-cue time;
- pre-go velocity RMSE;
- go-cue aligned velocity and position profiles;
- reaction/movement onset after go cue;
- peak velocity and time-to-peak in movement age;
- endpoint error and terminal velocity;
- per-go-cue-bin losses and kinematics;
- PGD inner-loop scalars by movement epoch;
- perturbation energy by time/component, with prep period expected near zero for
  movement-only PGD.

Required standard diagnostics:

- training diagnostics, gradient norm, clipping fraction, update/parameter ratio,
  LR schedule, and PGD inner-loop metrics;
- standard certificate;
- objective comparator with extLQG and robust output-feedback analytical rows
  where defined;
- perturbation-response bank;
- same-channel worst-case epsilon audit;
- H-infinity phenotype sidecar;
- feedback ablation/lens bundle;
- task-aligned/covariance-weighted map-error decomposition;
- velocity profiles and loss figures;
- all-replicate tables and checkpoint-selection summaries.

### 5. Expand only after the delayed pair is healthy

If the delayed pair passes the behavioral and diagnostic gate, expand in this
order:

1. Increase delayed PGD replication at the same moderate budget/10-step pressure.
2. Add moderate-vs-strong budget comparison.
3. Add PGD pressure comparison, such as 10 vs 20 steps or step-size fraction
   changes, only after the budget direction is interpretable.
4. Add a minimal immediate-reach anchor only if the delayed result needs a fresh
   matched comparator; otherwise reuse the existing immediate rows.

Stop before broad expansion if:

- PGD diagnostics are missing, flat, or inconsistent with the configured budget;
- pre-go drift returns;
- go-cue timing bins are uneven or not reported;
- movement kinematics degrade enough that robustness is confounded with a worse
  task solution;
- materializers cannot load delayed checkpoints.

## What Not To Lose From The Broader Direction

- Preserve immediate-reach rows as separate baselines; delayed reach is a task
  evolution, not a relabeling of the same task.
- Keep h0 / initial-state recovery (`643f101`) as the simpler alternative or
  companion to delayed preparation. If h0 solves early-state recovery cleanly,
  delayed reach may be a scientific/interpretability lane rather than a required
  fix.
- Keep formal PGD-vs-Riccati adequacy/equivalence separate from the training
  experiment. PGD remains an open-loop same-channel surrogate until the formal
  adversary path supports stronger claims.
- Keep task breadth and multi-target generality active. Delayed reach should not
  regress back to a narrow single-target or hidden timing convention.
- Keep the analytical game-card / standard-certificate boundary explicit:
  training rows, evaluation lenses, and robustness certificates are different
  axes.
- Keep perturbation taxonomy clear: calibrated bank, random broad epsilon, and
  PGD worst-case epsilon are different claims.

