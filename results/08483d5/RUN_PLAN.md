# Adaptive Soft-Adversary No-Launch Run Plan

## Status

This is a no-launch planning packet for issue `08483d5`. It does not authorize
local training, remote training, pod acquisition, push, protected auth request,
merge, issue closure, or any comment on `2e60620`.

The adaptive soft-epsilon implementation is present on local `main` at commit
`95c14a87339d67bf54241bcceb88e01ba578b87e`, merged from
`feature/08483d5-adaptive-curriculum` via auth request
`8cf3442b-c8c2-4fd6-8bd4-f04fa384db52`.

## Locked Initial Trial

Recommended row label: `adaptive_curriculum_3500to1000`.

The initial launch-facing trial should continue from the clean 6D no-PGD H0
`const_band16` baseline and add 7,500 controller batches of cap-free
`direct_epsilon` soft-energy training. The damage target ramps from `0` to
`3500` over the first 2,500 additional batches, then cosine-anneals to `1000`
over the next 5,000 batches. The outer adversarial weight independently ramps
from `0` to `1` over the first 2,500 additional batches and then holds at `1`.

Use `max_log_step=0.1` for this first trial. The merged default is `0.25`, but
the earlier no-launch replays were oscillatory, and this curriculum deliberately
starts with an aggressive high-damage ramp. A `0.1` cap limits a 50-batch update
to about a 10.5% multiplicative lambda change while still allowing adaptation
over the 150 update opportunities in the run.

## Operational Checkpoint Contract

The current training CLI resumes only from
`<output-dir>/checkpoints/checkpoint_latest`; it does not expose a first-class
`--init-checkpoint` or `--continue-from` source flag. Therefore the future run
manager must stage the baseline checkpoint into the new output directory before
launch, or add a supported source-checkpoint flag first. Do not run the new row
inside the baseline output directory.

Baseline source:

- tracked recipe: `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json`
- checkpoint tree:
  `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/`
- latest checkpoint:
  `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_latest`
  -> `checkpoint_0012000`

Future output paths after explicit launch approval:

- tracked run recipe:
  `results/08483d5/runs/adaptive_curriculum_3500to1000.json`
- graph sidecar dir:
  `results/08483d5/runs/adaptive_curriculum_3500to1000/`
- bulk output dir:
  `_artifacts/08483d5/runs/adaptive_curriculum_3500to1000/`

## Candidate Command Shape

This command shape was dry-run validated with an explicit `--spec-dir` so the
future run recipe lands at the intended flat path:

```bash
PYTHONPATH=src uv run --no-sync python scripts/train_cs_nominal_gru.py \
  --run-spec results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json \
  --output-dir _artifacts/08483d5/runs/adaptive_curriculum_3500to1000 \
  --spec-dir results/08483d5/runs/adaptive_curriculum_3500to1000 \
  --issue 08483d5 \
  --n-train-batches 19500 \
  --broad-epsilon-pgd-training \
  --broad-epsilon-pgd-objective soft_energy \
  --broad-epsilon-pgd-energy-lambda 281032999.21861446 \
  --broad-epsilon-pgd-inner-optimizer-method adam \
  --broad-epsilon-pgd-adam-lr 2e-5 \
  --broad-epsilon-pgd-steps 12 \
  --adaptive-epsilon-curriculum \
  --adaptive-epsilon-damage-start 0 \
  --adaptive-epsilon-damage-peak 3500 \
  --adaptive-epsilon-damage-final 1000 \
  --adaptive-epsilon-damage-ramp-batches 2500 \
  --adaptive-epsilon-damage-anneal-batches 5000 \
  --adaptive-epsilon-update-interval-batches 50 \
  --adaptive-epsilon-ema-alpha 0.1 \
  --adaptive-epsilon-eta 0.1 \
  --adaptive-epsilon-deadband-frac 0.1 \
  --adaptive-epsilon-max-log-step 0.1 \
  --adaptive-epsilon-outer-weight-start 0 \
  --adaptive-epsilon-outer-weight-final 1 \
  --adaptive-epsilon-outer-weight-ramp-batches 2500 \
  --checkpoint-interval-batches 500 \
  --log-step 100 \
  --resume
```

`--n-train-batches 19500` means "resume from the staged 12,000-batch baseline
checkpoint and stop after 7,500 additional batches." Without a staged checkpoint
in the new output directory, `--resume` would initialize from scratch.

## Post-Run Plan

After the run completes, use `scripts/post_run.sh` from the owning feature
worktree and then materialize a review packet. The first analysis should report:

- Adaptive trajectory: lambda, EMA damage, raw damage, target damage, deadband
  decisions, and outer adversarial weight. This checks whether the scheduler is
  controlling damage rather than merely producing finite epsilons.
- Clean/adversarial cost decomposition: total, running state, control, and
  terminal components where available. This distinguishes useful damage from
  blow-up-like trajectory degradation.
- Epsilon diagnostics: energy, norm, max absolute epsilon, nonfinite flags, and
  confirmation that no cap, radius, projection, or trust-region guard was active.
  These are sidecars, not pass criteria.
- Nominal phenotype: validation loss, endpoint quality, peak velocity,
  time-to-peak, forward-velocity RMSE, and pre-go drift where applicable. This
  tests whether robustness pressure damages baseline reach quality.
- Baseline comparisons: the source no-PGD H0 baseline, beta 1.05 output-feedback
  deterministic and paired-noise damage values, beta 1.4 output-feedback values
  as context, and older PGD rows only as historical sidecars.
- Perturbation-bank behavior: moderate randomized perturbation-bank metrics
  should remain interpretable as an orthogonal training axis, not as something
  weighted by the adaptive outer adversarial schedule.

## Residual Uncertainties

- Resume staging is operational, not first-class CLI support. Add a
  source-checkpoint flag before launch if the run manager should avoid copying
  checkpoint directories.
- The merged adaptive lambda update tracks damage on the live stochastic
  training batch. It does not use a separate fixed held-out damage-evaluation
  batch at the 50-batch cadence. Treat damage tracking as training-batch control
  evidence, and use fixed held-out replay post-run for the scientific report.
- The merged lambda update uses a clipped log-lambda step based on linear
  relative error, `eta * (EMA - target) / target`, not the older log-ratio
  formula. This lock accepts the merged behavior for the first trial.
- The starting lambda remains the beta 1.05 candidate
  `281032999.21861446`, used as a cap-independent soft-energy initialization.
  It is not a radius, cap, trust-region, or hard budget.
- The narrow beta spike near `1.342` remains stress-test context only and is not
  a curriculum target.
