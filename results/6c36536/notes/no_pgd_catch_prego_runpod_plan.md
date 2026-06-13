# Delayed No-PGD Catch/Pre-Go RunPod Plan

Issue: 6c36536
Related: 4d79e07, ffff699, 198be43

## Decision Summary

The next delayed training pass should focus on no-PGD 8D rows only. PGD should
wait until the delayed task reliably suppresses pre-go anticipation under the
target-visible/go-cue contract.

The canonical C&S objective should apply only to the fixed movement window from
go cue through go+60. The extra ragged tail is still rolled out because the
batch tensor has fixed trial length, but it is masked out of the exact Q/R/Q_f
objective and out of the standard movement-window comparator. Tail behavior is
therefore unconstrained by the canonical objective. If tail quieting becomes
important, add a separately named auxiliary tail term rather than folding it
into the analytical Q/R/Q_f claim.

## Planned Rows

Both rows should use the full 8D C&S physical state, target-relative multi-target
delayed reach, force/filter feedback, hidden size 180, batch size 64, five
replicates, 12k training batches, lr 3e-3 with warmup/cosine scheduling, gradient
clip 5, full analytical Q/R/Q_f movement objective, and no PGD/broad-epsilon
training.

1. `delayed_8d_no_pgd_catch0p5_prego0_lr3e-3_clip5_b64`
   - `p_catch_trial=0.5`
   - `nn_output_pre_go=0.0`

2. `delayed_8d_no_pgd_catch0p5_prego1_lr3e-3_clip5_b64`
   - `p_catch_trial=0.5`
   - `nn_output_pre_go=1.0`

## Required Gate Before Launch

The current delayed worktree exposes `--delayed-reach-p-catch-trial` and records
`nn_output_pre_go`, but `get_reach_loss(... full_analytical_qrf ...)` currently
returns a loss with only the full-QRF term. Before these rows are launched, the
pre-go row must be made semantically real by explicitly adding a separately
labeled auxiliary `nn_output_pre_go` term to the full-QRF loss when the weight is
nonzero. Otherwise the two planned rows would train under the same objective
except for metadata.

Focused verification before RunPod:

- A unit test showing `full_analytical_qrf` delayed loss includes only Q/R/Q_f
  when `nn_output_pre_go=0.0`.
- A unit test showing it includes both `full_analytical_qrf` and
  `nn_output_pre_go` when `nn_output_pre_go=1.0`, with the pre-go mask limited
  to the prep epoch.
- A catch-trial test showing target visibility is preserved while go/movement
  is withheld and the scored target remains the initial position.
- A short local smoke for each planned row, enough to confirm finite losses,
  correct run metadata, 8D physical state, p_catch_trial=0.5, and distinct
  active loss labels.

## RunPod Execution Prep

Use the existing RunPod runbook and secure RTX 4090 path. After `uv sync`, install
CUDA JAX and run all training with `uv run --no-sync`; verify `jax.devices()` is
CUDA before launch. Use the previously measured delayed no-PGD packing result:
launch both rows concurrently on one secure 4090 only after the local gates pass.

Monitoring should be tighter than a normal mature run:

- Check startup/import/JIT immediately.
- At about 1 minute, confirm finite losses and nonzero batch progress.
- During the first few hundred batches, monitor pre-go velocity/drift and active
  loss labels if diagnostics are available.
- After early descent is stable, monitor at the usual longer cadence.

Post-run diagnostics should initially emphasize:

- validation-selected checkpoints,
- go-aligned velocity profiles with 10 pre-go steps and the canonical 60-step
  movement window,
- pre-go RMS drift / forward velocity,
- endpoint error and terminal speed over the canonical movement window,
- tail behavior as a sidecar only, not as part of the comparator gate.
