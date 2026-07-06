# Retro Equivalence Verdict

Issue: `9c0ad75`

Historical checkout: `02a9d12^` (`6f3b0aa`), created as a temporary `wt`
worktree. Commands were run through `uv run --no-sync python`.

## Verdicts

| family | case | verdict | model diff | optimizer diff | key diff | notes |
|---|---|---:|---:|---:|---:|---|
| `cs_supervised` | plain | pass | `0.0` | `0.0` | `0.0` | One fixed-seed batch, forced legacy path vs forced native path. |
| `cs_supervised` | PGD pre-step | pass | `0.0` | `0.0` | `0.0` | One fixed-seed batch with `--broad-epsilon-pgd-training --broad-epsilon-pgd-steps 1`. |
| `minimax` | gaussian_bump | fail | `0.14897151034167244` | `1.2265456213160557` controller / `0.7111212476732048` adversary | `3371135417.0` | Real historical adversarial loop vs native executor, one adversarial batch. |
| `minimax` | linear_dynamics | blocked | n/a | n/a | n/a | Historical comparison could not be completed cleanly: no bounded route materialized a comparable legacy run under the historical linear-dynamics graph and current dependency set. |

## Minimax Gaussian Details

The gaussian-bump comparison used `n_warmup_batches=0`, `n_adversary_batches=1`,
`n_adversary_steps=1`, `batch_size=1`, `adv_batch_size=1`, and historical default
replicate count. A fixed warmup artifact stored the minimax initial controller
with standard-training metadata so the historical loader could deserialize it.
The legacy progress callback factory was disabled inside the one-off process
because current Feedbax raises on the historical callback dictionary path before
training semantics are reached.

Observed loss scalars:

| scalar | value |
|---|---:|
| legacy adversary loss | `0.7847103498727289` |
| native adversary loss | `42.43777390838417` |
| legacy controller loss | `0.7853422260775287` |
| native controller loss | `42.44496647121827` |

This is a real divergence signal and should be tracked as a separate error
issue. Attempts to create that issue from this session failed even for a minimal
probe body (`mandible issue new` returned `Failed to create issue`), so this note
and the `9c0ad75` closeout comment preserve the evidence without silently fixing
it in this lane.

## Minimax Linear-Dynamics Blocker

The follow-up bounded pass extracted `6f3b0aa` with `git archive` into a
temporary snapshot and ran the historical `scripts/train_minimax.py` through the
current worktree's `uv run --no-sync` environment with `PYTHONPATH` pointed at
the snapshot. No additional git worktree was created.

Two minimal historical legacy runs were attempted:

1. `--adversary-type linear_dynamics --n-warmup-batches 0
   --n-adversary-batches 1 --n-adversary-steps 1 --batch-size 1
   --adv-batch-size 1 --n-replicates 1 --fused`
2. The same run with `--n-warmup-batches 1`, to exercise the direct historical
   warmup path instead of the zero-warmup shortcut.

The zero-warmup run failed before phase 2 because current Optax rejects the
historical zero-length warmup schedule:

```text
ValueError: The cosine_decay_schedule requires positive decay_steps, got decay_steps=0.
```

The one-batch direct warmup run also failed before phase 2. It reached Feedbax
runtime execution, then the historical linear-dynamics graph fed a 4-wide signal
into the GRU path that expected 11 inputs:

```text
TypeError: dot_general requires contracting dimensions to have the same shape, got (11,) and (4,).
```

That second failure is consistent with the `cbbcdb6` native linear-dynamics
residual: the native path only became usable after the component-parameter input
routing was repaired. Current-tree native linear-dynamics driver/resume checks
still pass, but they compare native executor against the post-repair native
driver loop; they do not supply real legacy-vs-native numbers.

Bounded verdict: `linear_dynamics` remains blocked/bracketed rather than
completed. Unblocking a clean comparable number requires either an environment
that exactly matches the historical `6f3b0aa` dependency behavior or a dedicated
archaeology harness that constructs the zero-warmup initial controller and runs
the historical linear-dynamics adversarial loop in-process while bypassing the
historical warmup/materialization path. That harness should first be validated
against the already-recorded gaussian-bump divergence route so it does not
silently create another new-vs-new comparison.
