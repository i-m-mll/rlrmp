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
| `minimax` | linear_dynamics | blocked | n/a | n/a | n/a | Historical comparison could not be completed cleanly: the warmup-artifact route that lets gaussian_bump isolate the adversarial loop mismatches the linear-dynamics graph, while running warmup directly hits current-dependency incompatibilities in the historical warmup path. |

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
