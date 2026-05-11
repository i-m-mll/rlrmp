# Cell-2 (gru__jerk_motor_pre) Spike Investigation

**Cell:** gru__jerk_motor_pre
**N batches:** 12000
**N replicates:** 5

## Context

The original broken plot showed ~3-4 sharp downward spikes mid-training for cell-2
(orange trace). This investigation determines whether those spikes are real (present
in the true weighted total loss) or an artifact of the broken loss computation.

## End-of-training total loss (TRUE weighted total)

Mean (last 100 batches): 5.75e+00
Std (last 100 batches): 4.94e-01

| Replicate | Final loss | Min loss | At batch |
|-----------|-----------|---------|---------|
| 0 | 5.27e+00 | 4.75e+00 | 10688 |
| 1 | 6.20e+00 | 5.10e+00 | 8925 |
| 2 | 5.40e+00 | 4.27e+00 | 11222 |
| 3 | 5.70e+00 | 4.41e+00 | 9181 |
| 4 | 5.17e+00 | 4.57e+00 | 10059 |

## Detected spikes in TRUE weighted total loss: NONE

Using the threshold: total_loss < (rolling 50-batch median) / 10, no spikes were
found. The per-replicate minimum ratio to rolling median was 0.72 (replicate 2,
batch 4242), which is a gradual improvement, not a spike.

## Root cause of apparent spikes in the original broken plot

The original plot was computing an **unweighted raw sum** of all TermTree leaf
arrays (via `jt.leaves(history.loss)`, filtering for ndim==2 arrays). This is
WRONG because:

1. `jt.leaves` on a `TermTree` returns both the value arrays AND the Python float
   `weight` scalars (since weights are dynamic leaves in TermTree.tree_flatten).
2. The unweighted sum mixes terms at wildly different scales: `nn_hidden` and
   `nn_output` (raw values ~100-8000) vs `effector_pos_*` (~1-20). The sum is
   dominated by `nn_hidden` + `nn_output`.
3. These raw `nn_hidden`/`nn_output` values RISE during training (the network
   grows larger weights as it learns the task), while positional error terms FALL.
   The net effect is a "U-shape" or rising broken total that looks nothing like
   the true loss curve.

The spikes in the broken plot were most likely noise in the `nn_hidden` or
`nn_output` unweighted values at isolated batches where the network transiently
produced a lower-than-usual activity level. They are **not real** in the sense
that the weighted total loss shows no corresponding spike.

## Loss term contributions at end of training (cell-2)

| Term | Weighted value (final 100 batches mean) |
|------|---------------------------------------|
| effector_pos_late | 1.87e-01 |
| effector_pos_running | 4.73e+00 |
| effector_vel_late | 2.90e-01 |
| nn_hidden | 4.60e-02 |
| nn_output | 2.41e-02 |
| nn_output_jerk | 3.15e-01 |
| nn_output_pre_go | 1.61e-01 |

The dominant term at end of training is `effector_pos_running` (~85% of total),
which is the running position error across the full post-go movement window.

## nn_output_pre_go term trajectory (cell-2 specific)

`nn_output_pre_go` (weight=1e-2, pre-go motor silence penalty) shows a monotonic
RISE from ~2e-2 to ~1.6e+1 (unweighted), while weighted it rises from ~2e-4 to
~1.6e-1. This is expected: as the network learns to reach more effectively
post-go, its pre-go motor activity increases too (more "preparation" activity),
which this term penalises. There is no transient collapse of pre-go activity to
near-zero — confirming that the original "basin-discovery" hypothesis was not
supported by the data.
