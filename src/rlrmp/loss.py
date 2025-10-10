import equinox as eqx
from feedbax.loss import AbstractLoss, StopAtGoalLoss
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax_experiments.training.loss import get_readout_norm_loss
from feedbax_experiments.types import TreeNamespace


def get_reach_loss(hps: TreeNamespace):
    """Get loss function for reaching task, with some optional extras."""
    loss_fn = simple_reach_loss()

    if not getattr(hps, "loss", False):
        return loss_fn

    if getattr(hps.loss, "stop_at_goal", False):
        loss_fn = loss_fn + StopAtGoalLoss(**hps.loss.stop_at_goal)

    if getattr(hps.loss, "fix_readout_norm", False):
        loss_fn = loss_fn + get_readout_norm_loss(**hps.loss.fix_readout_norm)

    if getattr(hps.loss, "weights", False):
        loss_fn = eqx.tree_at(
            lambda loss_fn: loss_fn.weights,
            loss_fn,
            {**loss_fn.weights, **hps.loss.weights},
        )

    return loss_fn
