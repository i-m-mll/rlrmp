from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.intervene import schedule_intervenor
from feedbax_experiments.misc import vector_with_gaussian_length
from feedbax_experiments.types import LDict, TaskModelPair, TreeNamespace
from jaxtyping import PRNGKeyArray

from rlrmp.disturbance import (
    PLANT_DISTURBANCE_CLASSES,
    PLANT_INTERVENOR_LABEL,
)
from rlrmp.disturbances import get_gusts_fn
from rlrmp.loss import get_reach_loss
from rlrmp.models import create_point_mass_nn_ensemble
from rlrmp.task import TASK_TYPES

#! TODO: limit curl and constant fields to movement epoch!
disturbance_params = LDict.of("pert__type")(
    {
        "curl": lambda hps: dict(
            amplitude=lambda trial_spec, batch_info, key: jr.normal(key, ()),
        ),
        "constant": lambda hps: dict(
            field=lambda trial_spec, batch_info, key: vector_with_gaussian_length(key),
        ),
        "gusts": get_gusts_fn,
    }
)


def setup_task_model_pair(hps: TreeNamespace, *, key):
    """Returns a skeleton PyTree for reloading trained models."""
    scaleup_batches = hps.intervention_scaleup_batches
    n_batches_scaleup = scaleup_batches[1] - scaleup_batches[0]
    if n_batches_scaleup > 0:

        def batch_scale_up(batch_start, n_batches, batch_info, x):
            progress = jax.nn.relu(batch_info.current - batch_start) / n_batches
            progress = jnp.minimum(progress, 1.0)
            scale = 0.5 * (1 - jnp.cos(progress * jnp.pi))
            return x * scale
    else:

        def batch_scale_up(batch_start, n_batches, batch_info, x):
            return x

    hps_task = {k: v for k, v in hps.task.omitting_attrs("eval_n", "type").items() if v is not None}

    task_base = TASK_TYPES[hps.task.type](loss_func=get_reach_loss(hps), **hps_task)

    models = create_point_mass_nn_ensemble(
        hps,
        task_base,
        n_extra_inputs=0,
        key=key,
    )

    def disturbance(field_std, active=True):
        return PLANT_DISTURBANCE_CLASSES[hps.pert.type].with_params(
            scale=field_std,
            active=active,
            # **disturbance_params(partial(batch_scale_up, scaleup_batches[0], n_batches_scaleup))[
            #     hps.pert.type
            # ],
            **disturbance_params[hps.pert.type](hps),
        )

    return TaskModelPair(
        *schedule_intervenor(
            task_base,
            models,
            lambda model: model.step.mechanics,
            disturbance(hps.pert.std),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        )
    )
