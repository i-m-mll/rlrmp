from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.intervene import schedule_intervenor
from feedbax.task import DelayedReaches, SimpleReaches
from feedbax.xabdeef.models import point_mass_nn
from feedbax_experiments.analysis.disturbance import (
    PLANT_DISTURBANCE_CLASSES,
    PLANT_INTERVENOR_LABEL,
)
from feedbax_experiments.misc import vector_with_gaussian_length

# from rlrmp.loss import get_reach_loss
from feedbax_experiments.training.loss import get_reach_loss
from feedbax_experiments.types import LDict, TaskModelPair, TreeNamespace
from jax_cookbook.tree import get_ensemble
from jaxtyping import PRNGKeyArray

from rlrmp.disturbances import get_gusts_fn

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

    # task_base = SimpleReaches(
    #     loss_func=get_reach_loss(hps),
    #     **hps.task.omitting_attrs("eval_n"),
    # )

    if hps.task.type == "simple_reach":
        task_base = SimpleReaches(
            loss_func=get_reach_loss(hps),
            **hps.task.omitting_attrs("eval_n", "type"),
        )
    elif hps.task.type == "delayed_reach":
        task_base = DelayedReaches(
            loss_func=get_reach_loss(hps),
            **hps.task.omitting_attrs("eval_n", "type"),
        )
    else:
        raise ValueError(f"Unrecognized task type: {hps.task.type}")

    models = get_ensemble(
        point_mass_nn,
        task_base,
        n=hps.model.n_replicates,
        dt=hps.dt,
        mass=hps.model.effector_mass,
        damping=hps.model.damping,
        hidden_size=hps.model.hidden_size,
        n_steps=hps.task.n_steps,
        feedback_delay_steps=hps.model.feedback_delay_steps,
        feedback_noise_std=hps.model.feedback_noise_std,
        motor_noise_std=hps.model.motor_noise_std,
        tau_rise=hps.model.tau_rise,
        tau_decay=hps.model.tau_rise,
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
