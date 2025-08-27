from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.intervene import schedule_intervenor
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.xabdeef.models import point_mass_nn
from feedbax_experiments.analysis.disturbance import (
    PLANT_DISTURBANCE_CLASSES,
    PLANT_INTERVENOR_LABEL,
)
from feedbax_experiments.constants import (
    MASS,
)
from feedbax_experiments.misc import vector_with_gaussian_length
from feedbax_experiments.setup_utils import get_base_reaching_task
from feedbax_experiments.types import LDict, TaskModelPair, TreeNamespace
from jax_cookbook.tree import get_ensemble
from jaxtyping import PRNGKeyArray


def disturbance_params(scale_func):
    """Returns a dict of disturbance parameter functions, scaled by `scale_func`."""
    return {
        "curl": dict(
            amplitude=lambda trial_spec, batch_info, key: scale_func(
                batch_info,
                jr.normal(key, ()),
            )
        ),
        "constant": dict(
            field=lambda trial_spec, batch_info, key: scale_func(
                batch_info,
                vector_with_gaussian_length(key),
            )
        ),
    }


def setup_task_model_pair(hps_train: TreeNamespace, *, key):
    """Returns a skeleton PyTree for reloading trained models."""
    scaleup_batches = hps_train.intervention_scaleup_batches
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

    loss_func = simple_reach_loss()

    loss_func = eqx.tree_at(
        lambda loss_func: loss_func.weights["nn_output"],
        loss_func,
        hps_train.model.control_loss_scale * loss_func.weights["nn_output"],
    )

    task_base = get_base_reaching_task(
        n_steps=hps_train.model.n_steps,
        loss_func=loss_func,
    )

    models = get_ensemble(
        point_mass_nn,
        task_base,
        n=hps_train.model.n_replicates,
        dt=hps_train.model.dt,
        mass=MASS,
        damping=hps_train.model.damping,
        hidden_size=hps_train.model.hidden_size,
        n_steps=hps_train.model.n_steps,
        feedback_delay_steps=hps_train.model.feedback_delay_steps,
        feedback_noise_std=hps_train.model.feedback_noise_std,
        motor_noise_std=hps_train.model.motor_noise_std,
        tau_rise=hps_train.model.tau_rise,
        tau_decay=hps_train.model.tau_rise,
        key=key,
    )

    def disturbance(field_std, active=True):
        return PLANT_DISTURBANCE_CLASSES[hps_train.pert.type].with_params(
            scale=field_std,
            active=active,
            **disturbance_params(partial(batch_scale_up, scaleup_batches[0], n_batches_scaleup))[
                hps_train.pert.type
            ],
        )

    return TaskModelPair(
        *schedule_intervenor(
            task_base,
            models,
            lambda model: model.step.mechanics,
            disturbance(hps_train.pert.std),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        )
    )
