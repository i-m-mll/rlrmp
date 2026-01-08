from collections.abc import Callable
from functools import partial
from typing import Literal as L
from typing import TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.intervene import schedule_intervenor
from feedbax.task import DelayedReaches, SimpleReaches
from feedbax.misc import get_field_amplitude, vector_with_gaussian_length

# from rlrmp.loss import get_reach_loss
from feedbax.training.loss import get_reach_loss
from feedbax.training.train import always_active, bernoulli_active
from feedbax.types import LDict, TaskModelPair, TreeNamespace
from jaxtyping import PRNGKeyArray

from rlrmp.disturbance import (
    PLANT_DISTURBANCE_CLASSES,
    PLANT_INTERVENOR_LABEL,
)
from rlrmp.disturbances import get_gusts_fn
from rlrmp.loss import get_loss_update_func
from rlrmp.models import create_point_mass_nn_ensemble
from rlrmp.task import TASK_TYPES

TrainingMethodLabel: TypeAlias = L["bcs", "dai", "pai-asf", "pai-n"]


P_PERTURBED = LDict.of("train__method")(
    {
        "bcs": 0.5,
        "dai": 1.0,
        "pai-asf": 1.0,
    }
)

# Define whether the disturbance is active on each trial
disturbance_active: LDict[str, Callable] = LDict.of("train__method")(
    {
        "bcs": bernoulli_active,
        "dai": bernoulli_active,  # or always_active?
        "pai-asf": always_active,  # or bernoulli_active? and let hps control it
    }
)


def scaled_sampler(sample_fn, scale=1.0):
    def _fn(trial_spec, batch_info, key):
        return scale * sample_fn(key)

    return _fn


# Separate this def by training method so that we can multiply by `field_std` in the "pai-asf" case,
# without it affecting the SISU. That is, in all three cases `field_std` is a factor of
# the actual field strength, but in `"bcs"` and `"dai"` it is multiplied by the
# `scale` parameter, which is not seen by the network in those cases; and in `"pai-asf"` it is
# multiplied by the `field` parameter, which is not seen by the network in that case.
# (See the definition of `SCALE_FNS` below.)
#! TODO: limit curl and constant fields to movement epoch!
disturbance_params = LDict.of("train__method")(
    {
        "bcs": {
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal)),
            "constant": lambda hps: dict(field=scaled_sampler(vector_with_gaussian_length)),
        },
        "dai": {
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal)),
            "constant": lambda hps: dict(field=scaled_sampler(vector_with_gaussian_length)),
        },
        "pai-asf": {
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal, hps.pert.std)),
            "constant": lambda hps: dict(
                field=scaled_sampler(vector_with_gaussian_length, hps.pert.std)
            ),
            "gusts": get_gusts_fn,
        },
    }
)


# Define how the network's SISU will be determined from the trial specs, to which it is then added
SISU_FNS = LDict.of("train__method")(
    {
        "bcs": lambda trial_specs, key: trial_specs.intervene[PLANT_INTERVENOR_LABEL].active.astype(
            float
        ),
        "dai": lambda trial_specs, key: get_field_amplitude(
            trial_specs.intervene[PLANT_INTERVENOR_LABEL]
        ),
        "pai-asf": lambda trial_specs, key: trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale,
    }
)


"""Either scale the field strength by a constant std, or sample the std for each trial.

Note that in the `"pai-asf"` case the actual field amplitude is still scaled by `field_std`, 
but this is done in `disturbance_params` so that the magnitude of the SISU 
is the same on average between the `"dai"` and `"pai-asf"` methods.
"""
SCALE_FNS = LDict.of("train__method")(
    {
        "bcs": lambda field_std: field_std,
        "dai": lambda field_std: field_std,
        "pai-asf": lambda field_std: (
            lambda trial_spec, _, key: jr.uniform(key, (), minval=0, maxval=1)
        ),
    }
)


def disturbance(hps: TreeNamespace):
    return PLANT_DISTURBANCE_CLASSES[hps.pert.type].with_params(
        scale=SCALE_FNS[hps.method](hps.pert.std),
        active=disturbance_active[hps.method](P_PERTURBED[hps.method]),
        **disturbance_params[hps.method][hps.pert.type](hps),
    )


def setup_task_model_pair(
    hps: TreeNamespace = TreeNamespace(),
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    """Returns a skeleton PyTree for reloading trained models."""
    hps = hps | kwargs

    # TODO: Implement scale-up for this experiment
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

    models_base = create_point_mass_nn_ensemble(
        hps,
        task_base,
        n_extra_inputs=1,  # for SISU
        key=key,
    )

    try:
        task = task_base.add_input(
            name="sisu",
            input_fn=SISU_FNS[hps.method],
        )
    except AttributeError:
        raise ValueError("No training method label assigned to hps_train.method")

    return TaskModelPair(
        *schedule_intervenor(
            task,
            models_base,
            lambda model: model.step.mechanics,
            disturbance(hps),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        )
    )
