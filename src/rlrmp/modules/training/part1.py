import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.intervene import (
    CurlFieldParams,
    FixedFieldParams,
    schedule_intervenor,
)
from feedbax.misc import vector_with_gaussian_length
from feedbax.types import LDict, TaskModelPair, TreeNamespace

from rlrmp.disturbance import (
    PLANT_INTERVENOR_LABEL,
    get_gusts_fn,
)
from rlrmp.intervention_compat import add_plant_intervention_to_ensemble
from rlrmp.loss import get_reach_loss
from rlrmp.models import create_point_mass_nn_ensemble
from rlrmp.stochastic_runtime import (
    apply_stochastic_runtime_to_ensemble,
    stochastic_runtime_config_from_model,
)
from rlrmp.task import TASK_TYPES


# Parameter builders for different disturbance types
# These return dicts of additional params to merge into the base params
disturbance_extra_params = LDict.of("pert__type")(
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


def get_disturbance_params(pert_type: str, hps: TreeNamespace, scale, active=True):
    """Build disturbance params for the given type.

    Args:
        pert_type: Type of perturbation ("curl", "constant", "gusts")
        hps: Hyperparameters
        scale: Scale factor for the disturbance
        active: Whether the disturbance is active

    Returns:
        Appropriate params object (CurlFieldParams, FixedFieldParams, etc.)
    """
    extra_params = disturbance_extra_params[pert_type](hps)

    if pert_type == "curl":
        return CurlFieldParams(scale=scale, active=active, **extra_params)
    elif pert_type == "constant":
        return FixedFieldParams(scale=scale, active=active, **extra_params)
    elif pert_type == "gusts":
        return FixedFieldParams(scale=scale, active=active, **extra_params)
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")


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

    # Create base models
    models_base = create_point_mass_nn_ensemble(
        hps,
        task_base,
        n_extra_inputs=0,
        key=key,
    )

    # Insert intervention components into models via graph surgery
    models = add_plant_intervention_to_ensemble(
        models_base,
        hps.pert.type,
        PLANT_INTERVENOR_LABEL,
        active=False,  # Default to inactive; schedule_intervenor will control activation
    )
    models = apply_stochastic_runtime_to_ensemble(
        models,
        stochastic_runtime_config_from_model(hps.model),
    )

    # Build disturbance params for scheduling
    disturbance_params = get_disturbance_params(
        hps.pert.type,
        hps,
        scale=hps.pert.std,
        active=False,  # default_active is False
    )

    # Schedule the intervention params on the task
    task, models = schedule_intervenor(
        task_base,
        models,
        label=PLANT_INTERVENOR_LABEL,
        intervenor_params=disturbance_params,
        default_active=False,
    )

    return TaskModelPair(task, models)
