from collections.abc import Callable
from functools import partial
from typing import Literal as L
from typing import TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.intervene import (
    CurlFieldParams,
    FixedFieldParams,
    schedule_intervenor,
)
from feedbax.task import DelayedReaches, SimpleReaches
from feedbax.misc import get_field_amplitude, vector_with_gaussian_length
from feedbax.training.train import always_active, bernoulli_active

from rlrmp.loss import get_reach_loss
from feedbax.types import LDict, TaskModelPair, TreeNamespace
from jax_cookbook import is_module
from jaxtyping import PRNGKeyArray

from rlrmp.disturbance import (
    PLANT_INTERVENOR_LABEL,
)
from rlrmp.disturbances import get_gusts_fn
from rlrmp.intervention_compat import add_plant_intervention_to_ensemble
from rlrmp.loss import get_loss_update_func
from rlrmp.models import (
    LINEAR_HIDDEN_TYPES,
    create_point_mass_linear_ensemble,
    create_point_mass_nn_ensemble,
)
from rlrmp.task import TASK_TYPES

TrainingMethodLabel: TypeAlias = L["bcs", "dai", "pai-asf", "pai-n", "nominal-cs-gru"]


P_PERTURBED = LDict.of("train__method")(
    {
        "nominal-cs-gru": 0.0,
        "bcs": 0.5,
        "dai": 1.0,
        "pai-asf": 1.0,
    }
)

# Define whether the disturbance is active on each trial
disturbance_active: LDict[str, Callable] = LDict.of("train__method")(
    {
        "nominal-cs-gru": always_active,
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
disturbance_extra_params = LDict.of("train__method")(
    {
        "nominal-cs-gru": {
            "gusts": get_gusts_fn,
            "constant": lambda hps: dict(field=scaled_sampler(vector_with_gaussian_length)),
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal)),
        },
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
        "nominal-cs-gru": lambda trial_specs, key: jnp.zeros(
            (
                (trial_specs.timeline.epoch_bounds.shape[0], trial_specs.timeline.n_steps)
                if trial_specs.timeline.epoch_bounds is not None
                and trial_specs.timeline.epoch_bounds.ndim > 1
                else (trial_specs.timeline.n_steps,)
            ),
            dtype=float,
        ),
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
but this is done in `disturbance_extra_params` so that the magnitude of the SISU
is the same on average between the `"dai"` and `"pai-asf"` methods.
"""
SCALE_FNS = LDict.of("train__method")(
    {
        "nominal-cs-gru": lambda field_std: field_std,
        "bcs": lambda field_std: field_std,
        "dai": lambda field_std: field_std,
        "pai-asf": lambda field_std: (
            lambda trial_spec, _, key: jr.uniform(key, (), minval=0, maxval=1)
        ),
    }
)


def get_disturbance_params(hps: TreeNamespace):
    """Build disturbance params for the given hyperparameters.

    Args:
        hps: Hyperparameters including method, pert.type, pert.std

    Returns:
        Appropriate params object (CurlFieldParams, FixedFieldParams, etc.)
    """
    pert_type = hps.pert.type
    method = hps.method

    extra_params = disturbance_extra_params[method][pert_type](hps)
    scale = SCALE_FNS[method](hps.pert.std)
    active = disturbance_active[method](P_PERTURBED[method])

    if pert_type == "curl":
        return CurlFieldParams(scale=scale, active=active, **extra_params)
    elif pert_type == "constant":
        return FixedFieldParams(scale=scale, active=active, **extra_params)
    elif pert_type == "gusts":
        return FixedFieldParams(scale=scale, active=active, **extra_params)
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")


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

    task_type = hps.task.type
    hps_task = {k: v for k, v in hps.task.omitting_attrs("eval_n", "type").items() if v is not None}
    if task_type == "simple_reach":
        delayed_only_keys = {
            "epoch_len_ranges",
            "target_on_epochs",
            "hold_epochs",
            "move_epochs",
            "p_catch_trial",
            "train_endpoint_mode",
        }
        hps_task = {k: v for k, v in hps_task.items() if k not in delayed_only_keys}

    task_base = TASK_TYPES[task_type](loss_func=get_reach_loss(hps), **hps_task)

    # Resolve hidden_type from hps if present; default (None) falls back to GRUCell
    hidden_type = getattr(hps, 'hidden_type', None)
    # Resolve SISU gating mode; default "additive" preserves existing behavior
    sisu_gating = getattr(hps, 'sisu_gating', 'additive')

    # Dispatch: linear-controller MVP variants (Bug: 410d7ac) bypass
    # ``create_point_mass_nn_ensemble`` entirely because they replace
    # ``SimpleStagedNetwork`` with a purpose-built ``Component``. Detected via
    # the sentinel strings ``"linear"`` / ``"linear_tracker"``; for these,
    # hidden_type is a str (not a class), and SISU is still threaded through
    # the task input pipeline (controller ignores it) so n_extra_inputs is 0.
    if isinstance(hidden_type, str) and hidden_type in LINEAR_HIDDEN_TYPES:
        models_base = create_point_mass_linear_ensemble(
            hps,
            task_base,
            controller_type=hidden_type,
            key=key,
        )
    else:
        # Create base models with extra input for SISU
        models_base = create_point_mass_nn_ensemble(
            hps,
            task_base,
            n_extra_inputs=1,  # for SISU (even when multiplicative, task still provides it)
            hidden_type=hidden_type,
            sisu_gating=sisu_gating,
            key=key,
        )

    # Insert intervention components into models via graph surgery
    models = add_plant_intervention_to_ensemble(
        models_base,
        hps.pert.type,
        PLANT_INTERVENOR_LABEL,
        active=False,  # Default to inactive; schedule_intervenor will control activation
    )

    # Add SISU input to task
    try:
        task = task_base.add_input(
            name="sisu",
            input_fn=SISU_FNS[hps.method],
        )
    except AttributeError:
        raise ValueError("No training method label assigned to hps_train.method")

    # Build disturbance params for scheduling
    disturbance_params = get_disturbance_params(hps)

    # Schedule the intervention params on the task
    task, models = schedule_intervenor(
        task,
        models,
        label=PLANT_INTERVENOR_LABEL,
        intervenor_params=disturbance_params,
        default_active=False,
    )

    return TaskModelPair(task, models)
