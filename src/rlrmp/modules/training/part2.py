from collections.abc import Callable
from functools import partial
from typing import Literal as L
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_cookbook.tree as jtree
from feedbax.intervene import schedule_intervenor
from feedbax.task import TimeSeriesParam
from feedbax.xabdeef.models import point_mass_nn
from feedbax_experiments.analysis.disturbance import (
    PLANT_DISTURBANCE_CLASSES,
    PLANT_INTERVENOR_LABEL,
)
from feedbax_experiments.constants import (
    MASS,
)
from feedbax_experiments.misc import get_field_amplitude, vector_with_gaussian_length
from feedbax_experiments.setup_utils import get_base_reaching_task
from feedbax_experiments.training.train import always_active, bernoulli_active
from feedbax_experiments.types import LDict, TaskModelPair, TreeNamespace
from jaxtyping import Array, Float, Int, PRNGKeyArray

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


class Gusts(eqx.Module):
    signal: Float[Array, "k d=2"]
    starts: Int[Array, " k"]
    durations: Int[Array, " k"]
    forces: Float[Array, "k d=2"]


def get_gusts_fn(hps):
    n_steps = hps.model.n_steps - 1
    amplitude_std = hps.pert.std
    n_expected = hps.pert.n_expected
    duration_mean = hps.pert.duration_mean

    p_offset = 1.0 / jnp.maximum(1.0, duration_mean)
    max_gusts = n_steps // 2

    def _gusts_fn(trial_spec, batch_info, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        n_gusts = jr.poisson(key1, n_expected, shape=())
        n_gusts = jnp.minimum(n_gusts, max_gusts)

        def no_gusts():
            # return Gusts(
            #     signal=jnp.zeros((n_steps, 2), dtype=jnp.float32),
            #     starts=jnp.zeros((max_gusts,), dtype=jnp.int32),
            #     durations=jnp.zeros((max_gusts,), dtype=jnp.int32),
            #     forces=jnp.zeros((max_gusts, 2), dtype=jnp.float32),
            # )
            return jnp.zeros((n_steps, 2), dtype=jnp.float32)

        def some_gusts():
            starts_all = jr.choice(key2, n_steps, (max_gusts,), replace=False)
            durations_all = jr.geometric(key3, p_offset, (max_gusts,))
            forces_all = amplitude_std * vector_with_gaussian_length(key4, shape=(max_gusts,))

            # mask valid gusts
            mask = jnp.arange(max_gusts) < n_gusts
            starts = jnp.where(mask, starts_all, 0)
            durations = jnp.where(mask, durations_all, 0)
            forces = jnp.where(mask[:, None], forces_all, 0.0)

            # Build the signal
            ts = jnp.arange(n_steps)[None, :]  # (1, n_steps)
            s = starts[:, None]  # (n_gusts, 1)
            d = durations[:, None]  # (n_gusts, 1)
            in_window = (ts >= s) & (ts < (s + d))  # (n_gusts, n_steps)
            signal = jnp.einsum("kd,kt->td", forces, in_window.astype(jnp.float32))  # (n_steps, 2)
            # return Gusts(signal=signal, starts=starts, durations=durations, forces=forces)
            return signal

        return TimeSeriesParam(jax.lax.cond(n_gusts == 0, no_gusts, some_gusts))

    return dict(field=_gusts_fn)


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
    hps_train: TreeNamespace = TreeNamespace(),
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    """Returns a skeleton PyTree for reloading trained models."""
    hps_train = hps_train | kwargs

    # TODO: Implement scale-up for this experiment
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

    task_base = get_base_reaching_task(n_steps=hps_train.model.n_steps)

    models_base = jtree.get_ensemble(
        point_mass_nn,
        task_base,
        n_extra_inputs=1,  # for SISU
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

    try:
        task = task_base.add_input(
            name="sisu",
            input_fn=SISU_FNS[hps_train.method],
        )
    except AttributeError:
        raise ValueError("No training method label assigned to hps_train.method")

    return TaskModelPair(
        *schedule_intervenor(
            task,
            models_base,
            lambda model: model.step.mechanics,
            disturbance(hps_train),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        )
    )
