from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.analysis.state_utils import vmap_eval_ensemble
from feedbax.intervene import (
    AddNoise,
    CurlField,
    CurlFieldParams,
    FixedField,
    FixedFieldParams,
    TimeSeriesParam,
)
from rlrmp.misc import vector_with_gaussian_length
from feedbax.config.namespace import TreeNamespace
from jaxtyping import Array, Float, Int

FB_INTERVENOR_LABEL = "FeedbackPert"
PLANT_INTERVENOR_LABEL = "DisturbanceField"


PLANT_DISTURBANCE_CLASSES = {
    "curl": CurlField,
    "constant": FixedField,
    "gusts": FixedField,
    "noise": AddNoise,
}


def orthogonal_field(trial_spec, _, key):
    init_pos = trial_spec.inits["mechanics.effector"].pos
    goal_pos = jnp.take(trial_spec.targets["mechanics.effector.pos"].value, -1, axis=-2)
    direction_vec = goal_pos - init_pos
    direction_vec = direction_vec / (jnp.linalg.norm(direction_vec) + 1e-9)
    return jnp.array([-direction_vec[1], direction_vec[0]])


#! TODO: Control start and end time via hps
def get_fixed_gust_fn(hps: TreeNamespace, start_prop: float = 0.1, end_prop: float = 0.75):
    """Returns a fixed orthogonal field that is active during a fixed portion of the trial."""
    n_steps = hps.task.n_steps - 1
    idxs = jnp.arange(n_steps)

    def _get_active_ts(trial_spec, _, key):
        if trial_spec.timeline.has_epochs:
            move_start, move_end = trial_spec.timeline.window_for_epoch("movement")
            move_len = move_end - move_start

            gust_start = move_start + jnp.floor(move_len * start_prop).astype(jnp.int32)
            gust_end = move_start + jnp.floor(move_len * end_prop).astype(jnp.int32)
            gust_end = jnp.minimum(gust_end, move_end)  # clamp to epoch end
        else:
            gust_start = jnp.floor(n_steps * start_prop).astype(jnp.int32)
            gust_end = jnp.floor(n_steps * end_prop).astype(jnp.int32)

        gust_end = jnp.maximum(gust_end, gust_start)
        active_ts = (idxs >= gust_start) & (idxs < gust_end)
        return TimeSeriesParam(active_ts)

    def fixed_gust_params_fn(scale: float):
        return FixedFieldParams(
            scale=scale,
            field=orthogonal_field,
            active=_get_active_ts,
        )

    return fixed_gust_params_fn


def get_gusts_fn(hps: TreeNamespace):
    """Return a trial-sampled gust force-field parameter function."""
    n_steps = hps.task.n_steps - 1
    amplitude_std = hps.pert.std
    n_expected = hps.pert.n_expected
    duration_mean = hps.pert.duration_mean

    p_offset = 1.0 / jnp.maximum(1.0, duration_mean)
    max_gusts = n_steps // 2

    def _gusts_fn(trial_spec, batch_info, key):
        del batch_info
        key_n, key_start, key_duration, key_force = jr.split(key, 4)
        n_gusts = jr.poisson(key_n, n_expected, shape=())
        n_gusts = jnp.minimum(n_gusts, max_gusts)

        if trial_spec.timeline.has_epochs:
            move_start, move_end = trial_spec.timeline.window_for_epoch("movement")
        else:
            move_start = 0
            move_end = n_steps

        starts_all = jr.randint(key_start, (max_gusts,), minval=move_start, maxval=move_end)
        durations_all = jr.geometric(key_duration, p_offset, (max_gusts,))
        max_dur_all = jnp.maximum(0, move_end - starts_all)
        durations_all = jnp.minimum(durations_all, max_dur_all)
        forces_all = amplitude_std * vector_with_gaussian_length(key_force, shape=(max_gusts,))

        mask = jnp.arange(max_gusts) < n_gusts
        starts = jnp.where(mask, starts_all, 0)
        durations = jnp.where(mask, durations_all, 0)
        forces = jnp.where(mask[:, None], forces_all, 0.0)

        ts = jnp.arange(n_steps)[None, :]
        starts_expanded = starts[:, None]
        durations_expanded = durations[:, None]
        active = (ts >= starts_expanded) & (ts < (starts_expanded + durations_expanded))
        signal = jnp.einsum("kd,kt->td", forces, active.astype(jnp.float32))
        return TimeSeriesParam(signal)

    return dict(field=_gusts_fn)


def get_plant_intervention_params(intervention_type: str, hps: TreeNamespace, scale: float):
    """Get intervention params for the given type.

    Args:
        intervention_type: Type of intervention ("curl", "constant", "gusts")
        hps: Hyperparameters
        scale: Scale factor for the intervention

    Returns:
        Params object appropriate for the intervention type.
    """
    if intervention_type == "curl":
        return CurlFieldParams(scale=scale)
    elif intervention_type == "constant":
        return FixedFieldParams(scale=scale, field=orthogonal_field)
    elif intervention_type == "gusts":
        return get_fixed_gust_fn(hps)(scale)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


# Legacy compatibility - functions that return params-creating callables
PLANT_PERT_FNS = {
    "curl": lambda hps: lambda scale: CurlFieldParams(scale=scale),
    "constant": lambda hps: lambda scale: FixedFieldParams(scale=scale, field=orthogonal_field),
    "gusts": get_fixed_gust_fn,
}


def task_with_pert_amp(task, pert_amp, intervenor_label):
    """Returns a task with the given disturbance amplitude.

    Note: In the eager-models architecture, intervention params are stored
    directly on InterventionSpec.params, not via .intervenor.params.
    """
    return eqx.tree_at(
        lambda task: task.intervention_specs.validation[intervenor_label].params.scale,
        task,
        pert_amp,
    )


def get_pert_amp_vmap_eval_fn(
    where_pert_amps_in_hps: Callable[[TreeNamespace], Sequence[float]],
    intervenor_label: str,
):
    """Returns a function for evaluating models across a range of perturbation amplitudes.

    Args:
        where_pert_amps_in_hps: Callable that selects the sequence of amplitudes from the tree of hyperparameters.
        intervenor_label: The same argument passed to `schedule_intervenor` when setting up the task+models, which
            identifies which intervention to scale by the vmap argument.
    """

    def eval_fn(key_eval, hps, models, task):
        """Vmap over impulse amplitude."""

        states = eqx.filter_vmap(
            lambda amplitude: vmap_eval_ensemble(
                key_eval,
                hps,
                models,
                task_with_pert_amp(task, amplitude, intervenor_label),
            ),
        )(jnp.array(where_pert_amps_in_hps(hps)))

        # I am not sure why this moveaxis is necessary.
        # I tried using `out_axes=2` (with or without `in_axes=0`) and
        # the result has the trial (axis 0) and replicate (axis 1) swapped.
        # (I had expected vmap to simply insert the new axis in the indicated position.)
        return jt.map(
            lambda arr: jnp.moveaxis(arr, 0, 2),
            states,
        )

    return eval_fn


class Gusts(eqx.Module):
    signal: Float[Array, "k d=2"]
    starts: Int[Array, " k"]
    durations: Int[Array, " k"]
    forces: Float[Array, "k d=2"]
