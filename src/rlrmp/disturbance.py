import jax.numpy as jnp
import jax.random as jr
from feedbax.config.namespace import TreeNamespace
from feedbax.intervene import TimeSeriesParam

from rlrmp.misc import vector_with_gaussian_length

PLANT_INTERVENOR_LABEL = "DisturbanceField"


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
