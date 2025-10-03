import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.task import TimeSeriesParam
from feedbax_experiments.misc import vector_with_gaussian_length
from jaxtyping import Array, Float, Int


class Gusts(eqx.Module):
    signal: Float[Array, "k d=2"]
    starts: Int[Array, " k"]
    durations: Int[Array, " k"]
    forces: Float[Array, "k d=2"]


def get_gusts_fn(hps):
    n_steps = hps.task.n_steps - 1
    amplitude_std = hps.pert.std
    n_expected = hps.pert.n_expected
    duration_mean = hps.pert.duration_mean

    p_offset = 1.0 / jnp.maximum(1.0, duration_mean)
    max_gusts = n_steps // 2

    def _gusts_fn(trial_spec, batch_info, key):
        key1, key2, key3, key4 = jr.split(key, 4)
        n_gusts = jr.poisson(key1, n_expected, shape=())
        n_gusts = jnp.minimum(n_gusts, max_gusts)

        if trial_spec.timeline.has_epochs:
            move_start, move_end = trial_spec.timeline.window_for_epoch("movement")
        else:
            move_start = 0
            move_end = n_steps  # exclusive

        # Always sample a fixed-size pool; later, mask down to n_gusts.
        # Use randint over [start, end) → avoids population-size issues entirely.
        starts_all = jr.randint(key2, (max_gusts,), minval=move_start, maxval=move_end)

        # Geometric durations (>=1), then clamp so pulses never extend past the epoch end.
        durations_all = jr.geometric(key3, p_offset, (max_gusts,))
        max_dur_all = jnp.maximum(0, move_end - starts_all)  # per-gust budget
        durations_all = jnp.minimum(durations_all, max_dur_all)

        # Forces: random direction with Gaussian length scaled by amplitude_std
        forces_all = amplitude_std * vector_with_gaussian_length(key4, shape=(max_gusts,))

        # Keep only the first n_gusts; zero out the rest
        mask = jnp.arange(max_gusts) < n_gusts
        starts = jnp.where(mask, starts_all, 0)
        durations = jnp.where(mask, durations_all, 0)
        forces = jnp.where(mask[:, None], forces_all, 0.0)

        # Build the time-series by summing rectangular pulses
        ts = jnp.arange(n_steps)[None, :]  # (1, T)
        s = starts[:, None]  # (K, 1)
        d = durations[:, None]  # (K, 1)
        in_window = (ts >= s) & (ts < (s + d))  # (K, T)
        signal = jnp.einsum("kd,kt->td", forces, in_window.astype(jnp.float32))  # (T, 2)

        return TimeSeriesParam(signal)

    return dict(field=_gusts_fn)
