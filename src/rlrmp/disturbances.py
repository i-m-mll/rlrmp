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
